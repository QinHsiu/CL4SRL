import time
import os
import random
import numpy as np
import torch
import torch.utils.data

from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_spk
from data_aug import PhoneAug,SpecAug


# spec data loader
class SpecDataLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id pairs
        2) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid, args):
        self.audiopaths_sid = load_filepaths_and_spk(audiopaths_sid)
        self.max_wav_value = args.max_wav_value
        self.sampling_rate = args.sampling_rate
        self.filter_length  = args.filter_length
        self.hop_length     = args.hop_length
        self.win_length     = args.win_length

        # data augmentation
        self.use_spec_aug  = True
        self.specAug  = SpecAug(3,10,10,10,2,2)
        random.seed(1234)
        random.shuffle(self.audiopaths_sid)
        self.get_lengths()

    def spec_aug_pair(self,spec):
        specAugMode=random.randint(0,6)
        spec_=spec.unsqueeze(0)    
        specAug1=self.specAug.dataProcess(specAugMode,spec_)
        specAug2=self.specAug.dataProcess(specAugMode,spec_)
        specAug1=specAug1.squeeze(0)
        specAug2=specAug2.squeeze(0)
        return specAug1,specAug2

    def get_audio_speaker_pair(self, audiopath_sid):
        # separate filename, speaker_id
        audiopath, sid = audiopath_sid[0], audiopath_sid[1]
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        # data augmentation
        specAug1,specAug2=self.spec_aug_pair(spec)
        return (spec, wav, sid,specAug1,specAug2)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        os.system("rm -rf {}".format(spec_filename))
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_lengths(self):
        lengths = []
        for audiopath, sid in self.audiopaths_sid:
            lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.lengths = lengths

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_speaker_pair(self.audiopaths_sid[index])

    def __len__(self):
        return len(self.audiopaths_sid)


class SpecSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self):
        pass

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [spec_normalized, wav_normalized, sid]
        """
        

        max_spec_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[1].size(1) for x in batch])

        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        
        spec_padded.zero_()
        wav_padded.zero_()
        
        # spec augmentation
        spec_padded0 = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        spec_padded1 = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        

        spec_padded0.zero_()
        spec_padded1.zero_()
        for i in range(len(batch)):
            row = batch[i]

            spec = row[0]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[1]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[2]
            
            spec_aug0=row[3]
            spec_padded0[i,:,:spec_aug0.size(1)]=spec_aug0
            spec_aug1=row[4]
            spec_padded1[i,:,:spec_aug1.size(1)]=spec_aug1          
        
        return spec_padded, spec_lengths, wav_padded, wav_lengths, sid, spec_padded0,spec_padded1


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size

if __name__=="__main__":
    pass