"""SpecAugment Implementation for Pytorch.
Related paper : https://arxiv.org/pdf/1904.08779.pdf
In this paper, show summarized parameters by each open datasets in Tabel 1.
-----------------------------------------
Policy | W  | F  | m_F |  T  |  p  | m_T
-----------------------------------------
None   |  0 |  0 |  -  |  0  |  -  |  -
-----------------------------------------
LB     | 80 | 27 |  1  | 100 | 1.0 | 1
-----------------------------------------
LD     | 80 | 27 |  2  | 100 | 1.0 | 2
-----------------------------------------
SM     | 40 | 15 |  2  |  70 | 0.2 | 2
-----------------------------------------
SS     | 40 | 27 |  2  |  70 | 0.2 | 2
-----------------------------------------
LB : LibriSpeech basic
LD : LibriSpeech double
SM : Switchboard mild
SS : Switchboard strong
"""


import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from random import sample,randrange,randint,uniform,seed
from copy import deepcopy
from scipy.io.wavfile import read
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
from modules import InfoNCE
from sparse_image_warp_pytorch import sparse_image_warp
# matplotlib.use('TkAgg')

class PhoneAug():
    def __init__(self,reorderRatio,cropRatio,maskRatio,mixRatio=None):
        self.reorderRatio=reorderRatio
        self.cropRatio=cropRatio
        self.maskRatio=maskRatio
        self.mixRatio=mixRatio
        
    def dataProcess(self,augMode,phone,phone1=None):
        self.augMode=augMode
        if self.augMode==0:
            return self.Reorder(phone,self.reorderRatio)
        elif self.augMode==1:
            return self.Crop(phone,self.cropRatio)
        elif self.augMode==2:
            return self.Mask(phone,self.maskRatio)
        elif self.augMode==3:
            return self.MixUp(phone,phone1,self.mixRatio)
        else:
            raise NotImplementedError
        
    def Reorder(self,phone,reorderRatio=0.5):
        phone_output=deepcopy(phone)
        reorder_l=int(len(phone)*reorderRatio)
        idx_begin=sample(range(len(phone)-reorder_l),1)[0]
        idx_end=idx_begin+reorder_l
        phone_output[idx_begin:idx_end]=phone_output[idx_begin:idx_end][::-1]
        return phone_output
        
    
    def Crop(self,phone,cropRatio=0.5):
        phone_output=deepcopy(phone)
        crop_l=int(len(phone)*cropRatio)
        idx_begin=sample(range(len(phone)-crop_l),1)[0]
        idx_end=idx_begin+crop_l
        phone_output=phone_output[:idx_begin]+phone_output[idx_end:]
        phone_output=[0]*(len(phone)-len(phone_output))+phone_output
        return phone_output
    
    def Mask(self,phone,maskRatio=0.5):
        phone_output=deepcopy(phone)
        mask_l=int(len(phone)*maskRatio)
        idx_begin=sample(range(len(phone)-mask_l),1)[0]
        idx_end=idx_begin+mask_l
        phone_output[idx_begin:idx_end]=[0 for _ in range(mask_l)]
        return phone_output
    
    def MixUp(self,phone1,phone2,mixRatio=0.5):
        phone1_output=deepcopy(phone1)
        phone2_output=deepcopy(phone2)
        mix_length=int(len(phone1)*mixRatio)
        phone1_output=phone1_output[:mix_length]+phone2_output[mix_length:]
        phone2_output=phone2_output[:mix_length]+phone1_output[mix_length:]
        return phone1_output,phone2_output
    

class SpecAug():
    def __init__(self,W=5, time_warping_para=10, frequency_masking_para=10,
                 time_masking_para=10, frequency_mask_num=1, time_mask_num=1):
        self.W=W
        self.time_warping_para=time_warping_para
        self.frequency_masking_para=frequency_masking_para
        self.time_masking_para=time_masking_para
        self.frequency_mask_num=frequency_mask_num
        self.time_mask_num=time_mask_num
        
    def dataProcess(self,augMode,spec):
        if augMode==0:
            return self.TimeWarp(spec,self.W)
        elif augMode==1:
            return self.TimeMask(spec,self.time_mask_num,self.time_masking_para)
        elif augMode==2:
            return self.FrequenceMask(spec,self.frequency_mask_num,self.frequency_masking_para)
        elif augMode==3:
            return self.TimeFrequenceMask(spec)
        elif augMode==4:
            return self.FrequenceWarpMask(spec)
        elif augMode==5:
            return self.TimeWarpMask(spec)
        elif augMode==6:
            return self.TimeWarpFrequenceMask(spec)
        else:
            raise NotImplementedError
            
    
    def TimeWarp(self,spec,W=5):
        specT=deepcopy(spec)
        num_rows = specT.shape[1] # 1
        spec_len = specT.shape[2] # 2

        y = num_rows // 2
        horizontal_line_at_ctr = specT[0][y]
        assert len(horizontal_line_at_ctr) == spec_len

        point_to_warp = horizontal_line_at_ctr[randrange(W, spec_len-W)]
        # assert isinstance(point_to_warp, torch.Tensor)

        # Uniform distribution from (0,W) with chance to be up to W negative
        dist_to_warp = randrange(-W, W)
        src_pts = torch.tensor([[[y, point_to_warp]]])
        dest_pts = torch.tensor([[[y, point_to_warp + dist_to_warp]]])
        warped_spectro, dense_flows = sparse_image_warp(specT, src_pts, dest_pts)
        return warped_spectro.squeeze(3)
    
    def TimeMask(self,spec,time_mask_num,time_masking_para):
        specT=deepcopy(spec)
        tau=specT.shape[1]
        for i in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=time_masking_para)
            t = int(t)
            t0 = randint(0, tau-t)
            specT[:, :, t0:t0+t] = 0
        return specT
    
        
    def FrequenceMask(self,spec,frequency_mask_num,frequency_masking_para):
        specF=deepcopy(spec)
        v=specF.shape[1]
        for i in range(frequency_mask_num):
            f = np.random.uniform(low=0.0, high=frequency_masking_para)
            f = int(f)
            f0 = randint(0, v-f)
            specF[:, f0:f0+f, :] = 0
        return specF
        
    def TimeFrequenceMask(self,spec):
        # time mask
        specT=self.TimeMask(spec,self.time_mask_num,self.time_masking_para)
        # frequency mask
        specF=self.FrequenceMask(specT,self.frequency_mask_num,self.frequency_masking_para)
        return specF
        
        
    def TimeWarpMask(self,spec):
        # time warp
        specT=self.TimeWarp(spec,self.W)
        # time mask
        specT=self.TimeMask(specT,self.time_mask_num,self.time_masking_para)
        return specT
        
    def FrequenceWarpMask(self,spec):
        # time warp
        specT=self.TimeWarp(spec,self.W)
        # frequency mask
        specF=self.FrequenceMask(specT,self.frequency_mask_num,self.frequency_masking_para)
        return specF
        
    def TimeWarpFrequenceMask(self,spec):
        # time warp
        specT=self.TimeWarp(spec,self.W)
        # time mask
        specT=self.TimeMask(specT,self.time_mask_num,self.time_masking_para)
        # frequency mask
        specF=self.FrequenceMask(specT,self.frequency_mask_num,self.frequency_masking_para)
        return specF
   
    def VisualizationSpectrogram(self,mel_spectrogram, title):
        """visualizing result of SpecAugment
        # Arguments:
        mel_spectrogram(ndarray): mel_spectrogram to visualize.
        title(String): plot figure's title
        """
        # Show mel-spectrogram using librosa's specshow.
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(mel_spectrogram[0, :, :], ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
        # plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        # plt.show()
        plt.savefig("resources/{}.png".format(title))



if __name__=="__main__":
    seed(2023)
    aug_mode=0
    phone_ori=[i for i in range(10)]
    print("-----------------phone aug------------------------------")
    phoneAug=PhoneAug(0.5,0.5,0.5)
    print("ori: ",phone_ori)
    reoderPhone=phoneAug.dataProcess(0,phone_ori)
    cropPhone=phoneAug.dataProcess(1,phone_ori)
    maskPhone=phoneAug.dataProcess(2,phone_ori)
    print("reoder: ",reoderPhone)
    print("mask: ",maskPhone)
    print("crop: ",cropPhone)
    
    full_path="./data/toy/wav_ori/0108_交付_睡前儿童故事_s-yyx_英文字母_F_1.wav"
    # sampling_rate, data = read(full_path)
    # print(torch.FloatTensor(data.astype(np.float32)).shape)
    print("-----------------spec aug------------------------------")
    specAug=SpecAug(3,10,10,10,2,2)
    audio,sampling_rate=librosa.load(full_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                     sr=sampling_rate,
                                                     n_mels=256,
                                                     hop_length=128,
                                                     fmax=8000)
    print("specOri: ",mel_spectrogram,mel_spectrogram.shape)
    shape=mel_spectrogram.shape
    mel_spectrogram=np.reshape(mel_spectrogram,(-1,shape[0],shape[1]))
    mel_spectrogram=torch.from_numpy(mel_spectrogram)
    
    # Show Raw mel-spectrogram
    specAug.VisualizationSpectrogram(mel_spectrogram=mel_spectrogram,
                                                      title="Raw Mel Spectrogram")
    
    # Calculate 
    specAugTW=specAug.dataProcess(0,mel_spectrogram)
    specAugTM=specAug.dataProcess(1,mel_spectrogram)
    specAugFM=specAug.dataProcess(2,mel_spectrogram)
    specAugTFM=specAug.dataProcess(3,mel_spectrogram)
    specAugWFM=specAug.dataProcess(4,mel_spectrogram)
    specAugTWM=specAug.dataProcess(5,mel_spectrogram)
    specAugTTF=specAug.dataProcess(6,mel_spectrogram)
    # print("specTimeAug: ",specAugTime,specAugTime.shape)
    
    # Show time warped & masked spectrogram
    specAug.VisualizationSpectrogram(mel_spectrogram=specAugTW,title="specAugTW")
    specAug.VisualizationSpectrogram(mel_spectrogram=specAugTM,title="specAugTM")
    specAug.VisualizationSpectrogram(mel_spectrogram=specAugFM,title="specAugFM")
    specAug.VisualizationSpectrogram(mel_spectrogram=specAugTFM,title="specAugTFM")       
    specAug.VisualizationSpectrogram(mel_spectrogram=specAugWFM,title="specAugWFM")
    specAug.VisualizationSpectrogram(mel_spectrogram=specAugTWM,title="specAugTWM")
    specAug.VisualizationSpectrogram(mel_spectrogram=specAugTTF,title="specAugTTF")       
        
   
    
    
    print("--------------------calculate loss-------------------")
    infoNce=InfoNCE()
    loss=infoNce.info_nce(specAugTFM[0,:,:],specAugFM[0,:,:],1.0,256,'dot',False,None)/256
    print("infoNCE: ",loss)
    
    # specAugOutput=specAugOutput.numpy()
    # write back
    # step3 converting mel-spectrogrma back to wav file
    # mel_spectrogram = librosa.feature.inverse.mel_to_audio(
    #                                         specAugOutput[0,:,:], 
    #                                         sr=sampling_rate, 
    #                                         n_fft=2048, 
    #                                         hop_length=128 
    #                                         # win_length=None, 
    #                                         # window='hann', 
    #                                         # center=True, 
    #                                         # pad_mode='reflect', 
    #                                         # power=2.0, 
    #                                         # n_iter=32
    #                                         )

    # step4 - save it as a wav file
    # import soundfile as sf
    # sf.write("./resources/test1.wav", mel_spectrogram,sampling_rate)
    print("----------------------success-----------------")
    


    
