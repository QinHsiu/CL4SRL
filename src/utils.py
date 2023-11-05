import os
import random
import numpy as np
import torch

import glob
import sys
import argparse
import logging
import json
import subprocess

from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib



MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging



def plot_scatter(feature,label_id,epoches):
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['figure.figsize'] = (8, 8)
    plt.rcParams['font.size'] = 20
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True


    tsne = TSNE(n_components=2,init="pca",learning_rate="auto",n_iter=1000)#,learning_rate="auto",init="pca")
    X_tsne = tsne.fit_transform(feature)
    X_tsne_data = np.vstack((X_tsne.T, label_id)).T
    df_tsne = pd.DataFrame(X_tsne_data, columns=['Feat1','Feat2','label'])
    df_tsne['Feat1']=df_tsne['Feat1']
    df_tsne['Feat2']=df_tsne['Feat2']
    
    print(df_tsne["Feat1"].shape,df_tsne.shape)
    
    sns.set(style="darkgrid")
    sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2, 'lines.markersize': 20})

    cmap="tab20"
    s=60

    plt.scatter(df_tsne["Feat1"], df_tsne["Feat2"], s=s, c=df_tsne["label"],cmap=cmap,marker="o")
    # plt.colorbar()
    plt.xlabel("{0}".format("vctk"),size=25)      


    # plt.yticks([-2,-1, 0, 1, 2], fontproperties='SimHei', size=20)  # 设置大小及加粗
    # plt.xticks([-2, -1, 0, 1, 2], fontproperties='SimHei', size=20)
    
    # plt.yticks([-100, -50, 0, 50, 100], fontproperties='Microsoft YaHei', size=20)  # 设置大小及加粗
    # plt.xticks([-100, -50, 0, 50, 100], fontproperties='Microsoft YaHei', size=20)
    
    plt.yticks([-30, -15, 0, 15, 30], fontproperties='Microsoft YaHei', size=20)  # 设置大小及加粗
    plt.xticks([-30, -15, 0, 15, 30], fontproperties='Microsoft YaHei', size=20)
    # plt.yticks([-20, -10, 0, 10, 20], fontproperties='Microsoft YaHei', size=20)  # 设置大小及加粗
    # plt.xticks([-20, -10, 0, 10, 20], fontproperties='Microsoft YaHei', size=20)
    plt.savefig('./pics/{0}_{1}_{2}.png'.format("vctk",10,epoches))


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def visulization():
    pass
    
    

def load_checkpoint(checkpoint_path, model, optimizer=None):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  saved_state_dict = checkpoint_dict['model']
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict= {}
  for k, v in state_dict.items():
    try:
      new_state_dict[k] = saved_state_dict[k]
    except:
      logger.info("%s is not in the checkpoint" % k)
      new_state_dict[k] = v
  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict)
  else:
    model.load_state_dict(new_state_dict)
  logger.info("Loaded checkpoint '{}' (iteration {})" .format(
    checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
  logger.info("Saving model and optimizer state at iteration {} to {}".format(
    iteration, checkpoint_path))
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save({'model': state_dict,
              'iteration': iteration,
              'optimizer': optimizer.state_dict(),
              'learning_rate': learning_rate}, checkpoint_path)


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
  for k, v in scalars.items():
    writer.add_scalar(k, v, global_step)
  for k, v in histograms.items():
    writer.add_histogram(k, v, global_step)
  for k, v in images.items():
    writer.add_image(k, v, global_step, dataformats='HWC')
  for k, v in audios.items():
    writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x


def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np
  
  fig, ax = plt.subplots(figsize=(10,2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                  interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def plot_alignment_to_numpy(alignment, info=None):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(6, 4))
  im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                  interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
      xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_spk(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_spk = [line.strip().split(split)[:2] for line in f]
  return filepaths_and_spk


def get_hparams(init=True):
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                      help='JSON file for configuration')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Model name')
  
  args = parser.parse_args()
  model_dir = os.path.join("./logs", args.model)

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  config_path = args.config
  config_save_path = os.path.join(model_dir, "config.json")
  if init:
    with open(config_path, "r") as f:
      data = f.read()
    with open(config_save_path, "w") as f:
      f.write(data)
  else:
    with open(config_save_path, "r") as f:
      data = f.read()
  config = json.loads(data)
  
  hparams = HParams(**config)
  hparams.model_dir = model_dir
  return hparams


def get_hparams_from_dir(model_dir):
  config_save_path = os.path.join(model_dir, "config.json")
  with open(config_save_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  hparams.model_dir = model_dir
  return hparams


def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  return hparams


def check_git_hash(model_dir):
  source_dir = os.path.dirname(os.path.realpath(__file__))
  if not os.path.exists(os.path.join(source_dir, ".git")):
    logger.warn("{} is not a git repository, therefore hash value comparison will be ignored.".format(
      source_dir
    ))
    return

  cur_hash = subprocess.getoutput("git rev-parse HEAD")

  path = os.path.join(model_dir, "githash")
  if os.path.exists(path):
    saved_hash = open(path).read()
    if saved_hash != cur_hash:
      logger.warn("git hash values are different. {}(saved) != {}(current)".format(
        saved_hash[:8], cur_hash[:8]))
  else:
    open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)
  
  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger


class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()
