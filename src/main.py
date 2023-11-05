import argparse
import torch
import os
import torch.nn as nn
from models import SpecEncoder
from data_loader import SpecDataLoader,SpecSpeakerCollate,DistributedBucketSampler
from torch.utils.data import DataLoader
from os.path import join,exists
from trainer import Trainer
from utils import set_seed
# from utils import gen_train_test_file


def main():
    # setting config    
    parser=argparse.ArgumentParser()
    
    # data config  
    parser.add_argument('-d','--data_name',type=str,default="vctk",help="data name")
    parser.add_argument('-sr','--sampling_rate',type=int,default=24000,help="sampling rate")
    parser.add_argument('-mwv','--max_wav_value',type=float,default=32768.0,help="max wave value")
    parser.add_argument('-fl','--filter_length',type=int,default=1024,help="filter length")
    parser.add_argument('-hl','--hop_length',type=int,default=256,help="hope length")
    parser.add_argument('-wl','--win_length',type=int,default=1024,help="win length")
    parser.add_argument('-sn','--spk_num',type=int,default=109,help="speaker nums")
    
    
    
    # augmentation mode {0:none,1:hybrid}
    parser.add_argument('-a','--aug_mode',type=str,default="0",help="augmentation mode")

    # model config
    # spec encoder
    parser.add_argument('-ic','--spec_channel',type=int,default=513,help="spec channels")
    parser.add_argument('-oc','--out_channels',type=int,default=192,help="out channels")
    parser.add_argument('-hc','--hidden_channels',type=int,default=192,help="hidden channels")
    parser.add_argument('-ks','--kernel_size',type=int,default=5,help="kernel size")
    parser.add_argument('-dr','--dilation_rate',type=float,default=1,help="dilation rate")
    parser.add_argument('-nl','--n_layers',type=int,default=16,help="layers num")
    parser.add_argument('-gc','--gin_channels',type=int,default=256,help="gin channels")
    # cluster 
    parser.add_argument('-cn','--cluster_num',type=int,default=109,help="cluster num")

    # other config
    parser.add_argument('-lr','--learning_rate',type=float,default=2e-4,help="learning rate")
    parser.add_argument('-b','--betas',type=list,default=[0.8, 0.99],help="betas in adamw")
    parser.add_argument('-es','--eps',type=float,default=1e-9,help="eps in adamw")
    parser.add_argument('-gid','--gpu_id',type=str,default='0',help="gpu id")
    parser.add_argument('-t','--temp',type=float,default=1.0,help="temperature")
    parser.add_argument('-si','--sim',type=str,default='dot',help="similarity")
    parser.add_argument("--f_neg_mask", action="store_true", help="delete the FNM component")
    parser.add_argument('-bs','--batch_size',type=int,default=64,help="batch size")
    parser.add_argument('-e','--epoches',type=int,default=1,help="training epoches")
    parser.add_argument('-sd','--seed',type=int,default=2023,help="seed")
    parser.add_argument('-s','--save_dir',type=str,default="./output/vctk.pt",help="model save dir")
    parser.add_argument('--do_eval',action="store_true",help="do eval")
    parser.add_argument('-lfq','--log_freq',type=int,default=10,help="loginfo per epoches")
    parser.add_argument('-lfe','--log_file',type=str,default="./output/log.txt",help="loginfo")

    parser.add_argument('--no_cuda',action="store_true",help="cuda condation")

    parser.add_argument('-la','--lam',type=float,default=1.0,help="self-supervised contrastive loss weight")
    parser.add_argument('-be','--beta',type=float,default=0.1,help="supervised contrastive loss weight")
    

        
        
    args = parser.parse_args()
    
    data_train_file=join("../filelists/{0}_train_{1}.txt".format(args.data_name,10))
    data_test_file=join("../filelists/{0}_test_{1}.txt".format(args.data_name,10))
    
    if not exists(data_train_file):
        print("train file doesn't exist!")

    args.data_train_file=data_train_file
    args.data_test_file=data_test_file
    # set seed
    set_seed(args.seed)
    
    # set cuda condation
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", args.cuda_condition)
    
    # data loader
    train_dataset=SpecDataLoader(args.data_train_file,args)
    
    train_sampler = DistributedBucketSampler(
      train_dataset,
      args.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=1,
      rank=0,
      shuffle=True)
    collate_fn = SpecSpeakerCollate()
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
    
    test_dataset = SpecDataLoader(args.data_test_file, args)
    test_loader = DataLoader(test_dataset, num_workers=8, shuffle=False,
    batch_size=args.batch_size, pin_memory=True,
    drop_last=False, collate_fn=collate_fn)
        
    # model initilize
    specEncoder=SpecEncoder(args)
    
    trainer=Trainer(specEncoder,args)
    
    # eval or training
    if args.do_eval:
        print('--------------------------begin testing--------------------------------')
        trainer.load(args.save_dir[:-3]+"_{}.pt".format(args.epoches))
        trainer.test(test_loader)
    else:
        print('--------------------------begin training-------------------------------')
        trainer.train(train_loader)
        print('--------------------------begin testing--------------------------------')
        trainer.test(test_loader)
        

if __name__=="__main__":
    main()    


