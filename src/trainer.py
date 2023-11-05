import gc
import torch
import numpy as np
from models import KMeans
from modules import InfoNCE
from tqdm import tqdm
from torch.optim import AdamW
import matplotlib.pyplot as plt
from utils import plot_scatter
import joblib
import pickle

class Trainer:
    def __init__(self,specEncoder,args):
        self.enc=specEncoder
        self.args=args
        

        self.cluster=KMeans(
            num_cluster=self.args.cluster_num,
            seed=self.args.seed,
            hidden_size=self.args.hidden_channels,
            gpu_id=self.args.gpu_id,
            device=torch.device("cuda"),
        )
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        
        if self.cuda_condition:
            self.enc.cuda()

        
        self.cl=InfoNCE(self.args.temp,self.args.sim,self.args.f_neg_mask)
        self.optimizer= torch.optim.AdamW(
                self.enc.parameters(), 
                self.args.learning_rate, 
                betas=self.args.betas, 
                eps=self.args.eps)
        
    def save(self, file_name):
        torch.save(self.enc.cpu().state_dict(), file_name)
        self.enc.to(self.device)

    def load(self, file_name):
        self.enc.load_state_dict(torch.load(file_name))
        
    def train(self,train_loader):
        epoch=0
        train_data_iter=tqdm(enumerate(train_loader),total=len(train_loader))
        print("-------------------------begin clustering-------------------------")
        # clustering
        self.enc.eval()
        kmeans_training_data=[]
        for batch_idx, (spec_batch) in train_data_iter:
            spec_batch=tuple(t.to(self.device) for t in spec_batch)
            spec, spec_lengths, _, _, _,_,_=spec_batch
            # print("spec: ",spec)
            # print("spec_lengths: ",spec_lengths)

            z, m_q, logs_q, y_mask = self.enc(spec, spec_lengths) 
            # print("z shape: {}".format(z.shape))
            # [B,H,L]
            # sum,mean
            z_output=torch.mean(z,dim=2)
            z_output=z_output.detach().cpu().numpy()
            # print("z_output shape: {}".format(z_output.shape))
            kmeans_training_data.append(z_output)
            # print(z_output.shape)
            
        

        kmeans_training_data=np.concatenate(kmeans_training_data,axis=0)
        # print("t: ",kmeans_training_data.shape)
        for i,cluster in tqdm(enumerate([self.cluster]),total=len([kmeans_training_data])):
            cluster.train(kmeans_training_data)
            self.cluster=cluster
        # clear memory
        del kmeans_training_data
        gc.collect()
           
        # training
        self.enc.train()
        # drawing
        cl_losses_draw=[]
        scl_losses_draw=[]
        joint_losses_draw=[]
        while epoch<self.args.epoches:
            joint_losses=0.
            cl_losses=0.
            scl_losses=0.
            train_data_iter=tqdm(enumerate(train_loader),total=len(train_loader))
            for batch_idx, (spec_batch) in  train_data_iter:
                spec_batch=tuple(t.to(self.device) for t in spec_batch)
                
                spec, spec_lengths,wave,wave_lengths,speakers, spec_aug0, spec_aug1=spec_batch
                                
                z0, m_q_0, logs_q_0, y0_mask = self.enc(spec_aug0, spec_lengths)
                z1, m_q_1, logs_q_1, y1_mask = self.enc(spec_aug1, spec_lengths)
                
                # print(z0.shape,spec.shape)
                label_ids=torch.cat([speakers,speakers])
                # ------------self-supervised contrastive loss
                z0_=torch.mean(z0,dim=2)
                z1_=torch.mean(z1,dim=2)
                
                cl_loss=self.cl.info_nce(z0_,z1_,label_ids)
                # ------------supervised contrastive loss
                z_0=z0_.detach().cpu().numpy()
                z_1=z1_.detach().cpu().numpy()
                
                                                      # cluster query
                c_0_id,c_0=self.cluster.query(z_0)
                c_1_id,c_1=self.cluster.query(z_1)
                
                                            
                c_1 = c_1.view(c_1.shape[0], -1) # [BxH]
                c_0 = c_0.view(c_0.shape[0], -1) # [BxH]

                scl_loss=self.cl.info_nce(c_0,z0_)+self.cl.info_nce(c_1,z1_)
                
                joint_loss=self.args.lam * cl_loss + self.args.beta * scl_loss
                
                
                joint_losses += joint_loss.item()
                cl_losses += cl_loss.item()
                scl_losses += scl_loss.item()
                
                
                # print("loss {0} {1} {2}".format(cl_loss,scl_loss,joint_loss)) 


                self.optimizer.zero_grad()
                joint_loss.backward()
                self.optimizer.step()                
        
            epoch += 1
            post_fix = {
                "epoch": epoch,
                "cl_avg_loss": "{:.4f}".format(cl_losses / len(train_data_iter)),
                "scl_avg_loss": "{:.4f}".format(scl_losses/ len(train_data_iter)),
                "joint_avg_loss": "{:.4f}".format(joint_losses / len(train_data_iter)),
            }
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")
            # draw
            cl_losses_draw.append(cl_losses)
            scl_losses_draw.append(scl_losses)
            joint_losses_draw.append(joint_losses)
        # save sepc encoder
        # torch.save(self.enc.state_dict(),self.args.save_dir)
            if epoch%10==0:
                self.save(self.args.save_dir[:-3]+"_{}.pt".format(epoch))
            if epoch%self.args.epoches==0:
                plt.plot(range(len(cl_losses_draw)),cl_losses_draw,c="pink",label="cl")
                plt.plot(range(len(scl_losses_draw)),scl_losses_draw,c="red",label="scl")
                plt.savefig("loss.png")

                    
            
    def test(self,test_loader):
        self.enc.eval()
        test_data_iter=tqdm(enumerate(test_loader),total=len(test_loader))
        kmeans_testing_data=[]
        spk_ids=[]
        for batch_idx, (spec_batch) in test_data_iter:
            spec_batch=tuple(t.to(self.device) for t in spec_batch)
            spec, spec_lengths,_,_,speaker, _, _=spec_batch
            z, m_q, logs_q, y_mask = self.enc(spec, spec_lengths)
            
            z_output=torch.mean(z,dim=2)
            z_output=z_output.detach().cpu().numpy()
            kmeans_testing_data.append(z_output)
            spk_ids.append(speaker.detach().cpu().numpy())
            
        
        specs=np.concatenate(kmeans_testing_data,axis=0)   
        spk_id_data=np.concatenate(spk_ids,axis=0)        
        
        # clear memory
        del kmeans_testing_data
        gc.collect()
        
        
        plot_scatter(specs,spk_id_data,self.args.epoches)
            
            
        
            
            
            
            
            
            
            
            
            
            
 
            
