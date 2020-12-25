# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

from modules.dataset import MyDataset, IMUDataset
from modules.trainer import IMU_Trainer

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class IMU_VDANN(nn.Module):
    def __init__(self,input_dim,z_dim=20,label_num=2,domain_num=11,beta=10,reverse=True):
        super(IMU_VDANN, self).__init__()
        
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.label_num = label_num
        self.domain_num = domain_num
        self.beta = beta
        self.reverse = reverse
        
        self.enc_conv1 = nn.Conv1d(in_channels=9, out_channels=9, kernel_size=10, stride=5, padding=3, padding_mode='zeros')
        self.enc_conv2 = nn.Conv1d(in_channels=9, out_channels=9, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
        # z to mean
        self.encmean_fc11 = nn.Linear(90, self.z_dim)
        # z to var
        self.encvar_fc12 = nn.Linear(90, self.z_dim)

        # estimator
        self.clf_label = nn.Sequential()
        self.clf_label.add_module('clfl_fc1', nn.Linear(self.z_dim, self.label_num))

        self.clf_domain = nn.Sequential()
        self.clf_domain.add_module('clfd_fc1', nn.Linear(self.z_dim, self.z_dim))
        self.clf_domain.add_module('clfd_relu1', nn.ReLU(True))
        self.clf_domain.add_module('clfd_fc2', nn.Linear(self.z_dim, self.z_dim))
        self.clf_domain.add_module('clfd_relu2', nn.ReLU(True))
        self.clf_domain.add_module('clfd_fc2', nn.Linear(self.z_dim, self.domain_num))
        
    def encode(self, x):
        x = x.view(x.size()[0],9,-1)
        #x = self.encoder(x)
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = x.view(x.size()[0], -1)
        return self.encmean_fc11(x), self.encvar_fc12(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    #def decode(self, z):
    #    x = self.dec_fc1(z)
    #    x = x.view(x.size()[0],9,-1)
    #    x = self.decoder(x)
    #    x = x.view(x.size()[0],3,-1)
    #    return x

    def forward(self, x, alpha=0):
        # encode
        mu, logvar = self.encode(x.view(-1, 9, self.input_dim).float())
        # reparameterize
        z = self.reparameterize(mu, logvar)
        
        #label classification
        pre_label = self.clf_label(z)
        pre_label = pre_label.view(-1,self.label_num)
        # domain classification
        if self.reverse:
            reverse_d = ReverseLayerF.apply(z, alpha)
            pre_domain = self.clf_domain(reverse_d)
        else:
            pre_domain = self.clf_domain(z)
        pre_domain = pre_domain.view(-1,self.domain_num)
        return pre_label, pre_domain, mu, logvar
    
    def valid(self, x):
        mu, logvar = self.encode(x.view(-1, 9, self.input_dim).float())
        pre_label = self.clf_label(mu)
        pre_label = pre_label.view(-1,self.label_num)
        pre_domain = self.clf_domain(mu)
        pre_domain = pre_domain.view(-1,self.domain_num)
        
        return pre_label, pre_domain, mu, logvar
    
    def loss_KL(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD * self.beta
    
    def loss_label(self, pre_label, label):
        return F.cross_entropy(pre_label, label)*self.input_dim
    
    def loss_domain(self, pre_domain, domain):
        return F.cross_entropy(pre_domain, domain)*self.input_dim
    
    def acc(self, pre_tar, tar):
        _, p_tar = torch.max(pre_tar, 1)
        correct = (p_tar==tar).sum().item()
        return correct


class IMU_VDANN_Trainer(IMU_Trainer):
    def __init__(self, z_dim = 20, label_num = 2, domain_num = 11, beta = 10, reverse=True, device="cpu"):
        self.z_dim = z_dim
        self.label_num = label_num
        self.domain_num = domain_num
        self.beta = beta
        self.reverse = reverse
        
        # prepare cuda device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # prepare dataset
        self.dataset = IMUDataset(transforms.ToTensor(), mode='DANN')
        # define model
        self.model = IMU_VDANN(input_dim=self.dataset.data_size, z_dim=z_dim, label_num=label_num, domain_num=domain_num, beta=beta).to(self.device)
        # define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, epoch, max_epoch):
        # train mode
        self.model.train()
        train_loss = 0
        train_loss_KL = 0
        train_loss_label = 0
        train_loss_domain = 0
        train_acc_label = 0
        train_acc_domain = 0
        # calculate reverse layer alpha
        p = float(epoch * len(self.train_loader.dataset)) / max_epoch / len(self.train_loader.dataset)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        for data, label, domain in self.train_loader:
            data, label, domain = data.to(self.device), label.to(self.device), domain.to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            pre_label, pre_domain, mu, logvar = self.model(data, alpha)
            loss_KL = self.model.loss_KL(mu, logvar)
            loss_label = self.model.loss_label(pre_label, label)
            loss_domain = self.model.loss_domain(pre_domain, domain)
            loss = loss_KL + loss_label + loss_domain
            # backward
            loss.backward()
            # update the parameter
            self.optimizer.step()
            # logging
            train_loss += loss.item()
            train_loss_KL += loss_KL.item()
            train_loss_label += loss_label.item()
            train_loss_domain += loss_domain.item()
            train_acc_label += self.model.acc(pre_label, label)
            train_acc_domain += self.model.acc(pre_domain, domain)
                
        train_loss /= len(self.train_loader.dataset)
        train_loss_KL /= len(self.train_loader.dataset)
        train_loss_label /= len(self.train_loader.dataset)
        train_loss_domain /= len(self.train_loader.dataset)
        train_acc_label /= len(self.train_loader.dataset)
        train_acc_domain /= len(self.train_loader.dataset)
        
        print('====> Epoch: {} Average acc: {:.4f}'.format(epoch, train_acc_label))
        
        return train_loss, train_loss_KL, train_loss_label, train_loss_domain, train_acc_label, train_acc_domain

    def valid(self, epoch):
        # valid mode
        self.model.eval()
        valid_loss = 0
        valid_loss_KL = 0
        valid_loss_label = 0
        valid_loss_domain = 0
        valid_acc_label = 0
        valid_acc_domain = 0
        with torch.no_grad():
            for data, label, domain in self.valid_loader:
                data, label, domain = data.to(self.device), label.to(self.device), domain.to(self.device)
                pre_label, pre_domain, mu, logvar = self.model.valid(data)
                loss_KL = self.model.loss_KL(mu, logvar)
                loss_label = self.model.loss_label(pre_label, label)
                loss_domain = self.model.loss_domain(pre_domain, domain)
                loss = loss_KL + loss_label + loss_domain
                # logging
                valid_loss += loss.item()
                valid_loss_KL += loss_KL.item()
                valid_loss_label += loss_label.item()
                valid_loss_domain += loss_domain.item()
                valid_acc_label += self.model.acc(pre_label, label)
                valid_acc_domain += self.model.acc(pre_domain, domain)
                    
        valid_loss /= len(self.valid_loader.dataset)
        valid_loss_KL /= len(self.valid_loader.dataset)
        valid_loss_label /= len(self.valid_loader.dataset)
        valid_loss_domain /= len(self.valid_loader.dataset)
        valid_acc_label /= len(self.valid_loader.dataset)
        valid_acc_domain /= len(self.valid_loader.dataset)
        
        print('====> Validation set acc: {:.4f}'.format(valid_acc_label))
        
        return valid_loss, valid_loss_KL, valid_loss_label, valid_loss_domain, valid_acc_label, valid_acc_domain
        
    def auto_train(self, max_epoch, save_path = None):
        train_set, valid_set = torch.utils.data.random_split(self.dataset, [int(len(self.dataset)*0.8), len(self.dataset) - int(len(self.dataset)*0.8)])
        self.train_loader = torch.utils.data.DataLoader(train_set,batch_size=10,shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(valid_set,batch_size=10,shuffle=True)
        
        train_loss = []
        train_loss_KL = []
        train_loss_label = []
        train_loss_domain = []
        train_acc_label = []
        train_acc_domain = []
        valid_loss = []
        valid_loss_KL = []
        valid_loss_label = []
        valid_loss_domain = []
        valid_acc_label = []
        valid_acc_domain = []
        for epoch in range(1,max_epoch):
            t_loss, t_loss_KL, t_loss_label, t_loss_domain, t_acc_label, t_acc_domain = self.train(epoch, max_epoch)
            v_loss, v_loss_KL, v_loss_label, v_loss_domain, v_acc_label, v_acc_domain = self.valid(epoch)
            train_loss.append(t_loss)
            train_loss_KL.append(t_loss_KL)
            train_loss_label.append(t_loss_label)
            train_loss_domain.append(t_loss_domain)
            train_acc_label.append(t_acc_label)
            train_acc_domain.append(t_acc_domain)
            valid_loss.append(v_loss)
            valid_loss_KL.append(v_loss_KL)
            valid_loss_label.append(v_loss_label)
            valid_loss_domain.append(v_loss_domain)
            valid_acc_label.append(v_acc_label)
            valid_acc_domain.append(v_acc_domain)
        # plot result
        if save_path is not None:
            fig, ax = plt.subplots(4,1,figsize=(8, 16))
            ax[0].set_title('Loss')
            ax[1].set_title('KL Loss')
            ax[2].set_title('Label Loss')
            ax[3].set_title('Domain loss')
            for i in range(4):
                ax[i].set_xlabel('Epochs')
                ax[i].set_ylabel('Loss')
            ax[0].plot(range(1,max_epoch),train_loss,label="train")
            ax[0].plot(range(1,max_epoch),valid_loss,label="validation")
            ax[1].plot(range(1,max_epoch),train_loss_KL,label="train")
            ax[1].plot(range(1,max_epoch),valid_loss_KL,label="validation")
            ax[2].plot(range(1,max_epoch),train_loss_label,label="train")
            ax[2].plot(range(1,max_epoch),valid_loss_label,label="validation")
            ax[3].plot(range(1,max_epoch),train_loss_domain,label="train")
            ax[3].plot(range(1,max_epoch),valid_loss_domain,label="validation")
            for i in range(4):
                ax[i].legend()
            plt.tight_layout()
            plt.savefig(save_path+'/vdann_loss.png')
            plt.close()

            fig, ax = plt.subplots(2,1,figsize=(8, 16))
            ax[0].set_title('Label Accuracy')
            ax[1].set_title('Domain Accuracy')
            for i in range(2):
                ax[i].set_xlabel('Epochs')
                ax[i].set_ylabel('Accuracy')
            ax[0].plot(range(1,max_epoch),train_acc_label,label="train")
            ax[0].plot(range(1,max_epoch),valid_acc_label,label="validation")
            ax[1].plot(range(1,max_epoch),train_acc_domain,label="train")
            ax[1].plot(range(1,max_epoch),valid_acc_domain,label="validation")
            for i in range(2):
                ax[i].legend()
            plt.tight_layout()
            plt.savefig(save_path+'/vdann_acc.png')
            plt.close()

    def export_latent_space(self, save_path='../result/IMU_VDANN/model/result.png'):
        # print z all data
        loader = torch.utils.data.DataLoader(self.dataset,batch_size=len(self.dataset),shuffle=False)
        all_z = []
        all_ans = []
        self.model.eval()
        with torch.no_grad():
            for data, label, domain in loader:
                data = data.to(self.device)
                _, _, mu, logvar = self.model.valid(data)
                all_z = np.append(all_z, mu.to('cpu').clone().numpy())
                all_ans = np.append(all_ans, domain.to('cpu').clone().numpy())

        all_z = np.array(all_z).reshape(-1, self.model.z_dim)
        all_ans = np.array(all_ans).reshape(-1)
        
        ica = self.plot_z(all_z, all_ans, save_path)
        return all_z, all_ans, ica

def train_VDANN(key):
    vdann = IMU_VDANN_Trainer(beta=10)
    vdann.load(key)
    vdann.auto_train(1000, save_path = '../result/IMU_VDANN/' + key)
    vdann.export_latent_space(save_path = '../result/IMU_VDANN/' + key + '/z_map.png')
    vdann.save_weight(save_path = '../result/IMU_VDANN/' + key + '/vdann')
    del vdann

from modules.gradcam import GradCAM, GradCAMpp

def IMU_Grad_CAM(vae, model_dict, key, save_path, pp_mode=True):
    
    if pp_mode:
        gradcam = GradCAMpp(model_dict)
    else:
        gradcam = GradCAM(model_dict)
    
    src_loader = torch.utils.data.DataLoader(vae.dataset,batch_size=1,shuffle=False)
    # test mode
    for i, (data, label, domain) in enumerate(src_loader):
        pre_class, mask = gradcam(data, label)
        data=data.detach().clone().numpy()
        data=(data[0]-np.min(data[0]))/(np.max(data[0])-np.min(data[0]))
        mask=mask.detach().clone().numpy()
        
        fig, ax = plt.subplots(3,1,figsize=(16,12))
        ax[0].set_title('L')
        ax[1].set_title('C')
        ax[2].set_title('R')
        for j in range(3):
            ax[j].plot(range(len(mask[0])), mask[0], label='Grad_CAM')
            ax[j].plot(range(len(data[j])), data[j], label='input_data')
        plt.tight_layout()
        if pp_mode:
            plt.savefig(save_path+'/Grad_CAMpp/'+vae.dataset.filenames[i].split('.csv')[0]+'.png')
        else:
            plt.savefig(save_path+'/Grad_CAM/'+vae.dataset.filenames[i].split('.csv')[0]+'.png')
                       
        plt.close()

def Grad_CAM_VDANN(key):
    vdann = IMU_VDANN_Trainer(device='cpu')
    vdann.load_weight(load_path='../result/IMU_VDANN/'+key+'/vdann')
    vdann.load(key)
    vdann.model.eval()
    model_dict = dict(arch=vdann.model, layer_name=vdann.model.enc_conv3)
    IMU_Grad_CAM(vdann, model_dict, key, '../result/IMU_VDANN/'+key)
    IMU_Grad_CAM(vdann, model_dict, key, '../result/IMU_VDANN/'+key, pp_mode=False)
    
    del vdann

if __name__ == "__main__":
    keys = ['drive', 'block', 'push', 'stop', 'flick']
    for key in keys:
        train_VDANN(key)

















