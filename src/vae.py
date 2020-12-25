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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.dataset import MyDataset, SoundDataset
from modules.trainer import Sound_Trainer
from modules.myfunc import make_meshgrid

class VAE(nn.Module):
    def __init__(self,input_shape,z_shape=20,output_shape=11,beta=10):
        super(VAE, self).__init__()
        
        self.input_shape = input_shape
        self.z_shape = z_shape
        self.output_shape = output_shape
        self.beta = beta
        
        # encoder
        #self.encoder = nn.Sequential()
        #self.encoder.add_module('enc_conv1', nn.Conv1d(in_channels=3, out_channels=9, kernel_size=16, stride=10, padding=6, padding_mode='zeros'))
        #self.encoder.add_module('enc_relu1', nn.ReLU(True))
        #self.encoder.add_module('enc_conv2', nn.Conv1d(in_channels=9, out_channels=9, kernel_size=16, stride=10, padding=6, padding_mode='zeros'))
        #self.encoder.add_module('enc_relu2', nn.ReLU(True))
        #self.encoder.add_module('enc_conv3', nn.Conv1d(in_channels=9, out_channels=9, kernel_size=16, stride=10, padding=6, padding_mode='zeros'))
        #self.encoder.add_module('enc_relu3', nn.ReLU(True))
        self.enc_conv1 = nn.Conv1d(in_channels=3, out_channels=9, kernel_size=16, stride=10, padding=3, padding_mode='zeros')
        self.enc_conv2 = nn.Conv1d(in_channels=9, out_channels=9, kernel_size=16, stride=10, padding=3, padding_mode='zeros')
        self.enc_conv3 = nn.Conv1d(in_channels=9, out_channels=9, kernel_size=16, stride=10, padding=3, padding_mode='zeros')
        # z to mean
        self.encmean_fc11 = nn.Linear(int(input_shape/10/10/10*9), z_shape)
        # z to var
        self.encvar_fc12 = nn.Linear(int(input_shape/10/10/10*9), z_shape)
        
        # decoder
        self.dec_fc1 = nn.Linear(z_shape, int(input_shape/10/10/10*9))
        self.decoder = nn.Sequential()
        self.decoder.add_module('dec_deconv1', 
                                nn.ConvTranspose1d(in_channels=9, out_channels=9, kernel_size=16, stride=10, padding=3, padding_mode='zeros'))
        self.decoder.add_module('dec_relu1', nn.ReLU(True))
        self.decoder.add_module('dec_deconv2', 
                                nn.ConvTranspose1d(in_channels=9, out_channels=9, kernel_size=16, stride=10, padding=3, padding_mode='zeros'))
        self.decoder.add_module('dec_relu2', nn.ReLU(True))
        self.decoder.add_module('dec_deconv3', 
                                nn.ConvTranspose1d(in_channels=9, out_channels=3, kernel_size=16, stride=10, padding=3, padding_mode='zeros'))
        self.decoder.add_module('dec_sig1', nn.Sigmoid())
        
        # estimator
        self.classifier = nn.Sequential()
        #self.classifier.add_module('cla_fc1', nn.Linear(z_shape, 20))
        self.classifier.add_module('cla_fc1', nn.Linear(self.z_shape, self.output_shape))
        #self.classifier.add_module('cla_relu1', nn.ReLU(True))
        #self.classifier.add_module('cla_fc2', nn.Linear(20, self.output_shape))
        
    def encode(self, x):
        x = x.view(x.size()[0],3,-1)
        #x = self.encoder(x)
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = x.view(x.size()[0], -1)
        return self.encmean_fc11(x), self.encvar_fc12(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        x = self.dec_fc1(z)
        x = x.view(x.size()[0],9,-1)
        x = self.decoder(x)
        x = x.view(x.size()[0],3,-1)
        return x

    def forward(self, x):
        # encode
        mu, logvar = self.encode(x.view(-1, 3, self.input_shape).float())
        # reparameterize
        z = self.reparameterize(mu, logvar)
        pre_x = self.decode(z)
        # classification
        y = self.classifier(z)
        y = y.view(-1,self.output_shape)
        
        return pre_x, y, mu, logvar
    
    def valid(self, x):
        mu, logvar = self.encode(x.view(-1, 3, self.input_shape).float())
        y = self.classifier(mu)
        pre_x = self.decode(mu)
        y = y.view(-1,self.output_shape)
        
        return pre_x, y, mu, logvar
    
    def classify(self, z):
        y = self.classifier(z.view(-1, self.z_shape))
        y = y.view(-1,self.output_shape)

        return y

    def loss_function_vae(self, rec_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(rec_x, x.float(), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD * self.beta
    def loss_function_classifier(self, pre_y, y):
        return F.cross_entropy(pre_y, y)*self.input_shape
    
    def acc(self, pre_tar, tar):
        _, p_tar = torch.max(pre_tar, 1)
        correct = (p_tar==tar).sum().item()
        return correct


class VAE_Trainer(Sound_Trainer):
    def __init__(self, dim_z = 20, output_shape=11, device="cuda", beta = 10):
        # prepare cuda device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # prepare dataset
        self.dataset = SoundDataset(transform=transforms.ToTensor())
        # define model
        self.model = VAE(input_shape=self.dataset.data_size, z_shape=dim_z, output_shape=output_shape, beta=beta).to(self.device)
        # define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.dim_z = dim_z
    
    def train(self, epoch, max_epoch):
        # train mode
        self.model.train()
        train_loss = 0
        train_loss_vae = 0
        train_loss_classifier = 0
        train_acc = 0
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            rec_x, pre_y, mu, logvar = self.model(x)
            loss_vae = self.model.loss_function_vae(rec_x, x, mu, logvar)
            loss_classifier = self.model.loss_function_classifier(pre_y, y)
            loss = loss_vae + loss_classifier
            # backward
            loss.backward()
            # update the parameter
            self.optimizer.step()
            # logging
            train_loss += loss.item()
            train_loss_vae += loss_vae.item()
            train_loss_classifier += loss_classifier.item()
            train_acc += self.model.acc(pre_y, y)
            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(x)))
                
        train_loss /= len(self.train_loader.dataset)
        train_loss_vae /= len(self.train_loader.dataset)
        train_loss_classifier /= len(self.train_loader.dataset)
        train_acc /= len(self.train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
        
        return train_loss, train_loss_vae, train_loss_classifier, train_acc

    def valid(self, epoch):
        # test mode
        self.model.eval()
        valid_loss = 0
        valid_loss_vae = 0
        valid_loss_classifier = 0
        valid_acc = 0
        # test mode
        with torch.no_grad():
            for i, (x, y) in enumerate(self.valid_loader):
                x, y = x.to(self.device), y.to(self.device)
                rec_x, pre_y, mu, logvar = self.model.valid(x)
                loss_vae = self.model.loss_function_vae(rec_x, x, mu, logvar)
                loss_classifier = self.model.loss_function_classifier(pre_y, y)
                loss = loss_vae + loss_classifier
                valid_loss += loss.item()
                valid_loss_vae += loss_vae.item()
                valid_loss_classifier += loss_classifier.item()
                valid_acc += self.model.acc(pre_y, y)

        valid_loss /= len(self.valid_loader.dataset)
        valid_loss_vae /= len(self.valid_loader.dataset)
        valid_loss_classifier /= len(self.valid_loader.dataset)
        valid_acc /= len(self.valid_loader.dataset)
        print('====> Validation set loss: {:.4f}'.format(valid_loss))
        
        return valid_loss, valid_loss_vae, valid_loss_classifier, valid_acc
        
    def auto_train(self, max_epoch, save_path = None):
        train_set, valid_set = torch.utils.data.random_split(self.dataset, [int(len(self.dataset)*0.8), len(self.dataset) - int(len(self.dataset)*0.8)])
        self.train_loader = torch.utils.data.DataLoader(train_set,batch_size=10,shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(valid_set,batch_size=10,shuffle=True)
        
        train_loss = []
        train_loss_vae = []
        train_loss_classifier = []
        train_acc = []
        valid_loss = []
        valid_loss_vae = []
        valid_loss_classifier = []
        valid_acc = []
        for epoch in range(1,max_epoch):
            t_loss, t_loss_vae, t_loss_classifier, t_acc = self.train(epoch, max_epoch)
            v_loss, v_loss_vae, v_loss_classifier, v_acc = self.valid(epoch)
            train_loss.append(t_loss)
            train_loss_vae.append(t_loss_vae)
            train_loss_classifier.append(t_loss_classifier)
            train_acc.append(t_acc)
            valid_loss.append(v_loss)
            valid_loss_vae.append(v_loss_vae)
            valid_loss_classifier.append(v_loss_classifier)
            valid_acc.append(v_acc)
        # plot result
        if save_path is not None:
            fig, ax = plt.subplots(4,1,figsize=(8, 16))
            ax[0].set_title('Loss')
            ax[1].set_title('VAE Loss')
            ax[2].set_title('Classifier Loss')
            ax[3].set_title('Accuracy')
            for i in range(3):
                ax[i].set_xlabel('Epochs')
                ax[i].set_ylabel('Loss')
            ax[0].plot(range(1,max_epoch),train_loss,label="train")
            ax[0].plot(range(1,max_epoch),valid_loss,label="validation")
            ax[1].plot(range(1,max_epoch),train_loss_vae,label="train")
            ax[1].plot(range(1,max_epoch),valid_loss_vae,label="validation")
            ax[2].plot(range(1,max_epoch),train_loss_classifier,label="train")
            ax[2].plot(range(1,max_epoch),valid_loss_classifier,label="validation")
            ax[3].set_xlabel('Epochs')
            ax[3].set_ylabel('Accuracy')
            ax[3].plot(range(1,max_epoch),train_acc,label="train")
            ax[3].plot(range(1,max_epoch),valid_acc,label="validation")
            for i in range(3):
                ax[i].legend()
            plt.tight_layout()
            plt.savefig(save_path+'/loss.png')
            plt.close()
        
    def export_latent_space(self, save_path='../result/VAE/model/result.png'):
        # print z all data
        loader = torch.utils.data.DataLoader(self.dataset,batch_size=len(self.dataset),shuffle=False)
        all_z = []
        all_ans = []
        self.model.eval()
        with torch.no_grad():
            for i, (data, ans) in enumerate(loader):
                data = data.to(self.device)
                _, _, mu, logvar = self.model.valid(data)
                all_z = np.append(all_z, mu.to('cpu').clone().numpy())
                all_ans = np.append(all_ans, ans.to('cpu').clone().numpy())

        all_z = np.array(all_z).reshape(-1, self.model.z_shape)
        all_ans = np.array(all_ans).reshape(-1)

        mesh_z = make_meshgrid(all_z)
        mesh = torch.from_numpy(mesh_z.astype('float')).clone()
        mesh = mesh.to(self.device)
        mesh_ans = self.model.classify(mesh)
        mesh_ans = mesh_ans.to('cpu').clone().numpy()
        idx = np.argsort(mesh_ans)
        mesh_z = mesh_z[idx]

        ica = self.plot_z(all_z, all_ans, save_path, mesh_z, mesh_ans)        
        return all_z, all_ans, ica

    def reconstruct(self, save_path = '../result/VAE/reconstructed_sounds'):
        loader = torch.utils.data.DataLoader(self.dataset,batch_size=1,shuffle=False)
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                recon_x, _, _, _ = self.model.forward(x)
                recon_x = recon_x.to('cpu').clone().numpy()
                x = x.to('cpu').clone().numpy()
                x = x.reshape(3, -1)
                recon_x = recon_x.reshape(3, -1)
                # to png
        self.plot_reconstruct(x, recon_x, save_path)

def train_VAE(key):
    vae = VAE_Trainer(beta=20)
    vae.load(key)
    vae.auto_train(1000, save_path = '../result/VAE/' + key)
    vae.export_latent_space(save_path = '../result/VAE/' + key + '/z_map.png')
    vae.save_weight(save_path = '../result/VAE/' + key + '/vae')
    del vae

def plot_latent_VAE(key):
    vae = VAE_Trainer()
    vae.load_weight(load_path =  '../result/VAE/' + key + '/vae')
    vae.load(key)
    vae.export_latent_space(save_path = '../result/VAE/' + key + '/z_map.png')
    del vae

def reconstruct_VAE(key):
    vae = VAE_Trainer()
    vae.load_weight(load_path =  '../result/VAE/' + key + '/vae')
    vae.load(key)
    vae.reconstruct(save_path = '../result/VAE/' + key + '/reconstructed')


from modules.gradcam import GradCAM, GradCAMpp

def Sound_Grad_CAM(vae, model_dict, key, save_path, pp_mode=True):
    
    if pp_mode:
        gradcam = GradCAMpp(model_dict)
    else:
        gradcam = GradCAM(model_dict)
    
    src_loader = torch.utils.data.DataLoader(vae.dataset,batch_size=1,shuffle=False)
    # test mode
    for i, (data, label) in enumerate(src_loader):
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


def Grad_CAM_VAE(key):
    vae = VAE_Trainer(device='cpu')
    vae.load_weight(load_path='../result/VAE/'+key+'/vae')
    vae.load(key)
    vae.model.eval()
    model_dict = dict(arch=vae.model, layer_name=vae.model.enc_conv3)
    Sound_Grad_CAM(vae, model_dict, key, '../result/VAE/'+key)
    Sound_Grad_CAM(vae, model_dict, key, '../result/VAE/'+key, pp_mode=False)
    
    del vae


if __name__ == "__main__":
    keys = ['drive', 'block', 'push', 'stop', 'flick']
    for key in keys:
        plot_latent_VAE(key)














