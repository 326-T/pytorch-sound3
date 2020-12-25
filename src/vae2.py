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

# +
import torch

import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as snsR
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# -


from modules.dataset import MyDataset, SoundDataset
from modules.myfunc import ans2index_label_color_marker
from modules.success_and_false import result, results_list

from vae import VAE_Trainer, Sound_Grad_CAM
from modules.dataset import SoundDataset2


class VAE2(nn.Module):
    def __init__(self,input_shape,z_shape=20,output_shape=11,beta=10):
        super(VAE2, self).__init__()
        
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
        self.enc_conv1 = nn.Conv1d(in_channels=2, out_channels=6, kernel_size=16, stride=10, padding=3, padding_mode='zeros')
        self.enc_conv2 = nn.Conv1d(in_channels=6, out_channels=6, kernel_size=16, stride=10, padding=3, padding_mode='zeros')
        self.enc_conv3 = nn.Conv1d(in_channels=6, out_channels=6, kernel_size=16, stride=10, padding=3, padding_mode='zeros')
        # z to mean
        self.encmean_fc11 = nn.Linear(int(input_shape/10/10/10*6), z_shape)
        # z to var
        self.encvar_fc12 = nn.Linear(int(input_shape/10/10/10*6), z_shape)
        
        # decoder
        self.dec_fc1 = nn.Linear(z_shape, int(input_shape/10/10/10*6))
        self.decoder = nn.Sequential()
        self.decoder.add_module('dec_deconv1', 
                                nn.ConvTranspose1d(in_channels=6, out_channels=6, kernel_size=16, stride=10, padding=3, padding_mode='zeros'))
        self.decoder.add_module('dec_relu1', nn.ReLU(True))
        self.decoder.add_module('dec_deconv2', 
                                nn.ConvTranspose1d(in_channels=6, out_channels=6, kernel_size=16, stride=10, padding=3, padding_mode='zeros'))
        self.decoder.add_module('dec_relu2', nn.ReLU(True))
        self.decoder.add_module('dec_deconv3', 
                                nn.ConvTranspose1d(in_channels=6, out_channels=2, kernel_size=16, stride=10, padding=3, padding_mode='zeros'))
        self.decoder.add_module('dec_sig1', nn.Sigmoid())
        
        # estimator
        self.classifier = nn.Sequential()
        #self.classifier.add_module('cla_fc1', nn.Linear(z_shape, 20))
        self.classifier.add_module('cla_fc1', nn.Linear(self.z_shape, self.output_shape))
        #self.classifier.add_module('cla_relu1', nn.ReLU(True))
        #self.classifier.add_module('cla_fc2', nn.Linear(20, self.output_shape))
        
    def encode(self, x):
        x = x.view(x.size()[0],2,-1)
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
        x = x.view(x.size()[0],6,-1)
        x = self.decoder(x)
        x = x.view(x.size()[0],2,-1)
        return x

    def forward(self, x):
        # encode
        mu, logvar = self.encode(x.view(-1, 2, self.input_shape).float())
        # reparameterize
        z = self.reparameterize(mu, logvar)
        pre_x = self.decode(z)
        # classification
        y = self.classifier(z)
        y = y.view(-1,self.output_shape)
        
        return pre_x, y, mu, logvar
    
    def valid(self, x):
        mu, logvar = self.encode(x.view(-1, 2, self.input_shape).float())
        y = self.classifier(mu)
        pre_x = self.decode(mu)
        y = y.view(-1,self.output_shape)
        
        return pre_x, y, mu, logvar
    
    def loss_function_vae(self, rec_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(rec_x, x.float(), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD * self.beta
    def loss_function_classifier(self, pre_y, y):
        return F.cross_entropy(pre_y, y)*self.input_shape*2
    
    def acc(self, pre_tar, tar):
        _, p_tar = torch.max(pre_tar, 1)
        correct = (p_tar==tar).sum().item()
        return correct


class VAE2_Trainer(VAE_Trainer):
    def __init__(self, dim_z = 20, output_shape=11, device="cuda", beta = 20):
        # prepare cuda device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # prepare dataset
        self.dataset = SoundDataset2(transform=transforms.ToTensor())
        # define model
        self.model = VAE2(input_shape=self.dataset.data_size, z_shape=dim_z, output_shape=output_shape, beta=beta).to(self.device)
        # define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.dim_z = dim_z

    def reconstruct(self, save_path = '../result/VAE2/reconstructed_sounds'):
        loader = torch.utils.data.DataLoader(self.dataset,batch_size=1,shuffle=False)
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x = x.to(self.device)
                recon_x, _, _, _ = self.model.forward(x)
                recon_x = recon_x.to('cpu').clone().numpy()
                x = x.to('cpu').clone().numpy()
                x = x.reshape(2, -1)
                recon_x = recon_x.reshape(2, -1)
                # to png
                fig, ax = plt.subplots(2,2,figsize=(16, 12))
                ax[0][0].set_title('L')
                ax[0][1].set_title('R')
                ax[1][0].set_title('reconstructed L')
                ax[1][1].set_title('reconstructed R')
                time = range(len(x[0]))
                for j in range(2):
                    ax[0][j].set_ylim(0, 1)
                    ax[1][j].set_ylim(0, 1)
                    ax[0][j].plot(time, x[j], linewidth = 1)
                    ax[1][j].plot(time, recon_x[j], linewidth = 1)
                plt.tight_layout()
                plt.savefig(save_path + '/' + self.dataset.filenames[i].split('.csv')[0] + '.png')
                plt.close()
                # to csv
                save_data = pd.DataFrame(data = recon_x)
                save_data.to_csv(save_path + '/'+ self.dataset.filenames[i], index = False)


def train_VAE2(key):
    vae = VAE2_trainer()
    #vae.load_weight(load_path =  '../result/VAE/' + key + '/vae')
    vae.load(key)
    vae.auto_train(1000, save_path = '../result/VAE2/' + key)
    vae.plot_z(save_path = '../result/VAE2/' + key + '/z_map.png')
    vae.reconstruct(save_path = '../result/VAE2/' + key + '/reconstructed')
    vae.save_weight(save_path = '../result/VAE2/' + key + '/vae')
    del vae


def VAE2_Grad_CAM(key):
    vae = VAE2_trainer(device='cpu')
    vae.load_weight(load_path='../result/VAE2/'+key+'/vae')
    vae.load(key)
    vae.model.eval()
    model_dict = dict(arch=vae.model, layer_name=vae.model.enc_conv3)
    Sound_Grad_CAM(vae, model_dict, key, '../result/VAE2'+key)
    Sound_Grad_CAM(vae, model_dict, key, '../result/VAE2/'+key, pp_mode=False)
    
    del vae


if __name__ == "__main__":
    keys = ['push', 'stop', 'flick']
    keys2 = ['drive' ,'block', 'push', 'stop', 'flick']
    for key in keys:
        train_VAE2(key)
    for key in keys2:
        VAE2_Grad_CAM(key)






