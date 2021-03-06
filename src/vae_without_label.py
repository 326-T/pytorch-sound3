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
import seaborn as sns
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# -


from modules.dataset import SoundDataset
from modules.myfunc import ans2index_label_color_marker
from modules.success_and_false import result, results_list


class VAE_without_label(nn.Module):
    def __init__(self,input_shape,z_shape=20,output_shape=11):
        super(VAE_without_label, self).__init__()
        
        self.input_shape = input_shape
        self.z_shape = z_shape
        self.output_shape = output_shape
        
        # encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('enc_conv1', nn.Conv1d(in_channels=3, out_channels=9, kernel_size=16, stride=10, padding=6, padding_mode='zeros'))
        self.encoder.add_module('enc_relu1', nn.ReLU(True))
        self.encoder.add_module('enc_conv2', nn.Conv1d(in_channels=9, out_channels=9, kernel_size=16, stride=10, padding=6, padding_mode='zeros'))
        self.encoder.add_module('enc_relu2', nn.ReLU(True))
        self.encoder.add_module('enc_conv3', nn.Conv1d(in_channels=9, out_channels=9, kernel_size=16, stride=10, padding=6, padding_mode='zeros'))
        self.encoder.add_module('enc_relu3', nn.ReLU(True))
        # z to mean
        self.encmean_fc11 = nn.Linear(int(input_shape/10/10/10*9), z_shape)
        # z to var
        self.encvar_fc12 = nn.Linear(int(input_shape/10/10/10*9), z_shape)
        
        # decoder
        self.dec_fc1 = nn.Linear(z_shape, int(input_shape/10/10/10*9))
        self.decoder = nn.Sequential()
        self.decoder.add_module('dec_deconv1', nn.ConvTranspose1d(in_channels=9, out_channels=9, kernel_size=16, stride=10, padding=3, padding_mode='zeros'))
        self.decoder.add_module('dec_relu1', nn.ReLU(True))
        self.decoder.add_module('dec_deconv2', nn.ConvTranspose1d(in_channels=9, out_channels=9, kernel_size=16, stride=10, padding=3, padding_mode='zeros'))
        self.decoder.add_module('dec_relu2', nn.ReLU(True))
        self.decoder.add_module('dec_deconv3', nn.ConvTranspose1d(in_channels=9, out_channels=3, kernel_size=16, stride=10, padding=3, padding_mode='zeros'))
        self.decoder.add_module('dec_sig1', nn.Sigmoid())
        
    def encode(self, x):
        x = x.view(x.size()[0],3,-1)
        x = self.encoder(x)
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
        
        return pre_x, mu, logvar
    
    def valid(self, x):
        mu, logvar = self.encode(x.view(-1, 3, self.input_shape).float())
        pre_x = self.decode(mu)
        
        return pre_x, mu, logvar
    
    def loss_function_vae(self, rec_x, x, mu, logvar, beta=2):
        BCE = F.binary_cross_entropy(rec_x, x.float(), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD * beta


class VAE_without_label_trainer():
    def __init__(self, dim_z = 20, device="cuda"):
        # prepare cuda device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        # prepare dataset
        self.dataset = SoundDataset(transform=transforms.ToTensor())
        # define model
        self.model = VAE_without_label(self.dataset.data_size, dim_z).to(self.device)
        # define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.dim_z = dim_z
    
    #def __del__(self):
    #    self.save_weight()
    
    def load(self, key):
        self.dataset.load("../data/sounds/raw/sub1", key, 0)
        self.dataset.load("../data/sounds/raw/sub2", key, 1)
        self.dataset.load("../data/sounds/raw/sub3", key, 2)
        self.dataset.load("../data/sounds/raw/sub4", key, 3)
        self.dataset.load("../data/sounds/raw/sub5", key, 4)
        self.dataset.load("../data/sounds/raw/sub6", key, 5)
        self.dataset.load("../data/sounds/raw/sub7", key, 6)
        self.dataset.load("../data/sounds/raw/sub8", key, 7)
        self.dataset.load("../data/sounds/raw/sub9", key, 8)
        self.dataset.load("../data/sounds/raw/sub10", key, 9)
        self.dataset.load("../data/sounds/raw/sub11", key, 10)
        self.dataset.normalize()
    
    def train(self, epoch, max_epoch):
        # train mode
        self.model.train()
        train_loss = 0
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            rec_x, mu, logvar = self.model(x)
            loss = self.model.loss_function_vae(rec_x, x, mu, logvar)
            # backward
            loss.backward()
            # update the parameter
            self.optimizer.step()
            # logging
            train_loss += loss.item()
            if batch_idx % 1 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(x)))
                
        train_loss /= len(self.train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
        
        return train_loss

    def valid(self, epoch):
        # test mode
        self.model.eval()
        valid_loss = 0
        # test mode
        with torch.no_grad():
            for i, (x, y) in enumerate(self.valid_loader):
                x = x.to(self.device)
                rec_x, mu, logvar = self.model.valid(x)
                loss = self.model.loss_function_vae(rec_x, x, mu, logvar)
                valid_loss += loss.item()

        valid_loss /= len(self.valid_loader.dataset)
        print('====> Validation set loss: {:.4f}'.format(valid_loss))
        
        return valid_loss
        
    def auto_train(self, max_epoch, save_path = '../result/VAE/model'):
        train_set, valid_set = torch.utils.data.random_split(self.dataset, [int(len(self.dataset)*0.8), len(self.dataset) - int(len(self.dataset)*0.8)])
        self.train_loader = torch.utils.data.DataLoader(train_set,batch_size=10,shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(valid_set,batch_size=10,shuffle=True)
        
        train_loss = []
        valid_loss = []
        for epoch in range(1,max_epoch):
            t_loss = self.train(epoch, max_epoch)
            v_loss = self.valid(epoch)
            train_loss.append(t_loss)
            valid_loss.append(v_loss)
        # plot result
        fig, ax = plt.subplots(1,1,figsize=(8, 4))
        ax.set_title('Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.plot(range(1,max_epoch),train_loss,label="train")
        ax.plot(range(1,max_epoch),valid_loss,label="validation")
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path+'/loss.png')
        plt.close()
        
    def save_weight(self, save_path = '../result/VAE/model/vae'):
        torch.save(self.model.state_dict(), save_path)
        
    def load_weight(self, load_path = '../result/VAE/model/vae'):
        self.model.load_state_dict(torch.load(load_path))
    
    
    def plot_z(self, save_path='../result/VAE/model/result.png'):
        # print z all data
        loader = torch.utils.data.DataLoader(self.dataset,batch_size=len(self.dataset),shuffle=False)
        all_z = []
        all_ans = []
        self.model.eval()
        with torch.no_grad():
            for i, (data, ans) in enumerate(loader):
                data = data.to(self.device)
                _, mu, logvar = self.model.forward(data)
                all_z = np.append(all_z, mu.to('cpu').clone().numpy())
                all_ans = np.append(all_ans, ans.to('cpu').clone().numpy())

        all_z = np.array(all_z).reshape(-1, self.model.z_shape)
        all_ans = np.array(all_ans).reshape(-1)
        
        # LDA
        self.lda = LDA(n_components = 2)
        self.lda.fit(all_z, all_ans)
        lda_z = self.lda.transform(all_z)
        lda_z = lda_z.transpose()
        
        z_xrange = [np.min(lda_z[0]), np.max(lda_z[0])]
        z_yrange = [np.min(lda_z[1]), np.max(lda_z[1])]        
        plot_z(lda_z[0], lda_z[1], all_ans, "z map", save_path.split('.png')[0] + '_LDA.png', z_xrange, z_yrange)
        plot_z_each(lda_z, all_ans, self.dataset.filenames, '../data/succeed_list.csv', "z map",
                   save_path.split('.png')[0] + '_LDA_each.png', z_xrange, z_yrange)
        
        # ICA
        self.ica = FastICA(n_components = 2)
        self.ica.fit(all_z)
        ica_z = self.ica.transform(all_z)
        ica_z = ica_z.transpose()
        
        z_xrange = [np.min(ica_z[0]), np.max(ica_z[0])]
        z_yrange = [np.min(ica_z[1]), np.max(ica_z[1])]        
        plot_z(ica_z[0], ica_z[1], all_ans, "z map", save_path.split('.png')[0] + '_ICA.png', z_xrange, z_yrange)
        plot_z_each(ica_z, all_ans, self.dataset.filenames, '../data/succeed_list.csv', "z map",
                   save_path.split('.png')[0] + '_ICA_each.png', z_xrange, z_yrange)
        
    def reconstruct(self, save_path = '../result/VAE/reconstructed_sounds'):
        loader = torch.utils.data.DataLoader(self.dataset,batch_size=1,shuffle=False)
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x = x.to(self.device)
                recon_x, _, _ = self.model.forward(x)
                recon_x = recon_x.to('cpu').clone().numpy()
                x = x.to('cpu').clone().numpy()
                x = x.reshape(3, -1)
                recon_x = recon_x.reshape(3, -1)
                # to png
                fig, ax = plt.subplots(2,3,figsize=(24, 12))
                ax[0][0].set_title('L')
                ax[0][1].set_title('C')
                ax[0][2].set_title('R')
                ax[1][0].set_title('reconstructed L')
                ax[1][1].set_title('reconstructed C')
                ax[1][2].set_title('reconstructed R')
                time = range(len(x[0]))
                for j in range(3):
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


def plot_z(x, y, ans, title, save_path, xrange=None, yrange=None):
    plt.figure(figsize=(8, 8))
    if xrange is not None:
        plt.xlim(xrange[0], xrange[1])
    if yrange is not None:
        plt.ylim(yrange[0], yrange[1])
    idxs, labels, colors, markers = ans2index_label_color_marker(ans)
    for i, (label, color, marker) in enumerate(zip(labels,colors,markers)):
        plt.scatter(x[idxs[i]:idxs[i+1]], y[idxs[i]:idxs[i+1]], label=label, s=10, color=color, marker=marker)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_z_each(data, ans, names, sf_filepath, title, save_path, xrange=None, yrange=None):
    data_list = results_list(data, ans, names)
    data_list.classify(sf_filepath)
    
    # the number of the classes == 11
    fig, ax = plt.subplots(4, 3, figsize=(24,32))
    for i in range(0,12):
        ax[i//3][i%3].set_title(str(i), fontsize=20)
    if xrange is not None:
        for i in range(0,12):
            ax[i//3][i%3].set_xlim(xrange[0], xrange[1])
    if yrange is not None:
        for i in range(1,12):
            ax[i//3][i%3].set_ylim(yrange[0], yrange[1])
            
    idxs, labels, colors, markers = ans2index_label_color_marker(ans)
    for i, (label, color, marker) in enumerate(zip(labels,colors,markers)):
        ax[0][0].scatter(data[0,idxs[i]:idxs[i+1]], data[1,idxs[i]:idxs[i+1]], label=label, s=20, color=color, marker=marker)
    ax[0][0].legend()
    ax[0][0].set_title(title, fontsize=20)
    
    for i, (result, label, color) in enumerate(zip(data_list,labels,colors)):
        if len(result.success) > 0:
            ax[(i+1)//3][(i+1)%3].scatter(result.success[:,0], result.success[:,1], label=label, s=20, color=color, marker='.')
        if len(result.false) > 0:
            ax[(i+1)//3][(i+1)%3].scatter(result.false[:,0], result.false[:,1], label=label, s=20, color=color, marker='x')
        ax[(i+1)//3][(i+1)%3].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


vae = VAE_without_label_trainer()
vae.load_weight(load_path =  '../result/VAE_without_label/drive/vae')
vae.load('drive')
#vae.auto_train(1000, save_path = '../result/VAE_without_label/drive')
vae.plot_z(save_path = '../result/VAE_without_label/drive/z_map.png')
#vae.reconstruct()
#vae.save_weight(save_path = '../result/VAE_without_label/drive/vae')
del vae

vae = VAE_without_label_trainer()
vae.load_weight(load_path =  '../result/VAE_without_label/block/vae')
vae.load('block')
#vae.auto_train(1000, save_path = '../result/VAE_without_label/block')
vae.plot_z(save_path = '../result/VAE_without_label/block/z_map.png')
#vae.reconstruct()
#vae.save_weight(save_path = '../result/VAE_without_label/block/vae')
del vae

vae = VAE_without_label_trainer()
vae.load_weight(load_path =  '../result/VAE_without_label/push/vae')
vae.load('push')
#vae.auto_train(1000, save_path = '../result/VAE_without_label/push')
vae.plot_z(save_path = '../result/VAE_without_label/push/z_map.png')
#vae.reconstruct()
#vae.save_weight(save_path = '../result/VAE_without_label/push/vae')
del vae

vae = VAE_without_label_trainer()
vae.load_weight(load_path =  '../result/VAE_without_label/stop/vae')
vae.load('stop')
#vae.auto_train(1000, save_path = '../result/VAE_without_label/stop')
vae.plot_z(save_path = '../result/VAE_without_label/stop/z_map.png')
#vae.reconstruct()
#vae.save_weight(save_path = '../result/VAE_without_label/stop/vae')
del vae

vae = VAE_without_label_trainer()
vae.load_weight(load_path =  '../result/VAE_without_label/flick/vae')
vae.load('flick')
#vae.auto_train(1000, save_path = '../result/VAE_without_label/flick')
vae.plot_z(save_path = '../result/VAE_without_label/flick/z_map.png')
#vae.reconstruct()
#vae.save_weight(save_path = '../result/VAE_without_label/flick/vae')
del vae


