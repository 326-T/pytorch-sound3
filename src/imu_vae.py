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


from modules.dataset import IMUDataset
from modules.myfunc import ans2index_label_color_marker
from modules.success_and_false import result, results_list


class VAE(nn.Module):
    def __init__(self,input_shape,z_shape=10,output_shape=11):
        super(VAE, self).__init__()
        
        self.input_shape = input_shape
        self.z_shape = z_shape
        self.output_shape = output_shape
        
        # encoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('enc_conv1', nn.Conv1d(in_channels=9, out_channels=9, kernel_size=10, stride=5, padding=3, padding_mode='zeros'))
        self.encoder.add_module('enc_relu1', nn.ReLU(True))
        self.encoder.add_module('enc_conv2', nn.Conv1d(in_channels=9, out_channels=9, kernel_size=4, stride=2, padding=1, padding_mode='zeros'))
        self.encoder.add_module('enc_relu2', nn.ReLU(True))
        # z to mean
        self.encmean_fc11 = nn.Linear(90, z_shape)
        # z to var
        self.encvar_fc12 = nn.Linear(90, z_shape)
        
        # decoder
        self.dec_fc1 = nn.Linear(z_shape, 90)
        self.decoder = nn.Sequential()
        self.decoder.add_module('dec_deconv1', nn.ConvTranspose1d(in_channels=9, out_channels=9, kernel_size=4, stride=2, padding=1, padding_mode='zeros'))
        self.decoder.add_module('dec_relu1', nn.ReLU(True))
        self.decoder.add_module('dec_deconv2', nn.ConvTranspose1d(in_channels=9, out_channels=9, kernel_size=10, stride=5, padding=3, output_padding=1, padding_mode='zeros'))
        self.decoder.add_module('dec_sig1', nn.Sigmoid())
        
        # estimator
        self.classifier = nn.Sequential()
        self.classifier.add_module('cla_fc1', nn.Linear(z_shape, self.output_shape))
        
    def encode(self, x):
        x = x.view(x.size()[0],9,-1)
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
        x = x.view(x.size()[0],9,-1)
        return x

    def forward(self, x):
        # encode
        mu, logvar = self.encode(x.view(-1, 9, self.input_shape).float())
        # reparameterize
        z = self.reparameterize(mu, logvar)
        pre_x = self.decode(z)
        # classification
        y = self.classifier(z)
        y = y.view(-1,self.output_shape)
        
        return pre_x, y, mu, logvar
    
    def valid(self, x):
        mu, logvar = self.encode(x.view(-1, 9, self.input_shape).float())
        y = self.classifier(mu)
        pre_x = self.decode(mu)
        y = y.view(-1,self.output_shape)
        
        return pre_x, y, mu, logvar
    
    def loss_function_vae(self, rec_x, x, mu, logvar, beta=1):
        BCE = F.binary_cross_entropy(rec_x, x.float(), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD * beta
    def loss_function_classifier(self, pre_y, y):
        return F.cross_entropy(pre_y, y)*self.input_shape*9
    
    def acc(self, pre_tar, tar):
        _, p_tar = torch.max(pre_tar, 1)
        correct = (p_tar==tar).sum().item()
        return correct


class VAE_trainer():
    def __init__(self, dim_z = 10, device="cuda"):
        # prepare cuda device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        # prepare dataset
        self.dataset = IMUDataset(transform=transforms.ToTensor())
        # define model
        self.model = VAE(self.dataset.data_size, dim_z).to(self.device)
        # define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.dim_z = dim_z
    
    #def __del__(self):
    #    self.save_weight()
    
    def load(self, key):
        self.dataset.load("../data/3DM_GX3s/raw/sub2", key, 1)
        self.dataset.load("../data/3DM_GX3s/raw/sub3", key, 2)
        self.dataset.load("../data/3DM_GX3s/raw/sub4", key, 3)
        self.dataset.load("../data/3DM_GX3s/raw/sub5", key, 4)
        self.dataset.load("../data/3DM_GX3s/raw/sub6", key, 5)
        self.dataset.load("../data/3DM_GX3s/raw/sub8", key, 7)
        self.dataset.load("../data/3DM_GX3s/raw/sub9", key, 8)
        self.dataset.load("../data/3DM_GX3s/raw/sub11", key, 10)
        self.dataset.normalize()
    
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
            if batch_idx % 1 == 0:
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
        
    def auto_train(self, max_epoch, save_path = '../result/IMU_VAE/model'):
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
        
    def save_weight(self, save_path = '../result/IMU_VAE/model/vae'):
        torch.save(self.model.state_dict(), save_path)
        
    def load_weight(self, load_path = '../result/IMU_VAE/model/vae'):
        self.model.load_state_dict(torch.load(load_path))
    
    
    def plot_z(self, save_path='../result/IMU_VAE/model/result.png'):
        # print z all data
        loader = torch.utils.data.DataLoader(self.dataset,batch_size=len(self.dataset),shuffle=False)
        all_z = []
        all_ans = []
        self.model.eval()
        with torch.no_grad():
            for i, (data, ans) in enumerate(loader):
                data = data.to(self.device)
                _, _, mu, logvar = self.model.forward(data)
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
        plot_z_each(lda_z, all_ans, self.dataset.filenames, '../data/succeed_list_imu.csv', "z map",
                   save_path.split('.png')[0] + '_LDA_each.png', z_xrange, z_yrange)
        
        # ICA
        self.ica = FastICA(n_components = 2)
        self.ica.fit(all_z)
        ica_z = self.ica.transform(all_z)
        ica_z = ica_z.transpose()
        
        z_xrange = [np.min(ica_z[0]), np.max(ica_z[0])]
        z_yrange = [np.min(ica_z[1]), np.max(ica_z[1])]        
        plot_z(ica_z[0], ica_z[1], all_ans, "z map", save_path.split('.png')[0] + '_ICA.png', z_xrange, z_yrange)
        plot_z_each(ica_z, all_ans, self.dataset.filenames, '../data/succeed_list_imu.csv', "z map",
                   save_path.split('.png')[0] + '_ICA_each.png', z_xrange, z_yrange)
        
    def reconstruct(self, save_path = '../result/IMU_VAE/reconstructed_3DM_GX3s'):
        loader = torch.utils.data.DataLoader(self.dataset,batch_size=1,shuffle=False)
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x = x.to(self.device)
                recon_x, _, _, _ = self.model.forward(x)
                recon_x = recon_x.to('cpu').clone().numpy()
                x = x.to('cpu').clone().numpy()
                x = x.reshape(9, -1)
                recon_x = recon_x.reshape(9, -1)
                # to png
                fig, ax = plt.subplots(3,3,figsize=(24, 24))
                ax[0][0].set_title('Pitch Y')
                ax[1][0].set_title('Roll X')
                ax[2][0].set_title('Heading Z')
                ax[0][1].set_title('Acc X')
                ax[1][1].set_title('Acc Y')
                ax[2][1].set_title('Acc Z')
                ax[0][2].set_title('AR X')
                ax[1][2].set_title('AR Y')
                ax[2][2].set_title('AR Z')
                time = range(len(x[0]))
                for j in range(9):
                    ax[j%3][j//3].set_ylim(0, 1)
                    ax[j%3][j//3].plot(time, x[j], linewidth = 1, label='raw')
                    ax[j%3][j//3].plot(time, recon_x[j], linewidth = 1, label='reconstructed')
                plt.tight_layout()
                plt.legend()
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
    fig, ax = plt.subplots(3, 3, figsize=(24,24))
    for i in range(0,9):
        ax[i//3][i%3].set_title(str(i), fontsize=20)
    if xrange is not None:
        for i in range(0,9):
            ax[i//3][i%3].set_xlim(xrange[0], xrange[1])
    if yrange is not None:
        for i in range(0,9):
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


def train_VAE(key):
    vae = VAE_trainer()
    #vae.load_weight(load_path =  '../result/VAE/' + key + '/vae')
    vae.load(key)
    vae.auto_train(1000, save_path = '../result/IMU_VAE/' + key)
    vae.plot_z(save_path = '../result/IMU_VAE/' + key + '/z_map.png')
    vae.reconstruct()
    vae.save_weight(save_path = '../result/IMU_VAE/' + key + '/vae')
    del vae


if __name__ == "__main__":
    train_VAE('drive')
    #train_VAE('block')
    train_VAE('push')
    train_VAE('stop')
    train_VAE('flick')



