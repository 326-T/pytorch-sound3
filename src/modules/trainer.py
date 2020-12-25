import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.myfunc import ans_score2index_label_color_marker
from modules.success_and_false import result, results_list

class Trainer():
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def save_weight(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        
    def load_weight(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

class Sound_Trainer(Trainer):

    def __init__(self, device='cuda'):
        super().__init__(device)

    def load(self, key):
        self.dataset.load_npz('../data/sounds/raw/'+key+'.npz')
        self.dataset.normalize()

    def plot_z(self, all_z, all_ans, save_path):
        # LDA
        self.lda = LDA(n_components = 2)
        self.lda.fit(all_z, all_ans)
        lda_z = self.lda.transform(all_z)
        lda_z = lda_z.transpose()
        
        z_xrange = [np.min(lda_z[0]), np.max(lda_z[0])]
        z_yrange = [np.min(lda_z[1]), np.max(lda_z[1])]        
        self.plot_z_simple(lda_z[0], lda_z[1], all_ans, self.dataset.score, "z map", save_path.split('.png')[0] + '_LDA.png', z_xrange, z_yrange)
        self.plot_z_each(lda_z, all_ans, self.dataset.score, self.dataset.filenames, '../data/succeed_list_sound.csv', "z map",
                        save_path.split('.png')[0] + '_LDA_each.png', z_xrange, z_yrange)
        
        # ICA
        self.ica = FastICA(n_components = 2, max_iter=1000)
        self.ica.fit(all_z)
        ica_z = self.ica.transform(all_z)
        ica_z = ica_z.transpose()
        
        z_xrange = [np.min(ica_z[0]), np.max(ica_z[0])]
        z_yrange = [np.min(ica_z[1]), np.max(ica_z[1])]        
        self.plot_z_simple(ica_z[0], ica_z[1], all_ans, self.dataset.score, "z map", save_path.split('.png')[0] + '_ICA.png', z_xrange, z_yrange)
        self.plot_z_each(ica_z, all_ans, self.dataset.score, self.dataset.filenames, '../data/succeed_list_sound.csv', "z map",
                        save_path.split('.png')[0] + '_ICA_each.png', z_xrange, z_yrange)
        return ica_z.transpose()

    def plot_z_simple(self, x, y, ans, score, title, save_path, xrange=None, yrange=None):
        plt.figure(figsize=(8, 8))
        if xrange is not None:
            plt.xlim(xrange[0], xrange[1])
        if yrange is not None:
            plt.ylim(yrange[0], yrange[1])
        idxs, labels, colors, markers = ans_score2index_label_color_marker(ans, score)
        for i, (label, color, marker) in enumerate(zip(labels,colors,markers)):
            plt.scatter(x[idxs[i]:idxs[i+1]], y[idxs[i]:idxs[i+1]], label=label, s=10, color=color, marker=marker)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_z_each(self, data, ans, score, names, sf_filepath, title, save_path, xrange=None, yrange=None):
        data_list = results_list(data, ans, names)
        data_list.classify(sf_filepath)
        
        # the number of the classes == 11
        fig, ax = plt.subplots(4, 3, figsize=(24,32))
        ax[0][0].set_title('All', fontsize=20)
        if xrange is not None:
            for i in range(0,12):
                ax[i//3][i%3].set_xlim(xrange[0], xrange[1])
        if yrange is not None:
            for i in range(1,12):
                ax[i//3][i%3].set_ylim(yrange[0], yrange[1])
                
        idxs, labels, colors, markers = ans_score2index_label_color_marker(ans, score)
        for i, (label, color, marker) in enumerate(zip(labels,colors,markers)):
            ax[0][0].scatter(data[0,idxs[i]:idxs[i+1]], data[1,idxs[i]:idxs[i+1]], label=label, s=20, color=color, marker=marker)
        ax[0][0].legend()
        ax[0][0].set_title(title, fontsize=20)

        for i, (result, label, color) in enumerate(zip(data_list,labels,colors)):
            if len(result.success) > 0:
                ax[(i+1)//3][(i+1)%3].scatter(result.success[:,0], result.success[:,1], label=label, s=20, color=color, marker='.')
            if len(result.false) > 0:
                ax[(i+1)//3][(i+1)%3].scatter(result.false[:,0], result.false[:,1], label=label, s=20, color=color, marker='x')
            ax[(i+1)//3][(i+1)%3].set_title('Player '+label)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_reconstruct(self, x, recon_x, save_path):
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
    
class IMU_Trainer(Sound_Trainer):

    def load(self, key):
        self.dataset.load_npz('../data/3DM_GX3s/raw/'+key+'.npz')
        self.dataset.normalize()

    def plot_z_each(self, data, ans, score, names, sf_filepath, title, save_path, xrange=None, yrange=None):
        data_list = results_list(data, ans, names)
        data_list.classify(sf_filepath)
        
        # the number of the classes == 11
        fig, ax = plt.subplots(3, 3, figsize=(24,24))
        ax[0][0].set_title('All', fontsize=20)
        if xrange is not None:
            for i in range(0,9):
                ax[i//3][i%3].set_xlim(xrange[0], xrange[1])
        if yrange is not None:
            for i in range(0,9):
                ax[i//3][i%3].set_ylim(yrange[0], yrange[1])
                
        idxs, labels, colors, markers = ans_score2index_label_color_marker(ans, score)
        for i, (label, color, marker) in enumerate(zip(labels,colors,markers)):
            ax[0][0].scatter(data[0,idxs[i]:idxs[i+1]], data[1,idxs[i]:idxs[i+1]], label=label, s=20, color=color, marker=marker)
        ax[0][0].legend()
        ax[0][0].set_title(title, fontsize=20)
        
        for i, (result, label, color) in enumerate(zip(data_list,labels,colors)):
            if len(result.success) > 0:
                ax[(i+1)//3][(i+1)%3].scatter(result.success[:,0], result.success[:,1], label=label, s=20, color=color, marker='.')
            if len(result.false) > 0:
                ax[(i+1)//3][(i+1)%3].scatter(result.false[:,0], result.false[:,1], label=label, s=20, color=color, marker='x')
                ax[(i+1)//3][(i+1)%3].set_title('Player '+label, fontsize=20)
            #ax[(i+1)//3][(i+1)%3].legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_reconstruct(self, x, recon_x, save_path):
        fig, ax = plt.subplots(3,3,figsize=(24, 24))
        time = range(len(x[0]))
        for j in range(9):
            ax[j%3][j//3].set_title(self.dataset.columns[j])
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
