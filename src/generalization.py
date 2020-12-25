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

from vae import VAE, VAE_Trainer
from vdann import VDANN, VDANN_Trainer
from imu_vae import IMU_VAE, IMU_VAE_Trainer
from imu_vdann import IMU_VDANN, IMU_VDANN_Trainer
from vae2 import VAE2, VAE2_Trainer
import torch.utils.data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.potential import Potential


class Generalization_Performance():
    
    def __init__(self, model_type, key, coach=None, mode='rank'):
        self.model_type = model_type
        self.key = key
        self.coach = coach
        self.all_path = ['../data/sounds/raw/sub1/sub1_', '../data/sounds/raw/sub2/sub2_', '../data/sounds/raw/sub3/sub3_', 
                         '../data/sounds/raw/sub4/sub4_', '../data/sounds/raw/sub5/sub5_', '../data/sounds/raw/sub6/sub6_', 
                         '../data/sounds/raw/sub7/sub7_', '../data/sounds/raw/sub8/sub8_', '../data/sounds/raw/sub9/sub9_', 
                         '../data/sounds/raw/sub10/sub10_', '../data/sounds/raw/sub11/sub11_']
        
        potential = Potential()
        if coach is None:
            potential.load('../data/ranking', key)
        else:
            potential.load('../data/ranking', coach+'_'+key)
        
        score = potential.score
        rank = potential.rank

        if 'IMU' in self.model_type:
            self.all_path = ['../data/3DM_GX3s/raw/sub2/sub2_', '../data/3DM_GX3s/raw/sub3/sub3_', 
                             '../data/3DM_GX3s/raw/sub4/sub4_', '../data/3DM_GX3s/raw/sub5/sub5_', '../data/3DM_GX3s/raw/sub6/sub6_', 
                             '../data/3DM_GX3s/raw/sub8/sub8_', '../data/3DM_GX3s/raw/sub9/sub9_', 
                             '../data/3DM_GX3s/raw/sub11/sub11_']
            score = np.delete(score, [0, 6, 9])
            rank = np.delete(rank, [0, 6, 9])
        if mode == 'score':
            idx = np.argsort(score)
            self.binary = np.ones(len(idx), dtype='int')
            self.binary[idx[:int((len(idx)+1)/2)]] = 0
            self.hml_idx = [idx[int((len(idx))/4*3):], idx[int((len(idx)+2)/4*2):int((len(idx))/4*3)], 
                            idx[int((len(idx))/4):int((len(idx)+2)/4*2)], idx[int((len(idx)+3)/4):]]
        elif mode == 'rank':
            idx = np.argsort(rank)
            self.binary = np.zeros(len(idx), dtype='int')
            self.binary[idx[:int(len(idx)/2)]] = 1
            self.hml_idx = [idx[:int((len(idx)+3)/4)], idx[int((len(idx)+3)/4):int((len(idx)+2)/4*2)], 
                            idx[int((len(idx)+2)/4*2):int((len(idx))/4*3)], idx[:int((len(idx))/4*3)]]

    def load_vae(self):
        if self.model_type == 'VAE':
            self.vae = VAE_Trainer(output_shape=2, beta=20)
        elif self.model_type == 'IMU_VAE':
            self.vae = IMU_VAE_Trainer(output_shape=2, beta=20)
        elif self.model_type == 'VAE2':
            self.vae = VAE2_Trainer(output_shape=2, beta=20)
        if self.model_type =='VDANN':
            self.vae = VDANN_Trainer(z_dim=20, beta=20)
        if self.model_type =='IMU_VDANN':
            self.vae = IMU_VDANN_Trainer(z_dim=20, beta=20)
    
    def train(self, exception):
        for i, (path, binary) in enumerate(zip(self.all_path, self.binary)):
            if i != exception:
                self.vae.dataset.load_npz(path+self.key+'.npz', ans=binary)
        self.vae.dataset.normalize()
        self.vae.auto_train(1000)

    def validate(self, exception):
        self.vae.dataset.clear()
        self.vae.dataset.load_npz(self.all_path[exception]+self.key+'.npz', self.binary[exception])
        self.vae.dataset.normalize()
        self.vae.valid_loader = torch.utils.data.DataLoader(self.vae.dataset, batch_size=len(self.vae.dataset), shuffle=False)
        if 'DANN' in self.model_type:
            _, _, _, _, v_acc, _ = self.vae.valid(1)
        else:
            _, _, _, v_acc = self.vae.valid(1)
        del self.vae
        return v_acc

    def check(self, exception):
        load_vae()
        train(exception)
        acc = validate(exception)
        return acc 
    
    def check_all(self, filename='generalization/generalization'):
        self.acc = np.zeros(len(self.all_path)+1)
        names = []
        for i in range(len(self.all_path)):
            self.acc[i] = self.check(i)
            names.append(self.all_path[i].split('/sub')[-1].split('_')[0])
        
        self.acc[-1] = np.mean(self.acc[:-1])
        names.append("mean")
        df = pd.DataFrame(data=self.acc.reshape(1,-1), columns=names)
        if self.coach is None:
            df.to_csv('../result/' + self.model_type + '/' + self.key + '/' + filename + '.csv')
        else:
            df.to_csv('../result/' + self.model_type + '/' + self.key + '/' + filename + '_' + self.coach + '.csv')
        
        plt.figure()
        plt.ylabel('accuracy')
        x = np.arange(len(names))
        plt.bar(x[self.hml_idx[0]], self.acc[self.hml_idx[0]], color="red", label='high', align="center")
        plt.bar(x[self.hml_idx[1]], self.acc[self.hml_idx[1]], color="pink", label='middle', align="center")
        plt.bar(x[self.hml_idx[2]], self.acc[self.hml_idx[2]], color="skyblue", label='middle', align="center")
        plt.bar(x[self.hml_idx[3]], self.acc[self.hml_idx[3]], color="blue", label='low', align="center")
        plt.bar(x[-1], self.acc[-1], color="black")
        plt.legend()
        plt.xticks(x, names)
        if self.coach is None:
            plt.savefig('../result/' + self.model_type + '/' + self.key + '/' + filename + '.png')
        else:
            plt.savefig('../result/' + self.model_type + '/' + self.key + '/' + filename + '_' + self.coach + '.png')
        plt.close()
    
class Generalization_Performance_VAE(Generalization_Performance):
    
    def load_vae(self):
        if self.model_type == 'VAE':
            self.vae = VAE_Trainer(output_shape=11, beta=20)
        elif self.model_type == 'IMU_VAE':
            self.vae = IMU_VAE_Trainer(output_shape=11, beta=20)

    def train(self, exception):
        for i, path in enumerate(self.all_path):
            if i != exception:
                self.vae.dataset.load_npz(path+self.key+'.npz')
        self.vae.dataset.normalize()
        self.vae.auto_train(1000)

    def validate(self, exception):
        self.vae.dataset.clear()
        self.vae.dataset.load_npz(self.all_path[exception]+self.key+'.npz')
        self.vae.dataset.normalize()
        self.vae.valid_loader = torch.utils.data.DataLoader(self.vae.dataset, batch_size=len(self.vae.dataset), shuffle=False)
        _, _, _, v_acc = self.vae.valid(1)
        del self.vae
        return v_acc

def test_gp(model, key, coach=None):
    gene = Generalization_Performance(model, key, coach=coach, mode='rank')
    gene.check_all()
    del gene


if __name__ == "__main__":
    models = ['VDANN', 'IMU_VDANN']
    coachs = [None, 'Y-Takahashi', 's-nagashima', 'S-TAKEDA']
    keys = ['drive', 'block', 'push', 'stop', 'flick']
    for model in models:
        for coach in coachs:
           for key in keys:
                test_gp(model, key, coach)
