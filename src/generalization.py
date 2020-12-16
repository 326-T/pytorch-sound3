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

from imu_vae import IMU_VAE, IMU_VAE_trainer
from vae import VAE, VAE_trainer
import torch.utils.data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.potential import Potential


class Generalization_Performance():
    
    def __init__(self, model_type, key):
        self.model_type = model_type
        self.key = key
        self.all_path = ['../data/sounds/raw/sub1/sub1_', '../data/sounds/raw/sub2/sub2_', '../data/sounds/raw/sub3/sub3_', 
                         '../data/sounds/raw/sub4/sub4_', '../data/sounds/raw/sub5/sub5_', '../data/sounds/raw/sub6/sub6_', 
                         '../data/sounds/raw/sub7/sub7_', '../data/sounds/raw/sub8/sub8_', '../data/sounds/raw/sub9/sub9_', 
                         '../data/sounds/raw/sub10/sub10_', '../data/sounds/raw/sub11/sub11_']
        
        potential = Potential()
        potential.load('../data/ranking', key)
        score = potential.score
        
        if self.model_type == 'IMU_VAE':
            self.all_path = ['../data/3DM_GX3s/raw/sub2/sub2_', '../data/3DM_GX3s/raw/sub3/sub3_', 
                             '../data/3DM_GX3s/raw/sub4/sub4_', '../data/3DM_GX3s/raw/sub5/sub5_', '../data/3DM_GX3s/raw/sub6/sub6_', 
                             '../data/3DM_GX3s/raw/sub8/sub8_', '../data/3DM_GX3s/raw/sub9/sub9_', 
                             '../data/3DM_GX3s/raw/sub11/sub11_']
            score = np.delete(score, [0, 6, 9])
        
        idx = np.argsort(score).tolist()
        self.score = np.ones(len(idx), dtype='int')
        self.score[idx[:int((len(idx)+1)/2)]] = 0
        self.hml_idx = [idx[int((len(idx)+2)/3*2):], idx[int((len(idx)+1)/3):int((len(idx)+2)/3*2)], idx[:int((len(idx)+1)/3)]]
        
            
    def check(self, exception):
        if self.model_type == 'VAE':
            vae = VAE_trainer(beta=20)
        elif self.model_type == 'IMU_VAE':
            vae = IMU_VAE_trainer(beta=20)
        for i, (path, score) in enumerate(zip(self.all_path, self.score)):
            if i != exception:
                vae.dataset.load_npz(path+self.key+'.npz', score)
        vae.dataset.normalize()
        vae.auto_train(1000)
        vae.dataset.clear()
        vae.dataset.load_npz(self.all_path[exception]+self.key+'.npz', self.score[exception])
        vae.dataset.normalize()
        vae.valid_loader = torch.utils.data.DataLoader(vae.dataset, batch_size=len(vae.dataset), shuffle=False)
        v_loss, v_loss_vae, v_loss_classifier, v_acc = vae.valid(1)
        del vae
        return v_acc
    
    def check_all(self):
        self.acc = np.zeros(len(self.all_path)+1)
        names = []
        for i in range(len(self.all_path)):
            self.acc[i] = self.check(i)
            names.append(self.all_path[i].split('/')[-1])
        
        self.acc[-1] = np.mean(self.acc[:-1])
        names.append("mean")
        
        df = pd.DataFrame(data=self.acc.reshape(1,-1), columns=names)
        df.to_csv('../result/' + self.model_type + '/' + self.key + '/generalization.csv')
        
        plt.figure()
        x = np.arange(len(names))
        plt.bar(x[self.hml_idx[0]], self.acc[self.hml_idx[0]], color="red", label='high', align="center")
        plt.bar(x[self.hml_idx[1]], self.acc[self.hml_idx[1]], color="green", label='middle', align="center")
        plt.bar(x[self.hml_idx[2]], self.acc[self.hml_idx[2]], color="blue", label='low', align="center")
        plt.bar(x[-1], self.acc[-1], color="black")
        plt.legend()
        plt.xticks(x, names)
        plt.savefig('../result/' + self.model_type + '/' + self.key + '/generalization.png')


def test_gp(model, key):
    gene = Generalization_Performance(model, key)
    gene.check_all()
    del gene


if __name__ == "__main__":
    models = ['IMU_VAE', 'VAE']
    keys = ['drive', 'block', 'push', 'stop', 'flick']
    for model in models:
        for key in keys:
            test_gp(model, key)




