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
        self.all_path = ["../data/sounds/raw/sub1", "../data/sounds/raw/sub2", "../data/sounds/raw/sub3", "../data/sounds/raw/sub4",
                         "../data/sounds/raw/sub5", "../data/sounds/raw/sub6", "../data/sounds/raw/sub7", "../data/sounds/raw/sub8",
                         "../data/sounds/raw/sub9", "../data/sounds/raw/sub10", "../data/sounds/raw/sub11"]
        
        potential = Potential()
        potential.load('../data/ranking', key)
        score = potential.score
        
        if self.model_type == 'IMU_VAE':
            self.all_path = ["../data/3DM_GX3s/raw/sub2", "../data/3DM_GX3s/raw/sub3", "../data/3DM_GX3s/raw/sub4",
                            "../data/3DM_GX3s/raw/sub5", "../data/3DM_GX3s/raw/sub6", "../data/3DM_GX3s/raw/sub7", "../data/3DM_GX3s/raw/sub8",
                            "../data/3DM_GX3s/raw/sub9", "../data/3DM_GX3s/raw/sub10", "../data/3DM_GX3s/raw/sub11",]
            score = np.delete(score, 0)
        
        idx = np.argsort(score).tolist()
        self.score = np.ones(len(idx), dtype='int')
        self.score[idx[:int((len(idx)+1)/2)]] = 0
        self.hml_idx = [idx[int((len(idx)+2)/3*2):], idx[int((len(idx)+1)/3):int((len(idx)+2)/3*2)], idx[:int((len(idx)+1)/3)]]
        
            
    def check(self, exception):
        if self.model_type == 'VAE':
            vae = VAE_trainer()
        elif self.model_type == 'IMU_VAE':
            vae = IMU_VAE_trainer()
        for i, (path, score) in enumerate(zip(self.all_path, self.score)):
            if i != exception:
                vae.dataset.load_npz(path, self.key, score)
        vae.dataset.normalize()
        vae.auto_train(100)
        vae.dataset.clear()
        vae.dataset.load_npz(self.all_path[exception], self.key, self.score[exception])
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


def IMU_all():
    gene = Generalization_Performance('IMU_VAE', 'drive')
    gene.check_all()
    del gene
    gene = Generalization_Performance('IMU_VAE', 'block')
    gene.check_all()
    del gene
    gene = Generalization_Performance('IMU_VAE', 'push')
    gene.check_all()
    del gene
    gene = Generalization_Performance('IMU_VAE', 'stop')
    gene.check_all()
    del gene
    gene = Generalization_Performance('IMU_VAE', 'flick')
    gene.check_all()
    del gene


if __name__ == "__main__":
    model = ['IMU_VAE', 'VAE']
    key = ['drive', 'block', 'push', 'stop', 'flick']
    IMU_all()




