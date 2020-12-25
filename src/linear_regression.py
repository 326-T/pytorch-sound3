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

from sklearn.decomposition import FastICA
from sklearn import svm
import torch.utils.data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.dataset import MyDataset, SoundDataset, IMUDataset
from generalization import Generalization_Performance
from modules.potential import Potential


class LR():
    
    def __init__(self, z_dim = 20):
        self.z_dim = z_dim
        self.ica = FastICA(n_components = self.z_dim, max_iter=1000)
        self.ica_plot = FastICA(n_components = 2)
        self.svm = svm.SVC()
        self.dataset=None
        
    def load(self, path):
        self.dataset.load_npz(path)
        self.dataset.normalize()
        
    def fit(self):
        train, test = random_split(self.dataset)
        train_X = train.data.reshape(len(train), -1)
        self.ica.fit(train_X)
        train_Z = self.ica.transform(train_X)
        self.ica_plot.fit(train_Z)
        self.svm.fit(train_Z, train.ans)
        
        _, train_acc = self.predict(train_X, train.ans)
        _, test_acc = self.predict(test.data, test.ans)
        return train_acc, test_acc
        
    def predict(self, _X, _Y=None):
        X = _X.reshape(len(_X), -1)
        Z = self.ica.transform(X)
        z = self.ica_plot.transform(Z)
        prediction = self.svm.predict(Z)
        if _Y is None:
            return z
        else:
            acc = self.acc(prediction, _Y)
            return z, acc
        
    def acc(self, pre_Y, Y):
        correct = 0
        for (p_y, y) in zip(pre_Y, Y):
            if p_y == y:
                correct += 1
        return correct / len(pre_Y)
        


class Sound_LR(LR):
    def __init__(self, z_dim = 20):
        super(Sound_LR, self).__init__(z_dim)
        self.dataset = SoundDataset()
        
    def load(self, key):
        super().load('../data/sounds/raw/'+key+'.npz')


class IMU_LR(LR):
    def __init__(self, z_dim = 20):
        super(Sound_LR, self).__init__(z_dim)
        self.dataset = IMUDataset()
        
    def load(self, key):
        super().load('../data/3DM_GX3s/raw/'+key+'.npz')


# +
class simple_dataset():
    
    def __init__(self, data, ans):
        self.data = data
        self.ans = ans
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.ans[idx]
    

def random_split(dataset):
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)
    train = simple_dataset(dataset.data[idx[:int(len(idx)/5+1)]], dataset.ans[idx[:int(len(idx)/5+1)]])
    test = simple_dataset(dataset.data[idx[int(len(idx)/5+1):]], dataset.ans[idx[int(len(idx)/5+1):]])
    
    return train, test


# -

def train_SVM(model, key, z_dim=20):
    if model == 'VAE':
        clf = Sound_LR(z_dim=z_dim)
    elif model == 'IMU_VAE':
        clf = IMU_LR(z_dim=z_dim)
    clf.load(key)
    train_acc, test_acc = clf.fit()
    print(train_acc, test_acc)


class Generalization_Performance_SVM(Generalization_Performance):
            
    def check(self, exception):
        if self.model_type == 'VAE':
            clf = Sound_LR()
        elif self.model_type == 'IMU_VAE':
            clf = IMU_LR()
        for i, (path, score) in enumerate(zip(self.all_path, self.score)):
            if i != exception:
                clf.dataset.load_npz(path+self.key+'.npz', ans=score)
        clf.dataset.normalize()
        clf.fit()
        clf.dataset.clear()
        clf.dataset.load_npz(self.all_path[exception]+self.key+'.npz', self.score[exception])
        clf.dataset.normalize()
        _, acc = clf.predict(clf.dataset.data, clf.dataset.ans)
        del clf
        return acc
    
    def check_all(self, filename='generalization_svm'):
        super(Generalization_Performance_SVM, self).check_all(filename=filename)


def test_gp(model, key):
    gene = Generalization_Performance_SVM(model, key)
    gene.check_all()
    del gene


if __name__ == "__main__":
    models = ['VAE', 'IMU_VAE']
    keys = ['drive', 'block', 'push', 'stop', 'flick']
    for model in models:
        for key in keys:
            #test_gp(model, key)
            train_SVM(model, key)


