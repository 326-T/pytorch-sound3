# -*- coding: utf-8 -*-
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

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from modules.tournament import Swiss_System_Tournament, Player


class MyDataset(Dataset):
    
    def __init__(self, dim=11, data_size = 20000, transform = None, mode = 'label'):
        self.dim = dim
        self.data_size = data_size
        self.transform = transform
        self.mode = mode
        self.data_num = 0
        self.init_flag = True
        self.results = []
            
    def load_score(self, path, key):
        filenames = [temp for temp in os.listdir(path) if ('.csv' in temp and key in temp)]
        self.key = key
        for filename in filenames:
            tournament = Swiss_System_Tournament()
            tournament.Load(path + '/' + filename)
            self.results.append(tournament)
        self.score = np.zeros(self.results[0].participants)
        self.calc_score()
        idx = np.argsort(self.score).tolist()
        self.high_low = np.ones(len(idx), dtype='int')
        self.high_low[idx[:int((len(idx)+1)/2)]] = 0
        
    def calc_score(self):
        for result in self.results:
            for player in result.players:
                if player.name != "Bye":
                    id = int(player.name.split('選手')[-1]) - 1
                    self.score[id] += player.score
        self.score = (self.score - np.min(self.score)) / (np.max(self.score) - np.min(self.score))
            
    def load_csv(self, data_path, key, label):
        all_filenames = [temp for temp in os.listdir(data_path) if ('.csv' in temp and key in temp)]
        all_filenames.sort()
        data = []
        filenames = []
        for filename in all_filenames:
            x = pd.read_csv(data_path+'/'+filename)
            x.iloc[:,:].astype('float')
            x = x.values
            if x.shape[1] >= self.data_size:
                data.append(x[:,int(x.shape[1]/2-self.data_size/2):int(x.shape[1]/2+self.data_size/2)])
                filenames.append(filename)
            else:
                print(filename)
        data = np.array(data)
        ans = np.full(len(data),label)
        score = np.full(len(data), self.score[int(data_path.split('/sub')[-1])-1])
        
        if(self.init_flag):
            self.filenames = filenames
            self.data = data
            self.ans = ans
            self.score = score
            self.init_flag = False
        else:
            self.filenames = self.filenames + filenames
            self.data = np.concatenate([self.data, data])
            self.ans = np.concatenate([self.ans, ans])
            self.score = np.concatenate([self.score, score])
        
    def normalize(self):
        self.data_max = np.max(self.data)
        self.data_min = np.min(self.data)
        self.data = (self.data - self.data_min) / (self.data_max - self.data_min)
        self.data_num = len(self.data)
        self.ans = self.ans.astype(int)

    def clear(self):
        self.filenames.clear()
        self.results.clear()
        self.init_flag = True
        self.data_num = 0
    
    def export_npz(self, save_path):
        np.savez(save_path, data = self.data, ans = self.ans, score = self.score, binary = self.binary, filenames = np.array(self.filenames))
        
    def load_npz(self, load_path, ans=None, score=None):
        npz = np.load(load_path, allow_pickle=True)
        data = npz['data']
        #binary = npz['binary']
        filenames = npz['filenames'].tolist()
        if ans is not None:
            ans = np.full(len(data), ans)
        else:
            ans = npz['ans']
        if score is not None:
            score = np.full(len(data), score)
        else:
            score = npz['score']
        
        if(self.init_flag):
            self.filenames = filenames
            self.data = data
            self.ans = ans
            self.score = score
            #self.binary = binary
            self.init_flag = False
        else:
            self.filenames += filenames
            self.data = np.concatenate([self.data, data])
            self.ans = np.concatenate([self.ans, ans])
            self.score = np.concatenate([self.score, score])
            #self.binary = np.concatenate([self.binary, binary])

    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        out_data = self.data[idx]
        if self.mode == 'label':
            return self.data[idx], self.ans[idx]
        elif self.mode == 'score':
            return self.data[idx], self.score[idx]
        elif self.mode == 'DANN':
            return self.data[idx], self.binary[idx], self.ans[idx]    


class SoundDataset(MyDataset):
    
    def __init__(self, dim=11, data_size = 20000, transform = None, mode = 'label'):
        super().__init__(dim, data_size, transform, mode)


class SoundDataset2(SoundDataset):
    
    def __getitem__(self, idx):
        out_data = self.data[idx][0:4:2]
        if self.mode == 'label':
            out = self.ans[idx]
        elif self.mode == 'score':
            out = self.score[idx]
        return out_data, out


class IMUDataset(MyDataset):
    
    def __init__(self, dim=11, data_size = 100, transform = None, mode = 'label'):
        super().__init__(dim, data_size, transform, mode)
        self.columns = ['Pitch Y', 'Roll X', 'Heading Z', 'Acc X', 'Acc Y', 'Acc Z', 'AR X', 'AR Y', 'AR Z']


def csv2npz_all():
    def csv2npz(model, name, key, label):
        if model == '3DM_GX3s':
            if name == 'sub1' or name == 'sub7' or name == 'sub10':
                return 1
            dataset = IMUDataset()
        elif model == 'sounds':
            dataset = SoundDataset()
        dataset.load_csv('../../data/'+model+'/raw/'+name, key, label)
        dataset.export_npz('../../data/'+model+'/raw/'+name+'/'+name+'_'+key+'.npz')
        del dataset
        
    models = ['sounds']
    names = ['sub1', 'sub2', 'sub3', 'sub4', 'sub5', 'sub6', 'sub7', 'sub8', 'sub9', 'sub10', 'sub11']
    keys = ['drive', 'block', 'push', 'stop', 'flick']
    for model in models:
        for name in names:
            print(name)
            for key in keys:
                csv2npz(model, name, key, 0)


def add_score_all():
    def add_score(model, name, key):
        if model == '3DM_GX3s':
            if name == 'sub1' or name == 'sub7' or name == 'sub10':
                return 1
            dataset = IMUDataset()
        elif model == 'sounds':
            dataset = SoundDataset()
        dataset.load_score('../../data/ranking', key)
        dataset.load_npz('../../data/'+model+'/raw/'+name+'/'+name+'_'+key+'.npz')
        dataset.export_npz('../../data/'+model+'/raw/'+name+'/'+name+'_'+key+'.npz')
        del dataset
    
    def merge_npz(model, key, names):
        if model == '3DM_GX3s':
            dataset = IMUDataset()
        elif model == 'sounds':
            dataset = SoundDataset()
        for name in names:
            if not (model == '3DM_GX3s' and (name == 'sub1' or name == 'sub7' or name == 'sub10')):
                dataset.load_npz('../../data/'+model+'/raw/'+name+'/'+name+'_'+key+'.npz')
        dataset.normalize()
        dataset.export_npz('../../data/'+model+'/raw/'+key+'.npz')
        del dataset

    models = ['sounds', '3DM_GX3s']
    names = ['sub1', 'sub2', 'sub3', 'sub4', 'sub5', 'sub6', 'sub7', 'sub8', 'sub9', 'sub10', 'sub11']
    keys = ['drive', 'block', 'push', 'stop', 'flick']
    for model in models:
        for key in keys:
            #for name in names:
                #add_score(model, name, key)
            merge_npz(model, key, names)


if __name__ == "__main__":
    add_score_all()

