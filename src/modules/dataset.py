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


class SoundDataset(Dataset):
    
    def __init__(self, dim=11, data_size = 20000, transform = None):
        self.transform = transform
        self.data_size = data_size
        self.data_num = 0
        self.init_flag = True
        self.dim = dim
            
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
        
        if(self.init_flag):
            self.filenames = filenames
            self.data = data
            self.ans = ans
            self.init_flag = False
        else:
            self.filenames = self.filenames + filenames
            self.data = np.concatenate([self.data, data])
            self.ans = np.concatenate([self.ans, ans])
        
    def normalize(self):
        self.data_max = np.max(self.data)
        self.data_min = np.min(self.data)
        self.data = (self.data - self.data_min) / (self.data_max - self.data_min)
        self.data_num = len(self.data)
    
    def clear(self):
        self.filenames.clear()
        self.init_flag = True
        self.data_num = 0
    
    def export_npz(self, save_path):
        np.savez(save_path, data = self.data, ans = self.ans, filenames = np.array(self.filenames))
        
    def load_npz(self, load_path, label=None):
        npz = np.load(load_path, allow_pickle=True)
        data = npz['data']
        filenames = npz['filenames'].tolist()
        if label is not None:
            ans = np.full(len(data), int(label))
        else:
            ans = npz['ans'].tolist()
        
        if(self.init_flag):
            self.filenames = filenames
            self.data = data
            self.ans = ans
            self.init_flag = False
        else:
            self.filenames += filenames
            self.data = np.concatenate([self.data, data])
            self.ans = np.concatenate([self.ans, ans])

    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_ans = self.ans[idx]
        return out_data, out_ans


class IMUDataset(Dataset):
    
    
    def __init__(self, dim=11, data_size = 100, transform = None):
        self.transform = transform
        self.data_size = data_size
        self.data_num = 0
        self.init_flag = True
        self.dim = dim
            
    def load_csv(self, data_path, key, label):
        all_filenames = [temp for temp in os.listdir(data_path) if ('.csv' in temp and key in temp)]
        all_filenames.sort()
        data = []
        filenames = []
        for filename in all_filenames:
            x = pd.read_csv(data_path+'/'+filename, index_col = 0)
            if(self.init_flag):
                self.columns = x.columns.values
            x.iloc[:,:].astype('float')
            x = x.values.transpose()
            if x.shape[1] >= self.data_size:
                data.append(x[1:,int(x.shape[1]/2-self.data_size/2):int(x.shape[1]/2+self.data_size/2)])
                filenames.append(filename)
            else:
                print(filename)
        data = np.array(data)
        ans = np.full(len(data),label)
        
        if(self.init_flag):
            if data.shape[0] > 0:
                self.filenames = filenames
                self.data = data
                self.ans = ans
                self.init_flag = False
        else:
            if data.shape[0] > 0:
                self.filenames += filenames
                self.data = np.concatenate([self.data, data])
                self.ans = np.concatenate([self.ans, ans])
        
    def normalize(self):
        self.data_max = np.max(self.data)
        self.data_min = np.min(self.data)
        self.data = (self.data - self.data_min) / (self.data_max - self.data_min)
        self.data_num = len(self.data)
        
    def clear(self):
        self.filenames.clear()
        self.init_flag = True
        self.data_num = 0    
    
    def export_npz(self, save_path):
        np.savez(save_path, data = self.data, ans = self.ans,
                 filenames = np.array(self.filenames), columns = self.columns)
        
    def load_npz(self, load_path, label=None):
        npz = np.load(load_path, allow_pickle=True)
        data = npz['data']
        filenames = npz['filenames'].tolist()
        if label is not None:
            ans = np.full(len(data), int(label))
        else:
            ans = npz['ans'].tolist()
        
        if(self.init_flag):
            self.filenames = filenames
            self.data = data
            self.ans = ans
            self.columns = npz['columns']
            self.init_flag = False
        else:
            self.filenames = self.filenames + filenames
            self.data = np.concatenate([self.data, data])
            self.ans = np.concatenate([self.ans, ans])
    
    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_ans = self.ans[idx]
        return out_data, out_ans


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


if __name__ == "__main__":
    csv2npz_all()




