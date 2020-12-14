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
            
    def load(self, data_path, key, label):
        filenames = [temp for temp in os.listdir(data_path) if ('.csv' in temp and key in temp)]
        filenames.sort()
        data = []
        ans = []
        for filename in filenames:
            x = pd.read_csv(data_path+'/'+filename)
            x.iloc[:,:].astype('float')
            x = x.values
            data.append(x[:,int(x.shape[1]/2-self.data_size/2):int(x.shape[1]/2+self.data_size/2)])
            #temp = np.zeros(self.dim)
            #temp[int(label)] = 1
            #ans.append(temp)
            ans.append(label)
        data = np.array(data)
        ans = np.array(ans)
        
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
            
    def load(self, data_path, key, label):
        filenames = [temp for temp in os.listdir(data_path) if ('.csv' in temp and key in temp)]
        filenames.sort()
        data = []
        ans = []
        for filename in filenames:
            x = pd.read_csv(data_path+'/'+filename, index_col = 0)
            if(self.init_flag):
                self.columns = x.columns.values
            x.iloc[:,:].astype('float')
            x = x.values.transpose()
            if x.shape[1] >= self.data_size:
                data.append(x[1:,int(x.shape[1]/2-self.data_size/2):int(x.shape[1]/2+self.data_size/2)])
                ans.append(label)
            else:
                print(filename)
        data = np.array(data)
        ans = np.array(ans)
        
        if(self.init_flag):
            if data.shape[0] > 0:
                self.filenames = filenames
                self.data = data
                self.ans = ans
                self.init_flag = False
        else:
            if data.shape[0] > 0:
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
    
    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_ans = self.ans[idx]
        return out_data, out_ans

if __name__ == "__main__":
    imu = IMUDataset()
    imu.load("../../data/3DM_GX3s/raw/sub2", 'drive', 2)
    imu.normalize()
    #sound = SoundDataset()
    #sound.load("../../data/sounds/raw/sub1", 'drive', 1)
    #sound.load("../../data/sounds/raw/sub2", 'drive', 2)
    #sound.load("../../data/sounds/raw/sub3", 'drive', 3)
    #sound.normalize()

