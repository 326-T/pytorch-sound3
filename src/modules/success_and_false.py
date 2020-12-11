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

from modules.myfunc import optional_split
from modules.myfunc import ans2index_label


class filename_list():
    def __init__(self):
        self.filenames = []
        
    def add(self, folder_path):
        filenames = [temp.split('-C.wav')[0] for temp in os.listdir(folder_path) if ('-C.wav' in temp)]
        filenames.sort()
        self.filenames += [temp for temp in filenames if 'drive' in temp]
        self.filenames += [temp for temp in filenames if 'block' in temp]
        self.filenames += [temp for temp in filenames if 'push' in temp]
        self.filenames += [temp for temp in filenames if 'stop' in temp]
        self.filenames += [temp for temp in filenames if 'flick' in temp]
        
    def export(self, save_path):
        df = pd.DataFrame(index=self.filenames, columns=[str(i) for i in range(1,41)])
        print(df)
        df.to_csv(save_path)


class result():
    def __init__(self, data, names):
        self.success = []
        self.false = []
        self.data = data
        self.names = names
        
    def classify(self, sheet):
        for i, name in enumerate(self.names):
            row, col = optional_split(name.split('.csv')[0], '_', -1)
            if sheet.at[row,col] is '2':
                self.false.append(self.data[i])
            elif sheet.isnull().at[row,col]:
                self.success.append(self.data[i])
                
        self.success = np.array(self.success)
        self.false = np.array(self.false)


class results_list():
    def __init__(self, data, ans, names):
        if len(data) == 2:
            self.data = data.transpose()
        else:
            self.data = data
        self.names = names
        
        indexs, labels = ans2index_label(ans)
        self.results = []
        for i in range(len(labels)):
            new = result(self.data[indexs[i]:indexs[i+1]], self.names[indexs[i]:indexs[i+1]])
            self.results.append(new)
        
    def classify(self, load_path):
        self.sf_list = pd.read_csv(load_path, index_col=0)
        for i in range(len(self.results)):
            self.results[i].classify(self.sf_list)
        
    def __getitem__(self, idx):
        return self.results[idx]


def make_succeed_list():
    a = filename_list()
    a.add('../../data/original/sound/sub1')
    a.add('../../data/original/sound/sub2')
    a.add('../../data/original/sound/sub3')
    a.add('../../data/original/sound/sub4')
    a.add('../../data/original/sound/sub5')
    a.add('../../data/original/sound/sub6')
    a.add('../../data/original/sound/sub7')
    a.add('../../data/original/sound/sub8')
    a.add('../../data/original/sound/sub9')
    a.add('../../data/original/sound/sub10')
    a.add('../../data/original/sound/sub11')
    a.export('../../data/succeed_list.csv')


def make_succeed_list_imu():
    a = filename_list()
    a.add('../../data/original/3DM_GX3/sub1')
    a.add('../../data/original/3DM_GX3/sub2')
    a.add('../../data/original/3DM_GX3/sub3')
    a.add('../../data/original/3DM_GX3/sub4')
    a.add('../../data/original/3DM_GX3/sub5')
    a.add('../../data/original/3DM_GX3/sub6')
    a.add('../../data/original/3DM_GX3/sub7')
    a.add('../../data/original/3DM_GX3/sub8')
    a.add('../../data/original/3DM_GX3/sub9')
    a.add('../../data/original/3DM_GX3/sub10')
    a.add('../../data/original/3DM_GX3/sub11')
    a.export('../../data/succeed_list_imu.csv')


if __name__ == "__main__":
    make_succeed_list()


