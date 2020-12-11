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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import os


# +
class IMU:
    
    
    def __init__(self, name, data, columns, start, end, frame_rate=100):
        self.name = name
        self.data = data
        self.columns = columns
        self.start = start
        self.end = end
        self.frame_rate = frame_rate
        
    def save(self):
        save_data = pd.DataFrame(data = self.data.transpose(), columns = self.columns)
        save_data.to_csv(self.name + '.csv')
        
    def print_simple(self):
        fig = plt.figure(figsize=(24,24))
        ax = []
        for i in range(9):
            ax.append(fig.add_subplot(3,3,i+1))
            ax[i].set_title(self.columns[i+1])
            ax[i].plot(self.data[0], self.data[i+1])
        plt.tight_layout()
        plt.savefig(self.name +'.png')
        plt.close()
        
    def fft(self):
        frequency = np.fft.rfftfreq(self.data.shape[1], 1.0/self.frame_rate)
        spectrum = np.block([[np.fft.rfft(self.data[1])], [np.fft.rfft(self.data[2])], [np.fft.rfft(self.data[3])],
                            [np.fft.rfft(self.data[4])], [np.fft.rfft(self.data[5])], [np.fft.rfft(self.data[6])],
                            [np.fft.rfft(self.data[7])], [np.fft.rfft(self.data[8])], [np.fft.rfft(self.data[9])]])
        
        fig, ax = plt.subplots(3,3,figsize=(24, 24))
        for i in range(9):
            ax[i%3][i//3].set_yscale("log")
            ax[i%3][i//3].set_title(self.columns[i+1])
            ax[i%3][i//3].scatter(frequency, np.abs(spectrum[i]), s=1)
        plt.tight_layout()
        plt.savefig(self.name.split('raw/')[0] + 'fft/' + self.name.split('raw/')[-1] +'.png')
        plt.close()
        save_data = pd.DataFrame(data = spectrum.transpose(), index = frequency, columns = self.columns[1:])
        save_data.to_csv(self.name.split('raw/')[0] + 'fft/' + self.name.split('raw/')[-1] + '.csv')
        
    def spectrogram(self, width = 32, step = 16):
        window = np.hamming(width)
        ampLists = []
        argLists = []
        for i in range(1, 10):
            ampList = []
            argList = []
            for j in range(int((self.data.shape[1] - width) / step)):
                data = self.data[i,j*step:j*step+width] * window
                spec = np.fft.rfft(data)
                ampList.append(np.abs(spec))
                argList.append(np.angle(spec))
            ampList = np.array(ampList) + 1e-6
            ampList = np.log(ampList)
            argList = np.array(argList)
            ampList = np.transpose(ampList)
            argList = np.transpose(argList)
            
            ampLists.append(ampList)
            argLists.append(argList)
            
        ampLists = np.array(ampLists)
        argLists = np.array(argLists)
        
        #calculate frequency
        freq = np.fft.rfftfreq(width, 1.0/self.frame_rate)
        #calculate time
        time = np.arange(0, len(ampList), 1) * step / self.frame_rate
        #plot
        
        fig, ax = plt.subplots(3,3,figsize=(24, 24))
        for i in range(9):
            ax[i%3][i//3].set_title(self.columns[i+1])
            ax[i%3][i//3].imshow(ampLists[i], extent=[time[0], time[-1], freq[0], freq[-1]], aspect="auto", origin='lower', cmap='afmhot')
        plt.tight_layout()
        plt.savefig(self.name.split('raw/')[0] + 'spectrogram/' + self.name.split('raw/')[-1] +'.png')
        plt.close()
        
class IMUs:
    def __init__(self, name, ex_path, frame_rate = 100, size = 50):
        self.devided = []
        self.index = []
        self.name = name
        self.ex_path = ex_path
        self.frame_rate = frame_rate
        self.size = size
        x = pd.read_csv(name + '.csv', skiprows = 3)
        x.iloc[:,:].astype('float')
        self.data = x.iloc[0:,:].values.transpose()
        self.columns = x.columns.tolist()
        
    def print_simple(self):
        fig, ax = plt.subplots(3,3,figsize=(24,24))
        for i in range(9):
            ax[i%3][i//3].set_title(self.columns[i+1])
            ax[i%3][i//3].plot(self.data[0], self.data[i+1])
            ax[i%3][i//3].axhline(np.mean(self.data[i+1]), color = "green")
            for idx in self.index:
                #ax[i%3,i//3].axvline(self.data[0,int(np.mean(idx))]/self.frame_rate, ls = "--", color = "red")
                ax[i%3, i//3].scatter(self.data[0,int(np.mean(idx))], self.data[i+1,int(np.mean(idx))], s = 30, color='red')
        plt.tight_layout()
        fig.suptitle(str(len(self.index)), fontsize=60)
        plt.subplots_adjust(top=0.9)
        plt.savefig(self.name +'.png')
        plt.close()
        
    def devide(self):
        pre_index = 0
        count = 1
        acc_mean = []
        acc_max = []
        for i in range(3):
            acc_mean.append(np.mean(self.data[i+4]))
            acc_max.append(np.max(np.abs(self.data[i+4] - acc_mean[-1])))
        for i in range(1, len(self.data[0])):
            if(i > pre_index + 100
               and (np.abs(self.data[4][i] - acc_mean[0]) > acc_max[0] / 2
                   or np.abs(self.data[5][i] - acc_mean[1]) > acc_max[1] / 2
                   or np.abs(self.data[6][i] - acc_mean[2]) > acc_max[2] / 2)):
                sound_name = self.ex_path + '_' + str(count)
                self.devided.append(IMU(sound_name, self.data[:, i-self.size : i+self.size], self.columns, i-self.size, i+self.size, self.frame_rate))
                pre_index = i
                count += 1
                self.index.append([i - self.size, i + self.size])
                
    def devide_stop(self):
        pre_index = 0
        count = 1
        acc_mean = []
        acc_max = []
        for i in range(3):
            acc_mean.append(np.mean(self.data[i+1]))
            acc_max.append(np.max(self.data[i+1] - acc_mean[-1]))
        for i in range(1, len(self.data[0])):
            if(i > pre_index + 100
               and self.data[3][i] - acc_mean[2] > acc_max[2] / 1.5):
                sound_name = self.ex_path + '_' + str(count)
                self.devided.append(IMU(sound_name, self.data[:, i-self.size : i+self.size], self.columns, i-self.size, i+self.size, self.frame_rate))
                pre_index = i
                count += 1
                self.index.append([i - self.size, i + self.size])
    
    def devide_acc(self):
        pre_index = 0
        count = 1
        acc = (self.data[:,2:-1] - self.data[:, 1:-2] - (self.data[:, 1:-2] - self.data[:, 0:-3])) * self.frame_rate**2
        acc_mean = []
        acc_max = []
        for i in range(3):
            acc_mean.append(np.mean(acc[i+4]))
            acc_max.append(np.max(np.abs(acc[i+4] - acc_mean[-1])))
        for i in range(1, len(acc[0])):
            if(i > pre_index + 100
               and (np.abs(acc[4][i] - acc_mean[0]) > acc_max[0] / 1e1
                   and np.abs(acc[5][i] - acc_mean[1]) > acc_max[1] / 1e1
                   and np.abs(acc[6][i] - acc_mean[2]) > acc_max[2] / 1e1)):
                sound_name = self.ex_path + '_' + str(count)
                self.devided.append(IMU(sound_name, self.data[:, i-self.size : i+self.size], self.columns, i-self.size, i+self.size, self.frame_rate))
                pre_index = i
                count += 1
                self.index.append([i - self.size, i + self.size])
    
    def fft(self):
        frequency = np.fft.rfftfreq(self.data.shape[1], 1.0/self.frame_rate)
        spectrum = np.block([[np.fft.rfft(self.data[1])], [np.fft.rfft(self.data[2])], [np.fft.rfft(self.data[3])],
                            [np.fft.rfft(self.data[4])], [np.fft.rfft(self.data[5])], [np.fft.rfft(self.data[6])],
                            [np.fft.rfft(self.data[7])], [np.fft.rfft(self.data[8])], [np.fft.rfft(self.data[9])]])
        
        fig, ax = plt.subplots(3,3,figsize=(24, 24))
        for i in range(9):
            ax[i%3][i//3].set_yscale("log")
            ax[i%3][i//3].set_title(self.columns[i+1])
            ax[i%3][i//3].scatter(frequency, np.abs(spectrum[i]), s=1)
        plt.tight_layout()
        plt.savefig(self.name +'_fft.png')
        plt.close()
    
    def do_IMU(self):
        for imu in self.devided:
            imu.print_simple()
            imu.fft()
            imu.spectrogram()
            imu.save()



# +
def fft_all(filepath, save_path):
    imus = IMUs(filepath, save_path)
    if imus.data.shape[1] > 0:
        imus.fft()
        imus.devide_stop()
        imus.print_simple()
        imus.do_IMU()
    
def IMUs_all(folderpath, save_path):
    filenames = [temp for temp in os.listdir(folderpath) if '.csv' in temp]
    filenames.sort()
    for filename in filenames:
        fft_all(folderpath+'/'+filename.split('.csv')[0], save_path+'/'+filename.split('.csv')[0])


# -

if __name__ == "__main__":
    #IMUs_all("../../data/original/3DM_GX3/sub7", "../../data/3DM_GX3s/raw/sub7")
    #fft_all("../../data/original/3DM_GX3/sub7/sub7_flick_002", "../../data/3DM_GX3s/raw/sub7/sub7_flick_002")    



