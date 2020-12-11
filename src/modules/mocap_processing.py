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
class Mocap:
    
    
    def __init__(self, name, data, columns, start, end, frame_rate=240):
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
        
class Mocaps:
    def __init__(self, name, ex_path, frame_rate = 240):
        self.devided = []
        self.index = []
        self.name = name
        x = pd.read_csv(name + '.csv', skiprows = 6)
        x.iloc[:,:].astype('float')
        data = x.iloc[:,[1,
                         2,3,4,5,6,7,8,6,7,8,
                         21,22,23,24,25,26,27,25,26,27, 
                         28,29,30,31,32,33,34,32,33,34, 
                         47,48,49,50,51,52,53,51,52,53, 
                         54,55,56,57,58,59,60,58,59,60, 
                         70,71,72,73,74,75,76,74,75,76, 
                         83,84,85,86,87,88,89,87,88,89, 
                         96,97,98,99,100,101,102,100,101,102, 
                         103,104,105,106,107,108,109,107,108,109, 
                         119,120,121,122,123,124,125,123,124,125, 
                         132,133,134,135,136,137,138,136,137,138, 
                         145,146,147,148,149,150,151,149,150,151, 
                         152,153,154,155,156,157,158,156,157,158, 
                         168,169,170,171,172,173,174,172,173,174, 
                         181,182,183,184,185,186,187,185,186,187, 
                         194,195,196,197,198,199,200,198,199,200,
                         207,208,209,210,211,212,213,211,212,213,
                         220,221,222,223,224,225,226,224,225,226,
                         233,234,235,236,237,238,239,237,238,239,
                         246,247,248,249,250,251,252,250,251,252,
                         253,254,255,256,257,258,259,257,258,259,
                         260,261,262,263,264,265,266,264,265,266]]
        print(data)
        value = data.values
        self.data = value[:-1,:].copy()
        self.data[:,8::10] = (self.data[:,8::10] - value[1:,8::10]) * frame_rate
        self.data[:,9::10] = (self.data[:,9::10] - value[1:,9::10]) * frame_rate
        self.data[:,10::10] = (self.data[:,10::10] - value[1:,10::10]) * frame_rate
        self.data = self.data.transpose()
        self.columns = ['Time',
                       'Hip_Rot_X', 'Hip_Rot_Y', 'Hip_Rot_Z', 'Hip_Rot_W', 'Hip_Pos_X', 'Hip_Pos_Y', 'Hip_Pos_Z', 'Hip_Vel_X', 'Hip_Vel_Y', 'Hip_Vel_Z',
                       'Ab_Rot_X', 'Ab_Rot_Y', 'Ab_Rot_Z', 'Ab_Rot_W', 'Ab_Pos_X', 'Ab_Pos_Y', 'Ab_Pos_Z', 'Ab_Vel_X', 'Ab_Vel_Y', 'Ab_Vel_Z',
                       'Chest_Rot_X', 'Chest_Rot_Y', 'Chest_Rot_Z', 'Chest_Rot_W', 'Chest_Pos_X', 'Chest_Pos_Y', 'Chest_Pos_Z', 'Chest_Vel_X', 'Chest_Vel_Y', 'Chest_Vel_Z',
                       'Neck_Rot_X', 'Neck_Rot_Y', 'Neck_Rot_Z', 'Neck_Rot_W', 'Neck_Pos_X', 'Neck_Pos_Y', 'Neck_Pos_Z', 'Neck_Vel_X', 'Neck_Vel_Y', 'Neck_Vel_Z',
                       'Head_Rot_X', 'Head_Rot_Y', 'Head_Rot_Z', 'Head_Rot_W', 'Head_Pos_X', 'Head_Pos_Y', 'Head_Pos_Z', 'Head_Vel_X', 'Head_Vel_Y', 'Head_Vel_Z',
                       'LShoulder_Rot_X', 'LShoulder_Rot_Y', 'LShoulder_Rot_Z', 'LShoulder_Rot_W', 'LShoulder_Pos_X', 'LShoulder_Pos_Y', 'LShoulder_Pos_Z', 'LShoulder_Vel_X', 'LShoulder_Vel_Y', 'LShoulder_Vel_Z',
                       'LUArm_Rot_X', 'LUArm_Rot_Y', 'LUArm_Rot_Z', 'LUArm_Rot_W', 'LUArm_Pos_X', 'LUArm_Pos_Y', 'LUArm_Pos_Z', 'LUArm_Vel_X', 'LUArm_Vel_Y', 'LUArm_Vel_Z',
                       'LFArm_Rot_X', 'LFArm_Rot_Y', 'LFArm_Rot_Z', 'LFArm_Rot_W', 'LFArm_Pos_X', 'LFArm_Pos_Y', 'LFArm_Pos_Z', 'LFArm_Vel_X', 'LFArm_Vel_Y', 'LFArm_Vel_Z',
                       'LHand_Rot_X', 'LHand_Rot_Y', 'LHand_Rot_Z', 'LHand_Rot_W', 'LHand_Pos_X', 'LHand_Pos_Y', 'LHand_Pos_Z', 'LHand_Vel_X', 'LHand_Vel_Y', 'LHand_Vel_Z',
                       'RShoulder_Rot_X', 'RShoulder_Rot_Y', 'RShoulder_Rot_Z', 'RShoulder_Rot_W', 'RShoulder_Pos_X', 'RShoulder_Pos_Y', 'RShoulder_Pos_Z', 'RShoulder_Vel_X', 'RShoulder_Vel_Y', 'RShoulder_Vel_Z',
                       'RUArm_Rot_X', 'RUArm_Rot_Y', 'RUArm_Rot_Z', 'RUArm_Rot_W', 'RUArm_Pos_X', 'RUArm_Pos_Y', 'RUArm_Pos_Z', 'RUArm_Vel_X', 'RUArm_Vel_Y', 'RUArm_Vel_Z',
                       'RFArm_Rot_X', 'RFArm_Rot_Y', 'RFArm_Rot_Z', 'RFArm_Rot_W', 'RFArm_Pos_X', 'RFArm_Pos_Y', 'RFArm_Pos_Z', 'RFArm_Vel_X', 'RFArm_Vel_Y', 'RFArm_Vel_Z',
                       'RHand_Rot_X', 'RHand_Rot_Y', 'RHand_Rot_Z', 'RHand_Rot_W', 'RHand_Pos_X', 'RHand_Pos_Y', 'RHand_Pos_Z', 'RHand_Vel_X', 'RHand_Vel_Y', 'RHand_Vel_Z',
                       'LThigh_Rot_X', 'LThigh_Rot_Y', 'LThigh_Rot_Z', 'LThigh_Rot_W', 'LThigh_Pos_X', 'LThigh_Pos_Y', 'LThigh_Pos_Z', 'LThigh_Vel_X', 'LThigh_Vel_Y', 'LThigh_Vel_Z',
                       'LShin_Rot_X', 'LShin_Rot_Y', 'LShin_Rot_Z', 'LShin_Rot_W', 'LShin_Pos_X', 'LShin_Pos_Y', 'LShin_Pos_Z', 'LShin_Vel_X', 'LShin_Vel_Y', 'LShin_Vel_Z',
                       'LFoot_Rot_X', 'LFoot_Rot_Y', 'LFoot_Rot_Z', 'LFoot_Rot_W', 'LFoot_Pos_X', 'LFoot_Pos_Y', 'LFoot_Pos_Z', 'LFoot_Vel_X', 'LFoot_Vel_Y', 'LFoot_Vel_Z',
                       'RThigh_Rot_X', 'RThigh_Rot_Y', 'RThigh_Rot_Z', 'RThigh_Rot_W', 'RThigh_Pos_X', 'RThigh_Pos_Y', 'RThigh_Pos_Z', 'RThigh_Vel_X', 'RThigh_Vel_Y', 'RThigh_Vel_Z',
                       'RShin_Rot_X', 'RShin_Rot_Y', 'RShin_Rot_Z', 'RShin_Rot_W', 'RShin_Pos_X', 'RShin_Pos_Y', 'RShin_Pos_Z', 'RShin_Vel_X', 'RShin_Vel_Y', 'RShin_Vel_Z',
                       'RFoot_Rot_X', 'RFoot_Rot_Y', 'RFoot_Rot_Z', 'RFoot_Rot_W', 'RFoot_Pos_X', 'RFoot_Pos_Y', 'RFoot_Pos_Z', 'RFoot_Vel_X', 'RFoot_Vel_Y', 'RFoot_Vel_Z',
                       'LToe_Rot_X', 'LToe_Rot_Y', 'LToe_Rot_Z', 'LToe_Rot_W', 'LToe_Pos_X', 'LToe_Pos_Y', 'LToe_Pos_Z', 'LToe_Vel_X', 'LToe_Vel_Y', 'LToe_Vel_Z',
                       'RToe_Rot_X', 'RToe_Rot_Y', 'RToe_Rot_Z', 'RToe_Rot_W', 'RToe_Pos_X', 'RToe_Pos_Y', 'RToe_Pos_Z', 'RToe_Vel_X', 'RToe_Vel_Y', 'RToe_Vel_Z',
                       'Racket_Rot_X', 'Racket_Rot_Y', 'Racket_Rot_Z', 'Racket_Rot_W', 'Racket_Pos_X', 'Racket_Pos_Y', 'Racket_Pos_Z', 'Racket_Vel_X', 'Racket_Vel_Y', 'Racket_Vel_Z']
                       
        self.frame_rate = frame_rate
        self.ex_path = ex_path
        
    def print_simple(self):
        fig, ax = plt.subplots(3,1,figsize=(24,24))
        for i in range(9):
            ax[i%3][i//3].set_title(self.columns[i+1])
            ax[i%3][i//3].plot(self.data[0], self.data[i+1])
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
                self.devided.append(Mocap(sound_name, self.data[:, i-50 : i+50], self.columns, i-50, i+50, self.frame_rate))
                pre_index = i
                count += 1
                self.index.append([i - 50, i + 50])
    
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
    
    def do_Mocap(self):
        for imu in self.devided:
            imu.print_simple()
            imu.fft()
            imu.spectrogram()
            imu.save()



# +
def fft_all(filepath, save_path):
    mocaps = Mocaps(filepath, save_path)
    mocaps.fft()
    mocaps.devide()
    mocaps.print_simple()
    mocaps.do_IMU()
    
def Mocaps_all(folderpath, save_path):
    filenames = [temp for temp in os.listdir(folderpath) if '.csv' in temp]
    filenames.sort()
    for filename in filenames:
        mocaps = Mocaps(folderpath+'/'+filename.split('.csv')[0], save_path+'/'+filename.split('.csv')[0])
        if mocaps.data.shape[1] > 0:
            mocaps.fft()
            mocaps.devide()
            mocaps.print_simple()
            mocaps.do_IMU()


# -

mocaps = Mocaps("../../data/original/mocap/sub1/sub1_drive_001", "../../data/mocaps/sub1/sub1_drive_001")


if __name__ == "__main__":
    #IMUs_all("../../data/original/3DM_GX3/sub1", "../../data/3DM_GX3s/raw/sub1")


