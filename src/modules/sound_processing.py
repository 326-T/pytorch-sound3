#!/usr/bin/env python
# coding: utf-8
# %%
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import os


# %%
class Sound:
    
    
    def __init__(self, name, src_data, start, end, sample_width = 4, frame_rate = 192000, channels=1):
        self.data = src_data
        self.name = name
        self.start = start
        self.end = end
        self.sample_width = sample_width
        self.frame_rate = frame_rate
        self.channels = channels
        
    def save(self):
        #sound_L = AudioSegment(self.data[0].astype("int32").tobytes(),
        #                      sample_width=self.sample_width,
        #                      frame_rate=self.frame_rate,
        #                      channels=self.channels)
        #sound_L.export(self.name + '-L.wav', format="wav")
        
        #sound_C = AudioSegment(self.data[1].astype("int32").tobytes(),
        #                      sample_width=self.sample_width,
        #                      frame_rate=self.frame_rate,
        #                      channels=self.channels)
        #sound_C.export(self.name + '-C.wav', format="wav")
        
        #sound_R = AudioSegment(self.data[2].astype("int32").tobytes(),
        #                      sample_width=self.sample_width,
        #                      frame_rate=self.frame_rate,
        #                      channels=self.channels)
        #sound_R.export(self.name + '-R.wav', format="wav")
        save_data = pd.DataFrame(data = self.data)
        save_data.to_csv(self.name + '.csv', index = False)
        
        
    def print_simple(self):
        time = range(self.start, self.end)
        fig, ax = plt.subplots(3,1,figsize=(8, 12))
        ax[0].set_title('L')
        ax[1].set_title('C')
        ax[2].set_title('R')
        for i in range(3):
            ax[i].plot(time, self.data[i], label="L", linewidth = 1)
        plt.tight_layout()
        plt.savefig(self.name + '.png')
        plt.close()
        
    def fft(self):
        frequency = np.fft.rfftfreq(self.data.shape[1], 1.0/self.frame_rate)
        spectrum = np.block([[np.fft.rfft(self.data[0])], [np.fft.rfft(self.data[1])], [np.fft.rfft(self.data[2])]])
        fig, ax = plt.subplots(3,1,figsize=(8, 12))
        ax[0].set_title('L')
        ax[1].set_title('C')
        ax[2].set_title('R')
        for i in range(3):
            ax[i].set_yscale("log")
            ax[i].set_xlabel('frequency[Hz]')
            ax[i].scatter(frequency, np.abs(spectrum[i]), s=1)
        plt.tight_layout()
        plt.savefig(self.name.split('raw/')[0] + 'fft/' + self.name.split('raw/')[-1] + '.png')
        plt.close()
        save_data = pd.DataFrame(data = spectrum, index = ['L', 'C', 'R'], columns = frequency)
        save_data.to_csv(self.name.split('raw/')[0] + 'fft/' + self.name.split('raw/')[-1] + '.csv')
        
    def spectrogram(self, width = 128, step = 32):
        window = np.hamming(width)
        ampLists = []
        argLists = []
        for i in range(3):
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
        time = np.arange(0, len(ampList), 1) * step
        
        #plot
        fig, ax = plt.subplots(3,1,figsize=(8, 12))
        ax[0].set_title('L')
        ax[1].set_title('C')
        ax[2].set_title('R')
        for i in range(3):
            ax[i].imshow(ampLists[i], extent=[time[0], time[-1], freq[0], freq[-1]], aspect="auto", origin='lower', cmap='afmhot')
        plt.tight_layout()
        plt.savefig(self.name.split('raw/')[0] + 'spectrogram/' + self.name.split('raw/')[-1] + '.png')
        plt.close()
        
class Sounds:
    def __init__(self, name, ex_path, size=20000):
        self.devided = []
        self.size = size
        self.index = []
        self.name = name
        data_l = AudioSegment.from_file(name+'-L.wav', "wav")
        data_c = AudioSegment.from_file(name+'-C.wav', "wav")
        data_r = AudioSegment.from_file(name+'-R.wav', "wav")
        array_l = np.array(data_l.get_array_of_samples())
        array_c = np.array(data_c.get_array_of_samples())
        array_r = np.array(data_r.get_array_of_samples())
        len_l = len(array_l[::data_l.channels])
        len_c = len(array_c[::data_c.channels])
        len_r = len(array_r[::data_r.channels])
        if len_l == len_c and len_c == len_r and len_r == len_l:
            self.data = np.block([[array_l[::data_l.channels]], [array_c[::data_c.channels]], [array_r[::data_r.channels]]])
        else:
            start = np.min([len_l, len_c, len_r])
            self.data = np.block([[array_l[-start::data_l.channels]], [array_c[-start::data_c.channels]], [array_r[-start::data_r.channels]]])
            print(self.name)
        self.frame_rate = data_l.frame_rate
        self.ex_path = ex_path
        
    def print_simple(self):
        time = range(len(self.data[0]))
        fig, ax = plt.subplots(3,1,figsize=(16,12))
        ax[0].set_title('L')
        ax[1].set_title('C')
        ax[2].set_title('R')
        for i in range(3):
            ax[i].plot(time, self.data[i])
            for idx in self.index:
                #ax[i].axvline(np.mean(idx), ls = "--", color = "red")
                ax[i].scatter(np.mean(idx), self.data[i, int(np.mean(idx))], s=30, color = 'red')
        plt.tight_layout()
        plt.suptitle(str(len(self.index)), fontsize=40)
        plt.subplots_adjust(top=0.9)
        plt.savefig(self.name + '.png')
        plt.close()
        
    def devide(self):
        pre_index = 0
        count = 1
        sound_mean = []
        sound_max = []
        for i in range(3):
            sound_mean.append(np.mean(self.data[i]))
            sound_max.append(np.max(np.abs(self.data[i] - sound_mean[-1])))
        for i in range(self.size, self.data.shape[1]-self.size):
            if(i > pre_index + self.frame_rate
               and (np.abs(self.data[0][i] - sound_mean[0]) > sound_max[0] / 3
                    or np.abs(self.data[1][i] - sound_mean[1]) > sound_max[1] / 5
                    or np.abs(self.data[2][i] - sound_mean[2]) > sound_max[2] / 3)):
                sound_name = self.ex_path + '_' + str(count)
                self.devided.append(Sound(sound_name, self.data[:, i - self.size : i + self.size], i-self.size, i+self.size, self.frame_rate))
                pre_index = i
                count += 1
                self.index.append([i - self.size, i + self.size])
    
    def fft(self):
        frequency = np.fft.rfftfreq(self.data.shape[1], 1.0/self.frame_rate)
        spectrum = np.block([[np.fft.rfft(self.data[0])], [np.fft.rfft(self.data[1])], [np.fft.rfft(self.data[2])]])
        
        fig, ax = plt.subplots(3,1,figsize=(8, 12))
        ax[0].set_title('L')
        ax[1].set_title('C')
        ax[2].set_title('R')
        for i in range(3):
            ax[i].set_yscale("log")
            ax[i].set_xlabel('frequency[Hz]')
            ax[i].scatter(frequency, np.abs(spectrum[i]), s=1)
        plt.tight_layout()
        plt.savefig(self.name + '_fft.png')
        plt.close()
    
    def do_sound(self):
        for sound in self.devided:
            sound.print_simple()
            sound.fft()
            sound.spectrogram()
            sound.save()



# %%
def fft_all(filepath ,savepath):
    sounds = Sounds(filepath, savepath)
    sounds.fft()
    #sounds.save()
    sounds.devide()
    sounds.print_simple()
    sounds.do_sound()
    
def spectrogram_all(folderpath):
    filenames = [temp for temp in os.listdir(folderpath) if '.csv' in temp]
    for filename in filenames:
        sound = Sound(folderpath+'/'+filename)
        sound.spectrogram()

def sound_all(folderpath, save_path):
    filenames = [temp for temp in os.listdir(folderpath) if '-L.wav' in temp]
    filenames.sort()
    for filename in filenames:
        sounds = Sounds(folderpath+'/'+filename.split('-L.wav')[0], save_path+'/'+filename.split('-L.wav')[0])
        sounds.fft()
        sounds.devide()
        sounds.print_simple()
        sounds.do_sound()


# %%
class array2sound:
    def __init__(self, src, save_path):
        self.save_path = save_path
        self.data = []
        self.mix = []
        for s in src:
            temp = AudioSegment(data=s.astype("int16").tobytes(), frame_rate=4800, sample_width=2, channels=2)
            self.data.append(temp)
        
    def save(self, names):
        for i, data in enumerate(self.data):
            plt.figure()
            plt.plot(data.get_array_of_samples()[2400:6400])
            plt.savefig(self.save_path+names[i].split('.wav')[0] + '.png')
            plt.close()
            mix = data + self.data[0]
            plt.figure()
            plt.plot(mix.get_array_of_samples())
            plt.savefig(self.save_path+'mix_'+str(i)+'.png')
            plt.close()
            
            data = data * 10
            data.export(self.save_path+names[i], format="wav")
            mix = mix
            mix.export(self.save_path+'mix_'+str(i)+'.wav')


# %%
if __name__ == "__main__":
    fft_all("../../data/original/sound/sub5/sub5_drive_001", "../../data/sounds/raw/sub5/sub5_drive_001")
    fft_all("../../data/original/sound/sub5/sub5_drive_004", "../../data/sounds/raw/sub5/sub5_drive_003")
    fft_all("../../data/original/sound/sub5/sub5_drive_003", "../../data/sounds/raw/sub5/sub5_drive_004")
    fft_all("../../data/original/sound/sub11/sub11_drive_002", "../../data/sounds/raw/sub11/sub11_drive_002")
    fft_all("../../data/original/sound/sub11/sub11_drive_004", "../../data/sounds/raw/sub11/sub11_drive_004")
    #sound_all("../../data/original/sound/sub1", "../../data/sounds/raw/sub1")
    #sound_all("../../data/original/sound/sub2", "../../data/sounds/raw/sub2")
    #sound_all("../../data/original/sound/sub3", "../../data/sounds/raw/sub3")
    #sound_all("../../data/original/sound/sub4", "../../data/sounds/raw/sub4")
    #sound_all("../../data/original/sound/sub5", "../../data/sounds/raw/sub5")
    #sound_all("../../data/original/sound/sub6", "../../data/sounds/raw/sub6")
    #sound_all("../../data/original/sound/sub7", "../../data/sounds/raw/sub7")

# %%


# %%

# %%
