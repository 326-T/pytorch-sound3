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
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from modules.potential import Potential
from vae import VAE, VAE_trainer
from imu_vae import IMU_VAE, IMU_VAE_trainer
from modules.myfunc import ans2index_label_color_marker


class Potential_Map():
    
    def __init__(self, model, tech):
        self.model_type = model
        self.tech = tech
        self.potential = Potential()
        self.potential.load('../data/ranking', tech)
        self.vae = IMU_VAE_trainer()
        self.vae.load_weight(load_path =  '../result/' + self.model_type + '/' + self.tech + '/vae')
        self.vae.load(tech)
        _, self.Y, self.X = self.vae.plot_z(save_path = '../result/' + self.model_type + '/' + self.tech + '/z_map.png')
        self.potential.fit(self.X, self.Y, dim=4)
    
    def plot(self):
        fig = plt.figure(figsize=(20,12))
        ax1 = fig.add_subplot(1,2,1, projection='3d')
        x = np.arange(self.potential.min[0], self.potential.max[0], 0.01)
        y = np.arange(self.potential.min[1], self.potential.max[1], 0.01)
        x, y = np.meshgrid(x, y)
        z = self.potential.transform(np.stack([x.reshape(-1), y.reshape(-1)], 1))
        z = z.reshape(x.shape)
        ax1.plot_surface(x, y, z, cmap='plasma_r')
        ax2 = fig.add_subplot(1,2,2, projection='3d')
        x_each = self.X.transpose()
        z_each = self.potential.label2score(self.Y)
        idxs, labels, colors, markers = ans2index_label_color_marker(self.Y)
        for i, (label, color, marker) in enumerate(zip(labels,colors,markers)):
            ax2.scatter(x_each[0, idxs[i]:idxs[i+1]], x_each[1, idxs[i]:idxs[i+1]], z_each[idxs[i]:idxs[i+1]], 
                       label=label, s=10, color=color, marker=marker)
        plt.tight_layout()
        plt.legend()
        def spin_graph(i):
            ax1.view_init(30, i)
            ax2.view_init(30, i)
            return plt.gcf()
        
        ani = animation.FuncAnimation(fig, spin_graph, frames = 360, interval=50)
        #plt.show()
        ani.save('../result/' + self.model_type + '/' + self.tech + '/potential.mp4', writer="ffmpeg", dpi=100)


def plot_potential(model, tech):
    pmap = Potential_Map(model, tech)
    pmap.plot()
    del pmap


if __name__ == "__main__":
    plot_potential('IMU_VAE', 'drive')
    plot_potential('IMU_VAE', 'block')
    plot_potential('IMU_VAE', 'push')
    plot_potential('IMU_VAE', 'stop')
    plot_potential('IMU_VAE', 'flick')
    




