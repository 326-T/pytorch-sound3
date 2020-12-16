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

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import os

from modules.tournament import Swiss_System_Tournament, Player
from modules.myfunc import ans2index_label_color_marker


class Potential():
    
    def __init__(self):
        self.results = []
    
    def load(self, path, key):
        filenames = [temp for temp in os.listdir(path) if ('.csv' in temp and key in temp)]
        self.key = key
        for filename in filenames:
            tournament = Swiss_System_Tournament()
            tournament.Load(path + '/' + filename)
            self.results.append(tournament)
        self.score = np.zeros(self.results[0].participants)
        self.calc_score()
        
    def calc_score(self):
        for result in self.results:
            for player in result.players:
                if player.name != "Bye":
                    id = int(player.name.split('選手')[-1]) - 1
                    self.score[id] += player.score
        self.score = (self.score - np.min(self.score)) / (np.max(self.score) - np.min(self.score))
    
    def label2score(self, Y):
        y = Y.reshape(-1)
        s = np.zeros_like(y)
        for i, x in enumerate(y):
            s[i] = self.score[int(x)]
        S = s.reshape(Y.shape)
        return S


class MRA_Potential(Potential):
    
    def fit(self, X, Y, dim = 4):
        self.max = X.max(axis=0)
        self.min = X.min(axis=0)
        self.pf = PolynomialFeatures(dim)
        mat_X = self.pf.fit_transform(X)
        mat_Y = self.label2score(Y)
        self.A = np.linalg.pinv(mat_X)@mat_Y
        
    def transform(self, X):
        mat_X = X.copy()
        if mat_X.ndim == 1:
            mat_X = mat_X.reshape([1, -1])
        mat_X = self.pf.fit_transform(X)
        mat_Y = mat_X@self.A
        Y = mat_Y.reshape([-1])
        if len(Y) == 1:
            return Y[0]
        else:
            return Y


class GMM_Potential(Potential):
    
    def fit(self, X, Y, dim = 0): # dim is dummy
        idxs, _, _, _, = ans2index_label_color_marker(Y)
        self.gauss = []
        for i in range(len(idxs)-1):
            gauss.append(MultiDimGauss(X[idxs[i]:idxs[i+1]], label2score(Y[idxs[i]])))
            
    def transform(self, X):
        Y = np.zeros(len(X), dtype=float)
        for i, x in enumerate(X):
            for gauss in self.gauss:
                Y[i] += gauss.calc(x)
        if len(Y) == 1:
            return Y[0]
        else:
            return Y
        


class MultiDimGauss():
    
    def __init__(self, X, k=1):
        self.Mean = np.matrix(np.mean(X, axis=1))
        self.Sigma = np.cov(X.T)
        self.k = k
    
    def calc(self, _X):
        X = np.matrix(_X)
        a = np.sqrt(np.linalg.det(self.Sigma)*(2*np.pi)**self.Sigma.ndim)
        b = np.linalg.det(-0.5*(X - self.Mean)*Self.Sigma.I*(X - self.Mean).T)
        return self.k*np.exp(b)/a
        


if __name__ == "__main__":
    potential = Potential()
    potential.load('../data/ranking', 'drive')
    potential.fit(X, Y, dim=3)
    potential.transform(X)


