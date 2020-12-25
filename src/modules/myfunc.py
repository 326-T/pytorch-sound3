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
import os
import seaborn as sns

def sfm_ans2index(sfm_answer):
    index = []
    label = []
    answer = []
    for i in sfm_answer:
        answer.append(np.argmax(i)+1)
        
    for i, ans in enumerate(answer):
        if not str(ans) in label:
            index.append(i)
            label.append(str(ans))
    index.append(len(answer))
    return index, label


def ans2index_label(answer):
    index = []
    label = []
    for i, ans in enumerate(answer):
        if not str(int(ans)+1) in label:
            index.append(i)
            label.append(str(int(ans)+1))
    index.append(len(answer))
    return index, label


def ans2index_label_color_marker(answer):
    index = []
    label = []
    color = []
    marker = []
    for i, ans in enumerate(answer):
        if not str(int(ans)+1) in label:
            index.append(i)
            label.append(str(int(ans)+1))
            marker.append(".")
    index.append(len(answer))
    num_label = len(label)
    cm = plt.cm.get_cmap('tab20')
    for i in range(num_label):
        rgb = cm(i)
        color.append(rgb)
    
    return index, label, color, marker


def ans_score2index_label_color_marker(answer, _score):
    index = []
    label = []
    color = []
    marker = []
    score = []
    for i, ans in enumerate(answer):
        if not str(int(ans)+1) in label:
            index.append(i)
            label.append(str(int(ans)+1))
            score.append(_score[i])
    
    idx = np.argsort(score).tolist()
    hl = np.ones(len(idx), dtype='int')
    hl[idx[:int((len(idx)+1)/2)]] = 0
    for i in hl:
        if i == 0:
            marker.append("x")
        else:
            marker.append("o")
    
    index.append(len(answer))
    num_label = len(label)
    cm = plt.cm.get_cmap('tab20')
    for i in range(num_label):
        rgb = cm(i)
        color.append(rgb)
    
    return index, label, color, marker


def optional_split(line, key, place):
    temp = line.split(key)
    
    if place < 0:
        place = place + len(temp)
    
    if place <= 0:
        first = line
        latter = ''
        
    elif place >= len(temp):
        first = ''
        latter = line
    
    else:
        first = temp[0]
        latter = temp[place]
        for i in range(1,place):
            first += key + temp[i]
        for i in range(place+1, len(temp)):
            latter += key + temp[i]

    return first, latter

def print_ranking_bar(path, key):
    def names2numbers(names):
        numbers = []
        for name in names:
            if name != 'Bye':
                numbers.append(int(name.split('選手')[-1]))
        return numbers
    filenames = [temp for temp in os.listdir(path) if ('.csv' in temp and key in temp)]
    ranks = []
    for filename in filenames:
        df = pd.read_csv(path+'/'+filename, index_col=0)
        rank = names2numbers(df.columns.values)
        ranks.append(np.argsort(rank))
    ranks = np.array(ranks)
    idx = np.argsort(np.mean(ranks, 0))
    ranks = np.concatenate([idx.reshape(-1,1), ranks.transpose()[idx]], axis=1).transpose()
    ranks[1:] += 1 

    fig, ax = plt.subplots(1,1,figsize=(12,8))
    x = np.arange(1,len(ranks[0])+1)
    w = 0.8
    n = len(ranks[1:])
    for i, y in enumerate(ranks[1:]):
        ax.bar(x+(-1/2+1/2/n+i/n)*w, y, width=w/n, label='coach'+str(i), align="center")
    ax.xaxis.tick_top()
    ax.set_ylim(len(ranks[0]), 0)
    ax.set_ylabel('ranking')
    ax.legend(loc=3)
    plt.xticks(x, ['player'+str(id+1) for id in ranks[0]])
    plt.savefig(path+'/result_'+key+'.png')

    return ranks[:, np.argsort(ranks[0])]

def print_ranking(path, key):
    def names2numbers(names):
        numbers = []
        for name in names:
            if name != 'Bye':
                numbers.append(int(name.split('選手')[-1]))
        return numbers
    filenames = [temp for temp in os.listdir(path) if ('.csv' in temp and key in temp)]
    ranks = []
    for filename in filenames:
        df = pd.read_csv(path+'/'+filename, index_col=0)
        rank = names2numbers(df.columns.values)
        ranks.append(np.argsort(rank))
    ranks = np.array(ranks)
    idx = np.argsort(np.mean(ranks, 0))
    ranks = np.concatenate([idx.reshape(-1,1), ranks.transpose()[idx]], axis=1).transpose()
    ranks[1:] += 1 

    fig, ax = plt.subplots(1,1,figsize=(12,8))
    x = np.arange(1,len(ranks[0])+1)
    n = len(ranks[1:])
    for i, y in enumerate(ranks[1:]):
        ax.plot(x, y, label='coach'+str(i))
    ax.xaxis.tick_top()
    ax.set_ylim(len(ranks[0]), 0)
    ax.set_ylabel('ranking')
    ax.legend(loc=3)
    plt.xticks(x, ['player'+str(id+1) for id in ranks[0]])
    plt.savefig(path+'/result_'+key+'.png')

def calc_corr(path, keys):
    result = np.arange(1,12).reshape(1, -1)
    columns = []
    for key in keys:
        rank = print_ranking_bar(path, key)
        result = np.concatenate([result, rank[1:]], axis=0)
        columns.append(['coach'+str(i+1)+'_'+key for i in range(len(rank[1:]))])
    df = pd.DataFrame(data=result[1:].transpose(), index=['player'+str(name) for name in result[0]], columns = np.array(columns).reshape(-1))
    print(df.corr())
    corr = df.corr()
    corr.to_csv(path + '/correlation.csv')
    for i in range(len(rank[1:])):
        cols = ['coach'+str(i+1)+'_'+key for key in keys]
        corr_i = corr.loc[cols,cols].copy()
        corr_i.index = keys
        corr_i.columns = keys
        corr_i.to_csv(path+'/correlation_coach'+str(i+1)+'.csv')
        fig, ax = plt.subplots(1,1,figsize=(12,12))
        ax = sns.heatmap(corr_i, annot=True, fmt='.3f', square=True)
        ax.invert_yaxis()
        plt.savefig(path+'/correlation_coach'+str(i+1)+'.png')

if __name__=="__main__":
    keys = ['drive', 'block', 'push', 'stop', 'flick']
    calc_corr('../../data/ranking', keys)