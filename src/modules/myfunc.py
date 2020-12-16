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
import matplotlib as plt


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
            if (int(ans)+1 == 1
                or int(ans)+1 == 5 
                or int(ans)+1 == 8 
                or int(ans)+1 == 11):
                marker.append(".")
            else:
                marker.append(".")
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
