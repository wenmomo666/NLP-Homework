#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 01:54:24 2019

@author: yiwenzhang
"""
import numpy as np

def getIndex(filename = None):
    index = dict()
    with open(filename, 'r') as file1:
        for line in file1:
            token, ind = line.strip().split()
            index[token] = int(ind)    
    return index

def Indexing(file1 = None, file2 = None, indices = None):
    index_of_words, index_of_pos, index_of_labels, index_of_actions = indices
    with open(file1, 'r') as f1, open(file2, 'w+') as f2:
        for line in f1:
            tokens = line.strip().split()            
            words = tokens[0:20]
            poss = tokens[20: 20 + 20]
            labels = tokens[40: 40 + 12]
            action = tokens[52]
            newline = ''            
            for word in words:
                if(word in index_of_words):
                    newline += str(index_of_words[word]) + ' '
                else:
                    newline += str(index_of_words['<unk>']) + ' '                
            for pos in poss:
                if(pos in index_of_pos):
                    newline += str(index_of_pos[pos]) + ' '
                else:
                    newline += str(index_of_pos['<null>'])                
            for label in labels:
                newline += str(index_of_labels[label]) + ' '            
            newline += str(index_of_actions[action]) + '\n'
            f2.write(newline)
def loadnew(filename = 'data/train_with_indices.data'):
    train_data = []
    train_label = []
    with open(filename) as lines:
        for line in lines:
            tokens = [int(x) for x in line.strip().split()]         
            train_data.append(tokens[0:52])
            train_label.append(tokens[52])    
    return np.asarray(train_data), np.asarray(train_label)



