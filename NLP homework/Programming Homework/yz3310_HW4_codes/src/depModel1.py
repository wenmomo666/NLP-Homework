#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 01:50:52 2019

@author: yiwenzhang
"""

import os,sys
from decoder import *
from utils_1 import getIndex
from keras.models import load_model
class DepModel:
    def __init__(self):
        '''
            You can add more arguments for examples actions and model paths.
            You need to load your model here.
            actions: provides indices for actions.
            it has the same order as the data/vocabs.actions file.
        '''       
        # Load the indices from vocab files
        self.index_of_words = getIndex(filename = 'data/vocabs.word')
        self.index_of_pos = getIndex(filename = 'data/vocabs.pos')
        self.index_of_labels = getIndex(filename = 'data/vocabs.labels')
        self.index_of_actions = getIndex(filename = 'data/vocabs.actions')
        
        index_of_actions_items = self.index_of_actions.items()
        sorted_index_of_actions =  sorted(index_of_actions_items, key = lambda x: x[1])
        sorted_actions = [x[0] for x in sorted_index_of_actions] 
        self.actions = sorted_actions

        self.model = load_model('models/model1')

    def score(self, str_features):
        '''
        :param str_features: String features
        20 first: words, next 20: pos, next 12: dependency labels.
        DO NOT ADD ANY ARGUMENTS TO THIS FUNCTION.
        :return: list of scores
        '''
        # transform this feature vector to vector of indices
        words = str_features[0:20]
        pos = str_features[20: 20 + 20]
        labels = str_features[40: 40 + 12]
        
        indiceFeature = []
        for word in words:
            if(word in self.index_of_words):
                indiceFeature.append(self.index_of_words[word])
            else:
                indiceFeature.append(self.index_of_words['<unk>'])

        for tag in pos:
            if(tag in self.index_of_pos):
                indiceFeature.append(self.index_of_pos[tag])
            else:
                indiceFeature.append(self.index_of_pos['<null>'])
        
        for label in labels:
            indiceFeature.append(self.index_of_labels[label])
        return self.model.predict(x = np.array([indiceFeature]),batch_size = 1)[0]

if __name__=='__main__':
    m = DepModel()
    input_p = os.path.abspath(sys.argv[1])
    output_p = os.path.abspath(sys.argv[2])
    Decoder(m.score, m.actions).parse(input_p, output_p)

