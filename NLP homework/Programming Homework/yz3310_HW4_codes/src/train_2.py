#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 01:57:28 2019

@author: yiwenzhang
"""

import tensorflow as tf
import numpy as np
from utils_1 import Indexing, getIndex, loadnew
import sys
from keras.models import Model
from keras.layers import Dense, Input, Embedding, Reshape, Concatenate, Lambda
import keras

index_of_words = getIndex(filename = './data/vocabs.word')
index_of_labels = getIndex(filename = './data/vocabs.labels')
index_of_pos = getIndex(filename = './data/vocabs.pos')
index_of_actions = getIndex(filename = './data/vocabs.actions')
Indexing(file1 = './data/train.data', file2= './data/train_with_indices.data', indices = [index_of_words, index_of_pos, index_of_labels, index_of_actions])
train_data, train_labels = loadnew(filename = './data/train_with_indices.data')

def output_shape_words(input_shape):
    assert(len(list(input_shape)) == 2)
    assert(input_shape[1] == 52)
    return (input_shape[0], 20)

def output_shape_tags(input_shape):
    assert(len(list(input_shape)) == 2)
    assert(input_shape[1] == 52)
    return (input_shape[0], 20)

def output_shape_labels(input_shape):
    assert(len(list(input_shape)) == 2)
    assert(input_shape[1] == 52)
    return (input_shape[0], 12)
I=Input(shape = (52, ))
words = Lambda(function = lambda x: x[:, 0: 20], output_shape = output_shape_words)(I)
tags = Lambda(function = lambda x: x[:, 20: 20 + 20], output_shape = output_shape_tags)(I)
labels = Lambda(function = lambda x: x[:, 40: 40 + 41], output_shape = output_shape_labels)(I)

embedding_words = Reshape(target_shape = (20 * 64, ))(Embedding(input_dim = len(index_of_words),output_dim = 64,input_length = 20)(words))
embedding_tags = Reshape(target_shape = (32 * 20,) )(Embedding(input_dim = len(index_of_pos),output_dim = 32,input_length = 20)(tags))
embedding_labels = Reshape(target_shape = (32 * 12, ))(Embedding(input_dim = len(index_of_labels),output_dim = 32,input_length = 12)(labels))
embeddings = Concatenate(axis = 1)([embedding_words, embedding_tags, embedding_labels])
h1 = Dense(units = 400, activation = 'relu')(embeddings)
h2 = Dense(units = 400, activation = 'relu')(h1)

q = Dense(units = 93, activation = 'softmax')(h2)
model = Model(inputs = [I], outputs = [q])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())
model.fit(train_data, train_labels, epochs = 7, batch_size = 1000)
model.save(filepath = './models/model2')


