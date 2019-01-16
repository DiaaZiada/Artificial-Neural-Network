#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:24:48 2019

@author: diaa
"""
import numpy as np
import math

def batches_maker(X, Y, mini_batch_size = 64, shuffle=False):
    """
    Function:
        Creates a list of random minibatches from (X, Y)
    
    Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        mini_batch_size -- size of the mini-batches, integer
        shuffle -- bool value to make shuffled data or not
        
    Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[1]
    mini_batches = []
    
    if shuffle:
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1,m))
    else:
        shuffled_X = X
        shuffled_Y = Y
        
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(num_complete_minibatches):
        
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch = (shuffled_X, shuffled_Y)
       
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size :]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size :]
        
        mini_batch = (np.array(mini_batch_X), np.array(mini_batch_Y))
        mini_batches.append(mini_batch)
    return mini_batches