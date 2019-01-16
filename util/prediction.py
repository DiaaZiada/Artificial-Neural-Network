#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:35:50 2019

@author: diaa
"""
import numpy as np
from .forward import L_model_forward#(x, parameters,keep_prob)

def predict(X,parameters):
    """
    Function:
        This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
    
    Display:
        predictions for the given dataset X
    """
    Y_, _, _ = L_model_forward(X, parameters, keep_prob=None)
    for y_ in range(0, Y_.shape[1]):
        print("class precitions is: {}".format(np.argmax(y_)))     
        
def accuracy_precitor(X, Y, parameters):
    """
    Function:
        This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
        X -- data set of examples 
        Y -- data set of labels
        parameters -- parameters of the trained model
    
    Returns:
    p -- accuracy of the model
    """
    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)
    Y_, _, _ = L_model_forward(X, parameters, keep_prob=None)
    
    for i in range(0, Y_.shape[1]):
        if Y_[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    print("Accuracy: "  + str(np.mean((p[0,:] == Y[0,:]))))
    return p


