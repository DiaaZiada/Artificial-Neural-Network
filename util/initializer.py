#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 05:23:17 2019

@author: diaa
"""
import numpy as np

def initialize_parameters_(layer_dims):
    """
    Function:
        initialize parameters using random number from 0 to 0.99.. range, multiplied by 0.01
    Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    L = len(layer_dims)
    
    parameters = {}
    
    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters["b"+str(l)] = np.zeros((layer_dims[l], 1))
        assert parameters["W"+str(l)].shape == (layer_dims[l], layer_dims[l - 1])
        assert parameters["b"+str(l)].shape == (layer_dims[l], 1)
    return parameters


def initialize_parameters(layer_dims): 
    """
    Function:
        initialize parameters using random number from 0 to 0.99.. range, multiplied by sqrt(layer_dims[l-1])
    Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """    
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['W' + str(l)].shape == layer_dims[l], 1)

        
    return parameters

        
    return parameters
def initialize_parameters_he(layer_dims):
    """
    Function:
        initialize parameters using random number from 0 to 0.99.. range, multiplied by sqrt(2 / layer_dims[l - 1])
    Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """   

    L = len(layer_dims)
    
    parameters = {}
    
    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        parameters["b"+str(l)] = np.zeros((layer_dims[l], 1))
        assert parameters["W"+str(l)].shape == (layer_dims[l], layer_dims[l - 1])
        assert parameters["b"+str(l)].shape == (layer_dims[l], 1)
    return parameters




def initialize_parameters_xavier(layer_dims):
    """
    Function:
        initialize parameters using random number from 0 to 0.99.. range, multiplied by sqrt(1 / layer_dims[l - 1])
    Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """   

    L = len(layer_dims)
    
    parameters = {}
    
    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(1 / layer_dims[l - 1])
        parameters["b"+str(l)] = np.zeros((layer_dims[l], 1))
        assert parameters["W"+str(l)].shape == (layer_dims[l], layer_dims[l - 1])
        assert parameters["b"+str(l)].shape == (layer_dims[l], 1)
    return parameters


def initialize_beta(parameters):
    """
    Function:
        Initializes the velocity/S as a python dictionary with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL" 
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
    Returns:
    v_s -- python dictionary containing the current velocity/S.
                    v_s['dW' + str(l)] = velocity/S of dWl
                    v_s['db' + str(l)] = velocity/S of dbl
    """
 
    
    L = len(parameters) // 2 
    v_s = {}
    
    for l in range(L):
        v_s["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l+1)])
        v_s["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l+1)])        
    return v_s

