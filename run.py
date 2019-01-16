#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 05:27:25 2019

@author: diaa
"""

import argparse

from data import *
from util.forward import *
from util.backward import *
from util.cost import *
from util.initializer import *
from util.updater import *
from util.batches import batches_maker
from util.saver_loader import *
from util.prediction import *


def L_layer_model(X, Y, layers_dims, initializer=0, n_epochs = 3000, batch_size=32, shuffle=False, learning_rate = 0.0075, lambd=None, keep_prob=None,beta1=None, beta2=None, epsilon=None):
    """
    Function:
        layer neural network model which can be run in different optimizer modes and regularization methods.
    
    Arguments:
        X -- input data, of shape (2, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        layers_dims -- python list, containing the size of each layer
        n_epochs -- number of epochs
        batch_size -- the size of a mini batch
        shuffle -- bool value to check for shuffling data or not
        learning_rate -- the learning rate, scalar.
        lambd -- regularization hyperparameter, scalar
        keep_prob -- probability of keeping a neuron active during drop-out, scalar
        beta1 -- Exponential decay hyperparameter for the past gradients estimates 
        beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
        epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
        parameters -- python dictionary containing your updated parameters 
        costs -- list of costs during training
    """
    costs = []                        

    if initializer == 1:
        parameters = initialize_parameters_(layers_dims)
    elif initializer == 2:        
        parameters = initialize_parameters_he(layers_dims)
    elif initializer == 3:
        parameters = initialize_parameters_xavier(layers_dims)
    else:
        parameters = initialize_parameters(layers_dims)

    
    batches = batches_maker(X, Y, batch_size,shuffle=shuffle)
    v = None
    s = None
    if beta1:
        v = initialize_beta(parameters)
    if beta2:    
        s = initialize_beta(parameters)

    for e in range(1,n_epochs + 1):
    
        for x, y in batches:
            
            AL, caches,Ds = L_model_forward(x, parameters,keep_prob)
            
            cost = compute_cost(AL, y, parameters, lambd)
            
            grads = L_model_backward(AL, y, caches,lambd,keep_prob,Ds)
            
            parameters,v,s = update_parameters(parameters=parameters, grads=grads, learning_rate=learning_rate,v=v,beta1=beta1,s=s,beta2=beta2,t=e,epsilon=epsilon)
        if e % (n_epochs/10) == 0:
            print ("Cost after iteration {}: {}" .format(e, cost))
            costs.append(cost)
        
    return parameters, costs


def manager():
    
    
        
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--train', type=bool,default=False)
    parser.add_argument('--predict', type=bool,default=False)
    parser.add_argument('--layers', type=int, nargs='+')
    
    parser.add_argument('--initalizer', type=int)
    
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--shuffle', type=bool,default=False)
    
    
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--lambd', type=float)
    parser.add_argument('--keep_prob', type=float)
    parser.add_argument('--beta1', type=float)
    parser.add_argument('--beta2', type=float)
    parser.add_argument('--epsilon', type=float)
    
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--saving_path', type=str)
    parser.add_argument('--loading_path', type=str)

    parser.add_argument('--print_accuracy', type=bool, default=True)

    
    
    
    args = parser.parse_args()

    train = args.train
    predict = args.predict
    
    layers_dims = args.layers
    
    initalizer = args.initalizer
    
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    shuffle = args.shuffle
    
    learning_rate = args.learning_rate
    lambd = args.lambd
    keep_prob = args.keep_prob
    beta1 = args.beta1
    beta2 = args.beta2
    epsilon = args.epsilon
    
    
    if epsilon is None:
        epsilon = 1e-8
        
    data_path = args.data_path    
    saving_path = args.saving_path
    loading_path = args.loading_path
    
    print_accuracy = args.print_accuracy
    
    
    if train:
        X_train,Y_train,X_test,Y_test = train_data_loader(data_path)
        if batch_size == -1:
            batch_size = X_train.shape[1]
        parameters,costs = L_layer_model(X=X_train, Y=Y_train, layers_dims=layers_dims, initalizer=initalizer, n_epochs=n_epochs, batch_size=batch_size, shuffle=shuffle, learning_rate=learning_rate, lambd=lambd, keep_prob=keep_prob, beta1=beta1, beta2=beta2,epsilon=epsilon)
        
        if print_accuracy:  
            accuracy_precitor(X_test, Y_test, parameters)

        if saving_path:
            saver(parameters, saving_path)
            
    elif predict:
        parameters = loader(loading_path) 
        X = predict_data_loader(data_path)
        predict(X,parameters)

manager()