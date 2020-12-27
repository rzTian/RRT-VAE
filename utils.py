#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import argparse
from torch.nn import init


"""
Functions for loading the 20NG dataset.

"""

def to_BOW(data, min_length):
    return np.bincount(data, minlength=min_length)

def load_data(path_tr, path_te, path_vocab):
    
    data_tr = np.load(path_tr, allow_pickle=True, encoding="latin1")
    data_te = np.load(path_te, allow_pickle=True, encoding="latin1")

    with open(path_vocab, 'rb') as data_file:  
        vocab = pickle.load(data_file)

    vocab_size=len(vocab)
    #--------------convert to bag of words representation------------------
    print ('Converting data to bag of words representation')
    data_tr = np.array([to_BOW(doc.astype('int'),vocab_size) for doc in data_tr if np.sum(doc)!=0])
    data_te = np.array([to_BOW(doc.astype('int'),vocab_size) for doc in data_te if np.sum(doc)!=0])
    #--------------print the data dimentions--------------------------
    print ('Data Loaded')
    print ('Dim Training Data',data_tr.shape)
    print ('Dim Test Data',data_te.shape)
    return data_tr, data_te, vocab


"""
Other functions

"""

#-----------initialize model parameter--------#
def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.2) 
 
        
#----------- Count model parameters------------#
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())


#----------- Compute running time--------------#

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs