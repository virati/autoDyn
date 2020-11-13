#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:21:34 2020

@author: virati
Library of various dynamics functions
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pdb
import scipy.signal as sig

# WC should be N regions x D elements - D = 2 for excitatory and inhibitory
def wc_nmm(x,params):
    L = params['L']
    alpha = params['alpha']
    beta = params['beta']
    x_dot = np.zeros_like(x)
    for ii in x.shape[0]:
        x_dot[ii,0] = alpha[ii]*np.dot(L[ii,:],x[:,0]) - beta[ii]*x[ii,1]
        
    return x_dot

if __name__=='__main__':
    pass
