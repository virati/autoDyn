#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 22:44:18 2020

@author: virati
This is a playground for g(x) design where we're trying to get Lgh to be = 0
"""

from lieLib import *
from dynLib import *

import jax.numpy as np
from jax import grad

def relu_full(param,x):
    w,b = params
    return np.maximum(0,w*x-b)


def relu(c,x):
    return np.maximum(0,x-c)

relu_grad = grad(relu)

def linear(params,x):
    w,b = params
    return w*x + b

def loss(params,dataset):
    x,y = dataset
    pred = linear(params,x)
    return np.square(pred - y).mean()

N = 100
G = nx.erdos_renyi_graph(N,0.3)
struct_L = nx.laplacian_matrix(G).todense()




#%%

'''
D is the design matrix 
'''
@operable
def g(x,D):
    pass

def cost():
    return L_f(g,h)