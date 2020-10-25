#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 22:40:11 2020

@author: virati
Behaviors 
"""

import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as npo
import jax.numpy as np
from jax import grad, jit, vmap, jvp
from numpy import ndenumerate
import matplotlib.pyplot as plt
from jax import jit, jacfwd, jacrev

def gamma(kappa,jeta,x):
    #kappa,jeta = params
    
    return np.array([np.tanh(np.dot(kappa[0,:],x) + jeta[0]),np.tanh(np.dot(kappa[1,:],x) + jeta[1])])

def cost(kappa,jeta,x):
    beta = gamma(kappa,jeta,x)
    return np.dot(np.array([1,-1]).reshape(-1,1).T,beta).squeeze()

N = 100
x = npo.random.normal(0,1,size=(N,1))
kappa = npo.random.normal(0,1,size=(2,N))
jeta = npo.random.normal(0,1,size=(2,))

test = npo.array(gamma(kappa,jeta,x))
d_gamma = jacfwd(gamma,argnums=0)
d_cost = grad(cost,argnums=0)

for _ in range(100):
    grads = d_cost(kappa,jeta,x)
    kappa -= 0.01 * grads

    
