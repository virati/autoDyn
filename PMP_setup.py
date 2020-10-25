#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 06:19:09 2020

@author: virati
Simple PMP buildup script
"""
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as npo
import jax.numpy as np
from jax import grad, jit, vmap, jvp
from numpy import ndenumerate
import matplotlib.pyplot as plt
from jax import jit, jacfwd, jacrev

def dyn(x,u):
    return 1 + u

def L(x,u):
    return 0.5 * u**2

def H(x,p,u):
    return -L(x,u) + p * dyn(x,u)

lhs = jacfwd(H,argnums=1)
rhs = jacfwd(H,argnums=0)

def wrap(u,x,p):
    return np.abs(H(x,p,u))

x = np.
u_candidate = np.zeros(x.shape)