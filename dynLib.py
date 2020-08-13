#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:57:21 2019

@author: virati
Main library for autoDyn methods
"""

import numpy as npo
import jax.numpy as np
from jax import grad, jit, vmap, jvp
from numpy import ndenumerate
import matplotlib.pyplot as plt
from jax import jit, jacfwd, jacrev

import operator
import networkx as nx

# Generate a class that wraps functions
class operable:
    def __init__(self, f):
        self.f = f
    def __call__(self, x):
        return self.f(x)

# Convert oeprators to the corresponding function
def op_to_function_op(op):
    def function_op(self, operand):
        def f(x):
            return op(self(x), operand(x))
        return operable(f)
    return function_op
 
for name, op in [(name, getattr(operator, name)) for name in dir(operator) if "__" in name]:
    try:
        op(1,2)
    except TypeError:
        pass
    else:
        setattr(operable, name, op_to_function_op(op))

#%% Lie Derivatives Block
        
def L_d(h,f,order=1):
    c = [h]
    for ii in range(order):
        c.append(np.dot(operable(vmap(grad(c[ii]))),f))
    
    return c[-1]

def L_dot(h,f,order=1):
    return np.sum(L_d(h,f,order=order))

def L_bracket(f,g):
    c = operable(jcb(f)) * g
    cinv = operable(jcb(g)) * f
    
    return cinv
    #print(c(np.array([1.,1.,1.])))
    #print(cinv(np.array([1.,1.,1.])))

#%% Drift function libraries
@operable
def f1(x):
    #return np.array([-x[1],-x[0],-x[2] + x[1]])
    return np.array([-x[1]**2 + x[2],-x[0]**3,-x[2]**2 + x[1]])

def f2(x):
    return np.array([-x[0] + x[1],x[1],x[2] - x[0]])

def h(x):
    return 2*x[0] + 3*x[2]

if __name__ == '__main__':
    tanh_grad = grad(np.tanh)
    print(tanh_grad(0.0))
    