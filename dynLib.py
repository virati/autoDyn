#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:57:21 2019

@author: virati
Main library for autoDyn methods
This file will contain the primary JAX related methods that analyse variables of the (dyn_sys) class
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

def L_f_x(f,h,x):
    #return jacfwd(h)(x).squeeze()
    #print()
    #print(f(x))
    return npo.dot(jacfwd(h)(x).squeeze(),f(x))
    #return npo.dot(operable(vmap(jacfwd(h))),f)

def L_f(f,h):
    return npo.dot(operable(jacfwd(h)),operable(f))

def L_f_o(f,h,order=1):
    c = [h]
    for ii in range(order):
        c.append(npo.dot(operable(jacfwd(c[ii])),f))
    
    return c[-1]

def dotL_f(f,h,order=1):
    return npo.sum(L_d(h,f,order=order))

def brack_f_g(f,g):
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

@operable
def f2(x):
    return np.array([-x[0] + x[1],x[1],x[2] - x[0]])

def h(x):
    return 2*x[0] + 3*x[2]

if __name__ == '__main__':
    f_grad = jacfwd(f1)
    x0 = np.array([1.,1.,2.]).reshape(-1,1)
    print('F1, only\n',f_grad(x0).squeeze())
    #SUCCESS!
    
    f_all_grad = jacfwd(f1 + f2)
    print('F1+F2\n',f_all_grad(x0).squeeze())
    #SUCCESS!
    
    print('Lie deriv (actual)\n',L_f_x(f1,f2,x0))
    #SUCCESS!
    
    #lie
    L_f1_f2 = L_f_o(f1,f2,order=2)
    #print('Lie (jax)\n',np.sum(L_f1_f2(x0).squeeze(),axis=1).T) #if we are only doing 1st order Lie derivative
    
    print('Lie (jax)\n',np.sum(L_f1_f2(x0).squeeze(),axis=(1,2)).T) #WORKS!!!!!!!!
    
    