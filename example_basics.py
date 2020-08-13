#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 14:02:01 2019

@author: virati
Example script for Basics of Jax
"""

def simple_drift(x):
    x_dot = np.array([x[0] * x[2],-x[1],x[2]**2]).T
    return x_dot

def drift(G,inputs,params):
    D = nx.incidence_matrix(G)
    
    for W, b in params:
        outputs = np.dot(inputs,D.T) + b
        inputs = np.tanh(outputs)
    return outputs
        
def drift_function(G,x,params,f):
    #G is a networkx graph
    D = nx.incidence_matrix(G)
    
    x_dot = -D * f(np.dot(D.T,x))
    for W,b in params:
        outputs = np.dot(inputs,W) + b
    
grad_f = jacfwd(simple_drift)

#%%
def predict(params, inputs):
  for W, b in params:
    outputs = np.dot(inputs, W) + b
    inputs = np.tanh(outputs)
  return outputs

def logprob_fun(params, inputs, targets):
  preds = predict(params, inputs)
  return np.sum((preds - targets)**2)

#grad_fun = jit(grad(logprob_fun))  # compiled gradient evaluation function
#perex_grads = jit(vmap(grad_fun, in_axes=(None, 0, 0)))  # fast per-example grads

