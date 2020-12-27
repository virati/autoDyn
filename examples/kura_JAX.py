#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 00:12:38 2020

@author: virati
SIMPLE Kuramoto with JAX to demonstrate K transition
SIMPLE = DON'T USE DYNLIB CLASSES
"""

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/Control/autoDyn/lib/')

from dynSys import rk_integrator

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx

def k_dyn(params,theta):
    K = params['K']
    w = params['w']
    
    theta_dot = np.zeros_like(theta)
    N = theta.shape[0]
    
    #will spell out here instead of vectorize
    for ii in range(N):
        theta_dot[ii] = w
        for jj in range(N):
            theta_dot[ii] += K/N * np.sin(theta[jj] - theta[ii])
            
    return theta_dot


def k_dyn_v(params,theta):
    K = params['K']
    w = params['w']
    D = params['D']
    
    theta_dot = np.zeros_like(theta)
    N = theta.shape[0]
    
    theta_dot = w + K/N * D * np.sin(D.T * theta)
    #theta_dot = w-K/N * (D * np.sin(D.T * theta))
        
    return theta_dot

tvect = np.linspace(0,10,1000)
N = 10
x_init = np.random.normal(size=(N,1))
state = np.copy(x_init)

#Assume all-to-all for simple kuramoto case
D = nx.linalg.incidence_matrix(nx.complete_graph(N)).todense()

params = {'K':15,
          'w':15,
          'D':D
          }
state_raster = []

for tt,time in enumerate(tvect):
    state += rk_integrator(k_dyn_v,params,state)
    state_raster.append(np.copy(state))
#%%
state_raster = np.array(state_raster).squeeze()
plt.figure()
plt.plot(np.sin(state_raster))


