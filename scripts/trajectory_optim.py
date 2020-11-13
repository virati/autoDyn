#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 00:30:17 2020

@author: virati
JAX-based optimization of ccontrol-trajectory
This uses a Gaussian landscape model
"""

import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as npo
import jax.numpy as np
from jax import grad, jit, vmap, jvp
from numpy import ndenumerate
import matplotlib.pyplot as plt
from jax import jit, jacfwd, jacrev

from mpl_toolkits.mplot3d import Axes3D

mu, sigma = 0.0, 0.3

def get_traj_LIN(a):
    return np.array([a[0]*tvect**2 + tvect-1,a[1]*tvect**2 - tvect + 1]).T

def get_traj2(a):
    return np.array([(tvect+1)*(tvect-a[1])*(tvect-1)-1,a[0]*tvect-1]).T

def cost(a,L):
    x_traj = get_traj2(a)
    #total_cost = np.sum(L[x[0,:],x[1,:]])
    #calculate potential along a sequence of x1,x2
    total_cost = np.sum(potential(x_traj[:,0],x_traj[:,1]),axis=0)
    
    return total_cost

def potential(x1,x2):
    return -np.exp(-(np.sqrt((x1-0.5)**2 + (x2-0.5)**2)) / (2.0 * sigma**2)) + np.exp(-(np.sqrt((x1+0.5)**2 + (x2+0.5)**2)) / (2.0 * sigma**2))


#%%
x,y = npo.meshgrid(np.linspace(-1,1,30),np.linspace(-1,1,30))
#d = np.sqrt(x*x+y*y)
g = np.exp(-(np.sqrt((x-0.5)**2 + (y-0.5)**2)) / (2.0 * sigma**2)) + np.exp(-(np.sqrt((x+0.5)**2 + (y+0.5)**2)) / (2.0 * sigma**2))

#%%

grad_loss = jacfwd(cost,argnums=0)

tvect = np.linspace(0,1,1000)
#x_init = np.array([tvect-1,1-tvect]).T
x_init = get_traj2(np.array([0.,0.]))

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(x,y,g,alpha=0.3)
traj = ax.plot(x_init[:,0],x_init[:,1],potential(x_init[:,0],x_init[:,1]),linewidth=20,color='g')

x_cand = npo.copy(x_init) + npo.random.normal(0,0.02,size=x_init.shape)
a = np.array([0.,0.]).reshape(-1,1) + npo.random.normal(0,0.002,size=(2,1))
for _ in range(10):
    grads = grad_loss(a,g)
    a -= 0.01 * grads
    
x_cand = get_traj2(a)
traj_final = ax.plot(x_cand[:,0],x_cand[:,1],potential(x_cand[:,0],x_cand[:,1]),linewidth=10,color='r')
plt.xlim((-1,1))
plt.ylim((-1,1))
#%%
plt.figure()
plt.scatter(x_init[:,0],x_init[:,1])
plt.scatter(x_cand[:,0],x_cand[:,1])