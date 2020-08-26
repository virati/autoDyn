#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 00:30:17 2020

@author: virati
JAX-based optimization of ccontrol-trajectory
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

def cost(x,L):
    #total_cost = np.sum(L[x[0,:],x[1,:]])
    #calculate potential along a sequence of x1,x2
    total_cost = np.linalg.norm(potential(x[:,0],x[:,1]),axis=0)
    
    return total_cost

def potential(x1,x2,mu=0.0,sigma=0.3):
    return np.exp(-(np.sqrt(x1**2 + x2**2)-mu)**2 / (2.0 * sigma**2))


#%%
x,y = npo.meshgrid(np.linspace(-1,1,20),np.linspace(-1,1,20))
d = np.sqrt(x*x+y*y)
g = np.exp(-(d-mu)**2 / (2.0 * sigma**2))

#%%

grad_loss = jacfwd(cost,argnums=0)

tvect = np.linspace(0,2,1000)
x_init = np.array([tvect-1,1-tvect]).T

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(x,y,g,alpha=0.3)
traj = ax.plot(x_init[:,0],x_init[:,1],potential(x_init[:,0],x_init[:,1]),linewidth=20)

x_cand = npo.copy(x_init) + npo.random.normal(0,0.02,size=x_init.shape)
for _ in range(30):
    grads = grad_loss(x_cand,g)
    x_cand -= 0.1 * grads
traj_final = ax.plot(x_cand[:,0],x_cand[:,1],potential(x_cand[:,0],x_cand[:,1]),linewidth=20,color='r')

#%%
plt.scatter(x_init[:,0],x_init[:,1])
plt.scatter(x_cand[:,0],x_cand[:,1])