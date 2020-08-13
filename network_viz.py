#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:30:32 2019

@author: virati
An example of 
"""

# needs mayavi2
# run with ipython -wthread
import networkx as nx
import numpy as np
from mayavi import mlab
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

from net_viz_lib import *

# some graphs to try
#H=nx.krackhardt_kite_graph()
#H=nx.Graph();H.add_edge('a','b');H.add_edge('a','c');H.add_edge('a','d')
#H=nx.grid_2d_graph(4,5)
#H=nx.cycle_graph(20)
H = nx.erdos_renyi_graph(20,0.3)

L = np.abs(nx.laplacian_matrix(H)).todense()

theta = np.random.multivariate_normal(12*np.ones(L.shape[0]),L,10000).T
#%%
#plt.plot(y)

t = np.linspace(0,10,10000)
y = np.zeros((16,10000))
for ii in range(16):
    y[ii,:] = np.sin(2 * np.pi * theta[ii,:] * t)

plt.plot(y[:,0:1000].T)

#%%
# Now do empiric variance
empir_var = np.zeros((16,16))
for ii in range(16):
    for jj in range(16):
        empir_var[ii,jj] = np.cov(np.vstack((y[ii,:],y[jj,:])))[0,1]


plt.figure()
plt.subplot(211)
plt.title('Estimated Laplacian through Covar')
plt.imshow(empir_var > 0.01)
plt.subplot(212)
plt.imshow(L)
plt.title('True Laplacian')

#%%
def plot_graph():
    G = nx.erdos_renyi_graph(7, 0.2)
    plt.figure()
    nx.draw(G)

plot_graph()
#%%

read = np.zeros((20,)).astype(np.bool)
write = np.zeros((20,)).astype(np.bool)
read[2] = True
write[5] = True
render_graph(H,read,write)