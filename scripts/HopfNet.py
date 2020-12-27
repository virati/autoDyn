#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 03:50:39 2020

@author: virati
Hopf network
"""
import sys
sys.path.append('/home/virati/Dropbox/projects/Research/Control/autoDyn/lib/')
import numpy as np
from dynSys import dsys, brain_net
import matplotlib.pyplot as plt
import pdb
import scipy.signal as sig

import jax.numpy as jnp

plt.close('all')
#%%
def sigmoid(x):
    return np.tanh(x)

class HNet(dsys):
    def __init__(self,**kwargs):
        super().__init__(N=10,**kwargs)
        
    def fdyn(self,params,x,u):
        r = x[:,0].reshape(-1,1)
        theta = x[:,1].reshape(-1,1)
        c = params['c']
        d = params['d']
        w = params['w']
        L = params['L']
        K = params['K']
        D = params['D']
            
        
        try: r_dot = 0.1*(-r**2 * (r-c))# - 0.01*np.dot(L,r))# + np.random.normal(size=r.shape)
        except: pdb.set_trace()
        
        theta_dot = w + d/r - K * np.dot(D,np.sin(np.dot(D.T,theta)))
        
        return np.array([r_dot,theta_dot]).squeeze().T

def readout(raster):
    #assume raster is an T x N x D - Time x Nodes x internal dims
    #first, we want an BxN
    gamma = np.zeros((2,raster.shape[1]))
    gamma[0,[0,2,4,5,6]] = 1
    gamma[1,[0,1,3,7]] = 1
    
    states = raster.swapaxes(0,1)
    out_R = np.dot(gamma,states[:,:,0]) #output is B X T
    out_T = np.dot(gamma,states[:,:,1])
    
    return np.array([out_R,out_T]).T
    

def cost(behavs):
    return np.sum(behavs,axis=1)[:,0]


N = 10
main_net = brain_net(N,50)

param_set = {'c':5,
             'd':10,
             'w':4+np.random.normal(0,1.0,size=(N,1)),
             'L':main_net.laplacian(),
             'D':main_net.incidence(),
             'K':0.3}

test_net = HNet(params=param_set)
test_net.init_x(x=np.random.uniform(0,1,size=(N,2)))
test_net.run(tlen=20,u='sine')
#%%
plt.figure()
plt.plot(test_net.state_raster[:,:,0].squeeze())
plt.plot(test_net.state_raster[:,:,1].squeeze())
#%%
plt.figure()
plt.plot(test_net.state_raster[:,:,0].squeeze() * np.exp(-1j * test_net.state_raster[:,:,1].squeeze()))
plt.xlabel('time')

#%%
# Spectrogram
timeseries = test_net.state_raster[:,:,0].squeeze() * np.exp(-1j * test_net.state_raster[:,:,1].squeeze())
timeseries += np.random.normal(size=timeseries.shape)
dsf = 5
timeseries = np.sum(sig.decimate(timeseries.T,q=dsf),axis=0)

plt.figure()
plt.plot(timeseries)
cc = 2
#F,T,SG = sig.spectrogram(timeseries,fs=1/(dsf*0.001))
#plt.figure()
#plt.pcolormesh(T,F,np.log10(np.abs(SG)))

#%%
plt.figure()
plt.polar(test_net.state_raster[:,:,1].squeeze(),test_net.state_raster[:,:,0].squeeze(),linewidth=10)

#%%
plt.figure()
plt.plot(test_net.state_raster[:,:,1])
#%%
plt.figure()
behavs = readout(test_net.state_raster)
for b_idx in [0,1]:
    plt.plot(behavs[:,b_idx,0] * np.exp(-1j * behavs[:,b_idx,1]))
    plt.title(b_idx)
    