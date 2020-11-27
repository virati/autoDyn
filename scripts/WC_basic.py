#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:38:15 2020

@author: virati
Wilson-Cowan implemented with dynSys
"""


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pdb
import scipy.signal as sig
import sys
sys.path.append('/home/virati/Dropbox/projects/Research/Control/autoDyn/lib/')

from dynSys import dsys
from dynLib import *


G = nx.gnm_random_graph(10, 50)
network_L = nx.linalg.laplacian_matrix(G).todense()

params = {'T_e': 5,
          'T_i': 5,
          'beta': {'e':-1,
                 'i':-1},
          'w':{'ee':10,
               'ii':3,
               'ei':12,
               'ie':8},
          'alpha':0.1,
          'thresh':{'e':0.2,
                    'i':4},
          'tau':0,
          'net_k':1/10,
          'L':network_L}

wc_net = W_C(N=10,params=params,tlen=20)
wc_net.init_x(x=np.random.uniform(0,1,size=(10,2)))
wc_net.set_ctrl(u='sine')
wc_net.run()

#%%
plt.figure()
plt.plot(wc_net.state_raster[:,2,:].squeeze())

#consensus plotting
evals, consensus_dim = np.linalg.eig(network_L)
plt.figure()
cdist = np.dot(np.array(consensus_dim[0,:]),wc_net.state_raster[:,:,0].squeeze().T).T
plt.plot(cdist)

#%%
#spectrogram of the consensus distance
import scipy.signal as sig

ds = 4
lfp = sig.decimate(np.copy(cdist.T),q=ds,zero_phase=True)
lfp += 0.01*np.random.normal(size=lfp.shape)
t,f,sg = sig.spectrogram(lfp,fs=1/(wc_net.dt*ds),nperseg=1024,noverlap=1020,nfft=1024)
plt.figure()
plt.pcolormesh(f,t,np.log10(np.abs(sg.squeeze())))