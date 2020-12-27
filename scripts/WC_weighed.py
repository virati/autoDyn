#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 23:27:20 2020

@author: virati
Weighed Wilson-Cowan, want to extract dynamics with PySindy
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pdb
import scipy.signal as sig
import sys
sys.path.append('/home/virati/Dropbox/projects/Research/Control/autoDyn/lib/')

from dynSys import dsys, brain_net
from dynLib import *



if __name__=='__main__':
    main_net = brain_net(10,50)
    #main_net.render_graph()
    #setup our network first
    #G = nx.gnm_random_graph(10, 50)
    #network_L = nx.linalg.laplacian_matrix(G).todense()
    incidenceM = main_net.incidence()
    K_dsgn = np.zeros((incidenceM.shape[1],incidenceM.shape[1]))
    for nn,stren in enumerate():
        K_dsgn[]
    
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
              'G':main_net,
              'D':incidenceM,
              'K':K_dsgn}
#%%
    wc_net = W_C(N=10,params=params,tlen=20)
    wc_net.init_x(x=np.random.uniform(0,1,size=(10,2)))
    wc_net.set_ctrl(u='sine')
    wc_net.run()