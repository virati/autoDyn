#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:21:34 2020

@author: virati
Library of various dynamics functions
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pdb
import scipy.signal as sig
from dynSys import dyn_sys

# WC should be N regions x D elements - D = 2 for excitatory and inhibitory
class WC_unit(dyn_sys):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
    def fdyn(self,x,params):
        wie = params['wie']
        wei = params['wei']
        wee = params['wee']
        wii = params['wii']
        
        ze = params['ze']
        zi = params['zi']
        
        x_dot = np.array([
            - 1/wei * (np.log(1/(x[1] - 1)) - zi + wii * x[1]),
            1/wie * (np.log(1/x[0] - 1) - ze + wee * x[0])
            ])
        
        return x_dot
    
class WC_net(dyn_sys):
    def __init__(self):
        super().__init__(self)

    def fdyn(self,x,params):
        L = params['L']
        alpha = params['alpha']
        beta = params['beta']
        
        

if __name__=='__main__':
    params = {'wie':1,
             'wee':1,
             'wii':1,
             'wei':1,
             'ze':1,
             'zi':1,
             'L':np.array([[1]])}
    test_net = WC_unit(params=params)
    x_init = np.array([1.0,1.0]).T
    test_net.sim(x_i = x_init,t_end=10)
    plt.figure();plt.plot(test_net.states().T)
