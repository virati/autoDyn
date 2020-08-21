#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:21:34 2020

@author: virati
Barebones class for dynamical system
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pdb
import scipy.signal as sig

class dyn_sys:
    def __init__(self,dynamics,control,**kwargs):
        self.dt = 0.001
        self.N = kwargs['N']
        if 'H' in kwargs.keys(): self.H = kwargs['H'];
        else: H = np.ones_like
        
        if 'L' in kwargs.keys(): self.L = kwargs['L']
        if 'G' in kwargs.keys():
            self.G = kwargs['G']
            if 'L' in kwargs.keys(): print('You defined both G and L!! Dont do that')
            raise Warning
            self.L = nx.laplacian_matrix(self.G).todense()
        
        self.fdyn = dynamics
        self.gctr = control
        
    def integrator(self,exog=0):
        k1 = self.fdyn(self.state,ext_e = exog) * self.dt
        k2 = self.fdyn(self.state + .5*k1,ext_e = exog)*self.dt
        k3 = self.fdyn(self.state + .5*k2,ext_e = exog)*self.dt
        k4 = self.fdyn(self.state + k3,ext_e = exog)*self.dt
        
        new_state = self.state + (k1 + 2*k2 + 2*k3 + k4)/6
        new_state += np.random.normal(0,1,new_state.shape) * self.dt
        
        return new_state
        
    def run(self,x_i):
        end_time=10
        tvect = np.linspace(0,end_time,int(end_time/self.dt))
        self.state = x_i
        self.state_roster = []

        for tt,time in enumerate(tvect):
            print(time)
            self.state_roster.append(self.state)
            self.state = self.integrator()
        
        self.state_roster = np.array(self.state_roster).squeeze().T
        self.tvect = tvect
        
    def sim(self,x_i,element=0):
        self.run(x_i = x_i)
        self.measured_ts = self.measure(self.state_roster)
    
    
    def measure(self,x):
        H_matrix = np.ones_like(self.state)
        return np.sin(2 * np.pi * np.multiply(np.dot(H_matrix.T,x),self.tvect)).reshape(-1,1)
    
    
    ### PLOTTING FUNCTIONS
    def plot_states(self):
        plt.figure()
        plt.plot(self.state_roster.T)
    
    def plot_measured(self,element=0):
        plt.figure()
        plt.plot(self.measured_ts)
        
        T,F,SG = sig.spectrogram(self.measured_ts.T,fs = 1/self.dt,nfft=1024,nperseg=256,window='blackmanharris')
        plt.figure()
        #pdb.set_trace()
        plt.pcolormesh(F,T,np.log10(SG))
        
        
if __name__=='__main__':
    print('Test Run')
    node_N = 10
    network = nx.erdos_renyi_graph(node_N,0.8)
    #msys = dyn_sys(N=3,L = np.array([[1,2,-0.1],[2,-1,-0.1],[1,1,1]]))
    msys = dyn_sys(N=3,G = network)
    msys.sim(x_i = np.random.normal(0,1,size=(node_N,1)))
    msys.plot_states()
    msys.plot_measured()
