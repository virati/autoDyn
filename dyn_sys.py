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
    def __init__(self,**kwargs):
        
        self.dt = 0.001
        self.N = kwargs['N']
        self.L = kwargs['L']
        self.G = nx.Graph()
        self.dyn_const = np.array([[5,10,50]]).T
        
        self.fdyn = self.f_drift
        
    def integrator(self,exog=0):
        k1 = self.fdyn(self.state,ext_e = exog) * self.dt
        k2 = self.fdyn(self.state + .5*k1,ext_e = exog)*self.dt
        k3 = self.fdyn(self.state + .5*k2,ext_e = exog)*self.dt
        k4 = self.fdyn(self.state + k3,ext_e = exog)*self.dt
        
        new_state = self.state + (k1 + 2*k2 + 2*k3 + k4)/6
        #new_state += np.random.normal(0,10,new_state.shape) * self.dt
        
        return new_state
    
    def f_drift(self,x,ext_e=0):
        return -0.1*(np.dot(self.L,x) + self.dyn_const)
    
    def gctrl(self,u):
        return u
    
    def run(self,x_i):
        end_time=10
        tvect = np.linspace(0,end_time,int(end_time/self.dt))
        self.state = x_i
        self.state_roster = []

        for tt,time in enumerate(tvect):
            print(time)
            self.state_roster.append(self.state)
            self.state = self.integrator()
        
        self.state_roster = np.array(self.state_roster).squeeze()
        self.tvect = tvect
    def measure(self,x):
        return np.sin(2 * np.pi * x * self.tvect.reshape(-1,1)) + np.random.normal(0,1,size=x.shape)
    
    def sim(self,x_i,element=0):
        self.run(x_i = x_i)
        self.measured_tc = self.measure(self.state_roster)
    
    def plot_measured(self,element=0):
        plt.figure()
        plt.plot(self.measured_tc[:,element].squeeze())
        T,F,SG = sig.spectrogram(self.measured_tc[:,element],fs = 1/self.dt,nfft=512,nperseg=512,noverlap=500,window='blackmanharris')
        plt.figure()
        plt.pcolormesh(F,T,np.log10(SG))
        
        
if __name__=='__main__':
    print('Test Run')
    
    #msys = dyn_sys(N=3,L = np.array([[1,2,-0.1],[2,-1,-0.1],[1,1,1]]))
    msys = dyn_sys(N=3,L = np.array([[1,1,-0.5],[1,1,-0.5],[1,1,1]]))
    msys.sim()
    msys.plot_measured(element=0)
