 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:21:34 2020

@author: virati
Barebones class for dynamical systems
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pdb
import scipy.signal as sig

# Method to merge local and remote architectures

def consensus(x,params):
    L = params['L']
    return -np.dot(L,x)

def kuramoto(x,params):
    D = params['D']
    w = params['w']
    k = params['k']
    return w# - k * np.dot(D,np.cos(np.dot(D.T,x)))

def rand_squar(x,params):
    x_dot = np.zeros_like(x)
    L = params['L']
    c = params['c']
    
    for ii in range(x.shape[0]):
        for jj in range(x.shape[0]):
            x_dot[ii] = -(L[ii,jj] * x[ii] * (x[jj] - c[ii]))

    return x_dot

def zeros(x):
    return np.zeros_like(x)

# Basic class for dynamical system
class dyn_sys:
    dt = 0.001
    param_labels = ['L','C']
    
    def __init__(self,**kwargs):
        self.params = kwargs['params']
        #have to have at least a connectivity L
        self.L = self.params['L']
       
        # bring in main methods
        #self.fdyn = dynamics
        #self.gctr = control
        
    def set_dynamics(self,fn):
        self.fdyn = fn
        
    # Main Runge-Kutta Integrator
    def integrator_ctrl(self):
        k1 = (self.fdyn(self.state,self.params) + self.gctr(self.u)) * self.dt
        k2 = (self.fdyn(self.state + .5*k1,self.params)  + self.gctr(0.5*self.u))*self.dt
        k3 = (self.fdyn(self.state + .5*k2,self.params)  + self.gctr(0.5*self.u))*self.dt
        k4 = (self.fdyn(self.state + k3,self.params)  + self.gctr(self.u))*self.dt
        
        new_state = self.state + (k1 + 2*k2 + 2*k3 + k4)/6
        #new_state += np.random.normal(0,1,new_state.shape) * self.dt
        
        return new_state

    def integrator(self):
        k1 = (self.fdyn(self.state,self.params)) * self.dt
        k2 = (self.fdyn(self.state + .5*k1,self.params))*self.dt
        k3 = (self.fdyn(self.state + .5*k2,self.params))*self.dt
        k4 = (self.fdyn(self.state + k3,self.params))*self.dt
        
        new_state = self.state + (k1 + 2*k2 + 2*k3 + k4)/6
        #new_state += np.random.normal(0,1,new_state.shape) * self.dt
        
        return new_state
    
    # Method to run the simulationo along tvect
    def run(self,x_i):
        self.state = x_i
        state_roster = []

        for tt,time in enumerate(self.tvect):
            state_roster.append(self.state)
            self.state = self.integrator()
        
        self.state_roster = np.hstack(state_roster)
    
    # Method that does setup for the run, then runs, then does post-run organizing
    def sim(self,x_i,t_end,element=0):
        self.tvect = np.linspace(0,t_end,int(t_end/self.dt))
        self.u = np.zeros_like(self.tvect)
        self.run(x_i = x_i)
    
    # Return the full state trajectory directly
    def states(self):
        return self.state_roster

class measurement:
    def __init__(self,sys):
        self.dyn_sys = sys
    
    def measure(self,x):
        H_matrix = np.ones_like(self.state)
        
        return np.sin(2 * np.pi * np.multiply(np.dot(H_matrix.T,x),self.tvect)).reshape(-1,1)
    
    ### PLOTTING FUNCTIONS
    def plot_measured(self,element=0):
        plt.figure()
        plt.plot(self.measured_ts)
    
    def SG_measured(self,element=0):
        T,F,SG = sig.spectrogram(self.measured_ts.T,fs = 1/self.dt,nfft=1024,nperseg=256,window='blackmanharris')
        plt.figure()
        plt.pcolormesh(F,T,np.log10(SG))
        
        
if __name__=='__main__':
    print('Test Run')
    node_N = 10
    network = nx.erdos_renyi_graph(node_N,0.5)
    L_net = nx.linalg.laplacianmatrix.laplacian_matrix(network).todense()
    #msys = dyn_sys(N=3,L = np.array([[1,2,-0.1],[2,-1,-0.1],[1,1,1]]))
    
    params = {'L': L_net,'c': np.random.uniform(0.1,5,size=(10,1)),'alpha': np.random.uniform(0.1,5,size=(10,1)),
              'D':nx.linalg.graphmatrix.incidence_matrix(network).todense(),
              'w':10,
              'k':10}
    
    nx.draw(network)
    msys = dyn_sys(params=params)
    msys.set_dynamics(fn=kuramoto)
    x_init = np.random.uniform(-np.pi/2,np.pi/2,size=(node_N,1))
    msys.sim(x_i = x_init,t_end=10)
    plt.figure();plt.plot(np.sin(msys.states().T))
