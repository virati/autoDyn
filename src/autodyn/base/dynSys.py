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
import scipy.signal as sig

def rk_integrator(fdyn,params,state,dt=0.001):    
    k1 = fdyn(params,state) * dt
    k2 = fdyn(params,state + .5*k1)*dt
    k3 = fdyn(params,state + .5*k2)*dt
    k4 = fdyn(params,state + k3)*dt
    
    state_change = (k1 + 2*k2 + 2*k3 + k4)/6
    return state_change  

'''A simple 'brain network' class that does basic things on top of networkx'''
class brain_net:
    G = []
    
    def __init__(self,N=10,connectivity=50):
        self.G = nx.gnm_random_graph(N,connectivity)
        
    def laplacian(self):
        return nx.linalg.laplacian_matrix(self.G).todense()
    
    def incidence(self):
        return nx.linalg.incidence_matrix(self.G).todense()
    
    def plot(self):
        plt.figure()
        nx.draw(self.G)

class dsys:
    state = []
    dt = 0.001
    params = {}
    tlen = 10
    state_raster = None
    
    def __init__(self,N=2,d=1,**kwargs):
        self.state = np.zeros(shape=(N,d))
        
        if 'params' in kwargs:
            self.params = kwargs['params']
        if 'tlen' in kwargs:
            self.tlen=kwargs['tlen']
        
        self.tvect = np.arange(0,self.tlen,self.dt)
        
    '''Params depend on the dynamics being implemented'''
    
    def set_params(self,params):
        self.params = params

    def set_behavior(self, mapping : callable):
        self.Gamma = mapping

    def apply_behavior(self):
        if self.state_raster is None:
            raise ValueError("This system has not been run and there is no output behavior...")
        
        return self.Gamma(self.state_raster)

    
    '''Base Runge-Kutta Integrator Method'''
    def integrator(self,u=0):
        
        k1 = self.fdyn(self.params,self.state,u) * self.dt
        k2 = self.fdyn(self.params,self.state + .5*k1,u)*self.dt
        k3 = self.fdyn(self.params,self.state + .5*k2,u)*self.dt
        k4 = self.fdyn(self.params,self.state + k3,u)*self.dt
        
        self.state += (k1 + 2*k2 + 2*k3 + k4)/6
        
        #self.state += np.random.normal(0,1,self.state.shape) * self.dt
        
        #return new_state

    '''initialize x'''
    def init_x(self,x):
        self.state = np.copy(x)
        
    
    '''What do we do after integrator? This would be where we reset phases, for example'''
    def post_integrator(self):
        pass
    #    self.state = (self.state + np.pi) % (2 * np.pi) - np.pi
    
    '''run the dynamics for an initial x for tlen time'''
    def run(self,**kwargs):
        if not 'zero_start' in kwargs: self.state = np.random.normal(0,1,size=self.state.shape)
        if 'tlen' in kwargs:
            self.tvect = np.arange(0,kwargs['tlen'],self.dt)
        if 'params' in kwargs:
            self.params = kwargs['params']
      
        if 'u' not in kwargs: self.u = np.zeros_like(self.tvect)
        elif kwargs['u'] == 'sine': self.u = 20*np.sin(2 * np.pi * 10 * self.tvect)
        elif kwargs['u'] == 'impulse': self.u = np.zeros_like(self.tvect);self.u[0] = 1
        else: self.u = kwargs['u']
      
        self.state_raster = []
        for tt,time in enumerate(self.tvect):
            self.state_raster.append(np.copy(self.state))
            self.integrator(self.u[tt])
            self.post_integrator()
            
        self.state_raster = np.array(self.state_raster).squeeze()
        

''' Class for behavior'''
class behavior:
    dim = 2
    def __init__(self,dsys,d=2):
        self.dim = d
        self.dsys = dsys
        self.params = {'d':self.dim}

    def gamma(self,params,states):
        d = params['d']
        return states[:,:,-1] #default behavior just returns first d brain states
    
    def get_behav(self):
        betas = np.zeros((self.dim,self.dsys.state_raster.shape[0]))
        for dd in range(self.dim):
            betas[dd,:] = self.gamma(self.params,self.dsys.state_raster)
            
        return betas
        

''' Class for measuring a dynamical system'''
class measurement:
    def __init__(self,sys):
        self.dyn_sys = sys
    
    def measure(self,x,func=[]):
        if func == []: H_fn = np.ones_like(self.dyn_sys.state_raster)
        else: H_fn = func
        
        return np.sin(2 * np.pi * np.multiply(np.dot(H_fn.T,x),self.tvect)).reshape(-1,1)
    
    ### PLOTTING FUNCTIONS
    def plot_measured(self,element=0):
        plt.figure()
        plt.plot(self.measured_ts)
    
    def SG_measured(self,element=0):
        T,F,SG = sig.spectrogram(self.measured_ts.T,fs = 1/self.dt,nfft=1024,nperseg=256,window='blackmanharris')
        plt.figure()
        plt.pcolormesh(F,T,np.log10(SG))
        
        
if __name__=='__main__':
    print('Unit Testing the Setup of a Generic Dynamical System')
    node_N = 10
    print('Unit Test Complete')
