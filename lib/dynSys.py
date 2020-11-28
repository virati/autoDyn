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
from mayavi import mlab

'''A simple 'brain network' class that does basic things on top of networkx'''
class brain_net:
    G = []
    
    def __init__(self,N=10,connectivity=50):
        self.G = nx.gnm_random_graph(N,connectivity)
        
    def laplacian(self):
        return nx.linalg.laplacian_matrix(self.G).todense()
    
    def plot(self):
        plt.figure()
        nx.draw(self.G)
        
        
    def render_graph(self):
        # reorder nodes from 0,len(G)-1
        G=nx.convert_node_labels_to_integers(self.G)
        # 3d spring layout
        pos=nx.spring_layout(G,dim=3)
        # numpy array of x,y,z positions in sorted node order
        xyz=np.array([pos[v] for v in sorted(G)])
        # scalar colors
        scalars=np.array(G.nodes())+5
        
        mlab.figure(1, bgcolor=(0, 0, 0))
        mlab.clf()
        
        pts = mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2],
                            scalars,
                            scale_factor=0.1,
                            scale_mode='none',
                            colormap='Blues',
                            resolution=20)
        
        pts.mlab_source.dataset.lines = np.array(G.edges())
        tube = mlab.pipeline.tube(pts, tube_radius=0.01)
        mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8),opacity=0.1)
        
        #mlab.savefig('mayavi2_spring.png')
        #mlab.show() # interactive window
    

class dsys:
    state = []
    dt = 0.001
    params = {}
    tlen = 10
    
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
    
    def set_ctrl(self,u=[]):
        if u == []: self.u = np.zeros_like(self.tvect)
        elif u == 'sine': self.u = 20*np.sin(2 * np.pi * 10 * self.tvect)
        else: self.u = u
        
    '''run the dynamics for an initial x for tlen time'''
    def run(self):
        self.state_raster = []
        for tt,time in enumerate(self.tvect):
            self.state_raster.append(np.copy(self.state))
            self.integrator(self.u[tt])
            self.post_integrator()
            
        self.state_raster = np.array(self.state_raster).squeeze()
        
        
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
