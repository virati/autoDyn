#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 11:01:43 2020

@author: virati
Take a given NMM structure and make a laplacian out of it across N nodes
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pdb

#Single unit model
class unit:
    def __init__(self,**kwargs):
        if 'params' in kwargs.keys():
            self.params = kwargs['params']
        else:
            self.params = {'L':np.zeros((2,2))}
        if 'L' in self.params.keys():
            self.dim = self.params['L'].shape[0]
        else:
            self.dim = kwargs['dim']
        
        self.state = np.zeros((self.dim,1))
        self.dt = 0.001
        
        self.state_roster = []
    
    def dyn(self,state,params):
        return np.zeros_like(state)
    
    def integrator(self,exog=0):
        params = self.params
        k1 = self.dyn(self.state,params) * self.dt
        k2 = self.dyn(self.state + .5*k1,params)*self.dt
        k3 = self.dyn(self.state + .5*k2,params)*self.dt
        k4 = self.dyn(self.state + k3,params)*self.dt
        
        new_state = self.state + (k1 + 2*k2 + 2*k3 + k4)/6
        new_state += np.random.normal(0,1,new_state.shape) * self.dt
        
        return new_state
    
    def sim(self):
        pass

    def run(self,x0):
        self.state = x0
        for tt in np.linspace(0,10,1000):
            self.state = self.integrator()
            self.state_roster.append(self.state)
            
    def __call__(self,x0):
        
        return self.integrator(x0)
    
    def plot_roster(self):
        plt.figure()
        plt.plot(np.array(self.state_roster).squeeze())
class EI(unit):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
    def dyn(self,state,params):
        #pdb.set_trace()
        return np.dot(np.array([[1,-1.2],[0.5,1.0]]),state)

#test_params = {'L':[]}
#test_params['L'] = nx.laplacian_matrix(nx.erdos_renyi_graph(10,0.8)).todense()

test_unit = EI(dim=2)
for tt in np.linspace(0,100,10):
    test_unit.run(np.array([1.,0.]).reshape(-1,1))

test_unit.plot_roster()