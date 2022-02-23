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
import scipy.signal as sig
from .dynSys import dsys

def consensus(params,x):
    return -np.dot(params['L'],x)

def sindyn(params,x,u=0):
    return params['w'] - params['k']/len(x) *np.dot(params['D'],np.sin(np.dot(params['D'].T,x))) + u

def hopf2d(params,x):
    # x needs to be ~2d for this
    x_dot = 0
    y_dot = 0
    
    return np.array([x_dot,y_dot]).reshape(-1,1)

def sigm(x):
    return 1/(1+np.exp(-x))

#Oscillatory Dynamics - have to assume x \in \mathbb{C} for this to work since 2-d and all...
def oscillator(params,x):
    c = params['c'] #Coupling between real and imaginary
    #if x.shape[0] == 1 and x.dtype != np.complex_: raise Exception TypeError e
    
    return -np.dot(c,x)

def rand_squar(params,x):
    x_dot = np.zeros_like(x)
    L = params['L']
    c = params['c']
    
    for ii in range(x.shape[0]):
        for jj in range(x.shape[0]):
            x_dot[ii] = -(L[ii,jj] * x[ii] * (x[jj] - c[ii]))

    return x_dot

#%% Classes below
class W_C(dsys):
    
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
          'net_k':1/10}
    
    def __init__(self,**kwargs):
        super().__init__(d=2,**kwargs)
    
    def set_ctrl(self,u=[]):
        if u == []: self.u = np.zeros_like(self.tvect)
        elif u == 'sine': 
            self.u = 10*np.sin(2 * np.pi * 130 * self.tvect)
            halfway = int(round(self.u.shape[0]/2))
            self.u[:halfway] = 0
        else: self.u = u
        
        
    def fdyn(self,params,x,u=0):
        e = np.copy(x[:,0]) #region number is first, then element inside
        i = np.copy(x[:,1]) #region number is first, then element inside
        
        tau = params['tau']
        alpha = params['alpha']
        beta = params['beta']
        w = params['w']
        thresh = params['thresh']
        net_k = params['net_k']
        L = params['L']
        
        e_dot = np.zeros(shape=(x.shape[0],1))
        i_dot = np.zeros(shape=(x.shape[0],1))
        
        for nn in range(x.shape[0]):
            e_dot_p = params['T_e'] * (-e[nn] + sigm(-beta['e'] * (e[nn] * w['ee'] - i[nn] * w['ei'] - thresh['e'] + net_k*np.dot(L[nn,:],e))) + u)
            i_dot_p = params['T_i'] * (-i[nn] + sigm(-beta['i'] * (-i[nn] * w['ii'] + e[nn] * w['ie'] - thresh['i'])))
            
            e_dot[nn] = e_dot_p
            i_dot[nn] = i_dot_p
            
        return np.array([e_dot,i_dot]).squeeze().T

        
    '''What do we do after integrator? This would be where we reset phases, for example'''
    def post_integrator(self):
        pass
        self.state = (self.state + np.pi) % (2 * np.pi) - np.pi
    
class Ksys(dsys):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.fdyn = sindyn
        
    '''What do we do after integrator? This would be where we reset phases, for example'''
    def post_integrator(self):
        pass
        self.state = (self.state + np.pi) % (2 * np.pi) - np.pi


'''Weighed W-C Class'''
class W_C_w(W_C):
    def __init__(self,**kwargs):
        super.__init__(kwargs)
        
    def fdyn(self,params,x,u=0):
        e = np.copy(x[:,0]) #region number is first, then element inside
        i = np.copy(x[:,1]) #region number is first, then element inside
        
        tau = params['tau']
        alpha = params['alpha']
        beta = params['beta']
        w = params['w']
        thresh = params['thresh']
        net_k = params['net_k']
        D = params['D']
        K = params['K']
        
        e_dot = np.zeros(shape=(x.shape[0],1))
        i_dot = np.zeros(shape=(x.shape[0],1))
        
        for nn in range(x.shape[0]):
            e_dot_p = params['T_e'] * (-e[nn] + sigm(-beta['e'] * (e[nn] * w['ee'] - i[nn] * w['ei'] - thresh['e'] + net_k * np.dot(D[nn,:],np.dot(K,np.dot(D.T[:,nn],e))) + u)))
            i_dot_p = params['T_i'] * (-i[nn] + sigm(-beta['i'] * (-i[nn] * w['ii'] + e[nn] * w['ie'] - thresh['i'])))
            
            e_dot[nn] = e_dot_p
            i_dot[nn] = i_dot_p
            
        return np.array([e_dot,i_dot]).squeeze().T


''' The Delay-W-C class for DO Modeling'''

class W_C_d(dsys):
    
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
          'net_k':1/10}
    
    def __init__(self,**kwargs):
        super().__init__(d=2,**kwargs)
    
    def set_ctrl(self,u=[]):
        if u == []: self.u = np.zeros_like(self.tvect)
        elif u == 'sine': 
            self.u = 10*np.sin(2 * np.pi * 130 * self.tvect)
            halfway = int(round(self.u.shape[0]/2))
            self.u[:halfway] = 0
        else: self.u = u
        
        
    def fdyn(self,params,x,u=0):
        e = np.copy(x[:,0]) #region number is first, then element inside
        i = np.copy(x[:,1]) #region number is first, then element inside
        
        tau = params['tau']
        alpha = params['alpha']
        beta = params['beta']
        w = params['w']
        thresh = params['thresh']
        net_k = params['net_k']
        L = params['L']
        
        e_dot = np.zeros(shape=(x.shape[0],1))
        i_dot = np.zeros(shape=(x.shape[0],1))
        
        for nn in range(x.shape[0]):
            e_dot_p = params['T_e'] * (-e[nn] + sigm(-beta['e'] * (e[nn] * w['ee'] - i[nn] * w['ei'] - thresh['e'] + net_k*np.dot(L[nn,:],e))) + u)
            i_dot_p = params['T_i'] * (-i[nn] + sigm(-beta['i'] * (-i[nn] * w['ii'] + e[nn] * w['ie'] - thresh['i'])))
            
            e_dot[nn] = e_dot_p
            i_dot[nn] = i_dot_p
            
        return np.array([e_dot,i_dot]).squeeze().T

        
    '''What do we do after integrator? This would be where we reset phases, for example'''
    def post_integrator(self):
        pass
        self.state = (self.state + np.pi) % (2 * np.pi) - np.pi



class JR(dsys):
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
      'net_k':1/10}
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
    def fdyn(self,params,x,u=0):
        x_dot = np.zeros_like(x) #should give you R x D, where R is regions, D is internal dof
        
        alpha = params['alpha']
        
        x_dot[:,0] = x[3]
        x_dot[:,1] = x[4]
        x_dot[:,2] = x[5]
        x_dot[:,3] = A[0] * alpha * sigm['y1']['y2'] - 2.0 * alpha[0] * y[3] - alpha[0] ** 2 * x[0]
        x_dot[:,4] = A[0] * a[0] * (mu[0] + a_2[0] * J[0] * sigm['y0']['1'] + c[0] + src[0]) - 2.0 * a[0] * x[4] - a[0] ** 2 * x[1]
        x_dot[:,5] = B[0] * b[0] * (a_4[0] * J[0] * sigm['y0']['3']) - 2.0 * b[0] * x[5] - b[0] ** 2 * y[2]
        
        
        return x_dot
    
