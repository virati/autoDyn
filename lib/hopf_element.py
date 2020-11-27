#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 15:45:43 2016

A simple hopf system that encodes a value by "generating" a limit cycle -> r is the value being encoded
@author: virati
OBSOLETE!!!!!!!!!!!!!! - 11/20
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import pdb
import time

from scipy.integrate import odeint

from sklearn import preprocessing as pproc

plt.close('all')


input_val = []

class HopfNet():
    params = np.array([])
    flow = []
    mu = 1.0
    current_state = [1.5,0.0]
    fc = 0
    traj = {}
    rad = 0
    ctim = 0
    
    def __init__(self,center_freq=5,radius=1.5):
        self.params = np.array([0,0,0,0,0])
        self.fc = center_freq
        self.rad = radius
        #mu now needs to be a function of the desired/input radius
        self.mu = radius

    def plot_flow(self,plot_traj=False,state0=(2,2)):
        
        mesh_lim = 5
        xd = np.linspace(-mesh_lim,mesh_lim,20)
        yd = np.linspace(-mesh_lim,mesh_lim,20)
        X,Y = np.meshgrid(xd,yd)
        
        XX = np.array([X.ravel(),Y.ravel()])
        mu = self.mu
        
        Z = np.array(self.norm_form(XX,t=0))
        #Z = np.array(self.dyn(r,theta,0))
        #unit norm the Z vectors
        Z_n = pproc.normalize(Z.T,norm='l2').T
        #Z = Z.reshape(X.T.shape)
                
        plt.figure(figsize=(20,20))
        plt.subplot(211)
        plt.quiver(X,Y,Z_n[0,:],Z_n[1,:])
        
        plt.xlim((-5,5))
        plt.ylim((-5,5))
        plt.axis('tight')
        #overlay a trajectory
        if plot_traj:
            #state0 = self.current_state
            
            tvect,traj = self.trajectory(state0)
            plt.scatter(traj[:,0],traj[:,1])
            self.traj = {'X':traj,'T':tvect}
            
            plt.subplot(212)
            plt.plot(tvect,traj)
        #plt.show()
        
        #the timeseries of the trajectory
        
        #the trajectory just ran, so let's just set the last state as the current state
        #self.current_state = traj[-1,:]
        
        #I don't think I want this since it makes more sense for a "stateless syste" and the "last system" is just a snapshot
        #We want this to be a stateful system, with the ability to quickly look at the dynamics on demand
        self.flow = Z
        
    
    def dyn(self,r,theta,b):
        rd = b * r + a * r**3
        thetad = w + gamma * r ** 2
        
        return np.array([rd,thetad])

    def step_time(self):
        pass
        
    def tf_traj(self):
        #do TF analyses on trajectory
        tvect = self.traj['T']
        X = self.traj['X']
        
        plt.figure()
        plt.subplot(121)
        F,T,SG = sig.spectrogram(X[:,0],nperseg=512,noverlap=256,window=sig.get_window('blackmanharris',512),fs=100)
        plt.pcolormesh(T,F,10*np.log10(SG))  
        plt.subplot(122)
        F,T,SG = sig.spectrogram(X[:,1],nperseg=512,noverlap=256,window=sig.get_window('blackmanharris',512),fs=100)
        plt.pcolormesh(T,F,10*np.log10(SG))  
        
    def trajectory(self,state0):
        t = np.arange(0.0,30.0,0.01)
        
        traj = odeint(self.norm_form,state0,t)
        
        return t,traj
        
    def norm_form(self,state,t):
        x = state[0]
        y = state[1]
        
        mu = self.mu
        
        #these two can shape peakiness, be used for PAC?
        w = 0.5
        q = 1-w
        
        
        xd = w * (mu * x - y - x * (x**2 + y**2))
        yd = q * (x + mu * y - y * (x**2 + y**2))
    
        freq_fact = self.fc
        
        outv = freq_fact * np.array([xd,yd])
        
        return outv
        

def main():
    
    if 1:
        for mu in [2.0]:
            simpleNet = HopfNet(center_freq=140,radius=10)
            
            simpleNet.plot_flow()
            #traj = simpleNet.trajectory([12.0,13.0])

    plt.show()
    
main()
