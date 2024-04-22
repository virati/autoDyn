#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:46:03 2021

@author: virati
Animator class
"""

import numpy as np
import scipy.signal as sig
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/virati/Dropbox/projects/Research/Control/autoDyn/lib/')

from dynSys import dsys, rk_integrator
from dynLib import consensus

from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

def rkint(f,x0,tvec,params,dt):
    x_state = np.copy(x0)
    x_state_raster = []
    
    for tt in tvec:
        x_state_raster.append(np.copy(x_state))
        x_state += rk_integrator(f,params,x_state,dt=dt)
        
    return np.array(x_state_raster)


def rksim(**kwargs):
    for key,val in kwargs:
        print(key)

def oscillator(params,x):
    c = params['c']
    
    return -1*np.dot(c,-(x-5)*x)


#osc_drive = lambda x,t,c: -1*np.dot(c,-(x-5)*x)# + u

T = 1001
T_end=10
t = np.linspace(0,T_end,T)

x0 = np.array([1.,0.])

w=1
coupling = np.array([[0.1,w*2],[-2,0.5]])

params = {'c': coupling}
sol = rkint(oscillator,x0,t,params=params,dt=T_end/T)
#sol= odeint(osc_drive,x0,t,args=(coupling,))

plt.figure()
plt.plot(t,sol)
plt.show()


#%%
def init():
    line.set_data([],[])
    return line,

def animate(ii):
    line.set_data(t[:ii],sol[:ii])
    return line,

fig = plt.figure()
ax = plt.axes(xlim=(0,4), ylim=(-2,2))
line, = ax.plot([],[],lw=3)
anim = FuncAnimation(fig, animate, init_func=init,frames=200,interval=20,blit=True)

anim.save('test.gif',writer='imagemagick')