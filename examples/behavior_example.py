#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 00:32:10 2020

@author: virati
Behavior Example
"""


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pdb
import scipy.signal as sig
import sys
sys.path.append('/home/virati/Dropbox/projects/Research/Control/autoDyn/lib/')

from dynSys import dsys, brain_net, behavior
from dynLib import *

main_net = brain_net(10,50)
main_net.render_graph()

params = {'k':10,
          'w':15,
          'G':main_net,
          'D':main_net.incidence(),
          'L':main_net.laplacian()}

depr_net = Ksys(N=10,params=params,tlen=20)
depr_net.init_x(x=np.random.uniform(0,1,size=(10,2)))
depr_net.set_ctrl(u='sine')
depr_net.run()

depression = behavior(depr_net)
#%%
plt.plot(depression.get_behav())
