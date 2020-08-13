#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:21:34 2020

@author: virati
Barebones class for dynamical system
"""


class ctrl_sys:
    def __init__(self):
        self.x = []
        self.f = []
        self.g = []
        
        self.u = []
        
        self.tstep = 0.001
        
    def run(self,init_x):
        time = np.linspace(0,10,self.tstep)