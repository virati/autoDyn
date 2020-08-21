#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 22:44:18 2020

@author: virati
This is a playground for g(x) design where we're trying to get Lgh to be = 0
"""

from lieLib import *
from dynLib import *


'''
D is the design matrix 
'''
@operable
def g(x,D):

def cost():
    return L_f(g,h)