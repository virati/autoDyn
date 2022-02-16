#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 21:38:27 2021

@author: virati
Liouville Equation
"""
import sys
sys.path.append('/home/virati/Dropbox/projects/Research/Control/autoDyn/lib/')
import numpy as np
from dynSys import dsys, brain_net

import numpy as np
import jax.numpy as jnp
from jax import grad

def quadp(x,c=2,d=4):
    print(x.shape)
    #assert np.isscalar(x)
    return (x-d)*(x+d)*(x-c)*(x+c)

dquadp = grad(quadp)


def LV_dyn(params,x,u):
    return np.array([quadp(x[0]),np.dot(dquadp(x[0]),quadp(x[0]))])


sys = dsys()
sys.fdyn = LV_dyn
sys.init_x(x=np.array([0.0,0.0]))
sys.run(tlen=50)

