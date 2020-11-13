#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 06:19:09 2020

@author: virati
Simple PMP buildup script
"""
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as npo
import jax.numpy as np
from jax import grad, jit, vmap, jvp
from numpy import ndenumerate
import matplotlib.pyplot as plt
from jax import jit, jacfwd, jacrev

print('PMP Buildup Script')