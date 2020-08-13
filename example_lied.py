#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:57:37 2019

@author: virati
Example of a Lie Derivative
"""

#We want a function that does simple consensus dynamics

import numpy as onp
import jax.numpy as jnp
from jax import grad, jit, vmap, partial, jacrev
from jax import random as random
import jax

import networkx as nx

rng = jax.random.PRNGKey(111)

def consensus(L,x):
    return -jnp.dot(L,x)

def kuramoto(D,x):
    
    return -jnp.dot(D,jnp.sin(jnp.dot(D.T,x)))

G = nx.erdos_renyi_graph(10,0.4)
main_L = onp.array(nx.laplacian_matrix(G).todense())
main_L = jnp.asarray(main_L)

main_D = jnp.asarray(onp.array(nx.incidence_matrix(G).todense()))
#main_L = random.normal(rng,shape=(10,10))

container = jacrev(partial(kuramoto,main_D))

state_deriv = container(jnp.zeros((10,1))).squeeze()
print(state_deriv)
print(main_D)

x = jnp.zeros((10,1))
#%%
tvect = onp.linspace(0,10,1000)
for tidx,tt in enumerate(tvect):
    dx = container(x).squeeze()
    x += dx




