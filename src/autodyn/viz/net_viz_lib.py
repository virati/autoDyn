#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 19:53:34 2020

@author: virati
"""

import networkx as nx
from mayavi import mlab
import numpy as np
from ..base.dynSys import dsys

def plain_render_graph(H):
    # reorder nodes from 0,len(G)-1
    G=nx.convert_node_labels_to_integers(H)
    # 3d spring layout
    pos=nx.spring_layout(G,dim=3)
    # numpy array of x,y,z positions in sorted node order
    xyz=np.array([pos[v] for v in sorted(G)])
    # scalar colors
    scalars=np.array(G.nodes())+5
    
    mlab.figure(1, bgcolor=(0, 0, 0))
    mlab.clf()
    
    pts = mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2],
                        scalars,
                        scale_factor=0.1,
                        scale_mode='none',
                        colormap='Blues',
                        resolution=20)
    
    
    pts.mlab_source.dataset.lines = np.array(G.edges())
    tube = mlab.pipeline.tube(pts, tube_radius=0.01)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8),opacity=0.1)
    
def render_graph(H,read,write):
    # reorder nodes from 0,len(G)-1
    G=nx.convert_node_labels_to_integers(H)
    # 3d spring layout
    pos=nx.spring_layout(G,dim=3)
    # numpy array of x,y,z positions in sorted node order
    xyz=np.array([pos[v] for v in sorted(G)])
    # scalar colors
    scalars=np.array(G.nodes())+5
    
    mlab.figure(1, bgcolor=(0, 0, 0))
    mlab.clf()
    
    pts = mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2],
                        scalars,
                        scale_factor=0.1,
                        scale_mode='none',
                        colormap='Blues',
                        resolution=20)
    pts_reads = mlab.points3d(xyz[read,0], xyz[read,1], xyz[read,2],
                    scalars[read],
                    scale_factor=0.1,
                    scale_mode='none',
                    color=(0.0,1.0,0.0),
                    resolution=20)
    pts_reads = mlab.points3d(xyz[write,0], xyz[write,1], xyz[write,2],
                scalars[write],
                scale_factor=0.1,
                scale_mode='none',
                color=(1.0,0.0,0.0),
                resolution=20)
    
    
    pts.mlab_source.dataset.lines = np.array(G.edges())
    tube = mlab.pipeline.tube(pts, tube_radius=0.01)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8),opacity=0.1)
    
    mlab.savefig('mayavi2_spring.png')
    #mlab.show() # interactive window



def render_system(system : dsys):
    pass