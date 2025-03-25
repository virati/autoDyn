#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:21:34 2020

@author: virati
Barebones class for dynamical systems
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from autodyn.utils.functions import unity
from autodyn.core import control

from autodyn.core.integrators.runge_kutta import rk_integrator


class system:
    def __init__(self, f, D: int = 3, net_graph: nx.Graph = None):
        self.x = np.zeros((D, 1))
        self.D = D
        self.f = f
        self.u_func: Optional[callable] = None

        self.gen_connectivity()
        self.post_step = unity

    @property
    def u(self):
        self._u: control
        if self.u_func is None:
            return np.zeros_like(self.x)

        if not self._u:
            self._u = control(self.u_func)
        return self._u

    @u.setter
    def u(self, value: np.ndarray):
        self._u = control()
        self._u.raw = value

    def set_post_step(self, func: callable):
        self.post_step = func

    def gen_connectivity(self):
        nodes = [1, 2, 3, 4, 5, 6]
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(
            [
                (1, 2, 1),
                (2, 3, 1),
                (1, 6, 1),
                (1, 3, 1),
                (3, 4, 1),
                (4, 5, 1),
                (4, 6, 1),
                (5, 6, 1),
            ]
        )

        # L = nx.to_numpy_matrix(G)
        L = nx.linalg.laplacianmatrix.laplacian_matrix(G).todense()
        self.G = G
        self.L = L

    def simulate(self, T, dt=0.01, rasterize=True, **kwargs):
        tvect = np.arange(0, T, dt)
        controlled = False

        x_state = np.random.normal(0, 1, (self.D, 1))
        if "keep_positive" in kwargs.keys():
            x_state = np.abs(x_state)

        x_raster = []

        if "stim" in kwargs.keys():
            controlled = True

        for tidx, time in enumerate(tvect):
            if controlled:
                x_new = rk_integrator(
                    self.f, x_state, dt=0.01, u=kwargs["stim"][tidx], **kwargs
                )
            else:
                x_new = rk_integrator(self.f, x_state, dt=0.01, **kwargs)

            x_state = self.post_step(x_new)

            if rasterize:
                x_raster.append(x_state)

        if x_raster:
            self.raster = np.array(x_raster).squeeze()

    def plot_raster(self):
        plt.plot(self.raster)
        plt.show()

    def plot_phase(self, d1=0, d2=1):
        fig = plt.figure()
        plt.plot(self.raster[:, d1], self.raster[:, d2])
        plt.title("Phase")

    def plot_phase_full(self):
        if self.D == 3:
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            ax.plot(self.raster[:, 0], self.raster[:, 1], self.raster[:, 2])
            plt.draw()
            plt.title("Phase Portrait")
        else:
            fig = plt.figure()
            plt.plot(self.raster)
            plt.title("Trajectories in Time")

    def plot_measure(self):
        plt.figure()
        plt.plot(self.H(self.raster))
        plt.title("Measured Trajectories in Time")

    def plot_polar(self):
        plt.figure()
        plt.plot(np.real(self.raster[:, 0] * np.exp(1j * self.raster[:, 1])))
        plt.title("Polar Trajectories in Time")
