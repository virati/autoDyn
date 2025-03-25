from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from autodyn.utils.functions import unity
from autodyn.core import control

from autodyn.core.integrators.runge_kutta import rk_integrator


class system:
    def __init__(self):
        pass

    def transfer_function(self, inputs, params):
        pass

    def forward(self, inputs, params):
        return self.transfer_function(inputs, params)


class dsys(system):
    def __init__(self, f, D: int = 3):
        self.x = np.zeros((D, 1))
        self.D = D
        self.f = f

        self.post_step = None

    def forward(self, T, dt=0.01, rasterize=True, **kwargs):
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
            ax = fig.add_subplot(projection="3d")
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
