# %%
import numpy as np
from autodyn.core import dynamical as dyn
from autodyn.models.neuro.scalar import kuramoto

# %%
kmo = dyn.system(kuramoto, D=3)
param_set = {"D": 1 * np.eye(3), "w": np.pi / 2}
kmo.simulate(T=100, dt=0.1, params=param_set)
kmo.plot_phase(0, 1)
kmo.plot_polar()
