# %%
from autodyn.core import dynamical as dyn
from autodyn.models.canonical.standard import lorenz
from autodyn.core.network import connectivity
import numpy as np

# %%
lorenz_sys = dyn.dsys(lorenz, D=3)
lorenz_sys.simulate(T=50, dt=0.01, sigma=10, rho=28, beta=8 / 3)
lorenz_sys.plot_phase(d1=0, d2=1)
