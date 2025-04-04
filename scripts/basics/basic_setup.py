#%%
from autodyn.core import dynamical as dyn
from autodyn.models.canonical.standard import (
    lorenz,
    consensus,
    single_hopf,
    controlled_hopf,
)
from autodyn.core.network import connectivity
import numpy as np

#%%
lorenz_sys = dyn.system(lorenz, D=3)
lorenz_sys.simulate(T=50, dt=0.01, sigma=10, rho=28, beta=8 / 3)
lorenz_sys.plot_phase()


#%%
test_sys = dyn.system(consensus, D=10)
brain_net = connectivity(10, proportion=0.4)

test_sys.simulate(T=100, dt=0.1, D=brain_net.D.T)

test_sys.plot_phase()

brain_net.plot_incidence()
brain_net.plot_spectrum()
#%%
# Design our stimulation waveform here


times = 50
dt = 0.01
stim = np.zeros((int(times // dt) + 1, 1))
stim[stim.shape[0] // 2 :: 10] = 40

# Setup and run our dynamics
hopf_single = dyn.system(controlled_hopf, D=2)
hopf_single.simulate(
    T=times,
    dt=dt,
    sigma=10,
    rho=28,
    beta=8 / 3,
    c=3,
    w=1,
    g=1,
    stim=stim,
    keep_positive=True,
)
hopf_single.plot_polar()

# %%
