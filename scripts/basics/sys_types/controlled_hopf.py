# %%
import numpy as np
from autodyn.core import dynamical as dyn
from autodyn.models.canonical.standard import controlled_hopf

# %%
# Design our stimulation waveform here
times = 50
dt = 0.01
stim = np.zeros((int(times // dt) + 1, 1))
stim[stim.shape[0] // 2 :: 10] = 40

# Setup and run our dynamics
hopf_single = dyn.dsys(controlled_hopf, D=2)
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
