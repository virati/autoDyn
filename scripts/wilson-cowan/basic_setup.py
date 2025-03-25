#%%
from autodyn.core import dynamical as dyn
from autodyn.models.neuro.wc import wc_drift, wc_input
from autodyn.core.network import connectivity
import numpy as np

#%%
wilson_cowan = dyn.system(wc_drift, D=2)
param_set = {
    "T_e": 5,
    "T_i": 5,
    "beta": {"e": -1, "i": -1},
    "w": {"ee": 10, "ii": 3, "ei": 12, "ie": 8},
    "alpha": 0.1,
    "thresh": {"e": 0.2, "i": 4},
    "tau": 0,
    "net_k": 1 / 10,
}
wilson_cowan.simulate(T=100, dt=0.1, params=param_set)
wilson_cowan.plot_phase()

#%%
T = 100
dt = 0.1
tpts = int(T // dt) + 1
tvect = np.linspace(0, T, tpts)

u = np.zeros_like(tvect)
u[tpts // 2 :: 2] = 1

wilson_cowan = dyn.system(wc_input, D=2)
wilson_cowan.simulate(T=T, dt=dt, params=param_set, stim=u)
wilson_cowan.plot_phase()
