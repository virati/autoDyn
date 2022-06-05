#%%
from autodyn.core import dynamical as dyn
from autodyn.models.neuro.wc import wc_drift
from autodyn.core.network import connectivity

#%%
wilson_cowan = dyn.system(wc_drift)
param_set = 
wilson_cowan.simulate(T=100, dt=0.1, params=param_set)
wilson_cowan.plot_phase()

# %%
