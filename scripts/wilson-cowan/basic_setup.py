#%%
from autodyn.core import dynamical as dyn
from autodyn.models.canonical.standard import lorenz, consensus, single_hopf
from autodyn.core.network import connectivity

#%%
wilson_cowan = dyn.system(wc_vanilla)
