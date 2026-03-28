# %%
%load_ext autoreload
%autoreload 2
#%%
from autodyn.core import dynamical as dyn
from autodyn.models.canonical.standard import blip
from autodyn.core.network import connectivity

# %%

main_sys = dyn.dsys(blip, D=3)

main_sys.forward(T=50, dt=0.01, sigma=10, rho=28, beta=8 / 3)
main_sys.plot_phase(d1=0, d2=1)
