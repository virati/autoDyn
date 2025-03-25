from autodyn.core import dynamical as dyn
from autodyn.models.canonical.standard import consensus
from autodyn.core.network import connectivity

# %%
test_sys = dyn.system(consensus, D=10)
brain_net = connectivity(10, proportion=0.4)

test_sys.simulate(T=100, dt=0.1, D=brain_net.D.T)
test_sys.plot_phase(d1=0, d2=2)

brain_net.plot_incidence()
brain_net.plot_spectrum()
