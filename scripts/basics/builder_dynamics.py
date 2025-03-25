from autodyn.builder import microstruct, macrostruct
from autodyn.models.canonical.standard import hopf
import networkx as nx

# %%
neural_mass = microstruct()
brain_network = macrostruct(L=connectome)

main_system = microstruct + macrostruct
