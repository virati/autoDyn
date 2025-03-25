from autodyn.builder import microstruct, macrostruct
import networkx as nx

# %%
neural_mass = microstruct()
brain_network = macrostruct(L=brain_graph)

main_system = microstruct + macrostruct
