from autodyn.builder import microstruct, macrostruct

neural_mass = microstruct()
brain_network = macrostruct(L=brain_graph)

main_system = microstruct + macrostruct
