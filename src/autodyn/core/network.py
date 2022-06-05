import networkx as nx
import matplotlib.pyplot as plt

class network:
    G = []

    def __init__(self, N=10, structure = None, **kwargs):
        if callable(structure):
            self.G = structure(N,**kwargs)
        else:
            self.G = nx.gnm_random_graph(N, kwargs['connection_density'])

    @property
    def L(self):
        return nx.linalg.laplacian_matrix(self.G).todense()

    @property
    def D(self):
        return nx.linalg.incidence_matrix(self.G).todense()

    def plot(self):
        plt.figure()
        nx.draw(self.G)

    @property
    def spectrum(self):
        #L = nx.laplacian_matrix(self.G)
        #e = np.linalg.eigvals(L.A)

        return nx.laplacian_spectrum(self.G)