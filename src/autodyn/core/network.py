import networkx as nx
import matplotlib.pyplot as plt


class connectivity:
    def __init__(self, N, structure=None, **kwargs):
        self._topology = None
        self._arcs = None

        if len(self._arcs) != len(self._topology.edges):
            raise ValueError("Number of arcs must match number of edges in topology")

    @property
    def D(self):
        return nx.linalg.graphmatrix.incidence_matrix(self.G).todense().T

    @property
    def spectrum(self):
        # L = nx.laplacian_matrix(self.G)
        # e = np.linalg.eigvals(L.A)

        return nx.laplacian_spectrum(self.G)

    # %% PLOTTING METHODS

    def plot_incidence(self):
        plt.figure()
        plt.imshow(self.D.T)
        plt.title("Incidence")

    def plot_spectrum(self):
        plt.figure()
        plt.plot(self.spectrum)
        plt.title("Spectrum of Laplacian")

        plt.suptitle(f"Number of components: {nx.number_connected_components(self.G)}")
