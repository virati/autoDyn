import networkx as nx
import matplotlib.pyplot as plt


class connectivity:
    def __init__(self, N, structure=None, **kwargs):
        if callable(structure):
            self.G = structure(N, kwargs["proportion"])
        else:
            self.G = nx.generators.random_graphs.erdos_renyi_graph(
                N, kwargs["proportion"]
            )

    @property
    def D(self):
        return nx.linalg.graphmatrix.incidence_matrix(self.G).todense().T

    @property
    def spectrum(self):
        # L = nx.laplacian_matrix(self.G)
        # e = np.linalg.eigvals(L.A)

        return nx.laplacian_spectrum(self.G)

    #%% PLOTTING METHODS

    def plot_incidence(self):

        plt.figure()
        plt.imshow(self.D.T)
        plt.title("Incidence")

    def plot_spectrum(self):
        plt.figure()
        plt.plot(self.spectrum)
        plt.title("Spectrum of Laplacian")

        plt.suptitle(f"Number of components: {nx.number_connected_components(self.G)}")
