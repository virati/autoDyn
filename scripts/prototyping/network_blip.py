# %%
"""
Spiking Neural Network Simulation using Brian2
This simulation creates a network where neurons fire when their input
exceeds a threshold and immediately reset to zero.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from brian2 import (
    NeuronGroup,
    Synapses,
    SpikeMonitor,
    StateMonitor,
    SpikeGeneratorGroup,
    run,
    seed,
    prefs,
    mV,
    ms,
)


class SpikingNetworkSimulation:
    def __init__(
        self,
        num_neurons=20,
        connection_prob=0.3,
        threshold_voltage=-50.0,  # mV
        reset_voltage=-70.0,  # mV
        synaptic_weight=10.0,  # mV
        simulation_time=100,  # ms
        initial_spike_prob=0.2,
        random_seed=42,
    ):
        """
        Initialize the spiking neural network simulation using Brian2.

        Parameters:
        - num_neurons: Number of neurons in the network
        - connection_prob: Probability of connection between neurons
        - threshold_voltage: Firing threshold (mV)
        - reset_voltage: Reset voltage after spike (mV)
        - synaptic_weight: Weight of synaptic connections (mV)
        - simulation_time: Total simulation time (ms)
        - initial_spike_prob: Probability of initial spikes
        - random_seed: Random seed for reproducibility
        """
        self.num_neurons = num_neurons
        self.connection_prob = connection_prob
        self.threshold_voltage = threshold_voltage
        self.reset_voltage = reset_voltage
        self.synaptic_weight = synaptic_weight
        self.simulation_time = simulation_time
        self.random_seed = random_seed

        # Set random seed for reproducibility
        np.random.seed(random_seed)
        seed(random_seed)

        # Create the network
        self._build_network()
        self._create_connectivity()
        self._setup_monitoring()
        self._add_initial_stimulation(initial_spike_prob)

    def _build_network(self):
        """Build the neuron group with threshold dynamics."""
        # Define neuron model - simple threshold model
        neuron_eqs = """
        dv/dt = -v/(10*ms) : volt
        """

        # Create neuron group
        self.neurons = NeuronGroup(
            self.num_neurons,
            neuron_eqs,
            threshold="v > {}*mV".format(self.threshold_voltage),
            reset="v = {}*mV".format(self.reset_voltage),
            method="exact",
        )

        # Initialize membrane potentials
        self.neurons.v = self.reset_voltage * mV

    def _create_connectivity(self):
        """Create synaptic connections between neurons."""
        # Generate random connectivity using NetworkX
        G = nx.erdos_renyi_graph(
            self.num_neurons, self.connection_prob, directed=True, seed=self.random_seed
        )

        # Convert to Brian2 synapses
        sources = []
        targets = []
        for edge in G.edges():
            sources.append(edge[0])
            targets.append(edge[1])

        if sources:  # Only create synapses if there are connections
            self.synapses = Synapses(
                self.neurons, self.neurons, "w : volt", on_pre="v_post += w"
            )
            self.synapses.connect(i=sources, j=targets)
            self.synapses.w = self.synaptic_weight * mV
        else:
            self.synapses = None

        # Store network structure for visualization
        self.graph = G

    def _setup_monitoring(self):
        """Setup monitors to record network activity."""
        # Spike monitor
        self.spike_monitor = SpikeMonitor(self.neurons)

        # State monitor for membrane potentials
        self.state_monitor = StateMonitor(self.neurons, "v", record=True)

    def _add_initial_stimulation(self, spike_prob):
        """Add initial stimulation to some neurons."""
        # Select random neurons for initial stimulation
        num_initial = int(self.num_neurons * spike_prob)
        if num_initial > 0:
            initial_neurons = np.random.choice(
                self.num_neurons, size=num_initial, replace=False
            )

            # Create stimulus group
            self.stimulus = SpikeGeneratorGroup(
                num_initial, initial_neurons, [1] * num_initial * ms
            )

            # Connect stimulus to selected neurons
            self.stim_synapses = Synapses(
                self.stimulus, self.neurons, on_pre="v_post += 20*mV"
            )
            self.stim_synapses.connect(j="i")
        else:
            self.stimulus = None
            self.stim_synapses = None

    def run_simulation(self):
        """Run the spiking neural network simulation."""
        print(f"Running simulation for {self.simulation_time} ms...")
        print(
            f"Network: {self.num_neurons} neurons, "
            f"{len(self.graph.edges())} connections"
        )
        print(
            f"Threshold: {self.threshold_voltage} mV, "
            f"Synaptic weight: {self.synaptic_weight} mV"
        )

        # Run the simulation
        run(self.simulation_time * ms)

        print("Simulation completed!")
        print(f"Total spikes recorded: {len(self.spike_monitor.t)}")

    def visualize_network_structure(self):
        """Visualize the network connectivity."""
        plt.figure(figsize=(12, 8))

        # Create layout
        pos = nx.spring_layout(self.graph, seed=self.random_seed)

        # Draw network
        nx.draw_networkx_edges(
            self.graph, pos, alpha=0.6, arrows=True, arrowsize=20, edge_color="gray"
        )
        nx.draw_networkx_nodes(
            self.graph, pos, node_color="lightblue", node_size=500, alpha=0.8
        )
        nx.draw_networkx_labels(self.graph, pos, font_size=10)

        plt.title(
            f"Spiking Network Structure\n"
            f"{self.num_neurons} neurons, {len(self.graph.edges())} connections"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def visualize_spike_activity(self):
        """Visualize spike raster plot and network activity."""
        if len(self.spike_monitor.t) == 0:
            print(
                "No spikes recorded. Try lowering the threshold or increasing synaptic weights."
            )
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Spike raster plot
        axes[0].scatter(
            self.spike_monitor.t / ms, self.spike_monitor.i, s=2, alpha=0.7, color="red"
        )
        axes[0].set_xlabel("Time (ms)")
        axes[0].set_ylabel("Neuron ID")
        axes[0].set_title("Spike Raster Plot")
        axes[0].grid(True, alpha=0.3)

        # Population firing rate
        bin_size = 5  # ms
        bins = np.arange(0, self.simulation_time + bin_size, bin_size)
        spike_counts, _ = np.histogram(self.spike_monitor.t / ms, bins=bins)
        firing_rate = spike_counts / (bin_size / 1000) / self.num_neurons  # Hz

        axes[1].plot(bins[:-1], firing_rate, "b-", linewidth=2)
        axes[1].set_xlabel("Time (ms)")
        axes[1].set_ylabel("Population Firing Rate (Hz)")
        axes[1].set_title("Network Activity Over Time")
        axes[1].grid(True, alpha=0.3)

        # Sample membrane potentials
        sample_neurons = np.random.choice(
            self.num_neurons, size=min(5, self.num_neurons), replace=False
        )

        for neuron_id in sample_neurons:
            axes[2].plot(
                self.state_monitor.t / ms,
                self.state_monitor.v[neuron_id] / mV,
                label=f"Neuron {neuron_id}",
                alpha=0.7,
            )

        axes[2].axhline(
            y=self.threshold_voltage, color="red", linestyle="--", label="Threshold"
        )
        axes[2].set_xlabel("Time (ms)")
        axes[2].set_ylabel("Membrane Potential (mV)")
        axes[2].set_title("Sample Membrane Potentials")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_network_statistics(self):
        """Calculate and return network statistics."""
        total_spikes = len(self.spike_monitor.t)
        avg_firing_rate = (
            total_spikes / (self.simulation_time / 1000) / self.num_neurons
        )

        # Calculate per-neuron statistics
        unique_neurons, spike_counts = np.unique(
            self.spike_monitor.i, return_counts=True
        )
        neurons_that_fired = len(unique_neurons)

        # Network connectivity stats
        in_degrees = [self.graph.in_degree(n) for n in self.graph.nodes()]
        out_degrees = [self.graph.out_degree(n) for n in self.graph.nodes()]

        stats = {
            "total_neurons": self.num_neurons,
            "total_connections": len(self.graph.edges()),
            "network_density": nx.density(self.graph),
            "total_spikes": total_spikes,
            "neurons_that_fired": neurons_that_fired,
            "avg_firing_rate_hz": avg_firing_rate,
            "avg_in_degree": np.mean(in_degrees),
            "avg_out_degree": np.mean(out_degrees),
            "simulation_time_ms": self.simulation_time,
        }

        return stats

    def print_statistics(self):
        """Print comprehensive network statistics."""
        stats = self.get_network_statistics()

        print("\n=== Spiking Network Statistics ===")
        print("Network Structure:")
        print(f"  Total neurons: {stats['total_neurons']}")
        print(f"  Total connections: {stats['total_connections']}")
        print(f"  Network density: {stats['network_density']:.3f}")
        print(f"  Average in-degree: {stats['avg_in_degree']:.2f}")
        print(f"  Average out-degree: {stats['avg_out_degree']:.2f}")
        print("\nActivity:")
        print(f"  Simulation time: {stats['simulation_time_ms']} ms")
        print(f"  Total spikes: {stats['total_spikes']}")
        print(
            f"  Neurons that fired: {stats['neurons_that_fired']}/{stats['total_neurons']}"
        )
        print(f"  Average firing rate: {stats['avg_firing_rate_hz']:.2f} Hz")


def main():
    """Run a demonstration of the spiking neural network simulation."""
    print("=== Spiking Neural Network Simulation (Brian2) ===\n")

    # Create simulation with parameters that encourage propagation
    sim = SpikingNetworkSimulation(
        num_neurons=20,
        connection_prob=0.4,
        threshold_voltage=-55.0,  # Lower threshold for easier firing
        reset_voltage=-70.0,
        synaptic_weight=15.0,  # Higher weight for stronger connections
        simulation_time=150,  # Longer simulation
        initial_spike_prob=0.3,  # More initial spikes
        random_seed=42,
    )

    # Visualize network structure
    print("Visualizing network structure...")
    sim.visualize_network_structure()

    # Run simulation
    sim.run_simulation()

    # Print statistics
    sim.print_statistics()

    # Visualize results
    print("\nVisualizing spike activity...")
    sim.visualize_spike_activity()


if __name__ == "__main__":
    # Set Brian2 preferences for cleaner output
    prefs.codegen.target = "numpy"  # Use numpy backend for compatibility
    main()
