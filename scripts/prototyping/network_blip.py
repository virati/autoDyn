# %%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


class NetworkBlipSimulation:
    def __init__(
        self,
        num_nodes=20,
        connection_prob=0.3,
        threshold=2.0,
        initial_activation_prob=0.1,
        random_seed=42,
    ):
        """
        Initialize the network blip simulation.

        Parameters:
        - num_nodes: Number of nodes in the network
        - connection_prob: Probability of connection between any two nodes
        - threshold: Global threshold for node firing
        - initial_activation_prob: Probability of initial node activation
        - random_seed: Random seed for reproducibility
        """
        self.num_nodes = num_nodes
        self.threshold = threshold
        self.random_seed = random_seed

        # Set random seed
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Create directed graph
        self.graph = nx.erdos_renyi_graph(
            num_nodes, connection_prob, directed=True, seed=random_seed
        )

        # Initialize node states (0 = inactive, 1 = active/fired)
        self.node_states = np.zeros(num_nodes)

        # Randomly activate some nodes initially
        initial_active = np.random.choice(
            num_nodes, size=int(num_nodes * initial_activation_prob), replace=False
        )
        self.node_states[initial_active] = 1.0

        # Store history for visualization
        self.state_history = [self.node_states.copy()]
        self.firing_history = []

    def get_upstream_sum(self, node):
        """Calculate the sum of values from upstream neighbors (predecessors)."""
        predecessors = list(self.graph.predecessors(node))
        return sum(self.node_states[pred] for pred in predecessors)

    def step(self):
        """Perform one simulation step."""
        new_states = np.zeros(self.num_nodes)
        fired_nodes = []

        # Check each node to see if it should fire
        for node in range(self.num_nodes):
            upstream_sum = self.get_upstream_sum(node)

            # Node fires if upstream sum exceeds threshold
            if upstream_sum >= self.threshold:
                new_states[node] = 1.0
                fired_nodes.append(node)
            # Otherwise, node remains at 0 (or goes back to 0 if it was firing)
            else:
                new_states[node] = 0.0

        # Update states
        self.node_states = new_states

        # Store history
        self.state_history.append(self.node_states.copy())
        self.firing_history.append(fired_nodes.copy())

        return fired_nodes

    def simulate(self, num_steps=50):
        """Run the simulation for a specified number of steps."""
        print(
            f"Starting simulation with {self.num_nodes} nodes, threshold={self.threshold}"
        )
        print(f"Initial active nodes: {np.where(self.node_states > 0)[0].tolist()}")

        for step_num in range(num_steps):
            fired_nodes = self.step()
            active_nodes = np.where(self.node_states > 0)[0].tolist()

            if (
                step_num < 10 or step_num % 10 == 0
            ):  # Print first 10 steps, then every 10th
                print(
                    f"Step {step_num + 1}: Fired nodes: {fired_nodes}, Active nodes: {active_nodes}"
                )

        print(f"Simulation completed after {num_steps} steps")

    def visualize_network(self):
        """Visualize the network structure."""
        plt.figure(figsize=(12, 8))

        # Create layout
        pos = nx.spring_layout(self.graph, seed=self.random_seed)

        # Draw network
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, arrows=True, arrowsize=20)

        # Color nodes based on current state
        node_colors = [
            "red" if state > 0 else "lightblue" for state in self.node_states
        ]
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=500)

        # Add labels
        nx.draw_networkx_labels(self.graph, pos)

        plt.title(
            f"Network Structure (Red = Active, Blue = Inactive)\nThreshold = {self.threshold}"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def visualize_dynamics(self):
        """Visualize the dynamics over time."""
        if len(self.state_history) < 2:
            print("No simulation data to visualize. Run simulate() first.")
            return

        # Create activity matrix (time x nodes)
        activity_matrix = np.array(self.state_history)

        plt.figure(figsize=(12, 8))

        # Plot 1: Activity heatmap
        plt.subplot(2, 1, 1)
        plt.imshow(
            activity_matrix.T, aspect="auto", cmap="RdYlBu_r", interpolation="nearest"
        )
        plt.colorbar(label="Node State")
        plt.ylabel("Node ID")
        plt.title("Network Activity Over Time")

        # Plot 2: Total activity over time
        plt.subplot(2, 1, 2)
        total_activity = np.sum(activity_matrix, axis=1)
        plt.plot(total_activity, "b-", linewidth=2)
        plt.xlabel("Time Step")
        plt.ylabel("Total Active Nodes")
        plt.title("Total Network Activity")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_network_stats(self):
        """Get basic statistics about the network."""
        stats = {
            "num_nodes": self.num_nodes,
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "threshold": self.threshold,
            "current_active_nodes": int(np.sum(self.node_states)),
            "avg_in_degree": np.mean([d for n, d in self.graph.in_degree()]),
            "avg_out_degree": np.mean([d for n, d in self.graph.out_degree()]),
        }

        if len(self.state_history) > 1:
            activity_matrix = np.array(self.state_history)
            stats["avg_activity"] = np.mean(np.sum(activity_matrix, axis=1))
            stats["max_activity"] = np.max(np.sum(activity_matrix, axis=1))

        return stats

    def print_stats(self):
        """Print network statistics."""
        stats = self.get_network_stats()
        print("\n=== Network Statistics ===")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")


# %%


def main():
    """Run a demonstration of the network blip simulation."""
    print("=== Network Blip Simulation Demo ===\n")

    # Create simulation
    sim = NetworkBlipSimulation(
        num_nodes=15,
        connection_prob=0.4,
        threshold=2.2,
        initial_activation_prob=0.2,
        random_seed=42,
    )

    # Print initial stats
    sim.print_stats()

    # Visualize initial network
    print("\nVisualizing initial network structure...")
    sim.visualize_network()

    # Run simulation
    print("\nRunning simulation...")
    sim.simulate(num_steps=30)

    # Print final stats
    sim.print_stats()

    # Visualize dynamics
    print("\nVisualizing dynamics...")
    sim.visualize_dynamics()


if __name__ == "__main__":
    main()
