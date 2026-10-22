import importlib
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from binary_sync.acyclic_BP import acyclic_bp_corr2
from binary_sync.bp import bp_corr2
from binary_sync.low_degree import low_degree_corr2 as low_degree_corr
from binary_sync.low_degree import \
    low_degree_weights as compute_low_degree_weights
from binary_sync.low_degree import print_low_degree_weights
from binary_sync.model import Graph, sample_instance
from binary_sync.percolation import estimate_percolation_bound
from binary_sync.reporting import format_estimate
from binary_sync.spectral import spectral_corr2
from binary_sync.utils import draw_graph
from matplotlib.lines import Line2D

matplotlib.use("kitcat")


def generate_sbm_graph(p, sizes, edge_probs, rng=None):
    """
    Sample a graph according to SBM(edge_probabilities)
    p: flipping probability in the binary synchronization model
    sizes: np array of sizes of the communities
    edge_probs: a matrix of edge probabilities
    """
    if rng == None:
        rng = np.random.default_rng()
    num_vertices = np.sum(sizes)
    block_assignment = np.repeat(np.arange(len(sizes)), sizes)
    G = Graph()
    # add vertices
    for i in range(num_vertices):
        G.add_node(i)

    for v in range(num_vertices):
        for u in range(v + 1, num_vertices):
            label_v = block_assignment[v]
            label_u = block_assignment[u]
            if rng.random() < edge_probs[label_u, label_v]:
                G.add_edge(u, v, p)
    return G


# Plot Graph
def draw_sbm_graph(G, sizes, s, t, seed=0, node_size=300):
    """
    Draw an SBM graph, coloring vertices by community and labeling s and t.

    Parameters
    ----------
    G : Graph
        Custom Graph object.
    sizes : np.ndarray
        sizes[k] is the number of vertices in community k.
    s, t : int
        Distinguished source and target vertices.
    seed : int
        Seed for the graph layout.
    node_size : int
        Size of plotted vertices.
    """
    sizes = np.asarray(sizes, dtype=int)

    if sizes.sum() != len(G.nodes):
        raise ValueError("sum(sizes) must equal the number of vertices in G.")

    block_assignment = np.repeat(np.arange(len(sizes)), sizes)

    H = nx.Graph()
    H.add_nodes_from(G.nodes)

    for edge in G.edges:
        H.add_edge(edge.u, edge.v)

    pos = nx.spring_layout(H, seed=seed)

    nx.draw_networkx_edges(
        H,
        pos,
        edge_color="gray",
    )

    nx.draw_networkx_nodes(
        H,
        pos,
        node_color=[float(block_assignment[v]) for v in H.nodes],
        cmap=plt.cm.Set1,
        node_size=node_size,
    )

    labels = {v: ("s" if v == s else "t" if v == t else str(v)) for v in H.nodes}

    nx.draw_networkx_labels(
        H,
        pos,
        labels=labels,
    )

    plt.axis("off")
    plt.show()


p = 0.2
sizes = np.array([5, 10])
edge_probs = np.array([[0.4, 0.05], [0.05, 0.3]])

example = generate_sbm_graph(p, sizes, edge_probs)

s, t = 0, 14

# Plot graph
draw_sbm_graph(example, sizes, s, t)

# # Print Performance of different algorithms
result_low_degree = low_degree_corr(example, 8, s, t)
result_spectral = spectral_corr2(example, s, t, alg="naive", n_samples=10000, seed=3)
result_acyclic_sp = acyclic_bp_corr2(example, s, t, n_samples=50000)

percolation_bound = estimate_percolation_bound(example, s, t)

print(f"Low-degree: {format_estimate(result_low_degree)}")
print(
    f"Spectral: {format_estimate(result_spectral['corr2'], result_spectral['standard_error'])}"
)
print(
    f"Acyclic BP: {format_estimate(result_acyclic_sp['corr2'], result_acyclic_sp['standard error'])}"
)

print(
    f"Percolation Upper bound: {format_estimate(percolation_bound['bound'], percolation_bound['standard_error'])}"
)
