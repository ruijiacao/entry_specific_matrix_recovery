import importlib
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

from binary_sync.acyclic_BP import acyclic_bp_corr2
from binary_sync.bp import bp_corr2
from binary_sync.low_degree import low_degree_corr2 as low_degree_corr
from binary_sync.low_degree import low_degree_weights as compute_low_degree_weights
from binary_sync.low_degree import print_low_degree_weights
from binary_sync.model import Graph, sample_instance
from binary_sync.percolation import estimate_percolation_bound
from binary_sync.reporting import format_estimate
from binary_sync.spectral import spectral_corr2
from binary_sync.utils import draw_graph

matplotlib.use("kitcat")


def generate_er_graph(n, prob, p, rng=None):
    """
    n: number of vertices
    prob: edge probability
    p: probability in the synchronization
    """
    if rng is None:
        rng = np.random.default_rng()
    graph = Graph()
    for vertex in range(n):
        if hasattr(graph, "add_node"):
            graph.add_node(vertex)
        else:
            graph._node_set.add(vertex)
            graph.nodes.append(vertex)
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < prob:
                graph.add_edge(u, v, p)
    return graph


example = generate_er_graph(10, 0.4, 0.2)

s, t = 1, 3

# Plot graph
draw_graph(example, s, t)
plt.show()

# Print Performance of different algorithms
result_low_degree = low_degree_corr(example, 5, s, t)
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
