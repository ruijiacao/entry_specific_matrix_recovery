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
from binary_sync.simple_path import simple_path_corr2, simple_path_error
from binary_sync.spectral import naive_spectral_error, spectral_corr2
from binary_sync.utils import draw_graph

matplotlib.use("kitcat")


def test_small_graph(p: float = 0.2) -> Graph:
    """
    Construct a 6-vertex, 9-edge graph

        E = {
            (1,2), (1,3), (1,4), (1,5),
            (2,3), (2,4), (2,6),
            (3,5), (3,6)
        }.

    Every edge has crossover probability p.
    """
    G = Graph()

    edges = [(1, 2), (1, 3), (1, 5), (2, 4), (2, 6), (2, 5), (3, 4)]

    for u, v in edges:
        G.add_edge(u, v, p)

    return G


# Test Code

example = test_small_graph(0.2)
s, t = 6, 1

# Plot graph
draw_graph(example, s, t)
plt.show()

# Print Performance of different algorithms
result_low_degree = low_degree_corr(example, 7, s, t)
result_spectral = spectral_corr2(example, s, t, alg="naive", n_samples=10000, seed=3)
result_acyclic_sp = acyclic_bp_corr2(example, s, t, n_samples=50000)
result_simple_path = simple_path_corr2(example, s, t, 4, n_samples=50000)

percolation_bound = estimate_percolation_bound(example, s, t)

print(f"Low-degree: {format_estimate(result_low_degree)}")
print(
    f"Spectral: {format_estimate(result_spectral['corr2'], result_spectral['standard_error'])}"
)
print(
    f"Acyclic BP: {format_estimate(result_acyclic_sp['corr2'], result_acyclic_sp['standard error'])}"
)

print(
    f"Simple Path: {format_estimate(result_simple_path['corr2'], result_simple_path['standard error'])}"
)
print(
    f"Percolation Upper bound: {format_estimate(percolation_bound['bound'], percolation_bound['standard_error'])}"
)


# simple_path_prob = simple_path_error(example, s, t, 8, n_samples=10000)
# spectral_prob = naive_spectral_error(example, s, t, n_samples=10000)
#
# print(f"Simple path Prob: {simple_path_prob}")
# print(f"Spectral Prob: {spectral_prob}")
