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
from binary_sync.spectral import naive_spectral, naive_spectral_error, spectral_corr2
from binary_sync.utils import draw_graph

matplotlib.use("kitcat")


def make_two_cluster_graph(n, m, p=0.2):
    G = Graph()

    left = [f"a{0}"] + [f"a{i}" for i in range(1, n)]
    right = [f"b{0}"] + [f"b{i}" for i in range(1, m)]

    # Dense left cluster.
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(left[i], left[j], p)

    # Dense right cluster.
    for i in range(m):
        for j in range(i + 1, m, 5):
            G.add_edge(right[i], right[j], p)

    for i in range(m):
        for j in range(i + 1, m):
            G.add_edge(right[i], right[j], p)
    # Sparse cut between clusters.
    # for i in range(1, n):
    #     G.add_edge(f"a{i}", f"b{i}", p)
    G.add_edge("a4", "b1", p)
    G.add_edge("a2", "b2", p)

    s, t = "b1", "b2"
    # s, t = "a1", "a0"

    return G, s, t


def make_parallel_series():
    G = Graph()

    right = [f"a{0}"] + [f"al{i}" for i in range(3)]
    for i in range(3):
        for j in range(i + 1, m, 5):
            G.add_edge(right[i], right[j], p)
    G.add_edge("v", "a0", 0.4)
    G.add_edge("v", "a1", 0.4)


example, s, t = make_two_cluster_graph(6, 3, p=0.3)
draw_graph(example, s, t, title="Two-cluster graph")
plt.show()


# Print Performance of different algorithms
result_low_degree = low_degree_corr(example, 5, s, t)
result_spectral = spectral_corr2(example, s, t, alg="naive", n_samples=10000, seed=3)
# result_acyclic_sp = acyclic_bp_corr2(example, s, t, n_samples=50000)
# result_simple_path = simple_path_corr2(example, s, t, 9, n_samples=1000)

# percolation_bound = estimate_percolation_bound(example, s, t)
print(f"Low-degree: {format_estimate(result_low_degree)}")
print(
    f"Spectral: {format_estimate(result_spectral['corr2'], result_spectral['standard_error'])}"
)
# print(
#     f"Acyclic BP: {format_estimate(result_acyclic_sp['corr2'], result_acyclic_sp['standard error'])}"
# )

# print(
#     f"Simple Path: {format_estimate(result_simple_path['corr2'], result_simple_path['standard error'])}"
# )

# print(
#     f"Percolation Upper bound: {format_estimate(percolation_bound['bound'], percolation_bound['standard_error'])}"
# )
