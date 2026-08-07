from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from model import Graph, sample_instance

def _signed_adjacency(graph: Graph, observations: np.ndarray) -> np.ndarray:
    """Return the signed adjacency matrix, using the graph's node order."""
    index = {node: i for i, node in enumerate(graph.nodes)}
    matrix = np.zeros((len(graph), len(graph)))

    for edge_id, edge in enumerate(graph.edges):
        i, j = index[edge.u], index[edge.v]
        matrix[i, j] += observations[edge_id]
        matrix[j, i] += observations[edge_id]

    return matrix


def naive_spectral(
    graph: Graph,
    observations: np.ndarray,
    source: Any,
    target: Any,
) -> float:
    """Estimate the pair correlation from the leading signed-adjacency vector."""
    observations = np.asarray(observations)
    if observations.shape != (len(graph.edges),):
        raise ValueError("observations must have one entry per edge")
    if source == target:
        return 1.0

    eigenvalues, eigenvectors = np.linalg.eigh(
        _signed_adjacency(graph, observations)
    )
    vector = eigenvectors[:, np.argmax(eigenvalues)]
    index = {node: i for i, node in enumerate(graph.nodes)}
    signs = np.sign(vector)
    return float((1 if signs[index[source]] >= 0 else -1) *
                 (1 if signs[index[target]] >= 0 else -1))


def nb_spectral(
    graph: Graph,
    observations: np.ndarray,
    source: Any,
    target: Any,
) -> float:
    """Estimate the pair correlation using Algorithm 1 of Saade et al."""
    observations = np.asarray(observations)
    if observations.shape != (len(graph.edges),):
        raise ValueError("observations must have one entry per edge")
    if source == target:
        return 1.0

    index = {node: i for i, node in enumerate(graph.nodes)}
    n = len(graph)
    signed_adjacency = _signed_adjacency(graph, observations)
    degrees = np.diag(np.sum(np.abs(signed_adjacency), axis=1))

    operator = np.block([
        [np.zeros((n, n)), degrees - np.eye(n)],
        [-np.eye(n), signed_adjacency],
    ])

    eigenvalues, eigenvectors = np.linalg.eig(operator)
    eigenvalue = np.argmax(np.abs(eigenvalues))
    vector = np.real(eigenvectors[n:, eigenvalue])
    signs = np.sign(vector)
    return float((1 if signs[index[source]] >= 0 else -1) *
                 (1 if signs[index[target]] >= 0 else -1))

def spectral_corr2(graph, source, target, alg = "nb", n_samples=20_000, seed=0):
    """Estimate the squared correlation between two nodes using spectral methods.
    
    Parameters
    ----------
    graph : Graph
        The graph that defines the pairwise synchronization problem.
    source : a vertex of graph
    target : a vertex of graph
    alg : str, either "naive" or "nb"
    n_samples: number of samples used to estimate the correlation
    seed: random seed for reproducibility
    """

    rng = np.random.default_rng(seed)

    x_values = np.empty(n_samples)
    spectral_values = np.empty(n_samples)

    for i in range(n_samples):
        spins, observations = sample_instance(graph, rng)

        x_values[i] = spins[source] * spins[target]
        if alg == "naive":
            result = naive_spectral(
                graph,
                observations,
                source,
                target,
            )
        else:
            result = nb_spectral(
                graph,
                observations,
                source,
                target,
            )
        spectral_values[i] = result
        

    # U = X f(Y), V = f(Y)^2
    u = x_values * spectral_values
    v = spectral_values**2

    mean_u = np.mean(u)
    mean_v = np.mean(v)

    if mean_v == 0:
        return {
            "corr2": 0.0,
            "standard_error": 0.0,
            "cross_moment": 0.0,
            "estimate_second_moment": 0.0,
        }

    corr2 = mean_u**2 / mean_v

    # Delta-method standard error for g(u, v) = u^2 / v.
    sample_covariance = np.cov(
        np.column_stack([u, v]),
        rowvar=False,
        ddof=1,
    ) / n_samples

    gradient = np.array([
        2 * mean_u / mean_v,
        -(mean_u**2) / mean_v**2,
    ])

    variance = gradient @ sample_covariance @ gradient
    standard_error = np.sqrt(max(variance, 0.0))

    return {
        "corr2": corr2,
        "standard_error": standard_error,
        "cross_moment": mean_u,
        "estimate_second_moment": mean_v,
    }


def _run_basic_tests() -> None:
    # from binary_sync.model import Graph

    graph = Graph()
    graph.add_edge(0, 1, 0.2)
    graph.add_edge(1, 2, 0.2)
    graph.add_edge(2, 3, 0.2)

    nb_result = spectral_corr2(graph, 0, 3, "nb", n_samples = 20000)
    naive_result = spectral_corr2(graph, 0, 3, "naive", n_samples = 20000)
    print(nb_result["corr2"])
    print(naive_result["corr2"])

    # assert naive_spectral(graph, observations, 0, 0) == 1.0
    # assert nb_spectral(graph, observations, 1, 1) == 1.0
    # assert naive_spectral(graph, observations, 0, 2) in (-1.0, 1.0)
    # assert nb_spectral(graph, observations, 0, 2) in (-1.0, 1.0)


if __name__ == "__main__":
    _run_basic_tests()
    print("spectral.py: all basic tests passed")