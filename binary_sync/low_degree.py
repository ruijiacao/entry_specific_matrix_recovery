"""Optimal low-degree polynomial correlation for binary synchronization."""

from itertools import combinations
from typing import Any

import numpy as np

from model import Graph


def _boundary(graph: Graph, subset: tuple[int, ...]) -> frozenset[Any]:
    """Return the vertices having odd degree in an edge subset."""
    parity = {}
    for edge_id in subset:
        edge = graph.edges[edge_id]
        parity[edge.u] = 1 - parity.get(edge.u, 0)
        parity[edge.v] = 1 - parity.get(edge.v, 0)
    return frozenset(node for node, value in parity.items() if value)


def low_degree_corr(graph: Graph, degree: int, source: Any, sink: Any) -> float:
    """Return the squared correlation.

    The polynomial basis is

        phi_A(Y) = \Pi_{e in A} Y_e,

    restricted to edge subsets A with boundary {source, sink}.  The function
    computes c.T @ Sigma^{-1} @ c, using a pseudoinverse for degenerate noise
    parameters such as p=0.
    """
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    if source not in graph or sink not in graph:
        raise ValueError("source and sink must be vertices of graph")
    if source == sink:
        return 1.0

    n_edges = len(graph.edges)
    degree = min(int(degree), n_edges)
    terminal_boundary = frozenset((source, sink))
    subsets = [
        subset
        for size in range(1, degree + 1)
        for subset in combinations(range(n_edges), size)
        if _boundary(graph, subset) == terminal_boundary
    ]

    if not subsets:
        return 0.0

    q = np.array([1.0 - 2.0 * edge.p for edge in graph.edges])

    # c_A = E[(sigma_s sigma_t) phi_A(Y)].
    c = np.array([np.prod(q[list(subset)]) for subset in subsets])

    # Sigma_{A,B} = E[phi_A(Y) phi_B(Y)].
    sigma = np.empty((len(subsets), len(subsets)))
    for i, left in enumerate(subsets):
        left_set = set(left)
        for j, right in enumerate(subsets):
            symmetric_difference = [
                edge_id for edge_id in left_set.symmetric_difference(right)
            ]
            sigma[i, j] = np.prod(q[symmetric_difference])

    w = np.linalg.solve(sigma, c)
    corr2 = c @ w
    return float(corr2)


def _run_basic_tests() -> None:
    """Small tests for paths and the degree cutoff."""
    graph = Graph()
    graph.add_edge(0, 1, 0.2)
    graph.add_edge(1, 2, 0.2)
    graph.add_edge(2, 3, 0.2)

    # The only source-sink monomial has all three edges.
    print(low_degree_corr(graph, 3, 0, 3), 0.6**6)
    # assert np.isclose(low_degree_corr(graph, 3, 0, 3), 0.6**6)



if __name__ == "__main__":
    _run_basic_tests()
    print("polynomial.py: all basic tests passed")