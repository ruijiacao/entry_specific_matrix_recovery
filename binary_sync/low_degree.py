"""Optimal low-degree polynomial correlation for binary synchronization."""

from itertools import combinations
from typing import Any

import numpy as np

from .model import Graph
from .reporting import format_estimate


def _boundary(graph: Graph, subset: tuple[int, ...]) -> frozenset[Any]:
    """Return the vertices having odd degree in an edge subset."""
    parity = {}
    for edge_id in subset:
        edge = graph.edges[edge_id]
        parity[edge.u] = 1 - parity.get(edge.u, 0)
        parity[edge.v] = 1 - parity.get(edge.v, 0)
    return frozenset(node for node, value in parity.items() if value)


def low_degree_corr2(graph: Graph, degree: int, source: Any, sink: Any) -> float:
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


def low_degree_weights(G: Graph, s, t, p, degree=None):
    """
    Compute the optimal low-degree weights w = Sigma^{-1} c.

    A basis element F is represented by a tuple of edge IDs and corresponds
    to the monomial

        Y_F = product_{e in F} Y_e.

    Only subsets satisfying boundary(F) = {s, t} are included.

    Parameters
    ----------
    G : Graph
        Graph whose edges have attributes `u` and `v`.
    s, t :
        Source and target vertices.
    p : float
        Edge-flip probability.
    degree : int or None
        Maximum monomial degree. If None, use all edge subsets.

    Returns
    -------
    dict[tuple[int, ...], float]
        Maps each basis monomial, represented by its edge IDs, to its
        optimal coefficient.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1].")

    edges = G.edges
    num_edges = len(edges)

    if degree is None:
        degree = num_edges

    if not 0 <= degree <= num_edges:
        raise ValueError(
            f"degree must lie between 0 and {num_edges}."
        )

    target_boundary = set() if s == t else {s, t}
    basis_masks = []

    for size in range(degree + 1):
        for edge_ids in combinations(range(num_edges), size):
            boundary = set()
            mask = 0

            for edge_id in edge_ids:
                edge = edges[edge_id]

                # Toggle both endpoints. This also handles self-loops.
                for vertex in (edge.u, edge.v):
                    if vertex in boundary:
                        boundary.remove(vertex)
                    else:
                        boundary.add(vertex)

                mask |= 1 << edge_id

            if boundary == target_boundary:
                basis_masks.append(mask)

    if not basis_masks:
        return {}

    q = 1.0 - 2.0 * p
    num_basis = len(basis_masks)

    # c_i = E[(sigma_s sigma_t) Y_{F_i}]
    c = np.array(
        [q ** mask.bit_count() for mask in basis_masks],
        dtype=float,
    )

    # Sigma_ij = E[Y_{F_i} Y_{F_j}]
    #          = q^{|F_i symmetric_difference F_j|}.
    sigma = np.empty((num_basis, num_basis), dtype=float)

    for i, mask_i in enumerate(basis_masks):
        sigma[i, i] = 1.0

        for j in range(i):
            value = q ** (mask_i ^ basis_masks[j]).bit_count()
            sigma[i, j] = value
            sigma[j, i] = value

    try:
        weights = np.linalg.solve(sigma, c)
    except np.linalg.LinAlgError:
        # Handles degenerate cases such as p = 0 or p = 1.
        weights = np.linalg.lstsq(sigma, c, rcond=None)[0]

    def mask_to_edge_ids(mask):
        return tuple(
            edge_id
            for edge_id in range(num_edges)
            if mask & (1 << edge_id)
        )

    return {
        mask_to_edge_ids(mask): float(weight)
        for mask, weight in zip(basis_masks, weights)
    }
def print_low_degree_weights(G, weights):
    print("Edge IDs:")
    for edge_id, edge in enumerate(G.edges):
        print(f"  {edge_id}: ({edge.u}, {edge.v})")

    print("\nLow-degree weights:")
    for edge_ids, weight in weights.items():
        edge_list = [
            (G.edges[edge_id].u, G.edges[edge_id].v)
            for edge_id in edge_ids
        ]

        print(
            f"  basis={edge_ids}, "
            f"edges={edge_list}, "
            f"weight={weight:.6f}"
        )


def _run_basic_tests() -> None:
    """Small tests for paths and the degree cutoff."""
    graph = Graph()
    graph.add_edge(0, 1, 0.2)
    graph.add_edge(1, 2, 0.2)
    graph.add_edge(2, 3, 0.2)

    # The only source-sink monomial has all three edges.
    weights = low_degree_weights(
        graph,
        s=0,
        t=2,
        p=0.1,
        degree=2,)

    print_low_degree_weights(graph, weights)




if __name__ == "__main__":
    _run_basic_tests()
    print("polynomial.py: all basic tests passed")
