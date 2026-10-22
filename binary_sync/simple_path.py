from typing import Any

import numpy as np

from .model import Graph, sample_instance


def enumerate_simple_paths(G, s, t, L):
    """
    Return all simple s-t paths of length <= L,
    represented as tuples of edge IDs.
    """
    adjacency = {v: [] for v in G.nodes}

    for eid, edge in enumerate(G.edges):
        adjacency[edge.u].append((edge.v, eid))
        adjacency[edge.v].append((edge.u, eid))

    paths = []

    def dfs(v, visited, edge_path):
        if v == t:
            paths.append(tuple(edge_path))
            return

        if len(edge_path) == L:
            return

        for u, eid in adjacency[v]:
            if u in visited:
                continue

            visited.add(u)
            edge_path.append(eid)

            dfs(u, visited, edge_path)

            edge_path.pop()
            visited.remove(u)

    dfs(s, {s}, [])
    return paths


def simple_path_estimator(G, observations, paths):
    total = 0.0

    for path in paths:
        weight = 1.0

        for edge_id in path:
            edge = G.edges[edge_id]
            weight *= (1.0 - 2.0 * edge.p) * observations[edge_id]

        total += weight

    return total


def simple_path_corr2(G, s, t, L, n_samples=1000) -> dict["str", Any]:
    rng = np.random.default_rng(0)

    A = np.empty(n_samples)
    B = np.empty(n_samples)

    all_paths = enumerate_simple_paths(G, s, t, L)

    for i in range(n_samples):
        spins, observations = sample_instance(G, rng)

        x = spins[s] * spins[t]
        f = simple_path_estimator(G, observations, all_paths)

        A[i] = x * f
        B[i] = f * f

    mean_A = np.mean(A)
    mean_B = np.mean(B)

    if mean_B == 0:
        return 0.0, 0.0

    corr2 = mean_A**2 / mean_B

    # Delta method.
    #
    # g(a, b) = a^2 / b
    #
    # grad g =
    #   [2a/b, -a^2/b^2]
    grad = np.array(
        [
            2 * mean_A / mean_B,
            -(mean_A**2) / mean_B**2,
        ]
    )

    samples = np.column_stack([A, B])
    cov = np.cov(samples, rowvar=False, ddof=1) / n_samples

    se = np.sqrt(grad @ cov @ grad)

    return {"corr2": corr2, "standard error": se}


def simple_path_error(G, s, t, L, n_samples=1000):
    """
    Estimate the error of the simple path estimator
    """
    rng = np.random.default_rng(0)

    A = np.empty(n_samples)

    all_paths = enumerate_simple_paths(G, s, t, L)

    for i in range(n_samples):
        spins, observations = sample_instance(G, rng)

        x = spins[s] * spins[t]
        f = np.sign(simple_path_estimator(G, observations, all_paths))

        A[i] = x * f

    num_ones = np.count_nonzero(A == 1)

    return num_ones / n_samples
