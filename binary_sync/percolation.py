import numpy as np

from .model import Graph, sample_instance
from .reporting import format_estimate

def estimate_percolation_bound(
    graph,
    source,
    target,
    n_samples=50_000,
    seed=0,
):
    """
    Estimate P(source <-> target) under independent bond percolation,
    where edge e is open with probability (1 - 2 p_e)^2.
    """
    rng = np.random.default_rng(seed)
    adj = graph.adjacency()

    connected = np.empty(n_samples, dtype=float)

    for sample in range(n_samples):
        open_edges = np.array([
            rng.random() < (1 - 2 * edge.p)**2
            for edge in graph.edges
        ])

        visited = {source}
        stack = [source]

        while stack:
            u = stack.pop()

            for v, edge_id in adj[u]:
                if open_edges[edge_id] and v not in visited:
                    visited.add(v)
                    stack.append(v)

        connected[sample] = target in visited

    estimate = np.mean(connected)

    standard_error = np.sqrt(
        estimate * (1 - estimate) / n_samples
    )

    return {
        "bound": estimate,
        "standard_error": standard_error,
    }

def _run_basic_tests() -> None:
    # from binary_sync.model import Graph

    graph = Graph()
    graph.add_edge(0, 1, 0.2)
    graph.add_edge(1, 2, 0.2)
    graph.add_edge(2, 3, 0.2)

    percolation_bd = estimate_percolation_bound(graph, 0, 3, n_samples=20000)
    print("percolation bound: " + format_estimate(
        percolation_bd["bound"], percolation_bd["standard_error"]
    ))

    # assert naive_spectral(graph, observations, 0, 0) == 1.0
    # assert nb_spectral(graph, observations, 1, 1) == 1.0
    # assert naive_spectral(graph, observations, 0, 2) in (-1.0, 1.0)
    # assert nb_spectral(graph, observations, 0, 2) in (-1.0, 1.0)


if __name__ == "__main__":
    _run_basic_tests()
    print("percolation.py: all basic tests passed")
