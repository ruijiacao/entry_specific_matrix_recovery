"""Graph model and data generation for Z_2 synchronization.

The observation on an edge e=(u,v) is

    Y_e = sigma_u * sigma_v * X_e,

where the hidden spins sigma_u are i.i.d. Rademacher and X_e is -1 with
probability p_e and +1 otherwise.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Edge:
    """An undirected edge, identified by its position in ``Graph.edges``."""

    u: Any
    v: Any
    p: float


class Graph:
    """Undirected multigraph with an independent flipping probability per edge."""

    def __init__(self) -> None:
        self.nodes: List[Any] = []
        self._node_set = set()
        self.edges: List[Edge] = []

    def add_node(self, node: Any) -> None:
        """Add an isolated vertex if it is not already present."""
        if node not in self._node_set:
            self._node_set.add(node)
            self.nodes.append(node)

    def add_edge(self, u: Any, v: Any, p: float) -> int:
        """Add an edge and return its integer edge id.

        Parallel edges are allowed; self-loops are not. The noise parameter is
        allowed to lie in ``[0, 1)`` to match the original model specification.
        """
        if u == v:
            raise ValueError("Self-loops are not supported.")
        if not 0 <= p < 1:
            raise ValueError("p must lie in [0, 1).")

        for node in (u, v):
            self.add_node(node)

        self.edges.append(Edge(u, v, float(p)))
        return len(self.edges) - 1

    def adjacency(self) -> Dict[Any, List[Tuple[Any, int]]]:
        """Return ``node -> [(neighbor, edge_id), ...]``.

        Keeping edge ids in the adjacency list is important for multigraphs:
        two edges can connect the same pair of vertices but have different
        observations or noise levels.
        """
        adj = {u: [] for u in self.nodes}
        for edge_id, edge in enumerate(self.edges):
            adj[edge.u].append((edge.v, edge_id))
            adj[edge.v].append((edge.u, edge_id))
        return adj

    def __contains__(self, node: Any) -> bool:
        return node in self._node_set

    def __len__(self) -> int:
        return len(self.nodes)


def sample_instance(
    graph: Graph,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[Any, int], np.ndarray]:
    """Sample hidden spins and edge observations.

    Returns
    -------
    spins:
        Dictionary mapping each node to ``+1`` or ``-1``.
    observations:
        Integer NumPy array indexed by edge id.
    """
    if rng is None:
        rng = np.random.default_rng()

    spin_values = rng.choice(np.array([-1, 1], dtype=np.int8), size=len(graph))
    spins = {node: int(spin) for node, spin in zip(graph.nodes, spin_values)}

    observations = np.empty(len(graph.edges), dtype=np.int8)
    for edge_id, edge in enumerate(graph.edges):
        noise = -1 if rng.random() < edge.p else 1
        observations[edge_id] = spins[edge.u] * spins[edge.v] * noise

    return spins, observations


def _run_basic_tests() -> None:
    """Small dependency-free smoke tests for this module."""
    graph = Graph()
    first_id = graph.add_edge("a", "b", 0.0)
    second_id = graph.add_edge("a", "b", 0.25)
    print(graph.edges)


if __name__ == "__main__":
    _run_basic_tests()
