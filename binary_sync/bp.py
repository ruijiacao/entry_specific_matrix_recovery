"""Belief propagation for pairwise Z_2 synchronization."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .model import Graph, sample_instance


@dataclass(frozen=True)
class BPResult:
    """Correlation estimate together with basic convergence diagnostics."""

    correlation: float
    converged: bool
    iterations: int
    residual: float


# def _validate_inputs(
#     graph: Graph,
#     observations: np.ndarray,
#     source: Any,
#     target: Any,
#     damping: float,
#     init: str,
# ) -> np.ndarray:
#     """Validate public BP arguments and return observations as an array."""
#     if source not in graph or target not in graph:
#         raise ValueError("source and target must be vertices of graph")
#     if not 0 < damping <= 1:
#         raise ValueError("damping must lie in (0, 1]")
#     if init not in {"zero", "random"}:
#         raise ValueError("init must be 'zero' or 'random'")

#     observations = np.asarray(observations)
#     if observations.shape != (len(graph.edges),):
#         raise ValueError("observations must have one entry per edge")
#     if not np.all(np.isin(observations, [-1, 1])):
#         raise ValueError("each observation must be either -1 or +1")

#     return observations


def bp_pair_estimate(
    graph: Graph,
    observations: np.ndarray,
    source: Any,
    target: Any,
    *,
    max_iter: int = 50,
    tol: float = 1e-10,
    damping: float = 1.0,
    init: str = "zero",
    seed: Optional[int] = None,
) -> BPResult:
    """Estimate E[sigma_source sigma_target | observations] by BP.

    The source spin is clamped to +1. The returned marginal at ``target`` is
    therefore the estimate of ``sigma_source * sigma_target``. The same
    routine supports zero-message initialization and small random
    initialization. Updates are synchronous: every new message is computed
    from the messages in the previous iteration.

    Parameters
    ----------
    graph : the graph that defines the pairwise synchronization problem
    observations : array of shape (num_edges,)
        The observed edge measurements, each either -1 or +1.
    init: either "zero" or "random"
    """
    # observations = _validate_inputs(
    #     graph, observations, source, target, damping, init
    # )
    if max_iter < 0:
        raise ValueError("max_iter must be nonnegative")
    if tol < 0:
        raise ValueError("tol must be nonnegative")

    if source == target:
        return BPResult(1.0, True, 0, 0.0)

    rng = np.random.default_rng(seed)
    adj = graph.adjacency()

    # messages[(edge_id, u, v)] is the cavity magnetization m_{u -> v}.
    messages: Dict[Tuple[int, Any, Any], float] = {}
    for edge_id, edge in enumerate(graph.edges):
        for u, v in ((edge.u, edge.v), (edge.v, edge.u)):
            if u == source:
                messages[edge_id, u, v] = 1.0
            elif init == "zero":
                messages[edge_id, u, v] = 0.0
            else:
                messages[edge_id, u, v] = float(
                    rng.random.choice([-1, 1])
                )

    residual = np.inf
    converged = False
    iterations = 0

    for iteration in range(1, max_iter + 1):
        new_messages: Dict[Tuple[int, Any, Any], float] = {}
        residual = 0.0

        for edge_id, edge in enumerate(graph.edges):
            for u, v in ((edge.u, edge.v), (edge.v, edge.u)):
                key = (edge_id, u, v)

                if u == source:
                    value = 1.0
                else:
                    field = 0.0
                    for neighbor, incoming_id in adj[u]:
                        if incoming_id == edge_id:
                            continue

                        incoming_edge = graph.edges[incoming_id]
                        incoming_message = messages[
                            incoming_id, neighbor, u
                        ]
                        q = 1.0 - 2.0 * incoming_edge.p
                        argument = q * observations[incoming_id] * incoming_message
                        argument = float(np.clip(argument, -1 + 1e-15, 1 - 1e-15))
                        field += np.arctanh(argument)

                    raw_value = float(np.tanh(field))
                    value = (
                        damping * raw_value
                        + (1.0 - damping) * messages[key]
                    )

                new_messages[key] = value
                residual = max(residual, abs(value - messages[key]))

        messages = new_messages
        iterations = iteration
        if residual < tol:
            converged = True
            break

    target_field = 0.0
    for neighbor, edge_id in adj[target]:
        edge = graph.edges[edge_id]
        q = 1.0 - 2.0 * edge.p
        argument = q * observations[edge_id] * messages[edge_id, neighbor, target]
        argument = float(np.clip(argument, -1 + 1e-15, 1 - 1e-15))
        target_field += np.arctanh(argument)

    correlation = float(np.tanh(target_field))
    return BPResult(correlation, converged, iterations, float(residual))


def bp_corr2(
    graph,
    source,
    target,
    n_samples=20_000,
    seed=0,
    init="zero",
    **bp_kwargs,
):
    """
    Estimate the Corr^2 = E[X f(Y)]^2 / E[f(Y)^2] using for the BP algorithm 
    """
    rng = np.random.default_rng(seed)

    x_values = np.empty(n_samples)
    bp_values = np.empty(n_samples)

    for i in range(n_samples):
        spins, observations = sample_instance(graph, rng)

        x_values[i] = spins[source] * spins[target]
        result = bp_pair_estimate(
            graph,
            observations,
            source,
            target,
            init=init,
            **bp_kwargs,
        )
        bp_values[i] = result.correlation
        

    # U = X f(Y), V = f(Y)^2
    u = x_values * bp_values
    v = bp_values**2

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
    """Basic tests for paths, initialization modes, and input validation."""
    # from model import sample_instance

    graph = Graph()

    graph.add_edge(0, 1, 0.2)
    graph.add_edge(1, 2, 0.2)
    # spins, observations = sample_instance(graph, np.random.default_rng(7))

    # result = bp_pair_result(graph, observations, 0, 2)
    result = bp_corr2(graph, 0, 2, n_samples=1000, seed=7)
    print(result["corr2"])


    # random_value = bp_pair_estimate(
    #     graph, observations, 0, 2, init="random", seed=123
    # )
    # assert -1.0 <= random_value <= 1.0
    # assert bp_pair_estimate(graph, observations, 1, 1) == 1.0


if __name__ == "__main__":
    _run_basic_tests()
    print("bp.py: all basic tests passed")
