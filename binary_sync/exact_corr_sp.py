"""Exact correlation for two-terminal series-parallel graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from binary_sync.model import Graph
from binary_sync.reporting import format_estimate


Law = Dict[float, float]


@dataclass
class _Component:
    """A reduced two-terminal component and the law of its BP magnetization."""

    u: Any
    v: Any
    law: Law


def _add_mass(law: Law, value: float, probability: float) -> None:
    """Add probability mass, including when two float keys coincide (e.g. q=0)."""
    if probability != 0.0:
        law[value] = law.get(value, 0.0) + probability


def _edge_law(q: float) -> Law:
    # We condition on the edge's endpoint product being +1. Then Y is the
    # channel noise and M = E[sigma_u sigma_v | Y] = qY.
    law: Law = {}
    _add_mass(law, q, (1.0 + q) / 2.0)
    _add_mass(law, -q, (1.0 - q) / 2.0)
    return law


def _combine_laws(left: Law, right: Law, *, parallel: bool) -> Law:
    result: Law = {}

    for m1, prob1 in left.items():
        for m2, prob2 in right.items():
            probability = prob1 * prob2
            if probability == 0.0:
                continue

            if parallel:
                denominator = 1.0 + m1 * m2
                if denominator == 0.0:
                    # This can only represent contradictory probability-zero
                    # evidence for a valid channel, which was removed above.
                    continue
                value = (m1 + m2) / denominator
            else:
                value = m1 * m2

            _add_mass(result, value, probability)

    return result


def _other_endpoint(component: _Component, vertex: Any) -> Any:
    return component.v if component.u == vertex else component.u


def sp_corr2(G: Graph, s: Any, t: Any, p: float) -> float:
    """Return the exact Bayes correlation squared between ``s`` and ``t``.

    The graph must be a *two-terminal series-parallel graph* with terminals
    ``s`` and ``t``. Every edge is assumed to pass through the same binary
    symmetric channel with flip probability ``p``; the ``p`` values stored on
    the individual ``Graph.edges`` are not used.

    The function recognizes the S-P structure by repeatedly applying the
    reverse construction rules:

    * merge components with the same endpoints (parallel reduction), and
    * suppress a nonterminal vertex incident to exactly two components
      (series reduction).

    During each reduction it propagates the exact finite distribution of the
    effective posterior magnetization. It raises ``ValueError`` if the graph
    cannot be reduced to one component joining ``s`` and ``t``.
    """
    if s == t:
        raise ValueError("s and t must be distinct")
    if s not in G or t not in G:
        raise ValueError("s and t must both be vertices of G")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    if not G.edges:
        raise ValueError("G must contain at least one edge")

    q = 1.0 - 2.0 * float(p)
    components = [
        _Component(edge.u, edge.v, _edge_law(q)) for edge in G.edges
    ]

    while True:
        # Parallel reduction. A frozenset works for arbitrary hashable vertex
        # labels and does not require labels to be mutually orderable.
        groups: dict[frozenset[Any], list[_Component]] = {}
        for component in components:
            if component.u == component.v:
                raise ValueError("self-loops are not allowed")
            key = frozenset((component.u, component.v))
            groups.setdefault(key, []).append(component)

        parallel_reduced = False
        new_components: list[_Component] = []
        for group in groups.values():
            merged = group[0]
            for component in group[1:]:
                merged = _Component(
                    merged.u,
                    merged.v,
                    _combine_laws(merged.law, component.law, parallel=True),
                )
                parallel_reduced = True
            new_components.append(merged)
        components = new_components

        if (
            len(components) == 1
            and {components[0].u, components[0].v} == {s, t}
        ):
            root_law = components[0].law
            total_probability = sum(root_law.values())
            if total_probability == 0.0:
                raise ArithmeticError("the effective distribution has zero mass")
            corr2 = sum(probability * m * m for m, probability in root_law.items())
            return min(1.0, max(0.0, corr2 / total_probability))

        # Build incidence lists after parallel edges have been merged.
        incidence: dict[Any, list[int]] = {}
        for index, component in enumerate(components):
            incidence.setdefault(component.u, []).append(index)
            incidence.setdefault(component.v, []).append(index)

        series_vertex = next(
            (
                vertex
                for vertex, incident in incidence.items()
                if vertex not in (s, t) and len(incident) == 2
            ),
            None,
        )

        if series_vertex is None:
            if parallel_reduced:
                # Parallel reduction changed the graph; make another pass in
                # case it exposed a degree-two series vertex.
                continue
            raise ValueError(
                "G is not a two-terminal series-parallel graph with "
                f"terminals {s!r} and {t!r}"
            )

        first_index, second_index = incidence[series_vertex]
        first = components[first_index]
        second = components[second_index]
        u = _other_endpoint(first, series_vertex)
        v = _other_endpoint(second, series_vertex)

        if u == v:
            raise ValueError(
                "G is not a two-terminal series-parallel graph: "
                "a series reduction produced a self-loop"
            )

        series_component = _Component(
            u,
            v,
            _combine_laws(first.law, second.law, parallel=False),
        )
        components = [
            component
            for index, component in enumerate(components)
            if index not in (first_index, second_index)
        ]
        components.append(series_component)


def _run_sanity_check() -> None:
    """A length-three path has Corr^2 = (1 - 2p)^(2 * 3)."""
    p = 0.1
    graph = Graph()
    graph.add_edge("s", "u", p)
    graph.add_edge("u", "v", p)
    graph.add_edge("v", "t", p)

    actual = sp_corr2(graph, "s", "t", p)
    expected = (1.0 - 2.0 * p) ** 6
    assert abs(actual - expected) < 1e-12, (actual, expected)
    print("series-parallel exact: " + format_estimate(actual))


if __name__ == "__main__":
    _run_sanity_check()
    print("series_parallel.py: sanity check passed")
