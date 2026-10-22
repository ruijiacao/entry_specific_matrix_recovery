from typing import Any

import numpy as np

from .model import Graph, sample_instance


def acyclic_bp(G, s, t, observations):
    """
    Use nonbacktracking BP for estimating sigma_s * sigma_t.

    The estimator sums weighted nonbacktracking walks from s to t:

        f(Y) = sum_P prod_{e in P} (1 - 2p_e) Y_e.

    Returns a real number.
    """
    if s == t:
        return 1

    adj = G.adjacency()
    n = len(G)

    # Reasonable default depth.
    num_iter = max(2, int(np.ceil(np.log2(max(n, 2)))))

    # messages[(u, v)] is the total message reaching v from u.
    messages = {}

    # First step out of s.
    for v, edge_id in adj[s]:
        edge = G.edges[edge_id]
        q = 1 - 2 * edge.p

        messages[(s, v)] = messages.get((s, v), 0) + q * observations[edge_id]

    score = 0

    # Include length-1 paths s -> t.
    for (u, v), value in messages.items():
        if v == t:
            score += value

    # Propagate longer nonbacktracking paths.
    for _ in range(1, num_iter):
        new_messages = {}

        for (u, v), value in messages.items():

            # Once we reach t, stop this walk.
            if v == t:
                continue

            for w, edge_id in adj[v]:

                # Ordinary nonbacktracking condition.
                if w == u:
                    continue

                edge = G.edges[edge_id]
                q = 1.0 - 2.0 * edge.p

                new_value = value * q * observations[edge_id]

                key = (v, w)

                new_messages[key] = new_messages.get(key, 0.0) + new_value

        messages = new_messages

        for (u, v), value in messages.items():
            if v == t:
                score += value

        if not messages:
            break

    return score


def acyclic_bp_corr2(G, s, t, n_samples) -> dict[str:Any]:
    """
    Monte Carlo estimate of the squared correlation achieved by acyclic BP.

        corr^2 = E[X f(Y)]^2 / E[f(Y)^2],

    where X = sigma_s * sigma_t.

    Returns
    -------
    corr2 : float
        Estimated squared correlation.

    se : float
        Estimated Monte Carlo standard error of corr2,
        computed using the delta method.
    """
    rng = np.random.default_rng(0)

    A = np.empty(n_samples)
    B = np.empty(n_samples)

    for i in range(n_samples):
        spins, observations = sample_instance(G, rng)

        x = spins[s] * spins[t]
        f = acyclic_bp(G, s, t, observations)

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


if __name__ == "__main__":
    G = Graph()

    # K5, with moderate noise
    n = 5
    p = 0.2

    for u in range(n):
        for v in range(u + 1, n):
            G.add_edge(u, v, p)

    s = 0
    t = 1

    corr2 = acyclic_bp_corr2(
        G,
        s,
        t,
        n_samples=5000,
    )["corr2"]

    print("ABP corr^2:", corr2)

    # Basic sanity check
    assert 0.0 <= corr2 <= 1.0 + 1e-10
