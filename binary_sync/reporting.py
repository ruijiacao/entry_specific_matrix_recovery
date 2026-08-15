"""Formatting helpers for reporting estimates and confidence intervals."""


def format_estimate(estimate: float, standard_error: float | None = None) -> str:
    """Format an estimate with a normal-approximation 95% interval.

    Correlation-squared and percolation estimates lie in [0, 1], so the
    reported interval is clipped to that range. Missing standard errors are
    treated as deterministic estimates and produce a degenerate interval.
    """
    standard_error = 0.0 if standard_error is None else standard_error
    lower = max(0.0, estimate - 1.96 * standard_error)
    upper = min(1.0, estimate + 1.96 * standard_error)
    return f"estimate={estimate:.6f}, 95% CI=[{lower:.6f}, {upper:.6f}]"
