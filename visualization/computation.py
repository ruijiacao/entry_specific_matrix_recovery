from binary_sync.bp import bp_corr2
from binary_sync.spectral import spectral_corr2
from binary_sync.low_degree import low_degree_corr2


def compute_all_correlations(graph, source, target, degree=3):
    return {
        "BP": bp_corr2(graph, source, target)['corr2'],
        "Naive Spectral": spectral_corr2(graph, source, target)['corr2'],
        "Power Method": spectral_corr2(
            graph,
            source,
            target,
            alg="power",
        )['corr2'],
        "Low-degree": low_degree_corr2(
            graph,
            degree,
            source,
            target,
        ),
    }
