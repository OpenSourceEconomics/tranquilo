import numpy as np
import pandas as pd


def get_bootstrap_indices(data, rng, cluster_by=None, n_draws=1000):
    """Draw positional indices for the construction of bootstrap samples.

    Storing the positional indices instead of the full bootstrap samples saves a lot
    of memory for datasets with many variables.

    Args:
        data (pandas.DataFrame): original dataset.
        rng (numpy.random.Generator): A random number generator.
        cluster_by (str): column name of the variable to cluster by.
        n_draws (int): number of draws, only relevant if seeds is None.

    Returns:
        list: list of numpy arrays with positional indices

    """
    n_obs = len(data)
    if cluster_by is None:
        bootstrap_indices = list(rng.integers(0, n_obs, size=(n_draws, n_obs)))
    else:
        clusters = data[cluster_by].unique()
        drawn_clusters = rng.choice(
            clusters, size=(n_draws, len(clusters)), replace=True
        )

        bootstrap_indices = _convert_cluster_ids_to_indices(
            data[cluster_by], drawn_clusters
        )

    return bootstrap_indices


def _convert_cluster_ids_to_indices(cluster_col, drawn_clusters):
    """Convert the drawn clusters to positional indices of individual observations.

    Args:
        cluster_col (pandas.Series):

    """
    bootstrap_indices = []
    cluster_to_locs = pd.Series(np.arange(len(cluster_col)), index=cluster_col)
    for draw in drawn_clusters:
        bootstrap_indices.append(cluster_to_locs[draw].to_numpy())
    return bootstrap_indices


def get_bootstrap_samples(data, rng, cluster_by=None, n_draws=1000):
    """Draw bootstrap samples.

    If you have memory issues you should use get_bootstrap_indices instead and construct
    the full samples only as needed.

    Args:
        data (pandas.DataFrame): original dataset.
        rng (numpy.random.Generator): A random number generator.
        cluster_by (str): column name of the variable to cluster by.
        n_draws (int): number of draws, only relevant if seeds is None.

    Returns:
        list: list of resampled datasets.

    """
    indices = get_bootstrap_indices(
        data=data,
        rng=rng,
        cluster_by=cluster_by,
        n_draws=n_draws,
    )
    datasets = _get_bootstrap_samples_from_indices(data=data, bootstrap_indices=indices)
    return datasets


def _get_bootstrap_samples_from_indices(data, bootstrap_indices):
    """convert bootstrap indices into actual bootstrap samples.

    Args:
        data (pandas.DataFrame): original dataset.
        bootstrap_indices (list): List with numpy arrays containing positional indices
            of observations in data.

    Returns:
        list: list of DataFrames
    """
    out = [data.iloc[idx] for idx in bootstrap_indices]
    return out
