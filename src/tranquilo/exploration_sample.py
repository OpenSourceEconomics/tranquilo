import numpy as np
from scipy.stats import qmc, triang
from tranquilo.utilities import get_rng


def draw_exploration_sample(
    x,
    lower,
    upper,
    n_samples,
    sampling_distribution,
    sampling_method,
    seed,
):
    """Get a sample of parameter values for the first stage of the tiktak algorithm.

    The sample is created randomly or using a low discrepancy sequence. Different
    distributions are available.

    Args:
        x (np.ndarray): Internal parameter vector of shape (n_params,).
        lower (np.ndarray): Vector of internal lower bounds of shape (n_params,).
        upper (np.ndarray): Vector of internal upper bounds of shape (n_params,).
        n_samples (int): Number of sample points on which one function evaluation
            shall be performed. Default is 10 * n_params.
        sampling_distribution (str): One of "uniform", "triangular". Default is
            "uniform", as in the original tiktak algorithm.
        sampling_method (str): One of "sobol", "halton", "latin_hypercube" or
            "random". Default is sobol for problems with up to 200 parameters
            and random for problems with more than 200 parameters.
        seed (int): Random seed.

    Returns:
        np.ndarray: Numpy array of shape (n_samples, n_params).
            Each row represents a vector of parameter values.

    """
    valid_rules = ["sobol", "halton", "latin_hypercube", "random"]
    valid_distributions = ["uniform", "triangular"]

    if sampling_method not in valid_rules:
        raise ValueError(
            f"Invalid rule: {sampling_method}. Must be one of\n\n{valid_rules}\n\n"
        )

    if sampling_distribution not in valid_distributions:
        raise ValueError(f"Unsupported distribution: {sampling_distribution}")

    for name, bound in zip(["lower", "upper"], [lower, upper]):
        if not np.isfinite(bound).all():
            raise ValueError(
                f"multistart optimization requires finite {name}_bounds or "
                f"soft_{name}_bounds for all parameters."
            )

    if sampling_method == "sobol":
        # Draw `n` points from the open interval (lower, upper)^d.
        # Note that scipy uses the half-open interval [lower, upper)^d internally.
        # We apply a burn-in phase of 1, i.e. we skip the first point in the sequence
        # and thus exclude the lower bound.
        sampler = qmc.Sobol(d=len(lower), scramble=False, seed=seed)
        _ = sampler.fast_forward(1)
        sample_unscaled = sampler.random(n=n_samples)

    elif sampling_method == "halton":
        sampler = qmc.Halton(d=len(lower), scramble=False, seed=seed)
        sample_unscaled = sampler.random(n=n_samples)

    elif sampling_method == "latin_hypercube":
        sampler = qmc.LatinHypercube(d=len(lower), strength=1, seed=seed)
        sample_unscaled = sampler.random(n=n_samples)

    elif sampling_method == "random":
        rng = get_rng(seed)
        sample_unscaled = rng.uniform(size=(n_samples, len(lower)))

    if sampling_distribution == "uniform":
        sample_scaled = qmc.scale(sample_unscaled, lower, upper)
    elif sampling_distribution == "triangular":
        sample_scaled = triang.ppf(
            sample_unscaled,
            c=(x - lower) / (upper - lower),
            loc=lower,
            scale=upper - lower,
        )

    return sample_scaled
