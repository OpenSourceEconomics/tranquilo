def adjust_n_evals(n_evals, rho, rho_noise, options):
    """Adjust the number of evaluations based on the noise adaptation options.

    Args:
        n_evals (int): The current number of evaluations.
        rho_noise (np.ndarray): The simulated rho_noises.
        options (NoiseAdaptationOptions): Options for noise adaptation.

    Returns:
        int: The updated number of evaluations.
        bool: Whether the number of evaluations was increased.

    """
    # most rhos are very high -> decrease
    if (rho_noise > options.high_rho).mean() > options.min_share_high_rho:
        new_n_evals = max(n_evals - 1, options.min_n_evals)

    # most rhos are above rho low -> keep constant
    elif (
        rho_noise > options.low_rho
    ).mean() > options.min_share_low_rho or rho >= options.good_rho_threshold:
        new_n_evals = n_evals

    # many rhos are below rho low -> increase
    else:
        new_n_evals = min(n_evals + 1, options.max_n_evals)

    is_increased = new_n_evals > n_evals
    return new_n_evals, is_increased
