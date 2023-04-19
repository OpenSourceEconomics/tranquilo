import numpy as np


def adjust_radius_and_n_evals(
    radius,
    n_evals,
    rho,
    step_length,
    radius_options,
    rho_noise,
    noise_adaptation_options,
):
    is_large_step = step_length / radius >= radius_options.large_step

    # successful case
    if rho >= radius_options.rho_increase:
        new_radius = (
            radius * radius_options.expansion_factor if is_large_step else radius
        )
        if (rho_noise > noise_adaptation_options.high_rho).mean() > 0.7:
            new_n_evals = n_evals - 1
        else:
            new_n_evals = n_evals

    # neutral case
    elif rho >= radius_options.rho_decrease:
        new_radius = radius
        new_n_evals = n_evals

    # bad case
    else:
        if (rho_noise > noise_adaptation_options.low_rho).mean() > 0.9:
            new_n_evals = n_evals
            new_radius = radius * radius_options.shrinking_factor
        else:
            new_n_evals = n_evals + 1
            new_radius = radius

    new_radius = np.clip(
        new_radius, radius_options.min_radius, radius_options.max_radius
    )
    new_n_evals = np.clip(
        new_n_evals,
        noise_adaptation_options.min_n_evals,
        noise_adaptation_options.max_n_evals,
    )

    return new_radius, new_n_evals
