"""Functions that decide what is the next accepted point, given a candidate.

Decision functions can simply decide whether or not the candidate is accepted but can
also do own function evaluations and decide to accept a different point.

"""
from typing import NamedTuple

import numpy as np
import pandas as pd

from tranquilo.acceptance_sample_size import (
    get_acceptance_sample_sizes,
)
from tranquilo.get_component import get_component
from tranquilo.options import AcceptanceOptions


def get_acceptance_decider(acceptance_decider, acceptance_options):
    func_dict = {
        "classic": _accept_classic,
        "naive_noisy": accept_naive_noisy,
        "noisy": accept_noisy,
        "classic_line_search": accept_classic_line_search,
    }

    out = get_component(
        name_or_func=acceptance_decider,
        func_dict=func_dict,
        component_name="acceptance_decider",
        user_options=acceptance_options,
        default_options=AcceptanceOptions(),
    )

    return out


def _accept_classic(
    subproblem_solution,
    state,
    history,
    *,
    wrapped_criterion,
    min_improvement,
):
    """Do a classic acceptance step for a trustregion algorithm.

    Args:
        subproblem_solution (SubproblemResult): Result of the subproblem solution.
        state (State): Namedtuple containing the trustregion, criterion value of
            previously accepted point, indices of model points, etc.
        wrapped_criterion (callable): The criterion function.
        min_improvement (float): Minimum improvement required to accept a point.

    Returns:
        AcceptanceResult

    """
    out = _accept_simple(
        subproblem_solution=subproblem_solution,
        state=state,
        history=history,
        wrapped_criterion=wrapped_criterion,
        min_improvement=min_improvement,
        n_evals=1,
    )
    return out


def accept_naive_noisy(
    subproblem_solution,
    state,
    history,
    *,
    wrapped_criterion,
    min_improvement,
):
    """Do a naive noisy acceptance step, averaging over a fixed number of points."""
    out = _accept_simple(
        subproblem_solution=subproblem_solution,
        state=state,
        history=history,
        wrapped_criterion=wrapped_criterion,
        min_improvement=min_improvement,
        n_evals=10,
    )
    return out


def accept_classic_line_search(
    subproblem_solution,
    state,
    history,
    *,
    wrapped_criterion,
    min_improvement,
    batch_size,
    sample_points,
    search_radius_factor,
    rng,
):
    # ==================================================================================
    # Quick return if batch_size is 1

    if batch_size == 1:
        return _accept_classic(
            subproblem_solution=subproblem_solution,
            state=state,
            history=history,
            wrapped_criterion=wrapped_criterion,
            min_improvement=min_improvement,
        )

    # ==================================================================================
    # Add candidate to history

    candidate_x = subproblem_solution.x
    candidate_index = history.add_xs(candidate_x)

    eval_info = {candidate_index: 1}

    # ==================================================================================
    # Determine whether the candidate it sufficiently close to the border of the
    # trustregion, in which case we perform a line search

    perform_line_search = _is_on_border(state.trustregion, x=candidate_x, rtol=1e-1)

    if perform_line_search:
        alpha_grid = _generate_alpha_grid(batch_size)

        line_search_xs = _sample_on_line(
            start_point=state.x, direction_point=candidate_x, alpha_grid=alpha_grid
        )
    else:
        line_search_xs = None

    # ==================================================================================
    # Check whether there are any unallocated evaluations left, and if yes perform a
    # speculative sampling

    n_evals_line_search = 0 if line_search_xs is None else len(line_search_xs)
    n_unallocated_evals = batch_size - 1 - n_evals_line_search

    if n_unallocated_evals > 0:
        speculative_xs = _generate_speculative_sample(
            new_center=candidate_x,
            search_radius_factor=search_radius_factor,
            trustregion=state.trustregion,
            sample_points=sample_points,
            n_points=n_unallocated_evals,
            history=history,
            rng=rng,
        )
    else:
        speculative_xs = None

    # ==================================================================================
    # Consolidate newly sampled points

    if line_search_xs is not None and speculative_xs is not None:
        new_xs = np.vstack([line_search_xs, speculative_xs])
    elif line_search_xs is not None:
        new_xs = line_search_xs
    elif speculative_xs is not None:
        new_xs = speculative_xs

    # ==================================================================================
    # Add new points to history and evaluate criterion

    new_indices = history.add_xs(new_xs)

    for idx in new_indices:
        eval_info[idx] = 1

    wrapped_criterion(eval_info)

    # ==================================================================================
    # Calculate rho

    candidate_fval = np.mean(history.get_fvals(candidate_index))

    actual_improvement = -(candidate_fval - state.fval)

    rho = calculate_rho(
        actual_improvement=actual_improvement,
        expected_improvement=subproblem_solution.expected_improvement,
    )

    # ==================================================================================
    # Check if there are any better points

    new_fvals = history.get_fvals(new_indices)
    new_fvals = pd.Series({i: np.mean(fvals) for i, fvals in new_fvals.items()})
    new_fval_argmin = new_fvals.idxmin()

    found_better_candidate = new_fvals.loc[new_fval_argmin] < candidate_fval

    # If a better point was found, update the candidates
    if found_better_candidate:
        candidate_x = history.get_xs(new_fval_argmin)
        candidate_fval = new_fvals.loc[new_fval_argmin]
        candidate_index = new_fval_argmin

    # ==================================================================================
    # Calculate the overall improvement using a potentially updated candidate and draw
    # the acceptance conclusions based on that.

    overall_improvement = -(candidate_fval - state.fval)
    is_accepted = overall_improvement >= min_improvement

    # ==================================================================================
    # Return results

    res = _get_acceptance_result(
        candidate_x=candidate_x,
        candidate_fval=candidate_fval,
        candidate_index=candidate_index,
        rho=rho,
        is_accepted=is_accepted,
        old_state=state,
        n_evals=1,
    )
    return res


def _accept_simple(
    subproblem_solution,
    state,
    history,
    *,
    wrapped_criterion,
    min_improvement,
    n_evals,
):
    """Do a classic acceptance step for a trustregion algorithm.

    Args:
        subproblem_solution (SubproblemResult): Result of the subproblem solution.
        state (State): Namedtuple containing the trustregion, criterion value of
            previously accepted point, indices of model points, etc.
        wrapped_criterion (callable): The criterion function.
        min_improvement (float): Minimum improvement required to accept a point.

    Returns:
        AcceptanceResult

    """
    candidate_x = subproblem_solution.x

    candidate_index = history.add_xs(candidate_x)

    wrapped_criterion({candidate_index: n_evals})

    candidate_fval = np.mean(history.get_fvals(candidate_index))

    actual_improvement = -(candidate_fval - state.fval)

    rho = calculate_rho(
        actual_improvement=actual_improvement,
        expected_improvement=subproblem_solution.expected_improvement,
    )

    is_accepted = actual_improvement >= min_improvement

    res = _get_acceptance_result(
        candidate_x=candidate_x,
        candidate_fval=candidate_fval,
        candidate_index=candidate_index,
        rho=rho,
        is_accepted=is_accepted,
        old_state=state,
        n_evals=n_evals,
    )

    return res


def accept_noisy(
    subproblem_solution,
    state,
    noise_variance,
    history,
    *,
    wrapped_criterion,
    min_improvement,
    power_level,
    confidence_level,
    n_min,
    n_max,
):
    candidate_x = subproblem_solution.x
    candidate_index = history.add_xs(candidate_x)
    existing_n1 = len(history.get_fvals(state.index))

    n_1, n_2 = get_acceptance_sample_sizes(
        sigma=np.sqrt(noise_variance),
        existing_n1=existing_n1,
        expected_improvement=subproblem_solution.expected_improvement,
        power_level=power_level,
        confidence_level=confidence_level,
        n_min=n_min,
        n_max=n_max,
    )

    eval_info = {
        state.index: n_1,
        candidate_index: n_2,
    }

    wrapped_criterion(eval_info)

    current_fval = history.get_fvals(state.index).mean()
    candidate_fval = history.get_fvals(candidate_index).mean()

    actual_improvement = -(candidate_fval - current_fval)

    rho = calculate_rho(
        actual_improvement=actual_improvement,
        expected_improvement=subproblem_solution.expected_improvement,
    )

    is_accepted = actual_improvement >= min_improvement

    res = _get_acceptance_result(
        candidate_x=candidate_x,
        candidate_fval=candidate_fval,
        candidate_index=candidate_index,
        rho=rho,
        is_accepted=is_accepted,
        old_state=state,
        n_evals=n_2,
    )

    return res


class AcceptanceResult(NamedTuple):
    x: np.ndarray
    fval: float
    index: int
    rho: float
    accepted: bool
    step_length: float
    relative_step_length: float
    candidate_index: int
    candidate_x: np.ndarray
    n_evals_acceptance: int


def _get_acceptance_result(
    candidate_x,
    candidate_fval,
    candidate_index,
    rho,
    is_accepted,
    old_state,
    n_evals,
):
    x = candidate_x if is_accepted else old_state.x
    fval = candidate_fval if is_accepted else old_state.fval
    index = candidate_index if is_accepted else old_state.index
    step_length = np.linalg.norm(x - old_state.x, ord=2)
    relative_step_length = step_length / old_state.trustregion.radius

    out = AcceptanceResult(
        x=x,
        fval=fval,
        index=index,
        rho=rho,
        accepted=is_accepted,
        step_length=step_length,
        relative_step_length=relative_step_length,
        candidate_index=candidate_index,
        candidate_x=candidate_x,
        n_evals_acceptance=n_evals,
    )
    return out


def calculate_rho(actual_improvement, expected_improvement):
    if expected_improvement == 0 and actual_improvement > 0:
        rho = np.inf
    elif expected_improvement == 0:
        rho = -np.inf
    else:
        rho = actual_improvement / expected_improvement
    return rho


# ======================================================================================
# Helper functions for line search
# ======================================================================================


def _generate_speculative_sample(
    new_center, trustregion, sample_points, n_points, history, search_radius_factor, rng
):
    """Generative a speculative sample.

    Args:
        new_center (np.ndarray): New center of the trust region.
        trustregion (Region): Current trust region.
        sample_points (callable): Function to sample points.
        n_points (int): Number of points to sample.
        history (History): Tranquilo history.
        search_radius_factor (float): Factor to multiply the trust region radius by to
            get the search radius.
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: Speculative sample.

    """
    search_region = trustregion._replace(
        center=new_center, radius=search_radius_factor * trustregion.radius
    )

    old_indices = history.get_x_indices_in_region(search_region)

    old_xs = history.get_xs(old_indices)

    model_xs = old_xs

    new_xs = sample_points(
        search_region,
        n_points=n_points,
        existing_xs=model_xs,
        rng=rng,
    )
    return new_xs


def _sample_on_line(start_point, direction_point, alpha_grid):
    """Sample points on a line defined by startind and direction points.

    Args:
        start_point (np.ndarray): Starting point of the line.
        direction_point (np.ndarray): Direction point of the line.
        alpha_grid (np.ndarray): Grid of alphas to sample points on the line. 0
            corresponds to the starting point and 1 corresponds to the direction point.
            Points larger than 1 are beyond the direction point.

    Returns:
        np.ndarray: Sampled points on the line.

    """
    xs = start_point + alpha_grid.reshape(-1, 1) * (direction_point - start_point)
    return xs


def _is_on_border(trustregion, x, rtol):
    """Check whether a point is sufficiently close to the border of a trust region.

    Args:
        trustregion (Region): Trust region.
        x (np.ndarray): Point to check.
        rtol (float): Relative tolerance.

    Returns:
        bool: True if the point is sufficiently close to the border of the trust region.

    """
    if trustregion.shape == "sphere":
        candidate_on_border = _is_on_sphere_border(trustregion, x=x, rtol=rtol)
    else:
        candidate_on_border = _is_on_cube_border(trustregion, x=x, rtol=rtol)
    return candidate_on_border


def _is_on_sphere_border(trustregion, x, rtol):
    x_center_dist = np.linalg.norm(x - trustregion.center, ord=2)
    return np.isclose(x_center_dist, trustregion.radius, rtol=rtol)


def _is_on_cube_border(trustregion, x, rtol):
    cube_bounds = trustregion.cube_bounds
    is_on_lower_border = np.isclose(x, cube_bounds.lower, rtol=rtol).any()
    is_on_upper_border = np.isclose(x, cube_bounds.upper, rtol=rtol).any()
    return is_on_lower_border or is_on_upper_border


def _generate_alpha_grid(batch_size):
    n_points = min(batch_size, 4) - 1
    return 2 ** np.arange(1, n_points + 1, dtype=float)
