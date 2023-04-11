"""Functions that decide what is the next accepted point, given a candidate.

Decision functions can simply decide whether or not the candidate is accepted but can
also do own function evaluations and decide to accept a different point.

"""
from typing import NamedTuple

import pandas as pd
import numpy as np

from tranquilo.acceptance_sample_size import (
    get_acceptance_sample_sizes,
)
from tranquilo.get_component import get_component
from tranquilo.options import AcceptanceOptions


def get_acceptance_decider(acceptance_decider, acceptance_options):
    func_dict = {
        "classic": _accept_classic,
        "classic_speculative": _accept_classic_speculative,
        "classic_line_search": _accept_classic_line_search,
        "naive_noisy": accept_naive_noisy,
        "noisy": accept_noisy,
    }

    mandatory_args = [
        "subproblem_solution",
        "state",
        "history",
    ]

    out = get_component(
        name_or_func=acceptance_decider,
        func_dict=func_dict,
        component_name="acceptance_decider",
        default_options=AcceptanceOptions(),
        user_options=acceptance_options,
        mandatory_signature=mandatory_args,
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


def _accept_classic_speculative(
    subproblem_solution,
    state,
    history,
    batch_size,
    sample_points,
    rng,
    *,
    wrapped_criterion,
    min_improvement,
):
    """Do a speculative classic acceptance step for a trustregion algorithm.

    Args:
        subproblem_solution (SubproblemResult): Result of the subproblem solution.
        state (State): Namedtuple containing the trustregion, criterion value of
            previously accepted point, indices of model points, etc.
        history (History): The tranquilo history.
        batch_size (int): Number of points to evaluate in parallel.
        sample_points (callable): Function that samples points from the trustregion.
        rng (np.random.Generator): Random number generator.
        wrapped_criterion (callable): The criterion function.
        min_improvement (float): Minimum improvement required to accept a point.

    Returns:
        AcceptanceResult: The acceptance result.

    """
    if batch_size == 1:
        # quick return when only one point should be evaluated
        return _accept_classic(
            subproblem_solution=subproblem_solution,
            state=state,
            history=history,
            wrapped_criterion=wrapped_criterion,
            min_improvement=min_improvement,
        )

    # Perform candidate and speculative evaluations
    # ==================================================================================

    candidate_x = subproblem_solution.x
    candidate_index = history.add_xs(candidate_x)

    eval_info = {candidate_index: 1}

    # draw remaining points from trustregion with same radius but centered at candidate
    anticipated_trustregion = state.trustregion._replace(center=candidate_x)
    new_xs = sample_points(
        trustregion=anticipated_trustregion, n_points=batch_size - 1, rng=rng
    )

    new_indices = history.add_xs(new_xs)
    eval_info = {**eval_info, **{i: 1 for i in new_indices}}

    wrapped_criterion(eval_info)

    # Take acceptance decision
    # ==================================================================================
    candidate_fval = np.mean(history.get_fvals(candidate_index))

    actual_improvement = -(candidate_fval - state.fval)

    rho = calculate_rho(
        actual_improvement=actual_improvement,
        expected_improvement=subproblem_solution.expected_improvement,
    )

    is_accepted = actual_improvement >= min_improvement

    out = _get_acceptance_result(
        candidate_x=candidate_x,
        candidate_fval=candidate_fval,
        candidate_index=candidate_index,
        rho=rho,
        is_accepted=is_accepted,
        old_state=state,
        suggestive_radius=None,
    )
    return out


def _accept_classic_line_search(
    subproblem_solution,
    state,
    history,
    batch_size,
    *,
    wrapped_criterion,
    min_improvement,
):
    """Do a line search acceptance step for a trustregion algorithm.

    Args:
        subproblem_solution (SubproblemResult): Result of the subproblem solution.
        state (State): Namedtuple containing the trustregion, criterion value of
            previously accepted point, indices of model points, etc.
        history (History): The tranquilo history.
        batch_size (int): Number of points to evaluate in parallel.
        sample_points (callable): Function that samples points from the trustregion.
        rng (np.random.Generator): Random number generator.
        wrapped_criterion (callable): The criterion function.
        min_improvement (float): Minimum improvement required to accept a point.

    Returns:
        AcceptanceResult: The acceptance result.

    """
    if batch_size == 1:
        # quick return when only one point should be evaluated
        return _accept_classic(
            subproblem_solution=subproblem_solution,
            state=state,
            history=history,
            wrapped_criterion=wrapped_criterion,
            min_improvement=min_improvement,
        )

    # Perform candidate and line search evaluations
    # ==================================================================================

    candidate_x = subproblem_solution.x
    candidate_index = history.add_xs(candidate_x)

    eval_info = {candidate_index: 1}

    candidate_on_border = _is_on_border(state.trustregion, x=candidate_x, rtol=1e-12)

    if candidate_on_border:
        # a better point is most likely outside the trustregion, so we perform a line
        # search outside of the trustregion
        alpha_grid = 2 ** np.arange(1, batch_size)

        new_xs = _sample_on_line(
            start_point=state.x, direction_point=candidate_x, alpha_grid=alpha_grid
        )

        new_indices = history.add_xs(new_xs)

        eval_info.update({i: 1 for i in new_indices})

    wrapped_criterion(eval_info)

    # Take acceptance decision: Compare function values from line search and candidate
    # ==================================================================================

    candidate_fval = np.mean(history.get_fvals(candidate_index))

    if candidate_on_border:
        new_fvals = history.get_fvals(new_indices)
        new_fvals = pd.Series({i: np.mean(fvals) for i, fvals in new_fvals.items()})
        new_fval_argmin = new_fvals.idxmin()

    if candidate_on_border and new_fvals.loc[new_fval_argmin] < candidate_fval:
        # a better point was found during the line search
        candidate_x = history.get_xs(new_fval_argmin)
        candidate_fval = new_fvals.loc[new_fval_argmin]
        candidate_index = new_fval_argmin

        actual_improvement = -(candidate_fval - state.fval)

        rho = None
        suggestive_radius = (3 / 4) * np.linalg.norm(candidate_x - state.x)

    else:
        # no better point was found during the line search
        actual_improvement = -(candidate_fval - state.fval)

        rho = calculate_rho(
            actual_improvement=actual_improvement,
            expected_improvement=subproblem_solution.expected_improvement,
        )
        suggestive_radius = None

    is_accepted = actual_improvement >= min_improvement

    out = _get_acceptance_result(
        candidate_x=candidate_x,
        candidate_fval=candidate_fval,
        candidate_index=candidate_index,
        rho=rho,
        is_accepted=is_accepted,
        old_state=state,
        suggestive_radius=suggestive_radius,
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
        n_evals=5,
    )
    return out


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
        suggestive_radius=None,
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
        suggestive_radius=None,
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
    suggestive_radius: float


def _get_acceptance_result(
    candidate_x,
    candidate_fval,
    candidate_index,
    rho,
    is_accepted,
    old_state,
    suggestive_radius,
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
        suggestive_radius=suggestive_radius,
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


def _is_on_border(trustregion, x, rtol):
    """Check if x is on the border of the trustregion.

    Args:
        trustregion (Region): Trustregion.
        x (np.ndarray): Point to check.
        rtol (float): Relative tolerance.

    Returns:
        bool: True if x is on the border of the trustregion.

    """
    if trustregion.shape == "sphere":
        candidate_on_border = _is_on_sphere_border(trustregion, x=x, rtol=rtol)
    else:
        candidate_on_border = _is_on_cube_border(trustregion, x=x, rtol=rtol)
    return candidate_on_border


def _is_on_sphere_border(trustregion, x, rtol):
    """Check if x is on the border of the trustregion sphere.

    Args:
        trustregion (Region): Spherical trustregion.
        x (np.ndarray): Point to check.
        rtol (float): Relative tolerance.

    Returns:
        bool: True if x is on the border of the trustregion sphere.

    """
    x_center_dist = np.linalg.norm(x - trustregion.center, ord=2)
    return np.isclose(x_center_dist, trustregion.radius, rtol=rtol)


def _is_on_cube_border(trustregion, x, rtol):
    """Check if x is on the border of the trustregion cube.

    Args:
        trustregion (Region): Spherical trustregion.
        x (np.ndarray): Point to check.
        rtol (float): Relative tolerance.

    Returns:
        bool: True if x is on the border of the trustregion cube.

    """
    cube_bounds = trustregion.cube_bounds
    is_on_lower_border = np.isclose(x, cube_bounds.lower, rtol=rtol).any()
    is_on_upper_border = np.isclose(x, cube_bounds.upper, rtol=rtol).any()
    return is_on_lower_border or is_on_upper_border


def _sample_on_line(start_point, direction_point, alpha_grid):
    """Sample points on a line defined by start_point and direction_point.

    Args:
        start_point (np.ndarray): Starting point of the line.
        direction_point (np.ndarray): Direction point.
        alpha_grid (np.ndarray): Grid of positions to sample points on the line. 0 will
            corresponds to start_point and 1 to direction_point. Values between 0 and 1
            will result in samples between start_point and direction_point, and values
            greater than 1 will result in samples on the line further away from
            start_point than the direction_point.

    Returns:
        np.ndarray: Sampled points.

    """
    xs = start_point + alpha_grid.reshape(-1, 1) * (direction_point - start_point)
    return xs
