from collections import namedtuple

import numpy as np
import pytest
from tranquilo.acceptance_decision import (
    _accept_simple,
    _get_acceptance_result,
    calculate_rho,
    _generate_alpha_grid,
    _is_on_border,
    _is_on_cube_border,
    _is_on_sphere_border,
)
from tranquilo.history import History
from tranquilo.region import Region
from tranquilo.bounds import Bounds
from tranquilo.solve_subproblem import SubproblemResult
from numpy.testing import assert_array_equal

# ======================================================================================
# Fixtures
# ======================================================================================


@pytest.fixture()
def subproblem_solution():
    res = SubproblemResult(
        x=1 + np.arange(2.0),
        expected_improvement=1.0,
        n_iterations=1,
        success=True,
        x_unit=None,
        shape=None,
    )
    return res


# ======================================================================================
# Test accept_xxx
# ======================================================================================


trustregion = Region(center=np.zeros(2), radius=2.0)
State = namedtuple("State", "x trustregion fval index")
states = [  # we will parametrize over `states`
    State(np.arange(2.0), trustregion, 0.25, 0),  # better than candidate
    State(np.arange(2.0), trustregion, 1, 0),  # worse than candidate
]


@pytest.mark.parametrize("state", states)
def test_accept_simple(
    state,
    subproblem_solution,
):
    history = History(functype="scalar")

    idxs = history.add_xs(np.arange(10).reshape(5, 2))

    history.add_evals(idxs.repeat(2), np.arange(10))

    def wrapped_criterion(eval_info):
        indices = np.array(list(eval_info)).repeat(np.array(list(eval_info.values())))
        history.add_evals(indices, -indices)

    res_got = _accept_simple(
        subproblem_solution=subproblem_solution,
        state=state,
        history=history,
        wrapped_criterion=wrapped_criterion,
        min_improvement=0.0,
        n_evals=2,
    )

    assert res_got.accepted
    assert res_got.index == 5
    assert res_got.candidate_index == 5
    assert_array_equal(res_got.x, subproblem_solution.x)
    assert_array_equal(res_got.candidate_x, 1.0 + np.arange(2))


# ======================================================================================
# Test _get_acceptance_result
# ======================================================================================


def test_get_acceptance_result():
    candidate_x = 1 + np.arange(2)
    candidate_fval = 0
    candidate_index = 0
    rho = 1
    tr = Region(center=np.zeros(2), radius=2.0)
    old_state = namedtuple("State", "x fval index trustregion")(np.arange(2), 1, 1, tr)

    ar_when_accepted = _get_acceptance_result(
        candidate_x=candidate_x,
        candidate_fval=candidate_fval,
        candidate_index=candidate_index,
        rho=rho,
        old_state=old_state,
        is_accepted=True,
    )

    assert_array_equal(ar_when_accepted.x, candidate_x)
    assert ar_when_accepted.fval == candidate_fval
    assert ar_when_accepted.index == candidate_index
    assert ar_when_accepted.accepted is True
    assert ar_when_accepted.step_length == np.sqrt(2)
    assert ar_when_accepted.relative_step_length == np.sqrt(2) / 2

    ar_when_not_accepted = _get_acceptance_result(
        candidate_x=candidate_x,
        candidate_fval=candidate_fval,
        candidate_index=candidate_index,
        rho=rho,
        old_state=old_state,
        is_accepted=False,
    )

    assert_array_equal(ar_when_not_accepted.x, old_state.x)
    assert ar_when_not_accepted.fval == old_state.fval
    assert ar_when_not_accepted.index == old_state.index
    assert ar_when_not_accepted.accepted is False
    assert ar_when_not_accepted.step_length == 0
    assert ar_when_not_accepted.relative_step_length == 0


# ======================================================================================
# Test calculate_rho
# ======================================================================================


CASES = [
    (0, 0, -np.inf),
    (-1, 0, -np.inf),
    (1, 0, np.inf),
    (0, 1, 0),
    (1, 2, 1 / 2),
]


@pytest.mark.parametrize("actual_improvement, expected_improvement, expected", CASES)
def test_calculate_rho(actual_improvement, expected_improvement, expected):
    rho = calculate_rho(actual_improvement, expected_improvement)
    assert rho == expected


# ======================================================================================
# Test _generate_alpha_grid
# ======================================================================================

CASES = zip(
    [1, 2, 4, 6],
    [np.array([]), np.array([2]), np.array([2, 4, 8]), np.array([2, 4, 8])],
)


@pytest.mark.parametrize("batch_size, expected", CASES)
def test_generate_alpha_grid(batch_size, expected):
    alpha_grid = _generate_alpha_grid(batch_size)
    assert_array_equal(alpha_grid, expected)


# ======================================================================================
# Test border check functions
# ======================================================================================

CASES = [
    (np.array([0, 0]), 1, True),
    (np.array([0, 0]), 0.5, False),
    (np.array([0, 1]), 0.0, True),
    (np.array([0, 0.9]), 0.1, True),
    (np.array([0, 0.9]), 0.09, False),
]


@pytest.mark.parametrize("x, rtol, expected", CASES)
def test_is_on_sphere_border(x, rtol, expected):
    region = Region(center=np.zeros(2), radius=1.0)
    assert _is_on_sphere_border(region, x, rtol) == expected


CASES = [
    (np.ones(2), 0, True),
    (0.9 * np.ones(2), 0.1, True),
    (0.8 * np.ones(2), 0.1, False),
]


@pytest.mark.parametrize("x, rtol, expected", CASES)
def test_is_on_cube_border(x, rtol, expected):
    bounds = Bounds(lower=-np.ones(2), upper=np.ones(2))
    region = Region(center=np.zeros(2), radius=2.0, bounds=bounds)
    assert _is_on_cube_border(region, x, rtol) == expected


def test_is_on_border_sphere():
    region = Region(center=np.zeros(2), radius=1.0)
    assert _is_on_border(region, np.array([0, 0.9]), 0.1)
    assert not _is_on_border(region, np.array([0, 0.9]), 0.09)


def test_is_on_border_cube():
    bounds = Bounds(lower=-np.ones(2), upper=np.ones(2))
    region = Region(center=np.zeros(2), radius=2.0, bounds=bounds)
    assert _is_on_border(region, 0.9 * np.ones(2), 0.1)
    assert not _is_on_border(region, 0.8 * np.ones(2), 0.1)
