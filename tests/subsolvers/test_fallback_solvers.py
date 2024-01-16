import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae
from tranquilo.models import ScalarModel

from tranquilo.subsolvers.fallback_subsolvers import (
    robust_cube_solver,
    robust_cube_solver_multistart,
    robust_sphere_solver_inscribed_cube,
    robust_sphere_solver_norm_constraint,
    robust_sphere_solver_reparametrized,
)

FALLBACK_SPHERE_SOLVERS = [
    robust_sphere_solver_inscribed_cube,
    robust_sphere_solver_reparametrized,
    robust_sphere_solver_norm_constraint,
]

FALLBACK_CUBE_SOLVERS = [
    robust_cube_solver,
    robust_cube_solver_multistart,
]


@pytest.fixture
def model():
    """Simple quadratic scalar model.

    - Minimum in cube is at (1, 1)
    - Minimum in sphere is at (1/sqrt(2), 1/sqrt(2))

    """
    return ScalarModel(
        intercept=0.0,
        linear_terms=-np.ones(2),
        square_terms=-np.eye(2),
    )


@pytest.mark.parametrize("solver", FALLBACK_SPHERE_SOLVERS)
def test_fallback_sphere_solver(solver, model):
    x_candidate = np.zeros(2)

    calculated = solver(model, x_candidate)
    expected = np.ones(2) / np.sqrt(2)

    aaae(calculated["x"], expected)


@pytest.mark.parametrize("solver", FALLBACK_CUBE_SOLVERS)
def test_fallback_cube_solver(solver, model):
    x_candidate = np.zeros(2)

    calculated = solver(model, x_candidate)
    expected = np.ones(2)

    aaae(calculated["x"], expected)
