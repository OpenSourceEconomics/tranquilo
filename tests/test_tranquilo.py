import itertools
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from tranquilo.config import IS_OPTIMAGIC_INSTALLED
from tranquilo.tranquilo import _tranquilo

if IS_OPTIMAGIC_INSTALLED:
    from optimagic import mark
    from optimagic.optimization.optimize import minimize


tranquilo = partial(
    _tranquilo,
    functype="scalar",
)

tranquilo_ls = partial(
    _tranquilo,
    functype="least_squares",
)


# ======================================================================================
# Test tranquilo end-to-end
# ======================================================================================


def _product(sample_filter, model_fitter, model_type):
    # is used to create products of test cases
    return list(itertools.product(sample_filter, model_fitter, model_type))


# ======================================================================================
# Scalar Tranquilo
# ======================================================================================

TEST_CASES = {
    "ols": {
        "sample_filter": ["discard_all", "keep_all"],
        "model_fitter": ["ols"],
        "model_type": ["quadratic"],
    },
    "ols_keep_all": {
        "sample_filter": ["keep_all"],
        "model_fitter": ["ols"],
        "model_type": ["quadratic"],
    },
    "pounders_discard_all": {
        "sample_filter": ["discard_all"],
        "model_fitter": ["powell"],
        "model_type": ["quadratic"],
    },
    "pounders_keep_all": {
        "sample_filter": ["keep_all"],
        "model_fitter": ["powell"],
        "model_type": ["quadratic"],
    },
}

TEST_CASES = [_product(**kwargs) for kwargs in TEST_CASES.values()]
TEST_CASES = itertools.chain.from_iterable(TEST_CASES)


@pytest.mark.parametrize("sample_filter, model_fitter, model_type", TEST_CASES)
def test_internal_tranquilo_scalar_sphere_defaults(
    sample_filter,
    model_fitter,
    model_type,
):
    res = tranquilo(
        fun=lambda x: x @ x,
        x=np.arange(4),
        sample_filter=sample_filter,
        model_fitter=model_fitter,
        model_type=model_type,
    )
    aaae(res["solution_x"], np.zeros(4), decimal=4)


# ======================================================================================
# Imprecise options for scalar tranquilo
# ======================================================================================

TEST_CASES = {
    "ls_keep": {
        "sample_filter": ["keep_all"],
        "model_fitter": ["ols"],
        "model_type": ["quadratic"],
    },
    "pounders_discard_all": {
        "sample_filter": ["discard_all"],
        "model_fitter": ["powell"],
        "model_type": ["quadratic"],
    },
}

TEST_CASES = [_product(**kwargs) for kwargs in TEST_CASES.values()]
TEST_CASES = itertools.chain.from_iterable(TEST_CASES)


@pytest.mark.parametrize("sample_filter, model_fitter, model_type", TEST_CASES)
def test_internal_tranquilo_scalar_sphere_imprecise_defaults(
    sample_filter,
    model_fitter,
    model_type,
):
    res = tranquilo(
        fun=lambda x: x @ x,
        x=np.arange(4),
        sample_filter=sample_filter,
        model_fitter=model_fitter,
        model_type=model_type,
    )
    aaae(res["solution_x"], np.zeros(4), decimal=3)


# ======================================================================================
# External
# ======================================================================================


@pytest.mark.skipif(not IS_OPTIMAGIC_INSTALLED, reason="optimagic is not installed.")
def test_external_tranquilo_scalar_sphere_defaults():
    res = minimize(
        fun=lambda x: x @ x,
        params=np.arange(4),
        algorithm="tranquilo",
    )

    aaae(res.params, np.zeros(4), decimal=4)


# ======================================================================================
# Least-squares Tranquilo
# ======================================================================================


TEST_CASES = {
    "ols": {
        "sample_filter": ["keep_all", "discard_all"],
        "model_fitter": ["ols"],
        "model_type": ["linear"],
    },
    "tranquilo": {
        "sample_filter": ["keep_all", "discard_all"],
        "model_fitter": ["tranquilo"],
        "model_type": ["linear"],
    },
}

TEST_CASES = [_product(**kwargs) for kwargs in TEST_CASES.values()]
TEST_CASES = itertools.chain.from_iterable(TEST_CASES)


@pytest.mark.parametrize("sample_filter, model_fitter, model_type", TEST_CASES)
def test_internal_tranquilo_ls_sphere_defaults(
    sample_filter,
    model_fitter,
    model_type,
):
    res = tranquilo_ls(
        fun=lambda x: x,
        x=np.arange(5),
        sample_filter=sample_filter,
        model_fitter=model_fitter,
        model_type=model_type,
    )
    aaae(res["solution_x"], np.zeros(5), decimal=5)


# ======================================================================================
# External
# ======================================================================================


@pytest.mark.skipif(not IS_OPTIMAGIC_INSTALLED, reason="optimagic is not installed.")
def test_external_tranquilo_ls_sphere_defaults():
    res = minimize(
        fun=mark.least_squares(lambda x: x),
        params=np.arange(5),
        algorithm="tranquilo_ls",
    )

    aaae(res.params, np.zeros(5), decimal=5)


# ======================================================================================
# Noisy case
# ======================================================================================

if IS_OPTIMAGIC_INSTALLED:
    # Has to be defined here to avoid import errors when optimagic is not installed
    ALGORITHM_AND_CRITERION = [
        ("tranquilo", mark.scalar(lambda x: x @ x)),
        ("tranquilo_ls", mark.least_squares(lambda x: x)),
    ]
else:
    ALGORITHM_AND_CRITERION = []


@pytest.mark.skipif(not IS_OPTIMAGIC_INSTALLED, reason="optimagic is not installed.")
@pytest.mark.parametrize("algorithm, criterion", ALGORITHM_AND_CRITERION)
def test_tranquilo_with_noise_handling_and_deterministic_function(algorithm, criterion):
    res = minimize(
        fun=criterion,
        params=np.arange(5),
        algorithm=algorithm,
        algo_options={"noisy": True},
    )

    aaae(res.params, np.zeros(5), decimal=3)


@pytest.mark.skipif(not IS_OPTIMAGIC_INSTALLED, reason="optimagic is not installed.")
@pytest.mark.slow()
def test_tranquilo_ls_with_noise_handling_and_noisy_function():
    rng = np.random.default_rng(123)

    @mark.least_squares
    def _f(x):
        x_n = x + rng.normal(0, 0.05, size=x.shape)
        return x_n

    res = minimize(
        fun=_f,
        params=np.ones(3),
        algorithm="tranquilo_ls",
        algo_options={"noisy": True, "n_evals_per_point": 10},
    )

    aaae(res.params, np.zeros(3), decimal=1)


# ======================================================================================
# Bounded case
# ======================================================================================


@pytest.mark.skipif(not IS_OPTIMAGIC_INSTALLED, reason="optimagic is not installed.")
@pytest.mark.parametrize("algorithm, criterion", ALGORITHM_AND_CRITERION)
def test_tranquilo_with_binding_bounds(algorithm, criterion):
    res = minimize(
        fun=criterion,
        params=np.array([3, 2, -3]),
        bounds=[(1, np.inf), (-np.inf, np.inf), (-np.inf, -1)],
        algorithm=algorithm,
        collect_history=True,
        skip_checks=True,
    )
    assert res.success in [True, None]
    aaae(res.params, np.array([1, 0, -1]), decimal=3)
