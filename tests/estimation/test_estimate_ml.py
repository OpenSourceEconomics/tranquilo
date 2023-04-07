import itertools

import numpy as np
import pandas as pd
import pytest
import scipy as sp
import statsmodels.api as sm
from estimagic.estimation.estimate_ml import estimate_ml
from estimagic.examples.logit import logit_derivative
from estimagic.examples.logit import logit_hessian
from estimagic.examples.logit import logit_loglike
from estimagic.examples.logit import logit_loglike_and_derivative as llad
from scipy.stats import multivariate_normal
from statsmodels.base.model import GenericLikelihoodModel


def aaae(obj1, obj2, decimal=3):
    arr1 = np.asarray(obj1)
    arr2 = np.asarray(obj2)
    np.testing.assert_array_almost_equal(arr1, arr2, decimal=decimal)


# ==================================================================================
# Test case with constraints using multivariate Normal model
# ==================================================================================


def multivariate_normal_loglike(params, data):
    mean = params["mean"]
    cov = params["cov"]
    mn = multivariate_normal(mean=mean, cov=cov)
    contributions = mn.logpdf(data)
    return {
        "contributions": contributions,
        "value": contributions.sum(),
    }


def test_estimate_ml_with_constraints():

    # true parameters
    true_mean = np.arange(1, 4)
    true_cov = np.diag(np.arange(1, 4))

    # simulate 10.000 random samples
    mn = multivariate_normal(mean=true_mean, cov=true_cov)
    data = mn.rvs(size=10_000)

    loglike_kwargs = {"data": data}

    params = {"mean": np.ones(3), "cov": np.diag(np.ones(3))}

    constraints = [
        {"type": "fixed", "selector": lambda p: p["mean"][0]},
        {"type": "covariance", "selector": lambda p: p["cov"][np.tril_indices(3)]},
    ]

    results = estimate_ml(
        loglike=multivariate_normal_loglike,
        params=params,
        loglike_kwargs=loglike_kwargs,
        optimize_options="scipy_lbfgsb",
        constraints=constraints,
    )

    aaae(results.params["mean"], true_mean, decimal=1)
    aaae(results.params["cov"], true_cov, decimal=1)


# ======================================================================================
# Test case using Logit model
# ======================================================================================


@pytest.fixture
def logit_np_inputs():
    spector_data = sm.datasets.spector.load_pandas()
    spector_data.exog = sm.add_constant(spector_data.exog)
    x_df = sm.add_constant(spector_data.exog)
    out = {
        "y": spector_data.endog,
        "x": x_df.to_numpy(),
        "params": np.array([-10, 2, 0.2, 2]),
    }
    return out


@pytest.fixture
def fitted_logit_model(logit_object):
    """We need to use a generic model class to access all standard errors etc."""

    class GenericLogit(GenericLikelihoodModel):
        def nloglikeobs(self, params, *args, **kwargs):
            return -logit_object.loglikeobs(params, *args, **kwargs)

    generic_logit = GenericLogit(logit_object.endog, logit_object.exog)
    return generic_logit.fit()


def logit_jacobian(params, y, x):
    return logit_derivative(params, y, x)["contributions"]


def logit_loglike_and_derivative(params, y, x):
    loglike, loglike_derivative = llad(params, y, x)
    return loglike, loglike_derivative["value"]


test_cases = list(
    itertools.product(
        [
            {"algorithm": "scipy_lbfgsb"},
            "scipy_lbfgsb",
            {
                "algorithm": "scipy_lbfgsb",
                "criterion_and_derivative": logit_loglike_and_derivative,
            },
        ],  # optimize_options
        [None, logit_jacobian, False],  # jacobian
        [None, logit_hessian, False],  # hessian
    )
)


@pytest.mark.parametrize("optimize_options, jacobian, hessian", test_cases)
def test_estimate_ml_with_logit_no_constraints(
    fitted_logit_model,
    logit_np_inputs,
    optimize_options,
    jacobian,
    hessian,
):
    """
    Test that estimate_ml computes correct params and covariances under different
    scenarios.
    """

    if jacobian is False and hessian is False:
        pytest.xfail("jacobian and hessian cannot both be False.")

    # ==================================================================================
    # estimate
    # ==================================================================================

    kwargs = {"y": logit_np_inputs["y"], "x": logit_np_inputs["x"]}

    if "criterion_and_derivative" in optimize_options:
        optimize_options["criterion_and_derivative_kwargs"] = kwargs

    got = estimate_ml(
        loglike=logit_loglike,
        params=logit_np_inputs["params"],
        loglike_kwargs=kwargs,
        optimize_options=optimize_options,
        jacobian=jacobian,
        jacobian_kwargs=kwargs,
        hessian=hessian,
        hessian_kwargs=kwargs,
    )

    # ==================================================================================
    # test
    # ==================================================================================

    exp = fitted_logit_model

    if jacobian is not False and hessian is not False:
        methods = ["jacobian", "hessian", "robust"]
    elif jacobian is not False:
        methods = ["jacobian"]
    elif hessian is not False:
        methods = ["hessian"]

    statsmodels_suffix_map = {
        "jacobian": "jac",
        "hessian": "",
        "robust": "jhj",
    }

    # compare estimated parameters
    aaae(got.params, exp.params, decimal=4)

    for method in methods:

        # compare estimated standard errors
        exp_se = getattr(exp, f"bse{statsmodels_suffix_map[method]}")
        aaae(got.se(method=method), exp_se, decimal=3)

        # compare estimated confidence interval
        if method == "hessian":
            lower, upper = got.ci(method=method)
            exp_lower = exp.conf_int().T[0]
            exp_upper = exp.conf_int().T[1]
            aaae(lower, exp_lower, decimal=3)
            aaae(upper, exp_upper, decimal=3)

        # compare covariance
        if method == "hessian":
            aaae(got.cov(method=method), exp.cov_params(), decimal=3)
        elif method == "robust":
            aaae(got.cov(method=method), exp.covjhj, decimal=2)
        elif method == "jacobian":
            aaae(got.cov(method=method), exp.covjac, decimal=4)

        summary = got.summary(method=method)

        aaae(summary["value"], exp.params, decimal=4)
        aaae(summary["standard_error"], got.se(method=method))
        lower, upper = got.ci(method=method)
        aaae(summary["ci_lower"], lower)
        aaae(summary["ci_upper"], upper)
        aaae(summary["p_value"], got.p_values(method=method))

    if "jacobian" in methods:
        aaae(got._se, got.se())
        aaae(got._ci[0], got.ci()[0])
        aaae(got._ci[1], got.ci()[1])
        aaae(got._p_values, got.p_values())


def test_estimate_ml_optimize_options_false(fitted_logit_model, logit_inputs):
    """Test that estimate_ml computes correct covariances given correct params."""

    kwargs = {"y": logit_inputs["y"], "x": logit_inputs["x"]}

    params = pd.DataFrame({"value": fitted_logit_model.params})

    got = estimate_ml(
        loglike=logit_loglike,
        params=params,
        loglike_kwargs=kwargs,
        optimize_options=False,
    )

    summary = got.summary()

    # compare estimated parameters
    aaae(summary["value"], fitted_logit_model.params, decimal=4)

    # compare estimated standard errors
    aaae(summary["standard_error"], fitted_logit_model.bsejac, decimal=3)

    # compare covariance (if not robust case)
    aaae(got.cov(method="jacobian"), fitted_logit_model.covjac, decimal=4)


# ======================================================================================
# Univariate normal case using dict params
# ======================================================================================


def normal_loglike(params, y):
    contribs = sp.stats.norm.logpdf(y, loc=params["mean"], scale=params["sd"])
    return {
        "value": contribs.sum(),
        "contributions": np.array(contribs),
    }


@pytest.fixture
def normal_inputs():
    true = {
        "mean": 1.0,
        "sd": 1.0,
    }
    y = np.random.normal(loc=true["mean"], scale=true["sd"], size=10_000)
    return {"true": true, "y": y}


def test_estimate_ml_general_pytree(normal_inputs):

    # ==================================================================================
    # estimate
    # ==================================================================================

    kwargs = {"y": normal_inputs["y"]}

    start_params = {"mean": 5, "sd": 3}

    got = estimate_ml(
        loglike=normal_loglike,
        params=start_params,
        loglike_kwargs=kwargs,
        optimize_options="scipy_lbfgsb",
        lower_bounds={"sd": 0.0001},
        jacobian_kwargs=kwargs,
        constraints=[{"selector": lambda p: p["sd"], "type": "sdcorr"}],
    )

    # ==================================================================================
    # test
    # ==================================================================================

    true = normal_inputs["true"]

    assert (
        np.abs(true["mean"] - got.summary(method="jacobian")["mean"]["value"][0]) < 1e-1
    )
    assert np.abs(true["sd"] - got.summary(method="jacobian")["sd"]["value"][0]) < 1e-1


def test_to_pickle(normal_inputs, tmp_path):
    kwargs = {"y": normal_inputs["y"]}

    start_params = {"mean": 5, "sd": 3}

    got = estimate_ml(
        loglike=normal_loglike,
        params=start_params,
        loglike_kwargs=kwargs,
        optimize_options="scipy_lbfgsb",
        lower_bounds={"sd": 0.0001},
        jacobian_kwargs=kwargs,
        constraints=[{"selector": lambda p: p["sd"], "type": "sdcorr"}],
    )

    got.to_pickle(tmp_path / "bla.pkl")
