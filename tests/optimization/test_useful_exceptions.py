import numpy as np
import pandas as pd
import pytest
from estimagic.exceptions import InvalidFunctionError
from estimagic.exceptions import InvalidKwargsError
from estimagic.exceptions import UserFunctionRuntimeError
from estimagic.optimization.optimize import minimize


def test_missing_criterion_kwargs():
    def f(params, bla, blubb):  # noqa: ARG001
        return (params["value"].to_numpy() ** 2).sum()

    params = pd.DataFrame(np.ones((3, 1)), columns=["value"])

    with pytest.raises(InvalidKwargsError):
        minimize(f, params, "scipy_lbfgsb", criterion_kwargs={"bla": 3})


def test_missing_derivative_kwargs():
    def f(params):
        return (params["value"].to_numpy() ** 2).sum()

    def grad(params, bla, blubb):  # noqa: ARG001
        return params["value"].to_numpy() * 2

    params = pd.DataFrame(np.ones((3, 1)), columns=["value"])

    with pytest.raises(InvalidKwargsError):
        minimize(
            f, params, "scipy_lbfgsb", derivative=grad, derivative_kwargs={"bla": 3}
        )


def test_missing_criterion_and_derivative_kwargs():
    def f(params):
        return (params["value"].to_numpy() ** 2).sum()

    def f_and_grad(params, bla, blubb):  # noqa: ARG001
        return f(params), params["value"].to_numpy() * 2

    params = pd.DataFrame(np.ones((3, 1)), columns=["value"])

    with pytest.raises(InvalidKwargsError):
        minimize(
            f,
            params,
            "scipy_lbfgsb",
            criterion_and_derivative=f_and_grad,
            criterion_and_derivative_kwargs={"bla": 3},
        )


def test_typo_in_criterion_kwarg():
    def f(params, bla, foo):  # noqa: ARG001
        return (params["value"].to_numpy() ** 2).sum()

    params = pd.DataFrame(np.ones((3, 1)), columns=["value"])

    snippet = "Did you mean"
    with pytest.raises(InvalidKwargsError, match=snippet):
        minimize(f, params, "scipy_lbfgsb", criterion_kwargs={"bla": 3, "foa": 4})


def test_criterion_with_runtime_error_derivative_free():
    def f(params):
        x = params["value"].to_numpy()
        if x.sum() < 1:
            raise RuntimeError("Great error message")

        return x @ x

    params = pd.DataFrame(np.full((3, 1), 10), columns=["value"])
    snippet = "when evaluating criterion during optimization"
    with pytest.raises(UserFunctionRuntimeError, match=snippet):
        minimize(f, params, "scipy_neldermead")


def test_criterion_with_runtime_error_during_numerical_derivative():
    def f(params):
        x = params["value"].to_numpy()
        if (x != 1).any():
            raise RuntimeError("Great error message")

        return x @ x

    params = pd.DataFrame(np.ones((3, 1)), columns=["value"])
    snippet = "evaluating criterion to calculate a numerical derivative"
    with pytest.raises(UserFunctionRuntimeError, match=snippet):
        minimize(f, params, "scipy_lbfgsb")


def test_criterion_fails_at_start_values():
    def just_fail(params):  # noqa: ARG001
        raise RuntimeError()

    params = pd.DataFrame(np.ones((3, 1)), columns=["value"])
    snippet = "Error while evaluating criterion at start params."
    with pytest.raises(InvalidFunctionError, match=snippet):
        minimize(just_fail, params, "scipy_lbfgsb")
