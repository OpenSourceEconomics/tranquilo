import itertools

import numpy as np
import pandas as pd
import pytest
from estimagic.decorators import AlgoInfo
from estimagic.examples.criterion_functions import (
    sos_criterion_and_gradient,
    sos_dict_criterion,
    sos_dict_criterion_with_pd_objects,
    sos_gradient,
    sos_pandas_gradient,
    sos_scalar_criterion,
)
from estimagic.optimization.internal_criterion_template import (
    internal_criterion_and_derivative_template,
)
from estimagic.parameters.conversion import get_converter
from numpy.testing import assert_array_almost_equal as aaae


def reparametrize_from_internal(x):
    res = pd.DataFrame()
    res["value"] = x
    return res


def convert_derivative(external_derivative, internal_values):  # noqa: ARG001
    return external_derivative


@pytest.fixture()
def base_inputs():
    x = np.arange(5).astype(float)
    params = pd.DataFrame(data=np.arange(5).reshape(-1, 1), columns=["value"])
    inputs = {
        "x": x,
        "params": params,
        "algo_info": AlgoInfo(
            name="my_algorithm",
            primary_criterion_entry="value",
            needs_scaling=False,
            is_available=True,
            parallelizes=False,
            arguments=[],
        ),
        "error_handling": "raise",
        "numdiff_options": {},
        "logging": False,
        "db_kwargs": {"database": False, "fast_logging": False, "path": "logging.db"},
        "error_penalty_func": None,
        "fixed_log_data": {"stage": "optimization", "substage": 0},
    }
    return inputs


directions = ["maximize", "minimize"]
crits = [sos_dict_criterion, sos_dict_criterion_with_pd_objects, sos_scalar_criterion]
derivs = [sos_gradient, sos_pandas_gradient, None]
crits_and_derivs = [sos_criterion_and_gradient, None]

test_cases = list(itertools.product(directions, crits, derivs, crits_and_derivs))


@pytest.mark.parametrize("direction, crit, deriv, crit_and_deriv", test_cases)
def test_criterion_and_derivative_template(
    base_inputs, direction, crit, deriv, crit_and_deriv
):
    converter, _ = get_converter(
        params=base_inputs["params"],
        constraints=None,
        lower_bounds=None,
        upper_bounds=None,
        func_eval=crit(base_inputs["params"]),
        primary_key="value",
        scaling=False,
        scaling_options=None,
        derivative_eval=None,
    )
    inputs = {k: v for k, v in base_inputs.items() if k != "params"}
    inputs["converter"] = converter

    crit = crit if (deriv, crit_and_deriv) == (None, None) else crit

    inputs["criterion"] = crit
    inputs["derivative"] = deriv
    inputs["criterion_and_derivative"] = crit_and_deriv
    inputs["direction"] = direction

    calc_criterion, calc_derivative = internal_criterion_and_derivative_template(
        task="criterion_and_derivative", **inputs
    )

    calc_criterion2 = internal_criterion_and_derivative_template(
        task="criterion", **inputs
    )

    calc_derivative2 = internal_criterion_and_derivative_template(
        task="derivative", **inputs
    )

    if direction == "minimize":
        for c in calc_criterion, calc_criterion2:
            assert c == 30

        for d in calc_derivative, calc_derivative2:
            aaae(d, 2 * np.arange(5))
    else:
        for c in calc_criterion, calc_criterion2:
            assert c == -30

        for d in calc_derivative, calc_derivative2:
            aaae(d, -2 * np.arange(5))


@pytest.mark.parametrize("direction", directions)
def test_internal_criterion_with_penalty(base_inputs, direction):
    converter, _ = get_converter(
        params=base_inputs["params"],
        constraints=None,
        lower_bounds=None,
        upper_bounds=None,
        func_eval=sos_scalar_criterion(base_inputs["params"]),
        primary_key="value",
        scaling=False,
        scaling_options=None,
        derivative_eval=None,
    )
    inputs = {k: v for k, v in base_inputs.items() if k != "params"}

    inputs["converter"] = converter

    def raising_crit_and_deriv(x):  # noqa: ARG001
        raise ValueError()

    inputs["error_handling"] = "continue"
    inputs["x"] = inputs["x"] + 10
    inputs["criterion"] = sos_scalar_criterion
    inputs["derivative"] = sos_gradient
    inputs["criterion_and_derivative"] = raising_crit_and_deriv
    inputs["direction"] = direction
    inputs["error_penalty_func"] = lambda x, task: (42, 52)  # noqa: ARG005

    with pytest.warns():
        calc_criterion, calc_derivative = internal_criterion_and_derivative_template(
            task="criterion_and_derivative", **inputs
        )

    expected_crit = 42
    expected_grad = 52

    if direction == "minimize":
        assert calc_criterion == expected_crit
        aaae(calc_derivative, expected_grad)

    else:
        assert calc_criterion == -expected_crit
        aaae(calc_derivative, -expected_grad)
