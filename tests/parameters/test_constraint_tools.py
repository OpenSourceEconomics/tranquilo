import pytest
from estimagic.exceptions import InvalidParamsError
from estimagic.parameters.constraint_tools import check_constraints
from estimagic.parameters.constraint_tools import count_free_params


def test_count_free_params_no_constraints():
    params = {"a": 1, "b": 2, "c": [3, 3]}
    assert count_free_params(params) == 4


def test_count_free_params_with_constraints():
    params = {"a": 1, "b": 2, "c": [3, 3]}
    constraints = [{"selector": lambda x: x["c"], "type": "equality"}]
    assert count_free_params(params, constraints) == 3


def test_check_constraints():
    params = {"a": 1, "b": 2, "c": [3, 4]}
    constraints = [{"selector": lambda x: x["c"], "type": "equality"}]

    with pytest.raises(InvalidParamsError):
        check_constraints(params, constraints)
