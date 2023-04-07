import pytest
from tranquilo.utilities import propose_alternatives, get_rng
import numpy as np


def test_propose_alternatives():
    possibilities = ["scipy_lbfgsb", "scipy_slsqp", "nlopt_lbfgsb"]
    inputs = [["scipy_L-BFGS-B", 1], ["L-BFGS-B", 2]]
    expected = [["scipy_slsqp"], ["scipy_slsqp", "scipy_lbfgsb"]]
    for inp, exp in zip(inputs, expected):
        assert propose_alternatives(inp[0], possibilities, number=inp[1]) == exp


TEST_CASES = [
    0,
    1,
    10,
    1000000,
    None,
    np.random.default_rng(),
    np.random.Generator(np.random.MT19937()),
]


@pytest.mark.parametrize("seed", TEST_CASES)
def test_get_rng_correct_input(seed):
    rng = get_rng(seed)
    assert isinstance(rng, np.random.Generator)


TEST_CASES = [0.1, "a", object(), lambda x: x**2]


@pytest.mark.parametrize("seed", TEST_CASES)
def test_get_rng_wrong_input(seed):
    with pytest.raises(TypeError):
        get_rng(seed)
