from itertools import product

import numpy as np
import pytest
from tranquilo.exploration_sample import draw_exploration_sample
from numpy.testing import assert_array_almost_equal as aaae


dim = 2
distributions = ["uniform", "triangular"]
rules = ["sobol", "halton", "latin_hypercube", "random"]
lower = [np.zeros(dim), np.ones(dim) * 0.5, -np.ones(dim)]
upper = [np.ones(dim), np.ones(dim) * 0.75, np.ones(dim) * 2]
test_cases = list(product(distributions, rules, lower, upper))


@pytest.mark.parametrize("dist, rule, lower, upper", test_cases)
def test_draw_exploration_sample(dist, rule, lower, upper):
    results = []

    for _ in range(2):
        results.append(
            draw_exploration_sample(
                x=np.ones_like(lower) * 0.5,
                lower=lower,
                upper=upper,
                n_samples=3,
                sampling_distribution=dist,
                sampling_method=rule,
                seed=1234,
            )
        )

    aaae(results[0], results[1])
    calculated = results[0]
    assert calculated.shape == (3, 2)
