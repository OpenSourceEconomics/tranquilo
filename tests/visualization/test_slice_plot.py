import itertools

import numpy as np
import pandas as pd
import pytest
from estimagic.visualization.slice_plot import slice_plot


@pytest.fixture
def problem():
    params = pd.DataFrame(
        np.zeros(4),
        columns=["value"],
    )

    params["lower_bound"] = -5
    params["upper_bound"] = 5

    out = {
        "criterion": lambda params: (params["value"] ** 2).sum(),  # sphere
        "params": params,
    }
    return out


@pytest.mark.parametrize(
    "n_gridpoints, n_random_values, plots_per_row, return_dict",
    itertools.product([21, 41], [2, 3], [1, 2], [False, True]),
)
def test_slice_plot(problem, n_gridpoints, n_random_values, plots_per_row, return_dict):

    slice_plot(
        criterion=problem["criterion"],
        params=problem["params"],
        n_gridpoints=n_gridpoints,
        n_random_values=n_random_values,
        return_dict=return_dict,
        plots_per_row=plots_per_row,
    )
