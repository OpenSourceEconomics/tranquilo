import matplotlib.pyplot as plt
import pytest
from estimagic import get_benchmark_problems
from estimagic.benchmarking.run_benchmark import run_benchmark
from estimagic.visualization.convergence_plot import _check_only_allowed_subset_provided
from estimagic.visualization.convergence_plot import convergence_plot

# integration test to make sure non default argument do not throw Errors
profile_options = [
    {"n_cols": 3},
    {"distance_measure": "parameter_distance"},
    {"monotone": False},
    {"normalize_distance": False},
    {"runtime_measure": "walltime"},
    {"stopping_criterion": None},
    {"stopping_criterion": "x"},
    {"stopping_criterion": "x_and_y"},
    {"stopping_criterion": "x_or_y"},
    {"x_precision": 1e-5},
    {"y_precision": 1e-5},
]


@pytest.mark.parametrize("options", profile_options)
def test_convergence_plot_options(options):
    problems = get_benchmark_problems("example")
    stop_after_10 = {
        "stopping_max_criterion_evaluations": 10,
        "stopping_max_iterations": 10,
    }
    optimizers = {
        "lbfgsb": {"algorithm": "scipy_lbfgsb", "algo_options": stop_after_10},
        "nm": {"algorithm": "scipy_neldermead", "algo_options": stop_after_10},
    }
    results = run_benchmark(
        problems,
        optimizers,
        n_cores=1,  # must be 1 for the test to work
        logging_directory="logging",
    )

    convergence_plot(
        problems=problems,
        results=results,
        problem_subset=["bard_good_start"],
        **options
    )
    plt.close()


def test_check_only_allowed_subset_provided_none():
    allowed = ["a", "b", "c"]
    _check_only_allowed_subset_provided(None, allowed, "name")


def test_check_only_allowed_subset_provided_all_included():
    allowed = ["a", "b", "c"]
    _check_only_allowed_subset_provided(["a", "b"], allowed, "name")


def test_check_only_allowed_subset_provided_missing():
    allowed = ["a", "b", "c"]
    with pytest.raises(ValueError):
        _check_only_allowed_subset_provided(["d"], allowed, "name")
