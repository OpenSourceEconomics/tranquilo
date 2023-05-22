import pytest
from estimagic.optimization.optimize import minimize
from estimagic.benchmarking.get_benchmark_problems import get_benchmark_problems
from tranquilo.visualize import visualize_tranquilo
from tranquilo.tranquilo import _tranquilo
from estimagic.decorators import mark_minimizer
from functools import partial


tranquilo = mark_minimizer(
    func=partial(_tranquilo, functype="scalar"),
    name="tranquilo",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=True,
    is_global=False,
)


tranquilo_ls = mark_minimizer(
    func=partial(_tranquilo, functype="least_squares"),
    primary_criterion_entry="root_contributions",
    name="tranquilo_ls",
    needs_scaling=True,
    is_available=True,
    is_global=False,
)

cases = []
algo_options = {
    "random_hull": {
        "sampler": "random_hull",
        "sphere_subsolver": "gqtpar_fast",
        "sample_filter": "keep_all",
        "stopping_max_iterations": 10,
    },
    "optimal_hull": {
        "sampler": "optimal_hull",
        "sphere_subsolver": "gqtpar_fast",
        "sample_filter": "keep_all",
        "stopping_max_iterations": 10,
    },
}
for problem in ["rosenbrock_good_start", "watson_6_good_start"]:
    inputs = get_benchmark_problems("more_wild")[problem]["inputs"]
    criterion = inputs["criterion"]
    start_params = inputs["params"]
    for algorithm in [tranquilo, tranquilo_ls]:
        results = {}
        for s, options in algo_options.items():
            results[s] = minimize(
                criterion=criterion,
                params=start_params,
                algo_options=options,
                algorithm=algorithm,
            )
        cases.append(results)


@pytest.mark.parametrize("results", cases)
def test_visualize_tranquilo(results):
    visualize_tranquilo(results, 5)
    for res in results.values():
        visualize_tranquilo(res, [1, 5])
