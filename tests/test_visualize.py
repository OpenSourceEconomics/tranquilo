import pytest
from optimagic.optimization.optimize import minimize
from optimagic.benchmarking.get_benchmark_problems import get_benchmark_problems
from tranquilo.visualize import visualize_tranquilo


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
    fun = inputs["fun"]
    start_params = inputs["params"]
    for algorithm in ["tranquilo", "tranquilo_ls"]:
        results = {}
        for s, options in algo_options.items():
            results[s] = minimize(
                criterion=fun,
                params=start_params,
                algo_options=options,
                algorithm=algorithm,
            )
        cases.append(results)


skip_reason = (
    "History collection of tranquilo and tranquilo_ls is disabled in optimagic, but"
    "visualize_tranquilo requires a history."
)


@pytest.mark.skip(reason=skip_reason)
@pytest.mark.parametrize("results", cases)
def test_visualize_tranquilo(results):
    visualize_tranquilo(results, 5)
    for res in results.values():
        visualize_tranquilo(res, [1, 5])
