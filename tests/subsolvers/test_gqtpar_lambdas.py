import pytest
from tranquilo.config import IS_OPTIMAGIC_INSTALLED

if IS_OPTIMAGIC_INSTALLED:
    from optimagic.optimization.optimize import minimize
    from optimagic.benchmarking.get_benchmark_problems import get_benchmark_problems


@pytest.mark.skipif(not IS_OPTIMAGIC_INSTALLED, reason="optimagic is not installed.")
def test_gqtpar_lambdas():
    algo_options = {
        "disable_convergence": True,
        "stopping_maxiter": 30,
        "sample_filter": "keep_all",
        "sampler": "random_hull",
        "subsolver_options": {"k_hard": 0.001, "k_easy": 0.001},
    }
    problem_info = get_benchmark_problems("more_wild")["freudenstein_roth_good_start"]

    minimize(
        fun=problem_info["inputs"]["fun"],
        params=problem_info["inputs"]["params"],
        algo_options=algo_options,
        algorithm="tranquilo",
    )
