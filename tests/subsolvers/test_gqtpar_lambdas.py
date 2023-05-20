from estimagic.optimization.optimize import minimize
from estimagic.benchmarking.get_benchmark_problems import get_benchmark_problems
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


def test_gqtpar_lambdas():
    algo_options = {
        "disable_convergence": True,
        "stopping_max_iterations": 30,
        "sample_filter": "keep_all",
        "sampler": "random_hull",
        "subsolver_options": {"k_hard": 0.001, "k_easy": 0.001},
    }
    problem_info = get_benchmark_problems("more_wild")["freudenstein_roth_good_start"]

    minimize(
        criterion=problem_info["inputs"]["criterion"],
        params=problem_info["inputs"]["params"],
        algo_options=algo_options,
        algorithm=tranquilo,
    )
