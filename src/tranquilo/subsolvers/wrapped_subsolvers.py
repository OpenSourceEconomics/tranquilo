from functools import partial, wraps

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from tranquilo.exploration_sample import draw_exploration_sample


def add_fallback_to_subproblem_solver(solver, fallback):
    fallback_options = {
        "slsqp_sphere": slsqp_sphere,
        "lbfgsb_sphere": lbfgsb_sphere,
        "lbfgsb_sphere_reparametrized": lbfgsb_sphere_reparametrized,
    }

    if fallback not in fallback_options:
        raise ValueError(
            f"Unknown fallback solver: {fallback}. Must be in {list(fallback_options)}"
        )

    fallback_solver = fallback_options[fallback]

    @wraps(solver)
    def wrapped_solver(model, x_candidate, lower_bounds, upper_bounds, **kwargs):
        try:
            result = solver(
                model=model,
                x_candidate=x_candidate,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                **kwargs,
            )
        except Exception:
            result = fallback_solver(
                model=model,
                x_candidate=x_candidate,
            )
        return result

    return wrapped_solver


def solve_multistart(model, x_candidate, lower_bounds, upper_bounds):
    np.random.seed(12345)
    start_values = draw_exploration_sample(
        x=x_candidate,
        lower=lower_bounds,
        upper=upper_bounds,
        n_samples=100,
        sampling_distribution="uniform",
        sampling_method="sobol",
        seed=1234,
    )

    def crit(x):
        return model.predict(x)

    bounds = Bounds(lower_bounds, upper_bounds)

    best_crit = np.inf
    accepted_x = None
    critvals = []
    for x in start_values:
        res = minimize(
            crit,
            x,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if res.fun <= best_crit:
            accepted_x = res.x
        critvals.append(res.fun)

    return {
        "x": accepted_x,
        "std": np.std(critvals),
        "n_iterations": None,
        "success": None,
    }


def lbfgsb_sphere_reparametrized(model, x_candidate):
    def crit(x):
        x_norm = np.linalg.norm(x)
        if x_norm > 1:
            x_tilde = x / x_norm
        else:
            x_tilde = x
        return model.predict(x_tilde)

    lower_bounds = -np.ones(len(x_candidate))
    upper_bounds = np.ones(len(x_candidate))

    res = minimize(
        crit,
        x_candidate,
        method="L-BFGS-B",
        bounds=Bounds(lower_bounds, upper_bounds),
        options={
            "maxiter": len(x_candidate),
        },
    )

    solution_norm = np.linalg.norm(res.x)

    if solution_norm <= 1:
        _minimizer = res.x
        _criterion = res.fun
    else:
        _minimizer = _project_onto_unit_sphere(res.x)
        _criterion = crit(_minimizer)

    return {
        "x": _minimizer,
        "criterion": _criterion,
        "n_iterations": res.nit,
        "success": True,
    }


def lbfgsb_sphere(model, x_candidate):
    crit, grad = _get_crit_and_grad(model)

    # Run an unconstrained solver in the unit cube
    lower_bounds = -np.ones(len(x_candidate))
    upper_bounds = np.ones(len(x_candidate))

    res = minimize(
        crit,
        x_candidate,
        method="L-BFGS-B",
        jac=grad,
        bounds=Bounds(lower_bounds, upper_bounds),
        options={
            "maxiter": len(x_candidate),
        },
    )

    # Project the solution onto the unit sphere if necessary
    solution_lies_inside_sphere = np.linalg.norm(res.x) <= 1

    if solution_lies_inside_sphere:
        _minimizer = res.x
        _criterion = res.fun
    else:
        _minimizer = _project_onto_unit_sphere(res.x)
        _criterion = crit(_minimizer)

    return {
        "x": _minimizer,
        "criterion": _criterion,
        "n_iterations": res.nit,
        "success": True,
    }


def slsqp_sphere(model, x_candidate):
    crit, grad = _get_crit_and_grad(model)
    constraints = _get_constraints()

    res = minimize(
        crit,
        x_candidate,
        method="SLSQP",
        jac=grad,
        constraints=constraints,
        options={"maxiter": len(x_candidate)},
    )

    return {
        "x": res.x,
        "criterion": res.fun,
        "success": res.success,
        "n_iterations": res.nit,
    }


def _get_crit_and_grad(model):
    def _crit(x, c, g, h):
        return c + x @ g + 0.5 * x @ h @ x

    def _grad(x, g, h):
        return g + x @ h

    crit = partial(_crit, c=model.intercept, g=model.linear_terms, h=model.square_terms)
    grad = partial(_grad, g=model.linear_terms, h=model.square_terms)

    return crit, grad


def _get_constraints():
    def _constr_fun(x):
        return x @ x

    def _constr_jac(x):
        return 2 * x

    constr = NonlinearConstraint(
        fun=_constr_fun,
        lb=-np.inf,
        ub=1,
        jac=_constr_jac,
    )

    return (constr,)


def _project_onto_unit_sphere(x):
    return x / np.linalg.norm(x)
