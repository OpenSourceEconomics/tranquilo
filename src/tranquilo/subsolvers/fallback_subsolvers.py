import numpy as np
from functools import partial
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from tranquilo.exploration_sample import draw_exploration_sample


# ======================================================================================
# Cube solvers
# ======================================================================================
def robust_cube_solver(model, x_candidate, radius=1.0):
    """Robust cube solver.

    Argument `radius` corresponds to half of the side length of the cube.

    """
    crit, grad = _get_crit_and_grad(model)

    lower_bounds = -radius * np.ones(len(x_candidate))
    upper_bounds = radius * np.ones(len(x_candidate))

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

    return {
        "x": res.x,
        "criterion": res.fun,
        "n_iterations": res.nit,
        "success": True,
    }


def robust_cube_solver_multistart(model, x_candidate):
    np.random.seed(12345)
    start_values = draw_exploration_sample(
        x=x_candidate,
        lower=-np.ones(len(x_candidate)),
        upper=np.ones(len(x_candidate)),
        n_samples=100,
        sampling_distribution="uniform",
        sampling_method="sobol",
        seed=1234,
    )

    best_crit = np.inf
    accepted_x = None
    critvals = []

    for x in start_values:
        res = robust_cube_solver(model, x)

        if res["criterion"] <= best_crit:
            accepted_x = res["x"]
            best_crit = res["criterion"]

        critvals.append(res["criterion"])

    return {
        "x": accepted_x,
        "criterion": best_crit,
        "std": np.std(critvals),
        "n_iterations": None,
        "success": None,
    }


# ======================================================================================
# Sphere solvers
# ======================================================================================


def robust_sphere_solver_inscribed_cube(model, x_candidate):
    """Robust sphere solver that uses a cube solver in an inscribed cube.

    We let x be in the largest cube that is inscribed inside the unit sphere. Formula
    is taken from http://tinyurl.com/4astpuwn.

    This solver cannot find solutions on the hull of the sphere.

    """
    return robust_cube_solver(model, x_candidate, radius=1 / np.sqrt(len(x_candidate)))


def robust_sphere_solver_reparametrized(model, x_candidate):
    """Robust sphere solver that uses reparametrization.

    We let x be in the cube -1 <= x <= 1, but if the optimizer chooses a point outside
    the sphere x is projected onto the sphere inside the criterion function.

    This solver can find solutions on the hull of the sphere.

    """

    def crit(x):
        x_norm = np.linalg.norm(x)
        if x_norm <= 1:
            x_tilde = x
        else:
            x_tilde = x / x_norm
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
    else:
        _minimizer = res.x / solution_norm

    return {
        "x": _minimizer,
        "criterion": res.fun,
        "n_iterations": res.nit,
        "success": True,
    }


def robust_sphere_solver_norm_constraint(model, x_candidate):
    """Robust sphere solver that uses ||x|| <= 1 as a nonlinear constraint.

    This solver can find solutions on the hull of the sphere.

    """
    crit, grad = _get_crit_and_grad(model)
    constraint = _get_constraint()

    lower_bounds = -np.ones(len(x_candidate))
    upper_bounds = np.ones(len(x_candidate))

    res = minimize(
        crit,
        x_candidate,
        method="SLSQP",
        bounds=Bounds(lower_bounds, upper_bounds),
        jac=grad,
        constraints=constraint,
        options={"maxiter": 3 * len(x_candidate)},
    )

    return {
        "x": res.x,
        "criterion": res.fun,
        "success": res.success,
        "n_iterations": res.nit,
    }


# ======================================================================================
# Criterion, gradient, and spherical constraint
# ======================================================================================


def _get_crit_and_grad(model):
    def _crit(x, c, g, h):
        return c + x @ g + 0.5 * x @ h @ x

    def _grad(x, g, h):
        return g + x @ h

    crit = partial(_crit, c=model.intercept, g=model.linear_terms, h=model.square_terms)
    grad = partial(_grad, g=model.linear_terms, h=model.square_terms)

    return crit, grad


def _get_constraint():
    def _constr_fun(x):
        return x @ x

    def _constr_jac(x):
        return 2 * x

    return NonlinearConstraint(
        fun=_constr_fun,
        lb=-np.inf,
        ub=1,
        jac=_constr_jac,
        keep_feasible=True,
    )
