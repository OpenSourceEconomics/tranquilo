import numpy as np
import estimagic as em
import functools


def robustify_subproblem_sphere_solver(solver):
    @functools.wraps(solver)
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
            result = _robust_sphere_solver(
                model=model,
                x_candidate=x_candidate,
            )
        return result

    return wrapped_solver


def _robust_sphere_solver(model, x_candidate):
    """Robustly find the minimizer of a quadratic model on the unit sphere.

    Args:
        model (ScalarModel): The fitted model of which we want to find the minimum.
        x_candidate (np.ndarray): The candidate solution.

    Returns:
        dict: The result of the solver.
        - x (np.ndarray): The minimizer.
        - criterion (float): The value of the criterion at the minimizer.
        - n_iterations (int): The number of iterations.
        - success (bool): Whether the solver was successful.

    """
    criterion, derivative = _get_criterion_and_derivative(model)

    # Run an unconstrained solver in the unit cube
    # ==================================================================================
    lower_bounds = -np.ones(len(x_candidate))
    upper_bounds = np.ones(len(x_candidate))

    res = em.minimize(
        criterion=criterion,
        params=x_candidate,
        algorithm="scipy_lbfgsb",
        derivative=derivative,
        algo_options={
            "stopping.max_iterations": len(x_candidate),
        },
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    # Project the solution onto the unit sphere if necessary
    # ==================================================================================
    solution_lies_inside_sphere = np.linalg.norm(res.params) <= 1

    if solution_lies_inside_sphere:
        _minimizer = res.params
        _criterion = res.criterion
    else:
        _minimizer = _project_onto_unit_sphere(res.params)
        _criterion = criterion(_minimizer)

    return {
        "x": _minimizer,
        "criterion": _criterion,
        "n_iterations": res.n_iterations,
        "success": True,
    }


def _get_criterion_and_derivative(model):
    def criterion(x):
        return model.predict(x)

    def derivative(x):
        return model.linear_terms + 2 * model.square_terms @ x

    return criterion, derivative


def _project_onto_unit_sphere(x):
    return x / np.linalg.norm(x)
