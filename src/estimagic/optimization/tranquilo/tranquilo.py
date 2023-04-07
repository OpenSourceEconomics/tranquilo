import numbers
import warnings
from functools import partial
from typing import NamedTuple

import numpy as np
from estimagic.decorators import mark_minimizer
from estimagic.optimization.tranquilo.adjust_radius import adjust_radius
from estimagic.optimization.tranquilo.aggregate_models import get_aggregator
from estimagic.optimization.tranquilo.filter_points import get_sample_filter
from estimagic.optimization.tranquilo.fit_models import get_fitter
from estimagic.optimization.tranquilo.models import ModelInfo
from estimagic.optimization.tranquilo.models import ScalarModel
from estimagic.optimization.tranquilo.options import Bounds
from estimagic.optimization.tranquilo.options import ConvOptions
from estimagic.optimization.tranquilo.options import RadiusOptions
from estimagic.optimization.tranquilo.options import TrustRegion
from estimagic.optimization.tranquilo.sample_points import get_sampler
from estimagic.optimization.tranquilo.solve_subproblem import get_subsolver
from estimagic.optimization.tranquilo.tranquilo_history import History


def _tranquilo(
    criterion,
    x,
    functype,
    lower_bounds=None,
    upper_bounds=None,
    disable_convergence=False,
    stopping_max_iterations=200,
    random_seed=925408,
    sampler="sphere",
    sample_filter="keep_all",
    fitter="ols",
    subsolver="bntr",
    sample_size=None,
    surrogate_model=None,
    radius_options=None,
    sampler_options=None,
    fit_options=None,
    solver_options=None,
    conv_options=None,
    silence_experimental_warning=False,
):
    """Find the local minimum to a noisy optimization problem.

    Args:
        criterion (callable): Function that return values of the objective function.
        x (np.ndarray): Initial guess for the parameter vector.
        functype (str): String indicating whether the criterion is a scalar, a
            likelihood function or a least-square type of function. Valid arguments
            are:
            - "scalar"
            - "likelihood"
            - "least_squares"
        lower_bounds (np.ndarray or NoneTeyp): 1d array of shape (n,) with lower bounds
            for the parameter vector x.
        upper_bounds (np.ndarray or NoneTeyp): 1d array of shape (n,) with upper bounds
            for the parameter vector x.
        disable_convergence (bool): If True, check for convergence criterion and stop
            the iterations.
        stopping_max_iterations (int): Maximum number of iterations to run.
        random_seed (int): The seed used in random number generation.
        sample_filter (str): The method used to filter points in the current trust
            region.
        sampler (str): The sampling method used to sample points from the current
            trust region.
        fitter (str): The method used to fit the surrogate model.
        subsolver (str): The algorithm used for solving the nested surrogate model.
        sample_size (str): Target sample size. One of:
            - "linear": n + 1
            - "powell": 2 * n + 1
            - "quadratic: 0.5 * n * (n + 1) + n + 1
        surrogate_model (str): Type of surrogate model to fit. Both a "linear" and
            "quadratic" surrogate model are supported.
        radius_options (NemdTuple or NoneType): Options for trust-region radius
            management.
        sampler_options (dict or NoneType): Additional keyword arguments passed to the
            sampler function.
        fit_options (dict or NoneType): Additional keyword arguments passed to the
            fitter function.
        solver_options (dict or NoneType): Additional keyword arguments passed to the
            sub-solver function.
        conv_options (NamedTuple or NoneType): Criteria for successful convergence.

    Returns:
        (dict): Results dictionary with the following items:
            - solution_x (np.ndarray): Solution vector of shape (n,).
            - solution_criterion (np.ndarray): Values of the criterion function at the
                solution vector. Shape (n_obs,).
            - states (list): The history of optimization as a list of the State objects.
            - message (str or NoneType): Message stating which convergence criterion,
                if any has been reached at the end of optimization

    """
    # ==================================================================================
    # experimental warning
    # ==================================================================================
    if not silence_experimental_warning:
        warnings.warn(
            "Tranquilo is extremely experimental. algo_options and results will change "
            "frequently and without notice. Do not use."
        )

    # ==================================================================================
    # set default values for optional arguments
    # ==================================================================================
    sampling_rng = np.random.default_rng(random_seed)

    if radius_options is None:
        radius_options = RadiusOptions()
    if sampler_options is None:
        sampler_options = {}
    if fit_options is None:
        fit_options = {}
    if solver_options is None:
        solver_options = {}

    model_info = _process_surrogate_model(
        surrogate_model=surrogate_model,
        functype=functype,
    )

    target_sample_size = _process_sample_size(
        user_sample_size=sample_size,
        model_info=model_info,
        x=x,
    )

    if functype == "scalar":
        aggregator = "identity"
    elif functype == "likelihood":
        aggregator = "information_equality_linear"
    elif functype == "least_squares":
        aggregator = "least_squares_linear"
    else:
        raise ValueError(f"Invalid functype: {functype}")

    if conv_options is None:
        conv_options = ConvOptions()

    # ==================================================================================
    # initialize compoments for the solver
    # ==================================================================================

    history = History(functype=functype)
    bounds = Bounds(lower=lower_bounds, upper=upper_bounds)
    trustregion = TrustRegion(center=x, radius=radius_options.initial_radius)
    sample_points = get_sampler(sampler, bounds=bounds, user_options=sampler_options)
    filter_points = get_sample_filter(sample_filter)

    aggregate_vector_model = get_aggregator(
        aggregator=aggregator,
        functype=functype,
        model_info=model_info,
    )

    fit_model = get_fitter(
        fitter=fitter,
        user_options=fit_options,
        model_info=model_info,
    )

    solve_subproblem = get_subsolver(
        solver=subsolver,
        user_options=solver_options,
        bounds=bounds,
    )

    first_eval = criterion(x)
    history.add_entries(x, first_eval)

    state = State(
        index=0,
        model=None,
        rho=None,
        radius=trustregion.radius,
        x=history.get_xs(0),
        fvec=history.get_fvecs(0),
        fval=history.get_fvals(0),
    )
    states = [state]

    # ==================================================================================
    # main optimization loop
    # ==================================================================================
    converged, msg = False, None
    for _ in range(stopping_max_iterations):
        old_indices = history.get_indices_in_trustregion(trustregion)
        old_xs = history.get_xs(old_indices)

        filtered_xs, filtered_indices = filter_points(
            xs=old_xs,
            indices=old_indices,
            state=state,
        )

        if state.index not in filtered_indices:
            raise ValueError()

        new_xs = sample_points(
            trustregion=trustregion,
            target_size=target_sample_size,
            existing_xs=filtered_xs,
            rng=sampling_rng,
        )

        new_fvecs = [criterion(_x) for _x in new_xs]
        new_indices = np.arange(history.get_n_fun(), history.get_n_fun() + len(new_xs))
        history.add_entries(new_xs, new_fvecs)

        model_indices = np.hstack([filtered_indices, new_indices])

        model_xs = history.get_xs(model_indices)
        model_fvecs = history.get_fvecs(model_indices)

        centered_xs = (model_xs - trustregion.center) / trustregion.radius

        vector_model = fit_model(centered_xs, model_fvecs)

        scalar_model = aggregate_vector_model(
            vector_model=vector_model,
            fvec_center=state.fvec,
        )

        sub_sol = solve_subproblem(model=scalar_model, trustregion=trustregion)

        candidate_x = sub_sol["x"]

        candidate_fvec = criterion(candidate_x)
        candidate_index = history.get_n_fun()
        history.add_entries(candidate_x, candidate_fvec)
        candidate_fval = history.get_fvals(-1)
        actual_improvement = -(candidate_fval - state.fval)

        rho = _calculate_rho(
            actual_improvement=actual_improvement,
            expected_improvement=sub_sol["expected_improvement"],
        )

        new_radius = adjust_radius(
            radius=trustregion.radius,
            rho=rho,
            step=candidate_x - state.x,
            options=radius_options,
        )

        if actual_improvement > 0:
            trustregion = trustregion._replace(center=candidate_x, radius=new_radius)
            state = State(
                index=candidate_index,
                model=scalar_model,
                rho=rho,
                radius=new_radius,
                x=candidate_x,
                fvec=history.get_fvecs(candidate_index),
                fval=history.get_fvals(candidate_index),
            )

        else:
            trustregion = trustregion._replace(radius=new_radius)
            state = state._replace(
                model=scalar_model,
                rho=rho,
                radius=new_radius,
            )

        states.append(state)

        if actual_improvement >= 0 and not disable_convergence:
            converged, msg = _is_converged(states=states, options=conv_options)
            if converged:
                break

    # ==================================================================================
    # results processing
    # ==================================================================================
    res = {
        "solution_x": state.x,
        "solution_criterion": state.fval,
        "states": states,
        "message": msg,
    }

    return res


def _calculate_rho(actual_improvement, expected_improvement):
    if expected_improvement == 0 and actual_improvement > 0:
        rho = np.inf
    elif expected_improvement == 0:
        rho = -np.inf
    else:
        rho = actual_improvement / expected_improvement

    return rho


class State(NamedTuple):
    index: int
    model: ScalarModel
    rho: float
    radius: float
    x: np.ndarray
    fvec: np.ndarray
    fval: np.ndarray


def _is_converged(states, options):
    old, new = states[-2:]

    f_change_abs = np.abs(old.fval - new.fval)
    f_change_rel = f_change_abs / max(np.abs(old.fval), 0.1)
    x_change_abs = np.linalg.norm(old.x - new.x)
    x_change_rel = np.linalg.norm((old.x - new.x) / np.clip(np.abs(old.x), 0.1, np.inf))
    g_norm_abs = np.linalg.norm(new.model.linear_terms)
    g_norm_rel = g_norm_abs / max(g_norm_abs, 0.1)

    converged = True
    if g_norm_rel <= options.gtol_rel:
        msg = "Relative gradient norm smaller than tolerance."
    elif g_norm_abs <= options.gtol_abs:
        msg = "Absolute gradient norm smaller than tolerance."
    elif f_change_rel <= options.ftol_rel:
        msg = "Relative criterion change smaller than tolerance."
    elif f_change_abs <= options.ftol_abs:
        msg = "Absolute criterion change smaller than tolerance."
    elif x_change_rel <= options.xtol_rel:
        msg = "Relative params change smaller than tolerance."
    elif x_change_abs <= options.xtol_abs:
        msg = "Absolute params change smaller than tolerance."
    else:
        converged = False
        msg = None

    return converged, msg


def _process_surrogate_model(surrogate_model, functype):

    if surrogate_model is None:
        if functype == "scalar":
            surrogate_model = "quadratic"
        else:
            surrogate_model = "linear"

    if isinstance(surrogate_model, ModelInfo):
        out = surrogate_model
    elif isinstance(surrogate_model, str):
        if surrogate_model == "linear":
            out = ModelInfo(has_squares=False, has_interactions=False)
        elif surrogate_model == "quadratic":
            out = ModelInfo(has_squares=True, has_interactions=True)
        elif surrogate_model == "diagonal":
            out = ModelInfo(has_squares=True, has_interactions=False)
        else:
            raise ValueError(f"Invalid surrogate model: {surrogate_model}")

    else:
        raise ValueError(f"Invalid surrogate model: {surrogate_model}")
    return out


def _process_sample_size(user_sample_size, model_info, x):
    if user_sample_size is None:
        if model_info.has_squares or model_info.has_interactions:
            out = 2 * len(x) + 1
        else:
            out = len(x) + 1

    elif isinstance(user_sample_size, str):
        user_sample_size = user_sample_size.replace(" ", "")
        if user_sample_size in ["linear", "n+1"]:
            out = len(x) + 1
        elif user_sample_size in ["powell", "2n+1", "2*n+1"]:
            out = 2 * len(x) + 1
        elif user_sample_size == "quadratic":
            out = len(x) * (len(x) + 1) // 2 + len(x) + 1
        else:
            raise ValueError(f"Invalid sample size: {user_sample_size}")

    elif isinstance(user_sample_size, numbers.Number):
        out = int(user_sample_size)
    else:
        raise ValueError(f"invalid sample size: {user_sample_size}")
    return out


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
