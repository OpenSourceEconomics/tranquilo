import warnings

import numpy as np

from estimagic.config import IS_NLOPT_INSTALLED
from estimagic.optimization.algo_options import CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_CRITERION_TOLERANCE
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_PARAMS_TOLERANCE
from estimagic.optimization.algo_options import STOPPING_MAX_CRITERION_EVALUATIONS

if IS_NLOPT_INSTALLED:
    import nlopt


DEFAULT_ALGO_INFO = {
    "primary_criterion_entry": "value",
    "parallelizes": False,
    "needs_scaling": False,
}


def nlopt_bobyqa(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):

    """Minimize a scalar function using the BOBYQA algorithm.

    The implementation is derived from the BOBYQA subroutine of M. J. D. Powell.

    ADD REMAINING EXPLANATIONS FROM NLOPT DOCUMENTATION HERE.

    ``nlopt_bobyqa`` supports the following ``algo_options``:

    - convergence.relative_params_tolerance (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - convergence.relative_criterion_tolerance (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - stopping.max_criterion_evaluations (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.
    - stopping_max_iterations (int): If the maximum number of iterations is reached,
      the optimization stops, but we do not count this as convergence.

    """
    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LN_BOBYQA,
        algorithm_name="nlopt_bobyqa",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def nlopt_neldermead(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_relative_params_tolerance=CONVERGENCE_RELATIVE_PARAMS_TOLERANCE,
    convergence_absolute_params_tolerance=CONVERGENCE_ABSOLUTE_PARAMS_TOLERANCE,
    convergence_relative_criterion_tolerance=0,
    convergence_absolute_criterion_tolerance=CONVERGENCE_ABSOLUTE_CRITERION_TOLERANCE,
    stopping_max_criterion_evaluations=STOPPING_MAX_CRITERION_EVALUATIONS,
):
    """Minimize a scalar function using the Nelder-Mead simplex algorithm.

    Do not call this function directly but pass its name "nlopt_bobyqa" to
    estimagic's maximize or minimize function as `algorithm` argument. Specify
    your desired arguments as a dictionary and pass them as `algo_options` to
    minimize or maximize.

    The basic algorithm is described in:
    J. A. Nelder and R. Mead, "A simplex method for function minimization,"
    The Computer Journal 7, p. 308-313 (1965).

    The difference between the nlopt implementation an the original implementation is
    that the nlopt version supports bounds. This is done by moving all new points that
    would lie outside the bounds exactly on the bounds.

    Below, only details of the optional algorithm options are listed. For the mandatory
    arguments see :ref:`internal_optimizer_interface`. For more background on those
    options, see :ref:`naming_conventions`.

    Args:
        convergence_relative_params_tolerance (float): Stop when the relative movement
            between parameter vectors is smaller than this.
        convergence_relative_criterion_tolerance (float): Stop when the relative
            improvement between two iterations is smaller than this.
            In contrast to other algorithms the relative criterion tolerance is set
            to zero by default because setting it to any non-zero value made the
            algorithm stop too early even on the most simple test functions.
        stopping_max_criterion_evaluations (int): If the maximum number of function
            evaluation is reached, the optimization stops but we do not count this
            as convergence.
        stopping_max_iterations (int): If the maximum number of iterations is reached,
            the optimization stops, but we do not count this as convergence.

    Returns:
        dict: See :ref:`internal_optimizer_output` for details.

    """
    if np.isfinite(lower_bounds).any():
        warnings.warn(
            "nlopt_neldermead failed on simple benchmark functions if some but not all "
            "bounds were finite. Add finite bounds for all parameters for more safety."
        )

    out = _minimize_nlopt(
        criterion_and_derivative,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LN_NELDERMEAD,
        algorithm_name="nlopt_neldermead",
        convergence_xtol_rel=convergence_relative_params_tolerance,
        convergence_xtol_abs=convergence_absolute_params_tolerance,
        convergence_ftol_rel=convergence_relative_criterion_tolerance,
        convergence_ftol_abs=convergence_absolute_criterion_tolerance,
        stopping_max_eval=stopping_max_criterion_evaluations,
    )

    return out


def _minimize_nlopt(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    algorithm,
    algorithm_name,
    *,
    convergence_xtol_rel=None,
    convergence_xtol_abs=None,
    convergence_ftol_rel=None,
    convergence_ftol_abs=None,
    stopping_max_eval=None,
):
    """Run actual nlopt optimization argument, set relevant attributes."""
    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = algorithm_name

    def func(x, grad):
        if grad.size > 0:
            criterion, derivative = criterion_and_derivative(
                x,
                task="criterion_and_derivative",
                algorithm_info=algo_info,
            )
            grad[:] = derivative
        else:
            criterion = criterion_and_derivative(
                x,
                task="criterion",
                algorithm_info=algo_info,
            )
        return criterion

    opt = nlopt.opt(algorithm, x.shape[0])
    if convergence_ftol_rel is not None:
        opt.set_ftol_rel(convergence_ftol_rel)
    if convergence_ftol_abs is not None:
        opt.set_ftol_abs(convergence_ftol_abs)
    if convergence_xtol_rel is not None:
        opt.set_xtol_rel(convergence_xtol_rel)
    if convergence_xtol_abs is not None:
        opt.set_xtol_abs(convergence_xtol_abs)
    if lower_bounds is not None:
        opt.set_lower_bounds(lower_bounds)
    if upper_bounds is not None:
        opt.set_upper_bounds(upper_bounds)
    if stopping_max_eval is not None:
        opt.set_maxeval(stopping_max_eval)
    opt.set_min_objective(func)
    solution_x = opt.optimize(x)
    return _process_nlopt_results(opt, solution_x)


def _process_nlopt_results(nlopt_obj, solution_x):
    messages = {
        1: "Convergence achieved ",
        2: (
            "Optimizer stopped because maximum value of criterion function was reached"
        ),
        3: (
            "Optimizer stopped because convergence_relative_criterion_tolerance or "
            + "convergence_absolute_criterion_tolerance was reached"
        ),
        4: (
            "Optimizer stopped because convergence_relative_params_tolerance or "
            + "convergence_absolute_params_tolerance was reached"
        ),
        5: "Optimizer stopped because max_criterion_evaluations was reached",
        6: "Optimizer stopped because max running time was reached",
        -1: "Optimizer failed",
        -2: "Invalid arguments were passed",
        -3: "Memory error",
        -4: "Halted because roundoff errors limited progress",
        -5: "Halted because of user specified forced stop",
    }
    processed = {
        "solution_x": solution_x,
        "solution_criterion": nlopt_obj.last_optimum_value(),
        "solution_derivative": None,
        "solution_hessian": None,
        "n_criterion_evaluations": nlopt_obj.get_numevals(),
        "n_derivative_evaluations": None,
        "n_iterations": None,
        "success": nlopt_obj.last_optimize_result() in [1, 2, 3, 4],
        "message": messages[nlopt_obj.last_optimize_result()],
        "reached_convergence_criterion": None,
    }
    return processed