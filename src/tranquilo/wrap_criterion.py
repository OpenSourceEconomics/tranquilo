import numpy as np
from tranquilo.history import History


def get_wrapped_criterion(batch_fun, n_cores: int, batch_size: int, history: History):
    """Wrap the batch function to handle tranquilo's history management.

    Notes
    -----

    The wrapped criterion function takes a dict mapping x_indices to required numbers of
    evaluations as only argument. It evaluates the criterion function in parallel and
    saves the resulting function evaluations in the tranquilo history.

    The wrapped criterion function does not return anything.

    Args:
        batch_fun (callable): A function that takes (x_list, n_cores, batch_size) and
            returns a list of function values. When called from optimagic, this is
            InternalOptimizationProblem.batch_fun which handles parallelization and
            error handling internally.
        n_cores: The number of cores to use.
        batch_size: The batch size for parallel evaluation.
        history: The tranquilo history.

    Returns:
        callable: The wrapped criterion function.

    """

    def wrapper_criterion(eval_info):
        if not isinstance(eval_info, dict):
            raise ValueError("eval_info must be a dict.")

        if len(eval_info) == 0:
            return

        x_indices = list(eval_info)
        repetitions = list(eval_info.values())

        xs = history.get_xs(x_indices)
        xs = np.repeat(xs, repetitions, axis=0)

        x_list = list(xs)

        effective_n_cores = min(n_cores, len(x_list))

        # Call the batch function directly - it handles parallelization and error
        # handling internally. When called from optimagic, this also populates
        # optimagic's history automatically.
        raw_evals = batch_fun(x_list, effective_n_cores, batch_size)

        # replace NaNs but keep infinite values. NaNs would be problematic in many
        # places, infs are only a problem in model fitting and will be handled there.
        # Note: when using optimagic's batch_fun, errors are already handled and
        # replaced with penalty values, so we don't need to check for tracebacks.
        clipped_evals = [
            np.nan_to_num(critval, nan=np.inf, posinf=np.inf, neginf=-np.inf)
            for critval in raw_evals
        ]

        history.add_evals(
            x_indices=np.repeat(x_indices, repetitions),
            evals=clipped_evals,
        )

    return wrapper_criterion
