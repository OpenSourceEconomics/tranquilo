import functools

import numpy as np


def get_wrapped_criterion(criterion, batch_evaluator, n_cores, history):
    """Wrap the criterion function to do get parallelization and history handling.

    Notes
    -----

    The wrapped criterion function takes a dict mapping x_indices to required numbers of
    evaluations as only argument. It evaluates the criterion function in parallel and
    saves the resulting function evaluations in the history.

    The wrapped criterion function does not return anything.

    Args:
        criterion (function): The criterion function to wrap.
        batch_evaluator (function): The batch evaluator to use.
        n_cores (int): The number of cores to use.
        history (History): The tranquilo history.

    Returns:
        callable: The wrapped criterion function.

    """
    batch_evaluator = process_batch_evaluator(batch_evaluator)

    @functools.wraps(criterion)
    def wrapper_criterion(eval_info):
        if not isinstance(eval_info, dict):
            raise ValueError("eval_info must be a dict.")

        if len(eval_info) == 0:
            return

        x_indices = list(eval_info)
        repetitions = list(eval_info.values())

        xs = history.get_xs(x_indices)
        xs = np.repeat(xs, repetitions, axis=0)

        arguments = list(xs)

        effective_n_cores = min(n_cores, len(arguments))

        raw_evals = batch_evaluator(
            criterion,
            arguments=arguments,
            n_cores=effective_n_cores,
        )

        # replace NaNs but keep infinite values. NaNs would be problematic in many
        # places, infs are only a problem in model fitting and will be handled there
        clipped_evals = [
            np.nan_to_num(critval, nan=np.inf, posinf=np.inf, neginf=-np.inf)
            for critval in raw_evals
        ]

        history.add_evals(
            x_indices=np.repeat(x_indices, repetitions),
            evals=clipped_evals,
        )

    return wrapper_criterion


def process_batch_evaluator(batch_evaluator="joblib"):
    batch_evaluator = "joblib" if batch_evaluator is None else batch_evaluator

    if callable(batch_evaluator):
        out = batch_evaluator
    elif isinstance(batch_evaluator, str):
        if batch_evaluator == "joblib":
            from estimagic.batch_evaluators import joblib_batch_evaluator as out
        elif batch_evaluator == "pathos":
            from estimagic.batch_evaluators import pathos_mp_batch_evaluator as out
        else:
            raise ValueError(
                "Invalid batch evaluator requested. Currently only 'pathos' and "
                "'joblib' are supported."
            )
    else:
        raise TypeError("batch_evaluator must be a callable or string.")

    return out
