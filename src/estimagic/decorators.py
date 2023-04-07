"""This module contains various decorators.

There are two kinds of decorators defined in this module which consists of either two or
three nested functions. The former are decorators without and the latter with arguments.

For more information on decorators, see this `guide`_ on https://realpython.com which
provides a comprehensive overview.

.. _guide:
    https://realpython.com/primer-on-python-decorators/

"""
import functools
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from estimagic.exceptions import get_traceback
from estimagic.parameters.process_constraints import process_constraints
from estimagic.parameters.reparametrize import reparametrize_from_internal


def numpy_interface(func=None, *, params=None, constraints=None, numpy_output=False):
    """Convert x to params.

    This decorated function receives a NumPy array of parameters and converts it to a
    :class:`pandas.DataFrame` which can be handled by the user's criterion function.

    For convenience, the decorated function can also be called directly with a
    params DataFrame. In that case, the decorator does nothing.

    Args:
        func (callable): The function to which the decorator is applied.
        params (pandas.DataFrame): See :ref:`params`.
        constraints (list of dict): Contains constraints.
        numpy_output (bool): Whether pandas objects in the output should also be
            converted to numpy arrays.

    Returns:
        callable

    """
    constraints = [] if constraints is None else constraints

    pc, pp = process_constraints(constraints, params)

    fixed_values = pp["_internal_fixed_value"].to_numpy()
    pre_replacements = pp["_pre_replacements"].to_numpy().astype(int)
    post_replacements = pp["_post_replacements"].to_numpy().astype(int)

    def decorator_numpy_interface(func):
        @functools.wraps(func)
        def wrapper_numpy_interface(x, *args, **kwargs):
            if isinstance(x, pd.DataFrame):
                p = x
            elif isinstance(x, np.ndarray):
                p = reparametrize_from_internal(
                    internal=x,
                    fixed_values=fixed_values,
                    pre_replacements=pre_replacements,
                    processed_constraints=pc,
                    post_replacements=post_replacements,
                    params=params,
                    return_numpy=False,
                )
            else:
                raise ValueError(
                    "x must be a numpy array or DataFrame with 'value' column."
                )

            criterion_value = func(p, *args, **kwargs)

            if isinstance(criterion_value, (pd.DataFrame, pd.Series)) and numpy_output:
                criterion_value = criterion_value.to_numpy()

            return criterion_value

        return wrapper_numpy_interface

    if callable(func):
        return decorator_numpy_interface(func)
    else:
        return decorator_numpy_interface


def catch(
    func=None,
    *,
    exception=Exception,
    exclude=(KeyboardInterrupt, SystemExit),
    onerror=None,
    default=None,
    warn=True,
    reraise=False,
):
    """Catch and handle exceptions.

    This decorator can be used with and without additional arguments.

    Args:
        exception (Exception or tuple): One or several exceptions that
            are caught and handled. By default all Exceptions are
            caught and handled.
        exclude (Exception or tuple): One or several exceptionts that
            are not caught. By default those are KeyboardInterrupt and
            SystemExit.
        onerror (None or Callable): Callable that takes an Exception
            as only argument. This is called when an exception occurs.
        default: Value that is returned when as the output of func when
            an exception occurs. Can be one of the following:
            - a constant
            - "__traceback__", in this case a string with a traceback is returned.
            - callable with the same signature as func.
        warn (bool): If True, the exception is converted to a warning.
        reraise (bool): If True, the exception is raised after handling it.

    """

    def decorator_catch(func):
        @functools.wraps(func)
        def wrapper_catch(*args, **kwargs):
            try:
                res = func(*args, **kwargs)
            except exclude:
                raise
            except exception as e:

                if onerror is not None:
                    onerror(e)

                if reraise:
                    raise e

                tb = get_traceback()

                if warn:
                    msg = f"The following exception was caught:\n\n{tb}"
                    warnings.warn(msg)

                if default == "__traceback__":
                    res = tb
                elif callable(default):
                    res = default(*args, **kwargs)
                else:
                    res = default
            return res

        return wrapper_catch

    if callable(func):
        return decorator_catch(func)
    else:
        return decorator_catch


def unpack(func=None, symbol=None):
    def decorator_unpack(func):
        if symbol is None:

            @functools.wraps(func)
            def wrapper_unpack(arg):
                return func(arg)

        elif symbol == "*":

            @functools.wraps(func)
            def wrapper_unpack(arg):
                return func(*arg)

        elif symbol == "**":

            @functools.wraps(func)
            def wrapper_unpack(arg):
                return func(**arg)

        return wrapper_unpack

    if callable(func):
        return decorator_unpack(func)
    else:
        return decorator_unpack


def switch_sign(func):
    """Switch sign of all outputs of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        unswitched = func(*args, **kwargs)
        if isinstance(unswitched, dict):
            switched = {key: -val for key, val in unswitched.items()}
        elif isinstance(unswitched, (tuple, list)):
            switched = []
            for entry in unswitched:
                if isinstance(entry, dict):
                    switched.append({key: -val for key, val in entry.items()})
                else:
                    switched.append(-entry)
            if isinstance(unswitched, tuple):
                switched = tuple(switched)
        else:
            switched = -unswitched
        return switched

    return wrapper


class AlgoInfo(NamedTuple):
    primary_criterion_entry: str
    name: str
    parallelizes: bool
    disable_cache: bool
    needs_scaling: bool
    is_available: bool


def mark_minimizer(
    func=None,
    *,
    primary_criterion_entry="value",
    name=None,
    parallelizes=False,
    disable_cache=False,
    needs_scaling=False,
    is_available=True,
):
    """Decorator to mark a function as internal estimagic minimizer and add information.

    Args:
        func (callable): The function to be decorated
        primary_criterion_entry (str): One of "value", "contributions",
            "root_contributions" or "dict". Default: "value". This decides
            which part of the output of the user provided criterion function
            is needed by the internal optimizer.
        name (str): The name of the internal algorithm.
        parallelizes (bool): Must be True if an algorithm evaluates the criterion,
            derivative or criterion_and_derivative in parallel.
        disable_cache (bool): If True, no caching for the criterion function
            or its derivatives are used.
        needs_scaling (bool): Must be True if the algorithm is not reasonable
            independent of the scaling of the parameters.
        is_available (bool): Whether the algorithm is available. This is needed for
            algorithms that require optional dependencies.

    """
    if name is None:
        raise TypeError(
            "mark_minimizer() missing 1 required keyword-only argument: 'name'"
        )
    elif not isinstance(name, str):
        raise TypeError("name must be a string.")

    valid_entries = ["value", "dict", "contributions", "root_contributions"]
    if primary_criterion_entry not in valid_entries:
        raise ValueError(
            f"primary_criterion_entry must be one of {valid_entries} not "
            f"{primary_criterion_entry}."
        )

    if not isinstance(parallelizes, bool):
        raise TypeError("parallelizes must be a bool.")

    if not isinstance(disable_cache, bool):
        raise TypeError("disable_cache must be a bool.")

    if not isinstance(needs_scaling, bool):
        raise TypeError("needs_scaling must be a bool.")

    if not isinstance(is_available, bool):
        raise TypeError("is_available must be a bool.")

    algo_info = AlgoInfo(
        primary_criterion_entry=primary_criterion_entry,
        name=name,
        parallelizes=parallelizes,
        disable_cache=disable_cache,
        needs_scaling=needs_scaling,
        is_available=is_available,
    )

    def decorator_mark_minimizer(func):
        @functools.wraps(func)
        def wrapper_mark_minimizer(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper_mark_minimizer._algorithm_info = algo_info

        return wrapper_mark_minimizer

    if callable(func):
        return decorator_mark_minimizer(func)

    else:
        return decorator_mark_minimizer
