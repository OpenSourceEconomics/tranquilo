import numpy as np
import difflib
import warnings


def propose_alternatives(requested, possibilities, number=3):
    """Propose possible alternatives based on similarity to requested.

    Args:
        requested_algo (str): From the user requested algorithm.
        possibilities (list(str)): List of available algorithms
            are lists of algorithms.
        number (int) : Number of proposals.

    Returns:
        proposals (list(str)): List of proposed algorithms.

    Example:
        >>> possibilities = ["scipy_lbfgsb", "scipy_slsqp", "nlopt_lbfgsb"]
        >>> propose_alternatives("scipy_L-BFGS-B", possibilities, number=1)
        ['scipy_slsqp']
        >>> propose_alternatives("L-BFGS-B", possibilities, number=2)
        ['scipy_slsqp', 'scipy_lbfgsb']

    """
    number = min(number, len(possibilities))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        proposals = difflib.get_close_matches(
            requested, possibilities, n=number, cutoff=0
        )

    return proposals


def get_rng(seed):
    """Construct a random number generator.

    seed (Union[None, int, numpy.random.Generator]): If seed is None or int the
        numpy.random.default_rng is used seeded with seed. If seed is already a
        Generator instance then that instance is used.

    Returns:
        numpy.random.Generator: The random number generator.

    """
    if isinstance(seed, np.random.Generator):
        rng = seed
    elif seed is None or isinstance(seed, int):
        rng = np.random.default_rng(seed)
    else:
        raise TypeError("seed type must be in {None, int, numpy.random.Generator}.")
    return rng
