import numpy as np


def _standard_logistic(size):
    scale = np.sqrt(3) / np.pi
    return np.random.logistic(loc=0, scale=scale, size=size)


def _standard_uniform(size):
    ub = np.sqrt(3)
    lb = -ub
    return np.random.uniform(lb, ub, size=size)


def _standard_normal(size):
    return np.random.normal(size=size)


def _standard_gumbel(size):
    gamma = 0.577215664901532
    scale = np.sqrt(6) / np.pi
    loc = -scale * gamma
    return np.random.gumbel(loc=loc, scale=scale, size=size)


def _standard_laplace(size):
    return np.random.laplace(scale=np.sqrt(0.5), size=size)


NOISE_DISTRIBUTIONS = {
    "normal": _standard_normal,
    "gumbel": _standard_gumbel,
    "logistic": _standard_logistic,
    "uniform": _standard_uniform,
    "laplace": _standard_laplace,
}
