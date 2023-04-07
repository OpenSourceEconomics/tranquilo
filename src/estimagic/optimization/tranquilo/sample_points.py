from functools import partial

import numpy as np
from scipy.spatial.distance import pdist
from scipy.special import gammainc, logsumexp

import estimagic as em
from estimagic.optimization.tranquilo.get_component import get_component


def get_sampler(sampler, model_info=None, user_options=None):
    """Get sampling function partialled options.

    Args:
        sampler (str or callable): Name of a sampling method or sampling function.
            The arguments of sampling functions need to be: ``trustregion``,
            ``n_points``, ``rng``, ``existing_xs`` and ``bounds``.
            Sampling functions need to return a dictionary with the entry "points"
            (and arbitrary additional information). See ``reference_sampler`` for
            details.
        user_options (dict): Additional keyword arguments for the sampler. Options that
            are not used by the sampler are ignored with a warning. If sampler is
            'hull_sampler' or 'optimal_hull_sampler' the user options must contain the
            argument 'order', which is a positive integer.

    Returns:
        callable: Function that depends on trustregion, n_points, existing_xs and
            existing_fvals, model_info and  and returns a new sample.

    """

    built_in_samplers = {
        "box": _box_sampler,
        "ball": _ball_sampler,
        "hull_sampler": _hull_sampler,
        "optimal_hull_sampler": _optimal_hull_sampler,
        "cube": partial(_hull_sampler, order=np.inf),
        "sphere": partial(_hull_sampler, order=2),
        "optimal_cube": partial(_optimal_hull_sampler, order=np.inf),
        "optimal_sphere": partial(_optimal_hull_sampler, order=2),
    }

    if (
        isinstance(sampler, str)
        and "hull_sampler" in sampler
        and "order" not in user_options
    ):
        msg = (
            "The hull_sampler and optimal_hull_sampler require the argument 'order' to "
            "be prespecfied in the user_options dictionary. Order is a positive "
            "integer. For order = 2 the hull_sampler equals the sphere_sampler, and "
            "for order = np.inf it equals the cube_sampler."
        )
        raise ValueError(msg)

    default_options = {
        "model_info": model_info,
    }

    mandatory_args = [
        "trustregion",
        "n_points",
        "existing_xs",
        "rng",
    ]

    out = get_component(
        name_or_func=sampler,
        component_name="sampler",
        func_dict=built_in_samplers,
        default_options=default_options,
        user_options=user_options,
        mandatory_signature=mandatory_args,
    )

    return out


def _box_sampler(
    trustregion,
    n_points,
    rng,
    existing_xs=None,  # noqa: ARG001
):
    """Naive random generation of trustregion points inside a box.

    This is just a reference implementation to illustrate the interface of trustregion
    samplers. Mathematically it samples uniformaly from inside the cube defined by the
    intersection of the trustregion and the bounds.

    All arguments but seed are mandatory, even if not used.

    Samplers should not make unnecessary checks on input compatibility (e.g. that the
    shapes of existing_xs and existing_fvals match). This will be done automatically
    outside of the sampler.

    Args:
        trustregion (Region): Trustregion. See module region.py.
        n_points (int): how many new points to sample
        rng (numpy.random.Generator): Random number generator.
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies lower_bounds <= existing_xs <= upper_bounds.

    """
    n_params = len(trustregion.center)

    bounds = trustregion.cube_bounds

    points = rng.uniform(
        low=bounds.lower,
        high=bounds.upper,
        size=(n_points, n_params),
    )
    return points


def _ball_sampler(
    trustregion,
    n_points,
    rng,
    existing_xs=None,  # noqa: ARG001
):
    """Naive random generation of trustregion points inside a ball.

    Mathematically it samples uniformaly from inside the ball defined by the
    intersection of the trustregion and the bounds.

    Code is adapted from https://tinyurl.com/y3p2dz6b.

    Args:
        trustregion (Region): Trustregion. See module region.py.
        n_points (int): how many new points to sample
        rng (numpy.random.Generator): Random number generator.
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies lower_bounds <= existing_xs <= upper_bounds.

    """
    n_params = len(trustregion.center)

    raw = rng.normal(size=(n_points, n_params))
    norm = np.linalg.norm(raw, axis=1, ord=2)
    scale = gammainc(n_params / 2, norm**2 / 2) ** (1 / n_params) / norm
    points = raw * scale.reshape(-1, 1)
    out = trustregion.map_from_unit(points)
    return out


def _hull_sampler(
    trustregion,
    n_points,
    rng,
    order,
    distribution=None,
    existing_xs=None,  # noqa: ARG001
):
    """Random generation of trustregion points on the hull of general sphere / cube.

    Points are sampled randomly on a hull (of a sphere for order=2 and of a cube for
    order=np.inf). These points are then mapped into the feasible region, which is
    defined by the intersection of the trustregion and the bounds.

    Args:
        trustregion (Region): Trustregion. See module region.py.
        n_points (int): how many new points to sample
        rng (numpy.random.Generator): Random number generator.
        order (int): Type of norm to use when scaling the sampled points. For 2 it will
            result in sphere sampling, for np.inf in cube sampling.
        distribution (str): Distribution to use for initial sample before points are
            projected onto unit hull. Must be in {'normal', 'uniform'}.
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies lower_bounds <= existing_xs <= upper_bounds.

    """
    n_params = len(trustregion.center)

    if distribution is None:
        distribution = "normal" if order <= 3 else "uniform"
    raw = _draw_from_distribution(distribution, rng=rng, size=(n_points, n_params))
    points = _project_onto_unit_hull(raw, order=order)
    out = trustregion.map_from_unit(points)
    return out


def _optimal_hull_sampler(
    trustregion,
    n_points,
    rng,
    order,
    distribution=None,
    hardness=1,
    existing_xs=None,
    algorithm="scipy_lbfgsb",
    algo_options=None,
    criterion=None,
    n_points_randomsearch=1,
    return_info=False,
):
    """Optimal generation of trustregion points on the hull of general sphere / cube.

    Points are sampled optimally on a hull (of a sphere for order=2 and of a cube for
    order=np.inf), where the criterion that is maximized is the minimum distance of all
    pairs of points, except for pairs of existing points. These points are then mapped
    into the feasible region, which is defined by the intersection of the trustregion
    and the bounds. Instead of using a hard minimum we return the soft minimum, whose
    accuracy we govern by the hardness factor. For more information on the soft-minimum,
    seek: https://tinyurl.com/mrythbk4.

    Args:
        trustregion (Region): Trustregion. See module region.py.
        n_points (int): how many new points to sample
        rng (numpy.random.Generator): Random number generator.
        order (int): Type of norm to use when scaling the sampled points. For 2 it will
            result in sphere sampling, for np.inf in cube sampling.
        distribution (str): Distribution to use for initial sample before points are
            projected onto unit hull. Must be in {'normal', 'uniform'}.
        hardness (float): Positive scaling factor. As hardness tends to infinity the
            soft minimum (logsumexp) approaches the hard minimum. Default is 1. A
            detailed explanation is given in the docstring.
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies lower_bounds <= existing_xs <= upper_bounds.
        algorithm (str): Optimization algorithm.
        algo_options (dict): Algorithm specific configuration of the optimization. See
            :ref:`list_of_algorithms` for supported options of each algorithm. Default
            sets ``stopping_max_iterations=n_params``.
        criterion (str or None): "distance", "determinant" or None.
            - "distance": maximize the minimal distance between points, excluding
              distances between existing points. This is a fast and relatively simple
              optimization problem and yields the same points as "determinant" in
              many circumstances.
            - "determinant": maximize the determinant of the x'x where x is the matrix
              of points. This is known as d-optimality in the optimal design literature
              and as fekete points in the function approximation literature. This
              criterion has the best theoretical properties but is very hard to
              optimize. Thus the practical performance can be bad.
            - None: Use the "determinant" criterion if only one point is added and the
              "distance" criterion if multiple points are added.
        n_points_randomsearch (int): Number of random points to from which to select
            the best in terms of the Fekete criterion before starting the optimization.
            Default is 1.

    Returns:
        - np.ndarray: Generated points. Has shape (n_points, len(trustregion.center)).
        - dict: Information about the optimization. Only returned if ``return_info`` is
        True.

    """
    n_params = len(trustregion.center)

    if n_points <= 0:
        return np.array([])

    if criterion is None:
        if n_points == 1:
            criterion = "determinant"
        else:
            criterion = "distance"

    algo_options = {} if algo_options is None else algo_options
    if "stopping_max_iterations" not in algo_options:
        algo_options["stopping_max_iterations"] = 2 * n_params + 5

    if existing_xs is not None:
        # map existing points into unit space for easier optimization

        existing_xs_unit = trustregion.map_to_unit(existing_xs)

        if criterion == "distance":
            dist_to_center = np.linalg.norm(existing_xs_unit, axis=1)
            not_centric = dist_to_center >= 0.1
            if not_centric.any():
                existing_xs_unit = existing_xs_unit[not_centric]
            else:
                existing_xs_unit = None

    else:
        existing_xs_unit = None

    # Define criterion functions. "determinant" is the Fekete criterion and "distance"
    # corresponds to an approximation of the Fekete criterion.
    criterion_kwargs = {
        "existing_xs": existing_xs_unit,
        "order": order,
        "n_params": n_params,
    }

    func_dict = {
        "determinant": partial(_determinant_on_hull, **criterion_kwargs),
        "distance": partial(
            _minimal_pairwise_distance_on_hull,
            **criterion_kwargs,
            hardness=hardness,
        ),
    }

    # Select start params through random search
    if distribution is None:
        distribution = "normal" if order <= 3 else "uniform"
    candidates = _draw_from_distribution(
        distribution, rng=rng, size=(n_points_randomsearch, n_points, n_params)
    )
    candidates = [_project_onto_unit_hull(_x, order=order) for _x in candidates]

    if n_points_randomsearch == 1:
        x0 = candidates[0]
    else:
        _fekete_criterion = [func_dict["determinant"](_x) for _x in candidates]
        x0 = candidates[np.argmax(_fekete_criterion)]

    x0 = x0.flatten()  # flatten so that em.maximize uses fast path

    # This would raise an error because there are zero pairs to calculate the
    # pairwise distance
    if existing_xs_unit is None and n_points == 1:
        opt_params = x0
    else:
        res = em.maximize(
            criterion=func_dict[criterion],
            params=x0,
            algorithm=algorithm,
            lower_bounds=-np.ones_like(x0),
            upper_bounds=np.ones_like(x0),
            algo_options=algo_options,
        )
        opt_params = res.params

    # Make sure the optimal sampling is actually better than the initial one with
    # respect to the fekete criterion. This could be violated if the surrogate
    # criterion is not a good approximation or if the optimization fails.
    start_fekete = func_dict["determinant"](x0)
    end_fekete = func_dict["determinant"](opt_params)

    if start_fekete >= end_fekete:
        opt_params = x0

    points = _project_onto_unit_hull(opt_params.reshape(-1, n_params), order=order)
    points = trustregion.map_from_unit(points)

    # Collect additional information. Mostly used for testing.
    info = {
        "start_params": x0,
        "opt_params": opt_params,
        "start_fekete": start_fekete,
        "opt_fekete": end_fekete,
    }

    out = (points, info) if return_info else points
    return out


# ======================================================================================
# Helper functions
# ======================================================================================


def _minimal_pairwise_distance_on_hull(x, existing_xs, order, hardness, n_params):
    """Compute minimal pairwise distance of new and existing points.

    Instead of optimizing the distance of points in the feasible trustregion, this
    criterion function leads to the maximization of the minimum distance of the points
    in the unit space. These can then be mapped into the feasible trustregion. We do not
    consider the distances between existing points. Instead of using a hard minimum we
    return the soft minimum, whose accuracy we govern by the hardness factor. For more
    information on the soft-minimum, seek: https://tinyurl.com/mrythbk4.

    Args:
        x (np.ndarray): Flattened 1d array of internal points. Each value is in [-1, 1].
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies -1 <= existing_xs <= 1.
        order (int): Type of norm to use when scaling the sampled points. For 2 we
            project onto the hull of a sphere, for np.inf onto the hull of a cube.
        hardness (float): Positive scaling factor. As hardness tends to infinity the
            soft minimum (logsumexp) approaches the hard minimum. Default is 1. A
            detailed explanation is given in the docstring.
        n_params (int): Dimensionality of the problem.

    Returns:
        float: The criterion value.

    """
    x = x.reshape(-1, n_params)
    x = _project_onto_unit_hull(x, order=order)

    if existing_xs is not None:
        sample = np.row_stack([x, existing_xs])
        n_existing_pairs = len(existing_xs) * (len(existing_xs) - 1) // 2
        slc = slice(0, -n_existing_pairs) if n_existing_pairs else slice(None)
    else:
        sample = x
        slc = slice(None)

    dist = pdist(sample) ** 2

    # drop distances between existing points. They could introduce flat spots.
    dist = dist[slc]

    # soft minimum
    crit_value = -logsumexp(-hardness * dist)
    return crit_value


def _determinant_on_hull(x, existing_xs, order, n_params):
    """Compute d-optimality criterion of new and existing points.

    Instead of optimizing the distance of points in the feasible trustregion, this
    criterion function leads to the maximization of the minimum distance of the points
    in the unit space.

    Args:
        x (np.ndarray): Flattened 1d array of internal points. Each value is in [-1, 1].
        existing_xs (np.ndarray or None): 2d numpy array in which each row is an
            x vector at which the criterion function has already been evaluated, that
            satisfies -1 <= existing_xs <= 1.
        order (int): Type of norm to use when scaling the sampled points. For 2 we
            project onto the hull of a sphere, for np.inf onto the hull of a cube.
        n_params (int): Dimensionality of the problem.

    Returns:
        float: The criterion value.

    """
    x = x.reshape(-1, n_params)
    n_samples = len(x)

    x = _project_onto_unit_hull(x, order=order)

    if existing_xs is not None:
        sample = np.row_stack([x, existing_xs])
    else:
        sample = x

    crit_value = np.linalg.det(sample.T @ sample / n_samples)

    return crit_value


def _draw_from_distribution(distribution, rng, size):
    """Draw points from distribution.

    Args:
        distribution (str): Distribution to use for initial sample before points are
            projected onto unit hull. Must be in {'normal', 'uniform'}.
        rng (np.random.Generator): Random number generator.
        size (Union[int, tuple[int]]): Output shape.

    Returns:
        np.ndarray: Randomly drawn points.

    """
    if distribution == "normal":
        draw = rng.normal(size=size)
    elif distribution == "uniform":
        draw = rng.uniform(-1, 1, size=size)
    else:
        raise ValueError(
            f"distribution is {distribution}, but needs to be in ('normal', 'uniform')."
        )
    return draw


def _project_onto_unit_hull(x, order):
    """Project points from the unit space onto the hull of a geometric figure.

    Args:
        x (np.ndarray): 2d array of points to be projects. Each value is in [-1, 1].
        order (int): Type of norm to use when scaling the sampled points. For 2 we
            project onto the hull of a sphere, for np.inf onto the hull of a cube.

    Returns:
        np.ndarray: The projected points.

    """
    norm = np.linalg.norm(x, axis=1, ord=order).reshape(-1, 1)
    projected = x / norm
    return projected
