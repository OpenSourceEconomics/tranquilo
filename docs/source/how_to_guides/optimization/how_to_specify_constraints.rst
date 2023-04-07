
.. _constraints:

===========================
How to specify constraints
===========================

Constraints vs bounds
=====================

Estimagic distinguishes between bounds and constraints. Bounds are lower and upper
bounds for parameters. In the literature they are sometimes called box constraints.
Bounds are specified as "lower_bound" and "upper_bound" column of a params DataFrame
or as pytrees via the ``lower_bounds`` and ``upper_bounds`` argument to ``maximize`` and
``minimize``.

Examples with bounds can be found in `this tutorial`_.

.. _this tutorial: ../../getting_started/first_optimization_with_estimagic.ipynb

Constraints are more general constraints on the parameters. This ranges from rather
simple ones (e.g. Parameters are fixed to a value, a group of parameters is required
to be equal) to more complex ones (like general linear constraints).

Can you use constraints with all optimizers?
============================================

We implement constraints via reparametrizations. Details are explained `here`_. This
means that you can use all of the constraints with any optimizer that supports
bounds. Some constraints (e.g. fixing parameters) can even be used with optimizers
that do not support bounds.

.. _here: ../../explanations/optimization/implementation_of_constraints.rst


Example criterion function
==========================

Let's look at a variation of the sphere function to illustrate which consraints are
implemented and how you specify them in estimagic:

.. code-block:: python

    def criterion(params):
        offset = np.linspace(1, 0, len(params))
        x = params - offset
        return x @ x

The unconstrained optimum of a six-dimensional version of this problem is:

.. code-block:: python

    [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

The constrained optimum is usually also easy to see because all parameters enter the
criterion function in a additively separable way.

Types of constraints
====================

Below we show a very simple example of each type of constraint implemented in estimagic.
In each constraint we will select a subset of the parameters on which the constraint
is imposed via "loc". Generalizations to select subsets of ``params`` that are not a
flat numpy array are explained in the next section.


.. tabbed:: fixed

    The simplest (but very useful) constraint fixes parameters at their start values.

    Let's take the above example and fix the first and last parameter to 2.5 and
    -2.5, respectively.

    .. code-block:: python

        res = minimize(
            criterion=criterion,
            params=np.array([2.5, 1, 1, 1, 1, -2.5]),
            algorithm="scipy_lbfgsb",
            constraints=[{"loc": [0, 5], "type": "fixed"}],
        )

    Looking at the optimization result we get:

    >>> array([ 2.5,  0.8,  0.6,  0.4,  0.2, -2.5])

    Which is indeed the correct constrained optimum. Fixes are compatible with all
    optimizers.


.. tabbed:: increasing

    In our example, the unconstrained optimal parameters are decreasing from left to
    right. Let's impose the constraint that the second, third and fourth parameter
    decrease (weakly):

    .. code-block:: python

        res = minimize(
            criterion=criterion,
            params=np.array([1, 1, 1, 1, 1, 1]),
            algorithm="scipy_lbfgsb",
            constraints=[{"loc": [1, 2, 3], "type": "increasing"}],
        )

    Looking at the optimization result we get:


    >>> array([1. , 0.6, 0.6, 0.6, 0.2, 0. ])

    Which is indeed the correct constrained optimum. Decreasing constraints are only
    compatible with optimizers that support bounds.


.. tabbed:: decreasing

    In our example, the unconstrained optimal parameters are decreasing from left to
    right without any constraints. If we imposed a decreasing constraint without
    changing the order, it would have no effect.

    So let's impose one in a different order:

    .. code-block:: python

        res = minimize(
            criterion=criterion,
            params=np.array([1, 1, 1, 1, 1, 1]),
            algorithm="scipy_lbfgsb",
            constraints=[{"loc": [3, 0, 4], "type": "decreasing"}],
        )

    This should have no effect on ``params[4]`` because it is smaller than the other
    two anyways in the unconstrained optimum, but it will change the optimal values of
    ``params[3]`` and ``params[0]``. Indeed we get:

    >>> array([ 0.7,  0.8,  0.6,  0.7,  0.2, -0. ])

    Which is the correct optimum. Decreasing constraints are only compatible with
    optimizers that support bounds.

.. tabbed:: equality

    In our example, all optimal parameters are different. Let's constraint the first
    and last to be equal to each other:

    .. code-block:: python

        res = minimize(
            criterion=criterion,
            params=np.array([1, 1, 1, 1, 1, 1]),
            algorithm="scipy_lbfgsb",
            constraints=[{"loc": [0, 5], "type": "equality"}],
        )

    This yields:

    >>> array([0.5, 0.8, 0.6, 0.4, 0.2, 0.5])

    Which is the correct solution. Equality constraints are compatible with all
    optimizers.


.. tabbed:: pairwise_equality

    Pairwise equality constraints are similar to equality constraints but impose that
    two or more groups of parameters are pairwise equal. Let's look at an example:

    .. code-block:: python

        res = minimize(
            criterion=criterion,
            params=np.array([1, 1, 1, 1, 1, 1]),
            algorithm="scipy_lbfgsb",
            constraints=[{"locs": [[0, 1], [2, 3]], "type": "pairwise_equality"}],
        )

    This constraint imposes that ``params[0] == params[2]`` and
    ``params[1] == params[3]``. The optimal parameters with this constraint are:

    >>> array([ 0.8,  0.6,  0.8,  0.6,  0.2, -0. ])


.. tabbed:: probability

    Let's impose the constraint that the first four parameters form valid
    probabilities, i.e. they should add up to one and be between zero and one.

    .. code-block:: python

        res = minimize(
            criterion=criterion,
            params=np.full(6, 0.25),
            algorithm="scipy_lbfgsb",
            constraints=[{"loc": [0, 1, 2, 3], "type": "probability"}],
        )

    This yields again the correct result:

    >>> array([0.527, 0.333, 0.14 , 0.   , 0.2  , 0.   ])



.. tabbed:: covariance

    In many estimation problems, particularly when maximum likelihood estimation is
    used, one has to estimate the covariance matrix of a random variable. The
    ``covariance`` costraint ensures that such a covariance matrix is always valid,
    i.e. positive semi-definite and symmetric. Due to the symmetry, only the lower
    triangle of a covariance matrix actually has to be estimated.

    Let's look at an example. We want to impose that the first three elements form the
    lower triangle of a valid covariance matrix.

    .. code-block:: python

        res = minimize(
            criterion=criterion,
            params=np.ones(6),
            algorithm="scipy_lbfgsb",
            constraints=[{"loc": [0, 1, 2], "type": "covariance"}],
        )

    This yields the same solution as an unconstrained estimation because the constraint
    is not binding:

    >>> array([ 1. ,  0.8,  0.6,  0.4,  0.2, -0. ])

    We can now use one of estimagic's utility functions to actually build the covariance
    matrix out of the first three parameters:

    .. code-block:: python

        from estimagic.utilities import cov_params_to_matrix

        cov_params_to_matrix(res.params[:3]).round(3)

    This yields:

    >>> array([[1. , 0.8], [0.8, 0.6]])


.. tabbed:: sdcorr

    ``sdcorr`` constraints are very similar to ``covariance`` constraints. The only
    difference is that instead of estimating a covariance matrix, we estimate
    standard deviations and the correlation matrix of a random variable.

    Let's look at an example. We want to impose that the first three elements form valid
    standard deviations and a correlation matrix.

    .. code-block:: python

        res = minimize(
            criterion=criterion,
            params=np.ones(6),
            algorithm="scipy_lbfgsb",
            constraints=[{"loc": [0, 1, 2], "type": "sdcorr"}],
        )


    This yields the same solution as an unconstrained estimation because the constraint
    is not binding:

    >>> array([ 1. ,  0.8,  0.6,  0.4,  0.2, -0. ])

    We can now use one of estimagic's utility functions to actually build the standard
    deviations and the correlation matrix:

    .. code-block:: python

        from estimagic.utilities import sdcorr_params_to_sds_and_corr

        sdcorr_params_to_sds_and_corr(res.params[:3])

    This yields:

    >>> (array([1, 0.8]) array([[1. , 0.6], [0.6, 1]]))


.. tabbed:: linear

    Linear constraints are the most difficult but also most powerfull ones. They
    can be used to express constraints of the form
    ``lower_bound <= weights.dot(x) <= upper_bound`` or
    ``weights.dot(x) = value`` where ``x`` are the selected parameters.

    Linear constraints have many of the other constraint types as special cases, but
    typically it is more convenient to use the special cases instead of expressing
    them as a linear constraint. Internally, it will make no difference.

    Let's impose the constraint that the sum of the average of the first four parameters
    is at least 0.95.

    .. code-block:: python

        res = minimize(
            criterion=criterion,
            params=np.ones(6),
            algorithm="scipy_lbfgsb",
            constraints=[
                {"loc": [0, 1, 2, 3], "type": "linear", "lower_bound": 0.95, "weights": 0.25}
            ],
        )

    This yields:

    >>> array([ 1.25,  1.05,  0.85,  0.65,  0.2 , -0.  ])

    Where the first parameters have an average of 0.95.

    In the above example, ``lower_bound`` and ``weights`` were scalars. Instead they
    can also be arrays (or even pytrees) with bounds and weights for each selected
    parameter.



Imposing multiple constraints at once
=====================================

The above examples all just impose one constraint at a time. To impose multiple
constraints simultaneously, simple pass in a longer list of constraints. Example:

.. code-block:: python

    res = minimize(
        criterion=criterion,
        params=np.ones(6),
        algorithm="scipy_lbfgsb",
        constraints=[
            {"loc": [0, 1], "type": "equality"},
            {"loc": [2, 3, 4], "type": "linear", "weights": 1, "value": 3},
        ],
    )

This yields:

>>> array([0.9, 0.9, 1.2, 1. , 0.8, 0. ])

There are limits regarding the compatibility of constraints that overlap. You will
get a descriptive error message if your constraints are not compatible.


How to select the parameters?
=============================

All the above examples use a ``loc`` entry in the constraint dictionary to select
the subset of ``params`` on which the constraint is imposed. This is just one out
of several ways to do it. Which ways are available also depends on whether your
parameters are a numpy array, DataFrame or general pytree.


+---------------+---------------+----------------+---------------+
|               | loc           | query          | selector      |
+---------------+---------------+----------------+---------------+
| 1d-array      | ✅ (positions)| ❌             | ✅            |
+---------------+---------------+----------------+---------------+
| DataFrame     | ✅ (labels)   | ✅             | ✅            |
+---------------+---------------+----------------+---------------+
| Pytree        | ❌            | ❌             | ✅            |
+---------------+---------------+----------------+---------------+

Below we show how to use each of these selection methods in simple examples


.. tabbed:: loc

    You can look at any of the above examples to see constraints where params are
    a numpy array and ``loc`` is used to select parameters. So now, we focus on
    DataFrame params.

    Let's assume our ``params`` are a DataFrame with a two level index. The names of
    the index levels are ``category`` and ``name``. Something like this could for
    example be the params of an Ordered Logit model.

    +----------------+---------------+----------------+
    |                |               | **value**      |
    +----------------+---------------+----------------+
    | **category**   | **name**      |                |
    +----------------+---------------+----------------+
    | **betas**      | **a**         | 0.95           |
    +----------------+---------------+----------------+
    | **betas**      | **b**         | 0.9            |
    +----------------+---------------+----------------+
    | **cutoffs**    | **a**         | 0              |
    +----------------+---------------+----------------+
    | **cutoffs**    | **b**         | 0.4            |
    +----------------+---------------+----------------+

    Now let's impose the constraint that the cutoffs (i.e. the last two parameters)
    are increasing.

    .. code-block:: python

        res = minimize(
            criterion=some_criterion,
            params=params,
            algorithm="scipy_lbfgsb",
            constraints=[{"loc": "cutoffs", "type": "increasing"}],
        )

    The value corresponding to ``loc`` can be anything that you could pass into the
    ``DataFrame.loc`` method. This can be extremely powerful if you have a well
    designed MultiIndex, as you can easily select groups of parameters or single
    paramaters.


.. tabbed:: query

    Let's assume our ``params`` are a DataFrame with a two level index. The names of
    the index levels are ``category`` and ``name``. Something like this could for
    example be the params of an Ordered Logit model.

    +----------------+---------------+----------------+
    |                |               | **value**      |
    +----------------+---------------+----------------+
    | **category**   | **name**      |                |
    +----------------+---------------+----------------+
    | **betas**      | **a**         | 0.95           |
    +----------------+---------------+----------------+
    | **betas**      | **b**         | 0.9            |
    +----------------+---------------+----------------+
    | **cutoffs**    | **a**         | 0              |
    +----------------+---------------+----------------+
    | **cutoffs**    | **b**         | 0.4            |
    +----------------+---------------+----------------+

    This time we want to fix all betas as well as all parameters where the second index
    level is equal to ``"a"``. If we wanted to do that using ``loc``, we would have to
    type out three index tuples. So let's do it with query:

    .. code-block:: python

        res = minimize(
            criterion=some_criterion,
            params=params,
            algorithm="scipy_lbfgsb",
            constraints=[{"query": "category == 'betas' | name == 'a'", "type": "fixed"}],
        )

    The value corresponding to ``query`` can be anything you could pass to the
    ``DataFrame.query`` method.


.. tabbed:: selector

    Using ``selector`` to select the parameters is the most general way and works for
    all params. Let's assume we have defined parameters in a nested dictionary:

    .. code-block:: python

        params = {"a": np.ones(2), "b": {"c": 3, "d": pd.Series([4, 5])}}

    It is probably not a good idea to use a nested dictionary for so few parameters, but
    let's ignore that.

    Now assume, we want to fix the parameters in the pandas Series at their start
    values. We can do so as follows:

    .. code-block:: python

        res = minimize(
            criterion=some_criterion,
            params=params,
            algorithm="scipy_lbfgsb",
            constraints=[{"selector": lambda params: params["b"]["d"], "type": "fixed"}],
        )

    I.e. the value corresponding to ``selector`` is a python function that takes the
    full ``params`` and returns a subset. The selected subset does not have to be a
    numpy array, it can be an arbitrary pytree.

    Using lambda functions if often convenient, but we could have just as well defined
    the selector function using def.

    .. code-block:: python

        def my_selector(params):
            return params["b"]["d"]


        res = minimize(
            criterion=some_criterion,
            params=params,
            algorithm="scipy_lbfgsb",
            constraints=[{"selector": my_selector, "type": "fixed"}],
        )
