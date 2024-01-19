from tranquilo.filter_points import get_sample_filter
from tranquilo.filter_points import drop_worst_points
from tranquilo.tranquilo import State
from tranquilo.region import Region
from numpy.testing import assert_array_equal as aae
import pytest
import numpy as np


@pytest.fixture()
def state():
    out = State(
        trustregion=Region(center=np.ones(2), radius=0.3),
        model_indices=None,
        model=None,
        vector_model=None,
        candidate_index=5,
        candidate_x=np.array([1.1, 1.2]),
        index=2,
        x=np.ones(2),
        fval=15,
        rho=None,
        accepted=True,
        old_indices_used=None,
        old_indices_discarded=None,
        new_indices=None,
        step_length=0.1,
        relative_step_length=0.1 / 0.3,
    )
    return out


def test_discard_all(state):
    filter = get_sample_filter("discard_all")
    xs = np.arange(10).reshape(5, 2)
    indices = np.arange(5)
    got_xs, got_idxs = filter(xs=xs, indices=indices, state=state)
    expected_xs = np.ones((1, 2))
    aae(got_xs, expected_xs)
    aae(got_idxs, np.array([2]))


def test_keep_all():
    filter = get_sample_filter("keep_all")
    xs = np.arange(10).reshape(5, 2)
    indices = np.arange(5)
    got_xs, got_idxs = filter(xs=xs, indices=indices, state=None)
    aae(got_xs, xs)
    aae(got_idxs, indices)


def test_drop_worst_point(state):
    xs = np.array(
        [
            [1, 1.1],  # should be dropped
            [1, 1.2],
            [1, 1],  # center (needs to have index=2)
            [3, 3],  # should be dropped
        ]
    )

    got_xs, got_indices = drop_worst_points(
        xs, indices=np.arange(4), state=state, n_to_drop=2
    )

    expected_xs = np.array(
        [
            [1, 1.2],
            [1, 1],
        ]
    )
    expected_indices = np.array([1, 2])

    aae(got_xs, expected_xs)
    aae(got_indices, expected_indices)


def test_drop_excess(state):
    filter = get_sample_filter("drop_excess", user_options={"n_max_factor": 1.0})

    xs = np.array(
        [
            [1, 1.1],  # should be dropped
            [1, 1.2],
            [1, 1],  # center (needs to have index=2)
            [3, 3],  # should be dropped
        ]
    )

    got_xs, got_indices = filter(xs, indices=np.arange(4), state=state, target_size=2)

    expected_xs = np.array(
        [
            [1, 1.2],
            [1, 1],
        ]
    )
    expected_indices = np.array([1, 2])

    aae(got_xs, expected_xs)
    aae(got_indices, expected_indices)
