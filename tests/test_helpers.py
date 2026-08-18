"""Tests for core/helpers.py."""

import pytest

from teenygrad import argsort, prod


@pytest.mark.parametrize(
    "shape, expected",
    [((2, 3, 4), 24), ((5,), 5), ((), 1), ((1, 1), 1), ((0, 3), 0)],
)
def test_prod(shape, expected):
    """prod multiplies a shape out; the empty shape gives 1, a zero dim gives 0."""
    assert prod(shape) == expected


@pytest.mark.parametrize(
    "values, expected",
    [
        ([2, 0, 1], [1, 2, 0]),
        ([0, 1, 2], [0, 1, 2]),
        ([3, 2, 1, 0], [3, 2, 1, 0]),
        ([], []),
    ],
)
def test_argsort(values, expected):
    """argsort returns the indices that sort ascending."""
    assert argsort(values) == expected


def test_argsort_is_stable():
    """Equal values keep their original relative order."""
    assert argsort([1, 0, 1, 0]) == [1, 3, 0, 2]


def test_argsort_inverts_a_permutation():
    """The property Permute.backward depends on: argsort undoes a reordering."""
    order = [2, 0, 3, 1]
    inverse = argsort(order)
    assert [order[i] for i in inverse] == sorted(order)
