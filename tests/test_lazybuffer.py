"""Tests for core/lazybuffer.py, the numpy backend.

No gradients here. Just checking the six primitives compute the right values and
enforce the right contracts, since every layer above assumes they do.
"""

import numpy as np
import pytest

from teenygrad import BinaryOps, LazyBuffer, ReduceOps, UnaryOps


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [[1, 2, 3], np.array([1, 2, 3], dtype=np.int64), 5.0, [[1.0], [2.0]]],
)
def test_buffer_always_holds_float32(source):
    """Every buffer is float32, whatever it was constructed from."""
    assert LazyBuffer(source).dtype == np.float32


def test_buffer_shape_is_plain_ints():
    """Shapes are tuples of Python ints, not numpy scalars."""
    shape = LazyBuffer(np.zeros((2, 3))).shape
    assert shape == (2, 3)
    assert all(type(dim) is int for dim in shape)


def test_const_fills_a_shape():
    """const builds the constant buffers that backward passes multiply against."""
    assert np.array_equal(LazyBuffer.const(7, (2, 2))._np, np.full((2, 2), 7.0))
    assert LazyBuffer.const(0, ()).shape == ()


def test_rand_is_in_range_and_reproducible():
    """rand is uniform in [0, 1) and the same seed gives the same draw."""
    sample = LazyBuffer.rand((3, 3), seed=0)._np
    assert sample.shape == (3, 3)
    assert (sample >= 0).all() and (sample < 1).all()
    assert np.array_equal(sample, LazyBuffer.rand((3, 3), seed=0)._np)


# ---------------------------------------------------------------------------
# Elementwise ops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op, expected",
    [
        (UnaryOps.NEG, [-1.0, 2.0, -3.0]),
        (UnaryOps.RELU, [1.0, 0.0, 3.0]),
        (UnaryOps.EXP, np.exp([1.0, -2.0, 3.0])),
    ],
)
def test_unary_ops(op, expected):
    """Unary elementwise ops compute the expected values."""
    result = LazyBuffer([1.0, -2.0, 3.0]).e(op)
    assert np.allclose(result._np, expected, atol=1e-5)


@pytest.mark.parametrize(
    "op, expected",
    [
        (UnaryOps.LOG, np.log([1.0, 4.0, 9.0])),
        (UnaryOps.SQRT, [1.0, 2.0, 3.0]),
    ],
)
def test_unary_ops_on_positive_input(op, expected):
    """log and sqrt need a positive domain."""
    result = LazyBuffer([1.0, 4.0, 9.0]).e(op)
    assert np.allclose(result._np, expected, atol=1e-5)


def test_sigmoid_does_not_overflow():
    """The two-branch sigmoid stays finite at extreme values.

    The naive 1/(1+exp(-x)) gives inf (plus a RuntimeWarning) around x = -100 in
    float32.
    """
    extreme = LazyBuffer([-500.0, -100.0, 0.0, 100.0, 500.0])
    result = extreme.e(UnaryOps.SIGMOID)._np
    assert np.isfinite(result).all()
    assert result[2] == pytest.approx(0.5)
    assert result[0] == pytest.approx(0.0)
    assert result[4] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "op, expected",
    [
        (BinaryOps.ADD, [5.0, 7.0]),
        (BinaryOps.SUB, [-3.0, -3.0]),
        (BinaryOps.MUL, [4.0, 10.0]),
        (BinaryOps.MAX, [4.0, 5.0]),
    ],
)
def test_binary_ops(op, expected):
    """Binary elementwise ops compute the expected values."""
    result = LazyBuffer([1.0, 2.0]).e(op, LazyBuffer([4.0, 5.0]))
    assert np.allclose(result._np, expected)


def test_cmplt_returns_floats_not_booleans():
    """Comparison yields 1.0/0.0 so backward passes can do arithmetic with it."""
    mask = LazyBuffer([1.0, 5.0]).e(BinaryOps.CMPLT, LazyBuffer([3.0, 3.0]))
    assert mask.dtype == np.float32
    assert np.array_equal(mask._np, [1.0, 0.0])


def test_binary_ops_refuse_mismatched_shapes():
    """The buffer layer won't silently broadcast. That's Tensor's job.

    Raising here is what forces broadcasting through real expand nodes, which is
    what makes it differentiable.
    """
    with pytest.raises(ValueError, match="matching shapes"):
        LazyBuffer(np.zeros((2, 3))).e(BinaryOps.ADD, LazyBuffer(np.zeros((3,))))


def test_arity_mistakes_are_reported():
    """Passing the wrong number of sources raises instead of misbehaving."""
    with pytest.raises(ValueError, match="unary"):
        LazyBuffer([1.0]).e(UnaryOps.NEG, LazyBuffer([1.0]))
    with pytest.raises(ValueError, match="binary"):
        LazyBuffer([1.0]).e(BinaryOps.ADD)


# ---------------------------------------------------------------------------
# Reduce and movement ops
# ---------------------------------------------------------------------------


def test_reduce_keeps_dimensions():
    """Reductions keep the reduced axes at length 1, so expand can undo them."""
    buffer = LazyBuffer(np.arange(6, dtype=np.float32).reshape(2, 3))
    assert buffer.r(ReduceOps.SUM, (1,)).shape == (2, 1)
    assert buffer.r(ReduceOps.MAX, (0, 1)).shape == (1, 1)
    assert buffer.r(ReduceOps.SUM, (0, 1))._np.item() == pytest.approx(15.0)


def test_reduce_over_no_axes_is_a_no_op():
    """An empty axis tuple is fine. Expand.backward relies on it."""
    buffer = LazyBuffer(np.arange(6, dtype=np.float32).reshape(2, 3))
    assert np.array_equal(buffer.r(ReduceOps.SUM, ())._np, buffer._np)


def test_expand_returns_a_writable_copy():
    """expand copies rather than returning numpy's read-only broadcast view."""
    result = LazyBuffer(np.zeros((1, 3))).expand((2, 3))
    assert result.shape == (2, 3)
    result._np[0, 0] = 9.0  # would raise on a broadcast view
    assert result._np[1, 0] == 0.0  # and the rows must be independent


def test_reshape_and_permute():
    """Movement ops move elements without changing their values."""
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    buffer = LazyBuffer(values)
    assert buffer.reshape((3, 2)).shape == (3, 2)
    assert np.array_equal(buffer.permute((1, 0))._np, values.T)


# ---------------------------------------------------------------------------
# Interop
# ---------------------------------------------------------------------------


def test_numpy_interop():
    """np.asarray and float() work on a buffer."""
    values = np.array([[1.0, 2.0]], dtype=np.float32)
    assert np.array_equal(np.asarray(LazyBuffer(values)), values)
    assert float(LazyBuffer([3.5])) == 3.5
