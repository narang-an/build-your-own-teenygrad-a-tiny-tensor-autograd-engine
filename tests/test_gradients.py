"""Gradient checks for every differentiable op.

Each test builds a loss out of one op and checks the analytic gradient against a
finite-difference estimate. If a backward rule is wrong, one of these fails
loudly instead of quietly making training a bit worse.

The inputs are picked to stay away from each op's kinks: positive values for log
and sqrt, nothing near zero for relu, no ties for max. Those kinks are real
maths, not bugs, but finite differences straddle them and report nonsense.

Whether the values are right (does matmul actually compute a matmul?) is in
test_tensor.py. This file only asks whether the derivative is right.
"""

import numpy as np
import pytest

from teenygrad import Tensor
from tests.support.gradcheck import assert_gradients_match


def tensor_from(values):
    """Build a plain float32 Tensor from a nested list."""
    return Tensor(np.array(values, dtype=np.float32))


def random_tensor(*shape, seed):
    """A reproducible random Tensor of the given shape."""
    return Tensor(np.random.RandomState(seed).randn(*shape).astype(np.float32))


# Values spread across positive and negative, none near zero, no ties.
MIXED = [[1.5, -2.25, 0.75], [-0.5, 3.0, -1.75]]
# Strictly positive, for ops whose domain requires it.
POSITIVE = [[0.5, 1.5, 2.5], [3.5, 0.25, 4.0]]


# ---------------------------------------------------------------------------
# Unary ops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op_name, sample",
    [
        ("neg", MIXED),
        ("relu", MIXED),
        ("exp", MIXED),
        ("sigmoid", MIXED),
        ("log", POSITIVE),
        ("sqrt", POSITIVE),
    ],
)
def test_unary_op_gradients(op_name, sample):
    """Each unary op's backward matches finite differences."""
    x = tensor_from(sample)
    assert_gradients_match(lambda t: getattr(t, op_name)().sum(), [x])


def test_relu_blocks_gradient_where_it_clamped():
    """Negative inputs get exactly zero gradient, positive ones get exactly one."""
    x = tensor_from([-2.0, -0.5, 0.5, 2.0])
    x.requires_grad = True
    x.relu().sum().backward()
    assert np.array_equal(x.grad.numpy(), [0.0, 0.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Binary ops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", ["add", "sub", "mul", "div"])
def test_binary_op_gradients(op_name):
    """Each binary op produces correct gradients for *both* inputs."""
    x = tensor_from([[1.5, -2.0], [3.0, 0.5]])
    # Kept away from zero so that div's 1/y and -x/y^2 stay well conditioned.
    y = tensor_from([[2.0, 4.0], [-3.0, 1.5]])
    assert_gradients_match(lambda a, b: getattr(a, op_name)(b).sum(), [x, y])


def test_no_gradient_is_computed_for_untracked_inputs():
    """needs_input_grad suppresses work nobody asked for."""
    tracked = tensor_from([1.0, 2.0])
    tracked.requires_grad = True
    untracked = tensor_from([3.0, 4.0])

    tracked.mul(untracked).sum().backward()
    assert tracked.grad is not None
    assert untracked.grad is None


def test_gradient_accumulates_when_a_tensor_is_used_twice():
    """A tensor feeding two branches collects gradient from both.

    x * x reaches Mul through both inputs, so d/dx has to be 2x, not x. This is
    why backward accumulates into .grad instead of overwriting, and why
    zero_grad has to exist.
    """
    x = tensor_from([2.0, 3.0, -4.0])
    x.requires_grad = True
    x.mul(x).sum().backward()
    assert np.allclose(x.grad.numpy(), 2 * np.array([2.0, 3.0, -4.0]), atol=1e-5)


def test_backward_reaches_through_a_deep_chain():
    """Gradient makes it through a stack of ops, not just one.

    (3x + 1)^2 at x = 2, so dL/dx = 2*(3*2+1)*3 = 42.
    """
    x = tensor_from([2.0])
    x.requires_grad = True
    inner = x.mul(3.0).add(1.0)
    inner.mul(inner).sum().backward()
    assert x.grad.item() == pytest.approx(42.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "small_shape",
    [(3,), (1, 3), (2, 1), (1, 1)],
    ids=["row-vector", "explicit-row", "column", "scalar-like"],
)
def test_broadcast_gradients_reduce_to_the_smaller_shape(small_shape):
    """A broadcast operand gets its gradient summed back down.

    This was silently broken before broadcasting went through real expand nodes.
    The small operand has to come back with its own shape, holding gradient from
    every position it got copied into.
    """
    big = random_tensor(2, 3, seed=0)
    small = random_tensor(*small_shape, seed=1)

    assert_gradients_match(lambda a, b: a.mul(b).sum(), [big, small])
    assert small.grad.shape == small_shape


def test_broadcast_rejects_incompatible_shapes():
    """A genuine shape mismatch raises rather than silently doing something."""
    with pytest.raises(ValueError, match="cannot broadcast"):
        Tensor(np.zeros((2, 3), dtype=np.float32)).add(
            Tensor(np.zeros((4,), dtype=np.float32))
        )


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 1)])
@pytest.mark.parametrize("keepdim", [False, True])
def test_sum_gradients(axis, keepdim):
    """Sum's gradient is 1 everywhere, for every axis spec.

    Also the regression test for a bug I found while cleaning this up:
    Sum.backward was written as a loose module-level function and never actually
    attached to the class, so summing anything and calling .backward() blew up
    with AttributeError.
    """
    x = random_tensor(3, 4, seed=2)
    assert_gradients_match(lambda t: t.sum(axis=axis, keepdim=keepdim).sum(), [x])
    assert np.allclose(x.grad.numpy(), np.ones((3, 4)), atol=1e-5)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_max_gradients(axis):
    """Max routes gradient only to the winning elements."""
    # Distinct values, so there are no ties to split.
    x = tensor_from([[1.0, 5.0, 3.0], [7.0, 2.0, 4.0]])
    assert_gradients_match(lambda t: t.max(axis=axis).sum(), [x])


def test_max_splits_gradient_between_tied_elements():
    """Tied maxima split the gradient evenly.

    Finite differences can't check this one, since the kink is exactly at the
    tie, so the expected value is written out directly.
    """
    x = tensor_from([[4.0, 4.0, 1.0]])
    x.requires_grad = True
    x.max(axis=1).sum().backward()
    assert np.allclose(x.grad.numpy(), [[0.5, 0.5, 0.0]], atol=1e-6)


def test_mean_gradients():
    """Mean spreads gradient evenly, scaled by 1/count."""
    x = random_tensor(4, 5, seed=3)
    assert_gradients_match(lambda t: t.mean().sum(), [x])
    assert np.allclose(x.grad.numpy(), np.full((4, 5), 1 / 20), atol=1e-5)


# ---------------------------------------------------------------------------
# Movement ops
# ---------------------------------------------------------------------------


def test_reshape_gradients():
    """Reshape's gradient is the reverse reshape: values kept, shape restored."""
    x = random_tensor(2, 6, seed=4)
    assert_gradients_match(lambda t: t.reshape(3, 4).mul(t.reshape(3, 4)).sum(), [x])
    assert x.grad.shape == (2, 6)


def test_expand_gradients():
    """Expand's gradient sums over the axes that were stretched."""
    x = random_tensor(1, 4, seed=5)
    assert_gradients_match(lambda t: t.expand(3, 4).mul(t.expand(3, 4)).sum(), [x])
    assert x.grad.shape == (1, 4)


def test_permute_gradients():
    """Permute's gradient applies the inverse permutation."""
    x = random_tensor(2, 3, 4, seed=6)
    assert_gradients_match(
        lambda t: t.permute(2, 0, 1).mul(t.permute(2, 0, 1)).sum(), [x]
    )
    assert x.grad.shape == (2, 3, 4)


def test_transpose_gradients():
    """transpose inherits Permute's gradient."""
    x = random_tensor(3, 5, seed=7)
    assert_gradients_match(lambda t: t.T.mul(t.T).sum(), [x])
    assert x.grad.shape == (3, 5)


# ---------------------------------------------------------------------------
# Composite operations -- no hand-written derivative exists for any of these
# ---------------------------------------------------------------------------


def test_matmul_gradients_match_the_textbook_rules():
    """The composed matmul gives back dA = dOut @ B.T and dB = A.T @ dOut.

    Neither rule is written down anywhere in this project. They come out of the
    chain rule applied to reshape, expand, mul and sum, which is the clearest
    example of composition paying off that I ran into building this.
    """
    a = random_tensor(3, 2, seed=10)
    b = random_tensor(2, 4, seed=11)
    assert_gradients_match(lambda x, y: x.matmul(y).sum(), [a, b])

    # And compare against the closed-form rules explicitly.
    upstream = np.ones((3, 4), dtype=np.float32)
    assert np.allclose(a.grad.numpy(), upstream @ b.numpy().T, atol=1e-4)
    assert np.allclose(b.grad.numpy(), a.numpy().T @ upstream, atol=1e-4)


def test_log_softmax_gradients():
    """log_softmax is differentiable end to end despite the stability shift."""
    x = random_tensor(3, 4, seed=13)
    assert_gradients_match(lambda t: t.log_softmax().sum(), [x])


def test_softmax_gradients():
    """softmax, built as exp(log_softmax), differentiates correctly too."""
    x = random_tensor(3, 4, seed=15)
    assert_gradients_match(lambda t: t.softmax().mul(t).sum(), [x])
