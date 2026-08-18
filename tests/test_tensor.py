"""Tests for tensor.py.

Values, shapes and plumbing. Gradients are in test_gradients.py.
"""

import numpy as np
import pytest

from teenygrad import LazyBuffer, Tensor


# ---------------------------------------------------------------------------
# Construction and properties
# ---------------------------------------------------------------------------


def test_tensor_accepts_scalars_lists_arrays_and_buffers():
    """The constructor is the single entry point for making a Tensor."""
    assert Tensor(3.0).shape == ()
    assert Tensor([1, 2, 3]).shape == (3,)
    assert Tensor(np.zeros((2, 2))).shape == (2, 2)
    assert Tensor(LazyBuffer(np.zeros((4,)))).shape == (4,)
    assert Tensor([1, 2, 3]).dtype == np.float32


def test_leaf_defaults():
    """A fresh tensor is a leaf: no producing op, no gradient, not tracked."""
    x = Tensor([1.0, 2.0])
    assert x._ctx is None
    assert x.grad is None
    assert x.requires_grad is False


def test_ndim_and_len():
    """Convenience properties agree with the shape."""
    x = Tensor(np.zeros((7, 2, 3)))
    assert x.ndim == 3
    assert len(x) == 7
    assert Tensor(1.0).ndim == 0


def test_item_works_for_any_single_element_tensor():
    """item() unwraps a scalar, whether its shape is () or (1,) or (1, 1)."""
    assert Tensor(42.0).item() == 42.0
    assert Tensor([42.0]).item() == 42.0
    assert Tensor([[42.0]]).item() == 42.0
    with pytest.raises(ValueError):
        Tensor([1.0, 2.0]).item()


def test_data_setter_mutates_in_place():
    """Assigning .data keeps the same Tensor object, which the optimiser needs."""
    x = Tensor([1.0, 2.0])
    original_id = id(x)
    x.data = LazyBuffer([9.0, 9.0])
    assert id(x) == original_id
    assert np.array_equal(x.numpy(), [9.0, 9.0])


# ---------------------------------------------------------------------------
# Creation helpers
# ---------------------------------------------------------------------------


def test_creation_helpers():
    """zeros/ones/full/rand/randn produce the right shapes and values."""
    assert np.array_equal(Tensor.zeros((2, 3)).numpy(), np.zeros((2, 3)))
    assert np.array_equal(Tensor.ones((2, 3)).numpy(), np.ones((2, 3)))
    assert np.array_equal(Tensor.full((2,), 7.0).numpy(), [7.0, 7.0])
    assert Tensor.rand((3, 3), seed=0).shape == (3, 3)
    assert Tensor.randn((4, 2), seed=0).shape == (4, 2)
    assert Tensor.randn(5, seed=0).shape == (5,)


def test_creation_helpers_can_request_gradients():
    """Weight tensors are made with requires_grad=True."""
    assert Tensor.randn((2, 2), seed=0, requires_grad=True).requires_grad is True
    assert Tensor.zeros((2,), requires_grad=True).requires_grad is True


def test_randn_is_reproducible():
    """Same seed, same numbers."""
    assert np.array_equal(
        Tensor.randn((3, 3), seed=7).numpy(), Tensor.randn((3, 3), seed=7).numpy()
    )


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def test_operator_overloads_match_the_named_methods():
    """+ - * / @ are exactly their spelled-out equivalents."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    assert np.allclose((a + b).numpy(), a.add(b).numpy())
    assert np.allclose((a - b).numpy(), a.sub(b).numpy())
    assert np.allclose((a * b).numpy(), a.mul(b).numpy())
    assert np.allclose((a / b).numpy(), a.div(b).numpy())
    assert np.allclose((a @ b).numpy(), a.matmul(b).numpy())
    assert np.allclose((-a).numpy(), -a.numpy())


def test_scalars_are_promoted_on_both_sides():
    """`2 * t` works as well as `t * 2`, and order is respected for - and /."""
    t = Tensor([1.0, 2.0, 4.0])
    assert np.allclose((t * 2).numpy(), [2.0, 4.0, 8.0])
    assert np.allclose((2 * t).numpy(), [2.0, 4.0, 8.0])
    assert np.allclose((t - 1).numpy(), [0.0, 1.0, 3.0])
    assert np.allclose((10 - t).numpy(), [9.0, 8.0, 6.0])
    assert np.allclose((8 / t).numpy(), [8.0, 4.0, 2.0])


def test_binary_ops_broadcast_like_numpy():
    """Values agree with numpy for the usual broadcast patterns."""
    matrix = np.arange(6, dtype=np.float32).reshape(2, 3)
    row = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    column = np.array([[100.0], [200.0]], dtype=np.float32)

    assert np.allclose(Tensor(matrix).add(Tensor(row)).numpy(), matrix + row)
    assert np.allclose(Tensor(matrix).add(Tensor(column)).numpy(), matrix + column)
    assert np.allclose(Tensor(row).mul(Tensor(matrix)).numpy(), row * matrix)


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------


def test_shape_arguments_accept_both_styles():
    """t.reshape(2, 3) and t.reshape((2, 3)) mean the same thing."""
    t = Tensor(np.arange(6, dtype=np.float32))
    assert t.reshape(2, 3).shape == (2, 3)
    assert t.reshape((2, 3)).shape == (2, 3)
    assert t.reshape([2, 3]).shape == (2, 3)


def test_expand_and_permute_values():
    """Movement ops agree with numpy."""
    row = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert np.allclose(Tensor(row).expand(2, 3).numpy(), np.broadcast_to(row, (2, 3)))

    cube = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    assert np.allclose(
        Tensor(cube).permute(2, 0, 1).numpy(), cube.transpose(2, 0, 1)
    )


def test_transpose_matches_numpy():
    """transpose() and .T agree with numpy on values."""
    values = np.random.RandomState(7).randn(3, 5).astype(np.float32)
    x = Tensor(values)
    assert np.allclose(x.transpose().numpy(), values.T)
    assert np.allclose(x.T.numpy(), values.T)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_reductions_match_numpy(axis, keepdim):
    """sum/max/mean agree with numpy for every axis and keepdim setting."""
    values = np.random.RandomState(0).randn(3, 4).astype(np.float32)
    t = Tensor(values)
    assert np.allclose(
        t.sum(axis=axis, keepdim=keepdim).numpy(),
        values.sum(axis=axis, keepdims=keepdim),
        atol=1e-4,
    )
    assert np.allclose(
        t.max(axis=axis, keepdim=keepdim).numpy(),
        values.max(axis=axis, keepdims=keepdim),
        atol=1e-4,
    )
    assert np.allclose(
        t.mean(axis=axis, keepdim=keepdim).numpy(),
        values.mean(axis=axis, keepdims=keepdim),
        atol=1e-4,
    )


def test_reduction_to_a_scalar_has_empty_shape():
    """Reducing every axis without keepdim gives a true 0-d tensor."""
    total = Tensor(np.ones((2, 3), dtype=np.float32)).sum()
    assert total.shape == ()
    assert total.item() == pytest.approx(6.0)


def test_reducing_a_scalar_is_harmless():
    """A 0-d tensor has no axes, so every reduction is the identity."""
    assert Tensor(3.0).sum().item() == pytest.approx(3.0)
    assert Tensor(3.0).mean().item() == pytest.approx(3.0)


def test_multi_axis_reduction():
    """A tuple of axes reduces all of them at once."""
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    got = Tensor(values).sum(axis=(0, 2))
    assert got.shape == (3,)
    assert np.allclose(got.numpy(), values.sum(axis=(0, 2)))


# ---------------------------------------------------------------------------
# Composite operations
# ---------------------------------------------------------------------------


def test_matmul_matches_numpy():
    """The reshape/expand/mul/sum construction really does compute a matmul."""
    a = np.random.RandomState(8).randn(4, 3).astype(np.float32)
    b = np.random.RandomState(9).randn(3, 5).astype(np.float32)
    result = Tensor(a).matmul(Tensor(b))
    assert result.shape == (4, 5)
    assert np.allclose(result.numpy(), a @ b, atol=1e-4)


def test_matmul_rejects_bad_shapes():
    """Dimension errors are caught with a useful message, not a numpy traceback."""
    with pytest.raises(ValueError, match="inner dimensions disagree"):
        Tensor(np.zeros((2, 3))).matmul(Tensor(np.zeros((4, 5))))
    with pytest.raises(ValueError, match="two 2-D tensors"):
        Tensor(np.zeros((2, 3, 4))).matmul(Tensor(np.zeros((4, 5))))


def test_softmax_is_a_probability_distribution():
    """Softmax rows are nonnegative and sum to 1."""
    x = Tensor(np.random.RandomState(12).randn(5, 4).astype(np.float32) * 3)
    probs = x.softmax().numpy()
    assert (probs >= 0).all()
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)


def test_softmax_survives_huge_logits():
    """The max-subtraction keeps exp() from overflowing to inf/nan."""
    x = Tensor([[1000.0, 1001.0, 999.0]])
    probs = x.softmax().numpy()
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(), 1.0, atol=1e-5)


def test_log_softmax_matches_softmax_then_log():
    """The stable path agrees with the naive definition on well-scaled inputs."""
    values = np.random.RandomState(14).randn(3, 4).astype(np.float32)
    naive = np.log(np.exp(values) / np.exp(values).sum(axis=-1, keepdims=True))
    assert np.allclose(Tensor(values).log_softmax().numpy(), naive, atol=1e-5)


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------


def test_detach_cuts_the_graph():
    """A detached tensor keeps its values but stops gradient flow."""
    x = Tensor([1.0, 2.0], requires_grad=True)
    doubled = x.mul(2.0)
    cut = doubled.detach()
    assert np.allclose(cut.numpy(), doubled.numpy())
    assert cut._ctx is None
    assert cut.requires_grad is False


def test_topological_order_places_parents_before_children():
    """The ordering property backprop depends on, checked directly."""
    x = Tensor([1.0], requires_grad=True)
    y = x.mul(2.0)
    z = y.add(3.0)

    order = z._topological_order()
    assert order[-1] is z
    assert order.index(x) < order.index(y) < order.index(z)


def test_topological_order_visits_a_shared_node_once():
    """A diamond in the graph does not produce duplicate entries."""
    x = Tensor([1.0], requires_grad=True)
    shared = x.mul(2.0)
    combined = shared.add(shared)

    order = combined._topological_order()
    assert len(order) == len({id(node) for node in order})


def test_numpy_interop():
    """np.asarray and float() work on a Tensor."""
    values = np.array([[1.0, 2.0]], dtype=np.float32)
    assert np.array_equal(np.asarray(Tensor(values)), values)
    assert float(Tensor(5.0)) == 5.0


def test_repr_mentions_shape_and_grad_flag():
    """repr is informative enough to debug with."""
    assert "shape=(2,)" in repr(Tensor([1.0, 2.0]))
    assert "requires_grad=True" in repr(Tensor([1.0], requires_grad=True))
