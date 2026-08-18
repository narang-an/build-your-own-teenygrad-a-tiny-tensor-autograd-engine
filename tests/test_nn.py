"""Tests for nn/layers.py and nn/losses.py.

Layers and losses are just compositions of Tensor ops, so each test either
checks one against the arithmetic it claims to do, or gradient-checks it to
confirm the composition differentiates properly with no layer-specific
derivative code involved.
"""

import numpy as np
import pytest

from teenygrad import MLP, Linear, Tensor, one_hot, sparse_categorical_cross_entropy
from tests.support.gradcheck import assert_gradients_match


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------


def test_linear_shapes_and_parameters():
    """A Linear layer has a (in, out) weight, an (out,) bias, and both train."""
    layer = Linear(3, 5, seed=0)
    assert layer.weight.shape == (3, 5)
    assert layer.bias.shape == (5,)
    assert layer.parameters() == [layer.weight, layer.bias]
    assert all(p.requires_grad for p in layer.parameters())


def test_linear_forward_matches_the_arithmetic():
    """The layer really does compute x @ W + b."""
    layer = Linear(3, 4, seed=1)
    x = np.random.RandomState(0).randn(6, 3).astype(np.float32)
    expected = x @ layer.weight.numpy() + layer.bias.numpy()
    assert np.allclose(layer(Tensor(x)).numpy(), expected, atol=1e-4)


def test_linear_is_reproducible_from_its_seed():
    """Same seed, same layer."""
    first = Linear(3, 4, seed=5)
    again = Linear(3, 4, seed=5)
    assert np.array_equal(first.weight.numpy(), again.weight.numpy())
    assert np.array_equal(first.bias.numpy(), again.bias.numpy())


def test_linear_bias_gradient_sums_over_the_batch():
    """The bias broadcasts across the batch, so its gradient sums back down.

    With 6 rows and a loss that sums everything, each bias element picks up
    gradient 1 per row, so it should come out at 6.
    """
    layer = Linear(3, 4, seed=2)
    x = Tensor(np.random.RandomState(0).randn(6, 3).astype(np.float32))
    layer(x).sum().backward()

    assert layer.bias.grad.shape == (4,)
    assert np.allclose(layer.bias.grad.numpy(), np.full(4, 6.0), atol=1e-4)
    assert layer.weight.grad.shape == (3, 4)


def test_linear_gradients():
    """Both parameters gradient-check against finite differences."""
    layer = Linear(3, 2, seed=3)
    x = Tensor(np.random.RandomState(1).randn(4, 3).astype(np.float32))
    assert_gradients_match(
        lambda w, b: x.matmul(w).add(b).mul(x.matmul(w).add(b)).sum(),
        [layer.weight, layer.bias],
    )


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


def test_mlp_forward_matches_a_manual_relu_network():
    """The MLP is exactly relu(x @ W1 + b1) @ W2 + b2.

    This is the check that turned up the missing Tensor.relu in the first place.
    """
    np.random.seed(0)
    x = np.random.randn(4, 3).astype(np.float32)

    model = MLP(3, 5, 2, seed=42)
    w1, b1, w2, b2 = (p.numpy() for p in model.parameters())
    expected = np.maximum(x @ w1 + b1, 0.0) @ w2 + b2

    assert np.allclose(model(Tensor(x)).numpy(), expected, atol=1e-3)


def test_mlp_parameter_order_and_shapes():
    """parameters() returns [W1, b1, W2, b2], which train_mlp relies on."""
    model = MLP(9, 16, 3, seed=0)
    shapes = [p.shape for p in model.parameters()]
    assert shapes == [(9, 16), (16,), (16, 3), (3,)]


def test_mlp_accepts_raw_arrays():
    """Calling the model with a numpy array promotes it to a Tensor."""
    model = MLP(3, 4, 2, seed=0)
    out = model(np.zeros((5, 3), dtype=np.float32))
    assert isinstance(out, Tensor)
    assert out.shape == (5, 2)


def test_mlp_layers_get_different_seeds():
    """The two layers must not be initialised identically."""
    model = MLP(4, 4, 4, seed=0)
    assert not np.array_equal(model.l1.weight.numpy(), model.l2.weight.numpy())


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def test_one_hot():
    """Integer ids become float rows with a single 1.0."""
    encoded = one_hot([2, 0, 1], num_classes=3).numpy()
    assert np.array_equal(encoded, np.eye(3, dtype=np.float32)[[2, 0, 1]])
    assert encoded.dtype == np.float32


def test_cross_entropy_returns_a_scalar():
    """The loss is 0-d, which is what backward() expects to be seeded on."""
    logits = Tensor(np.random.RandomState(0).randn(5, 3).astype(np.float32))
    loss = sparse_categorical_cross_entropy(logits, [0, 1, 2, 1, 0])
    assert loss.shape == ()


def test_cross_entropy_matches_the_definition():
    """Agrees with an independent numpy computation of -mean(log p_correct)."""
    values = np.random.RandomState(1).randn(6, 4).astype(np.float32)
    labels = np.array([0, 3, 1, 2, 2, 0])

    shifted = values - values.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    expected = -log_probs[np.arange(6), labels].mean()

    got = sparse_categorical_cross_entropy(Tensor(values), labels).item()
    assert got == pytest.approx(expected, rel=1e-4)


def test_cross_entropy_is_near_zero_for_confident_correct_predictions():
    """A confident, correct model should have almost no loss."""
    logits = Tensor([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    assert sparse_categorical_cross_entropy(logits, [0, 1]).item() < 1e-5


def test_cross_entropy_is_large_for_confident_wrong_predictions():
    """And a confident, wrong one should be heavily penalised."""
    logits = Tensor([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    assert sparse_categorical_cross_entropy(logits, [1, 0]).item() > 10.0


def test_cross_entropy_gradients():
    """The loss differentiates correctly with respect to the logits."""
    logits = Tensor(np.random.RandomState(2).randn(4, 3).astype(np.float32))
    labels = np.array([0, 2, 1, 1])
    assert_gradients_match(
        lambda t: sparse_categorical_cross_entropy(t, labels), [logits]
    )


def test_cross_entropy_gradient_matches_the_closed_form():
    """Should come out to (softmax(logits) - onehot) / batch.

    Most frameworks hard-code this simplification as a single fused op. Here it
    just falls out of composing log_softmax, mul, sum, mean and neg, so it seemed
    worth checking directly.
    """
    values = np.random.RandomState(3).randn(5, 4).astype(np.float32)
    labels = np.array([1, 0, 3, 2, 1])

    logits = Tensor(values, requires_grad=True)
    sparse_categorical_cross_entropy(logits, labels).backward()

    shifted = values - values.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    expected = (probs - np.eye(4)[labels]) / 5

    assert np.allclose(logits.grad.numpy(), expected, atol=1e-5)


def test_cross_entropy_validates_its_inputs():
    """Shape mistakes get a clear message."""
    with pytest.raises(ValueError, match="batch, num_classes"):
        sparse_categorical_cross_entropy(Tensor([1.0, 2.0]), [0])
    with pytest.raises(ValueError, match="labels"):
        sparse_categorical_cross_entropy(Tensor(np.zeros((3, 2))), [0, 1])
