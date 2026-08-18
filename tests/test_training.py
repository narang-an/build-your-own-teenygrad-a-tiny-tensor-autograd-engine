"""End-to-end tests for data/ and training.py.

Every other test file checks one piece on its own. These check that the pieces
add up to something that actually learns.
"""

import numpy as np
import pytest

from teenygrad import (
    DIGIT_PROTOTYPES,
    MLP,
    accuracy,
    evaluate_mlp,
    make_toy_digit_dataset,
    train_mlp,
)


# ---------------------------------------------------------------------------
# Data and metrics
# ---------------------------------------------------------------------------


def test_dataset_shapes_and_label_range():
    """The generator returns the shapes and dtypes the loop expects."""
    X, y = make_toy_digit_dataset(num_samples=50, seed=0)
    assert X.shape == (50, 9)
    assert y.shape == (50,)
    assert X.dtype == np.float32
    assert set(np.unique(y)).issubset({0, 1, 2})


def test_dataset_is_reproducible_and_seed_dependent():
    """Same seed reproduces; different seeds give a genuinely different draw."""
    first, _ = make_toy_digit_dataset(20, seed=0)
    again, _ = make_toy_digit_dataset(20, seed=0)
    other, _ = make_toy_digit_dataset(20, seed=1)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_samples_stay_close_to_their_prototype():
    """Noise is small enough that the classes remain separable."""
    X, y = make_toy_digit_dataset(100, seed=0)
    assert np.abs(X - DIGIT_PROTOTYPES[y]).max() < 0.6


@pytest.mark.parametrize(
    "logits, labels, expected",
    [
        ([[0.0, 1.0], [1.0, 0.0]], [1, 0], 1.0),
        ([[0.0, 1.0], [1.0, 0.0]], [0, 1], 0.0),
        ([[0.0, 1.0], [1.0, 0.0]], [1, 1], 0.5),
    ],
)
def test_accuracy(logits, labels, expected):
    """Accuracy is the fraction of argmax-correct rows."""
    assert accuracy(np.array(logits), labels) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained():
    """Train once and share the result across the tests below."""
    X, y = make_toy_digit_dataset(200, seed=0)
    model, losses = train_mlp(X, y, epochs=30, learning_rate=0.1, hidden=16, seed=0)
    return model, losses


def test_training_returns_one_loss_per_epoch(trained):
    """The loss curve has the length it claims to."""
    _, losses = trained
    assert len(losses) == 30
    assert all(np.isfinite(losses))


def test_loss_decreases(trained):
    """The main one: gradient descent through the engine actually works."""
    _, losses = trained
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.1


def test_loss_decreases_roughly_monotonically(trained):
    """No blowing up or bouncing around at this learning rate.

    A few upticks are normal. A flipped gradient sign makes the loss rise almost
    every epoch, which this would catch.
    """
    _, losses = trained
    increases = sum(1 for a, b in zip(losses, losses[1:]) if b > a)
    assert increases < len(losses) // 3


def test_model_generalises_to_held_out_data(trained):
    """Accuracy on data the model never saw."""
    model, _ = trained
    X_test, y_test = make_toy_digit_dataset(60, seed=1)
    assert evaluate_mlp(model, X_test, y_test) > 0.9


def test_training_is_reproducible():
    """Same seed, same loss curve. Makes training bugs much easier to chase."""
    X, y = make_toy_digit_dataset(50, seed=0)
    _, first = train_mlp(X, y, epochs=5, seed=3)
    _, again = train_mlp(X, y, epochs=5, seed=3)
    assert first == again


def test_parameters_actually_change(trained):
    """Guards against a loop that runs fine but trains nothing.

    An earlier version of this computed gradients into detached tensors, so the
    loop looked perfectly healthy while the model's actual weights never moved.
    """
    model, _ = trained
    # An identically-seeded, untrained model is the baseline to compare against.
    untrained = MLP(in_features=9, hidden=16, out_features=3, seed=0)
    assert not np.allclose(
        model.l1.weight.numpy(), untrained.l1.weight.numpy(), atol=1e-6
    )
    assert not np.allclose(
        model.l2.bias.numpy(), untrained.l2.bias.numpy(), atol=1e-6
    )


def test_zero_epochs_leaves_the_model_untrained():
    """A degenerate but legal configuration."""
    X, y = make_toy_digit_dataset(20, seed=0)
    model, losses = train_mlp(X, y, epochs=0, seed=0)
    assert losses == []
    assert all(p.grad is None for p in model.parameters())
