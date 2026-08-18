"""Loss functions.

The loss sits at the root of the graph. Everything the model did flows into it,
and calling .backward() on it is what fills in every parameter's gradient. So
there's one hard rule: it has to be built entirely out of differentiable Tensor
ops, with no numpy shortcut anywhere between the logits and the returned scalar.
"""

from __future__ import annotations

import numpy as np

from teenygrad.core.types import ArrayLike, TensorLike
from teenygrad.tensor import Tensor


def one_hot(labels: ArrayLike, num_classes: int) -> Tensor:
    """Turn integer labels into a float one-hot matrix, shape (batch, num_classes).

    Class ids never go into a Tensor, since every buffer is float32. Converting
    to one-hot here is what lets "pick out the correct class's log-probability"
    be a differentiable multiply-and-sum, rather than fancy indexing, which
    wouldn't have a gradient.

    The result isn't tracked for gradients, since it's a constant of the problem.
    """
    ids = np.asarray(labels).astype(int).reshape(-1)
    encoded = np.zeros((ids.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(ids.shape[0]), ids] = 1.0
    return Tensor(encoded)


def sparse_categorical_cross_entropy(logits: TensorLike, labels: ArrayLike) -> Tensor:
    """Mean negative log-probability of the correct class.

    "Sparse" means the targets come in as integer class ids ([2, 0, 1]) rather
    than one-hot rows, same convention Keras uses.

    Steps, all in differentiable ops:
      1. log_softmax turns the logits into log-probabilities per row.
      2. Multiply by the one-hot targets and sum along the class axis, which
         picks out the right entry and zeroes the rest.
      3. Average over the batch and negate.

    Using log_softmax rather than log(softmax(x)) matters: the latter takes an
    exponential and immediately logs it again, losing precision and overflowing
    on large logits.

    logits is (batch, num_classes), labels is (batch,). Returns a scalar
    connected to the logits, so you can call .backward() on it.
    """
    if not isinstance(logits, Tensor):
        logits = Tensor(logits)
    if logits.ndim != 2:
        raise ValueError(f"expected (batch, num_classes) logits, got shape {logits.shape}")

    ids = np.asarray(labels).astype(int).reshape(-1)
    batch, num_classes = logits.shape
    if ids.shape[0] != batch:
        raise ValueError(f"got {batch} rows of logits but {ids.shape[0]} labels")

    log_probs = logits.log_softmax(axis=-1)
    correct = log_probs.mul(one_hot(ids, num_classes)).sum(axis=-1)
    return correct.mean().neg()


__all__ = ["one_hot", "sparse_categorical_cross_entropy"]
