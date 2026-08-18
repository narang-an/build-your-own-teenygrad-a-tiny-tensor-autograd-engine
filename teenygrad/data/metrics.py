"""Evaluation metrics.

Metrics are for reading, not for gradient descent. They get computed once an
epoch, they're often not differentiable anyway (accuracy is a step function of
the logits), and nothing backpropagates through them, so like the datasets next
door they're plain numpy.

Worth keeping the distinction straight: the loss is what gets optimised and has
to be smooth, the metric is what you actually care about and can be as jagged as
it likes.
"""

from __future__ import annotations

import numpy as np

from teenygrad.core.types import ArrayLike


def accuracy(logits: ArrayLike, labels: ArrayLike) -> float:
    """Fraction of rows where the highest-scoring class is the right one.

    No need to softmax first, since softmax is monotonic, so it can't change which
    entry is largest.
    """
    scores = np.asarray(logits)
    predictions = scores.argmax(axis=-1)
    return float((predictions == np.asarray(labels).reshape(-1)).mean())


__all__ = ["accuracy"]
