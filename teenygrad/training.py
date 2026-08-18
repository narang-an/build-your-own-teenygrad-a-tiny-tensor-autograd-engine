"""The training and evaluation loop.

Where it all comes together. Notice what isn't in here: no derivatives, no numpy
gradient maths, no mention of Function or LazyBuffer. The loop just says the
four steps of supervised learning and the rest of the package does the work.

    forward -> loss -> backward -> update

If you only read one file to see what the other nine are for, read this one.
"""

from __future__ import annotations

import numpy as np

from teenygrad.core.types import ArrayLike
from teenygrad.data.metrics import accuracy
from teenygrad.nn.layers import MLP
from teenygrad.nn.losses import sparse_categorical_cross_entropy
from teenygrad.optim import sgd_step, zero_grad
from teenygrad.tensor import Tensor


def train_mlp(
    X: ArrayLike,
    y: ArrayLike,
    epochs: int = 30,
    learning_rate: float = 0.1,
    hidden: int = 16,
    seed: int | None = 0,
) -> tuple[MLP, list[float]]:
    """Train a two-layer MLP with full-batch gradient descent.

    Full-batch means each epoch runs the whole dataset through in one forward
    pass instead of splitting it into minibatches. Simplest thing that works and
    it's fine at this size; minibatching would be the first thing to add for a
    real dataset.

    Returns (model, losses). Each loss is measured before that epoch's update,
    so losses[0] is the untrained model and a decreasing list means it's
    learning.
    """
    features = np.asarray(X, dtype=np.float32)
    labels = np.asarray(y).astype(int).reshape(-1)
    num_features = features.shape[1]
    num_classes = int(labels.max()) + 1

    model = MLP(
        in_features=num_features,
        hidden=hidden,
        out_features=num_classes,
        seed=seed,
    )
    params = model.parameters()

    # Built once and reused. The inputs never change and they're a leaf with
    # requires_grad=False, so no gradient ever gets computed for them.
    inputs = Tensor(features)

    losses: list[float] = []
    for _ in range(epochs):
        # 1. Forward. Running the model builds a fresh graph, since every op
        #    records its inputs in the output's _ctx as it goes.
        logits = model(inputs)
        loss = sparse_categorical_cross_entropy(logits, labels)
        losses.append(loss.item())

        # 2. Clear last epoch's gradients. Has to happen before backward(),
        #    because backward() accumulates.
        zero_grad(params)

        # 3. Backward. Walks the graph in reverse and fills in every parameter's
        #    .grad. This is the whole point of the autograd package.
        loss.backward()

        # 4. Nudge each parameter downhill.
        sgd_step(params, learning_rate)

        # This epoch's graph becomes garbage right here. Nothing references
        # loss or logits any more, so Python reclaims the whole chain of
        # intermediates. Only the parameters and their grads stick around.

    return model, losses


def evaluate_mlp(model: MLP, X_test: ArrayLike, y_test: ArrayLike) -> float:
    """Run a trained model on held-out data and return its accuracy."""
    logits = model(Tensor(np.asarray(X_test, dtype=np.float32)))
    return accuracy(logits.numpy(), y_test)


__all__ = ["train_mlp", "evaluate_mlp"]
