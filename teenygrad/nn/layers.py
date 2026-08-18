"""Network layers.

All of this is written with the plain Tensor API, so there's no calculus in the
file. A Linear layer is "matmul then add" and the gradients come from the engine
underneath. Adding a new layer type here doesn't require touching autograd at
all, as long as it's built out of existing Tensor operations.

Layers follow a small convention instead of inheriting from a base class, just
to keep the amount of machinery down: __init__ makes the parameters, __call__ is
the forward pass, parameters() lists what the optimiser should update.
"""

from __future__ import annotations

import numpy as np

from teenygrad.core.types import TensorLike
from teenygrad.tensor import Tensor


class Linear:
    """A fully connected layer: x @ W + b."""

    def __init__(
        self, in_features: int, out_features: int, seed: int | None = None
    ) -> None:
        """Set up the weight (in, out) and bias (out,), both trainable.

        Both come from a standard normal. Real frameworks scale the initial
        weights by something like 1/sqrt(in_features), Xavier/He init, to stop
        activations blowing up as layers stack. I skipped it because the nets
        here are two layers deep and it doesn't bite, but it's the first thing
        to add before going deeper.

        Both parameters come off one seeded generator, so the same seed gives
        the same layer.
        """
        rng = np.random.RandomState(seed)
        self.weight = Tensor(rng.randn(in_features, out_features), requires_grad=True)
        self.bias = Tensor(rng.randn(out_features), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass. (batch, in_features) -> (batch, out_features).

        The bias is (out_features,) while the matmul result is (batch,
        out_features), so add() broadcasts it through real expand nodes. That
        means the bias gradient gets summed back down over the batch axis during
        backprop without any reshaping by hand.
        """
        return x.matmul(self.weight).add(self.bias)

    def parameters(self) -> list[Tensor]:
        """The trainable tensors, in a fixed order."""
        return [self.weight, self.bias]


class MLP:
    """Two-layer network: Linear -> relu -> Linear.

    The output is raw scores (logits), not probabilities. The loss does its own
    log-softmax, which is both more stable and avoids normalising twice.
    """

    def __init__(
        self,
        in_features: int,
        hidden: int,
        out_features: int,
        seed: int | None = None,
    ) -> None:
        """Build both layers. out_features is the number of classes.

        The second layer uses seed + 1 so the two don't get correlated weights.
        """
        second_seed = seed + 1 if seed is not None else None
        self.l1 = Linear(in_features, hidden, seed=seed)
        self.l2 = Linear(hidden, out_features, seed=second_seed)

    def __call__(self, x: TensorLike) -> Tensor:
        """Forward pass, returning (batch, out_features) logits."""
        if not isinstance(x, Tensor):
            x = Tensor(x)
        hidden = self.l1(x).relu()
        return self.l2(hidden)

    def parameters(self) -> list[Tensor]:
        """Everything trainable, ordered [W1, b1, W2, b2]."""
        return self.l1.parameters() + self.l2.parameters()


__all__ = ["Linear", "MLP"]
