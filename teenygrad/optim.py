"""Updating parameters, the other half of a training step.

Autograd works out which direction each parameter should move. The optimiser
decides how far. This is plain SGD, plus the bookkeeping call that makes
repeated steps come out right.

It's at the top level rather than inside nn/ for the same reason torch.optim is
separate from torch.nn: an optimiser works on any list of Tensors and doesn't
care whether a network produced them.
"""

from __future__ import annotations

from collections.abc import Iterable

from teenygrad.core.lazybuffer import LazyBuffer
from teenygrad.core.ops import BinaryOps
from teenygrad.tensor import Tensor


def sgd_step(parameters: Iterable[Tensor], learning_rate: float) -> None:
    """One gradient descent update: p <- p - learning_rate * p.grad.

    The gradient points uphill, so we move the other way. learning_rate scales
    the step. Too small and it crawls, too large and it overshoots and blows up.

    Parameters whose .grad is None (they weren't involved in the loss) get
    skipped.

    The update goes straight into p.data, mutating the existing Tensor rather
    than making a new one. model.parameters() handed out references to these
    exact objects and the layers hold the same ones, so rebinding would leave
    the model pointing at stale weights.
    """
    for param in parameters:
        if param.grad is None:
            continue
        if param.grad.shape != param.shape:
            raise ValueError(
                f"gradient shape {param.grad.shape} does not match parameter "
                f"shape {param.shape}"
            )
        # Buffer-level arithmetic, not Tensor arithmetic. An optimiser update
        # isn't part of the model's computation and shouldn't show up in the
        # next forward pass's graph.
        scale = LazyBuffer.const(learning_rate, param.shape)
        step = param.grad.lazydata.e(BinaryOps.MUL, scale)
        param.data = param.lazydata.e(BinaryOps.SUB, step)


def zero_grad(parameters: Iterable[Tensor]) -> None:
    """Clear out gradients before the next backward pass.

    backward() adds into .grad rather than replacing it, which is what lets a
    tensor used in several places collect gradient from all of them. The price
    of that design is this function. Forget to call it and every step descends
    on the sum of every gradient so far, so the effective learning rate keeps
    growing and training falls apart.
    """
    for param in parameters:
        param.grad = None


__all__ = ["sgd_step", "zero_grad"]
