"""Movement ops: shuffle or duplicate elements, no arithmetic.

Not one of these does a floating point operation. They only decide where values
end up. They still carry gradients though, because moving a value changes which
output it affects, and duplicating one means it affects several.

The gradient of a movement op is whatever undoes it:

    reshape  ->  the reverse reshape
    expand   ->  sum over the axes that got stretched
    permute  ->  permute by the inverse permutation
"""

from __future__ import annotations

from collections.abc import Sequence

from teenygrad.autograd.function import Function
from teenygrad.core.helpers import argsort
from teenygrad.core.lazybuffer import LazyBuffer
from teenygrad.core.ops import ReduceOps
from teenygrad.core.types import Shape


class Reshape(Function):
    """Same elements under a new shape. Undone by reshaping back."""

    input_shape: Shape

    def forward(self, x: LazyBuffer, shape: Shape) -> LazyBuffer:
        self.input_shape = x.shape
        return x.reshape(shape)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.reshape(self.input_shape)


class Expand(Function):
    """Stretch size-1 axes. The opposite of duplicating a value is summing.

    If one input element got copied into N output positions, it affected the
    loss through all N of them, so by the chain rule its gradient is the sum of
    those N. Same fact as Sum.backward being an expand, just read the other way.

    This op is what makes broadcasting differentiable, which in turn is what
    makes layer_output + bias produce a correctly-shaped bias gradient without
    any special-casing.
    """

    input_shape: Shape

    def forward(self, x: LazyBuffer, shape: Shape) -> LazyBuffer:
        self.input_shape = x.shape
        return x.expand(shape)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        # Sum over exactly the axes that got stretched (input was 1, output
        # isn't). keepdims leaves those at length 1, which rebuilds input_shape.
        # If nothing was stretched this is an empty tuple and the sum is a no-op.
        stretched_axes = tuple(
            i
            for i, dim in enumerate(self.input_shape)
            if dim == 1 and grad_output.shape[i] != 1
        )
        return grad_output.r(ReduceOps.SUM, stretched_axes)


class Permute(Function):
    """Reorder axes. Undone with the inverse permutation."""

    order: Sequence[int]

    def forward(self, x: LazyBuffer, order: Sequence[int]) -> LazyBuffer:
        self.order = order
        return x.permute(order)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        # argsort of a permutation is its inverse. It answers "which slot did
        # each axis come from", which is how you put them all back.
        return grad_output.permute(argsort(self.order))


__all__ = ["Reshape", "Expand", "Permute"]
