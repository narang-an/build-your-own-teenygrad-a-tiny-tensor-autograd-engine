"""Reduce ops: collapse axes down to size 1.

First ops whose output shape differs from their input shape, so their gradients
have to grow a buffer back rather than just rescale it. Both keep the reduced
axes (keepdims), which is what lets the backward pass be a single expand: the
output is already lined up against the input.

Reduce and expand are mirror images of each other:

    summing N values into 1  ->  the gradient copies 1 value out to N
    copying 1 value into N   ->  the gradient sums those N back into 1

The other half of that is over in movement.py.
"""

from __future__ import annotations

from teenygrad.autograd.function import Function
from teenygrad.core.lazybuffer import LazyBuffer
from teenygrad.core.ops import BinaryOps, ReduceOps
from teenygrad.core.types import Shape


class Sum(Function):
    """y = x.sum(axis). Everything contributed equally, so everything gets the
    same gradient back.

    If y = a + b + c then dy/da = dy/db = dy/dc = 1, so each input element gets
    the full gradient of whichever output cell it fed into. Copying one value
    out to many positions is exactly what expand does.
    """

    input_shape: Shape

    def forward(self, x: LazyBuffer, axis: tuple[int, ...]) -> LazyBuffer:
        self.input_shape = x.shape
        return x.r(ReduceOps.SUM, axis)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.expand(self.input_shape)


class Max(Function):
    """y = x.max(axis). Only the winners get any gradient.

    Max is locally equal to whichever input happened to be biggest, so the
    derivative is 1 for that element and 0 for the rest. When several tie, the
    gradient gets split evenly between them. That's the subgradient convention,
    and it keeps things symmetric between the tied inputs.
    """

    x: LazyBuffer
    ret: LazyBuffer
    axis: tuple[int, ...]

    def forward(self, x: LazyBuffer, axis: tuple[int, ...]) -> LazyBuffer:
        self.x, self.axis = x, axis
        self.ret = x.r(ReduceOps.MAX, axis)
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        shape = self.x.shape
        maxima = self.ret.expand(shape)

        # 1.0 where x equals the max, 0.0 everywhere else. Written as
        # 1 - (x < max) because CMPLT is the only comparison we have, and since
        # nothing can be bigger than the max, "not less than" means "equal to".
        ones = LazyBuffer.const(1, shape)
        is_max = ones.e(BinaryOps.SUB, self.x.e(BinaryOps.CMPLT, maxima))

        # Divide by how many tied, so their gradients still add up to 1.
        tie_count = is_max.r(ReduceOps.SUM, self.axis).expand(shape)
        share = is_max.e(BinaryOps.DIV, tie_count)

        return share.e(BinaryOps.MUL, grad_output.expand(shape))


__all__ = ["Sum", "Max"]
