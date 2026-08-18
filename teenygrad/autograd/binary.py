"""Binary elementwise ops: two inputs of the same shape.

Two differences from the unary ops. These are the first with more than one
input, so they're the first to check needs_input_grad and hand back a tuple of
gradients instead of a single one.

They also assume both operands already match in shape. Tensor._broadcast puts
real Reshape/Expand nodes in front when they don't, so broadcasting gets
differentiated as its own separate step. That's why there's no "sum the gradient
back over the broadcast axes" logic in this file. By the time we get here
there's nothing left to undo.
"""

from __future__ import annotations

from teenygrad.autograd.function import Function
from teenygrad.core.lazybuffer import LazyBuffer
from teenygrad.core.ops import BinaryOps, UnaryOps


class Add(Function):
    """z = x + y. Both partials are 1, so the gradient just splits in two."""

    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        return x.e(BinaryOps.ADD, y)

    def backward(
        self, grad_output: LazyBuffer
    ) -> tuple[LazyBuffer | None, LazyBuffer | None]:
        return (
            grad_output if self.needs_input_grad[0] else None,
            grad_output if self.needs_input_grad[1] else None,
        )


class Sub(Function):
    """z = x - y. dz/dx is 1, dz/dy is -1."""

    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        return x.e(BinaryOps.SUB, y)

    def backward(
        self, grad_output: LazyBuffer
    ) -> tuple[LazyBuffer | None, LazyBuffer | None]:
        return (
            grad_output if self.needs_input_grad[0] else None,
            grad_output.e(UnaryOps.NEG) if self.needs_input_grad[1] else None,
        )


class Mul(Function):
    """z = x * y. Product rule: dz/dx is y and dz/dy is x."""

    x: LazyBuffer
    y: LazyBuffer

    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        # Each input is needed for the *other's* gradient, so both get cached.
        # This is why elementwise multiply costs memory during training and
        # plain addition doesn't.
        self.x, self.y = x, y
        return x.e(BinaryOps.MUL, y)

    def backward(
        self, grad_output: LazyBuffer
    ) -> tuple[LazyBuffer | None, LazyBuffer | None]:
        return (
            grad_output.e(BinaryOps.MUL, self.y) if self.needs_input_grad[0] else None,
            grad_output.e(BinaryOps.MUL, self.x) if self.needs_input_grad[1] else None,
        )


class Div(Function):
    """z = x / y. Quotient rule: dz/dx is 1/y, dz/dy is -x/y^2."""

    x: LazyBuffer
    y: LazyBuffer

    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        self.x, self.y = x, y
        return x.e(BinaryOps.DIV, y)

    def backward(
        self, grad_output: LazyBuffer
    ) -> tuple[LazyBuffer | None, LazyBuffer | None]:
        grad_x: LazyBuffer | None = None
        grad_y: LazyBuffer | None = None
        if self.needs_input_grad[0]:
            grad_x = grad_output.e(BinaryOps.DIV, self.y)
        if self.needs_input_grad[1]:
            numerator = grad_output.e(BinaryOps.MUL, self.x)
            y_squared = self.y.e(BinaryOps.MUL, self.y)
            grad_y = numerator.e(BinaryOps.DIV, y_squared).e(UnaryOps.NEG)
        return grad_x, grad_y


__all__ = ["Add", "Sub", "Mul", "Div"]
