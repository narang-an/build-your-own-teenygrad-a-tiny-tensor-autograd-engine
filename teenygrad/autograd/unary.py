"""Unary elementwise ops: one input, output the same shape.

The simplest Functions in the library. Since shapes match, every gradient is
just the incoming gradient scaled by the local derivative, no shape juggling.

A few of these cache the *output* rather than the input. For exp, sigmoid and
sqrt the derivative can be written in terms of the result, so keeping ret saves
holding both.
"""

from __future__ import annotations

from teenygrad.autograd.function import Function
from teenygrad.core.lazybuffer import LazyBuffer
from teenygrad.core.ops import BinaryOps, UnaryOps


class Neg(Function):
    """y = -x, so the derivative is just -1."""

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        return x.e(UnaryOps.NEG)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.e(UnaryOps.NEG)


class Relu(Function):
    """y = max(x, 0). Gradient only gets through where x was positive."""

    ret: LazyBuffer

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        # Caching the output works as well as the input here, since ret > 0
        # exactly where x > 0.
        self.ret = x.e(UnaryOps.RELU)
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        # (0 < ret) gives 1 where relu passed the value through and 0 where it
        # clamped. x == 0 ends up with gradient 0, which is the usual convention
        # for relu's kink at the origin.
        zero = LazyBuffer.const(0, self.ret.shape)
        mask = zero.e(BinaryOps.CMPLT, self.ret)
        return mask.e(BinaryOps.MUL, grad_output)


class Log(Function):
    """y = ln(x), derivative 1/x, so the gradient is grad / x."""

    x: LazyBuffer

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.x = x
        return x.e(UnaryOps.LOG)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return grad_output.e(BinaryOps.DIV, self.x)


class Exp(Function):
    """y = e^x, whose derivative is itself, so backward reuses the cached output."""

    ret: LazyBuffer

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.ret = x.e(UnaryOps.EXP)
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        return self.ret.e(BinaryOps.MUL, grad_output)


class Sqrt(Function):
    """y = sqrt(x), derivative 1/(2*sqrt(x)), which is 1/(2y)."""

    ret: LazyBuffer

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.ret = x.e(UnaryOps.SQRT)
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        two = LazyBuffer.const(2, self.ret.shape)
        return grad_output.e(BinaryOps.DIV, two.e(BinaryOps.MUL, self.ret))


class Sigmoid(Function):
    """y = 1/(1+e^-x). The derivative comes out to the neat y * (1 - y)."""

    ret: LazyBuffer

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        self.ret = x.e(UnaryOps.SIGMOID)
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        one = LazyBuffer.const(1, self.ret.shape)
        one_minus = one.e(BinaryOps.SUB, self.ret)
        return grad_output.e(BinaryOps.MUL, self.ret.e(BinaryOps.MUL, one_minus))


__all__ = ["Neg", "Relu", "Log", "Exp", "Sqrt", "Sigmoid"]
