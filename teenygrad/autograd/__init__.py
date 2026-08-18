"""Reverse-mode autodiff: the Function base class and all 15 ops.

This is the only place in the project where a derivative is written down.
Everything above it (matmul, softmax, cross-entropy, whole networks) is built
out of these and gets its gradients worked out automatically.

The op modules are split the same way core/ops.py is, because how an op treats
shapes is what decides what its gradient looks like:

    unary     same shape in and out, gradient is just a scale
    binary    two inputs, so the first ops to return a tuple
    reduce    shrink axes, so their gradients expand them back
    movement  pure shuffling, each gradient is the move that undoes it
"""

from teenygrad.autograd.binary import Add, Div, Mul, Sub
from teenygrad.autograd.function import Function
from teenygrad.autograd.movement import Expand, Permute, Reshape
from teenygrad.autograd.reduce import Max, Sum
from teenygrad.autograd.unary import Exp, Log, Neg, Relu, Sigmoid, Sqrt

__all__ = [
    "Function",
    "Neg",
    "Relu",
    "Log",
    "Exp",
    "Sqrt",
    "Sigmoid",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Sum",
    "Max",
    "Reshape",
    "Expand",
    "Permute",
]
