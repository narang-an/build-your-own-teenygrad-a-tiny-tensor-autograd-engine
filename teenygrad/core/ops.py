"""Names for every operation the backend can actually run.

Everything in this library eventually breaks down into some combination of the
17 ops listed here. Matmul, softmax, a whole training step, all of it. Keeping
this list short is the main trick behind tinygrad's design, and it's the reason
adding a new feature usually means composing rather than implementing.

They're grouped by what they do to shapes, because that's what decides how the
gradient behaves.
"""

import enum


class UnaryOps(enum.Enum):
    """One input, output is the same shape."""

    NEG = enum.auto()
    RELU = enum.auto()
    LOG = enum.auto()
    EXP = enum.auto()
    SQRT = enum.auto()
    SIGMOID = enum.auto()


class BinaryOps(enum.Enum):
    """Two inputs of the same shape, output is that shape too.

    CMPLT ("compare less than") gives back 1.0/0.0 instead of True/False. Having
    comparison produce a number means the backward passes can build masks with
    plain arithmetic instead of needing some kind of branching op.
    """

    ADD = enum.auto()
    SUB = enum.auto()
    MUL = enum.auto()
    DIV = enum.auto()
    CMPLT = enum.auto()
    MAX = enum.auto()


class ReduceOps(enum.Enum):
    """Collapse axes down to length 1.

    These always keep the reduced axes around as size 1 instead of dropping
    them. That way the output can still be broadcast against the input, which
    makes the backward pass a single expand. Dropping the axes happens later,
    in Tensor.sum / Tensor.max.
    """

    SUM = enum.auto()
    MAX = enum.auto()


class MovementOps(enum.Enum):
    """Shuffle elements around without doing any arithmetic.

    The gradient of one of these is always another one: reshape undoes reshape,
    permute undoes permute, and expand undoes with a sum.
    """

    RESHAPE = enum.auto()
    EXPAND = enum.auto()
    PERMUTE = enum.auto()


__all__ = ["UnaryOps", "BinaryOps", "ReduceOps", "MovementOps"]
