"""The bottom of the stack: types, helpers, op names, and the numpy backend.

Nothing in here knows what a gradient is. You could delete the autograd package
and core would still work as a (very small) array library.

Probably easiest to read in this order: types, helpers, ops, lazybuffer.
"""

from teenygrad.core.helpers import argsort, prod
from teenygrad.core.lazybuffer import DEFAULT_DTYPE, LazyBuffer
from teenygrad.core.ops import BinaryOps, MovementOps, ReduceOps, UnaryOps
from teenygrad.core.types import ArrayLike, Axis, GradResult, Shape, ShapeLike, TensorLike

__all__ = [
    "Shape",
    "ShapeLike",
    "Axis",
    "ArrayLike",
    "TensorLike",
    "GradResult",
    "prod",
    "argsort",
    "UnaryOps",
    "BinaryOps",
    "ReduceOps",
    "MovementOps",
    "LazyBuffer",
    "DEFAULT_DTYPE",
]
