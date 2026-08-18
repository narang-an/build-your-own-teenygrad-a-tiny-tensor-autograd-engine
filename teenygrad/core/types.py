"""Type aliases, kept in one place so a shape is spelled the same way everywhere.

None of it really exists at runtime, it's just annotations for the type checker.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, Union

import numpy as np

if TYPE_CHECKING:
    from teenygrad.core.lazybuffer import LazyBuffer
    from teenygrad.tensor import Tensor


# A real, resolved shape: (2, 3), or () for a scalar.
Shape: TypeAlias = tuple[int, ...]

# A shape as someone might pass it in, before it gets normalised. reshape takes
# either reshape(2, 3) or reshape((2, 3)).
ShapeLike: TypeAlias = Union[int, Sequence[int]]

# Which axes to reduce over. None means all of them, negatives count from the
# end, and a sequence picks several.
Axis: TypeAlias = Union[int, Sequence[int], None]

# Anything the LazyBuffer constructor can turn into a float32 array.
ArrayLike: TypeAlias = Any

# Anything you can use where a Tensor is expected. Scalars on the right of an
# operator (t * 2) get promoted automatically.
TensorLike: TypeAlias = Union["Tensor", "LazyBuffer", float, int, Sequence[Any], np.ndarray]

# What backward() hands back: one gradient for a single-input op, or a tuple
# with an entry per input (None where no gradient was asked for).
GradResult: TypeAlias = Union["LazyBuffer", tuple[Union["LazyBuffer", None], ...]]

__all__ = ["Shape", "ShapeLike", "Axis", "ArrayLike", "TensorLike", "GradResult"]
