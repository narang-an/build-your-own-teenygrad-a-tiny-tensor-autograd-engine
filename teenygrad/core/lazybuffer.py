"""A thin wrapper around a numpy array, plus the ops that run on it.

This is where the op names from ops.py actually get executed, so it's the
backend. It's also the only file in the engine that imports numpy for doing
maths. Everything above it is written using just the six methods below, so
swapping numpy out for something else would mean rewriting this file and
nothing else.

The name is a bit of a lie. In real tinygrad a LazyBuffer records what you
asked for and waits, so a compiler can fuse the graph before running anything.
Mine computes straight away. The interface is the part I wanted to learn; the
laziness is an optimisation on top of it.

Everything is float32. That's a simplification, but it kills off a whole class
of dtype-promotion bugs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from teenygrad.core.ops import BinaryOps, ReduceOps, UnaryOps
from teenygrad.core.types import ArrayLike, Shape

# The one dtype the whole engine uses. Integer labels never make it into a
# buffer. They stay as numpy arrays and get turned into one-hot floats at the
# loss instead.
DEFAULT_DTYPE = np.float32


class LazyBuffer:
    """An n-dimensional float32 array and the six ops you can run on it.

    _np is the actual numpy array. The underscore means it's backend-private:
    code above this layer should go through Tensor.numpy() rather than reaching
    in here.
    """

    __slots__ = ("_np", "shape", "dtype")

    def __init__(self, data: ArrayLike) -> None:
        """Wrap data as a float32 array. No copy if it's already float32."""
        self._np: np.ndarray = np.asarray(data, dtype=DEFAULT_DTYPE)
        # Plain ints rather than numpy ints, so shape tuples compare and print
        # the way you'd expect.
        self.shape: Shape = tuple(int(dim) for dim in self._np.shape)
        self.dtype: np.dtype = self._np.dtype

    # ---- making buffers --------------------------------------------------

    @staticmethod
    def const(value: float, shape: Shape) -> LazyBuffer:
        """A buffer of shape filled with value.

        The backward passes need this constantly. A rule like "multiply by 2"
        needs a real buffer full of 2s, because the binary ops only know how to
        combine two buffers.
        """
        return LazyBuffer(np.full(shape, value, dtype=DEFAULT_DTYPE))

    @staticmethod
    def rand(shape: Shape, seed: int | None = None) -> LazyBuffer:
        """Uniform random floats in [0, 1). Pass a seed to make it repeatable."""
        rng = np.random.RandomState(seed)
        return LazyBuffer(rng.random(shape))

    # ---- the six primitives ----------------------------------------------

    def e(self, op: UnaryOps | BinaryOps, *srcs: LazyBuffer) -> LazyBuffer:
        """Run an elementwise op. No srcs for unary, one src for binary.

        Both arities share a method because they look identical from the graph's
        point of view: shapes go in matched and come out matched, nothing moves.

        A binary source has to already be the same shape as self. Broadcasting
        is *not* applied here. Tensor._broadcast does it one layer up using real
        expand nodes, which is what makes the broadcast differentiable, so this
        raising is on purpose.
        """
        x = self._np

        if isinstance(op, UnaryOps):
            if srcs:
                raise ValueError(f"{op} is unary but got {len(srcs)} extra source(s)")
            if op is UnaryOps.NEG:
                return LazyBuffer(-x)
            if op is UnaryOps.RELU:
                return LazyBuffer(np.maximum(x, 0))
            if op is UnaryOps.LOG:
                return LazyBuffer(np.log(x))
            if op is UnaryOps.EXP:
                return LazyBuffer(np.exp(x))
            if op is UnaryOps.SQRT:
                return LazyBuffer(np.sqrt(x))
            if op is UnaryOps.SIGMOID:
                # Two branches so exp() can't blow up. The plain 1/(1+exp(-x))
                # overflows around x = -100 in float32, so for negative x use
                # exp(x)/(1+exp(x)), which is the same thing rearranged.
                positive = 1.0 / (1.0 + np.exp(-np.abs(x)))
                return LazyBuffer(np.where(x >= 0, positive, 1.0 - positive))
            raise ValueError(f"Unknown unary op: {op}")

        if isinstance(op, BinaryOps):
            if len(srcs) != 1:
                raise ValueError(f"{op} is binary but got {len(srcs)} source(s)")
            other = srcs[0]
            if self.shape != other.shape:
                raise ValueError(
                    f"binary op {op} needs matching shapes, got {self.shape} and "
                    f"{other.shape}; broadcast with Tensor.expand first"
                )
            y = other._np
            if op is BinaryOps.ADD:
                return LazyBuffer(x + y)
            if op is BinaryOps.SUB:
                return LazyBuffer(x - y)
            if op is BinaryOps.MUL:
                return LazyBuffer(x * y)
            if op is BinaryOps.DIV:
                return LazyBuffer(x / y)
            if op is BinaryOps.CMPLT:
                # Cast the bool mask to float so the backward passes can do
                # arithmetic with it.
                return LazyBuffer((x < y).astype(DEFAULT_DTYPE))
            if op is BinaryOps.MAX:
                return LazyBuffer(np.maximum(x, y))
            raise ValueError(f"Unknown binary op: {op}")

        raise ValueError(f"Not an elementwise op: {op}")

    def r(self, op: ReduceOps, axis: tuple[int, ...]) -> LazyBuffer:
        """Reduce over axis, leaving the reduced axes as size 1.

        axis is a tuple of non-negative indices. An empty tuple is fine and
        does nothing, which Expand.backward relies on.
        """
        if op is ReduceOps.SUM:
            return LazyBuffer(self._np.sum(axis=axis, keepdims=True))
        if op is ReduceOps.MAX:
            return LazyBuffer(self._np.max(axis=axis, keepdims=True))
        raise ValueError(f"Unknown reduce op: {op}")

    def reshape(self, new_shape: Shape) -> LazyBuffer:
        """Same elements, new shape. numpy raises if the counts don't match."""
        return LazyBuffer(self._np.reshape(new_shape))

    def expand(self, new_shape: Shape) -> LazyBuffer:
        """Stretch size-1 axes out to a bigger shape.

        Only axes that are currently 1 can grow; everything else has to match
        already. This is the op behind matrix + row_vector.
        """
        # broadcast_to hands back a read-only view with zero strides. Copying it
        # into a real array avoids handing anyone a buffer that refuses writes
        # or secretly shares memory with its source.
        stretched = np.broadcast_to(self._np, tuple(int(d) for d in new_shape))
        return LazyBuffer(np.array(stretched))

    def permute(self, order: Sequence[int]) -> LazyBuffer:
        """Reorder axes. Axis order[i] of the input ends up as axis i of the output."""
        return LazyBuffer(self._np.transpose(order))

    # ---- interop ---------------------------------------------------------

    def __array__(self, dtype: object = None, copy: bool | None = None) -> np.ndarray:
        """Makes np.asarray(buffer) work. dtype/copy are the numpy 2.x protocol."""
        arr = self._np if dtype is None else self._np.astype(dtype)
        return np.array(arr) if copy else arr

    def __float__(self) -> float:
        return float(self._np.item())

    def __repr__(self) -> str:
        return f"LazyBuffer(shape={self.shape}, {self._np!r})"

    def __str__(self) -> str:
        return str(self._np)


__all__ = ["LazyBuffer", "DEFAULT_DTYPE"]
