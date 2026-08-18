"""Tensor: the thing you actually write code against.

A Tensor bundles three things:

    lazydata    the numbers, as a LazyBuffer
    _ctx        the Function that made it, or None if it's a leaf
    grad        where backward() puts dL/d(this tensor)

Two ideas do most of the work here.

First, composition. Only a handful of methods call Function.apply directly.
matmul, mean, transpose, softmax and log_softmax are all written in terms of
those, which makes them differentiable for free. There's no derivative of a
matmul or a softmax anywhere in this project. They fall out of the chain rule
applied to reshape, expand, mul and sum.

Second, broadcasting is a real operation. x + bias with mismatched shapes
doesn't quietly reinterpret memory, it puts actual Reshape and Expand nodes in
the graph. Expand has a backward pass (sum over the stretched axes), so the
gradient reaches bias with the right shape and nothing else needs a special case.

I kept this as one file. Splitting a class this size across modules needs mixins,
and chasing one object's methods through four files seemed worse than scrolling.
The section banners are the map.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from teenygrad.autograd import (
    Add,
    Div,
    Exp,
    Expand,
    Function,
    Log,
    Max,
    Mul,
    Neg,
    Permute,
    Relu,
    Reshape,
    Sigmoid,
    Sqrt,
    Sub,
    Sum,
)
from teenygrad.core.helpers import prod
from teenygrad.core.lazybuffer import LazyBuffer
from teenygrad.core.ops import BinaryOps
from teenygrad.core.types import ArrayLike, Axis, Shape, ShapeLike, TensorLike


class Tensor:
    """An n-dimensional array that remembers how it was computed.

    Set requires_grad=True on things you want to train (weights, biases) and
    leave it False on inputs and targets. After backward(), grad holds a
    Tensor of the same shape. Gradients accumulate rather than overwrite, so
    call zero_grad between training steps.
    """

    __slots__ = ("lazydata", "requires_grad", "grad", "_ctx")

    def __init__(
        self,
        data: TensorLike,
        requires_grad: bool | None = False,
        _ctx: Function | None = None,
    ) -> None:
        """Wrap data in a Tensor.

        data can be a LazyBuffer, numpy array, nested list or scalar.
        Anything that isn't already a buffer gets converted (and so cast to
        float32). _ctx is internal: Function.apply sets it, not you.
        """
        self.lazydata: LazyBuffer = (
            data if isinstance(data, LazyBuffer) else LazyBuffer(data)
        )
        self.requires_grad: bool | None = requires_grad
        self.grad: Tensor | None = None
        self._ctx: Function | None = _ctx

    # =====================================================================
    # Properties
    # =====================================================================

    @property
    def data(self) -> LazyBuffer:
        """The underlying LazyBuffer."""
        return self.lazydata

    @data.setter
    def data(self, value: LazyBuffer | ArrayLike) -> None:
        """Swap the buffer out in place.

        The optimiser uses this to write updated weights into a parameter
        without making a new Tensor, which would break the object identity that
        model.parameters() handed out.
        """
        self.lazydata = value if isinstance(value, LazyBuffer) else LazyBuffer(value)

    @property
    def shape(self) -> Shape:
        """Shape, as a tuple of ints."""
        return self.lazydata.shape

    @property
    def dtype(self) -> np.dtype:
        """Always float32 here."""
        return self.lazydata.dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    def numpy(self) -> np.ndarray:
        """The values as a numpy array. This is the way to read data back out."""
        return self.lazydata._np

    def item(self) -> float:
        """Pull a single-element tensor out as a Python float."""
        return float(self.lazydata._np.item())

    def detach(self) -> Tensor:
        """Same values, but cut out of the graph so gradients stop here.

        Useful for things that are numerically part of a computation but
        shouldn't be differentiated through. The stability shift in log_softmax
        is the one place I use it.
        """
        return Tensor(self.lazydata)

    def __repr__(self) -> str:
        flag = ", requires_grad=True" if self.requires_grad else ""
        return f"<Tensor shape={self.shape}{flag}>\n{self.lazydata._np}"

    # =====================================================================
    # Making tensors
    # =====================================================================

    @staticmethod
    def zeros(shape: Shape, requires_grad: bool = False) -> Tensor:
        """Filled with 0.0."""
        return Tensor(LazyBuffer.const(0, shape), requires_grad=requires_grad)

    @staticmethod
    def ones(shape: Shape, requires_grad: bool = False) -> Tensor:
        """Filled with 1.0."""
        return Tensor(LazyBuffer.const(1, shape), requires_grad=requires_grad)

    @staticmethod
    def full(shape: Shape, value: float, requires_grad: bool = False) -> Tensor:
        """Filled with value."""
        return Tensor(LazyBuffer.const(value, shape), requires_grad=requires_grad)

    @staticmethod
    def rand(
        shape: Shape, seed: int | None = None, requires_grad: bool = False
    ) -> Tensor:
        """Uniform samples in [0, 1)."""
        return Tensor(LazyBuffer.rand(shape, seed=seed), requires_grad=requires_grad)

    @staticmethod
    def randn(
        shape: ShapeLike, seed: int | None = None, requires_grad: bool = False
    ) -> Tensor:
        """Standard normal samples. shape can be a tuple or a single int."""
        if not isinstance(shape, tuple):
            shape = tuple(shape) if hasattr(shape, "__iter__") else (shape,)
        rng = np.random.RandomState(seed)
        return Tensor(rng.randn(*shape), requires_grad=requires_grad)

    # =====================================================================
    # Unary ops
    #
    # One line each. Together they're the whole bridge from the nice API down
    # to the autograd machinery.
    # =====================================================================

    def neg(self) -> Tensor:
        """Elementwise -self."""
        return Neg.apply(self)

    def relu(self) -> Tensor:
        """Elementwise max(self, 0)."""
        return Relu.apply(self)

    def log(self) -> Tensor:
        """Elementwise natural log."""
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Elementwise e ** self."""
        return Exp.apply(self)

    def sqrt(self) -> Tensor:
        """Elementwise square root."""
        return Sqrt.apply(self)

    def sigmoid(self) -> Tensor:
        """Elementwise 1 / (1 + e ** -self)."""
        return Sigmoid.apply(self)

    def __neg__(self) -> Tensor:
        return self.neg()

    # =====================================================================
    # Broadcasting and binary ops
    # =====================================================================

    def _broadcast(self, other: TensorLike) -> tuple[Tensor, Tensor]:
        """Get self and other to a common shape, differentiably.

        Normal numpy rules: line the shapes up from the right, and any axis of
        length 1 can stretch to match the other side.

        The part that matters is how. Instead of letting numpy materialise the
        broadcast arrays (which would detach the result from the graph), this
        emits real reshape and expand nodes. Those have backward passes, so the
        gradient of a broadcast (summing back over the stretched axes) is
        handled automatically. It's why adding a (out,) bias to a (batch, out)
        activation gives back a correctly-shaped (out,) gradient with no
        special-case code.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # Match ranks first by left-padding the shorter shape with 1s, so
        # (3,) against (2, 3) becomes (1, 3) against (2, 3).
        rank = max(self.ndim, other.ndim)
        x = self
        if self.ndim < rank:
            x = self.reshape((1,) * (rank - self.ndim) + self.shape)
        y = (
            other.reshape((1,) * (rank - other.ndim) + other.shape)
            if other.ndim < rank
            else other
        )

        # Work out the result shape and reject anything genuinely incompatible.
        target: list[int] = []
        for axis, (dx, dy) in enumerate(zip(x.shape, y.shape)):
            if dx != dy and dx != 1 and dy != 1:
                raise ValueError(
                    f"cannot broadcast shapes {self.shape} and {other.shape}: "
                    f"axis {axis} has sizes {dx} and {dy}"
                )
            target.append(max(dx, dy))
        common: Shape = tuple(target)

        # Stretch each side up, skipping the expand where it already matches so
        # the graph doesn't fill up with no-op nodes.
        if x.shape != common:
            x = x.expand(common)
        if y.shape != common:
            y = y.expand(common)
        return x, y

    def add(self, other: TensorLike) -> Tensor:
        """Elementwise self + other, broadcasting if needed."""
        return Add.apply(*self._broadcast(other))

    def sub(self, other: TensorLike) -> Tensor:
        """Elementwise self - other, broadcasting if needed."""
        return Sub.apply(*self._broadcast(other))

    def mul(self, other: TensorLike) -> Tensor:
        """Elementwise self * other, broadcasting if needed."""
        return Mul.apply(*self._broadcast(other))

    def div(self, other: TensorLike) -> Tensor:
        """Elementwise self / other, broadcasting if needed."""
        return Div.apply(*self._broadcast(other))

    __add__ = add
    __sub__ = sub
    __mul__ = mul
    __truediv__ = div

    # Reflected versions so 2 * tensor works, not just tensor * 2. Sub and div
    # aren't commutative, hence the explicit ordering.
    def __radd__(self, other: TensorLike) -> Tensor:
        return Tensor(other).add(self)

    def __rmul__(self, other: TensorLike) -> Tensor:
        return Tensor(other).mul(self)

    def __rsub__(self, other: TensorLike) -> Tensor:
        return Tensor(other).sub(self)

    def __rtruediv__(self, other: TensorLike) -> Tensor:
        return Tensor(other).div(self)

    # =====================================================================
    # Movement
    # =====================================================================

    @staticmethod
    def _as_shape(args: tuple[Any, ...]) -> Shape:
        """Let both t.reshape(2, 3) and t.reshape((2, 3)) work."""
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return tuple(args[0])
        return tuple(args)

    def reshape(self, *shape: ShapeLike) -> Tensor:
        """Same elements, new shape. Takes separate ints or one tuple/list."""
        return Reshape.apply(self, shape=self._as_shape(shape))

    def expand(self, *shape: ShapeLike) -> Tensor:
        """Stretch size-1 axes out. Takes separate ints or one tuple/list."""
        return Expand.apply(self, shape=self._as_shape(shape))

    def permute(self, *order: ShapeLike) -> Tensor:
        """Reorder axes. Takes separate ints or one tuple/list."""
        return Permute.apply(self, order=self._as_shape(order))

    def transpose(self, ax1: int = -2, ax2: int = -1) -> Tensor:
        """Swap two axes, defaulting to the last two (a matrix transpose).

        Built on permute rather than being its own op, so it inherits the
        gradient for free.
        """
        order = list(range(self.ndim))
        order[ax1], order[ax2] = order[ax2], order[ax1]
        return self.permute(order)

    @property
    def T(self) -> Tensor:
        """Shorthand for transposing the last two axes."""
        return self.transpose()

    # =====================================================================
    # Reductions
    # =====================================================================

    def _resolve_axis(self, axis: Axis) -> tuple[int, ...]:
        """Turn an axis argument into a tuple of non-negative indices."""
        if self.ndim == 0:
            # A 0-d tensor has no axes, so every axis spec means "none".
            return ()
        if axis is None:
            return tuple(range(self.ndim))
        if isinstance(axis, int):
            return (axis % self.ndim,)
        return tuple(a % self.ndim for a in axis)

    def _reduce(self, fn: type[Function], axis: Axis, keepdim: bool) -> Tensor:
        """Shared plumbing for sum and max.

        The reduce Functions always keep the reduced axes at size 1, since that
        makes their backward a single expand. Dropping them for keepdim=False
        happens here with a reshape, which is differentiable too, so nothing
        gets lost.
        """
        axes = self._resolve_axis(axis)
        out = fn.apply(self, axis=axes)
        if keepdim:
            return out
        kept = tuple(dim for i, dim in enumerate(self.shape) if i not in axes)
        return out.reshape(kept)

    def sum(self, axis: Axis = None, keepdim: bool = False) -> Tensor:
        """Sum over axis (None for all of them, or an int, or a tuple)."""
        return self._reduce(Sum, axis, keepdim)

    def max(self, axis: Axis = None, keepdim: bool = False) -> Tensor:
        """Maximum over axis. Same arguments as sum."""
        return self._reduce(Max, axis, keepdim)

    def mean(self, axis: Axis = None, keepdim: bool = False) -> Tensor:
        """Mean over axis.

        Written as sum / count instead of being its own op, so its gradient
        (spread evenly over whatever got averaged) comes from the Sum and Div
        rules automatically.
        """
        total = self.sum(axis=axis, keepdim=keepdim)
        # How many input elements went into each output cell.
        count = prod(self.shape) // max(prod(total.shape), 1)
        return total.div(float(count))

    # =====================================================================
    # Matmul
    # =====================================================================

    def matmul(self, other: TensorLike) -> Tensor:
        """2-D matrix product, (m, k) @ (k, n) -> (m, n).

        Instead of adding a Matmul op with a hand-derived gradient, this writes
        out the definition out[i,j] = sum_k a[i,k] * b[k,j] literally: line both
        operands up in a shared (m, k, n) space, multiply elementwise, then sum
        away the shared k axis.

        Since reshape, expand, mul and sum all already have backward passes,
        composing them gives the textbook dA = dOut @ B.T and dB = A.T @ dOut
        without me having to write either one down. There's a test for that.

        The catch is memory: it builds an m*k*n intermediate that a real BLAS
        call wouldn't. Fine at this size, and a decent illustration of why
        production frameworks special-case matmul instead.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError(
                f"matmul expects two 2-D tensors, got shapes {self.shape} and {other.shape}"
            )
        m, k = self.shape
        k_other, n = other.shape
        if k != k_other:
            raise ValueError(
                f"matmul inner dimensions disagree: {self.shape} @ {other.shape}"
            )

        a = self.reshape(m, k, 1).expand(m, k, n)
        b = other.reshape(1, k, n).expand(m, k, n)
        return a.mul(b).sum(axis=1)

    def __matmul__(self, other: TensorLike) -> Tensor:
        return self.matmul(other)

    # =====================================================================
    # Softmax
    # =====================================================================

    def log_softmax(self, axis: int = -1) -> Tensor:
        """log(softmax(x)) along axis, done stably.

        The plain version is x - log(sum(exp(x))), but exp of a big logit
        overflows, so subtract the row max first. Softmax doesn't care about a
        constant shift, which is what makes that legal.

        The max gets detached. It's a numerical trick, not part of the function
        being differentiated, so it shouldn't contribute gradient. Saying that
        explicitly also avoids building a Max backward node every training step.
        """
        shift = self.max(axis=axis, keepdim=True).detach()
        shifted = self.sub(shift)
        log_sum_exp = shifted.exp().sum(axis=axis, keepdim=True).log()
        return shifted.sub(log_sum_exp)

    def softmax(self, axis: int = -1) -> Tensor:
        """Softmax along axis. Computed as exp(log_softmax) to reuse the
        stable path above."""
        return self.log_softmax(axis=axis).exp()

    # =====================================================================
    # Autograd
    # =====================================================================

    def _topological_order(self) -> list[Tensor]:
        """Every tensor this one depends on, parents before children.

        Depth-first walk over _ctx.parents, appending a node only after
        recursing into its parents. That ordering is what backprop needs: going
        through the reversed list, a node is only reached once everything that
        consumed it has been handled, so its gradient is complete.
        """
        visited: set[int] = set()
        order: list[Tensor] = []

        def visit(node: Tensor) -> None:
            visited.add(id(node))
            if node._ctx is not None:
                for parent in node._ctx.parents:
                    if id(parent) not in visited:
                        visit(parent)
            order.append(node)

        visit(self)
        return order

    def backward(self) -> Tensor:
        """Fill in gradients for everything this tensor was computed from.

        Call it on a scalar loss. It seeds self.grad with ones (dL/dL = 1), then
        walks the graph backwards handing each node's gradient to the Function
        that produced it and passing the results on to that Function's inputs.

        Gradients accumulate instead of overwriting. That's needed for
        correctness, since a tensor used twice has to collect gradient from both
        paths, and it's why zero_grad exists.
        """
        # Seeding with a full ones buffer instead of a scalar also means calling
        # backward() on a non-scalar tensor works out to "sum my elements, then
        # differentiate that", which is handy in tests.
        self.grad = Tensor(LazyBuffer.const(1, self.shape))

        for node in reversed(self._topological_order()):
            # Leaves have no op behind them, and a node can be in the graph
            # without having got any gradient (a branch the loss ignored).
            if node._ctx is None or node.grad is None:
                continue

            grads = node._ctx.backward(node.grad.lazydata)
            if not isinstance(grads, (tuple, list)):
                grads = (grads,)

            for parent, grad in zip(node._ctx.parents, grads):
                if grad is None or not parent.requires_grad:
                    continue
                if parent.grad is None:
                    parent.grad = Tensor(grad)
                else:
                    # Add at the buffer level. Gradients are data, not part of
                    # the graph, so combining them shouldn't create new nodes.
                    parent.grad = Tensor(parent.grad.lazydata.e(BinaryOps.ADD, grad))
        return self

    # =====================================================================
    # numpy interop
    # =====================================================================

    def __array__(self, dtype: object = None, copy: bool | None = None) -> np.ndarray:
        """Makes np.asarray(tensor) work."""
        arr = self.numpy() if dtype is None else self.numpy().astype(dtype)
        return np.array(arr) if copy else arr

    def __float__(self) -> float:
        return self.item()

    def __len__(self) -> int:
        return self.shape[0]


__all__ = ["Tensor"]
