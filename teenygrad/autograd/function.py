"""Function: the base class that makes autodiff work.

Anything differentiable subclasses this and defines two methods:

    forward(*buffers, **kwargs)  -> compute the output buffer
    backward(grad_output)        -> given dL/d(output), return dL/d(each input)

apply is what ties them together. It builds the Function, unwraps the input
Tensors into raw buffers, runs forward, wraps the result back up, and, if a
gradient is going to be needed, hangs the Function instance off the output as
out._ctx.

That last bit is the whole trick. The output holds a pointer back to the op that
made it, and that op holds pointers to its inputs. So running a forward pass
builds a graph as a side effect, without any separate graph-building step, and
backward() just walks it in reverse.

Since the Function instance is what sticks around, it's also where forward
stashes anything backward will need later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from teenygrad.core.lazybuffer import LazyBuffer
from teenygrad.core.types import GradResult

if TYPE_CHECKING:
    from teenygrad.tensor import Tensor


class Function:
    """Base class for every differentiable op.

    parents holds the input Tensors (only when a gradient is needed, see
    __init__), and backward() walks those edges.

    needs_input_grad has one flag per input, copied from each input's
    requires_grad. backward() checks it to avoid computing gradients nobody
    asked for. That's a real saving: for x @ W in a hidden layer you want dW but
    usually not dx.

    requires_grad says whether the output needs a gradient. True if any
    input does, False if none do, or None for the "nobody said" case that
    tinygrad uses. None is falsy so it acts like False everywhere it matters.
    """

    parents: tuple[Tensor, ...]
    needs_input_grad: list[bool | None]
    requires_grad: bool | None

    def __init__(self, *tensors: Tensor) -> None:
        self.needs_input_grad = [t.requires_grad for t in tensors]

        if any(flag is True for flag in self.needs_input_grad):
            self.requires_grad = True
            # Only hang on to the inputs when there's going to be a backward
            # pass. During plain inference this is what lets the forward graph
            # get garbage collected as each value is used up.
            self.parents = tensors
        elif None in self.needs_input_grad:
            self.requires_grad = None
        else:
            self.requires_grad = False

    def forward(self, *args: LazyBuffer, **kwargs: Any) -> LazyBuffer:
        """Compute the output buffer. Subclasses override this.

        Note it gets LazyBuffers, not Tensors. apply() unwraps them first, so
        op implementations never have to deal with graph bookkeeping.
        """
        raise NotImplementedError(f"forward not implemented for {type(self).__name__}")

    def backward(self, grad_output: LazyBuffer) -> GradResult:
        """Turn the output gradient into input gradients. Subclasses override this.

        Return a single buffer for a one-input op, or a tuple with an entry per
        input (None where needs_input_grad is false) for the rest.
        """
        raise NotImplementedError(f"backward not implemented for {type(self).__name__}")

    @classmethod
    def apply(cls, *tensors: Tensor, **kwargs: Any) -> Tensor:
        """Run this op on some Tensors and wire the result into the graph.

        kwargs are the non-tensor settings that get passed through to forward:
        the axis= of a reduction, the shape= of a reshape. They're separate from
        the positional tensors because you don't differentiate with respect to
        them.
        """
        # Imported in here rather than at the top because there's a real cycle:
        # tensor.py needs the op classes, which need this module, and apply
        # needs Tensor. Deferring to call time is the usual way out. By the
        # time anyone calls apply(), both modules have finished loading.
        from teenygrad.tensor import Tensor

        ctx = cls(*tensors)
        out_buffer = ctx.forward(*[t.lazydata for t in tensors], **kwargs)
        out = Tensor(out_buffer, requires_grad=ctx.requires_grad)
        if ctx.requires_grad:
            out._ctx = ctx
        return out


__all__ = ["Function"]
