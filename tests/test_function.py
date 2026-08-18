"""Tests for autograd/function.py, i.e. building the graph.

These check the bookkeeping apply() does, separately from whether any particular
derivative is correct. That's test_gradients.py's job.
"""

import pytest

from teenygrad import Function, Relu, Tensor


def test_function_stubs_raise():
    """The base forward/backward exist and refuse to run.

    Regression test. These used to be defined inside a helper function that
    never got called, so Function.forward didn't exist at all and forgetting to
    override it showed up as a confusing AttributeError instead of this.
    """

    class Incomplete(Function):
        pass

    with pytest.raises(NotImplementedError, match="forward"):
        Incomplete().forward()
    with pytest.raises(NotImplementedError, match="backward"):
        Incomplete().backward(None)


def test_apply_attaches_context_only_when_a_gradient_is_needed():
    """The graph edge is created exactly when someone downstream will want it."""
    tracked = Relu.apply(Tensor([1.0, -1.0], requires_grad=True))
    assert tracked.requires_grad is True
    assert isinstance(tracked._ctx, Relu)

    untracked = Relu.apply(Tensor([1.0, -1.0]))
    assert untracked.requires_grad is False
    assert untracked._ctx is None


def test_parents_are_only_retained_when_needed():
    """Inference doesn't pin the forward graph in memory.

    parents is the reference keeping intermediate tensors alive. With no input
    needing a gradient there's nothing to walk back through, so it never gets
    set and the graph can be collected right away.
    """
    tracked = Relu.apply(Tensor([1.0], requires_grad=True))
    assert len(tracked._ctx.parents) == 1

    ctx = Relu(Tensor([1.0]))
    assert not hasattr(ctx, "parents")


def test_needs_input_grad_reflects_the_inputs():
    """Per-input flags let backward skip gradients nobody asked for."""
    out = Tensor([1.0], requires_grad=True).add(Tensor([2.0]))
    assert out._ctx.needs_input_grad == [True, False]


def test_requires_grad_is_none_when_unspecified():
    """None means nobody decided. It has to stay falsy so every
    `if requires_grad` check treats it as False."""
    ctx = Relu(Tensor([1.0], requires_grad=None))
    assert ctx.requires_grad is None
    assert not ctx.requires_grad


def test_kwargs_are_not_treated_as_differentiable_inputs():
    """Settings like axis= are passed separately from the tensors.

    Only positional Tensors become graph parents. Keyword arguments are just
    configuration and never get a gradient.
    """
    out = Tensor([[1.0, 2.0]], requires_grad=True).sum(axis=1)
    reduce_node = out._ctx.parents[0]
    assert len(reduce_node._ctx.parents) == 1
