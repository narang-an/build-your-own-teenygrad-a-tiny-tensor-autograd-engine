"""Tests for optim.py.

Small file, but two of these cover things that are easy to break and horrible to
debug: the update has to mutate parameters in place, and it must not leave its
own arithmetic in the graph.
"""

import numpy as np
import pytest

from teenygrad import Tensor, sgd_step, zero_grad


def parameter(values, grad=None):
    """Build a trainable Tensor, optionally with a gradient already attached."""
    param = Tensor(np.array(values, dtype=np.float32), requires_grad=True)
    if grad is not None:
        param.grad = Tensor(np.array(grad, dtype=np.float32))
    return param


def test_sgd_moves_against_the_gradient():
    """p <- p - lr * grad."""
    param = parameter([1.0, 2.0], grad=[0.5, -1.0])
    sgd_step([param], learning_rate=0.1)
    assert np.allclose(param.numpy(), [0.95, 2.1], atol=1e-6)


def test_sgd_mutates_in_place():
    """The Tensor object has to survive the update.

    model.parameters() hands out references and the layers hold the same ones.
    Rebinding instead of mutating would train a detached copy while the model's
    real weights sat still, which shows up as "loss goes down, model never gets
    any better".
    """
    param = parameter([1.0], grad=[1.0])
    held_reference = param
    sgd_step([param], learning_rate=0.5)
    assert held_reference is param
    assert held_reference.numpy()[0] == pytest.approx(0.5)


def test_sgd_does_not_extend_the_graph():
    """An update is bookkeeping, not computation, so it must not create a _ctx.

    If it did, every step would chain onto the last one and the graph would grow
    forever across epochs.
    """
    param = parameter([1.0], grad=[1.0])
    sgd_step([param], learning_rate=0.1)
    assert param._ctx is None
    assert param.requires_grad is True


def test_sgd_skips_parameters_without_a_gradient():
    """A parameter that took no part in the loss is left alone."""
    param = parameter([1.0, 2.0])
    sgd_step([param], learning_rate=0.1)
    assert np.allclose(param.numpy(), [1.0, 2.0])


def test_sgd_rejects_a_mismatched_gradient_shape():
    """A wrong-shaped gradient means a broadcast went wrong upstream."""
    param = parameter([1.0, 2.0], grad=[[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="does not match parameter"):
        sgd_step([param], learning_rate=0.1)


def test_larger_learning_rate_takes_a_larger_step():
    """The learning rate scales the step linearly."""
    small = parameter([1.0], grad=[1.0])
    large = parameter([1.0], grad=[1.0])
    sgd_step([small], learning_rate=0.01)
    sgd_step([large], learning_rate=0.10)
    assert large.numpy()[0] < small.numpy()[0] < 1.0


def test_zero_grad_clears_every_gradient():
    """After zero_grad, nothing is left over from the previous step."""
    params = [parameter([1.0], grad=[1.0]), parameter([2.0], grad=[2.0])]
    zero_grad(params)
    assert all(p.grad is None for p in params)


def test_without_zero_grad_gradients_pile_up():
    """Shows exactly why zero_grad exists.

    Two identical backward passes with no clearing in between double the
    gradient, which would quietly double the effective learning rate.
    """
    x = Tensor([3.0], requires_grad=True)
    x.mul(2.0).sum().backward()
    first = x.grad.item()
    x.mul(2.0).sum().backward()
    assert x.grad.item() == pytest.approx(2 * first)

    zero_grad([x])
    x.mul(2.0).sum().backward()
    assert x.grad.item() == pytest.approx(first)
