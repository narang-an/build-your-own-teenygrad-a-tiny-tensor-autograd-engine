"""Numerical gradient checking.

Every backward pass in the autograd package is a claim about calculus, and it's
possible to get one subtly wrong in a way that still trains well enough to look
fine. This settles it by comparing against the definition of a derivative,
computed with nothing but forward passes:

    df/dx  ~=  (f(x + eps) - f(x - eps)) / (2 * eps)

That's the central difference, which is a lot more accurate than the one-sided
version because the first-order error terms cancel, so its error shrinks like
eps^2 rather than eps.

Picking eps is a real tradeoff. Too big and the leftover quadratic error takes
over. Too small and catastrophic cancellation does, since subtracting two nearly
equal float32 values throws away most of the digits. This engine is float32
(about 7 decimal digits) so 1e-2 is around the sweet spot, and the tolerances
below match. A float64 engine could use 1e-6 and demand much tighter agreement.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from teenygrad import LazyBuffer, Tensor


def numeric_gradient(
    scalar_fn: Callable[[], float], tensor: Tensor, eps: float = 1e-2
) -> np.ndarray:
    """Estimate d(scalar_fn)/d(tensor) with finite differences.

    Nudges one element at a time and re-runs the whole forward pass, which is
    why this is a testing tool and not a training strategy. It's two forward
    passes *per element*, where backprop does all of them in one backward pass.
    That gap is the entire reason autograd exists.

    scalar_fn takes no arguments and returns a float, and has to re-read the
    tensor's values each time it's called. The tensor gets put back how it was
    before returning.
    """
    values = tensor.numpy().copy()
    grad = np.zeros_like(values, dtype=np.float64)

    for index in np.ndindex(*values.shape):
        original = values[index]

        values[index] = original + eps
        tensor.data = LazyBuffer(values.copy())
        plus = scalar_fn()

        values[index] = original - eps
        tensor.data = LazyBuffer(values.copy())
        minus = scalar_fn()

        grad[index] = (plus - minus) / (2 * eps)
        values[index] = original

    tensor.data = LazyBuffer(values)
    return grad


def assert_gradients_match(
    build_loss: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-2,
    rtol: float = 2e-2,
    atol: float = 2e-3,
) -> None:
    """Check the engine's gradients against finite differences for each input.

    build_loss takes *inputs and returns a scalar Tensor. It gets called over
    and over, so it has to build a fresh graph each time rather than closing
    over a cached result.

    atol matters most where the true gradient is near zero and a relative
    comparison would be meaningless.
    """
    for tensor in inputs:
        tensor.requires_grad = True
        tensor.grad = None

    build_loss(*inputs).backward()

    for position, tensor in enumerate(inputs):
        assert tensor.grad is not None, f"input {position} got no gradient at all"
        analytic = tensor.grad.numpy().astype(np.float64)

        numerical = numeric_gradient(
            lambda: float(build_loss(*inputs).item()), tensor, eps=eps
        )

        if not np.allclose(analytic, numerical, rtol=rtol, atol=atol):
            worst = np.unravel_index(
                int(np.argmax(np.abs(analytic - numerical))), analytic.shape
            )
            raise AssertionError(
                f"gradient mismatch for input {position} at index {worst}: "
                f"analytic={analytic[worst]:.6g} numerical={numerical[worst]:.6g}\n"
                f"analytic:\n{analytic}\nnumerical:\n{numerical}"
            )


__all__ = ["numeric_gradient", "assert_gradients_match"]
