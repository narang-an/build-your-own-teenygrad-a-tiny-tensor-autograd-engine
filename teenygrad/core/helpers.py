"""Two small helper functions. No numpy, no tensors, no imports from anywhere else."""

from __future__ import annotations

from collections.abc import Sequence


def prod(shape: Sequence[int]) -> int:
    """Multiply out a shape to get its total element count.

    An empty shape gives 1, which is the right answer for a scalar.


    >>> prod((2, 3, 4))
    24
    """
    result = 1
    for dim in shape:
        result *= dim
    return result


def argsort(values: Sequence[int]) -> list[int]:
    """Return the indices that would sort values ascending.

    The only thing this is used for is flipping a permutation around. If the
    forward pass shuffled axes with order (2, 0, 1), the backward pass needs to
    undo that, and argsort((2, 0, 1)) == (1, 2, 0) is the undo order.

    >>> argsort([2, 0, 1])
    [1, 2, 0]
    """
    # Insertion sort. It's O(n^2), but n here is a tensor's number of
    # dimensions, so it's basically always under 5. Writing it by hand keeps
    # numpy out of this layer.
    order: list[int] = []
    for candidate in range(len(values)):
        position = 0
        # >= rather than > keeps equal values in their original order.
        while position < len(order) and values[candidate] >= values[order[position]]:
            position += 1
        order.insert(position, candidate)
    return order


__all__ = ["prod", "argsort"]
