"""A toy dataset to train on.

Datasets aren't part of the differentiable computation, so this works in plain
numpy and never builds a Tensor. Labels stay integer arrays, since a class id
has no meaningful gradient and shoving one into a float32 buffer would just be
confusing.
"""

from __future__ import annotations

import numpy as np

# Three 3x3 digit-ish stencils, flattened row-major into length-9 vectors:
#
#     . # .        # # #        # . #
#     # . #        . # .        # # #
#     . # .        # # #        # . #
#   class 0      class 1      class 2
#
# Before noise these are linearly separable, so a working MLP should get close
# to 100% test accuracy. That makes this a decent smoke test. If training
# stalls well under that, the problem is the engine, not the data.
DIGIT_PROTOTYPES = np.array(
    [
        [0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
    ],
    dtype=np.float32,
)


def make_toy_digit_dataset(
    num_samples: int = 200, seed: int | None = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Noisy samples of the three prototypes above.

    Each sample picks a prototype at random and adds Gaussian noise (sigma 0.1)
    to every pixel, small enough that the classes stay separable, big enough
    that the model can't just memorise three exact rows.

    Returns (X, y): X is (num_samples, 9) float32, y is (num_samples,) int64
    with values in [0, 3). Use different seeds for train and test so they're
    genuinely different draws.
    """
    rng = np.random.RandomState(seed)
    num_classes, num_features = DIGIT_PROTOTYPES.shape

    labels = rng.randint(0, num_classes, size=num_samples)
    noise = rng.randn(num_samples, num_features).astype(np.float32) * 0.1
    features = (DIGIT_PROTOTYPES[labels] + noise).astype(np.float32)

    return features, labels.astype(np.int64)


__all__ = ["DIGIT_PROTOTYPES", "make_toy_digit_dataset"]
