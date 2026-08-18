"""Datasets and metrics: the bits at the edges that don't need gradients.

Deliberately outside the engine's abstraction stack. Imports numpy directly and
hands back plain arrays.
"""

from teenygrad.data.datasets import DIGIT_PROTOTYPES, make_toy_digit_dataset
from teenygrad.data.metrics import accuracy

__all__ = ["DIGIT_PROTOTYPES", "make_toy_digit_dataset", "accuracy"]
