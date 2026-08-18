"""Layers and losses, roughly mirroring torch.nn.

Nothing in here knows autograd exists. It just calls Tensor methods and the
gradients happen underneath. Optimisers live in teenygrad.optim, same split as
PyTorch.
"""

from teenygrad.nn.layers import MLP, Linear
from teenygrad.nn.losses import one_hot, sparse_categorical_cross_entropy

__all__ = ["Linear", "MLP", "one_hot", "sparse_categorical_cross_entropy"]
