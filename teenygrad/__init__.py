"""teenygrad: a tiny tensor library with reverse-mode automatic differentiation.

A from-scratch numpy-backed miniature of tinygrad, written to be read rather
than to be fast. About a thousand lines gets you from "multiply two numbers" to
"train a neural network".

Train something:

    from teenygrad import evaluate_mlp, make_toy_digit_dataset, train_mlp

    X, y = make_toy_digit_dataset(200, seed=0)
    model, losses = train_mlp(X, y, epochs=30)
    print(losses[0], "->", losses[-1])

Or use the engine directly:

    from teenygrad import Tensor

    w = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    x = Tensor([[1.0, 1.0]])
    loss = (x @ w).sum()
    loss.backward()
    print(w.grad)

Layout, bottom to top. Each layer only imports from the ones above it:

    core.types          type aliases
    core.helpers        prod, argsort
    core.ops            names for the ~17 primitive operations
    core.lazybuffer     runs those primitives on numpy (the backend)
    autograd.function   Function base class and apply(); builds the graph
    autograd.unary      same-shape elementwise ops and their derivatives
    autograd.binary     two-input elementwise ops
    autograd.reduce     axis-collapsing ops
    autograd.movement   reshape, expand, permute
    tensor              the user-facing API, composed from everything above
    nn.layers           Linear, MLP
    nn.losses           cross-entropy
    optim               SGD update and gradient clearing
    data                toy dataset and accuracy
    training            the training loop

See README.md for how a gradient actually gets from the loss back to a weight.
"""

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
from teenygrad.core import (
    DEFAULT_DTYPE,
    BinaryOps,
    LazyBuffer,
    MovementOps,
    ReduceOps,
    UnaryOps,
    argsort,
    prod,
)
from teenygrad.data import DIGIT_PROTOTYPES, accuracy, make_toy_digit_dataset
from teenygrad.nn import MLP, Linear, one_hot, sparse_categorical_cross_entropy
from teenygrad.optim import sgd_step, zero_grad
from teenygrad.tensor import Tensor
from teenygrad.training import evaluate_mlp, train_mlp

__version__ = "1.0.0"

__all__ = [
    # core
    "prod",
    "argsort",
    "UnaryOps",
    "BinaryOps",
    "ReduceOps",
    "MovementOps",
    "LazyBuffer",
    "DEFAULT_DTYPE",
    # autograd
    "Function",
    "Neg",
    "Relu",
    "Log",
    "Exp",
    "Sqrt",
    "Sigmoid",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Sum",
    "Max",
    "Reshape",
    "Expand",
    "Permute",
    # tensor
    "Tensor",
    # nn
    "Linear",
    "MLP",
    "one_hot",
    "sparse_categorical_cross_entropy",
    # optim
    "sgd_step",
    "zero_grad",
    # data
    "DIGIT_PROTOTYPES",
    "make_toy_digit_dataset",
    "accuracy",
    # training
    "train_mlp",
    "evaluate_mlp",
]
