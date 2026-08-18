"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - prod
def prod(shape):
    # Multiply together the elements of a shape tuple to get the total number of elements.
    res = 1
    for num in shape:
        res *= num
    return res

# Step 2 - argsort
import numpy as np 

def argsort(values):
    # Return the indices that would sort values in ascending order.
    if len(values) == 0:
        return []

    result = [0]

    for val_idx in range(1, len(values)):
        result_idx = 0
        while result_idx < len(result) and values[val_idx] >= values[result[result_idx]]:
            result_idx += 1
        result.insert(result_idx, val_idx)

    return result

# Step 3 - make_op_enums
import enum

def make_op_enums():
    # four enum classes naming every supported operation kind
    UnaryOps = enum.Enum('UnaryOps', ['NEG', 'RELU', 'LOG', 'EXP', 'SQRT', 'SIGMOID'])
    BinaryOps = enum.Enum('BinaryOps', ['ADD', 'SUB', 'MUL', 'DIV', 'CMPLT', 'MAX'])
    ReduceOps = enum.Enum('ReduceOps', ['SUM', 'MAX'])
    MovementOps = enum.Enum('MovementOps', ['RESHAPE', 'EXPAND', 'PERMUTE'])

    return UnaryOps, BinaryOps, ReduceOps, MovementOps

UnaryOps, BinaryOps, ReduceOps, MovementOps = make_op_enums()

# Step 4 - LazyBuffer
class LazyBuffer:
    def __init__(self, np_array):
        # wrap np_array as an ndarray and expose shape and dtype
        self._np = np.asarray(np_array)
        self.shape = tuple(int(d) for d in self._np.shape)
        self.dtype = self._np.dtype

    def __array__(self):
        return np.asarray(self._np, dtype=self.dtype)
    
    def __float__(self):
        return float(self._np)

    def __repr__(self):
        return f"LazyBuffer({repr(self._np)})"

    def __str__(self):
        return str(self._np)

# Step 5 - lazybuffer_const
def const(value, shape):
    # Create a new LazyBuffer of the given shape filled with a constant value.
    arr = np.full(shape, value, dtype=np.float32)
    return LazyBuffer(arr)

LazyBuffer.const = staticmethod(const)

# Step 6 - rand
def rand(shape, seed=None):
    # return a LazyBuffer of uniform random floats in [0, 1) with given shape
    rng = np.random.RandomState(seed)
    data = rng.random(shape).astype(np.float32)
    return LazyBuffer(data)

# Step 7 - lazybuffer_unary_e
def e(self, op):
    # apply a unary elementwise op (NEG, RELU, LOG, EXP, SQRT, SIGMOID)
    x = self._np
    out = None

    if op == UnaryOps.NEG:
        out = -x
    elif op == UnaryOps.RELU:
        out = np.maximum(x, 0)
    elif op == UnaryOps.LOG:
        out = np.log(x)
    elif op == UnaryOps.EXP:
        out = np.exp(x)
    elif op == UnaryOps.SQRT:
        out = np.sqrt(x)
    elif op == UnaryOps.SIGMOID:
        out = 1.0 / (1.0 + np.exp(-x))
    else:
        raise ValueError(f"Unknown unary op: {op}")

    return LazyBuffer(out)

LazyBuffer.e = e

# Step 8 - lazybuffer_binary_e
def lazybuffer_binary_e(self, op, other):
    # apply a binary elementwise op between two LazyBuffers, return a new LazyBuffer
    a = self._np
    b = other._np

    out = None

    if op == BinaryOps.ADD:
        out = a + b
    elif op == BinaryOps.SUB:
        out = a - b
    elif op == BinaryOps.MUL:
        out = a * b
    elif op == BinaryOps.DIV:
        out = a / b
    elif op == BinaryOps.CMPLT:
        out = (a < b).astype(a.dtype)
    elif op == BinaryOps.MAX:
        out = np.maximum(a, b)
    else:
        raise ValueError(f"Unknown binary op: {op}")

    return LazyBuffer(out)

# Step 9 - lazybuffer_r
def r(self, op, axis):
    # reduce the underlying array along axis (SUM or MAX), keeping reduced dims as size 1
    x = self._np
    out = None
    if op == ReduceOps.SUM: 
        out = x.sum(axis=axis, keepdims=True)
    elif op == ReduceOps.MAX: 
        out = x.max(axis=axis, keepdims=True)
    else:
        raise ValueError(f"Unknown reduce op: {op}")

    return LazyBuffer(out)

# Step 10 - lazybuffer_reshape
def reshape(self, new_shape):
    # return a new LazyBuffer with the array reshaped to new_shape
    return LazyBuffer(self._np.reshape(new_shape))

# Step 11 - lazybuffer_expand
def expand(self, new_shape):
    # broadcast this buffer's size-1 dims out to new_shape
    new_shape = tuple(int(d) for d in new_shape)
    broadcasted = np.broadcast_to(self._np, new_shape)
    return LazyBuffer(np.array(broadcasted))

# Step 12 - lazybuffer_permute
def permute(self, order):
    # return a new LazyBuffer with axes reordered according to order
    arr = self._np.transpose(order)
    return LazyBuffer(arr)

# Step 13 - Function
class Function:
    def __init__(self, *tensors):
        # record needs_input_grad, requires_grad, and parents for backprop
        flags = [t.requires_grad for t in tensors]
        self.needs_input_grad = flags

        if any(f is True for f in flags): 
            self.requires_grad = True
            self.parents = tensors
        elif None in flags: 
            self.requires_grad = None
        else:
            self.requires_grad = False

# Step 14 - function_forward_backward_stubs
def function_forward_backward_stubs():
    # attach forward and backward stubs to Function that raise NotImplementedError
    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"forward not implemented for {type(self).__name__}")

    def backward(self, *args, **kwargs):
        raise NotImplementedError(f"backward not implemented for {type(self).__name__}")

    Function.forward = forward
    Function.backward = backward 

    return Function

# Step 15 - apply
@classmethod
def apply(cls, *tensors, **kwargs):
    # build the Function, run forward on the input buffers, wrap in a
    # Tensor, and link out._ctx when a gradient is needed.
    ctx = cls(*tensors)
    bufs = [t.lazydata for t in tensors]
    out_buf = ctx.forward(*bufs, **kwargs)
    out = Tensor(out_buf, requires_grad=ctx.requires_grad)
    if ctx.requires_grad:
        out._ctx = ctx
    return out

# Provided: attaches apply onto the Function base class. Leave this as-is.
for _obj in list(globals().values()):
    if isinstance(_obj, type):
        for _k in _obj.__mro__:
            if _k.__name__ == 'Function':
                _k.apply = apply

# Step 16 - Neg
class Neg(Function):
    def forward(self, x):
        return LazyBuffer(-x._np)

    def backward(self, grad_output):
        return LazyBuffer(-grad_output._np)

# Step 17 - Relu
class Relu(Function):
    def forward(self, x):
        # apply the rectified linear unit to lazy buffer x and cache the result
        self.ret = x.e(UnaryOps.RELU)
        return self.ret

    def backward(self, grad_output):
        # route the upstream gradient only through positions that were positive
        zero = LazyBuffer.const(0, self.ret._np.shape)
        mask = lazybuffer_binary_e(zero, BinaryOps.CMPLT, self.ret)
        return lazybuffer_binary_e(mask, BinaryOps.MUL, grad_output)

# Step 18 - Log
class Log(Function):
    def forward(self, x):
        # return the natural log of x and save x for backward
        self.x = x
        return x.e(UnaryOps.LOG)

    def backward(self, grad_output):
        # return the gradient of log with respect to its input
        return lazybuffer_binary_e(grad_output, BinaryOps.DIV, self.x)

# Step 19 - Exp
class Exp(Function):
    def forward(self, x):
        # compute the elementwise exponential and keep what backward needs
        self.ret = x.e(UnaryOps.EXP)
        return self.ret

    def backward(self, grad_output):
        # turn the upstream gradient into the gradient w.r.t. the input
        return lazybuffer_binary_e(self.ret, BinaryOps.MUL, grad_output)

# Step 20 - Sqrt
class Sqrt(Function):
    def forward(self, x):
        # compute the elementwise square root and cache it for backward
        self.ret = x.e(UnaryOps.SQRT)
        return self.ret

    def backward(self, grad_output):
        two = ones = LazyBuffer.const(2, self.ret._np.shape)
        return  lazybuffer_binary_e(grad_output, BinaryOps.DIV, lazybuffer_binary_e(two, BinaryOps.MUL, self.ret))

# Step 21 - Sigmoid
class Sigmoid(Function):
    def forward(self, x):
        self.ret = x.e(UnaryOps.SIGMOID)
        return self.ret

    def backward(self, grad_output):
        # return grad_output times the sigmoid derivative
        ones = LazyBuffer.const(1, self.ret._np.shape)
        return lazybuffer_binary_e(grad_output, BinaryOps.MUL, lazybuffer_binary_e(self.ret, BinaryOps.MUL, lazybuffer_binary_e(ones, BinaryOps.SUB, self.ret)))

# Step 22 - Add
class Add(Function):
    def forward(self, x, y):
        # return the elementwise sum of LazyBuffers x and y
        return lazybuffer_binary_e(x, BinaryOps.ADD, y)

    def backward(self, grad_output):
        # route grad_output to each input that requires a gradient
        grad_x = grad_output if self.needs_input_grad[0] else None
        grad_y = grad_output if self.needs_input_grad[1] else None
        return grad_x, grad_y

# Step 23 - Sub
class Sub(Function):
    def forward(self, x, y):
        # return the elementwise difference x - y as a LazyBuffer
        return lazybuffer_binary_e(x, BinaryOps.SUB, y)

    def backward(self, grad_output):
        # return gradients for x and y (None where grad is not needed)
        grad_x = grad_output if self.needs_input_grad[0] else None
        grad_y = grad_output.e(UnaryOps.NEG) if self.needs_input_grad[1] else None
        return grad_x, grad_y

# Step 24 - Mul
class Mul(Function):
    def forward(self, x, y):
        # compute the elementwise product and save what backward needs
        self.x = x
        self.y = y
        return lazybuffer_binary_e(x, BinaryOps.MUL, y)

    def backward(self, grad_output):
        # return the gradient w.r.t. each input (None if not needed)
        x_grad = lazybuffer_binary_e(grad_output, BinaryOps.MUL, self.y) if self.needs_input_grad[0] else None
        y_grad = lazybuffer_binary_e(grad_output, BinaryOps.MUL, self.x) if self.needs_input_grad[1] else None
        return x_grad, y_grad

# Step 25 - Div
class Div(Function):
    def forward(self, x, y):
        #  divide LazyBuffer x by y and cache inputs for backward
        self.x = x
        self.y = y
        return lazybuffer_binary_e(x, BinaryOps.DIV, y)

    def backward(self, grad_output):
        # return gradients w.r.t. x and y via the quotient rule
        grad_x = None
        grad_y = None
        twos = LazyBuffer.const(2, self.y._np.shape)
        if self.needs_input_grad[0]:
            grad_x = lazybuffer_binary_e(grad_output, BinaryOps.DIV, self.y)
        
        if self.needs_input_grad[1]:
            grad_y = lazybuffer_binary_e(lazybuffer_binary_e(
                grad_output, BinaryOps.MUL, self.x), 
                BinaryOps.DIV, 
                lazybuffer_binary_e(self.y, BinaryOps.MUL, self.y)).e(UnaryOps.NEG)
        return grad_x, grad_y

# Step 26 - sum_function_forward
class Sum(Function):
    def forward(self, x, axis):
        # Reduce x with ReduceOps.SUM over axis (keepdims) and cache shape/axis.
        self.input_shape = x._np.shape 
        self.axis = axis
        return r(x, ReduceOps.SUM, axis)

# Step 27 - sum_function_backward
def backward(self, grad_output):
    # broadcast the summed gradient back to the original input shape
    return expand(grad_output, self.input_shape)

Sum.backward = backward

# Step 28 - max_function_forward
class Max(Function):
    def forward(self, x, axis):
        # reduce x with the MAX reduce op along axis and cache for backward
        self.x = x
        self.axis = axis
        self.ret = r(x, ReduceOps.MAX, axis)
        return self.ret

# Step 29 - max_function_backward
def backward(self, grad_output):
    # route grad_output back to the input elements that were the maximum
    x_shape = self.x._np.shape
    ones = LazyBuffer(np.ones(x_shape, dtype=np.float32))
    max_is_1s = lazybuffer_binary_e(ones, BinaryOps.SUB, lazybuffer_binary_e(self.x, BinaryOps.CMPLT, self.ret))
    counts = r(max_is_1s, ReduceOps.SUM, self.axis)          
    counts_expanded = expand(counts, x_shape)                
    split_mask = lazybuffer_binary_e(max_is_1s, BinaryOps.DIV, counts_expanded)
    grad_x = lazybuffer_binary_e(split_mask, BinaryOps.MUL, expand(grad_output, x_shape))
    return grad_x
    


Max.backward = backward

# Step 30 - Reshape
class Reshape(Function):
    def forward(self, x, shape):
        # cache the input shape and return x reshaped to shape
        self.input_shape = x.shape
        return reshape(x, shape)

    def backward(self, grad_output):
        # reshape the gradient back to the cached input shape
        return reshape(grad_output, self.input_shape)

# Step 31 - expand_function_forward
def expand_function_forward(ctx, x, shape):
    # cache x.shape on ctx, then broadcast x to the target shape
    ctx.input_shape = x.shape
    return expand(x, shape)

# Step 32 - expand_function_backward
def expand_function_backward(ctx, grad_output):
    # Sum grad_output over the broadcast axes back to ctx.input_shape...
    axes = tuple(i for i in range(len(ctx.input_shape))
        if ctx.input_shape[i] == 1 and grad_output.shape[i] != 1)
    return r(grad_output, ReduceOps.SUM, axes)

# Step 33 - permute_function_forward_backward
def permute_function_forward_backward():
    # return (forward, backward); forward reorders axes, backward inverts the order
    def forward(ctx, x, order):
        ctx.order = order
        return permute(x, order)

    def backward(ctx, grad_output):
        inv = argsort(ctx.order)
        return permute(grad_output, inv)

    return forward, backward

# Step 34 - Tensor
class Tensor:
    def __init__(self, data, requires_grad=False, _ctx=None):
        # wrap data in a LazyBuffer and store grad/ctx bookkeeping
        if isinstance(data, LazyBuffer):
            self.lazydata = data
        else:
            self.lazydata = LazyBuffer(np.asarray(data, dtype=np.float32))
        
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = _ctx

    @property
    def data(self):
        # return the underlying LazyBuffer
        return self.lazydata

    @data.setter
    def data(self, value):
        # replace the underlying LazyBuffer
        if isinstance(value, LazyBuffer):
            self.lazydata = value
        else:
            self.lazydata = LazyBuffer(np.asarray(value, dtype=np.float32))

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.lazydata.dtype

    def numpy(self):
        return self.data._np

# Step 35 - tensor_from_data
def tensor_from_data(data, requires_grad=False):
    # wrap a number, list, or numpy array in a LazyBuffer held by a Tensor
    if isinstance(data, LazyBuffer):
        buf = data
    else:
        buf = LazyBuffer(np.asarray(data, dtype=np.float32))
    return Tensor(buf, requires_grad=requires_grad)

# Step 36 - tensor_creation_helpers
def tensor_creation_helpers():
    # return (zeros_fn, ones_fn, full_fn) building constant-filled Tensors
    def zeros_fn(shape):
        arr = np.full(shape, 0, dtype=np.float32)
        return Tensor(LazyBuffer(arr))
    def ones_fn(shape):
        arr = np.full(shape, 1, dtype=np.float32)
        return Tensor(LazyBuffer(arr))
    def full_fn(shape, value):
        arr = np.full(shape, value, dtype=np.float32)
        return Tensor(LazyBuffer(arr))

    return zeros_fn, ones_fn, full_fn

# Step 37 - tensor_randn
def tensor_randn(shape, seed=None, requires_grad=False):
    # Create a Tensor of standard-normal samples for the given shape.
    if not isinstance(shape, tuple):
        shape = tuple(shape) if hasattr(shape, "__iter__") else (shape,)

    rng = np.random.RandomState(seed)

    u = rng.rand(2, *shape)
    u1, u2 = u[0], u[1]
    u1 = np.clip(u1, 1e-12, 1.0)

    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    z = z.astype(np.float32)

    buf = LazyBuffer(z)
    return Tensor(buf, requires_grad=requires_grad)

# Step 38 - build_topological_order
def build_topological_order(tensor):
    # DFS over each node's _ctx.parents, append a node after its parents
    visited = set()
    order = []

    def dfs(node):
        visited.add(id(node))
        if node._ctx is not None:
            for p in node._ctx.parents:
                if id(p) not in visited:
                    dfs(p)
        order.append(node)

    dfs(tensor)
    return order

# Step 39 - tensor_backward
def tensor_backward(tensor):
    # seed root grad with ones, run each backward in reverse topo order
    tensor.grad = Tensor(LazyBuffer(np.ones(tensor.shape, dtype=np.float32)))
    order = build_topological_order(tensor)

    for node in reversed(order):
        if node._ctx is None or node.grad is None:
            continue

        grads = node._ctx.backward(node.grad.data)

        if not isinstance(grads, (tuple, list)):
            grads = (grads,)

        for parent, g in zip(node._ctx.parents, grads):
            if g is None or not parent.requires_grad:
                continue

            if parent.grad is None:
                parent.grad = Tensor(g)
            else:
                parent.grad = Tensor(
                    LazyBuffer(parent.grad.data._np + g._np)
                )

    return None

# Step 40 - bind_unary_tensor_methods
def bind_unary_tensor_methods():
    # map neg/relu/log/exp/sqrt/sigmoid names to callables using function_apply
    def _make(F):
        def method(t):
            return F.apply(t)
        return method

    methods = {
        'neg':     _make(Neg),
        'relu':    _make(Relu),
        'log':     _make(Log),
        'exp':     _make(Exp),
        'sqrt':    _make(Sqrt),
        'sigmoid': _make(Sigmoid),
    }

    for name, fn in methods.items():
        setattr(Tensor, name, fn)

    return methods

# Step 41 - broadcasted
def broadcasted(x, y):
    # align two tensors to one common shape so an elementwise op can run
    # Uses the graph-connected reshape/expand Tensor methods, so gradients still
    # flow back through the broadcast (Expand.backward sums the stretched axes).
    if not isinstance(x, Tensor):
        x = tensor_from_data(x)
    if not isinstance(y, Tensor):
        y = tensor_from_data(y)

    sx, sy = tuple(x.shape), tuple(y.shape)
    if sx == sy:
        return x, y

    # left-pad the shorter shape with 1s so both have the same rank
    ndim = max(len(sx), len(sy))
    px = (1,) * (ndim - len(sx)) + sx
    py = (1,) * (ndim - len(sy)) + sy

    out_shape = tuple(max(a, b) for a, b in zip(px, py))
    for a, b in zip(px, py):
        if a != b and a != 1 and b != 1:
            raise ValueError(f"cannot broadcast shapes {sx} and {sy}")

    if px != sx:
        x = x.reshape(px)
    if px != out_shape:
        x = x.expand(out_shape)

    if py != sy:
        y = y.reshape(py)
    if py != out_shape:
        y = y.expand(out_shape)

    return x, y

# Step 42 - bind_binary_tensor_methods
def bind_binary_tensor_methods():
    # attach broadcasting add/sub/mul/div methods onto the Tensor class
    def add(self, other):
        x, y = broadcasted(self, other)
        return Add.apply(x, y)

    Tensor.add = add
    Tensor.__add__ = add

    def sub(self, other):
        x, y = broadcasted(self, other)
        return Sub.apply(x, y)

    Tensor.sub = sub
    Tensor.__sub__ = sub

    def mul(self, other):
        x, y = broadcasted(self, other)
        return Mul.apply(x, y)

    Tensor.mul = mul
    Tensor.__mul__ = mul

    def div(self, other):
        x, y = broadcasted(self, other)
        return Div.apply(x, y)

    Tensor.div = div
    Tensor.__truediv__ = div

# Step 43 - bind_movement_tensor_methods
def bind_movement_tensor_methods():
    # return reshape/expand/permute methods that call function_apply on movement Functions
    def _normalize(args):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return tuple(args[0])
        return tuple(args)
        
    Expand = type('Expand', (Function,), {
        'forward': expand_function_forward,
        'backward': expand_function_backward,
    })

    permute_forward, permute_backward = permute_function_forward_backward()
    Permute = type('Permute', (Function,), {
        'forward': permute_forward,
        'backward': permute_backward,
    })

    def reshape(self, *args):
        return Reshape.apply(self, shape=_normalize(args))

    def expand(self, *args):
        return Expand.apply(self, shape=_normalize(args))

    def permute(self, *args):
        return Permute.apply(self, order=_normalize(args))

    methods = {'reshape': reshape, 'expand': expand, 'permute': permute}
    for name, fn in methods.items():
        setattr(Tensor, name, fn)

    return methods

# Step 44 - bind_reduce_tensor_methods
def bind_reduce_tensor_methods():
    # attach sum and max reduction methods to the Tensor class

    def _axes(ndim, axis):
        # normalize axis into a tuple of non-negative axis indices
        if axis is None:
            return tuple(range(ndim))
        if isinstance(axis, int):
            return (axis % ndim,)
        return tuple(a % ndim for a in axis)

    def _squeeze(out, in_shape, axes):
        # drop the size-1 dims the reduce left behind, via the graph-connected reshape
        return out.reshape(tuple(d for i, d in enumerate(in_shape) if i not in axes))

    def sum(self, axis=None, keepdim=False):
        in_shape = tuple(self.shape)
        axes = _axes(len(in_shape), axis)
        out = Sum.apply(self, axis=axes)
        return out if keepdim else _squeeze(out, in_shape, axes)

    def max(self, axis=None, keepdim=False):
        in_shape = tuple(self.shape)
        axes = _axes(len(in_shape), axis)
        out = Max.apply(self, axis=axes)
        return out if keepdim else _squeeze(out, in_shape, axes)

    Tensor.sum = sum
    Tensor.max = max

# Step 45 - tensor_mean
def tensor_mean(x, axis=None, keepdim=False):
    # sum x over axis then divide by the number of reduced elements
    in_shape = tuple(x.shape)
    ndim = len(in_shape)
    if axis is None:
        axes = tuple(range(ndim))
    elif isinstance(axis, int):
        axes = (axis % ndim,)
    else:
        axes = tuple(a % ndim for a in axis)

    n = prod([in_shape[a] for a in axes])
    return x.sum(axis=axes, keepdim=keepdim).div(tensor_from_data(float(n)))

# Step 46 - tensor_transpose
def tensor_transpose(x, ax1=-2, ax2=-1):
    # swap axes ax1 and ax2 of tensor x using a permutation
    n = len(x.shape)
    a1 = ax1 % n
    a2 = ax2 % n

    order = list(range(n))
    order[a1], order[a2] = order[a2], order[a1]

    return x.permute(order)

# Step 47 - tensor_matmul_2d
def tensor_matmul_2d(a, b):
    # Compute a 2D matrix product using reshape, expand, mul, and sum.
    if not isinstance(a, Tensor):
        a = tensor_from_data(a)
    if not isinstance(b, Tensor):
        b = tensor_from_data(b)

    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"cannot matmul shapes {a.shape} and {b.shape}")

    # (m,k) -> (m,k,1) -> (m,k,n)   and   (k,n) -> (1,k,n) -> (m,k,n)
    a3 = a.reshape((m, k, 1)).expand((m, k, n))
    b3 = b.reshape((1, k, n)).expand((m, k, n))

    # elementwise product, then contract the shared k axis away
    return a3.mul(b3).sum(axis=1)

# Step 48 - tensor_softmax
def tensor_softmax(x, axis=-1):
    # turn logits into a probability distribution along the given axis
    if not isinstance(x, Tensor):
        x = tensor_from_data(x)

    # subtract the max first so exp() cannot overflow; it cancels out of the result
    m = x.max(axis=axis, keepdim=True)
    e = x.sub(m).exp()
    return e.div(e.sum(axis=axis, keepdim=True))

# Step 49 - tensor_log_softmax
def tensor_log_softmax(x, axis=-1):
    # compute the log of the softmax of x along axis, numerically stable
    if not isinstance(x, Tensor):
        x = tensor_from_data(x)

    # log_softmax(x) = (x - max) - log(sum(exp(x - max)))
    # staying in log space avoids taking log() of an underflowed probability
    shifted = x.sub(x.max(axis=axis, keepdim=True))
    lse = shifted.exp().sum(axis=axis, keepdim=True).log()
    return shifted.sub(lse)

# Step 50 - sparse_categorical_cross_entropy
def sparse_categorical_cross_entropy(logits, labels):
    # mean negative log-probability of the correct class for each sample
    if not isinstance(logits, Tensor):
        logits = tensor_from_data(logits)

    log_probs = tensor_log_softmax(logits, axis=-1)
    n, n_classes = log_probs.shape

    # a constant one-hot mask picks out the correct class per row; multiplying by
    # it (instead of fancy-indexing) keeps the whole thing inside the autograd graph
    labels = np.asarray(labels).astype(int).reshape(-1)
    onehot = np.zeros((n, n_classes), dtype=np.float32)
    onehot[np.arange(n), labels] = 1.0

    picked = log_probs.mul(tensor_from_data(onehot))
    return picked.sum().neg().div(tensor_from_data(float(n)))

# Step 51 - Linear
class Linear:
    # randn weight [in,out] and bias [out]; call computes x @ W + b
    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.RandomState(seed)

        w = rng.randn(in_features, out_features).astype(np.float32)
        b = rng.randn(out_features).astype(np.float32)

        self.weight = Tensor(w, requires_grad=True)
        self.bias = Tensor(b, requires_grad=True)

    def __call__(self, x):
        if not isinstance(x, Tensor):
            x = tensor_from_data(x)
        # x @ W + b; add() broadcasts the (out,) bias across the batch, and
        # Expand.backward sums those copies back down when gradients flow
        return tensor_matmul_2d(x, self.weight).add(self.bias)

    def parameters(self):
        return [self.weight, self.bias]

# Step 52 - MLP
bind_unary_tensor_methods()
bind_binary_tensor_methods()
bind_movement_tensor_methods()  
bind_reduce_tensor_methods()  

class MLP:
    """Two-layer MLP: Linear -> relu -> Linear."""
    def __init__(self, in_features, hidden, out_features, seed=None):
        seed2 = seed + 1 if seed is not None else None
        self.l1 = Linear(in_features, hidden, seed=seed)
        self.l2 = Linear(hidden, out_features, seed=seed2)

    def __call__(self, x):
        if not isinstance(x, Tensor):
            x = tensor_from_data(x)

        h = self.l1(x).relu()
        return self.l2(h)

    def parameters(self):
        # return combined parameter list of both layers
        return self.l1.parameters() + self.l2.parameters()

# Step 53 - sgd_step
def _to_np(x):
    # pull a plain float32 ndarray out of a LazyBuffer, Tensor, or array-like
    if isinstance(x, LazyBuffer):
        return x._np
    if isinstance(x, Tensor):
        return x.lazydata._np
    return np.array(x, dtype=np.float32)

def sgd_step(parameters, learning_rate):
    # p <- p - lr * grad, in place, skipping params without a gradient
    for p in parameters:
        if p.grad is None:
            continue
        updated = _to_np(p.data) - learning_rate * _to_np(p.grad.data)
        p.data = LazyBuffer(updated.astype(np.float32))
    return None

# Step 54 - zero_grad
def zero_grad(parameters):
    # reset each parameter's .grad to None before the next backward pass
    for p in parameters:
        p.grad = None
    return None

# Step 55 - make_toy_digit_dataset
def make_toy_digit_dataset(num_samples, seed=0):
    # build N noisy samples around three flattened 3x3 digit prototypes
    prototypes = np.array([
        [0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
    ], dtype=np.float32)

    rng = np.random.RandomState(seed)
    y = rng.randint(0, prototypes.shape[0], size=num_samples)
    noise = rng.randn(num_samples, prototypes.shape[1]).astype(np.float32) * 0.1

    X = (prototypes[y] + noise).astype(np.float32)
    return X, y.astype(np.int64)

# Step 56 - accuracy
def accuracy(logits, labels):
    #  fraction of rows whose argmax over the class axis equals the label
    arr = logits.data if isinstance(logits, Tensor) else logits
    arr = np.asarray(arr)
    preds = arr.argmax(axis=-1)
    return float((preds == np.asarray(labels).reshape(-1)).mean())

# Step 57 - train_mlp
def train_mlp(X, y, epochs=50, learning_rate=0.1, hidden=16, seed=0):
    # build an MLP for X, y and run gradient descent, returning (model, loss_history)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int).reshape(-1)
    n_samples, n_features = X.shape
    n_classes = int(y.max()) + 1

    model = MLP(in_features=n_features, hidden=hidden, out_features=n_classes, seed=seed)
    params = model.parameters()
    inputs = tensor_from_data(X, requires_grad=False)

    losses = []
    for _ in range(epochs):
        zero_grad(params)                                   # backward accumulates, so clear first

        logits = model(inputs)                              # forward, building the graph
        loss = sparse_categorical_cross_entropy(logits, y)
        losses.append(float(loss.numpy()))

        tensor_backward(loss)                               # reverse-mode autodiff fills p.grad
        sgd_step(params, learning_rate)                     # p <- p - lr * p.grad

    return model, losses

# Step 58 - evaluate_mlp
def evaluate_mlp(model, X_test, y_test):
    # Run the model on X_test and return its classification accuracy
    logits = model(tensor_from_data(X_test, requires_grad=False))
    return accuracy(logits.numpy(), y_test)

