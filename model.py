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
        # TODO: return the elementwise difference x - y as a LazyBuffer
        return lazybuffer_binary_e(x, BinaryOps.SUB, y)

    def backward(self, grad_output):
        # TODO: return gradients for x and y (None where grad is not needed)
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
    expanded = expand(self.ret, x_shape)
    ones = LazyBuffer(np.ones(x_shape))
    max_is_1s = lazybuffer_binary_e(ones, BinaryOps.SUB, lazybuffer_binary_e(self.x, BinaryOps.CMPLT, self.ret))
    counts = r(max_is_1s, ReduceOps.SUM, self.axis)          
    counts_expanded = expand(counts, x_shape)                
    split_mask = lazybuffer_binary_e(max_is_1s, BinaryOps.DIV, counts_expanded)
    grad_x = lazybuffer_binary_e(split_mask, BinaryOps.MUL, expand(grad_output, x_shape))
    return grad_x
    


Max.backward = backward

# Step 30 - Reshape (not yet solved)
# TODO: implement

# Step 31 - expand_function_forward (not yet solved)
# TODO: implement

# Step 32 - expand_function_backward (not yet solved)
# TODO: implement

# Step 33 - permute_function_forward_backward (not yet solved)
# TODO: implement

# Step 34 - Tensor (not yet solved)
# TODO: implement

# Step 35 - tensor_from_data (not yet solved)
# TODO: implement

# Step 36 - tensor_creation_helpers (not yet solved)
# TODO: implement

# Step 37 - tensor_randn (not yet solved)
# TODO: implement

# Step 38 - build_topological_order (not yet solved)
# TODO: implement

# Step 39 - tensor_backward (not yet solved)
# TODO: implement

# Step 40 - bind_unary_tensor_methods (not yet solved)
# TODO: implement

# Step 41 - broadcasted (not yet solved)
# TODO: implement

# Step 42 - bind_binary_tensor_methods (not yet solved)
# TODO: implement

# Step 43 - bind_movement_tensor_methods (not yet solved)
# TODO: implement

# Step 44 - bind_reduce_tensor_methods (not yet solved)
# TODO: implement

# Step 45 - tensor_mean (not yet solved)
# TODO: implement

# Step 46 - tensor_transpose (not yet solved)
# TODO: implement

# Step 47 - tensor_matmul_2d (not yet solved)
# TODO: implement

# Step 48 - tensor_softmax (not yet solved)
# TODO: implement

# Step 49 - tensor_log_softmax (not yet solved)
# TODO: implement

# Step 50 - sparse_categorical_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - Linear (not yet solved)
# TODO: implement

# Step 52 - MLP (not yet solved)
# TODO: implement

# Step 53 - sgd_step (not yet solved)
# TODO: implement

# Step 54 - zero_grad (not yet solved)
# TODO: implement

# Step 55 - make_toy_digit_dataset (not yet solved)
# TODO: implement

# Step 56 - accuracy (not yet solved)
# TODO: implement

# Step 57 - train_mlp (not yet solved)
# TODO: implement

# Step 58 - evaluate_mlp (not yet solved)
# TODO: implement

