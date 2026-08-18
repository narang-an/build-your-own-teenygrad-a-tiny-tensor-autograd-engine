# A Tiny Tensor Autograd Engine

A miniature deep learning framework built from scratch on top of numpy. It has three core classes —
`LazyBuffer`, `Function`, and `Tensor` — and by stacking them you get automatic differentiation,
which is enough to train a small neural network. Same design as real libraries like
[tinygrad](https://github.com/tinygrad/tinygrad), just small enough to read in one sitting.

## How to run

```bash
python3 scaffold.py
```

```
initial loss:     7.377138137817383
initial accuracy: 0.32
loss[0], loss[-1]: 7.3771 0.018
test accuracy:    1.0
```

`model.py` holds the implementation (58 numbered steps, in dependency order). `scaffold.py` runs the
end-to-end demo: make data → train an MLP → measure accuracy.

---

## The big idea

The three classes split one job three ways. That separation is the whole design:

| Class | Job | Analogy |
| --- | --- | --- |
| `LazyBuffer` | **Compute.** Holds numbers, does math, remembers nothing. | a calculator |
| `Function` | **Differentiate.** Knows one op's derivative, and who its inputs were. | a receipt |
| `Tensor` | **Remember.** What you actually use. Holds a buffer + its history. | a notebook |

> `LazyBuffer` knows how to compute, `Function` knows how to differentiate, `Tensor` knows how to
> remember.

Why bother? Because to train a network you need derivatives, and derivatives require knowing *how a
number was produced*. Numpy throws that away — `c = a * b` gives you `c` and nothing else. So we
keep the arithmetic in `LazyBuffer`, and wrap it in a layer that records history.

---

## 1. `LazyBuffer` — the numbers

The bottom layer. It wraps exactly one numpy array:

```python
class LazyBuffer:
    def __init__(self, np_array):
        self._np   = np.asarray(np_array)
        self.shape = tuple(int(d) for d in self._np.shape)
        self.dtype = self._np.dtype
```

You never do math on it directly. Instead you call it with an **op**, one of four kinds:

```python
UnaryOps    = NEG, RELU, LOG, EXP, SQRT, SIGMOID     # same shape in and out
BinaryOps   = ADD, SUB, MUL, DIV, CMPLT, MAX         # two inputs -> one output
ReduceOps   = SUM, MAX                               # collapse an axis
MovementOps = RESHAPE, EXPAND, PERMUTE               # rearrange, no math
```

That's the framework's *entire* vocabulary — about 17 primitives. Everything else (matmul, softmax,
cross-entropy) gets built out of them. Keeping the list this short is the point: swap these few
numpy calls for GPU kernels and everything above keeps working untouched.

One rule matters: **buffers are immutable.** Every op returns a *new* buffer. That's what makes it
safe for a `Function` to stash an input and use it much later during the backward pass.

<details>
<summary><b>Why is it called "lazy" if it isn't?</b></summary>

In real tinygrad, a `LazyBuffer` doesn't compute right away — it records the op and its inputs, then
defers until you ask for the result, so a scheduler can fuse many ops into one GPU kernel.

Here it's **eager**: `e()`, `r()` and friends call numpy immediately. What's kept is the *interface* —
you only touch buffers through enum-tagged ops, never by grabbing `._np`. That boundary is exactly
what you'd need to drop in a real lazy backend later, which is why the name stuck.
</details>

---

## 2. `Function` — the derivative, and the link

A `Function` is one operation in the graph. Every subclass writes two halves:

```python
class Exp(Function):
    def forward(self, x):
        self.ret = x.e(UnaryOps.EXP)      # compute, and save what backward will need
        return self.ret
    def backward(self, grad_output):      # given d(loss)/d(output), return d(loss)/d(input)
        return lazybuffer_binary_e(self.ret, BinaryOps.MUL, grad_output)
```

`forward` computes. `backward` takes the gradient flowing in from downstream and converts it into a
gradient for the inputs. Each op saves only what its own formula needs — `Exp` saves its output
(since `d/dx eˣ = eˣ`), `Log` saves its input (since `d/dx ln x = 1/x`), `Add` saves nothing.

The other half of `Function` is the bookkeeping, and this is the part that builds the graph:

```python
class Function:
    def __init__(self, *tensors):
        if any(t.requires_grad for t in tensors):
            self.requires_grad = True
            self.parents = tensors      # <-- remember who my inputs were
```

`self.parents` **is** the graph. There's no graph object anywhere in the codebase — just each
`Function` holding references to the tensors that fed it.

### `apply` ties it together

Every single operation goes through this one classmethod:

```python
@classmethod
def apply(cls, *tensors, **kwargs):
    ctx     = cls(*tensors)                    # 1. make the node, record its parents
    bufs    = [t.lazydata for t in tensors]    # 2. unwrap Tensors -> LazyBuffers
    out_buf = ctx.forward(*bufs, **kwargs)     # 3. do the actual math
    out     = Tensor(out_buf, requires_grad=ctx.requires_grad)
    if ctx.requires_grad:
        out._ctx = ctx                         # 4. attach the node to the result
    return out
```

Line 4 is the hinge of the whole engine. The output tensor points back at the `Function` that made
it, which points back at its input tensors, which point at *their* functions. Chain a few operations
and a graph has assembled itself backwards — for free, as a side effect of computing forwards.

Note the `if`: when nothing needs a gradient, `_ctx` stays `None` and the node is thrown away. That's
inference mode, and it's why evaluation doesn't pile up memory.

---

## 3. `Tensor` — what you actually use

A thin wrapper with four fields:

```python
class Tensor:
    def __init__(self, data, requires_grad=False, _ctx=None):
        self.lazydata      = data     # the numbers  (a LazyBuffer)
        self.requires_grad = ...      # should gradients flow to me?
        self.grad          = None     # filled in by the backward pass
        self._ctx          = _ctx     # the Function that made me (None if I'm a leaf)
```

`_ctx is None` defines a **leaf** — weights, input data, constants. Anything not computed from
something else. Leaves are where the backward pass stops and gradients land.

Its methods are generated rather than hand-written: `bind_unary_tensor_methods()` and friends attach
`relu`, `exp`, `add`, `sum`, `reshape`, … onto the class at import time, each one just calling
`SomeFunction.apply`. The binary ones also wire up `__add__` / `__mul__` / etc., so plain Python
operators build a graph.

---

## How the three fit together

```python
x = tensor_from_data([1.0, -2.0, 3.0], requires_grad=True)
y = x.relu().exp()
tensor_backward(y)
x.grad.numpy()        # [2.718, 0.0, 20.086]
```

What happened on each line:

```
x.relu()        Relu.apply(x)  ->  ctx = Relu(x)          Function records parents=(x,)
                                   ctx.forward(x buffer)  LazyBuffer does max(v, 0)
                                   Tensor(out, _ctx=ctx)  Tensor remembers the link
                     |
                     v
     x  <--parents--  Relu  <--_ctx--  h  <--parents--  Exp  <--_ctx--  y
```

Then `tensor_backward(y)` walks that chain in reverse. `Exp.backward` hands a gradient to `h`,
`Relu.backward` hands one to `x`, and it lands in `x.grad`. The `0.0` in the middle is `Relu.backward`
zeroing the gradient where the input was negative.

### The backward pass

```python
tensor.grad = Tensor(LazyBuffer(np.ones(tensor.shape)))   # seed: d(loss)/d(loss) = 1
for node in reversed(build_topological_order(tensor)):    # visit consumers before producers
    if node._ctx is None: continue                        # a leaf; nothing to undo
    grads = node._ctx.backward(node.grad.data)            # this node's local derivative
    for parent, g in zip(node._ctx.parents, grads):
        parent.grad = g if parent.grad is None else parent.grad + g    # ACCUMULATE
```

Three things make this work:

1. **Reverse topological order.** A node's gradient isn't finished until everything downstream of it
   has contributed. Going in reverse order guarantees that.
2. **Gradients add, they don't overwrite.** If a tensor is used twice (like `x` in `x*x + x`), it gets
   two separate contributions, and the chain rule says they sum.
3. **Every op only knows its own derivative.** Correct gradients for a whole network emerge purely
   from the order of the walk. That's reverse-mode autodiff — about twenty lines.

---

## From ops to a neural network

Everything above is primitives. Everything here is *composed* from them — and because each piece is
built from graph-connected tensor methods, **none of them needs a hand-written backward.** The engine
differentiates them automatically.

A matrix multiply, for instance, isn't a primitive at all — it's four ops you already have:

```python
a3 = a.reshape((m, k, 1)).expand((m, k, n))     # stretch (m,k) across the output columns
b3 = b.reshape((1, k, n)).expand((m, k, n))     # stretch (k,n) across the batch rows
return a3.mul(b3).sum(axis=1)                   # multiply, then contract the shared axis
```

Softmax is `MAX → SUB → EXP → SUM → DIV`. Cross-entropy is log-softmax times a one-hot mask, summed.
And a layer is one line:

```python
class Linear:
    def __call__(self, x):
        return tensor_matmul_2d(x, self.weight).add(self.bias)

class MLP:
    def __call__(self, x):
        return self.l2(self.l1(x).relu())        # Linear -> relu -> Linear
```

## The training loop

Which makes the actual training almost anticlimactic:

```python
for _ in range(epochs):
    zero_grad(params)                                   # clear (backward accumulates!)
    logits = model(inputs)                              # forward — builds the graph
    loss   = sparse_categorical_cross_entropy(logits, y)
    tensor_backward(loss)                               # backward — fills every p.grad
    sgd_step(params, learning_rate)                     # p <- p - lr * p.grad
```

No derivative is written by hand anywhere. The `(probs - onehot) / N` formula you'd normally derive
on paper for softmax cross-entropy never appears — it *falls out* of the graph walk. That's the
payoff for building the engine.

The data (`make_toy_digit_dataset`) is three 3×3 pixel patterns — a diamond, an "I", an "H" —
flattened to 9 features with a little Gaussian noise:

```
   class 0        class 1        class 2
    . ##  .     ## ## ##     ##  . ##
   ##  . ##      . ##  .     ## ## ##
    . ##  .     ## ## ##     ##  . ##
```

Deliberately easy. The point is to confirm the engine learns at all, not to benchmark anything —
loss drops 7.38 → 0.018 and test accuracy hits 1.0.

---

## Where to start reading

`model.py` runs top to bottom in dependency order. The four steps that carry the design:

- **Step 15 (`apply`)** — how the graph gets built going forward
- **Step 39 (`tensor_backward`)** — how it gets consumed going backward
- **Step 47 (`tensor_matmul_2d`)** — how a "primitive" is really four simpler ops
- **Step 29 (`Max.backward`)** — the trickiest derivative in the file

<details>
<summary><b>All 58 steps</b></summary>

| | | | |
| --- | --- | --- | --- |
| 1. prod | 16. Neg | 31. expand_function_forward | 46. tensor_transpose |
| 2. argsort | 17. Relu | 32. expand_function_backward | 47. tensor_matmul_2d |
| 3. make_op_enums | 18. Log | 33. permute_function_fwd_bwd | 48. tensor_softmax |
| 4. LazyBuffer | 19. Exp | 34. Tensor | 49. tensor_log_softmax |
| 5. lazybuffer_const | 20. Sqrt | 35. tensor_from_data | 50. sparse_categorical_cross_entropy |
| 6. rand | 21. Sigmoid | 36. tensor_creation_helpers | 51. Linear |
| 7. lazybuffer_unary_e | 22. Add | 37. tensor_randn | 52. MLP |
| 8. lazybuffer_binary_e | 23. Sub | 38. build_topological_order | 53. sgd_step |
| 9. lazybuffer_r | 24. Mul | 39. tensor_backward | 54. zero_grad |
| 10. lazybuffer_reshape | 25. Div | 40. bind_unary_tensor_methods | 55. make_toy_digit_dataset |
| 11. lazybuffer_expand | 26. sum_function_forward | 41. broadcasted | 56. accuracy |
| 12. lazybuffer_permute | 27. sum_function_backward | 42. bind_binary_tensor_methods | 57. train_mlp |
| 13. Function | 28. max_function_forward | 43. bind_movement_tensor_methods | 58. evaluate_mlp |
| 14. function_fwd_bwd_stubs | 29. max_function_backward | 44. bind_reduce_tensor_methods | |
| 15. apply | 30. Reshape | 45. tensor_mean | |

</details>

---

Built on Deep-ML.
