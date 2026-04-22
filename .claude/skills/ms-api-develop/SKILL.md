---
name: ms-api
description: Use the MindSpore Python API correctly — Tensor operations, nn.Cell, @ms.jit, grad/autograd, vmap, context settings, mint PyTorch-compatible interface. Use when writing MindSpore training code, defining networks, computing gradients, using functional transforms, or configuring device/mode. Triggers on phrases like "how to use mindspore", "write a model", "compute gradient", "ms.jit", "nn.Cell", "value_and_grad", "mindspore API", "how to train", "Tensor operations", "set context", "mint", "mint.nn", "mint.optim", "PyTorch API in MindSpore".
---

# MindSpore Python API

## Context Setup (Always First)

```python
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

# Choose execution mode
ms.set_context(mode=ms.PYNATIVE_MODE)   # Eager (debug-friendly)
ms.set_context(mode=ms.GRAPH_MODE)      # JIT compiled (faster)

# Set device
ms.set_context(device_target="GPU")     # or "Ascend", "CPU"

# Short-hand for common setups:
ms.set_device("GPU")
```

## Tensor Creation

```python
import numpy as np

# From numpy
x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)

# Zeros/ones
x = ops.zeros((3, 4), ms.float32)
x = ops.ones((3, 4), ms.float32)

# Random
x = ops.randn(3, 4)          # Normal distribution
x = ops.rand(3, 4)           # Uniform [0, 1)

# Like existing tensor
y = ops.zeros_like(x)

# Type casting
x = x.astype(ms.float16)
```

## Tensor Operations (Functional API)

```python
# Math
y = ops.add(x, y)       # or x + y
y = ops.mul(x, y)       # or x * y
y = ops.matmul(x, y)
y = ops.sin(x)
y = ops.exp(x)
y = ops.sqrt(x)

# Shape
y = ops.reshape(x, (2, -1))    # or x.reshape(2, -1)
y = ops.transpose(x, (1, 0))  # or x.transpose(1, 0)
y = x.flatten()
y = ops.squeeze(x, axis=0)
y = ops.unsqueeze(x, axis=0)  # or x.unsqueeze(0)

# Reduction
y = ops.sum(x, axis=1, keepdims=True)
y = ops.mean(x, axis=-1)
y = ops.max(x, axis=0)

# Indexing / slicing
y = x[0, :]              # standard Python indexing
y = ops.gather(x, indices, axis=0)

# Concatenation
y = ops.cat([x, z], axis=0)    # or ops.concat
y = ops.stack([x, z], axis=0)
```

## Defining a Neural Network

```python
class MyNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Dense(128, 64)
        self.linear2 = nn.Dense(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def construct(self, x):
        # construct() = forward pass (replaces __call__ in PyTorch)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

net = MyNet()
output = net(input_tensor)
```

**Key difference from PyTorch:** Use `construct()` not `forward()`.

## JIT Compilation

```python
# Decorator form
@ms.jit
def my_func(x, y):
    return ops.add(x, y)

# With config
@ms.jit(jit_config=ms.JitConfig(jit_level="O2"))
def optimized_func(x):
    return ops.sin(x)

# Compile an entire Cell class
@ms.jit_class
class StatefulComputation:
    def __call__(self, x):
        return x * 2
```

**Restrictions in `@ms.jit`:** No arbitrary Python I/O, print statements are ok for debug, dynamic shapes need special handling.

## Autograd (Gradients)

```python
from mindspore import grad, value_and_grad

# grad: get gradient of output w.r.t. first input
df_dx = grad(ops.sin)(x)

# value_and_grad: get both output and gradient
f_val, f_grad = value_and_grad(ops.sin)(x)

# Gradient w.r.t. multiple inputs
def loss(x, y):
    return ops.mean((x - y) ** 2)

# Get gradient w.r.t. both x and y
dL_dx, dL_dy = grad(loss, grad_position=(0, 1))(x, y)

# Gradient w.r.t. model parameters (trainable weights)
def forward(params, x):
    # params = list of Parameter tensors
    ...

grad_fn = grad(forward, grad_position=0, weights=net.trainable_params())
```

### Training Loop Pattern

```python
loss_fn = nn.MSELoss()
optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)

def forward_fn(data, label):
    pred = net(data)
    loss = loss_fn(pred, label)
    return loss

grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

def train_step(data, label):
    loss, grads = grad_fn(data, label)
    optimizer(grads)
    return loss
```

## Common nn Layers

```python
# Linear
nn.Dense(in_features, out_features, has_bias=True)

# Convolutions
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, pad_mode='valid')
nn.Conv1d(...)

# Normalization
nn.BatchNorm2d(num_features)
nn.LayerNorm(normalized_shape)

# Activations
nn.ReLU(), nn.GELU(), nn.Sigmoid(), nn.Tanh(), nn.SiLU()

# Pooling
nn.MaxPool2d(kernel_size, stride)
nn.AvgPool2d(kernel_size, stride)

# Dropout
nn.Dropout(p=0.5)

# Embedding
nn.Embedding(vocab_size, embedding_dim)

# Sequential container
model = nn.SequentialCell([nn.Dense(64, 32), nn.ReLU(), nn.Dense(32, 10)])
```

## Dtype Reference

```python
ms.float16    # half precision
ms.float32    # single precision (default)
ms.float64    # double precision
ms.bfloat16   # bfloat16
ms.int8, ms.int16, ms.int32, ms.int64
ms.uint8, ms.uint16, ms.uint32, ms.uint64
ms.bool_
```

## Parameters and Checkpoints

```python
# Mark a tensor as trainable
param = ms.Parameter(Tensor(np.zeros((10,)), ms.float32), name="weight")

# Access model parameters
for name, param in net.parameters_and_names():
    print(name, param.shape)

# Save / load checkpoint
ms.save_checkpoint(net, "model.ckpt")
param_dict = ms.load_checkpoint("model.ckpt")
ms.load_param_into_net(net, param_dict)
```

---

## mint Module (PyTorch-Compatible API)

`mindspore.mint` is a PyTorch-compatible API layer designed to make migration from PyTorch easier. It exposes PyTorch-like function signatures and submodules.

```python
import mindspore.mint as mint
import mindspore.mint.nn as mint_nn
import mindspore.mint.nn.functional as F
import mindspore.mint.optim as mint_optim
```

### mint vs ops — When to Use Which

| Scenario | Recommendation |
|----------|---------------|
| Migrating from PyTorch | Use `mint` — signatures match PyTorch |
| New MindSpore code | Either works; `mint` preferred for PyTorch familiarity |
| `@ms.jit` / graph mode | Both supported; `mint` functions are pyboost-enabled |
| Key argument difference | `mint` uses `dim`/`keepdim` (PyTorch style); `ops` uses `axis`/`keepdims` |

### mint Functions

```python
import mindspore.mint as mint

# Math (PyTorch-compatible signatures)
mint.sin(x)
mint.cos(x)
mint.exp(x)
mint.log(x)
mint.sqrt(x)
mint.rsqrt(x)
mint.abs(x)
mint.pow(x, 2.0)
mint.square(x)
mint.reciprocal(x)
mint.neg(x)           # or mint.negative(x)

# Comparison
mint.eq(x, y)
mint.ne(x, y)
mint.gt(x, y)         # or mint.greater(x, y)
mint.ge(x, y)         # or mint.greater_equal(x, y)
mint.lt(x, y)         # or mint.less(x, y)
mint.le(x, y)         # or mint.less_equal(x, y)

# Logical
mint.logical_and(x, y)
mint.logical_or(x, y)
mint.logical_not(x)
mint.logical_xor(x, y)

# Reduction — note: uses dim/keepdim (PyTorch style)
mint.sum(x, dim=1, keepdim=True)
mint.mean(x, dim=-1)
mint.max(x, dim=0)
mint.min(x, dim=0)
mint.any(x)
mint.argmax(x, dim=1)
mint.argmin(x, dim=1)
mint.var(x, dim=0)
mint.norm(x, dim=-1)

# Shape
mint.transpose(x, 0, 1)          # Note: two ints, not a tuple
mint.unsqueeze(x, dim=0)
mint.flatten(x, start_dim=1)
mint.chunk(x, chunks=4, dim=0)
mint.split(x, split_size_or_sections=2, dim=0)
mint.narrow(x, dim=0, start=0, length=2)
mint.broadcast_to(x, (3, 4))
mint.stack([x, y], dim=0)
mint.sort(x, dim=-1, descending=False)
mint.topk(x, k=3, dim=-1)

# Indexing
mint.gather(x, dim=0, index=idx)
mint.index_select(x, dim=0, index=idx)
mint.masked_select(x, mask)
mint.nonzero(x)
mint.where(condition, x, y)

# Linear algebra
mint.matmul(x, y)
mint.bmm(x, y)
mint.mm(x, y)
mint.mv(mat, vec)
mint.outer(x, y)
mint.einsum("ij,jk->ik", x, y)
mint.norm(x, p=2, dim=-1)

# Trig / special
mint.sin(x); mint.cos(x); mint.tan(x)
mint.asin(x); mint.acos(x); mint.atan(x); mint.atan2(x, y)
mint.sinh(x); mint.cosh(x); mint.tanh(x)
mint.erf(x); mint.erfc(x); mint.erfinv(x)

# Random
mint.randn(*size)
mint.rand(*size)
mint.randint(low, high, size)
mint.randn_like(x)
mint.rand_like(x)
mint.normal(mean, std, size)
mint.multinomial(input, num_samples)
mint.bernoulli(x)
mint.randperm(n)

# Creation
mint.zeros(*size)       # via ops.zeros (mint re-exports many ops)
mint.ones(*size)
mint.full(size, fill_value)
mint.eye(n)
mint.arange(start, end, step)
mint.linspace(start, end, steps)
mint.empty(*size)

# Type / shape utilities
mint.clone(x)
mint.isinf(x)
mint.isneginf(x)
mint.isclose(x, y, rtol=1e-5, atol=1e-8)

# Bitwise
mint.bitwise_and(x, y)
mint.bitwise_or(x, y)
mint.bitwise_xor(x, y)
```

### mint.nn — Layers

PyTorch-compatible layer API. Key layers available:

```python
import mindspore.mint.nn as mint_nn

# Core layers
mint_nn.Linear(in_features, out_features, bias=True)
mint_nn.Embedding(num_embeddings, embedding_dim)
mint_nn.LayerNorm(normalized_shape, eps=1e-5)
mint_nn.GroupNorm(num_groups, num_channels)
mint_nn.BatchNorm1d(num_features)
mint_nn.BatchNorm2d(num_features)
mint_nn.BatchNorm3d(num_features)
mint_nn.SyncBatchNorm(num_features)

# Convolutions
mint_nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0)
mint_nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
mint_nn.Conv3d(in_channels, out_channels, kernel_size)
mint_nn.ConvTranspose2d(in_channels, out_channels, kernel_size)

# Pooling
mint_nn.MaxPool2d(kernel_size, stride=None)
mint_nn.AvgPool1d(kernel_size, stride=None)
mint_nn.AvgPool2d(kernel_size, stride=None)
mint_nn.AvgPool3d(kernel_size, stride=None)
mint_nn.AdaptiveAvgPool1d(output_size)
mint_nn.AdaptiveAvgPool2d(output_size)
mint_nn.AdaptiveAvgPool3d(output_size)
mint_nn.AdaptiveMaxPool1d(output_size)
mint_nn.MaxUnpool2d(kernel_size, stride=None)

# Activations
mint_nn.ReLU()
mint_nn.SiLU()
mint_nn.Sigmoid()
mint_nn.Softmax(dim=-1)
mint_nn.LogSoftmax(dim=-1)
mint_nn.LogSigmoid()
mint_nn.Tanh()
mint_nn.ELU(alpha=1.0)
mint_nn.PReLU()
mint_nn.GLU(dim=-1)
mint_nn.Threshold(threshold, value)
mint_nn.Hardsigmoid()
mint_nn.Hardswish()
mint_nn.Hardshrink(lambd=0.5)

# Regularization
mint_nn.Dropout(p=0.5)
mint_nn.Dropout2d(p=0.5)

# Shape
mint_nn.Flatten(start_dim=1, end_dim=-1)
mint_nn.Upsample(size=None, scale_factor=None, mode='nearest')
mint_nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)
mint_nn.Fold(output_size, kernel_size)

# Padding
mint_nn.ZeroPad2d(padding)
mint_nn.ConstantPad1d(padding, value)
mint_nn.ConstantPad2d(padding, value)
mint_nn.ReflectionPad2d(padding)
mint_nn.ReplicationPad2d(padding)

# Loss
mint_nn.L1Loss(reduction='mean')
```

### mint.nn.functional — Functional API

```python
import mindspore.mint.nn.functional as F

# Activations
F.relu(x)
F.sigmoid(x)
F.softmax(x, dim=-1)
F.log_softmax(x, dim=-1)
F.gelu(x)
F.silu(x)
F.tanh(x)
F.leaky_relu(x, negative_slope=0.01)
F.elu(x, alpha=1.0)
F.prelu(x, weight)
F.hardsigmoid(x)
F.hardswish(x)
F.softplus(x, beta=1)
F.mish(x)

# Loss functions
F.mse_loss(input, target, reduction='mean')
F.l1_loss(input, target, reduction='mean')
F.cross_entropy(input, target, weight=None, reduction='mean')
F.kl_div(input, target, reduction='mean')

# Normalization
F.layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5)
F.batch_norm(x, running_mean, running_var, weight=None, bias=None, training=False)
F.group_norm(x, num_groups, weight=None, bias=None, eps=1e-5)

# Linear
F.linear(x, weight, bias=None)        # equivalent to nn.Dense

# Convolution
F.conv1d(x, weight, bias=None, stride=1, padding=0)
F.conv2d(x, weight, bias=None, stride=1, padding=0)
F.conv3d(x, weight, bias=None, stride=1, padding=0)

# Pooling
F.max_pool2d(x, kernel_size, stride=None)
F.avg_pool2d(x, kernel_size, stride=None)
F.adaptive_avg_pool2d(x, output_size)
F.interpolate(x, size=None, scale_factor=None, mode='nearest')

# Other
F.embedding(x, weight, padding_idx=None)
F.dropout(x, p=0.5, training=True)
F.dropout2d(x, p=0.5, training=True)
F.pad(x, pad, mode='constant', value=0)
F.unfold(x, kernel_size, dilation=1, padding=0, stride=1)
F.pixel_shuffle(x, upscale_factor)
F.sigmoid_(x)        # in-place
F.relu_(x)           # in-place
```

### mint.optim — Optimizers

```python
import mindspore.mint.optim as mint_optim

# AdamW (recommended default)
optimizer = mint_optim.AdamW(
    net.trainable_params(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# Adam
optimizer = mint_optim.Adam(
    net.trainable_params(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8
)

# SGD
optimizer = mint_optim.SGD(
    net.trainable_params(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)

# FusedAdamW (Ascend-optimized, faster)
optimizer = mint_optim.FusedAdamW(
    net.trainable_params(),
    lr=1e-3,
    weight_decay=0.01
)
```

### mint.linalg — Linear Algebra

```python
import mindspore.mint.linalg as linalg

linalg.norm(x, ord=None, dim=None)    # matrix/vector norm
linalg.det(x)                          # determinant
linalg.inv(x)                          # matrix inverse
linalg.solve(A, b)                     # solve Ax=b
linalg.eig(x)                          # eigenvalues/eigenvectors
linalg.svd(x)                          # singular value decomposition
linalg.qr(x)                           # QR decomposition
linalg.cholesky(x)                     # Cholesky decomposition
```

### mint.special — Special Math Functions

```python
import mindspore.mint.special as special

special.erf(x)
special.erfc(x)
special.erfinv(x)
special.expit(x)       # sigmoid
special.log1p(x)
special.xlogy(x, y)
```

### mint.distributed — Distributed Operations

```python
import mindspore.mint.distributed as dist

dist.all_reduce(tensor, op='sum')
dist.all_gather(tensor_list, tensor)
dist.broadcast(tensor, src=0)
dist.reduce_scatter(output, input_list)
```

### Training with mint — Full Example

```python
import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.nn as mint_nn
import mindspore.mint.nn.functional as F
import mindspore.mint.optim as mint_optim

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")

class MLP(mint_nn.Linear.__bases__[0]):  # nn.Cell base
    def __init__(self):
        super().__init__()
        self.fc1 = mint_nn.Linear(128, 64)
        self.fc2 = mint_nn.Linear(64, 10)
        self.dropout = mint_nn.Dropout(p=0.1)

    def construct(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

net = MLP()
optimizer = mint_optim.AdamW(net.trainable_params(), lr=1e-3)

def forward_fn(x, y):
    logits = net(x)
    loss = F.cross_entropy(logits, y)
    return loss

grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

def train_step(x, y):
    loss, grads = grad_fn(x, y)
    optimizer(grads)
    return loss
```

### Key API Differences: mint vs ops

| Operation | `mint` (PyTorch style) | `ops` (MindSpore style) |
|-----------|----------------------|------------------------|
| Reduce axis | `dim=1` | `axis=1` |
| Keep dims | `keepdim=True` | `keepdims=True` |
| Transpose | `mint.transpose(x, 0, 1)` | `ops.transpose(x, (0, 1))` |
| Linear layer | `mint_nn.Linear` | `nn.Dense` |
| Optimizer | `mint_optim.AdamW` | `nn.AdamWeightDecay` |
| Functional | `F.relu(x)` | `ops.relu(x)` |
