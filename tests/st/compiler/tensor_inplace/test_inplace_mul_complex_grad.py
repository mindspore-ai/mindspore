# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Regression test for `tensor.mul_` backward with complex dtypes."""

import numpy as np
import pytest
import torch
import mindspore as ms
from mindspore import Tensor, context, ops
from tests.mark_utils import arg_mark

# Non-zero imaginary parts, so that a missing conjugation shows up as a wrong sign.
X = np.array([[1 + 2j, 3 - 4j], [-5 + 0.5j, -1j]])
Y = np.array([[5 + 6j, 7 + 8j], [1 - 2j, -3 - 4j]])


def _inplace_mul_loss(x, y):
    """Multiply a non-leaf tensor by `y` in place, then reduce to a real scalar."""
    out = x * 1
    return ops.real(out.mul_(y)).sum()


@ms.jit(backend="ms_backend")
def _inplace_mul_graph_grads(x, y):
    """Differentiate the in-place multiply using the MindSpore graph backend."""
    return ms.grad(_inplace_mul_loss, grad_position=(0, 1))(x, y)


def _torch_inplace_mul_grads(x_np, y_np):
    """PyTorch CPU gradients for the same in-place multiply."""
    torch_x = torch.tensor(x_np, requires_grad=True)
    torch_y = torch.tensor(y_np, requires_grad=True)
    out = torch_x * 1
    out.mul_(torch_y)
    out.real.sum().backward()
    return torch_x.grad.detach().numpy(), torch_y.grad.detach().numpy()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', mem_peak=4.00)
@pytest.mark.parametrize("mode_name", ["pynative", "graph"])
@pytest.mark.parametrize("dtype_name", ["complex64", "complex128"])
def test_inplace_mul_backward_matches_torch_for_complex(mode_name, dtype_name):
    """
    Feature: `tensor.mul_` backward for complex dtypes on Ascend.
    Description: Differentiate `out = x * 1; out.mul_(y)` for complex64 and complex128,
                 in PyNative and in Graph mode with ms_backend.
    Expectation: Gradients match the PyTorch CPU reference, namely dx = dout * conj(y)
                 and dy = dout * conj(x); graph complex64 must run without a TensorMove
                 kernel launch error.
    """
    context.set_context(mode=ms.PYNATIVE_MODE if mode_name == "pynative" else ms.GRAPH_MODE,
                        device_target="Ascend")
    np_dtype = np.complex64 if dtype_name == "complex64" else np.complex128
    x_np, y_np = X.astype(np_dtype), Y.astype(np_dtype)
    dx_expected, dy_expected = _torch_inplace_mul_grads(x_np, y_np)

    grad_fn = _inplace_mul_graph_grads if mode_name == "graph" else ms.grad(_inplace_mul_loss, grad_position=(0, 1))
    dx, dy = grad_fn(Tensor(x_np), Tensor(y_np))

    assert np.allclose(dx.asnumpy(), dx_expected), f"dx {dx.asnumpy()} != expected {dx_expected}"
    assert np.allclose(dy.asnumpy(), dy_expected), f"dy {dy.asnumpy()} != expected {dy_expected}"
