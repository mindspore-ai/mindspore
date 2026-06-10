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
"""Tests for mint.cdist API coverage on CPU and GPU."""

import numpy as np
import pytest
import torch

import mindspore as ms
from mindspore import Tensor, mint
from mindspore.ops.functional import vmap

from tests.mark_utils import arg_mark
from tests.st.utils import test_utils


@test_utils.run_with_cell
def mint_cdist_forward(x1, x2, p):
    return mint.cdist(x1, x2, p)


@test_utils.run_with_cell
def mint_cdist_forward_vmap(x1, x2):
    return mint.cdist(x1, x2)


@test_utils.run_with_cell
def mint_cdist_backward(x1, x2, p):
    grad_fn = ms.grad(mint_cdist_forward, grad_position=(0, 1))
    return grad_fn(x1, x2, p)


def set_context_mode(mode, device_target):
    ms.set_context(mode=mode, device_target=device_target)


def get_torch_forward_and_grads(x1_np, x2_np, p):
    x1 = torch.tensor(x1_np, dtype=torch.float32, requires_grad=True)
    x2 = torch.tensor(x2_np, dtype=torch.float32, requires_grad=True)
    out = torch.cdist(x1, x2, p=p)
    out.sum().backward()
    return out.detach().numpy(), x1.grad.detach().numpy(), x2.grad.detach().numpy()


def run_mint_cdist_api_forward_backward(device_target, mode):
    set_context_mode(mode, device_target)
    x1_np = np.array(
        [[[1.0, 1.0], [2.0, 2.0]],
         [[0.0, 1.0], [1.0, 0.0]]],
        dtype=np.float32,
    )
    x2_np = np.array(
        [[[3.0, 3.0], [3.0, 3.0]],
         [[2.0, 1.0], [1.0, 2.0]]],
        dtype=np.float32,
    )
    expect_out, expect_dx, expect_dy = get_torch_forward_and_grads(x1_np, x2_np, 2.0)
    output = mint_cdist_forward(Tensor(x1_np), Tensor(x2_np), 2.0)
    grads = mint_cdist_backward(Tensor(x1_np), Tensor(x2_np), 2.0)
    np.testing.assert_allclose(output.asnumpy(), expect_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(grads[0].asnumpy(), expect_dx, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(grads[1].asnumpy(), expect_dy, rtol=1e-4, atol=1e-4)


def run_mint_cdist_api_vmap(device_target, mode):
    set_context_mode(mode, device_target)
    x1_np = np.array(
        [[[[1.0, 1.0], [2.0, 2.0]],
          [[1.0, 1.0], [2.0, 2.0]]],
         [[[0.0, 1.0], [1.0, 0.0]],
          [[0.0, 1.0], [1.0, 0.0]]]],
        dtype=np.float32,
    )
    x2_np = np.array(
        [[[[3.0, 3.0], [3.0, 3.0]],
          [[3.0, 3.0], [3.0, 3.0]]],
         [[[2.0, 1.0], [1.0, 2.0]],
          [[2.0, 1.0], [1.0, 2.0]]]],
        dtype=np.float32,
    )
    expect = np.stack([torch.cdist(torch.tensor(x1_np[i]), torch.tensor(x2_np[i])).numpy() for i in range(2)])
    output = vmap(mint_cdist_forward_vmap, in_axes=(0, 0), out_axes=0)(Tensor(x1_np), Tensor(x2_np))
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level1',
    card_mark='onecard'
)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_cdist_api_forward_backward_cpu(mode):
    """
    Feature: mint.cdist API.
    Description: Verify forward and backward on CPU API path.
    Expectation: MindSpore matches torch.
    """
    run_mint_cdist_api_forward_backward('CPU', mode)


@arg_mark(
    plat_marks=['platform_gpu'],
    level_mark='level1',
    card_mark='onecard'
)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_cdist_api_forward_backward_gpu(mode):
    """
    Feature: mint.cdist API.
    Description: Verify forward and backward on GPU API path.
    Expectation: MindSpore matches torch.
    """
    run_mint_cdist_api_forward_backward('GPU', mode)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level1',
    card_mark='onecard'
)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_cdist_api_vmap_cpu(mode):
    """
    Feature: mint.cdist API.
    Description: Verify vmap on CPU API path.
    Expectation: MindSpore matches expected output.
    """
    run_mint_cdist_api_vmap('CPU', mode)


@arg_mark(
    plat_marks=['platform_gpu'],
    level_mark='level1',
    card_mark='onecard'
)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_cdist_api_vmap_gpu(mode):
    """
    Feature: mint.cdist API.
    Description: Verify vmap on GPU API path.
    Expectation: MindSpore matches expected output.
    """
    run_mint_cdist_api_vmap('GPU', mode)
