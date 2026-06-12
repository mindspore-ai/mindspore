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
"""Tests for mint.cdist dynamic forward and backward coverage."""

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, mint, nn

from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.grad.test_grad_of_dynamic import TestDynamicGrad
from tests.st.utils import test_utils


@test_utils.run_with_cell
def mint_cdist_forward_dynamic(x1, x2, p):
    return mint.cdist(x1, x2, p)


class MintCdistForwardNet(nn.Cell):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def construct(self, x1, x2):
        return mint.cdist(x1, x2, self.p)


class MintCdistNet(nn.Cell):
    def construct(self, x1, x2):
        return mint.cdist(x1, x2, 2.0)


def set_pynative_context(device_target):
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=device_target)


def run_dynamic_forward(device_target):
    set_pynative_context(device_target)
    dynamic_net = MintCdistForwardNet(1.0)
    dynamic_net.set_inputs(
        Tensor(shape=(None, None, None), dtype=ms.float32),
        Tensor(shape=(None, None, None), dtype=ms.float32),
    )
    x1_a = Tensor(np.random.randn(3, 7, 9).astype(np.float32))
    x2_a = Tensor(np.random.randn(3, 8, 9).astype(np.float32))
    x1_b = Tensor(np.random.randn(4, 8, 5).astype(np.float32))
    x2_b = Tensor(np.random.randn(4, 9, 5).astype(np.float32))
    static_out_a = mint_cdist_forward_dynamic(x1_a, x2_a, 1.0)
    dynamic_out_a = dynamic_net(x1_a, x2_a)
    np.testing.assert_allclose(dynamic_out_a.asnumpy(), static_out_a.asnumpy(), rtol=1e-4, atol=1e-4)
    static_out_b = mint_cdist_forward_dynamic(x1_b, x2_b, 1.0)
    dynamic_out_b = dynamic_net(x1_b, x2_b)
    np.testing.assert_allclose(dynamic_out_b.asnumpy(), static_out_b.asnumpy(), rtol=1e-4, atol=1e-4)


def run_dynamic_backward(device_target, is_dynamic_rank):
    set_pynative_context(device_target)
    test_dynamic = TestDynamicGrad(MintCdistNet())
    x1 = Tensor(np.random.randn(3, 7, 9).astype(np.float32))
    x2 = Tensor(np.random.randn(3, 8, 9).astype(np.float32))
    test_dynamic.test_dynamic_grad_net([x1, x2], is_dynamic_rank=is_dynamic_rank)


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    mem_peak=4.01
)
def test_mint_cdist_dynamic_forward_ascend():
    """
    Feature: mint.cdist dynamic forward.
    Description: Verify pynative dynamic forward on Ascend.
    Expectation: Success.
    """
    run_dynamic_forward('Ascend')


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level1',
    card_mark='onecard'
)
def test_mint_cdist_dynamic_forward_cpu():
    """
    Feature: mint.cdist dynamic forward.
    Description: Verify pynative dynamic forward on CPU.
    Expectation: Success.
    """
    run_dynamic_forward('CPU')


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    mem_peak=4.01
)
@pytest.mark.parametrize('is_dynamic_rank', [False, True])
def test_mint_cdist_dynamic_backward_ascend(is_dynamic_rank):
    """
    Feature: mint.cdist dynamic backward.
    Description: Verify pynative dynamic backward on Ascend.
    Expectation: Success.
    """
    run_dynamic_backward('Ascend', is_dynamic_rank)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level1',
    card_mark='onecard'
)
@pytest.mark.parametrize('is_dynamic_rank', [False, True])
def test_mint_cdist_dynamic_backward_cpu(is_dynamic_rank):
    """
    Feature: mint.cdist dynamic backward.
    Description: Verify pynative dynamic backward on CPU.
    Expectation: Success.
    """
    run_dynamic_backward('CPU', is_dynamic_rank)
