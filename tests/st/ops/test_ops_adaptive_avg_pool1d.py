# Copyright 2024 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest

import mindspore as ms
from mindspore import mint
from mindspore import Tensor
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


@test_utils.run_with_cell
def forward_adaptive_avg_pool1d_net(input_x, output_size):
    return mint.nn.functional.adaptive_avg_pool1d(input_x, output_size)

@test_utils.run_with_cell
def forward_adaptive_avg_pool1d_net_dyn(input_x):
    return mint.nn.functional.adaptive_avg_pool1d(input_x, 8)

@test_utils.run_with_cell
def grad_adaptive_avg_pool1d_net(input_x, output_size):
    net = mint.nn.functional.adaptive_avg_pool1d
    return ms.grad(net)(input_x, output_size)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_batchnorm1d_dyn():
    """
    Feature: Dynamic shape of adaptive_avg_pool1d
    Description: test adaptive_avg_pool1d with dynamic rank/shape.
    Expectation: success
    """
    in1 = Tensor(np.random.randn(4, 4), dtype=ms.float32)
    in2 = Tensor(np.random.randn(2, 4, 2), dtype=ms.float32)
    TEST_OP(forward_adaptive_avg_pool1d_net_dyn, [[in1], [in2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['ViewTensor', 'EmptyTensor', 'ScalarTensor'])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=36.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_adaptive_avg_pool1d(mode):
    """
    Feature: adaptive_avg_pool1d
    Description: Verify the result of adaptive_avg_pool1d.
    Expectation: success
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(jit_config={"jit_level": "O0"}, mode=ms.GRAPH_MODE)
    input_np = np.arange(0, 2 * 4 * 4, 1).reshape(2, 4, 4).astype(np.float32)
    input_ms = ms.Tensor(input_np)
    expect_out = ms.Tensor([[[0.5000, 2.5000], [4.5000, 6.5000],
                             [8.5000, 10.5000], [12.5000, 14.5000]],
                            [[16.5000, 18.5000], [20.5000, 22.5000],
                             [24.5000, 26.5000], [28.5000, 30.5000]]], dtype=ms.float32)
    expect_grad = ms.Tensor([[[0.5000, 0.5000, 0.5000, 0.5000],
                              [0.5000, 0.5000, 0.5000, 0.5000],
                              [0.5000, 0.5000, 0.5000, 0.5000],
                              [0.5000, 0.5000, 0.5000, 0.5000]],
                             [[0.5000, 0.5000, 0.5000, 0.5000],
                              [0.5000, 0.5000, 0.5000, 0.5000],
                              [0.5000, 0.5000, 0.5000, 0.5000],
                              [0.5000, 0.5000, 0.5000, 0.5000]]], dtype=ms.float32)
    output = forward_adaptive_avg_pool1d_net(input_ms, 2)
    assert np.allclose(output.asnumpy(), expect_out, 1e-03, 1e-03)
    output_grad = grad_adaptive_avg_pool1d_net(input_ms, 2)
    assert np.allclose(output_grad.asnumpy(), expect_grad, 1e-03, 1e-03)
