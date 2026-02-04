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
from tests.mark_utils import arg_mark
from tests.st.common.random_generator import generate_numpy_ndarray_by_randn
import mindspore as ms
import mindspore.nn as nn
from mindspore import context

def generate_random_input(shape, shape1, shape2):
    x = generate_numpy_ndarray_by_randn(shape, np.float32, 'x', seed=0)
    batch1 = generate_numpy_ndarray_by_randn(shape1, np.float32, 'batch1', seed=0)
    batch2 = generate_numpy_ndarray_by_randn(shape2, np.float32, 'batch2', seed=0)
    return x, batch1, batch2

def generate_expect_forward_output(input1, batch1, batch2, beta=1, alpha=1):
    return beta * input1 + alpha * (batch1 @ batch2)

class BaddbmmNet(nn.Cell):
    def construct(self, bias, mat1, mat2, beta=1.0, alpha=1.0):
        return bias.baddbmm(mat1, mat2, beta=beta, alpha=alpha)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize("mode", ['pynative', 'KBK'])
def test_baddbmm(mode):
    """
    Feature: test Tensor.clamp
    Description: Verify the result of Tensor.clamp
    Expectation: expect correct forward result
    """
    input_shape1 = (3, 4, 5)
    input_shape2 = (4, 5)
    batch1_shape = (3, 4, 2)
    batch2_shape = (3, 2, 5)
    beta = 1
    alpha = 2.0
    input1, batch1, batch2 = generate_random_input(input_shape1, batch1_shape, batch2_shape)
    input2, batch1, batch2 = generate_random_input(input_shape2, batch1_shape, batch2_shape)
    expect_forward = generate_expect_forward_output(input1, batch1, batch2)
    expect_forward2 = generate_expect_forward_output(input2, batch1, batch2, beta, alpha)

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

    net = BaddbmmNet()
    output = net(ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
    output2 = net(ms.Tensor(input2), ms.Tensor(batch1), ms.Tensor(batch2), beta, alpha)

    np.testing.assert_allclose(output.asnumpy(), expect_forward, 3e-3, 3e-3)
    np.testing.assert_allclose(output2.asnumpy(), expect_forward2, 3e-3, 3e-3)
