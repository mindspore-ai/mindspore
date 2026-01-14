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

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context

def generate_random_input(shape, dtype):
    x = np.random.randn(*shape).astype(dtype)
    y = np.random.randn(*shape).astype(dtype)
    expect = np.logical_xor(x, y)
    return x, y, expect


class LogicalXorNet(nn.Cell):
    def construct(self, x, other):
        return x.logical_xor(other)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize("mode", ['pynative', 'KBK'])
def test_logical_xor(mode):
    """
    Feature: test Tensor.clamp
    Description: Verify the result of Tensor.clamp
    Expectation: expect correct forward result
    """
    x, y, expect = generate_random_input((2, 3, 4, 5), np.int32)
    y2 = True
    expect2 = np.logical_xor(x, y2)
    x = Tensor(x, dtype=ms.int32)
    y = Tensor(y, dtype=ms.int32)
    net = LogicalXorNet()
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    output = net(x, y)
    output2 = net(x, y2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    np.testing.assert_allclose(output2.asnumpy(), expect2, rtol=1e-3)
