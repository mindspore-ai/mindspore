# Copyright 2020 Huawei Technologies Co., Ltd
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
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops.operations._grad_ops as G


class MinmumGradNet(Cell):
    def __init__(self):
        super(MinmumGradNet, self).__init__()
        self.minimum_grad = G.MinimumGrad()

    def construct(self, x, y, dy):
        return self.minimum_grad(x, y, dy)


def test_minimum_grad():
    np.random.seed(0)
    input_x = np.random.normal(0, 1, [2, 3]).astype(np.float32)
    input_y = np.random.normal(0, 1, [2, 3]).astype(np.float32)
    input_dout = np.minimum(input_x, input_y).astype(np.float32)
    net = MinmumGradNet()
    result = net(Tensor(input_x), Tensor(input_y), Tensor(input_dout))
    dx = input_dout * (input_x <= input_y)
    dy = input_dout - dx
    assert np.allclose(result[0].asnumpy(), dx, rtol=1.e-4, atol=1.e-8, equal_nan=True)
    assert np.allclose(result[1].asnumpy(), dy, rtol=1.e-4, atol=1.e-8, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_basic_gpu():
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="GPU")
    test_minimum_grad()


def test_basic_ascend():
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="Ascend")
    test_minimum_grad()
