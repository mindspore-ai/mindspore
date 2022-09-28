# Copyright 2022 Huawei Technologies Co., Ltd
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

context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="GPU")


class MaxmumGradNet(Cell):
    def __init__(self):
        super(MaxmumGradNet, self).__init__()
        self.maximum_grad = G.MaximumGrad()

    def construct(self, x, y, dy):
        return self.maximum_grad(x, y, dy)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_broadcast_grad_gpu_tpye():
    """
    Feature: ALL To ALL
    Description: test cases for broadcast_grad of two tensors
    Expectation: the result match to numpy
    """
    np.random.seed(1)
    input_x = np.arange(2 * 3 * 2).reshape((2, 3, 2))
    input_y = np.arange(88, 2 * 3 * 2 + 88).reshape((2, 3, 2))
    input_dout = np.maximum(input_x, input_y)
    net = MaxmumGradNet()
    dtypes = (np.int32, np.int64, np.float16, np.float32, np.float64,
              np.int16, np.uint16, np.uint32, np.uint64)
    for dtype in dtypes:
        result = net(Tensor(input_x.astype(dtype)), Tensor(input_y.astype(dtype)),
                     Tensor(input_dout.astype(dtype)))
        dx = input_dout * (input_x >= input_y)
        dy = input_dout - dx
        assert np.allclose(result[0].asnumpy(), dx, rtol=1.e-4, atol=1.e-8, equal_nan=True)
        assert np.allclose(result[1].asnumpy(), dy, rtol=1.e-4, atol=1.e-8, equal_nan=True)
    