# Copyright 2019 Huawei Technologies Co., Ltd
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
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops.operations import Tile

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

input_x0 = np.arange(2).reshape((2, 1, 1)).astype(np.float32)
mul0 = (8, 1, 1)
input_x1 = np.arange(32).reshape((2, 4, 4)).astype(np.float32)
mul1 = (2, 2, 2)
input_x2 = np.arange(1).reshape((1, 1, 1)).astype(np.float32)
mul2 = (1, 1, 1)


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.Tile = Tile()

        self.input_x0 = Parameter(initializer(Tensor(input_x0), input_x0.shape), name='x0')
        self.mul0 = mul0
        self.input_x1 = Parameter(initializer(Tensor(input_x1), input_x1.shape), name='x1')
        self.mul1 = mul1
        self.input_x2 = Parameter(initializer(Tensor(input_x2), input_x2.shape), name='x2')
        self.mul2 = mul2

    @ms_function
    def construct(self):
        output = (self.Tile(self.input_x0, self.mul0),
                  self.Tile(self.input_x1, self.mul1),
                  self.Tile(self.input_x2, self.mul2))
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tile():
    net = Net()
    output = net()

    expect0 = np.tile(input_x0, mul0)
    diff0 = output[0].asnumpy() - expect0
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output[0].shape == expect0.shape

    expect1 = np.tile(input_x1, mul1)
    diff1 = output[1].asnumpy() - expect1
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output[1].shape == expect1.shape

    expect2 = np.tile(input_x2, mul2)
    diff2 = output[2].asnumpy() - expect2
    error2 = np.ones(shape=expect2.shape) * 1.0e-5
    assert np.all(diff2 < error2)
    assert output[2].shape == expect2.shape
