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
import mindspore.ops.operations as P
from mindspore.nn.graph_kernels import ReLU


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.TensorAdd()
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.relu = ReLU()

    def construct(self, x, y):
        sub_res = self.sub(x, y)
        mul_res = self.mul(sub_res, x)
        relu_res = self.relu(mul_res)
        square_res = P.Square()(relu_res)
        add_res = self.add(relu_res, square_res)
        add1_res = self.add(add_res, add_res)
        return self.add(add1_res, add1_res)


def test_basic():
    input_x = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    input_y = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    sub_res = input_x - input_y
    mul_res = sub_res * input_x
    relu_res = np.maximum(mul_res, 0)
    square_res = np.square(relu_res)
    add_res = relu_res + square_res
    add1_res = add_res + add_res
    expect = add1_res + add1_res

    net = Net()
    result = net(Tensor(input_x), Tensor(input_y))

    res = np.allclose(expect, result.asnumpy(), rtol=1.e-4, atol=1.e-7, equal_nan=True)
    assert res


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_basic_gpu():
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="GPU")
    test_basic()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_basic_ascend():
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="Ascend")
    test_basic()
