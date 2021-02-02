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


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()
        self.mul = P.Mul()

    def construct(self, x):
        mul_res = self.mul(x, x)
        square_res = P.Square()(x)
        return self.add(mul_res, square_res)


def test_basic():
    input_x = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    mul_res = input_x * input_x
    square_res = np.square(input_x)
    expect = mul_res + square_res

    net = Net()
    result = net(Tensor(input_x))

    res = np.allclose(expect, result.asnumpy(), rtol=1.e-4, atol=1.e-7, equal_nan=True)
    assert res


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_basic_gpu():
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="GPU")
    test_basic()


def test_basic_ascend():
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="Ascend")
    test_basic()
