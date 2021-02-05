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
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.div = P.RealDiv()
        self.sqrt = P.Sqrt()
        self.pow = P.Pow()
        self.neg = P.Neg()
        self.reducemin = P.ReduceMin()
        self.reshape = P.Reshape()

    def construct(self, x, y):
        add_res1 = self.add(x, 4)
        add_res2 = self.add(add_res1, 5)
        sub_res = self.sub(y, 3)
        mul_res = self.mul(self.sqrt(add_res2), self.sqrt(sub_res))
        div_res = self.div(mul_res, self.sqrt(mul_res))
        pow_res = self.pow(y, 2)
        neg_res = self.neg(self.neg(pow_res))
        add_res3 = self.add(neg_res, div_res)
        resh_res = self.reshape(add_res3, (2, 12, 3))
        return self.reducemin(resh_res, 1)


def test_basic():
    input_x = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    input_y = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    input_y = np.abs(input_y) + 3
    add_res = input_x + 9
    sub_res = input_y + (-3)
    mul_res = np.sqrt(add_res * sub_res)
    div_res = np.sqrt(mul_res)
    pow_res = input_y * input_y
    neg_res = pow_res
    add_res3 = neg_res + div_res
    expect = np.min(add_res3, (1, 2))

    net = Net()
    result = net(Tensor(input_x), Tensor(input_y))

    res = np.allclose(expect, result.asnumpy(), rtol=1.e-4,
                      atol=1.e-7, equal_nan=True)
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
