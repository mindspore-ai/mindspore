# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
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
        self.reducesum = P.ReduceSum(keep_dims=True)
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
        neg_res = self.neg(resh_res)
        red_res = self.reducesum(neg_res, 0)
        return self.reducemin(self.reducemin(red_res, 1), 1)


class EmptyNet(Cell):
    def __init__(self):
        super(EmptyNet, self).__init__()
        self.add = P.Add()
        self.neg = P.Neg()

    def construct(self, x, y):
        add_res1 = self.add(x, y)
        neg_res1 = self.neg(x)
        add_res2 = self.add(add_res1, neg_res1)
        return add_res2


def run_basic():
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
    neg_res = np.negative(add_res3)
    red_res = np.sum(neg_res, axis=0, keepdims=True)
    expect = np.min(red_res, (1, 2, 3))

    net = Net()
    result = net(Tensor(input_x), Tensor(input_y))

    res = np.allclose(expect, result.asnumpy(), rtol=1.e-4,
                      atol=1.e-7, equal_nan=True)
    assert res


def run_empty_graph():
    input_x = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    input_y = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    expect = input_y

    net = EmptyNet()
    result = net(Tensor(input_x), Tensor(input_y))

    res = np.allclose(expect, result.asnumpy(), rtol=1.e-4,
                      atol=1.e-7, equal_nan=True)
    assert res


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_basic_gpu():
    """
    Feature: test graph kernel arithmetic simplify
    Description: run test case on GPU
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True)
    run_basic()
    run_empty_graph()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_basic_ascend():
    """
    Feature: test graph kernel arithmetic simplify
    Description: run test case on Ascend
    Expectation: the result match with expect
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True)
    run_basic()
    run_empty_graph()
