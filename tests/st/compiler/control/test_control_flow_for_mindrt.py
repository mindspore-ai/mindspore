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

"""test control flow for mindrt."""

import numpy as np
from tests.mark_utils import arg_mark
import mindspore
from mindspore import context, nn, ops, Tensor, CSRTensor, Parameter, jit, mutable
from mindspore.ops import functional as F


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.print = ops.Print()
        self.param_a = Parameter(Tensor(5, mindspore.int32), name='a')
        self.param_b = Parameter(Tensor(4, mindspore.int32), name='b')

    def construct(self, x):
        out = 0
        for _ in range(2):
            out += self.func1(x)
        return out

    def func1(self, x):
        out = x
        i = 0
        while i < 1:
            out += self.func2(x)
            i = i + 1
            self.print(out)
        return out

    def func2(self, x):
        if x > 10:
            return self.param_a
        return self.param_b


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_repeat_control_arrow_for_stack_actor():
    """
    Feature: Runtime.
    Description: Duplicate side effects depend on stack actors..
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array([1]), mindspore.int32)
    net = Net()
    out = net(x)
    result = 14
    assert out == result


@jit
def switch_op(x, y):
    z1 = y + 1
    z2 = Tensor(5, mindspore.int32)
    return F.switch(x, z1, z2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_switch_op():
    """
    Feature: Runtime.
    Description: Test switch op.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(False, mindspore.bool_)
    y = Tensor(1, mindspore.int32)
    out = switch_op(x, y)
    assert out == 5


@jit
def switch_single_op(x, y, z):
    return F.switch(x, y, z)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_switch_single_op():
    """
    Feature: Runtime.
    Description: Test switch single op.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(False, mindspore.bool_)
    y = Tensor(1, mindspore.int32)
    z = Tensor(2, mindspore.int32)
    out = switch_single_op(x, y, z)
    assert out == 2


class TupleNet(nn.Cell):
    def construct(self, x, y, z):
        while ops.less(x, y):
            z = ops.make_tuple(ops.add(F.tuple_getitem(z, 0), 3), ops.add(F.tuple_getitem(z, 1), 2))
            x = ops.add(x, 1)
        return z


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_tuple_parameter():
    """
    Feature: Runtime.
    Description: input a tuple parameter for root graph.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array([2]), mindspore.int32)
    y = Tensor(np.array([4]), mindspore.int32)
    z1 = Tensor(np.array([8]), mindspore.int32)
    z2 = Tensor(np.array([4]), mindspore.int32)
    z = mutable((z1, z2))
    net = TupleNet()
    out = net(x, y, z)
    assert out == (14, 8)


class CSRNet(nn.Cell):
    def construct(self, x, y, z):
        while x < y:
            z = CSRTensor(z.indptr, z.indices, z.values + x, z.shape)
            x = x + 1
        return z


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_csr_parameter():
    """
    Feature: Runtime.
    Description: input a tuple parameter for root graph.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(2, mindspore.float32)
    y = Tensor(4, mindspore.float32)
    indptr = Tensor([0, 1, 2], mindspore.int32)
    indices = Tensor([0, 1], mindspore.int32)
    values = Tensor([1, 2], mindspore.float32)
    shape = (2, 4)
    z = CSRTensor(indptr, indices, values, shape)
    net = CSRNet()
    out = net(x, y, z)
    assert np.all(out.values.asnumpy() == [6., 7.])
