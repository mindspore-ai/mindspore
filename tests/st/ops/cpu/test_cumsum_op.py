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

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

axis0 = 0
axis1 = 1
axis2 = 2
axis3 = 3
axis4 = 4
axis5 = -1
axis6 = -2

x0 = np.random.rand(3, 3, 4, 5, 3).astype(np.float32)
x1 = np.random.rand(2, 3, 4, 5, 3).astype(np.float16)
x2 = np.random.randint(-10000, 10000, size=(2, 3, 4, 5, 3)).astype(np.int32)
x3 = np.random.randint(-5, 5, size=(2, 3, 4, 5, 3)).astype(np.int8)
x4 = np.random.randint(0, 10, size=(2, 3, 4, 5, 3)).astype(np.uint8)
x5 = np.random.rand(3).astype(np.float32)

list1 = [x0, x1, x2, x3, x4]
list2 = [axis0, axis1, axis2, axis3, axis4, axis5, axis6]


class CumSum(nn.Cell):
    def __init__(self, exclusive=False, reverse=False):
        super(CumSum, self).__init__()
        self.cumsum_op = P.CumSum(exclusive, reverse)

        self.x0 = Tensor(x0)
        self.axis0 = axis0
        self.x1 = Tensor(x0)
        self.axis1 = axis1
        self.x2 = Tensor(x0)
        self.axis2 = axis2
        self.x3 = Tensor(x0)
        self.axis3 = axis3
        self.x4 = Tensor(x0)
        self.axis4 = axis4
        self.x5 = Tensor(x0)
        self.axis5 = axis5
        self.x6 = Tensor(x0)
        self.axis6 = axis6

        self.x7 = Tensor(x1)
        self.axis7 = axis0
        self.x8 = Tensor(x1)
        self.axis8 = axis1
        self.x9 = Tensor(x1)
        self.axis9 = axis2
        self.x10 = Tensor(x1)
        self.axis10 = axis3
        self.x11 = Tensor(x1)
        self.axis11 = axis4
        self.x12 = Tensor(x1)
        self.axis12 = axis5
        self.x13 = Tensor(x1)
        self.axis13 = axis6

        self.x14 = Tensor(x2)
        self.axis14 = axis0
        self.x15 = Tensor(x2)
        self.axis15 = axis1
        self.x16 = Tensor(x2)
        self.axis16 = axis2
        self.x17 = Tensor(x2)
        self.axis17 = axis3
        self.x18 = Tensor(x2)
        self.axis18 = axis4
        self.x19 = Tensor(x2)
        self.axis19 = axis5
        self.x20 = Tensor(x2)
        self.axis20 = axis6

        self.x21 = Tensor(x3)
        self.axis21 = axis0
        self.x22 = Tensor(x3)
        self.axis22 = axis1
        self.x23 = Tensor(x3)
        self.axis23 = axis2
        self.x24 = Tensor(x3)
        self.axis24 = axis3
        self.x25 = Tensor(x3)
        self.axis25 = axis4
        self.x26 = Tensor(x3)
        self.axis26 = axis5
        self.x27 = Tensor(x3)
        self.axis27 = axis6

        self.x28 = Tensor(x4)
        self.axis28 = axis0
        self.x29 = Tensor(x4)
        self.axis29 = axis1
        self.x30 = Tensor(x4)
        self.axis30 = axis2
        self.x31 = Tensor(x4)
        self.axis31 = axis3
        self.x32 = Tensor(x4)
        self.axis32 = axis4
        self.x33 = Tensor(x4)
        self.axis33 = axis5
        self.x34 = Tensor(x4)
        self.axis34 = axis6

        self.x35 = Tensor(x5)
        self.axis35 = axis0

    def construct(self):
        return (self.cumsum_op(self.x0, self.axis0),
                self.cumsum_op(self.x1, self.axis1),
                self.cumsum_op(self.x2, self.axis2),
                self.cumsum_op(self.x3, self.axis3),
                self.cumsum_op(self.x4, self.axis4),
                self.cumsum_op(self.x5, self.axis5),
                self.cumsum_op(self.x6, self.axis6),
                self.cumsum_op(self.x7, self.axis7),
                self.cumsum_op(self.x8, self.axis8),
                self.cumsum_op(self.x9, self.axis9),
                self.cumsum_op(self.x10, self.axis10),
                self.cumsum_op(self.x11, self.axis11),
                self.cumsum_op(self.x12, self.axis12),
                self.cumsum_op(self.x13, self.axis13),
                self.cumsum_op(self.x14, self.axis14),
                self.cumsum_op(self.x15, self.axis15),
                self.cumsum_op(self.x16, self.axis16),
                self.cumsum_op(self.x17, self.axis17),
                self.cumsum_op(self.x18, self.axis18),
                self.cumsum_op(self.x19, self.axis19),
                self.cumsum_op(self.x20, self.axis20),
                self.cumsum_op(self.x21, self.axis21),
                self.cumsum_op(self.x22, self.axis22),
                self.cumsum_op(self.x23, self.axis23),
                self.cumsum_op(self.x24, self.axis24),
                self.cumsum_op(self.x25, self.axis25),
                self.cumsum_op(self.x26, self.axis26),
                self.cumsum_op(self.x27, self.axis27),
                self.cumsum_op(self.x28, self.axis28),
                self.cumsum_op(self.x29, self.axis29),
                self.cumsum_op(self.x30, self.axis30),
                self.cumsum_op(self.x31, self.axis31),
                self.cumsum_op(self.x32, self.axis32),
                self.cumsum_op(self.x33, self.axis33),
                self.cumsum_op(self.x34, self.axis34),
                self.cumsum_op(self.x35, self.axis35))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cumsum():
    cumsum = CumSum()
    output = cumsum()

    k = 0

    for i in list1:
        for j in list2:
            expect = np.cumsum(i, axis=j)
            diff = abs(output[k].asnumpy() - expect)
            error = np.ones(shape=expect.shape) * 1.0e-5
            assert np.all(diff < error)
            assert output[k].shape == expect.shape
            k += 1

    expect = np.cumsum(x5, axis=axis0)
    diff = abs(output[k].asnumpy() - expect)
    error = np.ones(shape=expect.shape) * 1.0e-5
    assert np.all(diff < error)
    assert output[k].shape == expect.shape


def test_cumsum2():
    cumsum = CumSum(exclusive=False, reverse=True)
    output = cumsum()

    k = 0

    for i in list1:
        for j in list2:
            result1 = np.flip(i, axis=j)
            result2 = np.cumsum(result1, axis=j)
            expect = np.flip(result2, axis=j)
            diff = abs(output[k].asnumpy() - expect)
            error = np.ones(shape=expect.shape) * 1.0e-5
            assert np.all(diff < error)
            assert output[k].shape == expect.shape
            k += 1

    result1 = np.flip(x5, axis=axis0)
    result2 = np.cumsum(result1, axis=axis0)
    expect = np.flip(result2, axis=axis0)
    diff = abs(output[k].asnumpy() - expect)
    error = np.ones(shape=expect.shape) * 1.0e-5
    assert np.all(diff < error)
    assert output[k].shape == expect.shape


def test_cumsum3():
    cumsum = CumSum(exclusive=True, reverse=False)
    output = cumsum()

    k = 0

    for i in list1:
        for j in list2:
            result1 = np.insert(i, 0, [0], axis=j)
            result2 = np.delete(result1, -1, axis=j)
            expect = np.cumsum(result2, axis=j)
            diff = abs(output[k].asnumpy() - expect)
            error = np.ones(shape=expect.shape) * 1.0e-5
            assert np.all(diff < error)
            assert output[k].shape == expect.shape
            k += 1

    result1 = np.insert(x5, 0, [0], axis=axis0)
    result2 = np.delete(result1, -1, axis=axis0)
    expect = np.cumsum(result2, axis=axis0)
    diff = abs(output[k].asnumpy() - expect)
    error = np.ones(shape=expect.shape) * 1.0e-5
    assert np.all(diff < error)
    assert output[k].shape == expect.shape


def test_cumsum4():
    cumsum = CumSum(exclusive=True, reverse=True)
    output = cumsum()

    k = 0

    for i in list1:
        for j in list2:
            result1 = np.flip(i, axis=j)
            result2 = np.insert(result1, 0, [0], axis=j)
            result3 = np.delete(result2, -1, axis=j)
            result4 = np.cumsum(result3, axis=j)
            expect = np.flip(result4, axis=j)
            diff = abs(output[k].asnumpy() - expect)
            error = np.ones(shape=expect.shape) * 1.0e-5
            assert np.all(diff < error)
            assert output[k].shape == expect.shape
            k += 1

    result1 = np.flip(x5, axis=axis0)
    result2 = np.insert(result1, 0, [0], axis=axis0)
    result3 = np.delete(result2, -1, axis=axis0)
    result4 = np.cumsum(result3, axis=axis0)
    expect = np.flip(result4, axis=axis0)
    diff = abs(output[k].asnumpy() - expect)
    error = np.ones(shape=expect.shape) * 1.0e-5
    assert np.all(diff < error)
    assert output[k].shape == expect.shape


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.op = P.CumSum()

    def construct(self, x):
        return self.op(x, 0)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cumsum_dshape():
    """
    Feature: Test cumsum dynamic shape.
    Description: Test cumsum dynamic shape.
    Expectation: Success.
    """
    net = Net()
    input_x_dyn = Tensor(shape=[3, None], dtype=ms.float32)
    net.set_inputs(input_x_dyn)
    input_x = Tensor(np.random.random(([3, 10])), dtype=ms.float32)
    output = net(input_x)
    expect_shape = (3, 10)
    assert output.asnumpy().shape == expect_shape
