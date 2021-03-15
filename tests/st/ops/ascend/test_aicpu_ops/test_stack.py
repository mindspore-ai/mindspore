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
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops.operations import _inner_ops as P

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, index=0, shapes_and_types=None):
        super(Net, self).__init__()
        shapes_and_types.reverse()
        self.init = P.StackInit(index)
        self.push = P.StackPush(index)
        self.pop = [P.StackPop(index, shape, dtype) for (shape, dtype) in shapes_and_types]
        self.destroy = P.StackDestroy(index)

    def construct(self, x1, x2, x3):
        self.init()
        self.push(x1)
        self.push(x2)
        self.push(x3)
        y1 = self.pop[0]()
        y2 = self.pop[1]()
        y3 = self.pop[2]()
        self.destroy()
        return y1, y2, y3


class NetTwoStack(nn.Cell):
    def __init__(self, index=0, shapes_and_types=None):
        super(NetTwoStack, self).__init__()
        self.init_0 = P.StackInit(index)
        self.push_0 = P.StackPush(index)
        self.pop_0 = [P.StackPop(index, shape, dtype) for (shape, dtype) in shapes_and_types]
        self.destroy_0 = P.StackDestroy(index)

        index += 1
        self.init_1 = P.StackInit(index)
        self.push_1 = P.StackPush(index)
        self.pop_1 = [P.StackPop(index, shape, dtype) for (shape, dtype) in shapes_and_types]
        self.destroy_1 = P.StackDestroy(index)

    def construct(self, x1, x2, x3):
        self.init_0()
        self.init_1()

        self.push_0(x1)
        self.push_1(x3)
        y1 = self.pop_0[0]()
        z1 = self.pop_1[2]()
        self.push_0(x2)
        self.push_0(x3)
        self.push_1(x1)
        self.push_1(x2)
        y2 = self.pop_0[2]()
        z2 = self.pop_1[1]()
        y3 = self.pop_0[1]()
        z3 = self.pop_1[0]()

        self.destroy_0()
        self.destroy_1()
        return y1, y2, y3, z1, z2, z3


def test_net():
    x1 = Tensor(np.random.randn(4,).astype(np.float64))
    x2 = Tensor(np.random.randn(4, 6).astype(np.float32))
    x3 = Tensor(np.random.randint(100, size=(3, 4, 5)).astype(np.int32))

    shapes_and_types = []
    shapes_and_types.append((x1.shape, x1.dtype))
    shapes_and_types.append((x2.shape, x2.dtype))
    shapes_and_types.append((x3.shape, x3.dtype))

    net = Net(2018, shapes_and_types)
    y1, y2, y3 = net(x1, x2, x3)
    print(x1)
    print(x2)
    print(x3)
    print(y1)
    print(y2)
    print(y3)
    assert np.array_equal(y1.asnumpy(), x3.asnumpy())
    assert np.array_equal(y2.asnumpy(), x2.asnumpy())
    assert np.array_equal(y3.asnumpy(), x1.asnumpy())


def test_net_tow_stack():
    x1 = Tensor(np.random.randn(4,).astype(np.float64))
    x2 = Tensor(np.random.randn(4, 6).astype(np.float32))
    x3 = Tensor(np.random.randint(100, size=(3, 4, 5)).astype(np.int32))

    shapes_and_types = []
    shapes_and_types.append((x1.shape, x1.dtype))
    shapes_and_types.append((x2.shape, x2.dtype))
    shapes_and_types.append((x3.shape, x3.dtype))

    net = NetTwoStack(1998, shapes_and_types)
    y1, y2, y3, z1, z2, z3 = net(x1, x2, x3)
    print(x1)
    print(x2)
    print(x3)
    print(y1)
    print(y2)
    print(y3)
    assert np.array_equal(y1.asnumpy(), x1.asnumpy())
    assert np.array_equal(y2.asnumpy(), x3.asnumpy())
    assert np.array_equal(y3.asnumpy(), x2.asnumpy())

    assert np.array_equal(z1.asnumpy(), x3.asnumpy())
    assert np.array_equal(z2.asnumpy(), x2.asnumpy())
    assert np.array_equal(z3.asnumpy(), x1.asnumpy())
