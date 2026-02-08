# Copyright 2024 Huawei Technologies Co., Ltd
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

import numpy as np
import mindspore
from mindspore import context, ops, nn, Tensor
from tests.mark_utils import arg_mark
context.set_context(mode=context.GRAPH_MODE)
context.set_context(jit_config={"jit_level": "O0"})


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reshape = ops.Reshape()

    def construct(self, x, y, z):
        a = x + 2
        b = y * 2
        if z < 3:
            c = self.reshape(a, b.shape)
        else:
            c = x
        return c


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_free_by_entrance_actor():
    """
    Feature: eliminate nopnode.
    Description: base scene.
    Expectation: No exception.
    """
    x_dyn = Tensor(shape=[6, None], dtype=mindspore.float32)
    y_dyn = Tensor(shape=[None, None], dtype=mindspore.float32)
    z = Tensor(2, mindspore.float32)
    net = Net()
    net.set_inputs(x_dyn, y_dyn, z)
    x = Tensor(np.ones([6, 1]), mindspore.float32)
    y = Tensor(np.ones([3, 2]), mindspore.float32)
    out = net(x, y, z)
    assert out.shape == (3, 2)


class NetStack(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reshape = ops.Reshape()

    def construct(self, x, y, z):
        y = y + 5
        if y[0][0] < 3:
            z = z * 3
        else:
            z = z * 2
        return self.reshape(z, y.shape)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_free_by_stack_actor():
    """
    Feature: eliminate nopnode.
    Description: base scene.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(2, mindspore.float32)
    y_dyn = Tensor(shape=[None, None], dtype=mindspore.float32)
    z_dyn = Tensor(shape=[6, None], dtype=mindspore.float32)
    net = NetStack()
    net.set_inputs(x, y_dyn, z_dyn)
    y = Tensor(np.ones([3, 2]), mindspore.float32)
    z = Tensor(np.ones([6, 1]), mindspore.float32)
    out = net(x, y, z)
    assert out.shape == (3, 2)



class NetGather1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reshape = ops.Reshape()

    def construct(self, x, y, z):
        a = x + 2
        b = y * 2
        if z < 3:
            c = self.reshape(a, b.shape)
            d = b.shape[0] + 1
        else:
            c = x
            d = b.shape[0] - 2
        return c, d


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_free_by_gather_actor_1():
    """
    Feature: eliminate nopnode.
    Description: base scene.
    Expectation: No exception.
    """
    x_dyn = Tensor(shape=[6, None], dtype=mindspore.float32)
    y_dyn = Tensor(shape=[None, None], dtype=mindspore.float32)
    z = Tensor(2, mindspore.float32)
    net = NetGather1()
    net.set_inputs(x_dyn, y_dyn, z)
    x = Tensor(np.ones([6, 1]), mindspore.float32)
    y = Tensor(np.ones([3, 2]), mindspore.float32)
    out = net(x, y, z)
    assert out[0].shape == (3, 2)


class NetGather2(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reshape = ops.Reshape()

    def construct(self, x, y, z):
        a = x + 2
        b = y * 2
        if z < 3:
            c = self.reshape(a, b.shape)
            d = b.shape[0] + 1
        else:
            c = x
            d = b.shape[0] * 2
        f = b + c
        g = f / 3
        return c, d, g


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_free_by_gather_actor_2():
    """
    Feature: eliminate nopnode.
    Description: base scene.
    Expectation: No exception.
    """
    x_dyn = Tensor(shape=[6, None], dtype=mindspore.float32)
    y_dyn = Tensor(shape=[None, None], dtype=mindspore.float32)
    z = Tensor(2, mindspore.float32)
    net = NetGather2()
    net.set_inputs(x_dyn, y_dyn, z)
    x = Tensor(np.ones([6, 1]), mindspore.float32)
    y = Tensor(np.ones([3, 2]), mindspore.float32)
    out = net(x, y, z)
    assert out[0].shape == (3, 2)


class NetGather3(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reshape = ops.Reshape()

    def construct(self, x, y, z):
        a = x + 2
        b = y * 2
        if z < 3:
            c = self.reshape(a, b.shape)
            d = b + 3
        else:
            c = x
            d = b + 4
        e = 1 + d
        f = e * 2
        g = f / 3
        return c, d, g


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_free_by_gather_actor_3():
    """
    Feature: eliminate nopnode.
    Description: base scene.
    Expectation: No exception.
    """
    x_dyn = Tensor(shape=[6, None], dtype=mindspore.float32)
    y_dyn = Tensor(shape=[None, None], dtype=mindspore.float32)
    z = Tensor(2, mindspore.float32)
    net = NetGather3()
    net.set_inputs(x_dyn, y_dyn, z)
    x = Tensor(np.ones([6, 1]), mindspore.float32)
    y = Tensor(np.ones([3, 2]), mindspore.float32)
    out = net(x, y, z)
    assert out[0].shape == (3, 2)


class NetGather4(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reshape = ops.Reshape()

    def construct(self, x, y, z):
        a = x + 2
        b = y * 2
        if z < 3:
            e = z * 2
            c = self.reshape(a, b.shape)
            d = b + 3
        else:
            e = z * 3
            c = x
            d = b + 4
        e = 1 + d
        return c, d, e


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_free_by_gather_actor_4():
    """
    Feature: eliminate nopnode.
    Description: base scene.
    Expectation: No exception.
    """
    x_dyn = Tensor(shape=[6, None], dtype=mindspore.float32)
    y_dyn = Tensor(shape=[None, None], dtype=mindspore.float32)
    z = Tensor(2, mindspore.float32)
    net = NetGather4()
    net.set_inputs(x_dyn, y_dyn, z)
    x = Tensor(np.ones([6, 1]), mindspore.float32)
    y = Tensor(np.ones([3, 2]), mindspore.float32)
    out = net(x, y, z)
    assert out[0].shape == (3, 2)
