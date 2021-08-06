# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore import context
from mindspore import Tensor, nn
from mindspore.common.parameter import Parameter
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

grad_all = C.GradOperation(get_all=True)
context.set_context(device_target="Ascend")

def test_single_for_01():
    class SingleForNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.mul = P.Mul()

        def construct(self, x, y, z):
            x = self.add(x, y)
            for _ in range(0, 3):
                z = self.add(z, x)
            y = self.mul(z, y)
            return y

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([2], mstype.int32)
    y = Tensor([5], mstype.int32)
    z = Tensor([4], mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_net_foward = SingleForNet()
    graph_forward_res = for_net_foward(x, y, z)

    for_net = SingleForNet()
    net = GradNet(for_net)
    graph_backward_res = net(x, y, z)

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    for_net_foward = SingleForNet()
    pynative_forward_res = for_net_foward(x, y, z)

    for_net = SingleForNet()
    net = GradNet(for_net)
    pynative_backward_res = net(x, y, z)

    assert graph_forward_res == pynative_forward_res
    assert graph_backward_res == pynative_backward_res


def test_single_for_02():
    class SingleForNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.mul = P.Mul()

        def construct(self, x, y, z):
            x = self.add(x, y)
            for _ in range(10, -5, -3):
                z = self.add(z, x)
            y = self.mul(z, y)
            return y

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([2], mstype.int32)
    y = Tensor([5], mstype.int32)
    z = Tensor([4], mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_net = SingleForNet()
    net = GradNet(for_net)
    graph_forward_res = for_net(x, y, z)
    graph_backward_res = net(x, y, z)

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    for_net = SingleForNet()
    net = GradNet(for_net)
    pynative_forward_res = for_net(x, y, z)
    pynative_backward_res = net(x, y, z)

    assert graph_forward_res == pynative_forward_res
    assert graph_backward_res == pynative_backward_res


def test_single_for_03():
    class SingleForNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = P.Mul()
            self.add = P.Add()
            self.sub = P.Sub()
            self.assign = P.Assign()
            param_a = np.full((1,), 5, dtype=np.float32)
            self.param_a = Parameter(Tensor(param_a), name='a')
            param_b = np.full((1,), 2, dtype=np.float32)
            self.param_b = Parameter(Tensor(param_b), name='b')

        def func(self, x):
            x = self.mul(x, 2)
            for _ in range(0, 5):
                x = self.add(x, x)
                self.param_b = self.param_b + 1
            return x - self.param_b

        def construct(self, x, y):
            self.assign(self.param_a, x + self.param_a)
            z = self.func(x)
            x = self.param_a + y + z
            return x, self.param_b

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([2], mstype.int32)
    y = Tensor([5], mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    single_for_net = SingleForNet()
    net = GradNet(single_for_net)
    graph_forward_res = single_for_net(x, y)
    graph_backward_res = net(x, y)

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    single_for_net = SingleForNet()
    net = GradNet(single_for_net)
    pynative_forward_res = single_for_net(x, y)
    pynative_backward_res = net(x, y)

    assert graph_forward_res == pynative_forward_res
    assert graph_backward_res == pynative_backward_res

@pytest.mark.skip(reason="not supported side effect")
def test_single_for_04():
    class SingleForNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = P.Mul()
            self.add = P.Add()
            self.sub = P.Sub()
            self.assign = P.Assign()
            param_a = np.full((1,), 5, dtype=np.float32)
            self.param_a = Parameter(Tensor(param_a), name='a')
            param_b = np.full((1,), 2, dtype=np.float32)
            self.param_b = Parameter(Tensor(param_b), name='b')

        def construct(self, x):
            self.assign(self.param_a, x + self.param_a)
            for _ in range(1):
                self.param_b = x - self.param_a
            return self.param_b

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([2], mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    single_for_net = SingleForNet()
    net = GradNet(single_for_net)
    graph_forward_res = single_for_net(x)
    graph_backward_res = net(x)

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    single_for_net = SingleForNet()
    net = GradNet(single_for_net)
    pynative_forward_res = single_for_net(x)
    pynative_backward_res = net(x)

    assert graph_forward_res == pynative_forward_res
    assert graph_backward_res == pynative_backward_res


def test_single_for_05():
    class SingleForNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = P.Mul()
            self.add = P.Add()
            self.sub = P.Sub()
            self.assign = P.Assign()
            self.param_a = Parameter(Tensor(np.array(5), mstype.int32), name='a')
            self.param_b = Parameter(Tensor(np.array(2), mstype.int32), name='b')

        def construct(self, x):
            self.assign(self.param_a, x + self.param_a)
            for _ in range(0, 3):
                self.assign(self.param_b, x - self.param_a)
            return x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([6], mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    single_for_net = SingleForNet()
    net = GradNet(single_for_net)
    graph_forward_res = single_for_net(x)
    graph_backward_res = net(x)

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    single_for_net = SingleForNet()
    net = GradNet(single_for_net)
    pynative_forward_res = single_for_net(x)
    pynative_backward_res = net(x)

    assert graph_forward_res == pynative_forward_res
    assert graph_backward_res == pynative_backward_res
