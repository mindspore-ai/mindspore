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
from mindspore import context
from mindspore import Tensor, nn
from mindspore.common.parameter import Parameter
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

grad_all = C.GradOperation(get_all=True)
context.set_context(device_target="Ascend")


def test_for_in_if_01():
    class ForInIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = P.Mul()
            self.add = P.Add()
            param_a = np.full((1,), 5, dtype=np.float32)
            self.param_a = Parameter(Tensor(param_a), name='a')
            param_b = np.full((1,), 4, dtype=np.float32)
            self.param_b = Parameter(Tensor(param_b), name='b')

        def construct(self, x):
            if self.param_a > self.param_b:
                x = self.mul(x, 2)
                for _ in range(0, 5):
                    x = self.add(x, x)
                    self.param_b += 1
            return x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([10], mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_in_if_net = ForInIfNet()
    net = GradNet(for_in_if_net)
    graph_forward_res = for_in_if_net(x)
    graph_backward_res = net(x)

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    for_in_if_net = ForInIfNet()
    net = GradNet(for_in_if_net)
    pynative_forward_res = for_in_if_net(x)
    pynative_backward_res = net(x)

    assert graph_forward_res == pynative_forward_res
    assert graph_backward_res == pynative_backward_res


def test_for_in_if_02():
    class ForInIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = P.Mul()
            self.add = P.Add()
            param_a = np.full((1,), 5, dtype=np.float32)
            self.param_a = Parameter(Tensor(param_a), name='a')
            param_b = np.full((1,), 4, dtype=np.float32)
            self.param_b = Parameter(Tensor(param_b), name='b')

        def func(self, x):
            for _ in range(0, 5):
                x = self.add(x, x)
                self.param_b += 1
            return x

        def construct(self, x):
            if self.param_a > self.func(x):
                x = self.mul(x, 2)
                return x
            return x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([10], mstype.float32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_in_if_net = ForInIfNet()
    net = GradNet(for_in_if_net)
    graph_forward_res = for_in_if_net(x)
    graph_backward_res = net(x)

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    for_in_if_net = ForInIfNet()
    net = GradNet(for_in_if_net)
    pynative_forward_res = for_in_if_net(x)
    pynative_backward_res = net(x)

    assert graph_forward_res == pynative_forward_res
    assert graph_backward_res == pynative_backward_res


def test_for_in_if_03():
    class ForInIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = P.Mul()
            self.add = P.Add()
            param_a = np.full((1,), 5, dtype=np.float32)
            self.param_a = Parameter(Tensor(param_a), name='a')
            param_b = np.full((1,), 4, dtype=np.float32)
            self.param_b = Parameter(Tensor(param_b), name='b')

        def construct(self, x):
            y = x + self.param_b
            if self.param_a > self.param_b:
                x = self.mul(x, 2)
                for i in range(-1, 5):
                    x = self.add(i, x)
                    self.param_b += 1
            elif y > x:
                y = self.param_a * y
            else:
                x = self.param_b * x
            return x, y

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([10], mstype.float32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_in_if_net = ForInIfNet()
    net = GradNet(for_in_if_net)
    graph_forward_res = for_in_if_net(x)
    graph_backward_res = net(x)

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    for_in_if_net = ForInIfNet()
    net = GradNet(for_in_if_net)
    pynative_forward_res = for_in_if_net(x)
    pynative_backward_res = net(x)

    assert graph_forward_res == pynative_forward_res
    assert graph_backward_res == pynative_backward_res


def test_for_in_if_04():
    class ForInIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor(5, mstype.int32), name='a')
            self.param_b = Parameter(Tensor(4, mstype.int32), name='b')

        def construct(self, x):
            out = self.param_a
            x = self.func(x)
            out *= x
            return out

        def func(self, x):
            if self.param_a > self.param_b:
                for _ in range(0, 4):
                    self.param_a += 1
                    self.param_b -= 3
            self.param_b += 10
            return x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor(5, mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_in_if_net = ForInIfNet()
    net = GradNet(for_in_if_net)
    graph_forward_res = for_in_if_net(x)
    graph_backward_res = net(x)

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    for_in_if_net = ForInIfNet()
    net = GradNet(for_in_if_net)
    pynative_forward_res = for_in_if_net(x)
    pynative_backward_res = net(x)

    assert graph_forward_res == pynative_forward_res
    assert graph_backward_res == pynative_backward_res


def test_for_in_if_05():
    class ForInIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor(5, mstype.int32), name='a')
            self.param_b = Parameter(Tensor(4, mstype.int32), name='b')
            self.assign = P.Assign()

        def construct(self, x):
            out = self.param_a
            x = self.func(x)
            out *= x
            return out

        def func(self, x):
            if self.param_a > self.param_b:
                self.assign(self.param_a, self.param_b + self.param_a)
                for _ in range(0, 4):
                    self.param_a += 1
                    self.assign(self.param_b, self.param_b - 4)
            x += self.param_b
            return x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor(5, mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_in_if_net = ForInIfNet()
    net = GradNet(for_in_if_net)
    graph_forward_res = for_in_if_net(x)
    graph_backward_res = net(x)

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    for_in_if_net = ForInIfNet()
    net = GradNet(for_in_if_net)
    pynative_forward_res = for_in_if_net(x)
    pynative_backward_res = net(x)

    assert graph_forward_res == pynative_forward_res
    assert graph_backward_res == pynative_backward_res
