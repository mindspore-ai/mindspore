# Copyright 2025 Huawei Technologies Co., Ltd
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
""" test control ops """
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import context
from mindspore import nn
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import composite as C
from mindspore.ops import operations as P


context.set_context(jit_config={"jit_level": "O0"})
grad_by_list = C.GradOperation(get_by_list=True)


def test_if_const_grad():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()

        def construct(self, *inputs):
            out = self.add(*inputs)
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        def construct(self, *inputs):
            a = 1
            b = 2
            if a > 0:
                b = 1
            a += b
            return grad_by_list(self.net, self.weights)(*inputs)

    context.set_context(mode=context.GRAPH_MODE)
    my_net = MyNet()
    net = GradNet(my_net)
    a = Tensor(np.array(0), dtype=ms.int32)
    b = Tensor(np.array(1), dtype=ms.int32)
    net(a, b)


def test_if_by_if_const_grad():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()

        def construct(self, *inputs):
            out = self.add(*inputs)
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        def construct(self, *inputs):
            a = 1
            b = 2
            if a > 0:
                b = 1
            if a < 0:
                b = 0
            if a == 0:
                b = 3
            a += b
            return grad_by_list(self.net, self.weights)(*inputs)

    context.set_context(mode=context.GRAPH_MODE)
    my_net = MyNet()
    net = GradNet(my_net)
    a = Tensor(np.array(0), dtype=ms.int32)
    b = Tensor(np.array(1), dtype=ms.int32)
    net(a, b)


def test_while_const_grad():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()

        def construct(self, *inputs):
            out = self.add(*inputs)
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        def construct(self, *inputs):
            a = 1
            while a > 1:
                a = a - 1
            return grad_by_list(self.net, self.weights)(*inputs)

    context.set_context(mode=context.GRAPH_MODE)
    my_net = MyNet()
    net = GradNet(my_net)
    a = Tensor(np.array(0), dtype=ms.int32)
    b = Tensor(np.array(1), dtype=ms.int32)
    net(a, b)


def test_for_const_grad():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()

        def construct(self, *inputs):
            out = self.add(*inputs)
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        def construct(self, *inputs):
            a = 1
            for _ in (1,):
                a = a - 1
            return grad_by_list(self.net, self.weights)(*inputs)

    context.set_context(mode=context.GRAPH_MODE)
    my_net = MyNet()
    net = GradNet(my_net)
    a = Tensor(np.array(0), dtype=ms.int32)
    b = Tensor(np.array(1), dtype=ms.int32)
    net(a, b)


def test_switch_layer_const_grad():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()

        def construct(self, *inputs):
            out = self.add(*inputs)
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.layers = (self.net, self.net)
            self.weights = ParameterTuple(net.trainable_params())

        def construct(self, *inputs):
            if inputs[0][0][0] > 0:
                a = 1
            else:
                a = 0
            _ = self.layers[a](*inputs)
            return grad_by_list(self.net, self.weights)(*inputs)

    context.set_context(mode=context.GRAPH_MODE)
    my_net = MyNet()
    net = GradNet(my_net)
    a = Tensor(np.random.randn(2, 3), dtype=ms.int32)
    b = Tensor(np.random.randn(2, 3), dtype=ms.int32)
    net(a, b)


def test_if_by_while_const_grad():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()

        def construct(self, *inputs):
            out = self.add(*inputs)
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        def construct(self, *inputs):
            a = 1
            b = 2
            if a > 0:
                b = 0
            while a > 1:
                a = a - 1
            a += b
            return grad_by_list(self.net, self.weights)(*inputs)

    context.set_context(mode=context.GRAPH_MODE)
    my_net = MyNet()
    net = GradNet(my_net)
    a = Tensor(np.array(0), dtype=ms.int32)
    b = Tensor(np.array(1), dtype=ms.int32)
    net(a, b)
