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
""" test partial"""
from functools import partial
import copy
import pytest
import torch
import numpy as np

from mindspore import nn, context, Tensor
from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfAllInputs, GradOfFirstInput
from tests.st.compiler.utils import OpsFactory, allclose_nparray

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_partial_key_ward_arg_and_pos_arg_const_multi_assign_x():
    """
    Feature: ALL TO ALL
    Description: test cases for partial_key_ward_arg_and_pos_arg_const_multi_assign_x
    Expectation: the result match given one
    """

    class Net(nn.Cell):
        def show(self, x, y, z):
            return x, y, z

        def construct(self):
            f = partial(self.show, x=1)
            ret = f(1, 2, 3)
            return ret

    class Net2(nn.Cell):
        def __init__(self):
            super().__init__()
            self.show = lambda x, y, z: (x, y, z)

        def construct(self):
            f = partial(self.show, x=1)
            ret = f(1, 2, 3)
            return ret

    for net in [Net(), Net2()]:
        with pytest.raises(TypeError) as ex:
            net()
        assert "got multiple values for argument 'x'" in str(ex.value) or \
               "Multiply values for specific argument: x" in str(ex.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_partial_key_ward_arg_and_pos_arg_const_multi_assign_y():
    """
    Feature: ALL TO ALL
    Description: test cases for partial_key_ward_arg_and_pos_arg_const_multi_assign_y
    Expectation: the result match given one
    """

    class Net(nn.Cell):
        def show(self, x, y, z):
            return x, y, z

        def construct(self):
            f = partial(self.show, y=2)
            ret = f(1, 2, z=3)
            return ret

    class Net2(nn.Cell):
        def __init__(self):
            super().__init__()
            self.show = lambda x, y, z: (x, y, z)

        def construct(self):
            f = partial(self.show, y=2)
            ret = f(1, 2, z=3)
            return ret

    for net in [Net(), Net2()]:
        with pytest.raises(TypeError) as ex:
            net()
        assert "got multiple values for argument 'y'" in str(ex.value) or \
               "Multiply values for specific argument: y" in str(ex.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_partial_key_ward_arg_and_pos_arg_const_multi_assign_z():
    """
    Feature: ALL TO ALL
    Description: test cases for partial_key_ward_arg_and_pos_arg_const_multi_assign_z
    Expectation: the result match given one
    """

    class Net(nn.Cell):
        def show(self, x, y, z):
            return x, y, z

        def construct(self):
            f = partial(self.show, z=1)
            ret = f(1, 2, 3)
            return ret

    class Net2(nn.Cell):
        def __init__(self):
            super().__init__()
            self.show = lambda x, y, z: (x, y, z)

        def construct(self):
            f = partial(self.show, z=1)
            ret = f(1, 2, 3)
            return ret

    for net in [Net(), Net2()]:
        with pytest.raises(TypeError) as ex:
            net()
        assert "got multiple values for argument 'z'" in str(ex.value) or \
               "Multiply values for specific argument: z" in str(ex.value)


class ParserFactory(OpsFactory):
    def __init__(self, net_me, net_torch, *inputs):
        super().__init__()
        self._input_num = len(inputs)
        self.net_me = net_me
        self.net_me.set_grad()

        self.net_torch = net_torch

        self.input_me_list = []
        self.input_pt_list = []

        for item in inputs:
            self._input_me = Tensor(item)
            self.input_me_list.append(self._input_me)

            self._input_pt = torch.from_numpy(item)
            self._input_pt.requires_grad = True

            self.input_pt_list.append(self._input_pt)

        self.out_np_shape = self.forward_mindspore_impl().shape
        if not self.out_np_shape:
            self.out_np = np.array(1).astype(np.float32)
        else:
            self.out_np = np.random.randn(
                *self.out_np_shape).astype(np.float32)
        self.output_grad_me = Tensor(self.out_np)
        self.output_grad_pt = torch.from_numpy(self.out_np)

    def grad_mindspore_impl(self):
        grad_func = GradOfFirstInput if self._input_num == 1 else GradOfAllInputs
        grad_net = grad_func(self.net_me)
        grad_net.set_train()
        input_me_use_list = copy.deepcopy(self.input_me_list)
        grad_ms = grad_net(*input_me_use_list, self.output_grad_me)
        return grad_ms

    def grad_pytorch_impl(self):
        output_grad = torch.from_numpy(self.out_np.copy())
        input_pt_use_list = copy.deepcopy(self.input_pt_list)
        out = self.net_torch(*input_pt_use_list)
        out.backward(gradient=output_grad)
        grad_pt = []
        for item in input_pt_use_list:
            grad_pt.append(item.grad)
        return grad_pt

    def backward_cmp(self):
        grad_pt = self.grad_pytorch_impl()
        grad_ms = self.grad_mindspore_impl()
        for i in range(self._input_num):
            _grad_ms = grad_ms if self._input_num == 1 else grad_ms[i]
            input_grad_mindspore = _grad_ms.asnumpy()

            input_grad_pytorch = grad_pt[i].numpy()
            allclose_nparray(input_grad_pytorch,
                             input_grad_mindspore, self.loss, self.loss)

    def forward_mindspore_impl(self):
        input_me_use_list = copy.deepcopy(self.input_me_list)
        output_me = self.net_me(*input_me_use_list)
        return output_me


class NetPartial0011(nn.Cell):

    @staticmethod
    def calcu(x, y=2.5):
        return x - y

    def construct(self, x):
        f = partial(self.calcu)
        return f(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parser_partial_0011():
    """
    Feature: Partial
    Description: test partial with default input
    Expectation: the result match expectation
    """
    class NetPt(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        @staticmethod
        def forward(x):
            return torch.sub(x, 2.5)

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    net = NetPartial0011()
    net_pt = NetPt()
    fact = ParserFactory(net, net_pt, input_np_x)
    fact.backward_cmp()


class NetPartial0014(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    @staticmethod
    def item(items, i):
        return items[i]

    def construct(self, x):
        l = []
        l.append(self.relu)
        f = partial(self.item, l)
        return f(0)(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parser_partial_0014():
    """
    Feature: Partial
    Description: test partial with list input
    Expectation: the result match expectation
    """
    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)

    net = NetPartial0014()
    # grad
    net_pt = torch.nn.ReLU()
    fact = ParserFactory(net, net_pt, input_np_x)
    fact.backward_cmp()


class NetPartial0016(nn.Cell):
    def __init__(self, y, z):
        super().__init__()
        self.y = y
        self.z = z

    @staticmethod
    def calcu(x, y, z):
        return x - y + z

    def construct(self, x):
        f = partial(self.calcu, x)
        return f(self.y, self.z)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parser_partial_0016():
    """
    Feature: Partial
    Description: test partial with tensor and scalar inputs
    Expectation: the result match expectation
    """
    class NetPt(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        @staticmethod
        def forward(x):
            return torch.add(torch.sub(x, 2.5), 1.1)

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_np_y = 2.5
    input_np_z = 1.1
    net = NetPartial0016(input_np_y, input_np_z)
    # grad
    net_pt = NetPt()
    fact = ParserFactory(net, net_pt, input_np_x)
    fact.backward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parser_partial_0021_1():
    """
    Feature: Partial
    Description: test partial with pass one input multiple times
    Expectation: the result match expectation
    """
    class Net(nn.Cell):

        @staticmethod
        def calcu(x, y):
            return x - y

        def construct(self):
            f = partial(self.calcu, y=2)
            return f(10, 3)

    net = Net()
    with pytest.raises(TypeError):
        net()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parser_partial_0023_1():
    """
    Feature: Partial
    Description: test partial with pass one input multiple times
    Expectation: the result match expectation
    """
    class Net(nn.Cell):

        @staticmethod
        def calcu(x, y):
            return x - y

        def construct(self, x):
            f = partial(self.calcu, y=2)
            return f(x, 2)

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net()
    with pytest.raises(TypeError):
        net(input_me)
