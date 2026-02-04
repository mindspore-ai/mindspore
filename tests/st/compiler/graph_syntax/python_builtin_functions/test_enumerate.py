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
""" test enumerate"""
from functools import partial
import copy
import torch
import numpy as np

from mindspore import nn, context, Tensor
from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfFirstInput
from tests.st.compiler.utils import OpsFactory, allclose_nparray

context.set_context(mode=context.GRAPH_MODE)


class NetFactory(OpsFactory):
    def __init__(self, net, input_shape):
        super().__init__()
        self.net_me = net
        self.net_me1 = copy.deepcopy(net)
        self.net_pt = torch.nn.ReLU()
        self.input_np = np.random.randn(*input_shape).astype(np.float32)
        self.out_np = np.random.randn(*input_shape).astype(np.float32)

    def forward_mindspore_impl(self):
        input_me = Tensor(self.input_np)
        out = self.net_me(input_me)
        return out.asnumpy()

    def forward_pytorch_impl(self):
        input_pt = torch.from_numpy(self.input_np)
        out = self.net_pt(input_pt)
        return out.numpy()

    def grad_mindspore_impl(self):
        input_me = Tensor(self.input_np)
        output_grad = Tensor(self.out_np)
        grad_net = GradOfFirstInput(self.net_me1)
        grad_net.set_train()
        input_grad = grad_net(input_me, output_grad)
        return input_grad.asnumpy()

    def grad_pytorch_impl(self):
        input_pt = torch.from_numpy(self.input_np.copy())
        output_grad = torch.from_numpy(self.out_np.copy())
        input_pt.requires_grad = True
        out = self.net_pt(input_pt)
        out.backward(gradient=output_grad)
        return input_pt.grad.numpy()

    def forward_cmp(self):
        out_pytorch = self.forward_pytorch_impl()
        out_mindspore = self.forward_mindspore_impl()
        allclose_nparray(out_pytorch, out_mindspore, self.loss, self.loss)

    def grad_cmp(self):
        input_grad_pytorch = self.grad_pytorch_impl()
        input_grad_mindspore = self.grad_mindspore_impl()
        allclose_nparray(input_grad_pytorch,
                         input_grad_mindspore, self.loss, self.loss)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parser_enumerate_004():
    """
    Feature: Enumerate
    Description: Test enumerate with for loop
    Expectation: the result match expectation
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def construct(self, x):
            l = []
            l.append(x)
            l.append(x)
            for index, value in enumerate(l, start=10):
                if index == 11:
                    x = self.relu(value)
            return x

    fact = NetFactory(Net(), (2, 3, 4, 5))
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parser_enumerate_005():
    """
    Feature: Enumerate
    Description: Test enumerate with for loop and item
    Expectation: the result match expectation
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def construct(self, x):
            l = []
            l.append(x)
            l.append(x)
            for item in enumerate(l, start=10):
                x = self.relu(item[1])
            return x

    fact = NetFactory(Net(), (2, 3, 4, 5))
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parser_enumerate_012():
    """
    Feature: Enumerate
    Description: Test enumerate with nested for loop
    Expectation: the result match expectation
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def construct(self, x):
            l1 = []
            l2 = []
            l1.append(x)
            l1.append(x)
            l2.append(l1)
            l2.append(l1)
            for index1, value1 in enumerate(l2):
                for index2, value2 in enumerate(value1):
                    if index1 == 1 and index2 == 1:
                        x = self.relu(value2)
            return x

    fact = NetFactory(Net(), (2, 3, 4, 5))
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parser_enumerate_013():
    """
    Feature: Enumerate
    Description: Test enumerate with for loop and control flow
    Expectation: the result match expectation
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def construct(self, x):
            l = []
            while True:
                if len(l) < 2:
                    l.append(x)
                else:
                    for index, value in enumerate(l):
                        if index == 1:
                            x = self.relu(value)
                    break
            return x

    fact = NetFactory(Net(), (2, 3, 4, 5))
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parser_enumerate_and_partial():
    """
    Feature: Enumerate
    Description: Test enumerate with partial
    Expectation: the result match expectation
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.start = 10

        def construct(self, x):
            l = (x, x)
            f = partial(enumerate, l)
            for index, value in f(self.start):
                if index == 11:
                    return self.relu(value)
            return 0

    fact = NetFactory(Net(), (2, 3, 4, 5))
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parser_enumerate_and_len():
    """
    Feature: Enumerate
    Description: Test enumerate with len
    Expectation: the result match expectation
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def construct(self, x):
            l = []
            l.append(x)
            l.append(x)
            for index, value in enumerate(l, start=len(l)):
                if index == 3:
                    x = self.relu(value)
            return x

    fact = NetFactory(Net(), (2, 3, 4, 5))
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parser_enumerate_and_map():
    """
    Feature: Enumerate
    Description: Test enumerate with map
    Expectation: the result match expectation
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def construct(self, x):
            l1 = []
            l2 = []
            l1.append(x)
            l1.append(x)
            l2.append(l1)
            for item in map(enumerate, l2):
                for index, value in item:
                    if index == 1:
                        x = self.relu(value)
            return x

    fact = NetFactory(Net(), (2, 3, 4, 5))
    fact.forward_cmp()
    fact.grad_cmp()
