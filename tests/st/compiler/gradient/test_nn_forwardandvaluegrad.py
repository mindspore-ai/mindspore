# Copyright 2025 Huawei Technologies Co., Ltdallclose
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
"""test nn.ForwardValueAndGrad in graph mode"""
import numpy as np
import pytest
import torch
import torch.nn as nn_pt
from mindspore import nn, context, ops
from mindspore import jit, Tensor, Parameter, ParameterTuple, dtype
from mindspore.graph.api import _pynative_executor
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


def compare_result(ms_out, pt_out, get_all, get_list, loss=0.001):
    if isinstance(ms_out[0], Tensor):
        assert np.allclose(pt_out[0].detach().numpy(), ms_out[0].asnumpy(), loss, loss)
    else:
        for index, it in enumerate(ms_out[0]):
            assert np.allclose(pt_out[0][index].detach().numpy(), it.asnumpy(), loss, loss)

    if get_all and get_list:
        for index, it in enumerate(ms_out[1][0]):
            if index < len(pt_out[1][0]):
                assert np.allclose(pt_out[1][0][index].numpy(), it.asnumpy(), loss, loss)
        for index, it in enumerate(ms_out[1][1]):
            if not np.all(it.asnumpy() == 0):
                assert np.allclose(pt_out[1][1][index].numpy(), it.asnumpy(), loss, loss)

    if not get_all and not get_list:
        if ms_out[1] == ():
            assert True
        else:
            assert np.allclose(pt_out[1][0][0].numpy(), ms_out[1].asnumpy(), loss, loss)

    if not get_all and get_list:
        for index, it in enumerate(ms_out[1]):
            if not np.all(it.asnumpy() == 0):
                assert np.allclose(pt_out[1][1][index].numpy(), it.asnumpy(), loss, loss)

    if get_all and not get_list:
        for index, it in enumerate(ms_out[1]):
            if index < len(pt_out[1][0]):
                assert np.allclose(pt_out[1][0][index].numpy(), it.asnumpy(), loss, loss)


def extract_pt_input(inputs):
    input_pt_list = []
    for item in inputs:
        if isinstance(item, Tensor):
            input_pt = torch.from_numpy(item.asnumpy())
            if input_pt.dtype != torch.int32:
                input_pt.requires_grad = True
            input_pt_list.append(input_pt)
        elif isinstance(item, list):
            for list_it in item:
                if isinstance(list_it, Tensor):
                    input_pt = torch.from_numpy(list_it.asnumpy())
                    input_pt.requires_grad = False
                    input_pt_list.append(input_pt)
        elif isinstance(item, tuple):
            for tuple_it in item:
                if isinstance(tuple_it, Tensor):
                    input_pt = torch.from_numpy(tuple_it.asnumpy())
                    input_pt.requires_grad = False
                    input_pt_list.append(input_pt)
        elif isinstance(item, dict):
            for dict_it in item.values():
                if isinstance(dict_it, Tensor):
                    input_pt = torch.from_numpy(dict_it.asnumpy())
                    input_pt.requires_grad = False
                    input_pt_list.append(input_pt)
        elif isinstance(item, int):
            input_pt = torch.from_numpy(np.array(item))
            input_pt.requires_grad = False
            input_pt_list.append(input_pt)

    return input_pt_list


class GradFactory:
    def __init__(self, net_me, net_torch, get_all, get_by_list, sens_param, net_params=None,
                 default_para=False):
        self.net_me = net_me
        self.net_torch = net_torch
        self.get_all = get_all
        self.get_by_list = get_by_list
        self.sens_param = sens_param
        self.net_params = net_params
        self.default_para = default_para

    def get_grad(self, ms_input):
        output_grad_me = []
        output_grad_pt = []
        out = self.net_me(*ms_input)
        if isinstance(out, tuple):
            for it in out:
                if self.sens_param:
                    grad_np = np.random.randn(*it.shape).astype(np.float32)
                else:
                    grad_np = np.ones(it.shape).astype(np.float32)
                output_grad_me.append(Tensor(grad_np))
                output_grad_pt.append(torch.from_numpy(grad_np))
            output_grad_me = tuple(output_grad_me)
            output_grad_pt = tuple(output_grad_pt)
        else:
            if self.sens_param:
                grad_np = np.random.randn(*out.shape).astype(np.float32)
            else:
                grad_np = np.ones(out.shape).astype(np.float32)
            output_grad_me = Tensor(grad_np)
            output_grad_pt = torch.from_numpy(grad_np)
        return output_grad_me, output_grad_pt

    def grad_pytorch_impl(self, input_pt_list, output_grad):
        self.net_torch.zero_grad()
        out = self.net_torch(*input_pt_list)
        if torch.is_tensor(out):
            out.backward(gradient=output_grad)
        else:
            for i, ele in enumerate(out):
                ele.backward(gradient=output_grad[i], retain_graph=True)
        grad_pt = [[], []]
        for item in input_pt_list:
            if item.requires_grad:
                grad_pt[0].append(item.grad)
        for para in self.net_torch.parameters():
            grad_pt[1].append(para.grad)

        return out, grad_pt

    def one_backnet_call_twice(self, first_ms_input, second_ms_input, loss=0.001):
        grad_input = self.get_grad(first_ms_input)
        if self.default_para:
            back_net = nn.ForwardValueAndGrad(self.net_me)
            ms_out = back_net(*first_ms_input)
        else:
            if self.get_by_list:
                weight = self.net_params
            else:
                weight = None
            back_net = nn.ForwardValueAndGrad(self.net_me,
                                              weights=weight, get_all=self.get_all,
                                              get_by_list=self.get_by_list,
                                              sens_param=self.sens_param)
            if self.sens_param:
                ms_out = back_net(*first_ms_input, grad_input[0])
            else:
                ms_out = back_net(*first_ms_input)

        pt_input = extract_pt_input(first_ms_input)
        pt_out = self.grad_pytorch_impl(pt_input, grad_input[1])
        compare_result(ms_out, pt_out, self.get_all, self.get_by_list, loss)

        # second call
        grad_input = self.get_grad(second_ms_input)
        if self.default_para:
            ms_out = back_net(*second_ms_input)
        else:
            if self.sens_param:
                ms_out = back_net(*second_ms_input, grad_input[0])
            else:
                ms_out = back_net(*second_ms_input)
        pt_input = extract_pt_input(second_ms_input)
        pt_out = self.grad_pytorch_impl(pt_input, grad_input[1])
        compare_result(ms_out, pt_out, self.get_all, self.get_by_list)

    def two_backnet_call_twice(self, first_ms_input, second_ms_input, loss=0.001):
        grad_input = self.get_grad(first_ms_input)
        if self.default_para:
            back_net = nn.ForwardValueAndGrad(self.net_me)
            ms_out = back_net(*first_ms_input)
        else:
            if self.get_by_list:
                weight = self.net_params
            else:
                weight = None
            back_net = nn.ForwardValueAndGrad(self.net_me,
                                              weights=weight, get_all=self.get_all,
                                              get_by_list=self.get_by_list,
                                              sens_param=self.sens_param)
            if self.sens_param:
                ms_out = back_net(*first_ms_input, grad_input[0])
            else:
                ms_out = back_net(*first_ms_input)
        pt_input = extract_pt_input(first_ms_input)
        pt_out = self.grad_pytorch_impl(pt_input, grad_input[1])
        compare_result(ms_out, pt_out, self.get_all, self.get_by_list, loss)

        # second call
        grad_input = self.get_grad(second_ms_input)
        if self.default_para:
            back_net2 = nn.ForwardValueAndGrad(self.net_me)
            ms_out = back_net2(*second_ms_input)
        else:
            back_net2 = nn.ForwardValueAndGrad(self.net_me,
                                               weights=weight, get_all=self.get_all,
                                               get_by_list=self.get_by_list,
                                               sens_param=self.sens_param)
            if self.sens_param:
                ms_out = back_net2(*second_ms_input, grad_input[0])
            else:
                ms_out = back_net2(*second_ms_input)
        pt_input = extract_pt_input(second_ms_input)
        pt_out = self.grad_pytorch_impl(pt_input, grad_input[1])
        compare_result(ms_out, pt_out, self.get_all, self.get_by_list)

    def first_forward_second_backnet(self, first_ms_input, second_ms_input, loss=0.001):
        forward_ms_out = self.net_me(*first_ms_input)
        pt_input = extract_pt_input(first_ms_input)
        forward_pt_out = self.net_torch(*pt_input)
        if isinstance(forward_ms_out, Tensor):
            assert np.allclose(forward_pt_out.detach().numpy(), forward_ms_out.asnumpy(), loss, loss)
        else:
            for index, it in enumerate(forward_ms_out):
                assert np.allclose(forward_pt_out[index].detach().numpy(), it.asnumpy(), loss, loss)

        # second call
        grad_input = self.get_grad(second_ms_input)
        if self.default_para:
            back_net2 = nn.ForwardValueAndGrad(self.net_me)
            ms_out = back_net2(*second_ms_input)
        else:
            if self.get_by_list:
                weight = self.net_params
            else:
                weight = None
            back_net2 = nn.ForwardValueAndGrad(self.net_me,
                                               weights=weight, get_all=self.get_all,
                                               get_by_list=self.get_by_list,
                                               sens_param=self.sens_param)
            if self.sens_param:
                ms_out = back_net2(*second_ms_input, grad_input[0])
            else:
                ms_out = back_net2(*second_ms_input)
        pt_input = extract_pt_input(second_ms_input)
        pt_out = self.grad_pytorch_impl(pt_input, grad_input[1])
        compare_result(ms_out, pt_out, self.get_all, self.get_by_list)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("sens_param", [False, True])
@pytest.mark.parametrize("get_by_list", [False])
@pytest.mark.parametrize("get_all", [False, True])
def test_forward_value_and_grad_003(get_all, get_by_list, sens_param):
    """
    Feature: Test value and grad under graph mode
    Description: Test the code for value and grad under graph mode
    Expectation: success
    """
    def tensor_add(x, y):
        add = ops.Add()
        z = add(x, y)
        return z

    class NetPytorch(nn_pt.Module):
        def forward(self, x, y):
            z = torch.add(x, y)
            return z

    tensor_add_torch = NetPytorch()
    fact = GradFactory(net_me=tensor_add,
                       net_torch=tensor_add_torch,
                       get_all=get_all,
                       get_by_list=get_by_list,
                       sens_param=sens_param)

    input_1 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    input_2 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    first_input = (input_1, input_2)
    input_1 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    input_2 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    second_input = (input_1, input_2)
    fact.one_backnet_call_twice(first_input, second_input)

    input_3 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    input_4 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    first_input_2 = (input_3, input_4)
    input_3 = Tensor(np.random.randn(1, 2, 3, 4).astype(np.float32))
    input_4 = Tensor(np.random.randn(1, 2, 3, 4).astype(np.float32))
    second_input_2 = (input_3, input_4)
    fact.two_backnet_call_twice(first_input_2, second_input_2)
    fact.first_forward_second_backnet(first_input_2, second_input_2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_forward_value_and_grad_008():
    """
    Feature: Test value and grad under graph mode
    Description: Test the code for value and grad under graph mode
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()

            self.para = Parameter(Tensor(1, dtype.float32), name="para")

        def construct(self, z):
            x = self.para * self.para
            y = self.para - x
            return y

    input_1 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    net_me = Net()
    back_net = nn.ForwardValueAndGrad(net_me)
    out = back_net(input_1)
    assert np.all(out[1].asnumpy() == 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("get_all, get_by_list, sens_param",
                         [(True, False, False), (False, False, False)])
def test_forward_value_and_grad_022(get_all, get_by_list, sens_param):
    """
    Feature: Test value and grad under graph mode
    Description: Test the code for value and grad under graph mode
    Expectation: success
    """
    class Net(nn.Cell):
        def construct(self, x):
            x = x + x
            return x

    class NetPytorch(nn_pt.Module):
        def forward(self, x):
            x = x + x
            return x

    net_me = Net()
    net_torch = NetPytorch()
    fact = GradFactory(net_me=net_me,
                       net_torch=net_torch,
                       get_all=get_all,
                       get_by_list=get_by_list,
                       sens_param=sens_param)

    input_1 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    first_input = (input_1,)

    input_1 = Tensor(np.random.randn(1, 2, 3, 4).astype(np.float32))
    second_input = (input_1,)
    fact.one_backnet_call_twice(first_input, second_input)
    fact.two_backnet_call_twice(first_input, second_input)
    fact.first_forward_second_backnet(first_input, second_input)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_forward_value_and_grad_023():
    """
    Feature: Test value and grad under graph mode
    Description: Test the code for value and grad under graph mode
    Expectation: success
    """
    class Net(nn.Cell):
        def construct(self, x):
            x = x + x
            return x

    input_1 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    net_me = Net()
    with pytest.raises(TypeError):
        back_net = nn.ForwardValueAndGrad(net_me, get_all=True, get_by_list=True, sens_param=False)
        back_net(input_1)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_forward_value_and_grad_034():
    """
    Feature: Test value and grad under graph mode
    Description: Test the code for value and grad under graph mode
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self, a, b):
            super().__init__()

            self.a = a
            self.b = b

        def construct(self):
            x = self.a * self.b
            return x

    input_1 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    input_2 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    net_me = Net(input_1, input_2)
    back_net = nn.ForwardValueAndGrad(net_me)
    out = back_net()
    assert out[1] == ()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_forward_value_and_grad_040():
    """
    Feature: Test value and grad under graph mode
    Description: Test the code for value and grad under graph mode
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.para = Parameter(Tensor(1, dtype.float32), name="para")
            self.para2 = Parameter(Tensor(2, dtype.float32), name="para2")

        def construct(self, x):
            if x.ndim > 2:
                y = x * self.para
            else:
                y = self.para2 - x
            return y

    class NetPytorch(nn_pt.Module):
        def __init__(self):
            super().__init__()
            self.para = nn_pt.Parameter(torch.from_numpy(np.array(1).astype(np.float32)))
            self.register_parameter('para', self.para)
            self.para2 = nn_pt.Parameter(torch.from_numpy(np.array(2).astype(np.float32)))
            self.register_parameter('para2', self.para2)

        def forward(self, x):
            if x.dim() > 2:
                y = x * self.para
            else:
                y = self.para2 - x
            return y

    net_me = Net()
    net_torch = NetPytorch()
    fact = GradFactory(net_me=net_me,
                       net_torch=net_torch,
                       get_all=False,
                       get_by_list=False,
                       sens_param=False,
                       default_para=True)

    input_1 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    first_input = (input_1,)

    input_1 = Tensor(np.random.randn(2, 3).astype(np.float32))
    second_input = (input_1,)
    fact.one_backnet_call_twice(first_input, second_input)
    fact.two_backnet_call_twice(first_input, second_input)
    fact.first_forward_second_backnet(first_input, second_input)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_forward_value_and_grad_058():
    """
    Feature: Test value and grad under graph mode
    Description: Test the code for value and grad under graph mode
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()

            self.relu = nn.ReLU()

        def construct(self, x):
            y = self.relu(x)
            return y

        def bprop(self, x, out, dout):
            y = self.relu(x)
            grads = y * dout
            expand = ops.ExpandDims()
            squezze0 = ops.Squeeze(0)
            squezze1 = ops.Squeeze(1)
            if x[0, 0, 0] > 0:
                out = expand(grads, 0)
            else:
                out = expand(grads, 1)
            if x[0, 0, 0] > 0:
                out = squezze0(out)
            else:
                out = squezze1(out)
            return (out,)

    net_me = Net()
    input_np = np.array([[[-2, 3, 4, 5]]]).astype(np.float32)
    input_me = Tensor(input_np)
    grad_np = np.random.randn(*net_me(input_me).shape).astype(np.float32)
    output_grad_me = Tensor(grad_np)

    back_net = nn.ForwardValueAndGrad(net_me, get_all=True, sens_param=True)
    out_me = back_net(input_me, output_grad_me)

    out_me2 = net_me.bprop(input_me, None, output_grad_me)
    assert (out_me[1][0].asnumpy() == out_me2[0].asnumpy()).all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("sens_param", [True, False])
@pytest.mark.parametrize("get_by_list", [True, False])
@pytest.mark.parametrize("get_all", [True, False])
def test_forward_value_and_grad_060(get_all, get_by_list, sens_param):
    """
    Feature: Test value and grad under graph mode
    Description: Test the code for value and grad under graph mode
    Expectation: success
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()

            self.para = Parameter(Tensor(1, dtype.float32), name="para")

        @jit
        def construct(self, x):
            x = self.para * x
            return x

    class NetPytorch(nn_pt.Module):
        def __init__(self):
            super().__init__()
            self.para = nn_pt.Parameter(torch.from_numpy(np.array(1).astype(np.float32)))
            self.register_parameter('para', self.para)

        def forward(self, x):
            x = self.para * x
            return x

    net_me = Net()
    net_torch = NetPytorch()
    fact = GradFactory(net_me=net_me,
                       net_torch=net_torch,
                       get_all=get_all,
                       get_by_list=get_by_list,
                       sens_param=sens_param,
                       net_params=ParameterTuple(net_me.trainable_params()))

    input_1 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    first_input = (input_1,)
    input_1 = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    second_input = (input_1,)
    fact.one_backnet_call_twice(first_input, second_input)
    fact.two_backnet_call_twice(first_input, second_input)
    fact.first_forward_second_backnet(first_input, second_input)
