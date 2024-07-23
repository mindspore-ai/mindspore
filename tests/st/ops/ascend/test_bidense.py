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
# ============================================================================
from tests.mark_utils import arg_mark
import pytest
import torch
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit

from grad import GradOfAllInputsAndParams

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.bidense = nn.BiDense(20, 30, 40)

    @jit
    def construct(self, x1, x2):
        return self.bidense(x1, x2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net():
    """
    Feature: Assert BiDense output shape
    Description: test the output.shape == (128, 40).
    Expectation: match the shape.
    """
    x1 = np.random.randn(128, 20).astype(np.float32)
    x2 = np.random.randn(128, 30).astype(np.float32)
    net = Net()
    output = net(Tensor(x1), Tensor(x2))
    print(output.asnumpy())
    assert output.shape == (128, 40)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net_nd():
    """
    Feature: Assert BiDense output shape for n-dimensional input
    Description: test the output.shape == (128, 4, 40).
    Expectation: match the shape.
    """
    x1 = np.random.randn(128, 4, 20).astype(np.float32)
    x2 = np.random.randn(128, 4, 30).astype(np.float32)
    net = Net()
    output = net(Tensor(x1), Tensor(x2))
    print(output.asnumpy())
    assert output.shape == (128, 4, 40)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net_1d():
    """
    Feature: Assert BiDense output shape for 1-dimensional input
    Description: test the output.shape == (40,).
    Expectation: match the shape.
    """
    x1 = np.random.randn(20).astype(np.float32)
    x2 = np.random.randn(30).astype(np.float32)
    net = Net()
    output = net(Tensor(x1), Tensor(x2))
    print(output.asnumpy())
    assert output.shape == (40,)


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    nan_diff = np.not_equal(np.isnan(data_expected), np.isnan(data_me))
    inf_diff = np.not_equal(np.isinf(data_expected), np.isinf(data_me))
    neginf_diff = np.not_equal(np.isneginf(data_expected), np.isneginf(data_me))
    greater = greater + nan_diff + inf_diff + neginf_diff
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pynative_precision():
    """
    Feature: Test bidense ops precision
    Description: Compare the forward result and grad result between mindspore and torch.
    Expectation: assert pass.
    """
    # prepare input
    context.set_context(mode=context.PYNATIVE_MODE)
    dtype = np.float32
    input1_shape = (1024, 44)
    input2_shape = (1024, 55)
    in1_channel = 44
    in2_channel = 55
    out_channel = 32
    input1_np = np.random.randn(*input1_shape).astype(dtype)
    input2_np = np.random.randn(*input2_shape).astype(dtype)
    weight_np = np.random.randn(out_channel, in1_channel, in2_channel).astype(dtype)
    bias_np = np.random.randn(out_channel).astype(dtype)

    # calculate ms forward
    input_1 = Tensor(input1_np)
    input_2 = Tensor(input2_np)
    weight = Tensor(weight_np)
    bias = Tensor(bias_np)
    ms_net = nn.BiDense(in1_channel, in2_channel, out_channel, weight_init=weight,
                        bias_init=bias, has_bias=True)
    out = ms_net(input_1, input_2)
    ms_forward = out.asnumpy()

    # calculate torch forward
    input_1 = torch.from_numpy(input1_np.copy().astype(np.float32))
    input_2 = torch.from_numpy(input2_np.copy().astype(np.float32))
    weight = torch.from_numpy(weight_np.copy().astype(np.float32))
    bias = torch.from_numpy(bias_np.copy().astype(np.float32))
    torch_net = torch.nn.Bilinear(in1_channel, in2_channel, out_channel, bias=True)
    torch_net.register_parameter('weight', torch.nn.Parameter(weight))
    torch_net.register_parameter('bias', torch.nn.Parameter(bias))
    out = torch_net(input_1, input_2)
    torch_forward = out.detach().numpy().astype(dtype)

    # compare ms forward and torch forward
    loss = 7e-03
    allclose_nparray(torch_forward, ms_forward, loss, loss)

    # calculate torch grad
    weight_grad = torch.nn.Parameter(torch.from_numpy(weight_np.copy().astype(np.float32)))
    torch_net.register_parameter('weight', weight_grad)
    bias_grad = torch.nn.Parameter(torch.from_numpy(bias_np.copy().astype(np.float32)))
    torch_net.register_parameter('bias', bias_grad)
    input_1.requires_grad = True
    input_2.requires_grad = True
    output = torch_net(input_1, input_2)
    output_grad_np = np.ones(list(output.size())).astype(dtype)
    output_grad = torch.from_numpy(output_grad_np.copy().astype(np.float32))
    output.backward(gradient=output_grad)
    torch_grad = input_1.grad.detach().numpy(), input_2.grad.detach().numpy(), \
                   weight_grad.grad.detach().numpy(), bias_grad.grad.detach().numpy()

    # calculate ms grad
    output_grad = Tensor(output_grad_np)
    grad_net = GradOfAllInputsAndParams(ms_net)
    grad_net.set_train()
    input_1_grad = Tensor(input1_np)
    input_2_grad = Tensor(input2_np)
    input_grad = grad_net(input_1_grad, input_2_grad, output_grad)
    ms_grad = input_grad[0][0].asnumpy(), input_grad[0][1].asnumpy(), input_grad[1][
                0].asnumpy(), input_grad[1][1].asnumpy()

    allclose_nparray(torch_grad[0].astype(dtype), ms_grad[0], loss, loss)
    allclose_nparray(torch_grad[1].astype(dtype), ms_grad[1], loss, loss)
    allclose_nparray(torch_grad[2].astype(dtype), ms_grad[2], loss, loss)
