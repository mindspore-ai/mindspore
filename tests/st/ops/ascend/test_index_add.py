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

import numpy as np
import pytest

import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore.common import dtype as mstype


class NetIndexAdd(nn.Cell):
    def __init__(self, x, axis):
        super(NetIndexAdd, self).__init__()
        self.input_x = Parameter(Tensor(x), name='x')
        self.index_add = ops.IndexAdd(axis)

    def construct(self, idx, y):
        return self.index_add(self.input_x, idx, y)


def index_add_forward(nptype):
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(nptype)
    y = np.ones((2, 2, 4), dtype=nptype)
    idx = np.array([0, 2]).astype(np.int32)
    axis = 1
    expect = np.copy(x)
    expect[:, idx, :] = expect[:, idx, :] + y
    net = NetIndexAdd(x, axis)
    output = net(Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_index_add_float32():
    """
    Feature: test IndexAdd forward.
    Description: test float32 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    index_add_forward(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    index_add_forward(np.float32)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_index_add_float16():
    """
    Feature: test IndexAdd forward.
    Description: test float16 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    index_add_forward(np.float16)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    index_add_forward(np.float16)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_index_add_int32():
    """
    Feature: test IndexAdd forward.
    Description: test int32 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    index_add_forward(np.int32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    index_add_forward(np.int32)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_index_add_int8():
    """
    Feature: test IndexAdd forward.
    Description: test int8 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    index_add_forward(np.int8)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    index_add_forward(np.int8)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_index_add_uint8():
    """
    Feature: test IndexAdd forward.
    Description: test uint8 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    index_add_forward(np.uint8)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    index_add_forward(np.uint8)


class IndexAddGradNet(nn.Cell):
    def __init__(self, network):
        super(IndexAddGradNet, self).__init__()
        self.grad = ops.GradOperation(get_all=True, sens_param=True, get_by_list=True)
        self.network = network
        self.params = ParameterTuple(network.trainable_params())

    def construct(self, idx, y, dout):
        out = self.grad(self.network, self.params)(idx, y, dout)
        return out


def index_add_grad_with_type(nptype):
    x = np.arange(15).reshape(5, 3).astype(nptype)
    net = NetIndexAdd(x, 1)
    grad_net = IndexAddGradNet(net)
    y = Tensor(np.arange(5).reshape(5, 1).astype(nptype))
    dout = Tensor(np.array([[63., 64., 65.],
                            [66., 67., 68.],
                            [69., 70., 71.],
                            [72., 73., 74.],
                            [75., 76., 77.]]).astype(nptype))
    index = Tensor(np.array([1]), dtype=mindspore.int32)
    output = grad_net(index, y, dout)
    ygrad = output[0][1]
    xgrad = output[1][0]
    expect_xgrad = np.array([[63., 64., 65.],
                             [66., 67., 68.],
                             [69., 70., 71.],
                             [72., 73., 74.],
                             [75., 76., 77.]]).astype(nptype)
    expect_ygrad = np.array([[64.],
                             [67.],
                             [70.],
                             [73.],
                             [76.]]).astype(nptype)
    np.testing.assert_array_equal(xgrad.asnumpy(), expect_xgrad)
    np.testing.assert_array_equal(ygrad.asnumpy(), expect_ygrad)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_index_add_grad_float32():
    """
    Feature: test IndexAdd backward.
    Description: test float32 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    index_add_grad_with_type(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    index_add_grad_with_type(np.float32)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_index_add_grad_float16():
    """
    Feature: test IndexAdd backward.
    Description: test float16 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    index_add_grad_with_type(np.float16)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    index_add_grad_with_type(np.float16)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_index_add_grad_int32():
    """
    Feature: test IndexAdd backward.
    Description: test int32 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    index_add_grad_with_type(np.int32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    index_add_grad_with_type(np.int32)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_index_add_grad_int8():
    """
    Feature: test IndexAdd backward.
    Description: test int8 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    index_add_grad_with_type(np.int8)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    index_add_grad_with_type(np.int8)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_index_add_grad_uint8():
    """
    Feature: test IndexAdd backward.
    Description: test uint8 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    index_add_grad_with_type(np.uint8)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    index_add_grad_with_type(np.uint8)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_index_add_dynamic_y():
    """
    Feature: test IndexAdd dynamic shape.
    Description: input y is dynamic shape.
    Expectation: the result match with numpy result
    """
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
    y = np.ones((2, 2, 4), dtype=np.float32)
    idx = np.array([0, 2]).astype(np.int32)
    axis = 1
    expect = np.copy(x)
    expect[:, idx, :] = expect[:, idx, :] + y
    y_dyn = Tensor(shape=[2, None, 4], dtype=mindspore.float32)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = NetIndexAdd(x, axis)
    net.set_inputs(Tensor(idx), y_dyn)
    output = net(Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_index_add_dynamic_indices():
    """
    Feature: test IndexAdd dynamic shape.
    Description: input indices is dynamic shape.
    Expectation: the result match with numpy result
    """
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
    y = np.ones((2, 2, 4), dtype=np.float32)
    idx = np.array([0, 2]).astype(np.int32)
    axis = 1
    expect = np.copy(x)
    expect[:, idx, :] = expect[:, idx, :] + y
    idx_dyn = Tensor(shape=[None], dtype=mindspore.int32)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = NetIndexAdd(x, axis)
    net.set_inputs(idx_dyn, Tensor(y))
    output = net(Tensor(idx), Tensor(y))
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_index_add_tensor_api_modes(mode):
    """
    Feature: Test index_add tensor api.
    Description: Test index_add tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="Ascend")
    x = Parameter(Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mstype.float32), name="name_x")
    index = Tensor([0, 2], mstype.int32)
    source = Tensor([[0.5, 1.0], [1.0, 1.5], [2.0, 2.5]], mstype.float32)
    dim = 1
    output = x.index_add(dim, index, source)
    expected = np.array([[1.5, 2., 4.], [5., 5., 7.5], [9., 8., 11.5]], np.float32)
    np.testing.assert_array_equal(output.asnumpy(), expected)
