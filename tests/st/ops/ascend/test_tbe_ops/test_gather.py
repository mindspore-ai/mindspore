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
from mindspore import Tensor
from mindspore.ops import operations as P


class GatherTensorTestNet(nn.Cell):
    def construct(self, x, indices, axis):
        return x.gather(indices, axis)


class GatherNet(nn.Cell):
    def __init__(self):
        super(GatherNet, self).__init__()
        self.gather = P.Gather()

    def construct(self, input_x, indices, axis):
        return self.gather(input_x, indices, axis)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tensor_graph_mode():
    """
    Feature: gather tensor test on graph mode.
    Description: test gather tensor's interface on graph mode.
    Expectation: the result equal to expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    input_params = Tensor(np.array([1, 2, 3, 4, 5, 6, 7]), mindspore.float32)
    input_indices = Tensor(np.array([0, 2, 4, 2, 6]), mindspore.int32)
    axis = 0
    net = GatherTensorTestNet()
    output = net(input_params, input_indices, axis)
    expect_np = np.array([1., 3., 5., 3., 7.])
    rtol = 1.e-4
    atol = 1.e-4
    assert np.allclose(output.asnumpy(), expect_np, rtol, atol, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tensor_pynative_mode():
    """
    Feature: gather tensor test on pynative mode.
    Description: test gather tensor's interface on pynative mode.
    Expectation: the result equal to expect.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    input_params = Tensor(np.array([1, 2, 3, 4, 5, 6, 7]), mindspore.float32)
    input_indices = Tensor(np.array([0, 2, 4, 2, 6]), mindspore.int32)
    axis = 0
    net = GatherTensorTestNet()
    output = net(input_params, input_indices, axis)
    expect_np = np.array([1., 3., 5., 3., 7.])
    rtol = 1.e-4
    atol = 1.e-4
    assert np.allclose(output.asnumpy(), expect_np, rtol, atol, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_functional_pynative_mode():
    """
    Feature: gather functional test on pynative mode.
    Description: test gather_nd functional's interface on pynative mode.
    Expectation: the result equal to expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    input_params = Tensor(np.array([1, 2, 3, 4, 5, 6, 7]), mindspore.float32)
    input_indices = Tensor(np.array([0, 2, 4, 2, 6]), mindspore.int32)
    axis = 0
    output = ops.gather(input_params, input_indices, axis)
    expect_np = np.array([1., 3., 5., 3., 7.])
    rtol = 1.e-4
    atol = 1.e-4
    assert np.allclose(output.asnumpy(), expect_np, rtol, atol, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_functional_graph_mode():
    """
    Feature: gather functional test on graph mode.
    Description: test gather functional's interface on graph mode.
    Expectation: the result equal to expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    input_params = Tensor(np.array([1, 2, 3, 4, 5, 6, 7]), mindspore.float32)
    input_indices = Tensor(np.array([0, 2, 4, 2, 6]), mindspore.int32)
    axis = 0
    output = ops.gather(input_params, input_indices, axis)
    expect_np = np.array([1., 3., 5., 3., 7.])
    rtol = 1.e-4
    atol = 1.e-4
    assert np.allclose(output.asnumpy(), expect_np, rtol, atol, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gather_static_pynative_mode():
    """
    Feature: gather static shape test on pynative mode.
    Description: test static shape for gather on pynative mode.
    Expectation: the result equal to expect.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    input_params = Tensor(np.array([1, 2, 3, 4, 5, 6, 7]), mindspore.float32)
    input_indices = Tensor(np.array([0, 2, 4, 2, 6]), mindspore.int32)
    axis = 0
    net = GatherNet()
    output = net(input_params, input_indices, axis)
    expect_np = np.array([1., 3., 5., 3., 7.])
    rtol = 1.e-4
    atol = 1.e-4
    assert np.allclose(output.asnumpy(), expect_np, rtol, atol, equal_nan=True)
