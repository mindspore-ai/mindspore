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

import numpy as np
import pytest
import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter, ParameterTuple

# all cases tested against dchip


class TestScatterMinNet(nn.Cell):
    def __init__(self, inputx):
        super(TestScatterMinNet, self).__init__()

        self.scatter_min = ops.ScatterMin()
        self.inputx = Parameter(inputx, name="inputx")

    def construct(self, indices, updates):
        out = self.scatter_min(self.inputx, indices, updates)
        return out


def scatter_min_forward(nptype):
    inputx = Tensor(np.arange(0, 9).reshape((3, 3)).astype(nptype))
    indices = Tensor(np.array([[[1, 0, 2], [2, 2, 0]], [[1, 0, 1], [2, 1, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(34, 70).reshape((2, 2, 3, 3)).astype(nptype))

    net = TestScatterMinNet(inputx)
    output = net(indices, updates)
    expected = inputx.asnumpy()
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def scatter_min_dynamic_updates():
    inputx = Tensor(np.ones((4, 2, 3, 4)).astype(np.float32))
    indices = Tensor(np.array([[0, 2], [3, 1]]).astype(np.int32))
    updates = Tensor(np.arange(96).reshape((2, 2, 2, 3, 4)).astype(np.float32))
    updates_dy = Tensor(shape=(2, 2, 2, None, 4), dtype=mindspore.float32)

    net = TestScatterMinNet(inputx)
    net.set_inputs(indices, updates_dy)
    output = net(indices, updates)
    expected = np.ones((4, 2, 3, 4)).astype(np.float32)
    expected[0][0][0][0] = 0.0
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def scatter_min_dynamic_indices():
    inputx = Tensor(np.ones((4, 2, 3, 4)).astype(np.int32))
    indices = Tensor(np.array([[0, 2], [3, 1]]).astype(np.int32))
    indices_dy = Tensor(shape=(2, None), dtype=mindspore.int32)
    updates = Tensor(np.arange(96).reshape((2, 2, 2, 3, 4)).astype(np.int32))

    net = TestScatterMinNet(inputx)
    net.set_inputs(indices_dy, updates)
    output = net(indices, updates)
    expected = np.ones((4, 2, 3, 4)).astype(np.int32)
    expected[0][0][0][0] = 0
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


class TestScatterMinGradNet(nn.Cell):
    def __init__(self, network):
        super(TestScatterMinGradNet, self).__init__()
        self.grad = ops.GradOperation(get_all=True, sens_param=True, get_by_list=True)
        self.network = network
        self.params = ParameterTuple(network.trainable_params())

    def construct(self, indices, updates, dout):
        out = self.grad(self.network, self.params)(indices, updates, dout)
        return out


def scatter_min_grad(nptype):
    inputx = Tensor(np.flip(np.arange(34, 46).reshape(3, 4).astype(nptype)))
    indices = Tensor(np.array([[[0, 1, 2], [2, 1, 0]], [[0, 0, 0], [2, 2, 2]]]).astype(np.int32))
    updates = Tensor(np.arange(63, 111).reshape((2, 2, 3, 4)).astype(nptype))
    dout = Tensor(np.flip(np.arange(0, 12).reshape((3, 4)).astype(nptype)))

    net = TestScatterMinGradNet(TestScatterMinNet(inputx))
    output = net(indices, updates, dout)
    indices_grad = output[0][0]
    updates_grad = output[0][1]

    indices_expected = np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]).astype(nptype)
    updates_expected = np.array(
        [
            [
                [
                    [11, 10, 9, 8], [7, 6, 5, 4], [3, 2, 1, 0]
                ],
                [
                    [3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]
                ]
            ],
            [
                [
                    [11, 10, 9, 8], [11, 10, 9, 8], [11, 10, 9, 8]
                ],
                [
                    [3, 2, 1, 0], [3, 2, 1, 0], [3, 2, 1, 0]
                ]
            ]
        ]).astype(nptype)
    np.testing.assert_array_almost_equal(indices_grad, indices_expected)
    np.testing.assert_array_almost_equal(updates_grad, updates_expected)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_min_forward_float16():
    """
    Feature: test scatter_min forward.
    Description: test float16 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_min_forward(np.float16)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_min_forward(np.float16)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_min_forward_float32():
    """
    Feature: test scatter_min forward.
    Description: test float32 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_min_forward(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_min_forward(np.float32)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_min_forward_int32():
    """
    Feature: test scatter_min forward.
    Description: test int32 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_min_forward(np.int32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_min_forward(np.int32)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_min_dynamic_indices():
    """
    Feature: test scatter_min dynamic shape.
    Description: indices is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_min_dynamic_indices()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_min_dynamic_indices()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_min_dynamic_updates():
    """
    Feature: test scatter_min dynamic shape.
    Description: updates is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_min_dynamic_updates()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_min_dynamic_updates()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_min_grad_float16():
    """
    Feature: test scatter_min grad.
    Description: test float16 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_min_grad(np.float16)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_min_grad(np.float16)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_min_grad_float32():
    """
    Feature: test scatter_min grad.
    Description: test float32 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_min_grad(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_min_grad(np.float32)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_min_grad_int32():
    """
    Feature: test scatter_min grad.
    Description: test int32 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_min_grad(np.int32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_min_grad(np.int32)
