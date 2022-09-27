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
from mindspore import Tensor, Parameter
from mindspore.ops import functional as F

# all cases tested against dchip


class TestScatterMulNet(nn.Cell):
    def __init__(self, inputx):
        super(TestScatterMulNet, self).__init__()

        self.scatter_mul = ops.ScatterMul()
        self.inputx = Parameter(inputx, name="inputx")

    def construct(self, indices, updates):
        out = self.scatter_mul(self.inputx, indices, updates)
        return out


def scatter_mul_forward(nptype):
    inputx = Tensor(np.arange(0, 9).reshape((3, 3)).astype(nptype))
    indices = Tensor(np.array([[[1, 0, 2], [2, 2, 0]], [[1, 0, 1], [2, 1, 2]]]).astype(np.int32))
    updates = Tensor(np.ones((2, 2, 3, 3)).astype(nptype))

    net = TestScatterMulNet(inputx)
    output = net(indices, updates)
    expected = inputx.asnumpy()
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def scatter_mul_forward_functional(nptype):
    inputx = Tensor(np.arange(0, 9).reshape((3, 3)).astype(nptype))
    indices = Tensor(np.array([[[1, 0, 2], [2, 2, 0]], [[1, 0, 1], [2, 1, 2]]]).astype(np.int32))
    updates = Tensor(np.ones((2, 2, 3, 3)).astype(nptype))

    output = F.scatter_mul(Parameter(inputx, name="inputx"), indices, updates)
    expected = inputx.asnumpy()
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def scatter_mul_dynamic_updates():
    inputx = Tensor(np.arange(0, 9).reshape((3, 3)).astype(np.float32))
    indices = Tensor(np.array([[[1, 0, 2], [2, 2, 0]], [[1, 0, 1], [2, 1, 2]]]).astype(np.int32))
    updates = Tensor(np.ones((2, 2, 3, 3)).astype(np.float32))
    updates_dy = Tensor(shape=(2, 2, None, 3), dtype=mindspore.float32)

    net = TestScatterMulNet(inputx)
    net.set_inputs(indices, updates_dy)
    output = net(indices, updates)
    expected = inputx.asnumpy()
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def scatter_mul_dynamic_indices():
    inputx = Tensor(np.arange(0, 9).reshape((3, 3)).astype(np.float32))
    indices = Tensor(np.array([[[1, 0, 2], [2, 2, 0]], [[1, 0, 1], [2, 1, 2]]]).astype(np.int32))
    updates = Tensor(np.ones((2, 2, 3, 3)).astype(np.float32))
    indices_dy = Tensor(shape=(2, None, 3), dtype=mindspore.int32)

    net = TestScatterMulNet(inputx)
    net.set_inputs(indices_dy, updates)
    output = net(indices, updates)
    expected = inputx.asnumpy()
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_mul_forward_float16():
    """
    Feature: test scatter_mul forward.
    Description: test float16 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_mul_forward(np.float16)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_mul_forward(np.float16)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_mul_forward_float32():
    """
    Feature: test scatter_mul forward.
    Description: test float32 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_mul_forward(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_mul_forward(np.float32)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_mul_forward_int32():
    """
    Feature: test scatter_mul forward.
    Description: test int32 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_mul_forward(np.int32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_mul_forward(np.int32)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_mul_dynamic_indices():
    """
    Feature: test scatter_mul dynamic shape.
    Description: indices is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_mul_dynamic_indices()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_mul_dynamic_indices()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_mul_dynamic_updates():
    """
    Feature: test scatter_mul dynamic shape.
    Description: updates is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_mul_dynamic_updates()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_mul_dynamic_updates()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_mul_forward_int32_functional():
    """
    Feature: test scatter_mul forward.
    Description: test int32 inputs.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    scatter_mul_forward_functional(np.int32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    scatter_mul_forward_functional(np.int32)
