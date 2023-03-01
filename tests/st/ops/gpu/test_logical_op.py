# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

import mindspore.context as context
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore import dtype as mstype


class NetAnd(Cell):
    def __init__(self):
        super(NetAnd, self).__init__()
        self.logicaland = P.LogicalAnd()

    def construct(self, input_x, input_y):
        return self.logicaland(input_x, input_y)


class NetOr(Cell):
    def __init__(self):
        super(NetOr, self).__init__()
        self.logicalor = P.LogicalOr()

    def construct(self, input_x, input_y):
        return self.logicalor(input_x, input_y)


class NetNot(Cell):
    def __init__(self):
        super(NetNot, self).__init__()
        self.logicalnot = P.LogicalNot()

    def construct(self, input_x):
        return self.logicalnot(input_x)


x = np.array([True, False, False]).astype(np.bool)
y = np.array([False]).astype(np.bool)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_logicaland():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    logicaland = NetAnd()
    output = logicaland(Tensor(x), Tensor(y))
    assert np.all(output.asnumpy() == np.logical_and(x, y))

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    logicaland = NetAnd()
    output = logicaland(Tensor(x), Tensor(y))
    assert np.all(output.asnumpy() == np.logical_and(x, y))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_logicalor():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    logicalor = NetOr()
    output = logicalor(Tensor(x), Tensor(y))
    assert np.all(output.asnumpy() == np.logical_or(x, y))

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    logicalor = NetOr()
    output = logicalor(Tensor(x), Tensor(y))
    assert np.all(output.asnumpy() == np.logical_or(x, y))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("dtype", [np.bool_])
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_logicalnot(dtype, mode):
    """
    Feature: sigmoid
    Description: test cases for sigmoid
    Expectation: success
    """
    input_x = np.random.uniform(-2, 2, (2, 3, 4, 4)).astype(dtype)
    context.set_context(mode=mode, device_target="GPU")
    logicalnot = NetNot()
    output = logicalnot(Tensor(input_x))
    assert np.allclose(output.asnumpy(), np.logical_not(input_x), 1.0e-3, 1.0e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_logical_and_tensor_api_modes(mode):
    """
    Feature: Test logical_and tensor api.
    Description: Test logical_and tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="GPU")
    input_x = Tensor([True, False, True], mstype.bool_)
    other = Tensor([True, True, False], mstype.bool_)
    output = input_x.logical_and(other)
    expected = np.array([True, False, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_logical_not_tensor_api_modes(mode):
    """
    Feature: Test logical_not tensor api.
    Description: Test logical_not tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="GPU")
    input_x = Tensor([True, False, True], mstype.bool_)
    output = input_x.logical_not()
    expected = np.array([False, True, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_logical_or_tensor_api_modes(mode):
    """
    Feature: Test logical_or tensor api.
    Description: Test logical_or tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="GPU")
    input_x = Tensor([True, False, True], mstype.bool_)
    other = Tensor([True, True, False], mstype.bool_)
    output = input_x.logical_or(other)
    expected = np.array([True, True, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)
