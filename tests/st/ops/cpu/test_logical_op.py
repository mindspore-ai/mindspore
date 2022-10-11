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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class OpNetWrapper(nn.Cell):
    def __init__(self, op):
        super(OpNetWrapper, self).__init__()
        self.op = op

    def construct(self, *inputs):
        return self.op(*inputs)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logicaland():
    op = P.LogicalAnd()
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.array([True, False, False]))
    input_y = Tensor(np.array([True, True, False]))
    outputs = op_wrapper(input_x, input_y)

    assert np.allclose(outputs.asnumpy(), (True, False, False))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logicalor():
    op = P.LogicalOr()
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.array([True, False, False]))
    input_y = Tensor(np.array([True, True, False]))
    outputs = op_wrapper(input_x, input_y)

    assert np.allclose(outputs.asnumpy(), (True, True, False))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_logicalnot():
    op = P.LogicalNot()
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.array([True, False, False]))
    outputs = op_wrapper(input_x)

    assert np.allclose(outputs.asnumpy(), (False, True, True))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_logical_xor_functional_api_modes(mode):
    """
    Feature: Test logical_xor functional api.
    Description: Test logical_xor functional api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor([True, False, True], mstype.bool_)
    y = Tensor([True, True, False], mstype.bool_)
    output = F.logical_xor(x, y)
    expected = np.array([False, True, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_logical_and_tensor_api_modes(mode):
    """
    Feature: Test logical_and tensor api.
    Description: Test logical_and tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    input_x = Tensor([True, False, True], mstype.bool_)
    other = Tensor([True, True, False], mstype.bool_)
    output = input_x.logical_and(other)
    expected = np.array([True, False, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_logical_not_tensor_api_modes(mode):
    """
    Feature: Test logical_not tensor api.
    Description: Test logical_not tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    input_x = Tensor([True, False, True], mstype.bool_)
    output = input_x.logical_not()
    expected = np.array([False, True, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_logical_or_tensor_api_modes(mode):
    """
    Feature: Test logical_or tensor api.
    Description: Test logical_or tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    input_x = Tensor([True, False, True], mstype.bool_)
    other = Tensor([True, True, False], mstype.bool_)
    output = input_x.logical_or(other)
    expected = np.array([True, True, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_logical_xor_tensor_api_modes(mode):
    """
    Feature: Test logical_xor tensor api.
    Description: Test logical_xor tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    input_x = Tensor([True, False, True], mstype.bool_)
    other = Tensor([True, True, False], mstype.bool_)
    output = input_x.logical_xor(other)
    expected = np.array([False, True, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)


if __name__ == '__main__':
    test_logicaland()
    test_logicalor()
    test_logicalnot()
