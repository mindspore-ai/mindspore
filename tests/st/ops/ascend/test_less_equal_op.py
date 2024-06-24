# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
import numpy as np
from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype


def test_less_equal_functional_api():
    """
    Feature: test less_equal functional API.
    Description: test less_equal functional API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    other = Tensor(np.array([1, 1, 4]), mstype.int32)
    output = F.less_equal(x, other)
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)


def test_less_equal_functional_api_bfloat16():
    """
    Feature: test less_equal functional API for dtype bfloat16.
    Description: test less_equal functional API for dtype bfloat16.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([1.2, 2.5, 3.8]), mstype.bfloat16)
    other = Tensor(np.array([1.2, 1.8, 4.0]), mstype.bfloat16)
    output = F.less_equal(x, other)
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)


def test_less_equal_tensor_api():
    """
    Feature: test less_equal tensor API.
    Description: test case for less_equal tensor API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    other = Tensor(np.array([1, 1, 4]), mstype.int32)
    output = x.less_equal(other)
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_less_equal_functional_tensor_modes():
    """
    Feature: test less_equal functional and tensor APIs in PyNative and Graph modes.
    Description: test case for erfiless_equalnv functional API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_less_equal_functional_api()
    test_less_equal_tensor_api()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_less_equal_functional_api()
    test_less_equal_tensor_api()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_less_equal_functional_tensor_modes_910b():
    """
    Feature: test less_equal functional API in PyNative and Graph modes on 910B.
    Description: test case for less_equal functional API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_less_equal_functional_api_bfloat16()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_less_equal_functional_api_bfloat16()
