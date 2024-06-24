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
import numpy as np

from mindspore import Tensor
import mindspore.context as context
from mindspore.common import dtype as mstype


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_logical_and_tensor_api_modes(mode):
    """
    Feature: Test logical_and tensor api.
    Description: Test logical_and tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="Ascend")
    input_x = Tensor([True, False, True], mstype.bool_)
    other = Tensor([True, True, False], mstype.bool_)
    output = input_x.logical_and(other)
    expected = np.array([True, False, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_logical_not_tensor_api_modes(mode):
    """
    Feature: Test logical_not tensor api.
    Description: Test logical_not tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="Ascend")
    input_x = Tensor([True, False, True], mstype.bool_)
    output = input_x.logical_not()
    expected = np.array([False, True, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_logical_or_tensor_api_modes(mode):
    """
    Feature: Test logical_or tensor api.
    Description: Test logical_or tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="Ascend")
    input_x = Tensor([True, False, True], mstype.bool_)
    other = Tensor([True, True, False], mstype.bool_)
    output = input_x.logical_or(other)
    expected = np.array([True, True, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)
