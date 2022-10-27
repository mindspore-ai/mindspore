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

import pytest
import numpy as np

from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_greater_equal_functional_api_modes(mode):
    """
    Feature: Test greater_equal functional api.
    Description: Test greater_equal functional api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="Ascend")

    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    y = Tensor(np.array([1, 1, 4]), mstype.int32)
    output = F.greater_equal(x, y)
    expected = np.array([True, True, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_greater_equal_tensor_api_modes(mode):
    """
    Feature: Test greater_equal tensor api.
    Description: Test greater_equal tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="Ascend")

    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    y = Tensor(np.array([1, 1, 4]), mstype.int32)
    output = x.greater_equal(y)
    expected = np.array([True, True, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_greater_functional_api_modes(mode):
    """
    Feature: Test greater functional api.
    Description: Test greater functional api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="Ascend")

    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    y = Tensor(np.array([1, 1, 4]), mstype.int32)
    output = F.greater(x, y)
    expected = np.array([False, True, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_greater_tensor_api_modes(mode):
    """
    Feature: Test greater tensor api.
    Description: Test greater tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="Ascend")

    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    y = Tensor(np.array([1, 1, 4]), mstype.int32)
    output = x.greater(y)
    expected = np.array([False, True, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)
