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
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import functional as F

# all cases tested against dchip


def test_avg_pool1d_forward_functional(nptype):
    """
    Feature: test avg_pool1d forward for given input dtype.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    input_x = Tensor(np.ones((2, 3, 6)).astype(nptype))
    output = F.avg_pool1d(input_x, kernel_size=6, stride=1)
    expected = np.ones((2, 3, 1)).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_avg_pool1d_forward_float32_functional():
    """
    Feature: test avg_pool1d forward.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_avg_pool1d_forward_functional(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_avg_pool1d_forward_functional(np.float32)


def test_avg_pool2d_forward_functional(nptype):
    """
    Feature: test avg_pool2d forward for given input dtype.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    input_x = Tensor(np.ones((2, 3, 4, 6)).astype(nptype))
    output = F.avg_pool2d(input_x, kernel_size=3, stride=1)
    expected = np.ones((2, 3, 2, 4)).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_avg_pool2d_forward_float32_functional():
    """
    Feature: test avg_pool2d forward.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_avg_pool2d_forward_functional(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_avg_pool2d_forward_functional(np.float32)


def test_avg_pool3d_forward_functional(nptype):
    """
    Feature: test avg_pool3d forward for given input dtype.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    input_x = Tensor(np.ones((2, 3, 6, 3, 6)).astype(nptype))
    output = F.avg_pool3d(input_x, kernel_size=3, stride=1)
    expected = np.ones((2, 3, 4, 1, 4)).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_avg_pool3d_forward_float32_functional():
    """
    Feature: test avg_pool3d forward.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_avg_pool3d_forward_functional(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_avg_pool3d_forward_functional(np.float32)
