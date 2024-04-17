# Copyright 2023 Huawei Technologies Co., Ltd
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
import test_utils
import mindspore
from mindspore import Tensor
from mindspore.ops.auto_generate import abs
from mindspore import ops


@test_utils.run_with_cell
def abs_forward_func(x):
    return abs(x)


@test_utils.run_with_cell
def abs_backward_func(x):
    return ops.grad(abs_forward_func, (0))(x)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_pyboost_abs_forward():
    """
    Feature: test abs operator
    Description: test abs forward by pyboost
    Expectation: success
    """
    x = Tensor([1.0, -2.0, -3.0], mindspore.float32)
    output1 = abs_forward_func(x)
    assert np.allclose(output1.asnumpy(), [1.0, 2.0, 3.0])
    x = Tensor([1, 0, 0], mindspore.bool_)
    output2 = abs_forward_func(x)
    assert np.allclose(output2.asnumpy(), [True, False, False])


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_pyboost_abs_backward():
    """
    Feature: test abs operator
    Description: test abs backward by pyboost
    Expectation: success
    """
    x = Tensor([1.0, -2.0, -3.0], mindspore.float32)
    output1 = abs_backward_func(x)
    assert np.allclose(output1.asnumpy(), [1.0, -1.0, -1.0])
    x = Tensor([1, 0, 0], mindspore.float32)
    output2 = abs_backward_func(x)
    assert np.allclose(output2.asnumpy(), [1.0, 0, 0])
