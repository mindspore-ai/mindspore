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
import mindspore
from mindspore import Tensor, context
from mindspore.ops.auto_generate.gen_pyboost_func import sin


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sin_cpu():
    """
    Feature: test sin operator
    Description: test sin run by pyboost
    Expectation: success
    """
    context.set_context(device_target="CPU")
    x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), mindspore.float32)
    output = sin(x)
    assert np.allclose(output.asnumpy(), [0.5810352, 0.27635565, 0.41687083, 0.5810352])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sin_ascend():
    """
    Feature: test sin operator
    Description: test sin run by pyboost
    Expectation: success
    """
    context.set_context(device_target="Ascend")
    x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), mindspore.float32)
    output = sin(x)
    assert np.allclose(output.asnumpy(), [0.5810352, 0.27635565, 0.41687083, 0.5810352])
