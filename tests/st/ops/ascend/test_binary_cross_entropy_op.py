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


def test_binary_cross_entropy_forward_functional(nptype):
    """
    Feature: test binary_cross_entropy forward for given input dtype.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    logits = Tensor(np.array([0.2, 0.7, 0.1]).astype(nptype))
    labels = Tensor(np.array([0., 1., 0.]).astype(nptype))
    weight = Tensor(np.array([1, 2, 2]).astype(nptype))
    output = F.binary_cross_entropy(logits, labels, weight)
    expected = Tensor(np.array([0.38240486]).astype(nptype))
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_binary_cross_entropy_forward_float32_functional():
    """
    Feature: test binary_cross_entropy forward.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_binary_cross_entropy_forward_functional(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_binary_cross_entropy_forward_functional(np.float32)
