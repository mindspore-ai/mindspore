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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class RandomChoiceWithMaskNet(nn.Cell):
    def __init__(self):
        super(RandomChoiceWithMaskNet, self).__init__()
        self.random_choice_with_mask = P.RandomChoiceWithMask(count=4, seed=1)
        self.random_choice_with_mask.add_prim_attr("cust_aicpu", "mindspore_aicpu_kernels")

    def construct(self, x):
        return self.random_choice_with_mask(x)


@pytest.mark.skip(reason="fail on run package upgrade")
def test_random_choice_with_mask_graph():
    """
    Feature: Custom aicpu feature.
    Description: Test random_choice_with_mask kernel in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    input_tensor = Tensor(np.array([[1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1],
                                    [0, 0, 0, 1]]).astype(np.bool))
    expect1 = (4, 2)
    expect2 = (4,)
    net = RandomChoiceWithMaskNet()
    output1, output2 = net(input_tensor)
    assert output1.shape == expect1
    assert output2.shape == expect2


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_random_choice_with_mask_pynative():
    """
    Feature: Custom aicpu feature.
    Description: Test random_choice_with_mask kernel in pynative mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    input_tensor = Tensor(np.array([[1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1],
                                    [0, 0, 0, 1]]).astype(np.bool))
    expect1 = (4, 2)
    expect2 = (4,)
    net = RandomChoiceWithMaskNet()
    output1, output2 = net(input_tensor)
    assert output1.shape == expect1
    assert output2.shape == expect2
