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
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Roll(nn.Cell):
    def construct(self, x):
        return x.roll(shifts=2, dims=0)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_roll(mode):
    """
    Feature: tensor.roll
    Description: Verify the result of roll
    Expectation: success
    """
    ms.set_context(mode=mode)
    input_x = Tensor(np.array([0, 1, 2, 3, 4]).astype(np.float32))
    net = Roll()
    output = net(input_x)
    expect_output = [3., 4., 0., 1., 2.]
    assert np.allclose(output.asnumpy(), expect_output)
