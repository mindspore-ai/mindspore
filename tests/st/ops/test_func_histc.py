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
import numpy as np
import pytest
import mindspore as ms
from mindspore import ops
from mindspore import nn
from mindspore import Tensor
from mindspore import dtype as mstype


class Net(nn.Cell):
    def construct(self, x, bins, min_val, max_val):
        return ops.histc(x, bins, min_val, max_val)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_histogram_normal(mode):
    """
    Feature: ops.histc
    Description: Verify the result of histc
    Expectation: success
    """
    bins, min_val, max_val = 4, 0.0, 3.0
    net = Net()
    x = Tensor([1, 2, 1], mstype.int32)
    output = net(x, bins, min_val, max_val)
    expected_output = np.array([0, 2, 1, 0])
    assert np.array_equal(output.asnumpy(), expected_output)
