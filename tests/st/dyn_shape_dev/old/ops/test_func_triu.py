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
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore import context


class Net(nn.Cell):
    def construct(self, x, k):
        return ops.triu(x, diagonal=k)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_triu(mode):
    """
    Feature: test_ops_triu
    Description: Verify the result of test_ops_triu
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net()
    x = Tensor(np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [10, 11, 12, 13],
                         [14, 15, 16, 17]]))
    output = net(x, 1)
    expected = np.array([[0, 2, 3, 4],
                         [0, 0, 7, 8],
                         [0, 0, 0, 13],
                         [0, 0, 0, 0]])
    assert np.array_equal(output.asnumpy(), expected)
