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


class Net(nn.Cell):
    def construct(self, x1, x2):
        return x1.logaddexp(x2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_logaddexp(mode):
    """
    Feature: tensor.logaddexp
    Description: Verify the result of logaddexp
    Expectation: success
    """
    ms.set_context(mode=mode)
    x1 = ms.Tensor([-100, 1, 30], ms.float32)
    x2 = ms.Tensor([-1, -1, 3], ms.float32)
    net = Net()
    output = net(x1, x2)
    expect_output = [-0.99999994, 1.1269258, 30.]
    assert np.allclose(output.asnumpy(), expect_output)
