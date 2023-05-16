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
from mindspore import Tensor, nn
import mindspore.ops.function as F


class Net(nn.Cell):
    def construct(self, x, dim):
        return F.logcumsumexp(x, dim)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_net(mode):
    """
    Feature: test logcumsumexp op
    Description: verify the result of logcumsumexp
    Expectation: assertion success
    """
    ms.set_context(mode=mode)
    x1 = Tensor([1.0, 2.0, 3.0])
    dim = 0
    logcumsumexp = Net()
    output = logcumsumexp(x1, dim)
    np_out = np.array([1.0000, 2.3132617, 3.407606])
    assert np.allclose(output.asnumpy(), np_out)
