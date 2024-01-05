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
from mindspore import Tensor, nn
import mindspore.ops.function as F


class NetSoftmin(nn.Cell):
    def construct(self, x, axis=-1):
        return F.softmin(x, axis)


@pytest.mark.level2
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_net(mode):
    """
    Feature: test softmin op
    Description: verify the result of softmin
    Expectation: assertion success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([1, 2, 3, 4, 5]), ms.float32)
    softmin = NetSoftmin()
    output = softmin(x)
    np_out = np.array([0.6364086, 0.23412165, 0.08612854, 0.03168492, 0.01165623])
    assert np.allclose(output.asnumpy(), np_out)
