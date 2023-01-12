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
    def construct(self, x, y, z):
        return F.moveaxis(x, source=y, destination=z)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_net(mode):
    """
    Feature: test moveaxis op
    Description: verify the result of moveaxis
    Expectation: assertion success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([[[-0.3362], [-0.8437]], [[-0.9627], [0.1727]], [[0.5173], [-0.1398]]]))
    moveaxis = Net()
    output = moveaxis(x, 1, 0)
    np_out = np.array([[[-0.3362], [-0.9627], [0.5173]], [[-0.8437], [0.1727], [-0.1398]]])
    assert np.allclose(output.asnumpy(), np_out)
