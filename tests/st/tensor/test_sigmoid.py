# Copyright 2024 Huawei Technologies Co., Ltd
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


class Net(nn.Cell):
    def construct(self, x):
        return x.sigmoid()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_sigmoid(mode):
    """
    Feature: tensor.sigmoid
    Description: Verify the result of sigmoid
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([1, 2, 3, 4, 5]), ms.float32)
    net = Net()
    output = net(x)
    expect_output = np.array([0.7310586, 0.880797, 0.95257413, 0.98201376, 0.9933072], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
