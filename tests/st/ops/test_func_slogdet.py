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
from mindspore import ops


class Net(nn.Cell):
    def construct(self, x):
        return ops.slogdet(x)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_slogdet(mode):
    """
    Feature: slogdet
    Description: Verify the result of slogdet
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[-1.8297, -0.8474, 1.0292], [-1.2167, 0.5574, -0.6753], [-0.6702, 0.2276, 1.2421]])
    net = Net()
    output1, output2 = net(x)
    expect_output1 = np.array(-1, dtype=np.float32)
    expect_output2 = np.array(1.13549, dtype=np.float32)
    assert np.allclose(output1.asnumpy(), expect_output1)
    assert np.allclose(output2.asnumpy(), expect_output2)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_slogdet_complex(mode):
    """
    Feature: slogdet
    Description: Verify the result of slogdet
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[-1.5 + 7.8j, 3 + 5.75j, 2 + 2.4j],
                [-6.4 + 485.4j, 45 + 3.14j, 45 + 453j],
                [-3.5 + 5.8j, 63 + 12.75j, -5 + 6.4j]], dtype=ms.complex64)
    net = Net()
    output1, output2 = net(x)
    expect_output1 = np.array(0.749919+0.66153j, dtype=np.complex)
    expect_output2 = np.array(12.0614+0j, dtype=np.complex)
    assert np.allclose(output1.asnumpy(), expect_output1)
    assert np.allclose(output2.asnumpy(), expect_output2)
