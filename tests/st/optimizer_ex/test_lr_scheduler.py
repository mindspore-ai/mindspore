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

import pytest
import numpy as np
from mindspore import nn
import mindspore as ms


class Net(nn.Cell):
    def __init__(self, num_class=10):
        super(Net, self).__init__()
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_exponential_lr(mode):
    """
    Feature: ExponentialLR
    Description: Verify the result of ExponentialLR
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    optimizer = nn.optim_ex.Adam(net.trainable_params(), 0.01)
    scheduler = nn.ExponentialLR(optimizer, gamma=0.5)
    expect_list = [[0.005], [0.0025], [0.00125], [0.000625], [0.0003125]]
    for i in range(5):
        scheduler.step()
        current_lr = scheduler.get_last_lr()
        assert np.allclose(current_lr[0].asnumpy(), expect_list[i])


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_polynomial_lr(mode):
    """
    Feature: PolynomialLR
    Description: Verify the result of PolynomialLR
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    optimizer = nn.optim_ex.Adam(net.trainable_params(), 0.01)
    scheduler = nn.PolynomialLR(optimizer)
    expect_list = [[0.008], [0.006], [0.004], [0.002], [0], [0]]
    for i in range(6):
        scheduler.step()
        current_lr = scheduler.get_last_lr()
        assert np.allclose(current_lr[0].asnumpy(), expect_list[i])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_chained_scheduler(mode):
    """
    Feature: ChainedScheduler
    Description: Verify the result of ChainedScheduler
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    optimizer = nn.optim_ex.Adam(net.trainable_params(), 0.01)
    scheduler1 = nn.PolynomialLR(optimizer)
    scheduler2 = nn.ExponentialLR(optimizer, gamma=0.5)
    scheduler = nn.ChainedScheduler([scheduler1, scheduler2])
    expect_list = [[0.004], [0.0015], [0.0005], [0.000125], [0], [0]]
    for i in range(6):
        scheduler.step()
        current_lr = scheduler.get_last_lr()
        assert np.allclose(current_lr[0].asnumpy(), expect_list[i])
