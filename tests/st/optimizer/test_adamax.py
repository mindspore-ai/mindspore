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

import mindspore.context as context
from mindspore import nn, Tensor
from .optimizer_utils import build_network, default_fc1_weight_adamax, default_fc1_bias_adamax, \
    no_default_fc1_weight_adamax, no_default_fc1_bias_adamax, default_group_fc1_weight_adamax, \
    default_group_fc1_bias_adamax

w1 = np.array([[0.03909272, 0.08893055, -0.259909, -0.459185,
                -0.0195536, 0.12977135, -0.62942827, -0.53132117],
               [0.1542052, 0.6513571, -0.06453168, 0.44788414,
                -0.3775454, 0.6520292, 0.444174, -0.59306043],
               [0.2712369, 0.20890862, 0.6859066, 0.6629662,
                0.4724893, -0.34384444, -0.16007674, 0.21797538],
               [-0.3865972, 0.26727962, 0.23178828, -0.24629539,
                -0.68038213, -0.31262863, 0.10493469, -0.28973007]]).astype("float32")

b1 = np.array([0., 0., 0., 0.]).astype("float32")

w2 = np.array([[-0.6079024, -1.005364, 0.59004724, 0.7289244]]).astype("float32")

b2 = np.array([0.]).astype("float32")


class Net(nn.Cell):
    """
    build a 2-layers net to test adamax optimizer
    """

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Dense(8, 4, weight_init=Tensor(w1), bias_init=Tensor(b1))
        self.fc2 = nn.Dense(4, 1, weight_init=Tensor(w2), bias_init=Tensor(b2))
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_default_adamax(mode):
    """
    Feature: Test adamax optimizer
    Description: Test adamax with default parameter
    Expectation: Loss values and parameters conform to preset values.
    """
    context.set_context(mode=mode)
    config = {'name': 'adamax', 'lr': 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-07,
              'weight_decay': 0.0}
    _, cells = build_network(config, net=Net(), loss_fn=nn.L1Loss(reduction='mean'))
    assert np.allclose(cells.moment1[0].asnumpy(), default_fc1_weight_adamax, atol=1.e-3)
    assert np.allclose(cells.moment1[1].asnumpy(), default_fc1_bias_adamax, atol=1.e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_no_default_adamax(mode):
    """
    Feature: Test adamax optimizer
    Description: Test adamax with another set of parameter
    Expectation: Loss values and parameters conform to preset values.
    """
    context.set_context(mode=mode)
    config = {'name': 'adamax', 'lr': 0.01, "beta1": 0.9, "beta2": 0.98, "eps": 1e-06,
              'weight_decay': 0.0}
    _, cells = build_network(config, net=Net(), loss_fn=nn.L1Loss(reduction='mean'))
    assert np.allclose(cells.moment1[0].asnumpy(), no_default_fc1_weight_adamax, atol=1.e-3)
    assert np.allclose(cells.moment1[1].asnumpy(), no_default_fc1_bias_adamax, atol=1.e-3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_default_adamax_group(mode):
    """
    Feature: Test adamax optimizer
    Description: Test adamax with parameter grouping
    Expectation: Loss values and parameters conform to preset values.
    """
    context.set_context(mode=mode)
    config = {'name': 'adamax', 'lr': 0.002, "beta1": 0.9, "beta2": 0.999, "eps": 1e-08,
              'weight_decay': 0.0}
    _, cells = build_network(config, is_group=True, net=Net(), loss_fn=nn.L1Loss(reduction='mean'))
    assert np.allclose(cells.moment1[0].asnumpy(), default_group_fc1_weight_adamax, atol=1.e-3)
    assert np.allclose(cells.moment1[1].asnumpy(), default_group_fc1_bias_adamax, atol=1.e-3)
