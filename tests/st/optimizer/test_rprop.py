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
from .optimizer_utils import FakeNet, build_network, default_fc1_weight_rprop, default_fc1_bias_rprop, \
    no_default_fc1_weight_rprop, no_default_fc1_bias_rprop, default_group_fc1_weight_rprop, default_group_fc1_bias_rprop


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_default_rprop(mode):
    """
    Feature: Test Rprop optimizer
    Description: Test Rprop with default parameter
    Expectation: Loss values and parameters conform to preset values.
    """
    context.set_context(mode=mode)
    config = {'name': 'Rprop', 'lr': 0.01, 'etas': (0.5, 1.2), 'step_sizes': (1e-6, 50.), 'weight_decay': 0.0}
    _, cells = build_network(config, net=FakeNet())
    assert np.allclose(cells.prev[0].asnumpy(), default_fc1_weight_rprop, atol=1.e-2)
    assert np.allclose(cells.prev[1].asnumpy(), default_fc1_bias_rprop, atol=1.e-2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_no_default_rprop(mode):
    """
    Feature: Test Rprop optimizer
    Description: Test Rprop with another set of parameter
    Expectation: Loss values and parameters conform to preset values.
    """
    context.set_context(mode=mode)
    config = {'name': 'Rprop', 'lr': 0.01, 'etas': (0.6, 1.9), 'step_sizes': (1e-3, 20.), 'weight_decay': 0.0}
    _, cells = build_network(config, net=FakeNet())
    assert np.allclose(cells.prev[0].asnumpy(), no_default_fc1_weight_rprop, atol=1.e-2)
    assert np.allclose(cells.prev[1].asnumpy(), no_default_fc1_bias_rprop, atol=1.e-2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_default_rprop_group(mode):
    """
    Feature: Test Rprop optimizer
    Description: Test Rprop with parameter grouping
    Expectation: Loss values and parameters conform to preset values.
    """
    context.set_context(mode=mode)
    config = {'name': 'Rprop', 'lr': 0.1, 'etas': (0.6, 1.9), 'step_sizes': (1e-2, 10.), 'weight_decay': 0.0}
    _, cells = build_network(config, net=FakeNet(), is_group=True)
    assert np.allclose(cells.prev[0].asnumpy(), default_group_fc1_weight_rprop, atol=1.e-2)
    assert np.allclose(cells.prev[1].asnumpy(), default_group_fc1_bias_rprop, atol=1.e-2)
