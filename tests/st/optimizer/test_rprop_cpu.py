# Copyright 2021 Huawei Technologies Co., Ltd
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
import mindspore.context as context
from .optimizer_utils import build_network, loss_default_rprop, loss_group_rprop, loss_not_default_rprop


def test_default_rprop_graph():
    """
    Feature: Test Rprop optimizer
    Description: Test Rprop in Graph mode with default parameter
    Expectation: Loss values and parameters conform to preset values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    config = {'name': 'Rprop', 'lr': 0.01, 'etas': (0.5, 1.2), 'step_sizes': (1e-6, 50.), 'weight_decay': 0.0}
    loss = build_network(config)
    assert np.allclose(loss_default_rprop, loss, atol=1.e-5)


def test_no_default_rprop_graph():
    """
    Feature: Test Rprop optimizer
    Description: Test Rprop in Graph mode with another set of parameter
    Expectation: Loss values and parameters conform to preset values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    config = {'name': 'Rprop', 'lr': 0.001, 'etas': (0.6, 1.9), 'step_sizes': (1e-3, 20.), 'weight_decay': 0.0}
    loss = build_network(config)
    assert np.allclose(loss_not_default_rprop, loss, atol=1.e-5)


def test_default_rprop_group_graph():
    """
    Feature: Test Rprop optimizer
    Description: Test Rprop in Graph mode with parameter grouping
    Expectation: Loss values and parameters conform to preset values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    config = {'name': 'Rprop', 'lr': 0.001, 'etas': (0.6, 1.9), 'step_sizes': (1e-2, 10.), 'weight_decay': 0.0}
    loss = build_network(config, is_group=True)
    assert np.allclose(loss_group_rprop, loss, atol=1.e-5)
