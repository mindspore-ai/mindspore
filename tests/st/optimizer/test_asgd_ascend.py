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
from .optimizer_utils import build_network, loss_not_default_asgd, loss_default_asgd, loss_group_asgd


def test_default_asgd_pynative():
    """
    Feature: Test ASGD optimizer
    Description: Test ASGD in Pynative mode with default parameter
    Expectation: Loss values and parameters conform to preset values.
    """
    from .optimizer_utils import default_fc1_weight_asgd, \
        default_fc1_bias_asgd, default_fc2_weight_asgd, default_fc2_bias_asgd
    context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')
    config = {'name': 'ASGD', 'lr': 0.01, 'lambd': 1e-4, 'alpha': 0.75, 't0': 1e6, 'weight_decay': 0.0}
    loss, cells = build_network(config)
    assert np.allclose(cells.ax[0].asnumpy(), default_fc1_weight_asgd, atol=1.e-5)
    assert np.allclose(cells.ax[1].asnumpy(), default_fc1_bias_asgd, atol=1.e-5)
    assert np.allclose(cells.ax[2].asnumpy(), default_fc2_weight_asgd, atol=1.e-5)
    assert np.allclose(cells.ax[3].asnumpy(), default_fc2_bias_asgd, atol=1.e-5)
    assert np.allclose(loss_default_asgd, loss, atol=1.e-5)


def test_default_asgd_graph():
    """
    Feature: Test ASGD optimizer
    Description: Test ASGD in Graph mode with default parameter
    Expectation: Loss values and parameters conform to preset values.
    """
    from .optimizer_utils import default_fc1_weight_asgd, \
        default_fc1_bias_asgd, default_fc2_weight_asgd, default_fc2_bias_asgd
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    config = {'name': 'ASGD', 'lr': 0.01, 'lambd': 1e-4, 'alpha': 0.75, 't0': 1e6, 'weight_decay': 0.0}
    loss, cells = build_network(config)
    assert np.allclose(cells.ax[0].asnumpy(), default_fc1_weight_asgd, atol=1.e-5)
    assert np.allclose(cells.ax[1].asnumpy(), default_fc1_bias_asgd, atol=1.e-5)
    assert np.allclose(cells.ax[2].asnumpy(), default_fc2_weight_asgd, atol=1.e-5)
    assert np.allclose(cells.ax[3].asnumpy(), default_fc2_bias_asgd, atol=1.e-5)
    assert np.allclose(loss_default_asgd, loss, atol=1.e-5)


def test_no_default_asgd_pynative():
    """
    Feature: Test ASGD optimizer
    Description: Test ASGD in Pynative mode with another set of parameter
    Expectation: Loss values and parameters conform to preset values.
    """
    from .optimizer_utils import no_default_fc1_weight_asgd, \
        no_default_fc1_bias_asgd, no_default_fc2_weight_asgd, no_default_fc2_bias_asgd
    context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')
    config = {'name': 'ASGD', 'lr': 0.001, 'lambd': 1e-3, 'alpha': 0.8, 't0': 50., 'weight_decay': 0.001}
    loss, cells = build_network(config)
    assert np.allclose(cells.ax[0].asnumpy(), no_default_fc1_weight_asgd, atol=1.e-5)
    assert np.allclose(cells.ax[1].asnumpy(), no_default_fc1_bias_asgd, atol=1.e-5)
    assert np.allclose(cells.ax[2].asnumpy(), no_default_fc2_weight_asgd, atol=1.e-5)
    assert np.allclose(cells.ax[3].asnumpy(), no_default_fc2_bias_asgd, atol=1.e-5)
    assert np.allclose(loss_not_default_asgd, loss, atol=1.e-5, rtol=1e-3)


def test_no_default_asgd_graph():
    """
    Feature: Test ASGD optimizer
    Description: Test ASGD in Graph mode with another set of parameter
    Expectation: Loss values and parameters conform to preset values.
    """
    from .optimizer_utils import no_default_fc1_weight_asgd, \
        no_default_fc1_bias_asgd, no_default_fc2_weight_asgd, no_default_fc2_bias_asgd
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    config = {'name': 'ASGD', 'lr': 0.001, 'lambd': 1e-3, 'alpha': 0.8, 't0': 50., 'weight_decay': 0.001}
    loss, cells = build_network(config)
    assert np.allclose(cells.ax[0].asnumpy(), no_default_fc1_weight_asgd, atol=1.e-5)
    assert np.allclose(cells.ax[1].asnumpy(), no_default_fc1_bias_asgd, atol=1.e-5)
    assert np.allclose(cells.ax[2].asnumpy(), no_default_fc2_weight_asgd, atol=1.e-5)
    assert np.allclose(cells.ax[3].asnumpy(), no_default_fc2_bias_asgd, atol=1.e-5)
    assert np.allclose(loss_not_default_asgd, loss, atol=1.e-5, rtol=1e-3)


def test_default_asgd_group_pynative():
    """
    Feature: Test ASGD optimizer
    Description: Test ASGD in Pynative mode with parameter grouping
    Expectation: Loss values and parameters conform to preset values.
    """
    from .optimizer_utils import no_default_group_fc1_weight_asgd, no_default_group_fc1_bias_asgd, \
        no_default_group_fc2_weight_asgd, no_default_group_fc2_bias_asgd
    context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')
    config = {'name': 'ASGD', 'lr': 0.1, 'lambd': 1e-3, 'alpha': 0.8, 't0': 50., 'weight_decay': 0.001}
    loss, cells = build_network(config, is_group=True)
    assert np.allclose(cells.ax[0].asnumpy(), no_default_group_fc1_weight_asgd, atol=1.e-5)
    assert np.allclose(cells.ax[1].asnumpy(), no_default_group_fc1_bias_asgd, atol=1.e-5)
    assert np.allclose(cells.ax[2].asnumpy(), no_default_group_fc2_weight_asgd, atol=1.e-5)
    assert np.allclose(cells.ax[3].asnumpy(), no_default_group_fc2_bias_asgd, atol=1.e-5)
    assert np.allclose(loss_group_asgd, loss, atol=1.e-5, rtol=1e-3)


def test_default_asgd_group_graph():
    """
    Feature: Test ASGD optimizer
    Description: Test ASGD in Graph mode with parameter grouping
    Expectation: Loss values and parameters conform to preset values.
    """
    from .optimizer_utils import no_default_group_fc1_weight_asgd, no_default_group_fc1_bias_asgd, \
        no_default_group_fc2_weight_asgd, no_default_group_fc2_bias_asgd
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    config = {'name': 'ASGD', 'lr': 0.1, 'lambd': 1e-3, 'alpha': 0.8, 't0': 50., 'weight_decay': 0.001}
    loss, cells = build_network(config, is_group=True)
    assert np.allclose(cells.ax[0].asnumpy(), no_default_group_fc1_weight_asgd, atol=1.e-5)
    assert np.allclose(cells.ax[1].asnumpy(), no_default_group_fc1_bias_asgd, atol=1.e-5)
    assert np.allclose(cells.ax[2].asnumpy(), no_default_group_fc2_weight_asgd, atol=1.e-5)
    assert np.allclose(cells.ax[3].asnumpy(), no_default_group_fc2_bias_asgd, atol=1.e-5)
    assert np.allclose(loss_group_asgd, loss, atol=1.e-5, rtol=1e-3)
