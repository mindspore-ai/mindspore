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
"""ST tests for Rprop optimizer."""

import numpy as np
import pytest

from mindspore import Tensor, context, nn
from mindspore import ops, Parameter
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark
from tests.st.optimizer.optimizer_utils import (
    FakeNet,
    build_network,
    default_fc1_bias_rprop,
    default_fc1_weight_rprop,
    default_group_fc1_bias_rprop,
    default_group_fc1_weight_rprop,
    no_default_fc1_bias_rprop,
    no_default_fc1_weight_rprop,
)


@arg_mark(
    plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential',
)
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


@arg_mark(
    plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential',
)
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


@arg_mark(
    plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential',
)
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


@arg_mark(
    plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential',
)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [mstype.float16, mstype.bfloat16])
def test_rprop_mixed_precision(dtype, mode):
    """
    Feature: Rprop supports float16 and bfloat16 training
    Description: Train a minimal network (no MatMul) with Rprop using low-precision weights/inputs
    Expectation: Training runs without dtype conversion errors and prev/step_size keep dtype
    """
    context.set_context(mode=mode)

    class TinyNet(nn.Cell):
        """Minimal net: single parameter elementwise multiply (no MatMul)."""

        def __init__(self, weight_dtype):
            super().__init__()
            self.w = Parameter(Tensor([1.0, -1.0], weight_dtype), name="w")
            self.mul = ops.Mul()

        def construct(self, x):
            return self.mul(self.w, x)

    class TinyNetWithLoss(nn.Cell):
        """Wrap TinyNet with L1Loss."""

        def __init__(self, network, loss_fn):
            super().__init__()
            self.network = network
            self.loss_fn = loss_fn

        def construct(self, x, label):
            out = self.network(x)
            return self.loss_fn(out, label)

    net = TinyNet(dtype)

    loss_fn = nn.L1Loss(reduction='mean')
    net_with_loss = TinyNetWithLoss(net, loss_fn)
    net_with_loss.set_train()

    opt = nn.Rprop(net_with_loss.trainable_params(), learning_rate=0.01, etas=(0.5, 1.2),
                   step_sizes=(1e-6, 50.), weight_decay=0.0)
    train_one_step = nn.TrainOneStepCell(net_with_loss, opt)
    train_one_step.set_train()

    losses = []
    for i in range(5):
        data = Tensor(np.ones((2,)) * i, dtype=dtype)
        label = Tensor(np.ones((2,)) * (i + 1), dtype=dtype)
        loss = train_one_step(data, label)
        losses.append(loss.asnumpy())

    assert losses, "Loss list should not be empty."
    for param in net.trainable_params():
        assert param.dtype == dtype
    # prev and step_size in Rprop should keep the same dtype as parameters
    assert all(p.dtype == dtype for p in opt.prev)
    assert all(p.dtype == dtype for p in opt.step_size)
