# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Guard FTRL updates after moving its primitives to internal exports."""
from mindspore.ops.operations import _inner_ops
from mindspore import Parameter, Tensor, context, nn
from mindspore.ops.operations import nn_ops

import numpy as np

import pytest

from tests.mark_utils import arg_mark


@pytest.fixture(autouse=True)
def restore_mode():
    """Restore execution mode after each guard."""
    original = context.get_context("mode")
    yield
    context.set_context(mode=original)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_retained_ftrl_optimizer(mode):
    """
    Feature: Retained FTRL optimizer.
    Description: Perform a dense update with zero regularization in each execution mode.
    Expectation: The parameter matches the FTRL update equation.
    """
    context.set_context(mode=mode)
    assert _inner_ops.SparseApplyFtrl is nn_ops.SparseApplyFtrl
    original = np.array([1, 2], np.float32)
    gradient = np.array([0.5, 0.25], np.float32)
    parameter = Parameter(Tensor(original), name="weight")
    optimizer = nn.FTRL([parameter], initial_accum=0.1, learning_rate=0.1,
                        lr_power=-0.5, l1=0.0, l2=0.0)
    optimizer((Tensor(gradient),))
    accumulator = 0.1 + gradient ** 2
    linear = gradient - (np.sqrt(accumulator) - np.sqrt(0.1)) / 0.1 * original
    expected = -linear / (np.sqrt(accumulator) / 0.1)
    np.testing.assert_allclose(parameter.asnumpy(), expected, rtol=1e-5, atol=1e-6)
