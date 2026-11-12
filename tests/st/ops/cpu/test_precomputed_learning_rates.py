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
"""Optimizer updates using precomputed per-step learning rates."""
import numpy as np
import pytest

from mindspore import Parameter, Tensor, context, nn
from tests.mark_utils import arg_mark


@pytest.fixture(autouse=True)
def restore_mode():
    """Keep the caller's execution mode after each optimizer regression."""
    original = context.get_context("mode")
    yield
    context.set_context(mode=original)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="onecard")
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("as_tensor", [False, True])
def test_precomputed_rates_update_each_step(mode, as_tensor):
    """
    Feature: Precomputed learning rates for SGD.
    Description: Apply unit gradients with a list or Tensor schedule in graph and PyNative modes.
    Expectation: Each step decreases the weight by the corresponding scheduled rate.
    """
    context.set_context(mode=mode)
    rates = [0.1, 0.1, 0.05, 0.05, 0.01, 0.01]
    learning_rate = Tensor(np.array(rates, np.float32)) if as_tensor else rates
    weight = Parameter(Tensor(np.array([1.0], np.float32)), name="weight")
    optimizer = nn.SGD([weight], learning_rate=learning_rate)
    gradient = Tensor(np.array([1.0], np.float32))
    expected = np.array([1.0], np.float32)
    for rate in rates:
        optimizer((gradient,))
        expected -= np.float32(rate)
        np.testing.assert_allclose(weight.asnumpy(), expected, rtol=1e-6, atol=1e-7)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="onecard")
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_precomputed_group_rates_keep_default_rate(mode):
    """
    Feature: Per-group learning rates for SGD.
    Description: Apply unit gradients to one group with a schedule and one using the default rate.
    Expectation: The first weight follows the schedule; the second decreases by 0.125 at every step.
    """
    context.set_context(mode=mode)
    rates = [0.01, 0.0067, 0.0034] + [0.0001] * 7
    scheduled = Parameter(Tensor(np.array([1.0], np.float32)), name="scheduled")
    constant = Parameter(Tensor(np.array([1.0], np.float32)), name="constant")
    optimizer = nn.SGD([{"params": [scheduled], "lr": rates}, {"params": [constant]}], learning_rate=0.125)
    gradient = Tensor(np.array([1.0], np.float32))
    expected_scheduled = np.array([1.0], np.float32)
    expected_constant = np.array([1.0], np.float32)
    for rate in rates:
        optimizer((gradient, gradient))
        expected_scheduled -= np.float32(rate)
        expected_constant -= np.float32(0.125)
        np.testing.assert_allclose(scheduled.asnumpy(), expected_scheduled, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(constant.asnumpy(), expected_constant, rtol=1e-6, atol=1e-7)
