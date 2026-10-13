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
"""Regression tests for the existing ops.hann_window API."""
import numpy as np
import pytest
import mindspore as ms
from mindspore import nn, ops
from tests.mark_utils import arg_mark


class PeriodicHannWindowNet(nn.Cell):
    """Hann window with a graph compile-time periodic constant."""

    def construct(self):
        return ops.hann_window(5, periodic=True, dtype=ms.float32)


class SymmetricHannWindowNet(nn.Cell):
    """Hann window with a graph compile-time symmetric constant."""

    def construct(self):
        return ops.hann_window(5, periodic=False, dtype=ms.float32)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'cpu_linux'],
          level_mark='level2', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('periodic', [False, True])
def test_hann_window_legacy(mode, periodic):
    """
    Feature: Existing ops.hann_window API.
    Description: Preserve periodic and symmetric windows after adding the mint API.
    Expectation: Values, shape and dtype match the NumPy Hann window.
    """
    ms.set_context(mode=mode)
    net = PeriodicHannWindowNet() if periodic else SymmetricHannWindowNet()
    output = net()
    expected = np.hanning(6)[:-1] if periodic else np.hanning(5)
    assert output.shape == (5,)
    assert output.dtype == ms.float32
    np.testing.assert_allclose(output.asnumpy(), expected, rtol=1e-5, atol=1e-6)
