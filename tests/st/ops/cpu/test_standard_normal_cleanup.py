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
"""Guard shared normal generation after removing the standard_normal wrapper."""
import numpy as np
import pytest

import mindspore as ms
from mindspore import nn, ops
from mindspore.common._register_for_tensor import tensor_operator_registry
from tests.mark_utils import arg_mark


@pytest.fixture(autouse=True)
def restore_context():
    """Do not leak execution settings into other tests."""
    mode = ms.get_context('mode')
    target = ms.get_context('device_target')
    ms.set_context(device_target='CPU')
    yield
    ms.set_context(mode=mode, device_target=target)


class RandomNet(nn.Cell):
    """Use all retained normal function routes with explicit seeds."""
    def construct(self, value):
        return (ops.normal((2, 3), 0.0, 1.0, seed=7),
                ops.randn(2, 3, dtype=ms.float32, seed=8),
                ops.randn_like(value, seed=9))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', mem_peak=0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_retained_normal_functions(mode):
    """
    Feature: Retained normal, randn and randn_like functions.
    Description: Run all three routes in graph and pynative execution modes.
    Expectation: Each produces finite float32 samples of the requested shape.
    """
    assert tensor_operator_registry.get('standard_normal') is ops.StandardNormal
    ms.set_context(mode=mode)
    value = ms.Tensor(np.zeros((2, 3)), ms.float32)
    for output in RandomNet()(value):
        assert output.shape == (2, 3)
        assert output.dtype == ms.float32
        assert np.isfinite(output.asnumpy()).all()
    np.testing.assert_array_equal(value.asnumpy(), np.zeros((2, 3)))
