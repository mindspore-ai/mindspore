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
"""Regression tests for the existing ops.vstack API."""
import numpy as np
import pytest
import mindspore as ms
from mindspore import nn, ops, Tensor
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def construct(self, x, y):
        return ops.vstack((x, y))


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'cpu_linux'],
          level_mark='level2', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('shape', [(3,), (2, 3)])
def test_vstack_legacy(mode, shape):
    """
    Feature: Existing ops.vstack API.
    Description: Preserve vector and matrix stacking after adding mint.vstack.
    Expectation: Values, shape and dtype match NumPy.
    """
    ms.set_context(mode=mode)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    y = x + 10
    output = Net()(Tensor(x), Tensor(y))
    expected = np.vstack((x, y))
    assert output.shape == expected.shape
    assert output.dtype == ms.float32
    np.testing.assert_array_equal(output.asnumpy(), expected)
