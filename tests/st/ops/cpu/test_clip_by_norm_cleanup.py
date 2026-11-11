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
"""Protect independent clipping implementations after function cleanup."""
import numpy as np
import pytest
import mindspore as ms
from mindspore import nn, ops
from mindspore.ops import function, functional
from mindspore.ops.function import clip_func
from mindspore.ops.function.clip_func import _clip_by_global_norm
from tests.mark_utils import arg_mark


@pytest.fixture(autouse=True)
def restore_context():
    """Restore execution settings after each test."""
    mode = ms.get_context('mode')
    target = ms.get_context('device_target')
    ms.set_context(device_target='CPU')
    yield
    ms.set_context(mode=mode, device_target=target)


class GlobalNormNet(nn.Cell):
    """Retain the independent global clipping route."""
    def construct(self, x, y):
        return _clip_by_global_norm((x, y), 1.0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', mem_peak=0)
def test_clip_by_norm_exports_removed():
    """
    Feature: Deprecated clip_by_norm function removal.
    Description: Inspect the defining module and all public function export routes.
    Expectation: Retired entries are absent and value clipping remains callable.
    """
    for module in (ops, function, functional, clip_func):
        assert not hasattr(module, 'clip_by_norm')
        assert 'clip_by_norm' not in getattr(module, '__all__', ())
        assert not hasattr(module, 'clip_by_global_norm')
        assert callable(module.clip_by_value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', mem_peak=0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('values,expected', [([3, 4], [0.6, 0.8]), ([0, 0.5], [0, 0.5])])
def test_retained_nn_clip_by_norm(mode, values, expected):
    """
    Feature: Retained nn.ClipByNorm and internal primitive.
    Description: Clip a single vector above or below the threshold along axis zero.
    Expectation: Only the oversized vector is rescaled and the input is unchanged.
    """
    ms.set_context(mode=mode)
    value = ms.Tensor(values, ms.float32)
    result = nn.ClipByNorm(axis=0)(value, ms.Tensor([1], ms.float32))
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=1e-6)
    np.testing.assert_array_equal(value.asnumpy(), values)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', mem_peak=0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_retained_global_norm(mode):
    """
    Feature: Retained clip_by_global_norm.
    Description: Clip a tuple of tensors using the shared global L2 norm.
    Expectation: The two outputs equal 0.6 and 0.8, with the original dtype preserved.
    """
    ms.set_context(mode=mode)
    result = GlobalNormNet()(ms.Tensor([3], ms.float32), ms.Tensor([4], ms.float32))
    for output, expected in zip(result, [0.6, 0.8]):
        np.testing.assert_allclose(output.asnumpy(), [expected], rtol=1e-6)
        assert output.dtype == ms.float32
