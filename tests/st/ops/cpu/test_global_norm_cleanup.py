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
"""Protect internal global clipping after deleting its public aliases."""
import importlib
import numpy as np
import pytest
import mindspore as ms
from mindspore import nn, ops
from mindspore.ops import composite, function, functional
from mindspore.ops.function import clip_func
from mindspore.ops.function.clip_func import _clip_by_global_norm
from tests.mark_utils import arg_mark

thor_module = importlib.import_module('mindspore.nn.optim.thor')


@pytest.fixture(autouse=True)
def restore_context():
    """Restore global execution settings."""
    mode = ms.get_context('mode')
    target = ms.get_context('device_target')
    ms.set_context(device_target='CPU')
    yield
    ms.set_context(mode=mode, device_target=target)


class InternalClipNet(nn.Cell):
    """Use the retained internal algorithm without a public alias."""
    def construct(self, x, y):
        return _clip_by_global_norm((x, y), 1.0)


class ThorClipNet(nn.Cell):
    """Exercise the actual THOR clipping helper."""
    def __init__(self, enabled):
        super().__init__()
        self.enabled = enabled

    def construct(self, x, y):
        return thor_module.clip_gradient(self.enabled, (x, y))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', mem_peak=0)
def test_global_norm_public_aliases_removed():
    """
    Feature: Global norm public API cleanup.
    Description: Inspect all public aliases and the retained parallel marker.
    Expectation: Public aliases are absent and grad_scale remains enabled.
    """
    for module in (ops, function, functional, composite, clip_func):
        assert not hasattr(module, 'clip_by_global_norm')
        assert 'clip_by_global_norm' not in getattr(module, '__all__', ())
    assert clip_func.expand_dims.attrs['grad_scale'] is True
    assert callable(ops.clip_by_value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', mem_peak=0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [ms.float16, ms.float32])
@pytest.mark.parametrize('values', [(0.0, 0.0), (0.3, 0.4), (3.0, 4.0)])
def test_internal_global_norm(mode, dtype, values):
    """
    Feature: Retained internal global norm algorithm.
    Description: Clip zero, below-threshold and oversized gradients in both execution modes.
    Expectation: Results match a float32 L2 reference, preserve dtype and leave inputs intact.
    """
    ms.set_context(mode=mode)
    inputs = tuple(ms.Tensor([v], dtype) for v in values)
    raw = np.array([v.asnumpy()[0] for v in inputs], np.float32)
    expected = raw / max(float(np.linalg.norm(raw)), 1.0)
    outputs = InternalClipNet()(*inputs)
    for index, output in enumerate(outputs):
        assert output.dtype == dtype
        np.testing.assert_allclose(output.asnumpy(), [expected[index]], rtol=1e-3, atol=1e-4)
        np.testing.assert_array_equal(inputs[index].asnumpy(), [raw[index]])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', mem_peak=0)
@pytest.mark.parametrize('clip_norm,use_norm', [(0.0, None), (-1.0, None), (1.0, 2.0)])
def test_internal_global_norm_validation(clip_norm, use_norm):
    """
    Feature: Retained internal global norm input validation.
    Description: Supply a nonpositive threshold or unsupported use_norm argument.
    Expectation: The existing ValueError behavior is preserved.
    """
    with pytest.raises(ValueError):
        _clip_by_global_norm((ms.Tensor([1], ms.float32),), clip_norm, use_norm)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', mem_peak=0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('enabled', [False, True])
def test_thor_global_clipping(mode, enabled, monkeypatch):
    """
    Feature: THOR retains its original global clipping path.
    Description: Exercise the real helper with global clipping enabled or bypassed.
    Expectation: Enabled clipping returns 0.6/0.8; disabled clipping preserves 3/4.
    """
    ms.set_context(mode=mode)
    monkeypatch.setattr(thor_module, 'IS_ENABLE_GLOBAL_NORM', True)
    outputs = ThorClipNet(enabled)(ms.Tensor([3], ms.float32), ms.Tensor([4], ms.float32))
    for output, expected in zip(outputs, (0.6, 0.8) if enabled else (3, 4)):
        np.testing.assert_allclose(output.asnumpy(), [expected], rtol=1e-6)
