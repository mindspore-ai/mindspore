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
"""Guard retained scatter primitives and Tensor methods after wrapper removal."""
import numpy as np
import pytest

import mindspore as ms
from mindspore import nn, ops
from mindspore.ops import function, functional
from mindspore.ops.function import array_func
from mindspore.ops.operations import _scatter_ops
from tests.mark_utils import arg_mark


CASES = [('mul', _scatter_ops.ScatterMul, [16, 10]),
         ('div', _scatter_ops.ScatterDiv, [4, 2.5]),
         ('max', _scatter_ops.ScatterMax, [8, 5]),
         ('min', _scatter_ops.ScatterMin, [2, 2])]


@pytest.fixture(autouse=True)
def restore_context():
    """Avoid leaking execution context to other tests."""
    mode = ms.get_context('mode')
    target = ms.get_context('device_target')
    ms.set_context(device_target='CPU')
    yield
    ms.set_context(mode=mode, device_target=target)


class ScatterNet(nn.Cell):
    """Keep a mutable Parameter to check primitive side effects."""
    def __init__(self, primitive):
        super().__init__()
        self.weight = ms.Parameter(ms.Tensor([8, 5], ms.float32), name='weight')
        self.scatter = primitive()

    def construct(self, indices, updates):
        return self.scatter(self.weight, indices, updates)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', mem_peak=0)
@pytest.mark.parametrize('suffix', ['mul', 'div', 'max', 'min'])
def test_removed_scatter_wrappers(suffix):
    """
    Feature: Removal of deprecated scatter function wrappers.
    Description: Inspect all public function export routes.
    Expectation: Removed functions are absent; unrelated scatter functions remain.
    """
    for module in (ops, function, functional, array_func):
        assert not hasattr(module, 'scatter_' + suffix)
        assert 'scatter_' + suffix not in getattr(module, '__all__', ())
        assert callable(module.scatter_add)
        assert callable(getattr(module, 'tensor_scatter_' + suffix))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', mem_peak=0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('suffix,primitive,expected', CASES)
def test_retained_scatter_primitive(mode, suffix, primitive, expected):
    """
    Feature: Retained stateful scatter primitives.
    Description: Update two Parameter rows in graph and pynative modes.
    Expectation: Output and mutated Parameter match NumPy reference values.
    """
    ms.set_context(mode=mode)
    assert not hasattr(ops, 'Scatter' + suffix.capitalize())
    net = ScatterNet(primitive)
    output = net(ms.Tensor([0, 1], ms.int32), ms.Tensor([2, 2], ms.float32))
    np.testing.assert_allclose(output.asnumpy(), expected)
    np.testing.assert_allclose(net.weight.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', mem_peak=0)
@pytest.mark.parametrize('suffix,primitive,expected', CASES)
def test_retained_tensor_scatter_method(suffix, primitive, expected):
    """
    Feature: Retained Tensor scatter methods.
    Description: Call Tensor methods with ND indices, independently of stateful primitives.
    Expectation: Results match the reference and the original Tensor is unchanged.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    assert callable(primitive)
    value = ms.Tensor([8, 5], ms.float32)
    output = getattr(value, 'scatter_' + suffix)(ms.Tensor([[0], [1]], ms.int32),
                                               ms.Tensor([2, 2], ms.float32))
    np.testing.assert_allclose(output.asnumpy(), expected)
    np.testing.assert_array_equal(value.asnumpy(), [8, 5])
