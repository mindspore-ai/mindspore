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
"""Guard retained ND scatter operations after removing function wrappers."""
import numpy as np
import pytest

import mindspore as ms
from mindspore import nn, ops
from mindspore.ops import function, functional
from mindspore.ops.function import array_func
from mindspore.ops.operations import _scatter_nd_ops
from tests.mark_utils import arg_mark


CASES = [('add', _scatter_nd_ops.ScatterNdAdd, [[10, 6], [7, 4]]),
         ('sub', _scatter_nd_ops.ScatterNdSub, [[6, 2], [3, 0]]),
         ('mul', _scatter_nd_ops.ScatterNdMul, [[16, 8], [10, 4]]),
         ('div', _scatter_nd_ops.ScatterNdDiv, [[4, 2], [2.5, 1]]),
         ('max', _scatter_nd_ops.ScatterNdMax, [[8, 4], [5, 2]]),
         ('min', _scatter_nd_ops.ScatterNdMin, [[2, 2], [2, 2]])]


@pytest.fixture(autouse=True)
def restore_context():
    """Restore execution settings after each guard."""
    mode = ms.get_context('mode')
    target = ms.get_context('device_target')
    ms.set_context(device_target='CPU')
    yield
    ms.set_context(mode=mode, device_target=target)


class ScatterNdNet(nn.Cell):
    """Retained primitive with a mutable Parameter and selectable locking."""
    def __init__(self, primitive, use_locking):
        super().__init__()
        self.weight = ms.Parameter(ms.Tensor([[8, 4], [5, 2]], ms.float32), name='weight')
        self.scatter = primitive(use_locking=use_locking)

    def construct(self, indices, updates):
        return self.scatter(self.weight, indices, updates)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', mem_peak=0)
@pytest.mark.parametrize('suffix', ['add', 'sub', 'mul', 'div', 'max', 'min'])
def test_removed_scatter_nd_wrappers(suffix):
    """
    Feature: ND scatter function cleanup.
    Description: Inspect ops, functional, function and defining-module exports.
    Expectation: Removed wrappers are absent while unrelated scatter functions remain.
    """
    for module in (ops, function, functional, array_func):
        name = 'scatter_nd_' + suffix
        assert not hasattr(module, name)
        assert name not in getattr(module, '__all__', ())
        assert callable(module.scatter_update)
        assert callable(module.scatter_add)
        assert callable(getattr(module, 'tensor_scatter_' + suffix))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', mem_peak=0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('use_locking', [False, True])
@pytest.mark.parametrize('suffix,primitive,expected', CASES)
def test_retained_scatter_nd_primitive(mode, use_locking, suffix, primitive, expected):
    """
    Feature: Retained stateful ND scatter primitives.
    Description: Update Parameter slices with both locking options and execution modes.
    Expectation: Result and Parameter state match reference values.
    """
    ms.set_context(mode=mode)
    assert not hasattr(ops, 'ScatterNd' + suffix.capitalize())
    net = ScatterNdNet(primitive, use_locking)
    output = net(ms.Tensor([[0], [1]], ms.int32), ms.Tensor([[2, 2], [2, 2]], ms.float32))
    np.testing.assert_allclose(output.asnumpy(), expected)
    np.testing.assert_allclose(net.weight.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', mem_peak=0)
@pytest.mark.parametrize('suffix,primitive,expected', CASES)
def test_retained_tensor_scatter_functions(suffix, primitive, expected):
    """
    Feature: Independent functional Tensor scatter operations.
    Description: Apply tensor_scatter operations to slices of an ordinary Tensor.
    Expectation: Results match the reference and the input remains unchanged.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    assert callable(primitive)
    value = ms.Tensor([[8, 4], [5, 2]], ms.float32)
    output = getattr(ops, 'tensor_scatter_' + suffix)(
        value, ms.Tensor([[0], [1]], ms.int32), ms.Tensor([[2, 2], [2, 2]], ms.float32))
    np.testing.assert_allclose(output.asnumpy(), expected)
    np.testing.assert_array_equal(value.asnumpy(), [[8, 4], [5, 2]])
