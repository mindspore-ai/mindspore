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
"""Public sparse API retirement must preserve ordinary sparse-gradient training."""
import importlib
import numpy as np
import pytest
import mindspore as ms
from mindspore import nn, ops
from mindspore.common import RowTensor
from mindspore.common._sparse_tensor import RowTensorInner
from mindspore.ops._grad_experimental.grad_array_ops import get_bprop_sparse_gather_v2
from tests.mark_utils import arg_mark


@pytest.fixture(autouse=True)
def restore_mode():
    previous = ms.get_context('mode')
    yield
    ms.set_context(mode=previous)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_retired_public_sparse_surface():
    """
    Feature: Public sparse API retirement.
    Description: Check constructors, conversion methods and dedicated wrappers.
    Expectation: Removed APIs cannot be imported; shared operators remain available.
    """
    for module in (ms, ms.common):
        for name in ('COOTensor', 'CSRTensor', 'SparseTensor', 'RowTensorInner'):
            assert not hasattr(module, name), (module.__name__, name)
    assert not hasattr(ms, 'RowTensor')
    assert 'RowTensor' not in ms.common.__all__ and 'RowTensor' not in ms.__all__
    assert RowTensor is RowTensorInner
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('mindspore.common.sparse_tensor')
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('mindspore.ops.function.sparse_unary_func')
    for module in (ops, ops.functional, ops.function):
        assert not any(name.startswith(('_csr', '_coo', '_row_tensor', '_make_row',
                                        '_make_csr', '_make_coo', '_make_sparse', '_dense_to_sparse'))
                       for name in module.__all__)
        for name in ('csr_mul', 'csr_mv', 'coo_add', 'coo_concat', 'csr_sin', 'coo_abs',
                     'make_coo_tensor', 'make_csr_tensor', 'make_row_tensor', 'make_sparse_tensor',
                     'dense_to_sparse_coo', 'dense_to_sparse_csr'):
            assert not hasattr(module, name), (module.__name__, name)
    assert not hasattr(ms.Tensor, 'to_coo')
    assert not hasattr(ms.Tensor, 'to_csr')
    assert not hasattr(nn, 'SparseToDense')
    rewrite = importlib.import_module('mindspore.rewrite')
    for name in ('sparsify', 'ArgType', 'SparseFunc'):
        assert not hasattr(rewrite, name)
    assert hasattr(rewrite, 'SymbolTree')
    assert hasattr(nn, 'SparseTensorDenseMatmul')
    # This branch is independent of MapParameter and experimental.optim cleanup.
    assert hasattr(ms.experimental, 'MapParameter')
    assert hasattr(ms.experimental.optim, 'Adam')


class SparseParameters(nn.Cell):
    """Parameters updated by the retained sparse optimizer paths."""
    def __init__(self):
        super().__init__()
        self.weight = ms.Parameter(ms.Tensor(np.ones((4, 2), np.float32)), name='weight')

    def construct(self):
        return self.weight


class RemovedCOOConversion(nn.Cell):
    def construct(self, value):
        return value.to_coo()


class RemovedCSRConversion(nn.Cell):
    def construct(self, value):
        return value.to_csr()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('network', [RemovedCOOConversion, RemovedCSRConversion])
def test_removed_tensor_conversion(mode, network):
    """
    Feature: Public sparse conversion retirement in both execution modes.
    Description: Attempt the removed dense Tensor methods inside a Cell.
    Expectation: Python and graph method resolution both reject the old entry.
    """
    ms.set_context(mode=mode)
    with pytest.raises((AttributeError, RuntimeError), match='to_coo|to_csr'):
        network()(ms.Tensor([[1., 0.], [0., 2.]], ms.float32))


class SparseOptimizerStep(nn.Cell):
    """Build the sparse gradient within the optimizer graph, as training does."""
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer

    @ms.jit
    def construct(self, indices, values):
        gradient = RowTensorInner(indices, values, (4, 2))
        return self.optimizer((gradient,))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('optimizer_type', [nn.Adam, nn.LazyAdam, nn.FTRL])
def test_internal_sparse_optimizer(mode, optimizer_type, tmp_path):
    """
    Feature: Retained sparse optimizer dispatch and checkpointing.
    Description: Apply an internal sparse gradient and restore updated parameters.
    Expectation: Updated rows match reference; untouched rows and checkpoint values are preserved.
    """
    ms.set_context(mode=mode)
    net = SparseParameters()
    indices = ms.Tensor([0, 2], ms.int32)
    values = ms.Tensor(np.ones((2, 2), np.float32))
    optimizer = optimizer_type(net.trainable_params(), learning_rate=0.1)
    optimizer.target = 'CPU'
    SparseOptimizerStep(optimizer)(indices, values)
    expected = np.ones((4, 2), np.float32)
    if optimizer_type is nn.FTRL:
        updated = -(1.0 - (np.sqrt(1.1) - np.sqrt(0.1)) / 0.1) * 0.1 / np.sqrt(1.1)
    else:
        updated = 1.0 - 0.1 / (1.0 + 1e-8 / np.sqrt(1 - 0.999))
    expected[[0, 2]] = updated
    np.testing.assert_allclose(net.weight.asnumpy(), expected, rtol=1e-5, atol=1e-6)
    filename = str(tmp_path / 'lookup.ckpt')
    ms.save_checkpoint(net, filename)
    net.weight.set_data(ms.Tensor(np.zeros((4, 2), np.float32)))
    missing, unused = ms.load_param_into_net(net, ms.load_checkpoint(filename), strict_load=True)
    assert not missing and not unused
    np.testing.assert_allclose(net.weight.asnumpy(), expected, rtol=1e-5, atol=1e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_sparse_lookup_python_bprop():
    """
    Feature: Retained Python sparse lookup backward rule.
    Description: Execute the registered rule directly with a dense gather result.
    Expectation: The rule constructs the expected private RowTensor gradient.
    """
    # SparseGatherV2 has no CPU forward kernel; this checks its Python backward
    # rule, not end-to-end automatic differentiation on the target devices.
    ms.set_context(mode=ms.PYNATIVE_MODE)
    weight = ms.Tensor(np.ones((4, 2), np.float32))
    indices = ms.Tensor([0, 2], ms.int32)
    output = ops.gather(weight, indices, 0)
    backward = get_bprop_sparse_gather_v2(ops.SparseGatherV2())
    gradient, _, _ = backward(weight, indices, 0, output, ops.ones_like(output))
    assert isinstance(gradient, RowTensorInner)
    np.testing.assert_array_equal(gradient.indices.asnumpy(), [0, 2])
    np.testing.assert_allclose(gradient.values.asnumpy(), np.ones((2, 2), np.float32))
    assert gradient.dense_shape == (4, 2)


class SparseScale(nn.Cell):
    """The retained sparse loss-scale formula used by training wrappers."""
    def construct(self, scale, grad):
        result = RowTensor(grad.indices,
                           grad.values * ops.cast(ops.reciprocal(scale), ops.dtype(grad.values)),
                           grad.dense_shape)
        return result.indices, result.values, result.dense_shape


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_internal_sparse_scaling(mode):
    """
    Feature: Sparse gradient scaling.
    Description: Reconstruct a scaled internal gradient and clone its representation.
    Expectation: The MindFormers-compatible constructor preserves indices, shape and scaled values.
    """
    ms.set_context(mode=mode)
    row = RowTensorInner(ms.Tensor([1], ms.int32), ms.Tensor([[4., 8.]], ms.float32), (3, 2))
    clone = RowTensorInner(row_tensor=row)
    indices, values, shape = SparseScale()(ms.Tensor(4., ms.float32), clone)
    np.testing.assert_array_equal(indices.asnumpy(), [1])
    np.testing.assert_allclose(values.asnumpy(), [[1., 2.]])
    assert shape == (3, 2)
