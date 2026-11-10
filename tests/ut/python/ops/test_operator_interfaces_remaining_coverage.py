# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""CPU contracts for the remaining Python primitive interfaces.

Native AOT compilation is tested only at the subprocess protocol boundary.
Shape/type inference is executed directly and does not claim device arithmetic.
"""
# Unit coverage intentionally exercises internal Python inference and wrapper helpers.
# pylint: disable=protected-access
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
import types

import numpy as np
import pytest
from mindspore import Tensor, context, jit
from mindspore.common import dtype as dt
from mindspore.ops import primitive as prim
from mindspore.ops import _primitive_cache as cache
from mindspore.ops._register_for_op import Registry, PyFuncRegistry
from mindspore.ops import deprecated as dep
from mindspore.ops.operations import nn_ops as nn
from mindspore.ops.operations import math_ops as math
from mindspore.ops.operations import array_ops as array
from mindspore.ops.operations import custom_ops as custom
from mindspore.ops.operations import _custom_grad as grad
from mindspore.ops.operations import _embedding_cache_ops as embed
from mindspore.ops.operations import _rl_inner_ops as rl
from mindspore.ops.operations import _sequence_ops as seq
from mindspore.ops.operations import inner_ops as inner
from mindspore.ops.operations import debug_ops as debug
from mindspore.ops.operations import image_ops as image
from mindspore.ops.operations import sparse_ops as sparse

F32, F16, I32, I64 = [dt.TensorType(t) for t in (dt.float32, dt.float16, dt.int32, dt.int64)]


def abstract(shape, dtype=F32, value=None):
    return {'shape': shape, 'dtype': dtype, 'value': value}


@pytest.fixture(autouse=True, scope='module', name='cpu_context')
def fixture_cpu_context():
    old = {k: context.get_context(k) for k in ('mode', 'device_target')}
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    yield
    context.set_context(**old)


@pytest.fixture(name='custom_state')
def fixture_custom_state(monkeypatch, tmp_path):
    """Restore the mutable Custom registries and isolate all generated paths."""
    for name in ('compiled_bin', 'registered_func', 'attr_dict', 'tbe_path_checked',
                 'tbe_path_failed', 'op_path_in_cache', 'custom_aot_warning'):
        monkeypatch.setattr(custom.Custom, name, copy.deepcopy(getattr(custom.Custom, name)))
    monkeypatch.setenv('MS_COMPILER_CACHE_PATH', str(tmp_path / 'meta'))
    # Import at the test boundary so fixture isolation is established first.
    # pylint: disable-next=import-outside-toplevel
    from mindspore.ops.operations import _pyfunc_registry
    monkeypatch.setattr(_pyfunc_registry.CustomPyFuncRegistry, '_instance',
                        _pyfunc_registry.CustomPyFuncRegistry(), raising=False)
    yield


@pytest.mark.parametrize('n', range(1, 11))
@pytest.mark.parametrize('out,dout,tail', [(11, 13, (11, 13)), ((11, 12), (13, 14), (11, 12, 13, 14))])
def test_custom_gradient_argument_protocol(n, out, dout, tail):
    """
    Feature: Custom gradient argument protocol.
    Description: Real bprop wrappers preserve input ordering and multi-output sensitivities.
    Expectation: Real bprop wrappers preserve input ordering and multi-output sensitivities.
    """
    seen = []
    def gradient(*args):
        seen.append(args)
        return tuple(v * 2 for v in args[:n])
    inputs = tuple(range(1, n + 1))
    result = grad.autodiff_bprop(n)(gradient)(*inputs, out, dout)
    assert result == tuple(range(2, 2 * n + 1, 2))
    assert seen == [inputs + tail]


def test_custom_gradient_registry_absent_and_mixed_outputs():
    """
    Feature: Custom gradient registry absent and mixed outputs.
    Description: Check custom gradient registry absent and mixed outputs.
    Expectation: Missing gradient lookups return None and mixed scalar/tuple outputs flatten in their original
        order.
    """
    assert grad.Registry().get(0) is None
    assert grad.autodiff_bprop(11) is None
    assert grad.get_outs((1, 2), 3) == (1, 2, 3)
    assert grad.get_outs(1, (2, 3)) == (1, 2, 3)


def test_registry_dispatch_and_missing_key():
    """
    Feature: Registry dispatch and missing key.
    Description: Check registry dispatch and missing key.
    Expectation: String and primitive registrations resolve to the expected callables, replacement keeps the
        correct registry size, and missing keys honor defaults or raise.
    """
    registry = Registry()
    def named(x):
        return x + 1
    assert registry.register('Add')(named) is named
    assert registry.get('Add', None)(2) == 3
    assert registry.get(math.Add(), None) is named
    def typed(x):
        return x * 3
    registry.register(math.Add)(typed)
    assert registry.get(math.Add(), None)(2) == 6
    registry.register(int)(typed)
    assert len(registry) == 2
    assert registry.get(math.Sub(), 'missing') == 'missing'
    functions = PyFuncRegistry()
    functions.register(7, named)
    assert functions.get(7)(4) == 5
    with pytest.raises(ValueError, match='not registered'):
        functions.get(8)


def test_embedding_cache_contracts():
    """
    Feature: Embedding cache contracts.
    Description: Check embedding cache contracts.
    Expectation: Cache primitives preserve explicit dynamic shapes and tensor dtypes and reject invalid table
        dimensions, indices, and operand types.
    """
    op = embed.SubAndFilter()
    assert op.check_shape([4], [], []) == ((-1,), (-1,))
    assert op.check_dtype(I32, I32, I32) == I32
    with pytest.raises(TypeError):
        op.check_dtype(F32, I32, I32)
    swap = embed.CacheSwapTable()
    assert swap.check_shape([8, 3], [2], [2, 3]) == [2, 3]
    assert swap.check_dtype(F32, I64, F16) == F16
    with pytest.raises(ValueError, match='shape must be 2'):
        swap.check_shape([8], [2], [2, 3])
    with pytest.raises(TypeError):
        swap.check_dtype(F32, F32, F16)
    mapping = embed.MapCacheIdx()
    args = [abstract([8, 4], I32), abstract([2, 3], I32)] + [abstract([], I32)] * 3
    assert mapping.__check__(*args) == {'shape': ([2, 3], -1, -1, -1), 'dtype': (I32,) * 4, 'value': None}
    with pytest.raises(ValueError, match='dimension'):
        mapping.__check__(abstract([8], I32), *args[1:])
    with pytest.raises(TypeError):
        mapping.__check__(args[0], abstract([2, 3], I64), *args[2:])
    assign = embed.DynamicAssign()
    assert assign.check_dtype(F32, F32) is None
    assert assign.check_dtype(dt.type_refkey, F16) is None
    with pytest.raises(TypeError):
        assign.check_dtype(dt.string, F32)


@pytest.mark.parametrize('bidirectional,hidden_shape,out',
                         [(False, [2, 3, 4], (5, 3, 4)), (True, [4, 3, 4], (5, 3, 8))])
def test_gru_shape_dtype(bidirectional, hidden_shape, out):
    """
    Feature: Gru shape dtype.
    Description: Check gru shape dtype.
    Expectation: GRU inference retains the expected time, batch, direction, hidden-state shapes, and float32
        outputs while rejecting incompatible metadata.
    """
    op = rl.GRUV2(6, 4, 2, True, bidirectional, 0.25)
    assert op.infer_shape([5, 3, 6], hidden_shape, [100], [3]) == (out, hidden_shape, (1, 1), (1, 1))
    assert op.infer_dtype(F32, F32, F32, I32) == (F32,) * 4
    with pytest.raises(TypeError):
        op.infer_dtype(F32, F16, F32, I32)
    with pytest.raises(TypeError):
        op.infer_dtype(F32, F32, F32, I64)
    for x, h, lengths in [([5, 6], hidden_shape, [3]), ([5, 3, 7], hidden_shape, [3]),
                           ([5, 3, 6], [1, 3, 4], [3]), ([5, 3, 6], hidden_shape, [4])]:
        with pytest.raises(ValueError):
            op.infer_shape(x, h, [100], lengths)


@pytest.mark.parametrize('name', ['GRUV2', 'LSTMV2', 'CudnnGRU'])
def test_recurrent_constructor_validation(name):
    """
    Feature: Recurrent constructor validation.
    Description: Check recurrent constructor validation.
    Expectation: Valid recurrent sizes and direction flags are retained, and invalid constructor arguments are
        rejected.
    """
    cls = getattr(rl, name)
    op = cls(6, 4, 2, True, True, 0.5)
    assert op.input_size == 6 and op.hidden_size == 4 and op.bidirectional is True
    for args in [(0, 4, 2, True, False, 0.0), (6, 4, 2, 1, False, 0.0), (6, 4, 2, True, False, 1.1)]:
        with pytest.raises((TypeError, ValueError)):
            cls(*args)


def test_optimizer_inference_contracts():
    """
    Feature: Optimizer inference contracts.
    Description: Check optimizer inference contracts.
    Expectation: Optimizer inference preserves expected parameter/state shapes, reference dtypes, and attributes
        and rejects incompatible operands.
    """
    momentum = inner.FusedWeightScaleApplyMomentum()
    assert momentum.infer_shape([], [], [2, 3], [2, 3], [], [2, 3], []) == [2, 3]
    assert momentum.infer_dtype(F32, F32, F16, F16, F32, F16, F32) == F16
    assert momentum.infer_dtype(F32, F32, dt.type_refkey, dt.type_refkey, F32, F16, F32) == dt.type_refkey
    with pytest.raises(TypeError):
        momentum.infer_dtype(F32, F32, I32, I32, F32, F32, F32)
    adam = inner.FusedCastAdamWeightDecay(use_locking=True)
    shapes = [[2, 3]] * 3 + [[]] * 5 + [[2, 3], []]
    assert adam.infer_shape(*shapes) == ([2, 3],) * 3
    types_ = [F16, F32, F32] + [F32] * 5 + [F16, F32]
    assert adam.infer_dtype(*types_) == (F16, F32, F32)
    shapes[1] = [6]
    with pytest.raises(ValueError):
        adam.infer_shape(*shapes)
    with pytest.raises(TypeError):
        inner.FusedCastAdamWeightDecay(use_locking=1)
    for cls, n in [(inner.FusedAdaFactor, 12), (inner.FusedAdaFactorWithGlobalNorm, 13)]:
        op = cls()
        args = [[]] * n
        args[6] = args[7] = [2, 3]
        assert op.infer_shape(*args) == [2, 3]
        assert op.infer_dtype(*([F32] * n)) == F32
        args[6] = [3]
        with pytest.raises(ValueError):
            op.infer_shape(*args)
    eod = inner.GenerateEodMask(8, 2, [1, 2])
    assert eod.n_pos == 8 and eod.n_step == [1, 2]


def test_sequence_python_values():
    """
    Feature: Sequence python values.
    Description: Check sequence python values.
    Expectation: Python sequence primitives preserve literal counts, indices, ranges, comparisons, and tensor
        values and reject unsupported sequence inputs.
    """
    assert seq.shape_mul()((2, 3, 4)) == 24
    assert seq.shape_mul()(()) == 1
    with pytest.raises(ValueError):
        seq.shape_mul()([2, 3])
    assert seq.SequenceCount()((1, 2, 1), 1) == 2
    assert seq.SequenceIndex()((1, 2, 1), 1, 1, 3) == 2
    assert tuple(seq.make_range()(1, 8, 2)) == (1, 3, 5, 7)
    assert seq.tuple_equal()((1, 2), (1, 2)) is True
    assert seq.list_equal()([1], [2]) is False
    assert seq.sequence_len()([2, 3]) == 2
    for value in [range(1, 4), (1, 2, 3)]:
        np.testing.assert_array_equal(seq.TupleToTensor().infer_value(value, dt.int32).asnumpy(), [1, 2, 3])
    for value in [None, (1, None), [1, 2]]:
        assert seq.TupleToTensor().infer_value(value, dt.int32) is None
    np.testing.assert_array_equal(seq.ListToTensor()([1, 2], dt.float32).asnumpy(), [1., 2.])
    for value in [None, [1, None], (1, 2)]:
        with pytest.raises(RuntimeError):
            seq.ListToTensor()(value, dt.int32)
    for cls, inputs in [
    (seq.ListAppendAndInsertGrad, [
        'input_data', 'index']), (seq.ShapeMulGrad, [
            'input_data', 'dout']), (seq.SequenceSliceGrad, [
                'dy', 'x', 'start', 'stop', 'step']), (seq.SequenceAddN, ['inputs'])]:
        assert cls().input_names == inputs
    assert seq.SequenceUnstack(-1, 3).axis == -1
    with pytest.raises(TypeError):
        seq.SequenceUnstack(0, '3')


@pytest.mark.parametrize('axis,keep,expected', [((), False, []), ([], True, [1, 1, 1]), (-1, False, [2, 3]),
                                              ((0, -1), True, [1, 3, 1]), ([1, 1], False, [2, 4])])
def test_reduce_shape(axis, keep, expected):
    """
    Feature: Reduce shape.
    Description: Check reduce shape.
    Expectation: Reduction inference returns the independently specified shape for the selected axes and
        keep-dimensions flag.
    """
    assert math._infer_shape_reduce([2, 3, 4], axis, keep, 'ReduceMax') == expected


@pytest.mark.parametrize('name,expected', [('ReduceMax', [4, 8]), ('ReduceMin', [1, 2]),
    ('ReduceProd', [4, 16]), ('ReduceMean', [2.5, 5]), ('ReduceAll', [True, True]), ('ReduceAny', [True, True])])
def test_reduce_constant_folding(name, expected):
    # Generated public aliases bypass this legacy base; a named subclass uses its real Python contract.
    """
    Feature: Reduce constant folding.
    Description: Check reduce constant folding.
    Expectation: Constant reductions preserve the expected values and shapes; unresolved values or axes return
        None.
    """
    op = type(name, (math._Reduce,), {})(keep_dims=False)
    value = Tensor([[1., 4.], [2., 8.]], dt.float32)
    np.testing.assert_array_equal(op.infer_value(value, 1).asnumpy(), expected)
    assert op.infer_value(None, 1) is None
    assert op.infer_value(value, None) is None
    assert op.infer_value(value, [0, 0]).shape == (2,)
    assert op.infer_value(value, ()).shape == ()


def test_math_binary_dtype_and_constants():
    """
    Feature: Math binary dtype and constants.
    Description: Check math binary dtype and constants.
    Expectation: Binary math inference preserves complex promotion, broadcast shapes, comparison dtypes, and
        explicit folded constants while rejecting invalid metadata.
    """
    complex64 = dt.TensorType(dt.complex64)
    assert math._MathBinaryOp.do_infer_dtype(complex64, F32) == complex64
    assert math._MathBinaryOp.do_infer_dtype(F32, complex64) == complex64
    with pytest.raises(TypeError, match='Complex'):
        math._MathBinaryOp.do_infer_dtype(complex64, I32)
    with pytest.raises(TypeError, match='Complex'):
        math._MathBinaryOp.do_infer_dtype(complex64, dt.float32)
    assert math._MathBinaryOp.do_infer_dtype(F32, F32) == F32
    op = math.Sub()
    assert op._convert_back_shape([2, 3], (-1, 3)) == (-1, 3)
    assert op._convert_back_shape([2, 3], Tensor([2, 3])) == [2, 3]
    assert op._convert_back_shape([2, 3], [2, 3]) == [2, 3]
    for cls, expected in [(math.Sub, [4., 4.]), (math.Mod, [0., 1.]), (math.Xdivy, [5., 7./3.])]:
        instance = cls()
        result = instance.infer_value(Tensor([5., 7.], dt.float32), Tensor([1., 3.], dt.float32))
        np.testing.assert_allclose(result.asnumpy(), expected)
        assert instance.infer_value(None, None) is None
    assert math.Xdivy().infer_shape([2, 1], [3]) == [2, 3]
    assert math.Xdivy().infer_dtype(F32, F32) == F32
    assert math._BitwiseBinaryOp().infer_dtype(I32, I32) == I32
    with pytest.raises(TypeError):
        math._BitwiseBinaryOp().infer_dtype(F32, F32)
    assert math._LogicBinaryOp().infer_dtype(F32, F32) == dt.TensorType(dt.bool_)
    for cls in (math.AddN, math.AccumulateNV2):
        op = cls()
        value = Tensor([1.])
        assert op.check_elim([value]) == (True, value)
        assert op.check_elim([value, value]) == (False, None)
        with pytest.raises(TypeError):
            op.check_elim([1])
    nms = math.NMSWithMask()
    assert nms.infer_shape([4, 5]) == ([4, 5], (4,), (4,))
    assert nms.infer_shape([-1, 5]) == ([-1, 5], (-1,), (-1,))
    assert nms.infer_dtype(F32) == (F32, dt.int32, dt.bool_)
    with pytest.raises(ValueError):
        nms.infer_shape([4, 6])


@pytest.mark.parametrize('cls,kwargs,attrs', [
    (math.Einsum, {'equation':'ij,jk->ik'}, {'equation':'ij,jk->ik'}),
    (math.Quantile, {'keep_dims':None, 'ignore_nan':None}, {'dim':10000, 'keep_dims':False, 'ignore_nan':False}),
    (math.Lu, {'output_idx_type':dt.int64}, {'output_idx_type':dt.int64}),
    (math.TrilIndices, {'row':2, 'col':3}, {'row':2, 'col':3}),
    (math.TriuIndices, {'row':2, 'col':3}, {'row':2, 'col':3}),
    (math.Ormqr, {'left':False, 'transpose':True}, {'left':False, 'transpose':True}),
    (math.Renorm, {'p':2, 'dim':0, 'maxnorm':1.0}, {'p':2.0}),
    (array.Unsqueeze, {'axis':(0, 2)}, {'axis':(0, 2)}),
    (array.LogSpace, {'steps':0, 'base':2, 'dtype':dt.float64}, {'steps':0, 'base':2, 'dtype':dt.float64}),
    (array.CountNonZero, {}, {'dims':[]}),
    (array.CountNonZero, {'dims':(0, 1)}, {'dims':(0, 1)}),
    (array.SplitV, {'size_splits':[2, -1], 'split_dim':1, 'num_split':2}, {'num_split':2}),
    (array.Unstack, {'axis':-1, 'num':2}, {'axis':-1, 'num':2}),
    (sparse.SparseSplit, {'num_split':2}, {'num_split':2}),
    (sparse.SparseMatrixSoftmax, {'dtype':dt.float32}, {'dtype':dt.float32}),
    (sparse.SparseMatrixSparseMatMul, {'transpose_a':True}, {'transpose_a':True}),
    (image.ExtractGlimpse, {'uniform_noise':False, 'noise':'gaussian'}, {'noise':'gaussian'}),
    (image.CropAndResizeGradImage, {'T':dt.float64, 'method':'nearest'}, {'max_Byte':2000000000}),
])
def test_constructor_attribute_contracts(cls, kwargs, attrs):
    """
    Feature: Constructor attribute contracts.
    Description: Check constructor attribute contracts.
    Expectation: Each constructor retains the independently specified public attribute value.
    """
    op = cls(**kwargs)
    for key, expected in attrs.items():
        assert getattr(op, key) == expected


@pytest.mark.parametrize('cls,kwargs', [
    (math.Einsum, {'equation':3}), (math.Einsum, {'equation':'a->b->c'}),
    (math.Bernoulli, {'seed':-2}), (math.Renorm, {'p':0, 'dim':0, 'maxnorm':1.0}),
    (math.TrilIndices, {'row':-1, 'col':2}), (math.TriuIndices, {'row':2, 'col':3, 'dtype':dt.float32}),
    (math.Ormqr, {'left':1}), (math.Quantile, {'dim':'0'}), (math._Reduce, {'keep_dims':1}),
    (array.LogSpace, {'steps':-1}), (array.LogSpace, {'dtype':dt.int32}),
    (array.CountNonZero, {'dims':['a']}), (array.SplitV, {'size_splits':[0], 'split_dim':0, 'num_split':1}),
    (sparse.SparseMatrixSoftmax, {'dtype':dt.int32}), (sparse.SparseSplit, {'num_split':1.0}),
    (sparse.SparseMatrixSparseMatMul, {'transpose_a':1}),
    (image.ExtractGlimpse, {'noise':'zero'}), (image.CropAndResizeGradImage, {'T':dt.int32}),
    (image.CropAndResizeGradImage, {'T':dt.float32, 'method':'bilinear_v2'}),
])
def test_invalid_constructor_contracts(cls, kwargs):
    """
    Feature: Invalid constructor contracts.
    Description: Check invalid constructor contracts.
    Expectation: Invalid constructor contracts raise (TypeError, ValueError).
    """
    with pytest.raises((TypeError, ValueError)):
        cls(**kwargs)


@pytest.mark.xfail(strict=True, raises=pytest.fail.Exception,
                   reason='SparseMatrixSparseMatMul validates adjoint_b twice and omits adjoint_a')
def test_sparse_adjoint_a_requires_boolean():
    """
    Feature: Sparse adjoint a requires boolean.
    Description: Check sparse adjoint a requires boolean.
    Expectation: Invalid sparse adjoint a requires boolean raise TypeError. Diagnostics identify adjoint_a.
    """
    with pytest.raises(TypeError, match='adjoint_a'):
        sparse.SparseMatrixSparseMatMul(adjoint_a=1)


@pytest.mark.parametrize('parts', [[], ['BAD'], ['VALUE_ROWIDS'], ['FIRST_DIM_SIZE'],
                                  ['FIRST_DIM_SIZE', 'ROW_SPLITS'], ['ROW_SPLITS', 'VALUE_ROWIDS'], [1]])
def test_ragged_partition_validation(parts):
    """
    Feature: Ragged partition validation.
    Description: Check ragged partition validation.
    Expectation: Invalid ragged partition arguments raise (ValueError, TypeError).
    """
    with pytest.raises((ValueError, TypeError)):
        sparse.RaggedTensorToTensor(parts)


@pytest.mark.parametrize('parts,n', [(['ROW_SPLITS'], 1), (['FIRST_DIM_SIZE', 'VALUE_ROWIDS', 'VALUE_ROWIDS'], 3)])
def test_ragged_partition_count(parts, n):
    """
    Feature: Ragged partition count.
    Description: Check ragged partition count.
    Expectation: Each ragged partition layout records the expected number of row-partition tensors.
    """
    assert sparse.RaggedTensorToTensor(parts).num_row_partition_tensors == n


def test_array_constant_folding_and_fill():
    """
    Feature: Array constant folding and fill.
    Description: Check array constant folding and fill.
    Expectation: Fill, dtype, diagonal, permutation, and sequence folds retain the explicit values and dtypes;
        unresolved or invalid metadata takes the specified fallback.
    """
    fill = array.Fill()
    for dims, value in [((2, 3), 7), (Tensor([2, 3], dt.int32), Tensor(7, dt.int32))]:
        np.testing.assert_array_equal(fill(dt.int32, dims, value).asnumpy(), [[7, 7, 7], [7, 7, 7]])
        np.testing.assert_array_equal(fill.infer_value(dt.int32, dims, value).asnumpy(), [[7, 7, 7], [7, 7, 7]])
    assert fill.infer_value(dt.int32, (2, None), 7) is None
    for args in [('bad', (2,), 1), (dt.int32, [2], 1), (dt.int32, (2,), 'x')]:
        with pytest.raises(TypeError):
            fill(*args)
    for value, expected_dtype in [((1, 2), dt.int32), ((1., 2.), dt.float32)]:
        out = array.TupleToArray().infer_value(value)
        assert out.dtype == expected_dtype
        np.testing.assert_array_equal(out.asnumpy(), [1, 2])
    with pytest.raises(TypeError):
        array.TupleToArray().infer_value((1, 2.))
    assert array.DType()(Tensor([1.], dt.float32)) == dt.float32
    with pytest.raises(TypeError):
        array.DType()(1)
    diag = array.DiagPart()
    np.testing.assert_array_equal(diag.infer_value(Tensor([[1, 2], [3, 4]])).asnumpy(), [1, 4])
    assert diag.infer_value(Tensor([1, 2])) is None
    assert diag.infer_value(None) is None
    inverse = array.InvertPermutation()
    assert inverse.__infer__(abstract([(), (), ()], dt.tuple_, (2, 0, 1)))['value'] == (1, 2, 0)
    for value in [None, (1, 1), (1, 2), (-1, 0)]:
        with pytest.raises(ValueError):
            inverse.__infer__(abstract([(), ()], dt.tuple_, value))
    for typ, shape in [(F32, [3]), (dt.tuple_, [(2,)])]:
        with pytest.raises(ValueError):
            inverse.__infer__(abstract(shape, typ, (1, 0)))


def test_scatter_validation_and_dynamic_shapes():
    """
    Feature: Scatter validation and dynamic shapes.
    Description: Check scatter validation and dynamic shapes.
    Expectation: Scatter inference preserves the input shape for valid dynamic indices and rejects incompatible
        updates, ranks, and index dtypes.
    """
    op = array._ScatterOp()
    assert op.infer_shape([4, 3], [2], [2, 3]) == [4, 3]
    assert op.infer_shape([4, 3], [-1], [8]) == [4, 3]
    assert op.infer_dtype(F32, I32, F32) == F32
    with pytest.raises(ValueError):
        op.infer_shape([4, 3], [2], [2, 4])
    with pytest.raises(TypeError):
        op.infer_dtype(F32, I64, F32)
    dynamic = array._ScatterOpDynamic()
    dynamic._check_scatter_shape([4, 3], [-2], [-2], 'Scatter')
    with pytest.raises(ValueError, match='dynamic'):
        dynamic._check_scatter_shape([-1, 3], [2], [2, 3], 'Scatter')
    with pytest.raises(ValueError):
        dynamic._check_scatter_shape([4, 3], [2], [2, 4], 'Scatter')
    nd = array._ScatterNdOp()
    nd._check_scatter_shape([4, 3], [2, 1], [2, 3], 'ScatterNd')
    with pytest.raises(ValueError):
        nd._check_scatter_shape([4, 3], [2, 1], [2, 4], 'ScatterNd')
    tensor = array._TensorScatterOp("TensorScatterAdd")
    assert tensor.infer_shape([4, 3], [2, 1], [2, 3]) == [4, 3]
    assert tensor.infer_shape([4, 3], [-2], [-2]) == [4, 3]
    assert tensor.infer_dtype(F32, I64, F32) == F32
    for indices, updates in [([2], [2]), ([2, 3], [2]), ([2, 1], [2, 4])]:
        with pytest.raises(ValueError):
            tensor.infer_shape([4, 3], indices, updates)
    assert tensor._check_shape([-2], [1, 2]) is True
    assert tensor._check_shape([-1, 3], [2, 3]) is True
    assert tensor._check_shape([2], [2, 3]) is False
    assert tensor._check_shape([2], [3]) is False


def test_batch_space_stack_helpers():
    """
    Feature: Batch space stack helpers.
    Description: Check batch space stack helpers.
    Expectation: Batch/space transforms, stacking unknown dimensions, and window dtype encodings match
        independently specified metadata.
    """
    op = array.BatchToSpace(2, [[0, 1], [1, 0]])
    assert op.infer_shape([8, 3, 2, 4]) == [2, 3, 3, 7]
    assert op.infer_dtype(F32) == F32
    with pytest.raises(ValueError):
        op.infer_shape([7, 3, 2, 4])
    assert array.SpaceToBatchND(2, [[0, 0], [0, 0]]).block_shape == (2, 2)
    array._check_infer_attr_reduce((0, 1), True, 'Reduce')
    with pytest.raises(TypeError):
        array._check_infer_attr_reduce(('0',), True, 'Reduce')
    assert array._get_stack_shape(None, [[2, 3], [2, 3]], [F32, F32], -1, 'Stack') == [2, 3, 2]
    assert array._get_stack_shape(None, [[-2], [-2]], [F32, F32], 0, 'Stack') == {'shape':[-2]}
    assert array._get_stack_shape(None, [[-1, 3], [2, 3]], [F32, F32], 0, 'Stack') == {'shape':[2, -1, 3]}
    with pytest.raises(TypeError):
        array._get_stack_shape(None, [[2], [2]], [F32, F16], 0, 'Stack')
    with pytest.raises(ValueError):
        array._get_stack_shape(None, [[2], [3]], [F32, F32], 0, 'Stack')
    assert [array.HammingWindow(dtype=t).dtype for t in (dt.float16, dt.float32, dt.float64)] == [1, 0, 11]


@pytest.mark.parametrize('value,opts,expected', [
    (3, {}, (3, 3)), ((2, 3), {'ret_four':True}, (1, 1, 2, 3)),
    ((1, 1, 2, 3), {'allow_four':True}, (2, 3)),
    ((1, 1, 2, 3), {'allow_four':True, 'ret_four':True}, (1, 1, 2, 3)),
    (0, {'strict_positive':False}, (0, 0))])
def test_nn_dimension_normalization(value, opts, expected):
    """
    Feature: Nn dimension normalization.
    Description: Check nn dimension normalization.
    Expectation: Positive integer and tuple parameters normalize to the explicitly specified convolution
        dimensions.
    """
    assert nn._check_positive_int_or_tuple('kernel_size', value, 'Conv2D', **opts) == expected


@pytest.mark.parametrize('value', [(1,), (1, 2, 3), (1, 1, 2, 3), (0, 2), (True, 2)])
def test_nn_dimension_rejection(value):
    """
    Feature: Nn dimension rejection.
    Description: Check nn dimension rejection.
    Expectation: Invalid nn dimension rejection raise ValueError.
    """
    with pytest.raises(ValueError):
        nn._check_positive_int_or_tuple('kernel_size', value, 'Conv2D')


def test_nn_shape_helpers():
    """
    Feature: Nn shape helpers.
    Description: Check nn shape helpers.
    Expectation: Shape validation, NHWC attribute ordering, and transpose-convolution lengths match the explicit
        expected results.
    """
    assert nn._check_shape('shape', (2, 3), 'Op') == (2, 3)
    with pytest.raises(ValueError):
        nn._check_shape('shape', (2, 0), 'Op')
    assert nn._update_attr_by_format((1, 2, 3, 4), 'NHWC') == (2, 3, 4, 1)
    assert nn._deconv_output_length(4, 3, 2, 2) == 11
    assert nn._deconv_output_length(4, 1, 2, 1) == 8


@pytest.mark.parametrize('mode,shape,expected', [('valid', [2, 3, 6, 8], [2, 3, 2, 3]),
    ('same', [2, 3, 6, 8], [2, 3, 3, 4]), ('valid', [2, 3, -1, 8], [2, 3, -1, 3]),
    ('same', [2, 3, 6, -1], [2, 3, 3, -1])])
def test_pool_python_inference(mode, shape, expected):
    """
    Feature: Pool python inference.
    Description: Check pool python inference.
    Expectation: Pooling inference returns the independently specified dimensions and float32 dtype and rejects
        unsupported ranks, shapes, and dtypes.
    """
    op = nn._Pool(3, 2, mode)
    assert op.infer_shape(shape) == expected
    assert op.infer_dtype(F32) == F32
    with pytest.raises(TypeError):
        op.infer_dtype(dt.float32)
    with pytest.raises(ValueError):
        if mode == 'valid':
            op.infer_shape([2, 3, 1, 1])
        else:
            op.infer_shape([2, 3, 0, 0])


@pytest.mark.parametrize('kwargs,expected', [({'stride':1, 'dilation':1}, (1, 1, 1, 1)),
    ({'stride':(2, 3), 'dilation':(1, 2)}, (1, 1, 2, 3)),
    ({'stride':(1, 1, 2, 3), 'dilation':(1, 1, 1, 2)}, (1, 1, 2, 3))])
def test_dilation_constructor(kwargs, expected):
    """
    Feature: Dilation constructor.
    Description: Check dilation constructor.
    Expectation: The dilation constructor records the expected stride and normalized SAME padding.
    """
    op = nn.Dilation2D(**kwargs)
    assert op.stride == expected and op.pad_mode == 'SAME'


@pytest.mark.parametrize('kwargs', [{'stride':1, 'dilation':1, 'data_format':'NHWC'},
    {'stride':(2, 1, 1, 1), 'dilation':1}, {'stride':(1, 2, 3), 'dilation':1},
    {'stride':0, 'dilation':1}, {'stride':256, 'dilation':1}, {'stride':(1, 256), 'dilation':1},
    {'stride':1, 'dilation':(True, 2)}])
def test_dilation_validation(kwargs):
    """
    Feature: Dilation validation.
    Description: Check dilation validation.
    Expectation: Invalid dilation arguments raise ValueError.
    """
    with pytest.raises(ValueError):
        nn.Dilation2D(**kwargs)


def test_nn_loss_and_depthwise_inference():
    """
    Feature: Nn loss and depthwise inference.
    Description: Check nn loss and depthwise inference.
    Expectation: CTC and depthwise-convolution inference return the explicit shapes/dtypes, including
        int8-to-int32 promotion, and reject incompatible metadata.
    """
    loss = nn.RNNTLoss()
    assert loss.infer_shape([2, 5, 4, 8], [2, 3], [2], [2]) == ((2,), [2, 5, 4, 8])
    assert loss.infer_dtype(F32, I32, I32, I32) == (F32, F32)
    with pytest.raises(ValueError):
        loss.infer_shape([2, 5, 4, 8], [2, 4], [2], [2])
    with pytest.raises(TypeError):
        loss.infer_dtype(F32, I64, I32, I32)
    conv = nn.DepthwiseConv2dNative(2, 3, pad_mode='pad', pad=(1, 1, 1, 1))
    assert conv.infer_shape([2, 3, 5, 5], [1, 3, 3, 3]) == [2, 6, 5, 5]
    assert conv.infer_dtype(dt.TensorType(dt.int8), dt.TensorType(dt.int8)) == I32
    with pytest.raises(ValueError, match='weight'):
        conv.infer_shape([2, 3, 5, 5], [2, 3, 3, 3])


@pytest.mark.parametrize('cls,kwargs,attrs', [
    (nn.MaxUnpool2D, {'ksize':2, 'data_format':'NHWC'}, {'ksize':(1, 2, 2, 1), 'strides':(1, 2, 2, 1)}),
    (nn.MaxUnpool3D, {'ksize':2, 'data_format':'NDHWC'}, {'ksize':(1, 2, 2, 2, 1)}),
    (nn.CTCLossV2Grad, {'blank':0}, {'blank':0, 'zero_infinity':False}),
    (nn.ComputeAccidentalHits, {'num_true':2}, {'num_true':2}),
    (nn.Conv3D, {'out_channel':4, 'kernel_size':3, 'pad_mode':'pad', 'pad':(1,)*6}, {'pad':(1,)*6}),
    (nn.Conv3DBackpropInput, {'out_channel':4, 'kernel_size':3, 'pad_mode':'pad', 'pad':(1,)*6}, {'pad':(1,)*6}),
    (nn.Conv3DTranspose, {'in_channel':2, 'out_channel':4, 'kernel_size':3,
     'pad_mode':'pad', 'pad':(1,)*6}, {'pad_list':(1,)*6}),
    (nn.FractionalMaxPool3DWithFixedKsize, {'ksize':2, 'output_shape':3}, {'ksize':(2, 2, 2)}),
    (nn.DeformableOffsets, {'strides':(1, 2, 2, 1), 'pads':(0,)* \
     4, 'ksize':(3, 3), 'data_format':'NHWC'}, {'format':'NHWC'}),
    (nn.NuclearNorm, {}, {'dim':[1000]}), (nn.WKV, {}, {'output_names':['output', 'out_sp', 'out_sq', 'out_sm']}),
    (nn.AllFinite, {}, {'output_names':['is_finite']}),
    (nn.MaxPoolWithArgmaxV2, {'kernel_size':2, 'argmax_type':dt.int32}, {'argmax_type':3}),
    (nn.MaxPool3D, {'pad_mode':'PAD', 'pad_list':(1, 2, 3), 'ceil_mode':True},
     {'pad_list':(1, 1, 2, 2, 3, 3), 'ceil_mode':1}),
])
def test_nn_constructor_protocols(cls, kwargs, attrs):
    """
    Feature: Nn constructor protocols.
    Description: Check nn constructor protocols.
    Expectation: Each neural-network constructor retains its independently specified attribute value.
    """
    op = cls(**kwargs)
    for key, expected in attrs.items():
        assert getattr(op, key) == expected


@pytest.mark.parametrize('cls,kwargs', [
    (nn.FusedBatchNorm, {}), (nn.FusedBatchNormEx, {}),
    (nn.Conv2D, {'out_channel':4, 'kernel_size':3, 'data_format':'NHWC'}),
    (nn.Conv2DBackpropInput, {'out_channel':4, 'kernel_size':3, 'data_format':'NHWC'}),
    (nn.DepthwiseConv2dNative, {'channel_multiplier':2, 'kernel_size':3, 'stride':(1, 2)}),
    (nn.DepthwiseConv2dNative, {'channel_multiplier':2, 'kernel_size':3, 'dilation':(1, 2)}),
    (nn.MaxPool3D, {'ceil_mode':True}), (nn.MaxPool3D, {'pad_list':(1, 2)}),
    (nn.MaxPool3D, {'pad_list':1}), (nn.ChannelShuffle, {'group':1.0}),
    (nn.ComputeAccidentalHits, {'num_true':0}), (nn.Pdist, {'p':-1.0}),
    (nn.SGD, {'nesterov':True, 'dampening':1.0}), (nn.L2Normalize, {'axis':(0, 1)}),
    (nn.DeformableOffsets, {'strides':(2, 1, 1, 1), 'pads':(0,)*4, 'ksize':(3, 3)}),
    (nn.DeformableOffsets, {'strides':(1,)*4, 'pads':(0,)*4, 'ksize':(3, 3), 'dilations':(2, 1, 1, 1)}),
    (nn.DeformableOffsets, {'strides':(1,)*4, 'pads':(0,)*4, 'ksize':(3, 3), 'modulated':False}),
])
def test_nn_constructor_errors(cls, kwargs):
    """
    Feature: Nn constructor errors.
    Description: Check nn constructor errors.
    Expectation: Invalid nn constructor errors raise (TypeError, ValueError).
    """
    with pytest.raises((TypeError, ValueError)):
        cls(**kwargs)


def test_deprecated_alias_attributes_and_identity():
    """
    Feature: Deprecated alias attributes and identity.
    Description: Check deprecated alias attributes and identity.
    Expectation: Deprecated aliases preserve constructor names, attributes, and identity behavior and reject
        unsupported data formats.
    """
    for cls, kwargs, key, value in [(dep.BNTrainingReduce, {}, 'data_format', 'NCHW'),
        (dep.BNTrainingUpdate, {}, 'epsilon', 1e-5), (dep.DynamicShape, {}, 'is_dynamic_shape', True),
        (dep.GatherV2, {}, 'batch_dims', 0), (dep.Pack, {'axis':1}, 'axis', 1),
        (dep.Unpack, {'axis':-1}, 'axis', -1), (dep.InplaceUpdate, {'indices':2}, 'indices', (2,)),
        (dep.BatchToSpaceND, {'block_shape':2, 'crops':[[0, 0], [0, 0]]}, 'block_shape', (2, 2)),
        (dep.FastGelu, {}, 'input_names', ['x'])]:
        assert getattr(cls(**kwargs), key) == value
    assert dep.ScalarToArray().name == 'ScalarToArray'
    assert dep.identity()(object()) is not None
    for cls in (dep.BNTrainingReduce, dep.BNTrainingUpdate, dep.MaxPoolWithArgmax):
        with pytest.raises(ValueError, match='NHWC'):
            cls(data_format='NHWC')


def test_primitive_attribute_lifecycle_and_strategy():
    """
    Feature: Primitive attribute lifecycle and strategy.
    Description: Check primitive attribute lifecycle and strategy.
    Expectation: Primitive mutation methods retain chaining identity, set/remove the intended attributes,
        preserve clone metadata, and reject invalid strategies.
    """
    op = prim.Primitive('CoveragePrimitive')
    assert op.set_stage(2) is op and op.stage == 2
    assert op.add_prim_attr('extra', 7).del_prim_attr('extra') is op
    assert 'extra' not in op.attrs
    assert op.del_prim_attr('absent') is op
    assert op.infer_dynamic_shape is None
    with pytest.raises(AttributeError):
        _ = op.missing
    assert op.shard() is op
    assert op.shard(((2, 1),), ((2, 1),)).in_strategy == ((2, 1),)
    for strategy in ([1], (1,), ((1, 'x'),)):
        with pytest.raises(TypeError):
            op._check_shard_strategy(strategy, 'in_strategy')
    assert op._check_shard_strategy((), 'in_strategy') is None
    with pytest.raises(ValueError):
        op.shard(out_strategy=((1,),))
    with pytest.raises(TypeError):
        op._extract_layout_value([], 'layout')
    with pytest.raises(TypeError):
        op._extract_layout_value((1,), 'layout')
    with pytest.raises(TypeError):
        op._extract_layout_value(((1,),), 'layout')
    assert op._offload(1).backward_prefetch == 1
    assert op._offload().backward_prefetch == 'Auto'
    with pytest.raises(ValueError):
        op._offload('bad')
    with pytest.raises(ValueError):
        op.place('MS_WORKER', -1)
    with pytest.raises(ValueError):
        op.place('MS_SERVER', 0)
    op.place('MS_WORKER', 0)
    assert op.ms_role == 'MS_WORKER' and op.rank_id == 0
    assert op.update_parameter is False
    checker = embed.SubAndFilter()
    cloned = checker._clone()
    assert cloned is not checker and cloned.input_names == ['input_x', 'max_num', 'offset']


def test_primitive_constexpr_and_nested_variables():
    """
    Feature: Primitive constexpr and nested variables.
    Description: Check primitive constexpr and nested variables.
    Expectation: Constant-expression evaluation returns twelve, preserves deferred functions, and identifies
        variables correctly in nested containers.
    """
    def product(x, y):
        return x * y
    constexpr_class = prim.constexpr(get_instance=False)(product)
    op = constexpr_class()
    assert op(3, 4) == 12
    assert op.__infer__(abstract([], None, 3), abstract([], None, 4))['value'] == 12
    expr = prim._primexpr(product, reuse_result=False)
    assert expr(3, 4) == 12
    assert expr.forbid_reuse_result is True
    assert expr.__infer__(abstract([], F32, None))['fn'] == (product,)
    assert expr.__infer__(abstract([], None, 3), abstract([], None, 4))['value'] == 12
    assert prim._primexpr(get_instance=False)(product)()(3, 4) == 12
    assert prim._check_contains_variable([F32, F32], [1, None]) is True
    assert prim._check_contains_variable([None], {'key':None}) is False
    assert prim._check_contains_variable([F32], {'key':None}) is True
    empty = types.SimpleNamespace(created_empty_tensor=True)
    assert prim._check_contains_variable(F32, empty) is True


def test_primitive_cache_identity_and_graph_compile(monkeypatch):
    """
    Feature: Primitive cache identity and graph compile.
    Description: Check primitive cache identity and graph compile.
    Expectation: Repeated construction with identical arguments reuses the primitive, while different arguments
        produce a distinct instance.
    """
    monkeypatch.setattr(cache, '_PRIM_CACHE', {})
    assert cache._temp_func() == 0
    constructor = cache._get_cache_prim(nn.Pdist)
    first = constructor(p=2.0)
    assert constructor(p=2.0) is first
    assert constructor(p=1.0) is not first


def numpy_increment(x):
    """A numerical Python callback, also used for automatic inference."""
    return x + np.float32(1)


def infer_same_shape(shape):
    return tuple(shape)


def infer_same_type(dtype):
    return dtype


def test_custom_pyfunc_inference(custom_state):
    """
    Feature: Custom pyfunc inference.
    Description: Check custom pyfunc inference.
    Expectation: Custom PyFunc callbacks preserve output abstracts, concrete inferred values, callback identity,
        scalar/fake outputs, and backward metadata.
    """
    op = custom.Custom(numpy_increment, infer_same_shape, infer_same_type, func_type='pyfunc')
    assert op.__infer__(abstract([2, 3])) == {'shape': (2, 3), 'dtype': F32, 'value': None}
    assert op.inputs_num == 1 and op.primitive_target == 'CPU'
    # Import at the test boundary so fixture isolation is established first.
    # pylint: disable-next=import-outside-toplevel
    from mindspore.ops.operations._pyfunc_registry import get_pyfunc
    np.testing.assert_array_equal(get_pyfunc(op.fn_id)(np.array([2, 3], np.float32)), [3, 4])
    assert op._auto_infer(abstract([2], value=Tensor([2., 3.], dt.float32)))[:2] == ((2,), F32)
    np.testing.assert_array_equal(op._auto_infer(
        abstract([2], value=Tensor([2., 3.], dt.float32)))[2].asnumpy(), [3, 4])
    assert op._auto_infer(abstract([2])) == ((2,), F32, None)
    assert op._auto_infer(abstract([2], dt.float32)) == ((2,), F32, None)
    fake = custom.Custom(numpy_increment, (), (), 'pyfunc')
    assert fake.fake_output is True
    assert fake.__infer__(abstract([2])) == {'shape': (1,), 'dtype': dt.int32, 'value': None}
    scalar = custom.Custom(numpy_increment, (), dt.float32, 'pyfunc')
    assert scalar.single_scalar_output is True
    for shape, dtype in [(3, dt.float32), ((2,), 'bad')]:
        invalid = custom.Custom(numpy_increment, shape, dtype, 'pyfunc')
        with pytest.raises(TypeError):
            invalid.__infer__(abstract([2]))
    ext = custom._CustomExt('symbol', None, None, bprop='gradient')
    assert ext.__infer__(abstract([2])) == {'shape': (1,), 'dtype': dt.float16, 'value': None}
    assert ext.get_bprop() == 'gradient'
    ext = custom._CustomExt('symbol', infer_same_shape, infer_same_type)
    assert ext.__infer__(abstract([2])) == {'shape': (2,), 'dtype': F32, 'value': None}


@pytest.mark.parametrize('func,kind,error', [(3, 'pyfunc', TypeError), (numpy_increment, 'invalid', ValueError),
    (numpy_increment, 'hybrid', TypeError), (3, 'tbe', TypeError), ('symbol', 'aicpu', ValueError)])
def test_custom_function_validation(custom_state, func, kind, error):
    """
    Feature: Custom function validation.
    Description: Check custom function validation.
    Expectation: Invalid custom function arguments raise the parameterized exception type.
    """
    with pytest.raises(error):
        custom.Custom(func, (2,), dt.float32, kind)


def test_custom_metadata_protocol(custom_state, monkeypatch):
    """
    Feature: Custom metadata protocol.
    Description: Check custom metadata protocol.
    Expectation: Registration forwards the expected metadata and path, preserves names/targets, expands nested
        attributes, and rejects malformed declarations.
    """
    calls = []
    class NativeRegistry:
        def reg_op(self, info, imply_path):
            calls.append((json.loads(info), imply_path))
            return True
    monkeypatch.setattr(custom, 'Oplib', NativeRegistry)
    info = {'op_name':'ProtocolEntry', 'target':'CPU', 'inputs':[{'index':0, 'name':'x', 'param_type':'required'}],
            'outputs':[{'index':0, 'name':'y', 'param_type':'required'}],
            'attr':[{'name':'scale', 'type':'float', 'value':2.0}],
            'dtype_format':[[['float32', 'DefaultFormat'], ['float32', 'DefaultFormat']]]}
    op = custom.Custom('protocol_entry', (2,), dt.float32, 'aicpu', reg_info=copy.deepcopy(info))
    assert calls == [({**info, 'op_name':'ProtocolEntry', 'imply_type':'AiCPU', 'fusion_type':'OPAQUE'}, '')]
    assert op.uniq_name == 'ProtocolEntry' and op.scale == 2.0
    assert op.output_names == ['y'] and op.input_names == ['x']
    assert op._get_expanded_list([None, ('a', [1, 2])]) == [None, 'a', 1, 2]
    op._save_register_status('GPU')
    assert op._get_registered_targets() == ['CPU', 'GPU']
    op._save_attr(None)
    with pytest.raises(ValueError, match='attr names'):
        op._save_attr({'inputs':{'name':'x'}, 'attr':{'name':'new'}})
    op.func_type = 'akg'
    for info_, expected in [({'target':'CPU'}, 'CPU'), ({'processor':'CUDA'}, 'GPU'), ({'imply_type':'TBE'}, 'Ascend')]:
        assert op._get_target(info_) == expected
    with pytest.raises(ValueError, match='target'):
        op._get_target({})
    with pytest.raises(TypeError):
        op._reformat_reg_info('bad', 'CPU')
    reformatted = op._reformat_reg_info({'fusion_type':''}, 'CPU')
    assert reformatted['processor'] == 'CPU' and reformatted['imply_type'] == 'AKG'
    op.func_type = 'tbe'
    result = op._reformat_reg_info({'attr':[{'name':'size'}]}, 'Ascend')
    assert result['attr'] == [{'name':'size', 'value':'all'}]
    assert result['async_flag'] is False and result['partial_flag'] is True and result['kernel'] == 'protocol_entry'


@pytest.mark.parametrize('suffix,cuda_version,expected_extra', [('.cpp', None, []), ('.cc', None, []),
    ('.cu', 'Cuda compilation tools, release 11.8, V11.8.89', ['arch=compute_80,code=sm_80']),
    ('.cu', 'Cuda compilation tools, release 10.0, V10.0.130', [])])
def test_aot_compile_command_protocol(custom_state, monkeypatch, tmp_path, suffix, cuda_version, expected_extra):
    """
    Feature: Aot compile command protocol.
    Description: Check aot compile command protocol.
    Expectation: AOT compilation selects the expected compiler and flags, forwards explicit output/source paths,
        and reuses the generated shared library.
    """
    source = tmp_path / ('kernel' + suffix)
    source.write_text('/* protocol only: compiler subprocess is the external boundary */')
    commands = []
    class CompilerProcess:
        returncode = 0
        def __init__(self, cmd, **kwargs):
            commands.append((cmd, kwargs))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def communicate(self, timeout):
            assert timeout == 30
            return b'compile protocol completed', None
    monkeypatch.setattr(custom.subprocess, 'Popen', CompilerProcess)
    if cuda_version:
        def version_command(cmd, **kwargs):
            assert cmd == ['nvcc', '-V'] and kwargs == {'universal_newlines':True}
            return cuda_version
        monkeypatch.setattr(custom.subprocess, 'check_output', version_command)
    path = custom._compile_aot(str(source))
    assert path == str(tmp_path / 'meta' / (source.name + '.so'))
    cmd, kwargs = commands[0]
    assert cmd[0] == ('nvcc' if suffix == '.cu' else 'g++')
    assert cmd[-3:] == ['-o', path, str(source)]
    assert '-D_GLIBCXX_USE_CXX11_ABI=0' in cmd
    assert any(flag.endswith('/include/api/') for flag in cmd)
    assert '-I' + str(tmp_path) in cmd
    assert kwargs == {'stdout': custom.subprocess.PIPE, 'stderr': custom.subprocess.STDOUT, 'shell': False}
    for flag in expected_extra:
        assert flag in cmd
    assert custom._compile_aot(str(source)) == path and len(commands) == 1


def test_custom_paths_and_builder_protocol(custom_state, monkeypatch, tmp_path):
    """
    Feature: Custom paths and builder protocol.
    Description: Check custom paths and builder protocol.
    Expectation: Cache paths, containment checks, build flags, and function proxies match their explicit
        contracts and invalid environment paths are rejected.
    """
    meta = tmp_path / 'meta'
    assert custom._get_cache_path() == str(meta) + '/'
    assert meta.is_dir()
    assert custom._is_path_under_dir(str(meta / 'x'), str(meta)) is True
    assert custom._is_path_under_dir(str(tmp_path / 'other'), str(meta)) is False
    assert custom._is_path_under_custom_aot_generated_dir(str(meta / 'x')) is True
    assert custom._is_path_under_custom_aot_generated_dir(str(tmp_path / 'other')) is False
    monkeypatch.delenv('MS_COMPILER_CACHE_PATH')
    monkeypatch.chdir(tmp_path)
    assert custom._get_cache_path() == './custom_kernel_meta/'
    assert custom._is_path_under_custom_aot_generated_dir(str(tmp_path / 'akg_kernel_meta' / 'x')) is True
    builder = custom.CustomOpBuilder('protocol', ['kernel.cc'], backend='CPU',
                                     cflags='-DTEST=1', ldflags='-lm', build_dir=str(tmp_path / 'build'))
    # The exact empty-list result is part of this interface contract.
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert builder._get_op_def() == []
    assert '-DTEST=1' in builder.get_cflags() and '-O2' in builder.get_cflags()
    assert '-lm' in builder.get_ldflags() and '-shared' in builder.get_ldflags()
    debug_builder = custom.CustomOpBuilder('protocol_debug', [], backend='CPU', debug_mode=True)
    assert '-g' in debug_builder.get_cflags() and '-s' not in debug_builder.get_ldflags()
    for kwargs in [{'backend': 'CPU', 'enable_atb': True}, {'backend': 'CPU', 'enable_asdsip': True}]:
        with pytest.raises(ValueError, match='Ascend'):
            custom.CustomOpBuilder('bad', [], **kwargs)
    monkeypatch.delenv('ASCEND_OPP_PATH', raising=False)
    with pytest.raises(ValueError, match='ASCEND_OPP_PATH'):
        custom.CustomOpBuilder('bad', [], backend='Ascend')
    monkeypatch.setenv('ASCEND_OPP_PATH', str(tmp_path / 'opp'))
    monkeypatch.delenv('ATB_HOME_PATH', raising=False)
    with pytest.raises(ValueError, match='ATB_HOME_PATH'):
        custom.CustomOpBuilder('bad', [], backend='Ascend', enable_atb=True)
    module = types.SimpleNamespace(op=lambda x: x + 1)
    proxy = custom._MultiSoProxy(None, module)
    assert proxy.op(2) == 3 and proxy.op is module.op
    # Exercise the explicit lifecycle/proxy return contract.
    # pylint: disable-next=unnecessary-dunder-call
    assert proxy.__getattr__('op')(4) == 5
    with pytest.raises(AttributeError, match='neither'):
        _ = proxy.absent
    context.set_context(mode=context.GRAPH_MODE)
    try:
        with pytest.raises(AttributeError, match='GRAPH'):
            _ = proxy.other
        functions = custom._MultiSoProxy(module, None)
        assert functions.op(3) == 4
    finally:
        context.set_context(mode=context.PYNATIVE_MODE)


def test_debug_inference_and_summary_validation(tmp_path):
    """
    Feature: Debug inference and summary validation.
    Description: Check debug inference and summary validation.
    Expectation: Debug primitives preserve expected shape, dtype, and channel metadata and report invalid mode,
        summary, and assertion arguments.
    """
    debug._check_summary_param(abstract([], dt.string, 'metric'), abstract([]), 'ScalarSummary')
    for name, value, error in [(abstract([], dt.string, ''), abstract([]), ValueError),
                                (abstract([], dt.int32, 3), abstract([]), TypeError),
                                (abstract([], dt.string, 'metric'), abstract([], dt.float32), TypeError)]:
        with pytest.raises(error):
            debug._check_summary_param(name, value, 'Summary')
    for cls, channel in [(debug.ScalarSummary, 'ms_scalar_summary'), (debug.TensorSummary, 'ms_tensor_summary'),
                         (debug.ImageSummary, 'ms_image_summary'), (debug.HistogramSummary, 'ms_histogram_summary')]:
        assert cls().channel_name == channel
    assertion = debug.Assert(2)
    assert assertion.infer_shape([], [[2]]) == [1]
    assert assertion.infer_shape([1], [[2]]) == [1]
    with pytest.raises(ValueError):
        assertion.infer_shape([2], [[2]])
    morph = debug.Morph(numpy_increment, infer_same_shape, infer_same_type)
    assert morph.infer_shape([2, 3]) == (2, 3)
    assert morph.infer_dtype(F32) == F32
    with pytest.raises(RuntimeError, match='GRAPH_MODE'):
        morph(Tensor([1.]))
    with pytest.raises(ValueError):
        debug.TensorDump('invalid')
    dump = debug.TensorDump()
    with pytest.raises(ValueError, match='empty'):
        dump('', Tensor([1.]))
    with pytest.raises(TypeError):
        dump(str(tmp_path / 'invalid'), 1)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason=(
                       'Xdivy Python constant folding computes 0/0 as NaN although Xdivy specifies '
                       'zero when x is zero'
                   ))
def test_xdivy_zero_numerator_constant_contract():
    """
    Feature: Xdivy zero numerator constant contract.
    Description: Check xdivy zero numerator constant contract.
    Expectation: A zero numerator folds to zero even when the denominator is also zero.
    """
    with np.errstate(invalid='ignore'):
        value = math.Xdivy().infer_value(Tensor([0.], dt.float32), Tensor([0.], dt.float32))
    np.testing.assert_array_equal(value.asnumpy(), [0.])


def test_custom_aot_placeholders_and_targets(custom_state, tmp_path):
    """
    Feature: Custom aot placeholders and targets.
    Description: Check custom aot placeholders and targets.
    Expectation: AOT placeholders retain their specified abstract dtype/shape and resolve registered or default
        device targets correctly.
    """
    op = custom.Custom(str(tmp_path / 'lib.so') + ':entry', None, None, 'aot')
    assert op.__infer__(abstract([2])) == {'shape': (1,), 'dtype': dt.float16, 'value': None}
    # The exact empty-list result is part of this interface contract.
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert op._get_registered_targets() == []
    op._save_register_status('CPU')
    op._add_prim_target()
    assert op.primitive_target == 'CPU'
    op._save_register_status('GPU')
    assert op._get_registered_targets() == ['CPU', 'GPU']
    op._add_prim_target()
    op.func = 'new_symbol'
    op.uniq_name = 'new_symbol'
    op._save_register_status('GPU')
    op._add_prim_target()
    assert op.primitive_target == 'GPU'
    op.func_type = 'pyfunc'
    op._add_prim_target()
    assert op.primitive_target == 'CPU'


@pytest.mark.parametrize('remove_race', [False, True])
def test_custom_tbe_source_fallback(custom_state, tmp_path, monkeypatch, remove_race):
    """
    Feature: Custom tbe source fallback.
    Description: Import failures trigger isolated source copies; no device compiler is called.
    Expectation: Import failures trigger isolated source copies; no device compiler is called.
    """
    path = tmp_path / 'kernel_source.py'
    path.write_text('def tbe_entry(x):\n    return x + 1\n')
    spec = importlib.util.spec_from_file_location('interface_tbe_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    op = custom.Custom(module.tbe_entry, (2,), dt.float32, 'tbe')
    assert op.imply_path == str(path)
    assert op._update_func_imply_path() == str(path)
    assert str(path) in custom.Custom.tbe_path_checked
    custom.Custom.tbe_path_checked.remove(str(path))
    path.write_text("raise ImportError('optional vendor SDK absent')\n")
    expected_path = tmp_path / 'meta' / 'tbe_entry.py'
    expected_path.parent.mkdir(exist_ok=True)
    expected_path.write_text('old source')
    if remove_race:
        def disappearing_file(file):
            assert file == str(expected_path)
            os.unlink(file)
            raise FileNotFoundError(file)
        monkeypatch.setattr(custom.os, 'remove', disappearing_file)
    copied = op._update_func_imply_path()
    assert copied == str(expected_path)
    assert expected_path.read_text() == op.func_source_str
    assert expected_path.stat().st_mode & 0o777 == 0o600
    assert str(path) in custom.Custom.tbe_path_failed
    assert op._update_func_imply_path() == copied
    assert 'interface_tbe_fixture' not in sys.modules


def test_custom_hybrid_analysis_and_autodiff(custom_state):
    """
    Feature: Custom hybrid analysis and autodiff.
    Description: Check custom hybrid analysis and autodiff.
    Expectation: Hybrid custom operations retain alias metadata, inferred values and shapes, and available
        backward callbacks without invoking device compilation.
    """
    # Import at the test boundary so fixture isolation is established first.
    # pylint: disable-next=import-outside-toplevel
    from mindspore.ops.operations._ms_kernel import kernel
    @kernel
    def passthrough(x):
        return x
    op = custom.Custom(passthrough, None, None, 'hybrid')
    assert op.inplace_assign_output == '0 0'
    inferred = op.__infer__(abstract([2], value=Tensor([2., 3.], dt.float32)))
    assert inferred['shape'] == (2,) and inferred['dtype'] == F32
    np.testing.assert_array_equal(inferred['value'].asnumpy(), [2., 3.])
    assert callable(op.bprop)
    pyfunc = custom.Custom(passthrough, (2,), dt.float32, 'pyfunc')
    assert pyfunc.__infer__(abstract([2])) == {'shape': (2,), 'dtype': dt.float32, 'value': None}
    assert op.__infer__(abstract([2])) == {'shape': (2,), 'dtype': F32, 'value': None}
    # This AKG source contains neither ir_builder nor compute, so it is hybrid.
    akg = custom.Custom(passthrough, (2,), dt.float32, 'akg', bprop=True)
    assert akg.func_type == 'hybrid' and akg.inplace_assign_output == '0 0'
    @kernel
    def zero_inputs():
        result = output_tensor((1,), "float32")
        return result
    no_backward = custom.Custom(zero_inputs, (1,), dt.float32, 'hybrid')
    assert no_backward.bprop is None


def test_custom_builder_generation_and_import_protocol(custom_state, monkeypatch, tmp_path):
    """
    Feature: Custom builder generation and import protocol.
    Description: Check custom builder generation and import protocol.
    Expectation: Generation receives the expected YAML/doc paths and process flags, generated modules import
        correctly, and invalid generation results are rejected.
    """
    calls = []
    def generate_command(cmd, **kwargs):
        calls.append((cmd, kwargs))
        assert cmd[0] == sys.executable and cmd[1].endswith('ops_generate/gen_custom_ops.py')
        assert kwargs == {'check': True, 'text': True, 'capture_output': True}
        assert len(cmd) == 10
        assert cmd[2::2] == ['-i', '-o', '-m', '-d']
        args = dict(zip(cmd[2::2], cmd[3::2]))
        assert args['-m'] == 'yaml_protocol'
        assert Path(args['-i'], 'input.yaml').read_text(encoding='utf-8') == 'interface: test\n'
        assert Path(args['-d'], 'doc.yml').read_text(encoding='utf-8') == 'description: test\n'
        Path(args['-o'], 'gen_custom_ops_def.cc').write_text('// external generator result', encoding='utf-8')
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(custom.subprocess, 'run', generate_command)
    yaml = tmp_path / 'input.yaml'
    yaml.write_text('interface: test\n')
    doc = tmp_path / 'doc.yml'
    doc.write_text('description: test\n')
    builder = custom.CustomOpBuilder('yaml_protocol', [], backend='CPU', op_def=str(yaml),
                                     op_doc=str(doc), build_dir=str(tmp_path / 'build'))
    expected = str(tmp_path / 'build' / 'yaml_protocol_auto_generate' / 'gen_custom_ops_def.cc')
    assert builder._get_op_def() == [expected]
    assert builder._get_op_def() == [expected] and len(calls) == 2
    for path, error in [(tmp_path / 'absent.yaml', FileNotFoundError), (tmp_path / 'wrong.txt', ValueError)]:
        if path.suffix == '.txt':
            path.write_text('wrong')
        builder.yaml = [str(path)]
        with pytest.raises(error):
            builder._get_op_def()
    def failing_generator(cmd, **kwargs):
        raise custom.subprocess.CalledProcessError(2, cmd, output='generation output', stderr='invalid yaml')
    monkeypatch.setattr(custom.subprocess, 'run', failing_generator)
    with pytest.raises(RuntimeError, match='invalid yaml'):
        builder._generate_custom_op_def('yaml_protocol', '', '', '')
    import_path = tmp_path / 'interface_import.py'
    import_path.write_text('def increment(x):\n    return x + 1\n')
    old_modules, old_path = sys.modules.copy(), sys.path[:]
    try:
        loaded = custom.CustomOpBuilder._import_module(str(import_path))
        assert loaded.increment(3) == 4
        packaged = custom.CustomOpBuilder._import_module(str(import_path), True)
        assert packaged.increment(4) == 5
    finally:
        sys.path[:] = old_path
        for key in list(sys.modules):
            if key not in old_modules:
                del sys.modules[key]
        for key, value in old_modules.items():
            sys.modules[key] = value


def test_custom_builder_load_external_compiler_protocol(custom_state, monkeypatch, tmp_path):
    """
    Feature: Custom builder load external compiler protocol.
    Description: Check custom builder load external compiler protocol.
    Expectation: The external compiler receives the explicit source/include/compile/link arguments and its
        imported module is cached by identity.
    """
    monkeypatch.setattr(custom.CustomOpBuilder, '_loaded_ops', {})
    module_path = tmp_path / 'protocol_extension.py'
    module_path.write_text('def twice(x):\n    return 2*x\n')
    calls = []
    class Compiler:
        def __init__(self, build_dir):
            assert build_dir == str(tmp_path / 'build')
        def build(self, **kwargs):
            calls.append(kwargs)
            return str(module_path)
    monkeypatch.setattr(custom, 'ExtensionBuilder', Compiler)
    builder = custom.CustomOpBuilder(
        'protocol_extension', ['kernel.cc'], backend='CPU', build_dir=str(tmp_path / 'build'))
    old = sys.modules.get('protocol_extension')
    try:
        module = builder.load()
        assert module.twice(3) == 6
        assert builder.load() is module and len(calls) == 1
        assert calls[0] == {
    'module_name': 'protocol_extension',
    'sources': ['kernel.cc'],
    'extra_include_paths': builder.get_include_paths( ),
    'extra_cflags': builder.get_cflags(),
     'extra_ldflags': builder.get_ldflags()}
    finally:
        if old is None:
            sys.modules.pop('protocol_extension', None)
        else:
            sys.modules['protocol_extension'] = old


@pytest.mark.parametrize('annotation,expected', [(Tensor,'tensor'), (int,'int'), (float,'float'), (bool,'bool'),
    (list[Tensor],'list_tensor'), (list[int],'list_int'), (list[float],'list_float'), (list[bool],'list_bool'),
    (tuple[Tensor, ...],'tuple_tensor'), (tuple[int, ...],'tuple_int'), (tuple[float, ...],'tuple_float'),
    (tuple[bool, ...],'tuple_bool')])
def test_custom_typed_abi_mapping(annotation, expected):
    """
    Feature: Custom typed abi mapping.
    Description: Check custom typed abi mapping.
    Expectation: Each supported annotation maps to the independently specified typed ABI representation.
    """
    assert custom._map_type_to_spec(annotation) == expected


def test_custom_typed_abi_unsupported():
    """
    Feature: Custom typed abi unsupported.
    Description: Check custom typed abi unsupported.
    Expectation: Invalid custom typed abi unsupported raise TypeError. Diagnostics identify not supported.
    """
    for annotation in (str, dict, list[str], tuple[str, ...], type(None)):
        with pytest.raises(TypeError, match='not supported'):
            custom._map_type_to_spec(annotation)


@jit
def cached_relu_graph(value):
    return cache._get_cache_prim(nn.ReLU)()(value)


def test_primitive_cache_real_graph_execution():
    """
    Feature: Primitive cache real graph execution.
    Description: Check primitive cache real graph execution.
    Expectation: A graph using the cached ReLU primitive computes [0, 0, 3] from [-2, 0, 3].
    """
    np.testing.assert_array_equal(cached_relu_graph(Tensor([-2., 0., 3.], dt.float32)).asnumpy(), [0., 0., 3.])


def test_builder_ascend_flags_without_device(custom_state, monkeypatch, tmp_path):
    """
    Feature: Builder ascend flags without device.
    Description: Backend string flag generation needs no Ascend runtime or device context.
    Expectation: Backend string flag generation needs no Ascend runtime or device context.
    """
    monkeypatch.setenv('ASCEND_OPP_PATH', str(tmp_path / 'cann' / 'opp'))
    monkeypatch.setenv('ATB_HOME_PATH', str(tmp_path / 'atb'))
    builder = custom.CustomOpBuilder('flag_protocol', [], backend='Ascend', enable_atb=True, enable_asdsip=True)
    assert {'-DCUSTOM_ASCEND_OP', '-DCUSTOM_ENABLE_ATB', '-DCUSTOM_ENABLE_ASDSIP'}.issubset(builder.get_cflags())
    assert {'-lascendcl', '-latb', '-lmindspore_extension_ascend_atb',
            '-lmindspore_extension_ascend_asdsip'}.issubset(builder.get_ldflags())
    assert context.get_context('device_target') == 'CPU'


@pytest.mark.parametrize('cls,kwargs', [
    (nn.Conv2D, {'out_channel':4, 'kernel_size':3, 'pad':1}),
    (nn.Conv2DBackpropInput, {'out_channel':4, 'kernel_size':3, 'pad':1}),
    (nn.DepthwiseConv2dNative, {'channel_multiplier':1, 'kernel_size':3, 'pad':1}),
    (nn._Pool, {'data_format':'NHWC'}), (nn.AvgPool3D, {'pad':(1, 2)}),
    (nn.AvgPool3D, {'pad':1}), (nn.Conv3D, {'out_channel':4, 'kernel_size':3, 'pad':(1, 2)}),
    (nn.Conv3D, {'out_channel':4, 'kernel_size':3, 'pad':1}),
    (nn.Conv3D, {'out_channel':4, 'kernel_size':3, 'group':3}),
    (nn.Conv3DBackpropInput, {'out_channel':4, 'kernel_size':3, 'pad':1}),
    (nn.Conv3DTranspose, {'in_channel':2, 'out_channel':4, 'kernel_size':3, 'pad':(1, 2)}),
    (nn.Conv3DTranspose, {'in_channel':2, 'out_channel':4, 'kernel_size':3, 'pad':1}),
    (nn.Conv3DTranspose, {'in_channel':2, 'out_channel':4, 'kernel_size':3, 'group':3}),
    (nn.Conv3DTranspose, {'in_channel':2, 'out_channel':4, 'kernel_size':3, 'output_padding':1}),
    (nn.FractionalMaxPool3DWithFixedKsize, {'ksize':(1, 2), 'output_shape':2}),
])
def test_nn_remaining_pad_and_group_errors(cls, kwargs):
    """
    Feature: Nn remaining pad and group errors.
    Description: Check nn remaining pad and group errors.
    Expectation: Invalid nn remaining pad and group errors raise ValueError.
    """
    with pytest.raises(ValueError):
        cls(**kwargs)


def test_pool_argmax_and_remaining_dynamic_axes(monkeypatch):
    """
    Feature: Pool argmax and remaining dynamic axes.
    Description: Check pool argmax and remaining dynamic axes.
    Expectation: Pooling retains explicit normalized kernels and dynamic dimensions, recurrent directions remain
        correct, and primitive dispatch preserves operand identity.
    """
    max_arg = type('MaxPoolWithArgmax', (nn._Pool,), {})(3, 2, 'valid')
    assert max_arg.kernel_size == (1, 3, 3, 1) and max_arg.strides == (1, 2, 2, 1)
    assert max_arg.infer_shape([2, 3, 6, 8]) == [2, 3, 2, 3]
    assert nn._Pool(3, 2, 'valid').infer_shape([2, 3, 6, -1]) == [2, 3, 2, -1]
    assert nn._Pool(3, 2, 'same').infer_shape([2, 3, -1, 8]) == [2, 3, -1, 4]
    assert rl.CudnnGRU(6, 4, 2, True, False, 0.0).num_directions == 1
    assert nn.LSTM(6, 4, 2, True, True, 0.0).num_directions == 2
    assert nn.MaxPool3DWithArgmax(2, 2, 0, argmax_type=dt.int32).argmax_type == 'int32'
    assert sparse.SparseMatrixOrderingAMD().output_names == ['y']
    # ArgminV2 has no CPU kernel: verify dispatch protocol at the native executor boundary.
    value = Tensor([[3., 1.], [0., 2.]])
    result = Tensor([1, 0], dt.int32)
    calls = []
    def native_dispatch(op, name, args):
        calls.append((op, name, args))
        return result
    with monkeypatch.context() as boundary:
        boundary.setattr(prim._pynative_executor, 'run_op_async', native_dispatch)
        argmin = array.ArgminV2()
        assert argmin(value, 1) is result
    assert len(calls) == 1 and calls[0][0] is argmin
    assert len(calls[0][2]) == 2
    assert calls[0][1] == 'ArgminV2' and calls[0][2][0] is value and calls[0][2][1] == 1
    np.testing.assert_array_equal(array.TupleToArray()(range(3)).asnumpy(), [0, 1, 2])
    # The legacy two-input ReduceMax ABI differs from the generated CPU ABI.
    reducer = type('ReduceMax', (math._Reduce,), {})(keep_dims=False)
    reduction_result = Tensor([3., 4.])
    def native_reduce(op, name, args):
        assert op is reducer and name == 'ReduceMax'
        assert len(args) == 2
        assert args[0] is value and args[1] == 1
        return reduction_result
    with monkeypatch.context() as boundary:
        boundary.setattr(prim._pynative_executor, 'run_op_async', native_reduce)
        # _Reduce supplies __call__ for this dynamically created subclass.
        # pylint: disable-next=not-callable
        assert reducer(value, 1) is reduction_result


def test_primitive_signatures_layouts_and_argument_registration():
    """
    Feature: Primitive signatures layouts and argument registration.
    Description: Check primitive signatures layouts and argument registration.
    Expectation: Primitive signatures and layouts retain the expected structured metadata, Python-only arguments
        register correctly, and invalid representations are rejected.
    """
    # Import at the test boundary so fixture isolation is established first.
    # pylint: disable-next=import-outside-toplevel
    from mindspore.ops import signature as sig
    # Import at the test boundary so fixture isolation is established first.
    # pylint: disable-next=import-outside-toplevel
    from mindspore.parallel.shard import Layout
    op = prim.Primitive('SignatureExample')
    signatures = op._fill_signature([('x', sig.sig_rw.RW_READ, sig.sig_kind.KIND_POSITIONAL_KEYWORD)])
    assert len(signatures) == 1 and isinstance(signatures[0], sig.Signature)
    with pytest.raises(ValueError, match='Signature'):
        op._fill_signature([('x',)])
    layout = Layout((2, 2), ('data', 'model'))('data', 'model')
    assert op._extract_layout_value((layout,), 'layout') == (
        {'device_matrix':(2, 2), 'tensor_map':(1, 0), 'interleaved_parallel':False, 'alias_name':('data', 'model')},)
    with pytest.raises(ValueError, match='in_strategy type'):
        op.shard(((2, 1),), (layout,))
    op.attrs['python_only'] = 17
    assert op.python_only == 17
    class CheckedPrimitive(prim.PrimitiveWithCheck):
        substitute_name = 'CheckedAlias'
        @prim.prim_arg_register
        # prim_arg_register performs Primitive base initialization.
        # pylint: disable-next=super-init-not-called
        def __init__(self, scale=3):
            self.init_prim_io_names(['x'], ['y'])
    checked = CheckedPrimitive(4)
    assert checked.name == 'CheckedAlias' and checked.init_attrs['scale'] == 4
    assert prim._check_contains_variable([F32], {None:3}) is True


def test_debug_security_build_boundary(monkeypatch):
    """
    Feature: Debug security build boundary.
    Description: Binary security-build setting is external to Python constructor validation.
    Expectation: Binary security-build setting is external to Python constructor validation.
    """
    monkeypatch.setattr(debug.security, 'enable_security', lambda: True)
    for cls in (debug.ScalarSummary, debug.ImageSummary, debug.TensorSummary, debug.HistogramSummary,
                debug.TensorDump, debug.Print, debug.Assert):
        with pytest.raises(ValueError, match='not supported'):
            cls()


def test_debug_morph_class_dispatch_and_hook_warning():
    """
    Feature: Debug morph class dispatch and hook warning.
    Description: Check debug morph class dispatch and hook warning.
    Expectation: Class-level Morph dispatch preserves shape/dtype and HookBackward records the requested legacy
        cell identifier.
    """
    morph = debug.Morph(numpy_increment, infer_same_shape, infer_same_type)
    # prim_attr_register stores callback arguments under infer_shape/infer_dtype;
    # direct class dispatch exercises the real forwarding implementation as well.
    assert debug.Morph.infer_shape(morph, [2, 3]) == (2, 3)
    assert debug.Morph.infer_dtype(morph, F32) == F32
    def gradient_hook(gradient):
        return gradient
    assert debug.HookBackward(gradient_hook, cell_id='legacy_cell').cell_id == 'legacy_cell'


def test_custom_environment_and_compiler_failure_boundaries(custom_state, monkeypatch, tmp_path):
    """
    Feature: Custom environment and compiler failure boundaries.
    Description: Check custom environment and compiler failure boundaries.
    Expectation: Rank-specific cache paths are retained and compiler, import, platform, and unsupported-source
        failures propagate their specified diagnostics.
    """
    assert custom._is_path_under_dir('relative.cc', str(tmp_path)) is False
    monkeypatch.setattr(custom.GlobalComm, 'INITED', True)
    def rank_query():
        return 7
    monkeypatch.setattr(custom, 'get_rank', rank_query)
    assert custom._get_cache_path() == str(tmp_path / 'meta' / 'rank_7') + '/'
    monkeypatch.setattr(custom.GlobalComm, 'INITED', False)
    with pytest.raises(ValueError, match='cc/cpp/cu'):
        custom._compile_aot(str(tmp_path / 'invalid.txt'))
    class FailedCompiler:
        returncode = 1
        def __init__(self, cmd, **kwargs):
            assert cmd[-1] == str(tmp_path / 'invalid.cc')
            assert kwargs['shell'] is False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def communicate(self, timeout):
            assert timeout == 30
            return b'syntax error at line 1', None
    monkeypatch.setattr(custom.subprocess, 'Popen', FailedCompiler)
    with pytest.raises(RuntimeError, match='syntax error at line 1'):
        custom._compile_aot(str(tmp_path / 'invalid.cc'))
    with monkeypatch.context() as boundary:
        boundary.setattr(custom.importlib.util, 'find_spec',
                         lambda name: types.SimpleNamespace(origin='/missing/entry.py'))
        with pytest.raises(RuntimeError, match='__init__.py'):
            custom._compile_aot(str(tmp_path / 'another.cc'))
    with monkeypatch.context() as boundary:
        boundary.setattr(custom.platform, 'system', lambda: 'Darwin')
        with pytest.raises(Exception, match='only supported on Linux'):
            custom.Custom(numpy_increment, (2,), dt.float32, 'pyfunc')
    for shape, dtype in [(3, dt.float32), ((2,), 'invalid')]:
        with pytest.raises(TypeError):
            custom._CustomExt('entry', shape, dtype).__infer__(abstract([2]))


def test_custom_callable_registration_and_fake_output(custom_state, monkeypatch):
    """
    Feature: Custom callable registration and fake output.
    Description: Check custom callable registration and fake output.
    Expectation: Callable registration forwards the exact CPU metadata and source path once and retains target,
        input, and attribute declarations.
    """
    calls = []
    class NativeRegistry:
        def reg_op(self, info, path):
            calls.append((json.loads(info), path))
            return True
    monkeypatch.setattr(custom, 'Oplib', NativeRegistry)
    def callback(x):
        return x + 1
    callback.reg_info = {'op_name':'PyProtocol', 'target':'CPU', 'inputs':[{'name':'x', 'index':0}],
                         'outputs':[], 'attr':[{'name':'alpha', 'value':2}],
                         'dtype_format':[(('float32', 'DefaultFormat'),)]}
    op = custom.Custom(callback, (), (), 'pyfunc')
    assert len(calls) == 1
    info, path = calls[0]
    assert info == {'op_name':op.uniq_name, 'target':'CPU', 'inputs':[{'name':'x', 'index':0}],
                    'outputs':[{'index':0, 'name':'y', 'param_type':'required'}], 'attr':[{'name':'alpha', 'value':2}],
                    'dtype_format':[[['float32', 'DefaultFormat'], ['int32', 'DefaultFormat']]],
                    'imply_type':'CPU', 'fusion_type':'OPAQUE'}
    assert path == str(Path(__file__).resolve())
    assert callback.registered_targets == ['CPU']
    assert callback.func_attr == {'input_names':['x'], 'attr_names':['alpha'], 'pure_input_names':['x']}
    op._save_attr({'inputs':[{'name':'x'}], 'attr':[{'name':'alpha', 'value':2}]})
    callback.registered_targets = 'CPU'
    assert op._get_registered_targets() == ['CPU']
    callback.registered_targets = ['CPU']
    op._register_info(json.dumps({'op_name':'PyProtocol', 'target':'CPU', 'outputs':[], 'dtype_format':[]}))
    assert len(calls) == 1
    op._generate_get_worspace_size_func_by_types(['aclTensor*'])


@pytest.mark.xfail(strict=True, raises=AttributeError,
                   reason='Custom._auto_infer scalar fallback reads .dtype although Python scalars have no dtype')
def test_custom_auto_infer_python_scalar(custom_state):
    """
    Feature: Custom auto infer python scalar.
    Description: Check custom auto infer python scalar.
    Expectation: A scalar callback result infers shape (1,), int64 dtype, and concrete value two.
    """
    def scalar_output(x):
        return len(x)
    op = custom.Custom(scalar_output, (), dt.int64, 'pyfunc')
    shape, dtype, value = op._auto_infer(abstract([2], value=Tensor([1., 2.])))
    assert shape == (1,) and dtype == dt.int64 and value.item() == 2


def test_custom_backward_infer_callback_protocol(custom_state):
    """
    Feature: Custom backward infer callback protocol.
    Description: Check custom backward infer callback protocol.
    Expectation: Backward inference preserves the two gradient abstracts and autodiff flag, while unsupported
        large-input autodiff returns no backward callback.
    """
    # Import at the test boundary so fixture isolation is established first.
    # pylint: disable-next=import-outside-toplevel
    from mindspore.ops.operations._ms_kernel import kernel
    @kernel
    def pair(x, y):
        return x, y
    forward = custom.Custom(pair, ((2,), (3,)), (dt.float32, dt.float32), 'hybrid')
    backward = next(cell.cell_contents for cell in forward.bprop.__closure__
                    if isinstance(cell.cell_contents, custom.Custom))
    result = backward.__infer__(abstract([2]), abstract([3]), abstract([2]), abstract([3]),
                                abstract([2]), abstract([3]))
    assert result == {'shape': ([2], [3]), 'dtype': (F32, F32), 'value': None}
    assert backward.autodiff is True
    @kernel
    def eleven_inputs(a, b, c, d, e, f, g, h, i, j, k):
        return a
    assert custom.Custom(eleven_inputs, (2,), dt.float32, 'hybrid').bprop is None


def test_custom_registration_failure_protocol(custom_state, monkeypatch, tmp_path):
    """
    Feature: Custom registration failure protocol.
    Description: Check custom registration failure protocol.
    Expectation: Native rejection forwards the expected payload, raises ValueError, and leaves the failed name
        unregistered; JSON names remain deterministic.
    """
    calls = []
    class RejectingRegistry:
        def reg_op(self, info, path):
            calls.append((json.loads(info), path))
            return False
    monkeypatch.setattr(custom, 'Oplib', RejectingRegistry)
    info = {'op_name':'RejectedEntry', 'target':'CPU', 'outputs':[]}
    with pytest.raises(ValueError, match='registered failed'):
        custom.Custom('rejected', (2,), dt.float32, 'aicpu', reg_info=info)
    assert calls == [({'op_name':'RejectedEntry', 'target':'CPU', 'outputs':[],
                      'imply_type':'AiCPU', 'fusion_type':'OPAQUE'}, '')]
    assert 'RejectedEntry' not in custom.Custom.registered_func
    op = custom.Custom(str(tmp_path / 'lib.so') + ':entry', (2,), dt.float32, 'aot')
    op._update_func_info([None, json.dumps({'op_name':'JsonName'})])
    assert op.uniq_name == 'Custom_JsonName_' + str(tmp_path / 'lib.so') + ':entry'


def test_primitive_place_parallel_mode_rejection():
    """
    Feature: Primitive place parallel mode rejection.
    Description: Check primitive place parallel mode rejection.
    Expectation: Invalid primitive place parallel mode rejection raise RuntimeError. Diagnostics identify mixed
        with other parallel.
    """
    old = context.get_auto_parallel_context('parallel_mode')
    try:
        context.set_auto_parallel_context(parallel_mode='data_parallel')
        with pytest.raises(RuntimeError, match='mixed with other parallel'):
            prim.Primitive('PlacementCheck').place('MS_WORKER', 0)
    finally:
        context.set_auto_parallel_context(parallel_mode=old)


def test_crop_resize_grad_rejects_non_numeric_type():
    """
    Feature: Crop resize grad rejects non numeric type.
    Description: Check crop resize grad rejects non numeric type.
    Expectation: Invalid crop resize grad non numeric type raise TypeError. Diagnostics identify T.
    """
    with pytest.raises(TypeError, match='T'):
        image.CropAndResizeGradImage(dt.string)


def test_custom_elimination_extension_protocol(custom_state):
    """
    Feature: Custom elimination extension protocol.
    Description: Check custom elimination extension protocol.
    Expectation: An elimination extension returns the original tensor by identity.
    """
    class IdentityCustom(custom.Custom):
        def check_elim(self, value):
            return True, value
    op = IdentityCustom(numpy_increment, (2,), dt.float32, 'pyfunc')
    value = Tensor([2., 3.], dt.float32)
    assert op(value) is value


def test_custom_multiple_return_analysis_warning(custom_state):
    """
    Feature: Custom multiple return analysis warning.
    Description: Check custom multiple return analysis warning.
    Expectation: Multiple-return hybrid analysis retains hybrid metadata, the expected abstract output, and the
        callback's literal numeric behavior.
    """
    def choose(x):
        if x[0] > 0:
            return x
        return -x
    op = custom.Custom(choose, (2,), dt.float32, 'akg', bprop=True)
    assert op.func_type == 'hybrid'
    assert op.__infer__(abstract([2])) == {'shape': (2,), 'dtype': dt.float32, 'value': None}
    np.testing.assert_array_equal(op.func(np.array([-1., 2.], np.float32)), [1., -2.])


@pytest.mark.xfail(strict=True, raises=pytest.fail.Exception,
                   reason=(
                       'Primitive._check_shard_strategy guards uniformity with if not is_layout, '
                       'so mixed tuple/Layout inputs escape validation'
                   ))
def test_primitive_rejects_mixed_strategy_representation():
    """
    Feature: Primitive rejects mixed strategy representation.
    Description: Check primitive rejects mixed strategy representation.
    Expectation: Invalid primitive mixed strategy representation raise TypeError. Diagnostics identify all tuple
        type or all Layout type.
    """
    # Import at the test boundary so fixture isolation is established first.
    # pylint: disable-next=import-outside-toplevel
    from mindspore.parallel.shard import Layout
    layout = Layout((2,), ('data',))('data')
    with pytest.raises(TypeError, match='all tuple type or all Layout type'):
        prim.Primitive('UniformStrategy').shard(((2,), layout))
