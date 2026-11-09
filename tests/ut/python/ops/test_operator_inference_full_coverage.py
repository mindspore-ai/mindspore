# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""CPU tests of Python operator inference, validation, and constant folding."""
# Unit coverage intentionally exercises internal Python inference and wrapper helpers.
# pylint: disable=protected-access
import ast
import errno
import json
import math
import os
import subprocess
import sys
import types as pytypes
import threading
from collections import OrderedDict

import numpy as np
import pytest

from mindspore import Tensor, Parameter, context
from mindspore.common import dtype as dt
from mindspore.ops.operations import _grad_ops as grad
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.operations import _quant_ops as quant
from mindspore.ops.operations import _thor_ops as thor
from mindspore.ops.operations import _ms_kernel as hybrid
from mindspore.ops.operations import _custom_ops_utils as custom
from mindspore.ops.operations import other_ops as other
from mindspore.ops.operations.manually_defined import ops_def as manual

F32 = dt.TensorType(dt.float32)
F16 = dt.TensorType(dt.float16)
I32 = dt.TensorType(dt.int32)
I64 = dt.TensorType(dt.int64)
X = [2, 3, 4, 4]
C = [3]


@pytest.fixture(autouse=True, scope="module", name='cpu_context')
def fixture_cpu_context():
    old = {key: context.get_context(key) for key in ('mode', 'device_target')}
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    yield
    context.set_context(**old)


@pytest.fixture(autouse=True, name='ascend_registration_boundary')
def fixture_ascend_registration_boundary(monkeypatch):
    """Isolate only optional Ascend kernel registration imports, never inference.

    The CPU distribution lacks TE/TIK. These imports register device kernels;
    constructors do not call their symbols. All Python validation stays real.
    """
    names = {
        'batch_matmul_impl': 'cus_batch_matmul',
        'cholesky_trsm_impl': 'cus_cholesky_trsm',
        'fused_abs_max1_impl': 'cus_fused_abs_max1',
        'img2col_impl': 'cus_img2col',
        'matmul_cube_dense_left_impl': 'cus_matmul_cube_dense_left',
        'matmul_cube_fracz_right_mul_impl': 'cus_matmul_cube_fraczrightmul',
        'matmul_cube_impl': 'cus_matmul_cube',
        'matrix_combine_impl': 'cus_matrix_combine',
        'transpose02314_impl': 'cus_transpose02314',
        'matmul_cube_dense_right_impl': 'cus_matmul_cube_dense_right',
        'matmul_cube_fracz_left_cast_impl': 'cus_matmul_cube_fraczleftcast',
    }
    package_name = 'mindspore.ops._op_impl._custom_op'
    package = pytypes.ModuleType(package_name)
    package.__path__ = []
    registrations = [
        'fake_learned_scale_quant_perlayer_grad',
        'fake_learned_scale_quant_perlayer_grad_reduce',
        'fake_learned_scale_quant_perchannel_grad',
        'fake_learned_scale_quant_perchannel_grad_reduce',
        'batchnorm_fold', 'batchnorm_fold_grad', 'batchnorm_fold2',
        'batchnorm_fold2_grad', 'batchnorm_fold2_grad_reduce',
        'minmax_update_perlayer', 'minmax_update_perchannel',
        'fake_learned_scale_quant_perlayer', 'fake_learned_scale_quant_perchannel',
        'fake_quant_perlayer', 'fake_quant_perlayer_grad',
        'fake_quant_perchannel', 'fake_quant_perchannel_grad',
        'correction_mul', 'correction_mul_grad',
    ]
    def device_kernel_must_not_run(*args, **kwargs):
        raise AssertionError('Inference must not execute an Ascend device kernel')
    for name in registrations:
        setattr(package, name, pytypes.ModuleType(package_name + '.' + name))
    monkeypatch.setitem(sys.modules, package_name, package)
    for module_name, symbol in names.items():
        module = pytypes.ModuleType(package_name + '.' + module_name)
        setattr(module, symbol, device_kernel_must_not_run)
        monkeypatch.setitem(sys.modules, module.__name__, module)


def abstract(shape, dtype=F32, value=None):
    """Construct an input abstract, without replacing primitive inference."""
    return {'shape': shape, 'dtype': dtype, 'value': value}


# Every table entry records an independently specified public output contract.
QUANT_CASES = [
    ('MinMaxUpdatePerLayer', {}, [X, [1], [1]], ([1], [1]), [F32]*3, (F32, F32)),
    ('MinMaxUpdatePerChannel', {}, [X, C, C], (C, C), [F32]*3, (F32, F32)),
    ('FakeLearnedScaleQuantPerLayer', {}, [X, [1], [1]], X, [F32]*3, F32),
    ('FakeLearnedScaleQuantPerLayerGrad', {}, [X, X, [1], [1]], (X, [1]), [F32]*4, (F32, F32)),
    ('FakeLearnedScaleQuantPerLayerGradD', {}, [X, X, [1], [1]], (X, X), [F32]*4, (F32, F32)),
    ('FakeLearnedScaleQuantPerLayerGradDReduce', {}, [X], (1,), [F32], F32),
    ('FakeLearnedScaleQuantPerChannel', {}, [X, C, [1]], X, [F32]*3, F32),
    ('FakeLearnedScaleQuantPerChannelGrad', {}, [X, X, C, [1]], (X, C), [F32]*4, (F32, F32)),
    ('FakeLearnedScaleQuantPerChannelGradD', {}, [X, X, C, [1]], (X, X), [F32]*4, (F32, F32)),
    ('FakeLearnedScaleQuantPerChannelGradDReduce', {}, [X], (3,), [F32], F32),
    ('FakeQuantWithMinMaxVars', {}, [X, [1], [1]], X, [F32]*3, F32),
    ('FakeQuantWithMinMaxVarsGradient', {}, [X, X, [1], [1]], (X, [1], [1]), [F32]*4, (F32, F32, F32)),
    ('FakeQuantWithMinMaxVarsPerChannel', {}, [X, [4], [4]], X, [F32]*3, F32),
    ('FakeQuantWithMinMaxVarsPerChannelGradient', {}, [X, X, [4], [4]], (X, [4], [4]), [F32]*4, (F32, F32, F32)),
    ('FakeQuantPerLayer', {}, [X, [1], [1]], X, [F32]*3, F32),
    ('FakeQuantPerLayerGrad', {}, [X, X, [1], [1]], X, [F32]*4, F32),
    ('FakeQuantPerChannel', {}, [X, C, C], X, [F32]*3, F32),
    ('FakeQuantPerChannelGrad', {}, [X, X, C, C], X, [F32]*4, F32),
    ('BatchNormFold', {}, [X, C, C, [1]], (C, C, C, C), [F32]*3+[I32], (F32,)*4),
    ('BatchNormFoldGrad', {}, [C, C, X, C, C, [1]], X, [F32]*5+[I32], F32),
    ('CorrectionMul', {'channel_axis': 1}, [X, C, C], X, [F32]*3, F32),
    ('CorrectionMulGrad', {'channel_axis': 1}, [X, X, C, C], (X, C), [F32]*4, (F32, F32)),
    ('CorrectionMulGradReduce', {'channel_axis': 1}, [X], C, [F32], F32),
    ('BatchNormFold2', {}, [X, C, C, C, C, C, C, [1]], X, [F32]*7+[I32], F32),
    ('BatchNormFold2Grad', {}, [X, X, C, C, C, C, C, [1]], (C, C, C, C, X), [F32]*7+[I32], (F32,)*5),
    ('BatchNormFoldD', {}, [X, C, C, C, C], (X, C, C, C, C, C, C), [F32]*5, (F32,)*7),
    ('BatchNormFoldGradD', {}, [C, C, X, C, C], X, [F32]*5, F32),
    ('BatchNormFold2D', {}, [X, C, C, C, C, C], X, [F32]*6, F32),
    ('BatchNormFold2GradD', {}, [X, C, C, C, C, C, C], (C, C, C, X), [F32]*7, (F32,)*4),
    ('BatchNormFold2GradReduce', {}, [X, X], ((3,), (3,)), [F32]*2, (F32, F32)),
    ('ActsULQ', {}, [X, [1]*4, [1]*4], (X,)*4, [F32]*3, (F32, dt.bool_, dt.bool_, F32)),
    ('ActsULQInputGrad', {}, [X, X, X], X, [F32]*3, F32),
    ('ActULQClampMinGrad', {}, [X, X, X], (1,)*4, [F32]*3, dt.float32),
    ('ActULQClampMaxGrad', {}, [X, X, X], (1,)*4, [F32]*3, dt.float32),
    ('WtsARQ', {'num_bits': 8, 'offset_flag': True}, [X, [1]*4, [1]*4], X, [F32]*3, F32),
]


@pytest.mark.parametrize('name,kwargs,shapes,out_shape,types,out_type', QUANT_CASES, ids=[c[0] for c in QUANT_CASES])
def test_quantization_inference(name, kwargs, shapes, out_shape, types, out_type):
    """
    Feature: Quantization inference.
    Description: Check quantization inference.
    Expectation: Every quantization primitive infers the independently specified output shape and dtype for its
        abstract inputs.
    """
    op = getattr(quant, name)(**kwargs)
    assert op.infer_shape(*shapes) == out_shape
    assert op.infer_dtype(*types) == out_type


@pytest.mark.parametrize(
    'name,kwargs,shapes,out_shape,types,out_type', [
        c for c in QUANT_CASES if c[0] not in (
            'CorrectionMulGradReduce', 'ActULQClampMinGrad', 'ActULQClampMaxGrad')], ids=[
                c[0] for c in QUANT_CASES if c[0] not in (
                    'CorrectionMulGradReduce', 'ActULQClampMinGrad', 'ActULQClampMaxGrad')])
def test_quantization_rejects_integer_data(name, kwargs, shapes, out_shape, types, out_type):
    """
    Feature: Quantization rejects integer data.
    Description: Check quantization rejects integer data.
    Expectation: Invalid quantization integer data raise (TypeError, ValueError). Diagnostics identify
        type|Type|dtype.
    """
    del shapes, out_shape, out_type
    op = getattr(quant, name)(**kwargs)
    # Change all floating data, preserving special integer global_step inputs.
    invalid = [I32 if typ == F32 else typ for typ in types]
    if name == 'BatchNormFold2GradReduce':
        # This reduction checks equality rather than allowed floating types.
        invalid[-1] = F32
    with pytest.raises((TypeError, ValueError), match='type|Type|dtype'):
        op.infer_dtype(*invalid)


@pytest.mark.parametrize('name',
    ['MinMaxUpdatePerLayer',
    'MinMaxUpdatePerChannel',
    'FakeQuantPerLayer',
     'FakeQuantPerChannel'])
@pytest.mark.parametrize('kwargs,error,reason', [
    ({'ema': True, 'ema_decay': 0.0}, ValueError, 'together'),
    ({'ema': 'yes'}, TypeError, 'ema'),
    ({'ema_decay': 1.2}, ValueError, 'ema_decay'),
])
def test_quantization_ema_validation(name, kwargs, error, reason):
    """
    Feature: Quantization ema validation.
    Description: Check quantization ema validation.
    Expectation: Invalid quantization ema arguments raise the parameterized exception type.
    """
    with pytest.raises(error, match=reason):
        getattr(quant, name)(**kwargs)


@pytest.mark.parametrize('name', ['FakeQuantPerLayer', 'FakeQuantPerLayerGrad',
                         'FakeQuantPerChannel', 'FakeQuantPerChannelGrad'])
def test_quantization_bits_and_symmetric_validation(name):
    """
    Feature: Quantization bits and symmetric validation.
    Description: Check quantization bits and symmetric validation.
    Expectation: Invalid quantization bits and symmetric arguments raise ValueError, TypeError. Diagnostics
        identify num_bits, symmetric.
    """
    with pytest.raises(ValueError, match='num_bits'):
        getattr(quant, name)(num_bits=6)
    with pytest.raises(TypeError, match='symmetric'):
        getattr(quant, name)(symmetric=1)


@pytest.mark.parametrize('name', ['FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxVarsGradient'])
def test_quantization_broadcast(name):
    """
    Feature: Quantization broadcast.
    Description: Check quantization broadcast.
    Expectation: Scalar-sized and flattened compatible parameters are accepted, while incompatible broadcasts
        raise ValueError.
    """
    op = getattr(quant, name)()
    assert op.check_broadcast([1], [2, 3]) is None
    assert op.check_broadcast([6], [2, 3]) is None
    with pytest.raises(ValueError, match='broadcast'):
        op.check_broadcast([2], [2, 3])


@pytest.mark.parametrize('name', ['FakeLearnedScaleQuantPerChannel', 'FakeQuantPerChannel'])
def test_quantization_rank_one_channel_axis(name):
    """
    Feature: Quantization rank one channel axis.
    Description: Check quantization rank one channel axis.
    Expectation: Rank-one per-channel inputs preserve shape [5] and normalize the channel axis to zero;
        incompatible ranks are rejected.
    """
    op = getattr(quant, name)()
    assert op.infer_shape([5], [5], [5]) == [5]
    assert op.channel_axis == 0
    with pytest.raises(ValueError, match='rank|shape'):
        op.infer_shape([], [5], [5])


@pytest.mark.parametrize('scale,zp,expected_scale,expected_zp',
                         [(0.5, 2, [0.5], [2]), ([0.1, 0.2], [0, 1], [0.1, 0.2], [0, 1]), ((0.3,), (0,), (0.3,), (0,))])
def test_linear_quant_param(scale, zp, expected_scale, expected_zp):
    """
    Feature: Linear quant param.
    Description: Check linear quant param.
    Expectation: Scale, zero-point, algorithm, per-channel flag, and label attributes retain their explicitly
        specified values.
    """
    op = quant.FakeQuantParam.linear_quant_param(dt.QuantDtype.INT8, scale, zp, is_per_channel=True, label='test')
    assert op.attrs['linear_quant_scale'] == expected_scale
    assert op.attrs['linear_quant_zero_point'] == expected_zp
    assert op.attrs['quant_algo_name'] == 'linear_quant_algo'
    assert op.attrs['is_per_channel'] is True
    assert op.attrs['label'] == 'test'


@pytest.mark.parametrize('scale,zp,flag,reason', [('bad', 0, False, 'scale'),
                         (0.1, 0.5, False, 'zero_point'), (0.1, 0, 1, 'is_per_channel')])
def test_linear_quant_param_rejects_bad_types(scale, zp, flag, reason):
    """
    Feature: Linear quant param rejects bad types.
    Description: Check linear quant param rejects bad types.
    Expectation: Invalid linear quant param bad types raise TypeError.
    """
    with pytest.raises(TypeError, match=reason):
        quant.FakeQuantParam.linear_quant_param(dt.QuantDtype.INT8, scale, zp, flag)


THOR_CASES = [
    ('CusBatchMatMul', {}, [[2, 128, 128], [2, 128, 128]], [2, 128, 128], F32),
    ('CusCholeskyTrsm', {}, [[256, 256]], [2, 128, 128], F32),
    ('CusCholeskyTrsm', {}, [[64, 64]], [1, 64, 64], F32),
    ('CusFusedAbsMax1', {}, [[32, 64]], [1], F32),
    ('CusFusedAbsMax1', {}, [[2, 2, 16, 16]], [32, 64], F32),
    ('CusImg2Col', {'ksizes': (1, 3, 3, 1), 'strides': (1, 2, 2, 1)}, [[2, 3, 8, 8]], [32, 144], F32),
    ('CusImg2Col', {'ksizes': (1, 3, 3, 1), 'strides': (1, 2, 2, 1)}, [[2, 32, 8, 8]], [32, 288], F32),
    ('CusMatMulCubeDenseLeft', {}, [[2, 2, 16, 16], [32, 48]], [32, 48], dt.float16),
    ('CusMatMulCubeFraczRightMul', {}, [[32, 48], [48, 48], [1]], [32, 48], dt.float32),
    ('CusMatMulCube', {}, [[32, 48], [48, 64]], [32, 64], dt.float32),
    ('CusMatMulCube', {'transpose_a': True, 'transpose_b': True}, [[48, 32], [64, 48]], [32, 64], dt.float32),
    ('CusMatrixCombine', {}, [[3, 8, 12]], [24, 36], F32),
    ('CusTranspose02314', {}, [[2, 32, 4, 4]], (32, 32), F32),
    ('CusMatMulCubeDenseRight', {}, [[32, 48], [48, 64], [1]], [32, 48], dt.float32),
    ('CusMatMulCubeFraczLeftCast', {}, [[2, 2, 16, 16], [32, 48]], [32, 48], dt.float16),
    ('LoadIm2Col', {'ksizes': (3, 3), 'strides': (2, 2)}, [[2, 16, 8, 8]], [2, 1, 9, 16, 16], F32),
    ('LoadIm2Col', {'ksizes': (3, 3), 'strides': (2, 2)}, [[2, 16, 6, 6]], [1, 9, 16, 16], F32),
    ('UpdateThorGradient', {}, [[4, 4], [4, 8], [8, 8]], [4, 8], F32),
    ('_Cholesky', {}, [[10, 10]], [10, 10], F32),
    ('_Cholesky', {'split_dim': 4}, [[3, 3]], [1, 3, 3], F32),
    ('_Cholesky', {'split_dim': 4}, [[8, 8]], [2, 4, 4], F32),
    ('_Cholesky', {'split_dim': 4}, [[10, 10]], [3, 4, 4], F32),
    ('DetTriangle', {}, [[2, 3, 3]], [2], F32),
    ('ProdForceSeA', {'natoms': 5}, [[2, 5, 3], [2, 5, 3], [2, 5, 4]], [2, 5, 3], F32),
]


@pytest.mark.parametrize('name,kwargs,shapes,expected,dtype', THOR_CASES,
                         ids=[f'{x[0]}-{i}' for i,x in enumerate(THOR_CASES)])
def test_thor_inference(name, kwargs, shapes, expected, dtype):
    """
    Feature: Thor inference.
    Description: Check thor inference.
    Expectation: THOR primitives produce the parameterized output shapes and dtypes and record the expected
        output name.
    """
    op = getattr(thor, name)(**kwargs)
    assert op.infer_shape(*[list(s) for s in shapes]) == expected
    assert op.infer_dtype(*([F32]*len(shapes))) == dtype
    assert op.attrs['output_names'] == ['y']


@pytest.mark.parametrize('cls,kwargs,expected,pads', [
    (thor.ThorIm2Col, {'kernel_size': 3, 'stride': 2, 'pad_mode': 'valid'}, [3, 3, 3, 2, 3, 3], (0, 0, 0, 0)),
    (thor.ThorIm2Col, {'kernel_size': 3, 'stride': 2, 'pad_mode': 'same'}, [3, 3, 3, 2, 4, 4], (0, 1, 0, 1)),
    (thor.ThorIm2Col, {'kernel_size': 3, 'stride': 2, 'pad_mode': 'pad', 'pad': 1}, [3, 3, 3, 2, 4, 4], (1, 1, 1, 1)),
    (thor.NewIm2Col, {'ksizes': (3, 3), 'strides': 2, 'padding_mode': 'VALID'}, [2, 3, 3, 27], (0, 0, 0, 0)),
    (thor.NewIm2Col, {'ksizes': (3, 3), 'strides': 2, 'padding_mode': 'SAME'}, [2, 4, 4, 27], (0, 1, 0, 1)),
])
def test_thor_im2col_padding(cls, kwargs, expected, pads):
    """
    Feature: Thor im2col padding.
    Description: Check thor im2col padding.
    Expectation: Im2col inference derives the explicit output shape and padding, accepts float16, and rejects
        unsupported dtypes and ranks.
    """
    op = cls(**kwargs)
    assert op.infer_shape([2, 3, 8, 8]) == expected
    assert op.attrs['pad_list'] == pads
    assert op.infer_dtype(F16) == F16
    with pytest.raises(TypeError, match='type|Type'):
        op.infer_dtype(I32)
    with pytest.raises(ValueError, match='rank'):
        op.infer_shape([3, 8, 8])


INNER_CASES = [
    ('Quant', {'scale': 1.0, 'offset': 0.0}, [X], X, [F32], dt.int8),
    ('Lamb', {}, [X, X, X, [], [], [], [], [], [], X], X, [F32]*8+[I32, F32], F32),
    ('MatrixDiagPart', {}, [[2, 3], [2, 3]], [2], [F32, F32], F32),
    ('MatrixDiagPart', {}, [[3, 2], [3, 2]], [2], [F32, F32], F32),
    ('MatrixSetDiag', {}, [[2, 3], [2], [2, 3]], [2, 3], [F32]*3, F32),
    ('MatrixSetDiag', {}, [[3, 2], [2], [3, 2]], [3, 2], [F32]*3, F32),
    ('ConfusionMulGrad', {'axis': 1, 'keep_dims': True}, [
     [2, 3], [2, 3], [2, 3]], ([2, 3], [2, 1]), [F32]*3, (F32, F32)),
    ('DecodeImage', {'dtype': dt.float32}, [[40]], (-1, -1, 3), [dt.string], dt.float32),
    ('PsROIPooling', {'pooled_height': 2, 'pooled_width': 3, 'num_rois': 4, 'spatial_scale': 1.0,
     'out_dim': 5, 'group_size': 1}, [X, [4, 5]], ([4, 5, 2, 3], [4, 5, 2, 3]), [F32]*2, (F32, I32)),
    ('ClipByNorm', {'axis': (0, -1)}, [X, [1]], X, [F16, F32], dt.float32),
    ('_VirtualConverterEnd', {'input_nums': 2}, [[2, 3], [2, 3]], (4, 3), [F32]*2, F32),
    ('_VirtualConverterBegin', {'output_nums': 2}, [[4, 3]], ((2.0, 3), (2.0, 3)), [F32], (F32, F32)),
]


@pytest.mark.parametrize('name,kwargs,shapes,out_shape,types,out_type', INNER_CASES,
                         ids=[c[0]+str(i) for i,c in enumerate(INNER_CASES)])
def test_inner_inference(name, kwargs, shapes, out_shape, types, out_type):
    """
    Feature: Inner inference.
    Description: Check inner inference.
    Expectation: Each internal primitive returns the independently specified abstract output shape and dtype.
    """
    op = getattr(inner, name)(**kwargs)
    assert op.infer_shape(*shapes) == out_shape
    assert op.infer_dtype(*types) == out_type


@pytest.mark.parametrize('cls', [inner.GpuConvertToDynamicShape, inner.ConvertToDynamic])
def test_dynamic_shape_checkers(cls):
    """
    Feature: Dynamic shape checkers.
    Description: Check dynamic shape checkers.
    Expectation: Unknown dimensions are accepted with float32 tensors, while invalid ranks and dtypes are
        rejected.
    """
    op = cls()
    assert op.check_shape([2, -1]) is None
    assert op.check_dtype(F32) is None
    with pytest.raises(ValueError, match='rank'):
        op.check_shape([])
    with pytest.raises(TypeError, match='input_dtype'):
        op.check_dtype(dt.float32)


def test_error_on_dynamic_shape_and_sequence_mask():
    """
    Feature: Error on dynamic shape and sequence mask.
    Description: Check error on dynamic shape and sequence mask.
    Expectation: Static shapes and values retain identity; dynamic shapes and invalid sequence-mask dtypes or
        shapes are rejected.
    """
    op = inner.ErrorOnDynamicShapeInput()
    assert op.infer_shape([2, 3]) == [2, 3]
    assert op.infer_type(F32) == F32
    value = Tensor([1, 2], dt.int32)
    assert op.infer_value(value) is value
    with pytest.raises(ValueError, match='dynamically'):
        op.infer_shape([2, -1])
    op = inner.SequenceMask()
    assert op.check_shape([2], []) is None
    assert op.check_dtype(I32, dt.int32) is None
    for shapes in [([], []), ([2], [1])]:
        with pytest.raises(ValueError, match='shape'):
            op.check_shape(*shapes)
    with pytest.raises(TypeError, match='lengths_dtype'):
        op.check_dtype(dt.int32, dt.int32)


@pytest.mark.parametrize('axis', [1, (0, 1), [0], ()])
def test_centralization_const_axis(axis):
    """
    Feature: Centralization const axis.
    Description: Check centralization const axis.
    Expectation: Constant-axis inference preserves the input shape [2, 3] and expected float32 abstract
        metadata.
    """
    x = abstract([2, 3])
    assert inner.Centralization().__infer__(x, abstract([], dt.int64, axis)) == abstract([2, 3])


@pytest.mark.parametrize('axis,error,reason', [(None, ValueError, 'const'), (2,
                         ValueError, 'axis'), ((0, 'bad'), TypeError, 'axis'), ('bad', TypeError, 'axis')])
def test_centralization_bad_axis(axis, error, reason):
    """
    Feature: Centralization bad axis.
    Description: Check centralization bad axis.
    Expectation: Invalid centralization bad axis raise the parameterized exception type.
    """
    with pytest.raises(error, match=reason):
        inner.Centralization().__infer__(abstract([2, 3]), abstract([], dt.int64, axis))


def test_stack_and_dynamic_stitch():
    """
    Feature: Stack and dynamic stitch.
    Description: Check stack and dynamic stitch.
    Expectation: Stack metadata, dynamic stitch shapes/dtypes, and constructor indices match the expected
        contracts; malformed inputs are rejected.
    """
    for cls in (inner.StackInit, inner.StackPush, inner.StackDestroy):
        assert cls(index=3).attrs['index'] == 3
        with pytest.raises(TypeError, match='index'):
            cls(index='bad')
    op = inner.StackPop(index=2, shape=(2, 3), dtype=dt.int32)
    assert op.__infer__() == abstract([2, 3], dt.int32)
    with pytest.raises(ValueError, match='shape element'):
        inner.StackPop(shape=(0,))
    op = inner.DynamicStitch()
    assert op.check_shape([[2], [3]], [[2, 4], [3, 4]]) == [-1, 4]
    assert op.check_dtype([I32, I32], [F32, F32]) == F32
    with pytest.raises(ValueError, match='shape'):
        op.check_shape([[2], [3]], [[2, 4], [3, 5]])
    with pytest.raises(TypeError, match='indices'):
        op.check_dtype([I64], [F32])


@pytest.mark.parametrize('attribute,expected', [('start', 1), ('stop', 7), ('step', 2)])
@pytest.mark.parametrize('tensor_bound', [False, True])
def test_slice_getitem(attribute, expected, tensor_bound):
    """
    Feature: Slice getitem.
    Description: Check slice getitem.
    Expectation: Reading each supported slice attribute returns its independently specified start, stop, or step
        value.
    """
    values = [Tensor([i], dt.int32) for i in (1, 7, 2)] if tensor_bound else [1, 7, 2]
    assert inner.SliceGetItem()(slice(*values), attribute) == expected


def test_slice_getitem_errors():
    """
    Feature: Slice getitem errors.
    Description: Check slice getitem errors.
    Expectation: Invalid slice getitem errors raise TypeError, AttributeError. Diagnostics identify slice type,
        missing.
    """
    with pytest.raises(TypeError, match='slice type'):
        inner.SliceGetItem()(1, 'start')
    with pytest.raises(AttributeError, match='missing'):
        inner.SliceGetItem()(slice(1, 2), 'missing')


@pytest.mark.parametrize('cls,kwargs', [
    (inner.ParallelResizeBilinear, {
        'ori_image_size': (8, 8), 'split_size': (2, 3), 'src_start_w': 0,
        'dst_start_w': 0, 'align_corners': False,
    }),
    (grad.ParallelResizeBilinearGrad, {
        'ori_image_size': (8, 8), 'src_start_w': 0, 'dst_start_w': 0, 'align_corners': False,
    }),
])
def test_parallel_resize_const_size(cls, kwargs):
    """
    Feature: Parallel resize const size.
    Description: Check parallel resize const size.
    Expectation: A constant resize size produces the expected shape and half-pixel metadata; a nonconstant size
        is rejected.
    """
    op = cls(**kwargs)
    args = [abstract([2, 3, 4, 4])]
    if cls is grad.ParallelResizeBilinearGrad:
        args.append(abstract([2, 3, 8, 8]))
        shape = [2, 3, 8, 8]
    else:
        shape = [2, 3, 2, 3]
    assert op.__infer__(*args, abstract([2], I32, (2, 3))) == abstract(shape)
    assert op.attrs['half_pixel_centers'] is False
    with pytest.raises(ValueError, match='const'):
        op.__infer__(*args, abstract([2], I32))


@pytest.mark.parametrize('xshapes,yshapes', [(([2, 3],), ([2, 3],)),
                         (([-1, 3],), ([2, 3],)), (([-2],), ([2, 3],)), (([],), ([2, 3],))])
def test_check_bprop_dynamic_shapes(xshapes, yshapes):
    """
    Feature: Check bprop dynamic shapes.
    Description: Check check bprop dynamic shapes.
    Expectation: Compatible dynamic backward shapes preserve the original input-shape structure.
    """
    assert inner.CheckBprop().infer_shape(xshapes, yshapes) == xshapes


@pytest.mark.parametrize('method,args,error,reason', [
    ('infer_shape', (([2],), ()), ValueError, 'number'),
    ('infer_shape', (([2],), ([2, 3],)), ValueError, 'shape'),
    ('infer_shape', (([2, 4],), ([2, 3],)), ValueError, 'shape'),
    ('infer_dtype', ((F32,), ()), ValueError, 'number'),
    ('infer_dtype', ((F32,), (I32,)), TypeError, 'dtype'),
    ('infer_dtype', ((F32,), (dt.FunctionType(),)), TypeError, 'type'),
])
def test_check_bprop_errors(method, args, error, reason):
    """
    Feature: Check bprop errors.
    Description: Check check bprop errors.
    Expectation: Invalid check bprop errors raise the parameterized exception type.
    """
    with pytest.raises(error, match=reason):
        getattr(inner.CheckBprop(), method)(*args)


def test_inner_misc_python_protocols():
    """
    Feature: Inner misc python protocols.
    Description: Check inner misc python protocols.
    Expectation: Internal metadata, subclass checks, tile sizes, gradient lookup, and mixed conversions retain
        their specified values and identities.
    """
    op = inner.PartitionedCall(graph='graph', executor_type='test')
    assert op.infer_shape([2]) is NotImplementedError
    assert op.infer_dtype(F32) is NotImplementedError
    assert op.graph == 'graph'
    assert inner.KMeansCentroids(True).infer_shape([4, 3], [2, 3], [1, 2], [4, 1]) == ((2, 3), (2, 1), 1)
    assert inner.CheckBprop().infer_dtype((F32, dt.AnythingType()), (F32, I32))[0] == F32
    assert inner._is_subclass_(int, dt.number) is False
    assert inner._is_subclass_(dt.int32, dt.number) is True
    assert inner.TileSize()((2, 1), (2, 5), 3) == (1, 5, 1)
    value = Tensor([1.0], dt.float32)
    parameter = Parameter(value, name='weight')
    gradients = ((0, value), (('weight', value), (2, value)))
    assert inner.GetGrad()(gradients, 0) is value
    assert inner.GetGrad()(gradients, parameter) is value
    with pytest.raises(TypeError, match='integer or a Parameter'):
        inner.GetGrad()(gradients, 'weight')
    with pytest.raises(RuntimeError, match='Can not find'):
        inner.GetGrad()(gradients, 4)
    result = inner.MixedPrecisionCast()(dt.float16, (value, Tensor([1], dt.int32), 3))
    assert result[0].dtype == dt.float16
    assert result[1].dtype == dt.int32
    assert result[2] == 3


@pytest.mark.parametrize('mode', ['iou', 'iof'])
def test_iou_mode(mode):
    """
    Feature: Iou mode.
    Description: Check iou mode.
    Expectation: The IOU constructor records the requested supported overlap mode unchanged.
    """
    assert other.IOU(mode).attrs['mode'] == mode


def test_other_operator_contracts():
    """
    Feature: Other operator contracts.
    Description: Check other operator contracts.
    Expectation: State identity, side-effect metadata, scalar outputs, and dtype/shape inference match the
        explicit contracts and invalid settings are rejected.
    """
    with pytest.raises(KeyError, match='mode'):
        other.IOU('bad')
    op = other._DynamicLossScale(layer=2)
    assert op.infer_shape([2, 3], []) == [2, 3]
    assert op.infer_dtype(F32, F32) == F32
    with pytest.raises(TypeError, match='layer'):
        other._DynamicLossScale(layer='bad')
    state = object()
    assert other.UpdateState()(state, 3) is state
    assert other.Reusing()(state) is state
    for cls, dtype in [(other.Push, dt.uint64), (other.Pull, dt.float32)]:
        op = cls()
        assert op.infer_shape([3], [3]) == [1]
        assert op.infer_dtype(F32, F32) == dtype
        assert op.attrs['primitive_target'] == 'CPU'
    op = other.PyExecute()
    assert op.infer_shape() == ((1,),)
    assert op.infer_dtype() == (dt.int32,)
    assert op.attrs['side_effect_io'] is True


@pytest.mark.parametrize('weights', [None, [4]])
def test_confusion_matrix_inference(weights):
    """
    Feature: Confusion matrix inference.
    Description: Check confusion matrix inference.
    Expectation: Inference yields a 3-by-3 int32 matrix with or without weights and rejects incompatible shapes
        and dtypes.
    """
    op = other.ConfusionMatrix(3)
    assert op.infer_shape([4], [4], weights) == (3, 3)
    assert op.infer_dtype(I32, I32, I32 if weights else None) == I32
    with pytest.raises(ValueError, match='shape'):
        op.infer_shape([4], [3], weights)
    with pytest.raises(TypeError, match='type|Type'):
        op.infer_dtype(I32, F32)


@pytest.mark.parametrize('out_types,out_shapes,expected_shape,expected_type,fake', [
    ([dt.float32], [(2, 3)], ((2, 3),), (F32,), False),
    ([], [], ((1,),), (I32,), True),
])
def test_pyfunc_output_protocol(out_types, out_shapes, expected_shape, expected_type, fake):
    """
    Feature: Pyfunc output protocol.
    Description: Check pyfunc output protocol.
    Expectation: PyFunc records the callback identity, expected output shape/dtype, fake-output flag, and
        disabled IO side effect.
    """
    def callback(x):
        return x
    op = other.PyFunc(callback, [dt.float32], [(2, 3)], out_types, out_shapes, stateful=False)
    assert op.infer_shape() == expected_shape
    assert op.infer_dtype() == expected_type
    assert op.attrs['fake_output'] is fake
    assert op.attrs['fn_id'] == id(callback)
    assert op.attrs['side_effect_io'] is False


@pytest.mark.parametrize('args,reason', [(([], [()], [], []), 'in_types length'),
                         (([], [], [dt.float32], []), 'out_types length')])
def test_pyfunc_mismatched_metadata(args, reason):
    """
    Feature: Pyfunc mismatched metadata.
    Description: Check pyfunc mismatched metadata.
    Expectation: Invalid pyfunc mismatched metadata raise ValueError.
    """
    with pytest.raises(ValueError, match=reason):
        other.PyFunc(lambda: None, *args)


SCALAR_CASES = [
    ('ScalarDiv', (7, 2), 3.5), ('ScalarFloorDiv', (-7, 2), -4),
    ('ScalarAdd', (2, 3), 5), ('ScalarPow', (2, 3), 8), ('ScalarLog', (1,), 0.0),
    ('ScalarUadd', (-2,), -2), ('ScalarUsub', (-2,), 2), ('ScalarSub', (2, 3), -1),
    ('ScalarMul', (2, 3), 6), ('ScalarEq', (2, 3), False), ('ScalarGt', (3, 2), True),
    ('ScalarLt', (3, 2), False), ('ScalarGe', (2, 2), True), ('ScalarLe', (2, 2), True),
    ('ScalarMod', (-7, 2), 1), ('ScalarBool', (0,), False), ('ScalarBool', (-1,), True),
    ('ScalarMax', (2, 3), 3), ('ScalarMin', (2, 3), 2),
]


@pytest.mark.parametrize('name,args,expected', SCALAR_CASES, ids=[x[0]+str(i) for i,x in enumerate(SCALAR_CASES)])
def test_manual_scalar_values(name, args, expected):
    """
    Feature: Manual scalar values.
    Description: Check manual scalar values.
    Expectation: Each manually defined scalar operation returns the independently specified Python scalar
        result.
    """
    assert getattr(manual, name)()(*args) == expected


@pytest.mark.parametrize('name,reason', [('ScalarDiv', 'divisor'),
                         ('ScalarFloorDiv', 'divisor'), ('ScalarMod', 'modulo')])
def test_manual_scalar_zero_divisor(name, reason):
    """
    Feature: Manual scalar zero divisor.
    Description: Check manual scalar zero divisor.
    Expectation: Invalid manual scalar zero divisor raise ValueError.
    """
    with pytest.raises(ValueError, match=reason):
        getattr(manual, name)()(2, 0)


@pytest.mark.parametrize('name,axis,keepdims,expected', [
    ('ReduceMax', 1, False, [3, 6]), ('ReduceMin', (1, 1), True, [[1], [4]]),
    ('ReduceProd', [], False, 720), ('ReduceMean', 0, False, [2.5, 3.5, 4.5]),
    ('ReduceAll', (), False, True), ('ReduceAny', [], False, True),
])
def test_manual_reduce_constants(name, axis, keepdims, expected):
    """
    Feature: Manual reduce constants.
    Description: Check manual reduce constants.
    Expectation: Constant reductions match NumPy; unavailable tensor values and unknown reduction names return
        None.
    """
    tensor = Tensor([[1, 2, 3], [4, 5, 6]], dt.float32)
    result = manual._infer_value_for_Reduce(tensor, axis, keepdims, name)
    np.testing.assert_allclose(result.asnumpy(), expected)
    assert manual._infer_value_for_Reduce(None, axis, keepdims, name) is None
    assert manual._infer_value_for_Reduce(tensor, axis, keepdims, 'Unknown') is None


@pytest.mark.parametrize('name,axis,expected', [('MeanExt', 1, [2, 5]),
                         ('SumExt', (0, 0), [5, 7, 9]), ('ProdExt', [], 720)])
@pytest.mark.parametrize('cast_dtype', [None, 43])
def test_manual_extended_reduce(name, axis, expected, cast_dtype):
    """
    Feature: Manual extended reduce.
    Description: Check manual extended reduce.
    Expectation: Extended constant reductions match the independently specified values for the chosen axis and
        retained-dimension policy.
    """
    tensor = Tensor([[1, 2, 3], [4, 5, 6]], dt.float32)
    result = manual._infer_value_for_ReduceExtand(tensor, axis, False, cast_dtype, name)
    np.testing.assert_allclose(result.asnumpy(), expected)


def test_manual_constant_folding_tensor_values():
    """
    Feature: Manual constant folding tensor values.
    Description: Check manual constant folding tensor values.
    Expectation: Constant folding preserves exact equality, concatenation, gather, softmax, reduction, and
        broadcast values, with None for unresolved inputs.
    """
    x = Tensor([[1, 2], [3, 4]], dt.float32)
    y = Tensor([[1, 2], [3, 0]], dt.float32)
    assert manual.infer_value_for_EqualExt(x, x).item() is True
    assert manual.infer_value_for_EqualExt(x, y).item() is False
    assert manual.infer_value_for_EqualExt(None, x) is None
    np.testing.assert_array_equal(manual.infer_value_for_Concat((x, y), 0).asnumpy(), [[1, 2], [3, 4], [1, 2], [3, 0]])
    assert manual.infer_value_for_Concat([], 0) is None
    idx = Tensor([[1, 0], [0, 1]], dt.int32)
    np.testing.assert_array_equal(manual.infer_value_for_GatherD(x, 1, idx).asnumpy(), [[2, 1], [3, 4]])
    assert manual.infer_value_for_GatherD(None, 1, idx) is None
    np.testing.assert_allclose(manual.infer_value_for_Softmax(Tensor([[0., 0.]]), -1).asnumpy(), [[0.5, 0.5]])
    assert manual.infer_value_for_Softmax(x, None) is None
    assert manual.infer_value_for_ReduceSum(x, (), False, True) is x
    assert manual.infer_value_for_ReduceSum(x, [], False, False).item() == 10
    np.testing.assert_array_equal(manual.infer_value_for_ReduceSum(x, (0, 0), True, False).asnumpy(), [[4, 6]])
    np.testing.assert_array_equal(manual.infer_value_for_ReduceSum(x, 1, False, False).asnumpy(), [3, 7])
    assert manual.infer_value_for_ReduceSum(None, 0, False, False) is None
    np.testing.assert_array_equal(manual.infer_value_for_BroadcastTo(
        Tensor([1, 2]), (2, 2)).asnumpy(), [[1, 2], [1, 2]])
    np.testing.assert_array_equal(manual.infer_value_for_Reshape(x, (4,)).asnumpy(), [1, 2, 3, 4])


@pytest.mark.parametrize('mode,expected', [(1, [-3., 3.]), (2, [-4., 3.]), (0, None)])
def test_manual_division_rounding(mode, expected):
    """
    Feature: Manual division rounding.
    Description: Check manual division rounding.
    Expectation: Division folding honors the selected rounding mode, preserves the explicit expected values, and
        returns None for unresolved inputs.
    """
    x = Tensor([-7., 7.], dt.float32)
    for func, divisor in [(manual.infer_value_for_DivMod, Tensor([2., 2.], dt.float32)),
                           (manual.infer_value_for_DivMods, 2.)]:
        result = func(x, divisor, mode)
        if expected is None:
            assert result is None
        else:
            np.testing.assert_array_equal(result.asnumpy(), expected)
        assert func(None, divisor, mode) is None


@pytest.mark.parametrize('value,dtype,expected_dtype',
                         [(7, dt.int32, dt.float32), (7., dt.float16, dt.float16), ([7., -7.], dt.float32, dt.float32)])
def test_manual_divs_scalar_dtype(value, dtype, expected_dtype):
    """
    Feature: Manual divs scalar dtype.
    Description: Check manual divs scalar dtype.
    Expectation: Scalar division returns the expected promoted dtype and signed fractional values.
    """
    result = manual.infer_value_for_Divs(Tensor(value, dtype), 2)
    assert result.dtype == expected_dtype
    np.testing.assert_allclose(result.asnumpy(), [3.5, -3.5] if isinstance(value, list) else 3.5)


def test_manual_norm_cast_tile_typeas():
    """
    Feature: Manual norm cast tile typeas.
    Description: Check manual norm cast tile typeas.
    Expectation: Norms return their literal values, no-op casts retain identity, tiling preserves element order,
        and invalid operand types are rejected.
    """
    x = Tensor([3., 4.], dt.float32)
    assert manual.infer_value_for_LinalgVectorNorm(x, 2, None, False, None).item() == 5
    assert manual.infer_value_for_LinalgVectorNorm(x, 0, None, False, None).item() == 2
    assert manual.infer_value_for_LpNormV2(x).item() == 5
    assert manual.infer_value_for_LpNormV2(None) is None
    op = manual.Cast()
    assert op(x, dt.float32) is x
    assert op(3, dt.int32).item() == 3
    assert op(x, dt.float16).dtype == dt.float16
    assert manual.TypeAs()(x, x) is x
    tile = manual.Tile()
    yes, value = tile.check_elim(x, (1,))
    assert yes
    np.testing.assert_array_equal(value.asnumpy(), [3., 4.])
    assert tile.check_elim(x, (2,)) == (False, None)
    assert tile.check_elim(x, (1, 1)) == (False, None)
    np.testing.assert_array_equal(tile(x, (2,)).asnumpy(), [3., 4., 3., 4.])
    with pytest.raises(TypeError, match='Tensor'):
        tile.check_elim([1], (1,))
    with pytest.raises(TypeError, match='tuple'):
        tile.check_elim(x, [1])


GRAD_INIT_CASES = [
    ('SparseFillEmptyRowsGrad', {}, 'input_names', ['reverse_index_map', 'grad_values']),
    ('ScaleAndTranslateGrad', {}, 'kernel_type', 'lanczos3'),
    ('SoftmaxGrad', {}, 'output_names', ['z']),
    ('SyncBatchNormGrad', {}, 'device_num', 2),
    ('KLDivLossGrad', {'reduction': 'batchmean'}, 'reduction', 'batchmean'),
    ('LuUnpackGrad', {'L_grad_flag': True, 'U_grad_flag': False}, 'cust_aicpu', 'LuUnpackGrad'),
    ('Conv3DBackpropFilter', {'out_channel': 4, 'kernel_size': 3,
     'pad_mode': 'pad', 'pad': (1, 2, 1, 2, 1, 2)}, 'pad_list', (1, 2, 1, 2, 1, 2)),
    ('Conv2DBackpropFilter', {'out_channel': 4, 'kernel_size': 3}, 'pad', (0, 0, 0, 0)),
    ('DepthwiseConv2dNativeBackpropFilter', {'channel_multiplier': 2, 'kernel_size': 3}, 'data_format', 'NCHW'),
    ('DepthwiseConv2dNativeBackpropInput', {'channel_multiplier': 2,
     'kernel_size': 3, 'pad': (1, 2, 1, 2)}, 'pad', (1, 2, 1, 2)),
    ('FlattenGrad', {}, 'output_names', ['output']),
    ('InstanceNormGrad', {}, 'output_names', ['dx', 'bn_gamma', 'bn_beta']),
    ('EinsumGrad', {'equation': 'ij,jk->ik'}, 'equation', 'ij,jk->ik'),
    ('BNTrainingReduceGrad', {}, 'data_format', 'NCHW'),
    ('BNTrainingUpdateGrad', {'data_format': 'NHWC'}, 'data_format', 'NHWC'),
    ('AdaptiveAvgPool2DGrad', {}, 'output_names', ['output_grad']),
    ('AdaptiveAvgPool3DGrad', {}, 'output_names', ['x_grad']),
    ('AvgPool3DGrad', {}, 'pad_list', (0, 0, 0, 0, 0, 0)),
    ('MaxPool3DGrad', {'pad_mode': 'pad', 'pad_list': (1, 1, 2, 2, 0, 0)}, 'pad_list', (1, 1, 2, 2, 0, 0)),
    ('MaxPoolGradWithArgmaxV2', {'kernel_size': 2}, 'strides', (1, 1, 2, 2)),
    ('MaxPool3DGradWithArgmax', {'ksize': 2, 'strides': 1, 'pads': 0}, 'ksize', (2, 2, 2)),
    ('L2NormalizeGrad', {'axis': 1}, 'axis', [1]),
    ('DynamicRNNGrad', {}, 'forget_bias', 0.0),
    ('DynamicGRUV2Grad', {}, 'gate_order', 'rzh'),
    ('RandomGammaGrad', {}, 'side_effect_hidden', True),
    ('MirrorPadGrad', {'mode': 'SYMMETRIC'}, 'mode', 'SYMMETRIC'),
    ('MvlgammaGrad', {'p': 3}, 'p', 3),
    ('PdistGrad', {}, 'p', 2.0),
    ('MultiMarginLossGrad', {}, 'reduction', 'mean'),
    ('MaxUnpool2DGrad', {'ksize': (1, 1, 2, 2), 'strides': (1, 1, 2, 2), 'pads': (0, 0, 0, 0)}, 'ksize', (1, 1, 2, 2)),
    ('MaxUnpool3DGrad', {'ksize': (1, 1, 2, 2, 2), 'strides': (
        1, 1, 2, 2, 2), 'pads': (0, 0, 0, 0, 0)}, 'ksize', (1, 1, 2, 2, 2)),
    ('AdaptiveMaxPool3DGrad', {}, 'output_names', ['output_grad']),
    ('TraceGrad', {}, 'output_names', ['x_grad']),
    ('MapTensorGetGrad', {}, 'side_effect_mem', True),
    ('ResizeV2Grad', {}, 'cubic_coeff_a', -0.75),
    ('WKVGrad', {}, 'output_names', ['gw', 'gu', 'gk', 'gv']),
]


@pytest.mark.parametrize('name,kwargs,key,value', GRAD_INIT_CASES, ids=[x[0] for x in GRAD_INIT_CASES])
def test_gradient_constructor_contracts(name, kwargs, key, value):
    """
    Feature: Gradient constructor contracts.
    Description: Check gradient constructor contracts.
    Expectation: Each gradient constructor records its independently specified normalized attribute value.
    """
    assert getattr(grad, name)(**kwargs).attrs[key] == value


@pytest.mark.parametrize('cls', [grad.LSTMGradData, grad.LSTMGradWeight, grad.LSTMGrad,
                         grad.GruGradData, grad.GruGradWeight, grad.GRUV2Grad])
@pytest.mark.parametrize('bidirectional,directions', [(False, 1), (True, 2)])
def test_recurrent_gradient_constructor(cls, bidirectional, directions):
    """
    Feature: Recurrent gradient constructor.
    Description: Check recurrent gradient constructor.
    Expectation: Recurrent gradient constructors retain input/hidden sizes and direction count and reject
        invalid sizes and dropout.
    """
    op = cls(3, 4, 2, True, bidirectional, 0.5)
    assert op.num_directions == directions
    assert op.input_size == 3
    assert op.hidden_size == 4
    with pytest.raises(ValueError, match='input_size'):
        cls(0, 4, 2, True, bidirectional, 0.5)
    with pytest.raises(ValueError, match='dropout'):
        cls(3, 4, 2, True, bidirectional, 1.5)


@pytest.mark.parametrize('bidirectional,bias,layers,weight_size',
                         [(False, False, 1, 84), (False, True, 2, 228), (True, True, 2, 552)])
def test_gru_gradient_inference(bidirectional, bias, layers, weight_size):
    """
    Feature: Gru gradient inference.
    Description: Check gru gradient inference.
    Expectation: GRU gradient shapes and weight sizes match the explicit formulas and dtypes; inconsistent
        hidden shapes are rejected.
    """
    op = grad.GruGradWeight(3, 4, layers, bias, bidirectional, 0.0)
    assert op.infer_shape([5, 2, 3], [layers, 2, 4], [5, 2, 4], [1], [1]) == (weight_size, 1, 1)
    assert op.infer_dtype(F32, F16, F32, F32, F32) == F16
    directions = 2 if bidirectional else 1
    h = [layers*directions, 2, 4]
    y = [5, 2, 4*directions]
    op = grad.GruGradData(3, 4, layers, bias, bidirectional, 0.0)
    assert op.infer_shape(y, y, h, [1], h, [1], [1]) == ((5, 2, 3), h)
    assert op.infer_dtype(*([F32]*7)) == (F32, F32)
    with pytest.raises(ValueError, match='h_shape'):
        op.infer_shape(y, y, [99, 2, 4], [1], h, [1], [1])


@pytest.mark.parametrize('cls',
    [grad.MaxPoolGrad,
    grad.AvgPoolGradVm,
    grad.AvgPoolGradGe,
    grad.AvgPoolGradV1,
     grad.MaxPoolGradWithArgmax])
@pytest.mark.parametrize('size,expected', [(2, (1, 1, 2, 2)), ((2, 3), (1, 1, 2, 3)), ((1, 1, 2, 3), (1, 1, 2, 3))])
def test_pool_gradient_kernel_normalization(cls, size, expected):
    """
    Feature: Pool gradient kernel normalization.
    Description: Check pool gradient kernel normalization.
    Expectation: Pooling gradient kernel sizes and strides normalize to the expected tuples and padding mode
        becomes SAME.
    """
    op = cls(kernel_size=size, strides=size, pad_mode='same')
    if cls is grad.MaxPoolGradWithArgmax:
        expected = (1, 2, 2, 1) if isinstance(size, int) else ((1, 2, 3, 1) if len(size) == 2 else size)
    assert op.attrs['kernel_size'] == expected
    assert op.attrs['strides'] == expected
    assert op.attrs['pad_mode'] == 'SAME'


@pytest.mark.parametrize('cls', [grad.MaxPoolGrad, grad.AvgPoolGradV1, grad.MaxPoolGradV1, grad.MaxPoolGradWithArgmax])
@pytest.mark.parametrize('size', [0, (2,), (1, -1), (1, 1, 1, 'bad')])
def test_pool_gradient_rejects_bad_kernel(cls, size):
    """
    Feature: Pool gradient rejects bad kernel.
    Description: Check pool gradient rejects bad kernel.
    Expectation: Invalid pool gradient bad kernel raise ValueError. Diagnostics identify kernel_size.
    """
    with pytest.raises(ValueError, match='kernel_size'):
        cls(kernel_size=size)


def test_maxpoolgrad_v1_tuple_format():
    """
    Feature: Maxpoolgrad v1 tuple format.
    Description: Check maxpoolgrad v1 tuple format.
    Expectation: NHWC gradient kernel and stride tuples retain the expected spatial and batch/channel ordering.
    """
    op = grad.MaxPoolGradV1(kernel_size=(2, 3), strides=(1, 2), data_format='NHWC')
    assert op.attrs['kernel_size'] == (1, 2, 3, 1)
    assert op.attrs['strides'] == (1, 1, 2, 1)


@pytest.mark.parametrize('name,kwargs,error,reason', [
    ('ScaleAndTranslateGrad', {'kernel_type': 'bad'}, ValueError, 'kernel_type'),
    ('SyncBatchNormGrad', {'group': 3}, TypeError, 'group'),
    ('SyncBatchNormGrad', {'epsilon': 0.0}, ValueError, 'epsilon'),
    ('SyncBatchNormGrad', {'device_num': 1}, ValueError, 'device_num'),
    ('KLDivLossGrad', {'reduction': 'bad'}, ValueError, 'reduction'),
    ('Conv3DBackpropFilter', {'out_channel': 2, 'kernel_size': 3, 'pad': 1}, ValueError, 'pad_mode'),
    ('Conv3DBackpropFilter', {'out_channel': 2, 'kernel_size': 3, 'pad_mode': 'pad', 'pad': -1}, ValueError, 'pad'),
    ('Conv2DBackpropFilter', {'out_channel': 2, 'kernel_size': 3, 'data_format': 'NHWC'}, ValueError, 'GPU'),
    ('Conv2DBackpropFilter', {'out_channel': 2, 'kernel_size': 3, 'pad': (1, 2)}, ValueError, 'pad size'),
    ('MaxPoolGrad', {'data_format': 'NHWC'}, ValueError, 'GPU'),
    pytest.param('MaxPool3DGrad', {'pad_list': (1, 2, 3)}, ValueError, 'pad_list',
                 marks=pytest.mark.xfail(strict=True, reason='Short pad_list is indexed before length validation')),
    ('MaxPool3DGrad', {'pad_list': (1, 2)}, ValueError, 'pad_list'),
    ('MaxPool3DGrad', {'pad_list': 1}, ValueError, 'pad_mode'),
    ('MaxPool3DGrad', {'pad_mode': 'pad', 'pad_list': -1}, ValueError, 'pad_list'),
    ('L2NormalizeGrad', {'axis': (0, 1)}, TypeError, 'length'),
    ('DynamicRNNGrad', {'forget_bias': 0}, TypeError, 'forget_bias'),
    ('DynamicGRUV2Grad', {'direction': 'BIDIRECTIONAL'}, ValueError, 'direction'),
    ('MirrorPadGrad', {'mode': 'bad'}, ValueError, 'mode'),
    ('PdistGrad', {'p': -1.0}, ValueError, 'non-negative'),
    ('MultiMarginLossGrad', {'p': 3}, ValueError, 'p'),
    ('ResizeV2Grad', {'mode': 'bad'}, ValueError, 'mode'),
])
def test_gradient_constructor_errors(name, kwargs, error, reason):
    """
    Feature: Gradient constructor errors.
    Description: Check gradient constructor errors.
    Expectation: Invalid gradient constructor errors raise the parameterized exception type.
    """
    with pytest.raises(error, match=reason):
        getattr(grad, name)(**kwargs)


@pytest.mark.parametrize('cls', [grad.Dilation2DBackpropInput, grad.Dilation2DBackpropFilter])
@pytest.mark.parametrize('stride,expected', [(2, (1, 1, 2, 2)), ((2, 3), (1, 1, 2, 3)), ((1, 1, 2, 3), (1, 1, 2, 3))])
def test_dilation_gradient_normalizes_stride(cls, stride, expected):
    """
    Feature: Dilation gradient normalizes stride.
    Description: Check dilation gradient normalizes stride.
    Expectation: Dilation gradients normalize strides and dilation into the specified tuples and record SAME
        padding.
    """
    op = cls(stride=stride, dilation=(2, 3), pad_mode='same')
    assert op.attrs['stride'] == expected
    assert op.attrs['dilation'] == (1, 1, 2, 3)
    assert op.attrs['pad_mode'] == 'SAME'


@pytest.mark.parametrize('cls', [grad.Dilation2DBackpropInput, grad.Dilation2DBackpropFilter])
@pytest.mark.parametrize('stride', [(2,), (2, 1, 1, 1), (0, 2), (True, 2)])
def test_dilation_gradient_invalid_stride(cls, stride):
    """
    Feature: Dilation gradient invalid stride.
    Description: Check dilation gradient invalid stride.
    Expectation: Invalid dilation gradient stride raise ValueError. Diagnostics identify stride.
    """
    with pytest.raises(ValueError, match='stride'):
        cls(stride=stride, dilation=1)


@pytest.mark.parametrize('cls', [grad.Dilation2DBackpropInput, grad.Dilation2DBackpropFilter])
def test_dilation_gradient_unsupported_nhwc(cls):
    """
    Feature: Dilation gradient unsupported nhwc.
    Description: Check dilation gradient unsupported nhwc.
    Expectation: Invalid dilation gradient unsupported nhwc raise ValueError. Diagnostics identify NHWC.
    """
    with pytest.raises(ValueError, match='NHWC'):
        cls(1, 1, data_format='NHWC')


def test_gradient_infer_methods():
    """
    Feature: Gradient infer methods.
    Description: Check gradient infer methods.
    Expectation: Gradient primitives return their independently specified tensor shapes and dtypes and reject
        unsupported argument types.
    """
    assert grad.EinsumGrad('ij,jk->ik').infer_shape(([2, 3], [3, 4]), [2, 4]) == ([2, 3], [3, 4])
    assert grad.EinsumGrad('ij,jk->ik').infer_dtype((F32, F16), F32) == (F32, F16)
    assert grad.FlattenGrad().__infer__(abstract([6]), abstract([2], I32, (2, 3))) == abstract((2, 3))
    assert grad.MaxPoolGrad().infer_shape([2, 3], [1], [1]) == [2, 3]
    assert grad.MaxPoolGrad().infer_dtype(F32, F16, F16) == F32
    assert grad.MaxPoolGradGrad().infer_shape([2, 3], [1], [1]) == [1]
    assert grad.MaxPoolGradGrad().infer_dtype(F16, F16, F16) == F16
    with pytest.raises(TypeError, match='type|Type'):
        grad.MaxPoolGradGrad().infer_dtype(F32, F32, F32)
    op = grad.MaxPoolGradGradWithArgmax()
    assert op.infer_shape([2, 3], [2, 3], [2, 3]) == [2, 3]
    assert op.infer_dtype(F16, F16, I32) == F16
    with pytest.raises(TypeError, match='Tensor'):
        op.infer_shape([2], [], [2])
    assert grad._ActivationGrad().infer_shape([2], [3]) == [3]
    assert grad._ActivationGrad().infer_dtype(F16, F32) == F32
    for cls, count in [(grad.AvgPoolGradVm, 4), (grad.AvgPoolGradGe, 2)]:
        args = [abstract([], I32, (2, 3)), abstract([6])] + [abstract([1])]*(count-2)
        assert cls().__infer__(*args) == abstract((2, 3))
    op = grad.PsROIPoolingGrad(2, 3, 4, 5, 6, 2, 2, 1.0, 3)
    assert op.infer_shape([1], [1], [1]) == [2, 3, 4, 5]
    assert op.infer_dtype(F32, F32, I32) == F32


@pytest.mark.parametrize('axis,shapes,expected', [
    (1, [[2, 3], [2, 4]], {'shape': None, 'dtype': None, 'value': ((0, 0), (0, 3))}),
    (-1, [[2, 3], [2, 4]], {'shape': None, 'dtype': None, 'value': ((0, 0), (0, 3))}),
    (1, [[2, -1], [2, 4]], abstract([2, 2], dt.int64)),
    (0, [[-2], [2, 4]], abstract([2, 2], dt.int64)),
])
def test_concat_offset_infer(axis, shapes, expected):
    """
    Feature: Concat offset infer.
    Description: Check concat offset infer.
    Expectation: Concatenation offsets match the explicit inference contract, including unknown dimensions, and
        record float32 element metadata.
    """
    op = grad.ConcatOffset(axis=axis)
    assert op.__infer__({'shape': shapes, 'dtype': [F32, F32]}) == expected
    assert op.attrs['T'] == dt.float32


@pytest.mark.parametrize('pad_mode,expected', [('VALID', (0, 0, 0, 0, 0, 0)), ('SAME', (1, 1, 0, 1, 0, 0))])
def test_maxpool3d_gradgrad_padding(pad_mode, expected):
    """
    Feature: Maxpool3d gradgrad padding.
    Description: Check maxpool3d gradgrad padding.
    Expectation: Second pooling gradients preserve the expected output shape and dtype and derive the explicit
        padding list.
    """
    x = [2, 3, 5, 6, 4]
    op = grad.MaxPool3DGradGrad(kernel_size=(3, 3, 1), strides=(2, 2, 2), pad_mode=pad_mode)
    assert op.infer_shape(x, [2, 3, 3, 3, 2], x) == [2, 3, 3, 3, 2]
    assert op.attrs['pad_list'] == expected
    assert op.infer_dtype(F32, F32, F16) == F32
    with pytest.raises(ValueError, match='rank'):
        op.infer_shape([2, 3], [1], [2, 3])


@pytest.mark.parametrize('size', [(2, 3), (-1, 3), None])
def test_slice_gradient_size_inference(size):
    """
    Feature: Slice gradient size inference.
    Description: Check slice gradient size inference.
    Expectation: Constant slice sizes reconstruct the original input abstract and inconsistent sensitivity
        shapes are rejected.
    """
    dy = abstract([2, 3])
    x = abstract([4, 5])
    assert grad.SliceGrad().__infer__(dy, x, abstract([], I32, (2, 2)), abstract([], I32, size)) == x
    with pytest.raises(ValueError, match='dy_shape'):
        grad.SliceGrad().__infer__(abstract([5, 3]), x, abstract([], I32, (0, 0)), abstract([], I32, (5, 3)))


@pytest.mark.parametrize('cls', [grad.DepthwiseConv2dNativeBackpropFilter, grad.DepthwiseConv2dNativeBackpropInput])
def test_depthwise_gradient_shape_from_const(cls):
    """
    Feature: Depthwise gradient shape from const.
    Description: Check depthwise gradient shape from const.
    Expectation: A constant filter shape determines the exact depthwise gradient abstract (3, 1, 3, 3).
    """
    op = cls(1, 3)
    x = abstract([2, 3, 4, 4])
    shape = abstract([4], I32, (3, 1, 3, 3))
    args = (x, shape, x) if cls is grad.DepthwiseConv2dNativeBackpropFilter else (shape, x, x)
    assert op.__infer__(*args) == abstract((3, 1, 3, 3))


def test_hybrid_runtime_intrinsics():
    """
    Feature: Hybrid runtime intrinsics.
    Description: Check hybrid runtime intrinsics.
    Expectation: Hybrid allocation, grid ordering, reciprocal square roots, error function, and stub lifecycle
        match their explicit NumPy/runtime contracts.
    """
    output = hybrid._allocate((2, 3), 'int32', 'local')
    assert output.shape == (2, 3)
    assert output.dtype == np.int32
    np.testing.assert_array_equal(output, [[0, 0, 0], [0, 0, 0]])
    np.testing.assert_allclose(hybrid._rsqrt(np.array([1., 4., 16.])), [1., .5, .25])
    for x in (-2., 0., 1., 2.):
        assert hybrid._erf(x) == pytest.approx(math.erf(x), abs=2e-7)
    assert list(hybrid._grid((2, 3))) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    stub = hybrid.WithStub()
    assert stub('test', scope='local') is stub
    with stub as same:
        assert same is stub
    # Exercise the explicit lifecycle/proxy return contract.
    # pylint: disable-next=unnecessary-dunder-call
    assert stub.__del__() is stub


@pytest.mark.parametrize('body,error,reason', [
    ('b = unknown\n    return a', ValueError, 'undeclared variable'),
    ('for i in range(4):\n        i = 0\n    return a', ValueError, 'iter var'),
    ('for i in range(4):\n        temp = 1\n    b = temp\n    return a', ValueError, 'out of the scope'),
    ('b = bad(a)\n    return a', ValueError, 'not in intrinsics'),
    ('b = sqrt(a, a)\n    return a', TypeError, 'one input'),
    ('b = power(a)\n    return a', TypeError, 'two inputs'),
    ('b = float16(a)\n    return a', ValueError, 'not available'),
    ('with attr("x"):\n        b = 1\n    return a', ValueError, 'two inputs'),
    ('with attr(1, 2):\n        b = 1\n    return a', ValueError, 'first input'),
    ('with attr("x", None):\n        b = 1\n    return a', ValueError, 'second input'),
    ('with block_realize(a, a):\n        b = 1\n    return a', ValueError, 'only one input'),
    ('with block_realize(3):\n        b = 1\n    return a', ValueError, 'tensor name'),
    ('with wrong(a):\n        b = 1\n    return a', ValueError, 'Unsupported function'),
    ('b = c = 1\n    return a', ValueError, 'multiple assignments'),
    ('b = output_tensor((2,), "float32")\n    b = allocate((2,), "float32")\n    return b', ValueError, 'redefined'),
    ('for i in range(4):\n        break\n    return a', TypeError, 'break'),
    ('for i in range(4):\n        continue\n    return a', TypeError, 'continue'),
    ('while True:\n        b = 1\n    return a', TypeError, 'while'),
    ('b = a.shape.real\n    return a', ValueError, 'getattr'),
    ('b = unknown.shape\n    return a', ValueError, 'declaration'),
    ('b = a.bad\n    return a', ValueError, 'no attribute'),
    ('return 1', TypeError, 'single tensor or a tuple'),
    ('return (a, 1)', TypeError, 'name of a tensor'),
    ('b = allocate((2,), "float32")\n    return b', TypeError, 'allocated by output_tensor'),
    ('b = output_tensor((2,), "float32")\n    return a', TypeError, 'not in the return value'),
])
def test_hybrid_dsl_rejects_invalid_programs(body, error, reason):
    # Parsing a DSL input is the production API; no implementation is extracted.
    """
    Feature: Hybrid dsl rejects invalid programs.
    Description: Check hybrid dsl rejects invalid programs.
    Expectation: Invalid hybrid dsl programs raise the parameterized exception type.
    """
    tree = ast.parse('def kernel_case(a):\n    ' + body + '\n')
    with pytest.raises(error, match=reason):
        hybrid.determine_variable_usage(tree, 'kernel_case')


def test_hybrid_dsl_tracks_output_alias_and_scopes():
    """
    Feature: Hybrid dsl tracks output alias and scopes.
    Description: Check hybrid dsl tracks output alias and scopes.
    Expectation: Variable-usage analysis identifies the expected output/input alias relation across the program
        scopes.
    """
    source = '''def valid(a):
    temp = allocate(a.shape, a.dtype)
    out = output_tensor(a.shape, a.dtype)
    with attr("scope", True):
        "scope documentation"
        for i in serial(2):
            temp[i] = sqrt(a[i])
            out[i] = temp[i]
    with block_realize(out):
        out[0] = out[0]
    return (out, a)
'''
    assert hybrid.determine_variable_usage(ast.parse(source), 'valid') == [[1, 0]]


@pytest.mark.parametrize('kwargs,reason', [({'compile_attrs': []}, 'dict'),
                         ({'compile_attrs': {1: 'x'}}, 'str'), ({'reg_info': []}, 'reg_info')])
def test_hybrid_decorator_metadata_validation(kwargs, reason):
    """
    Feature: Hybrid decorator metadata validation.
    Description: Check hybrid decorator metadata validation.
    Expectation: Invalid hybrid decorator metadata arguments raise TypeError.
    """
    with pytest.raises(TypeError, match=reason):
        hybrid.kernel(**kwargs)


def test_hybrid_decorator_executes_numpy_runtime():
    # These names are injected by the real decorator when the wrapper is called.
    """
    Feature: Hybrid decorator executes numpy runtime.
    Description: Check hybrid decorator executes numpy runtime.
    Expectation: Hybrid decorators preserve metadata and execute the original NumPy function with the expected
        values.
    """
    def hybrid_add(x):
        result = output_tensor(x.shape, x.dtype)  # pylint: disable=undefined-variable
        for i in serial(x.shape[0]):  # pylint: disable=undefined-variable
            result[i] = x[i] + 2
        return result
    decorated = hybrid.kernel(reg_info={'name': 'AddTwo'}, compile_attrs={'debug': True})(hybrid_add)
    try:
        np.testing.assert_array_equal(decorated(np.array([1., 3.])), [3., 5.])
        assert decorated.ms_kernel_flag is True
        assert json.loads(decorated.compile_attrs) == {'debug': True}
        assert decorated.reg_info == {'name': 'AddTwo'}
        assert hybrid.ms_kernel(lambda x: x)(3) == 3
        assert hybrid.kernel(lambda x: x)(4) == 4
    finally:
        # The decorator deliberately inserts runtime intrinsics into function globals.
        for key in hybrid.VariableUsage.intrin_runtime:
            globals().pop(key, None)


def test_custom_version_manager_detects_source_flags_and_directory(tmp_path):
    """
    Feature: Custom version manager detects source flags and directory.
    Description: Check custom version manager detects source flags and directory.
    Expectation: Unchanged builds are reused; changes in source, flags, or directories advance the expected
        module version.
    """
    source = tmp_path / 'module.cc'
    source.write_text('int value = 0;')
    manager = custom.VersionManager()
    args = ['module', [str(source)], ['-O2'], [], [], str(tmp_path)]
    assert manager.check_version(*args) is True
    assert manager.check_version(*args) is False
    source.write_text('int value = 1;')
    assert manager.check_version(*args) is True
    assert manager._get_version('module') == 1
    assert manager.check_version(*args) is False
    args[2] = ['-O3']
    assert manager.check_version(*args) is True
    assert manager._get_version('module') == 2


def test_custom_file_locker_contention_and_release(tmp_path):
    """
    Feature: Custom file locker contention and release.
    Description: Check custom file locker contention and release.
    Expectation: The first lock excludes contenders, waiting times out as documented, and releasing allows the
        second locker to acquire it.
    """
    first = custom.FileLocker(str(tmp_path))
    second = custom.FileLocker(str(tmp_path))
    assert first.try_lock() is True
    try:
        assert second.try_lock() is False
        with pytest.raises(RuntimeError, match='Timeout waiting'):
            second.wait(timeout=0.01)
    finally:
        first.release_lock()
    assert second.wait(timeout=0.01) is None
    assert second.try_lock() is True
    second.release_lock()
    second.release_lock()


@pytest.mark.parametrize('typ,expected',
    [('tensor',
    'aclTensor*'),
    ('int',
    'int64_t'),
    ('float',
    'float'),
    ('double',
    'double'),
    ('bool',
    'bool'),
    ('str',
    'char*'),
    ('number',
    'aclScalar*'),
    ('listInt',
    'aclIntArray*'),
    ('listBool',
    'aclBoolArray*'),
    ('listFloat',
     'aclFloatArray*')])
def test_custom_callback_type_declarations(typ, expected):
    """
    Feature: Custom callback type declarations.
    Description: Check custom callback type declarations.
    Expectation: Each supported ABI type maps to the independently specified C++ declaration.
    """
    assert custom.CustomCodeGenerator._get_type_declaration(typ) == expected


def test_custom_callback_generation_protocol():
    """
    Feature: Custom callback generation protocol.
    Description: Check custom callback generation protocol.
    Expectation: Generated callbacks retain tensor/scalar declarations and operand ordering and reject
        inconsistent or unsupported ABI metadata.
    """
    generator = custom.CustomCodeGenerator()
    info = {'inputs': [{'paramType': 'required'}, {'paramType': 'dynamic'}], 'outputs': [
        {'paramType': 'dynamic'}], 'attr': [{'type': 'int'}, {'type': 'bool'}]}
    types = ['aclTensor*', 'aclTensorList*', 'int64_t', 'bool', 'aclTensorList*', 'int64_t*', 'aclOpExecutor**']
    assert generator.get_api_types_by_reg_info(info) == types
    code = generator.generate_callback_by_types('aclnnTest', info, types)
    assert 'int64_t* input2 = static_cast<int64_t*>(inputs[2])' in code
    assert 'aclTensorList* output0 = static_cast<aclTensorList*>(outputs[0])' in code
    assert 'input0, input1, *input2, *input3, output0, workspace_size, executor' in code
    # A scalar prototype differing from registration is allowed and dereferenced.
    changed = types.copy()
    changed[2] = 'uint64_t'
    assert '*input2' in generator.generate_callback_by_types('aclnnTest', info, changed)
    with pytest.raises(RuntimeError, match='input size'):
        generator.generate_callback_by_types('aclnnTest', info, types[:-1])
    with pytest.raises(RuntimeError, match='Unsupported type'):
        generator._get_type_declaration('bad')
    with pytest.raises(RuntimeError, match='Unsupported input type'):
        generator._generate_callback_inputs(['bad'])
    with pytest.raises(RuntimeError, match='Unsupported output type'):
        generator._generate_callback_outputs(['int64_t'])
    with pytest.raises(RuntimeError, match='Unsupported input type'):
        generator._generate_callback_func_params(['bad'], [])
    with pytest.raises(RuntimeError, match='Unsupported output type'):
        generator._generate_callback_func_params([], ['bad'])


@pytest.mark.parametrize('name,pure,prefix', [('TestOp', 'TestOp',
                         'aclnnTestOp'), ('aclnnTestOp', 'TestOp', 'aclnnTestOp')])
def test_custom_info_names(name, pure, prefix):
    """
    Feature: Custom info names.
    Description: Check custom info names.
    Expectation: Custom API names normalize to the expected pure/prefixed names and header filename, while empty
        names are rejected.
    """
    info = custom.CustomInfoGenerator(name)
    assert info.pure_op_name == pure
    assert info.prefix_op_name == prefix
    assert info.aclnn_api_file_name == 'aclnn_test_op.h'
    with pytest.raises(ValueError, match='empty'):
        info._get_pure_name('')
    with pytest.raises(ValueError, match='empty'):
        info._get_prefix_name(None)


@pytest.mark.parametrize('header,error', [
    ('// aclnnTestGetWorkspaceSize(x);', 'aclnnStatus'),
    ('aclnnStatus aclnnTestGetWorkspaceSize(aclTensor* x)', 'end marker'),
])
def test_custom_info_invalid_header(tmp_path, header, error):
    """
    Feature: Custom info invalid header.
    Description: Check custom info invalid header.
    Expectation: Invalid custom info header raise ValueError.
    """
    info = custom.CustomInfoGenerator('Test')
    (tmp_path / 'aclnn_test.h').write_text(header)
    with pytest.raises(ValueError, match=error):
        info._get_aclnn_api_from_file(str(tmp_path))


def test_custom_info_discovers_real_files(tmp_path, monkeypatch):
    """
    Feature: Custom info discovers real files.
    Description: Check custom info discovers real files.
    Expectation: Header discovery extracts the literal API types and vendor priority, and missing or empty files
        return the specified fallbacks.
    """
    root = tmp_path / 'custom'
    include = root / 'op_api/include'
    include.mkdir(parents=True)
    (include / 'aclnn_test.h').write_text('''// aclnnTestGetWorkspaceSize(wrong);
aclnnStatus aclnnTestGetWorkspaceSize(const aclTensor * x, int64_t axis,
aclTensor *out, uint64_t *workspace, aclOpExecutor **executor);
''')
    monkeypatch.setenv('ASCEND_CUSTOM_OPP_PATH', str(tmp_path / 'missing')+':'+str(root))
    monkeypatch.setenv('ASCEND_OPP_PATH', str(tmp_path / 'builtin'))
    info = custom.CustomInfoGenerator('Test')
    assert info.get_aclnn_api_types() == ['aclTensor*', 'int64_t', 'aclTensor*', 'uint64_t*', 'aclOpExecutor**']
    assert info._get_aclnn_api_from_file(str(tmp_path / 'missing')) is False
    config = tmp_path / 'config.ini'
    config.write_text('# vendors\nother=none\n load_priority = vendor1, vendor2,, \n')
    assert custom.CustomInfoGenerator._parse_load_priority(str(config)) == ['vendor1', 'vendor2']
    config.write_text('other=none\n')
    # The exact empty-list result is part of this interface contract.
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert custom.CustomInfoGenerator._parse_load_priority(str(config)) == []


def test_type_as_dispatch_protocol(monkeypatch):
    """
    Feature: Type as dispatch protocol.
    Description: Check type as dispatch protocol.
    Expectation: TypeAs forwards the original primitive, source tensor, and target once and returns the dispatch
        result by identity.
    """
    x = Tensor([1.], dt.float32)
    target = Tensor([1.], dt.float16)
    calls = []
    marker = object()
    def pyboost_boundary(op, args):
        calls.append((op, args))
        return marker
    monkeypatch.setattr(manual, 'pyboost_type_as', pyboost_boundary)
    op = manual.TypeAs()
    assert op(x, target) is marker
    assert len(calls) == 1
    assert calls[0][0] is op
    assert calls[0][1][0] is x
    assert calls[0][1][1] is target


def metadata_device(monkeypatch, target):
    """Supply device capability metadata without initializing unavailable devices."""
    get_context = context.get_context
    monkeypatch.setattr(context, 'get_context', lambda key=None: target if key == 'device_target' else get_context(key))


@pytest.mark.parametrize('target', ['Ascend', 'GPU'])
def test_quantization_target_specific_metadata(monkeypatch, target):
    """
    Feature: Quantization target specific metadata.
    Description: Check quantization target specific metadata.
    Expectation: Target-specific quantization primitives retain the expected output shapes/dtypes and reject
        invalid ranks and tensor types.
    """
    # Table cases exercise heterogeneous decorated/native API signatures.
    # pylint: disable=too-many-function-args
    metadata_device(monkeypatch, target)
    for name in ('MinMaxUpdatePerLayer', 'MinMaxUpdatePerChannel',
                 'FakeLearnedScaleQuantPerLayer', 'FakeLearnedScaleQuantPerLayerGrad',
                 'FakeLearnedScaleQuantPerChannel', 'FakeLearnedScaleQuantPerChannelGrad',
                 'FakeQuantPerLayer', 'FakeQuantPerLayerGrad', 'FakeQuantPerChannel', 'FakeQuantPerChannelGrad',
                 'CorrectionMul', 'CorrectionMulGrad', 'CorrectionMulGradReduce'):
        op = getattr(quant, name)()
        if name in (
    'MinMaxUpdatePerChannel',
    'FakeLearnedScaleQuantPerChannel',
     'FakeQuantPerChannel') and target == 'Ascend':
            with pytest.raises(ValueError, match='rank'):
                op.infer_shape([3], [3], [3])
        if name in (
    'FakeLearnedScaleQuantPerLayer',
    'FakeLearnedScaleQuantPerChannel',
    'FakeQuantPerLayer',
     'FakeQuantPerChannel'):
            assert op.infer_dtype(F32, F32, F32) == F32
            if target == 'GPU':
                with pytest.raises(TypeError, match='type|Type'):
                    op.infer_dtype(F16, F16, F16)
        if name in ('FakeLearnedScaleQuantPerLayerGrad', 'FakeLearnedScaleQuantPerChannelGrad'):
            assert op.infer_dtype(F32, F32, F32, F32) == (F32, F32)
        if name in ('FakeQuantPerLayerGrad', 'FakeQuantPerChannelGrad'):
            assert op.infer_dtype(F32, F32, F32, F32) == F32
        if name == 'CorrectionMulGrad' and target == 'Ascend':
            assert op.infer_shape([3, 4], [3, 4], [3], [3]) == ([3, 4], [3, 4])
            assert op.infer_dtype(F32, F32, F32, F32) == (F32, F32)


def test_te_optional_dependency_unavailable(monkeypatch):
    """
    Feature: Te optional dependency unavailable.
    Description: Check te optional dependency unavailable.
    Expectation: TE support detection returns false when the optional dependency is unavailable.
    """
    monkeypatch.setitem(sys.modules, 'te', None)
    assert quant._support_te() is False


def test_hybrid_ascend_compile_metadata(monkeypatch):
    """
    Feature: Hybrid ascend compile metadata.
    Description: Check hybrid ascend compile metadata.
    Expectation: Ascend hybrid compilation records debug mode and the required always-enabled polytops
        attribute.
    """
    metadata_device(monkeypatch, 'Ascend')
    fn = hybrid.kernel(compile_attrs={'debug': True})(lambda x: x)
    assert json.loads(fn.compile_attrs) == {'debug': True, 'enable_polytops': 'always'}


def test_gradient_target_metadata_and_remaining_constructor_branches(monkeypatch):
    """
    Feature: Gradient target metadata and remaining constructor branches.
    Description: Check gradient target metadata and remaining constructor branches.
    Expectation: Device-specific gradient kernel layouts match the explicit tuples and invalid targets, axes,
        and strides are rejected.
    """
    metadata_device(monkeypatch, 'GPU')
    op = grad.MaxPoolGrad(kernel_size=(2, 3), strides=(1, 2), data_format='NHWC')
    assert op.attrs['kernel_size'] == [1, 2, 3, 1]
    assert op.attrs['strides'] == [1, 1, 2, 1]
    assert grad.MaxPoolGradWithArgmax(data_format='NHWC').attrs['kernel_size'] == (1, 1, 1, 1)
    assert grad.MaxPoolGradV1(kernel_size=(1, 1, 2, 3), strides=(1, 1, 1, 2)).attrs['kernel_size'] == (1, 1, 2, 3)
    metadata_device(monkeypatch, 'Unknown')
    with pytest.raises(ValueError, match='unknown device'):
        grad.KLDivLossGrad()
    with pytest.raises(ValueError, match='axis'):
        grad.ConcatOffset(axis=3).__infer__({'shape': [[2, 3], [2, 4]], 'dtype': [F32, F32]})
    with pytest.raises(ValueError, match='stride'):
        grad.Dilation2DBackpropFilter(256, 1)


def test_gradient_python_only_primitive_dispatch(monkeypatch):
    # EmbeddingLookupCommGrad only delegates tuple_setitem to a C++ primitive.
    """
    Feature: Gradient python only primitive dispatch.
    Description: Check gradient python only primitive dispatch.
    Expectation: Python-only gradient inference forwards tuple updates and native arguments in order and returns
        the specified dispatch result.
    """
    # Import at the test boundary so fixture isolation is established first.
    # pylint: disable-next=import-outside-toplevel
    from mindspore.ops.primitive import Primitive
    calls = []
    original = Primitive.__call__
    def dispatch(op, *args):
        if op.name == 'tuple_setitem':
            calls.append((op.name, args))
            values, index, value = args
            assert index == 0
            return (value,) + values[1:]
        if op.name == 'MultiMarginLossGrad':
            calls.append((op.name, args))
            return 'gradient-result'
        return original(op, *args)
    monkeypatch.setattr(Primitive, '__call__', dispatch)
    op = grad.EmbeddingLookupCommGrad()
    assert op.__infer__(abstract([2, 3]), abstract([], dt.int64, 4)) == abstract((16, 3))
    assert calls == [('tuple_setitem', ((2, 3), 0, 16))]
    op = grad.MultiMarginLossGrad()
    assert op('dy', 'x', 'target') == 'gradient-result'
    assert calls[-1] == ('MultiMarginLossGrad', ('dy', 'x', 'target', None))


def test_inner_remaining_constructors_and_virtual_zero():
    """
    Feature: Inner remaining constructors and virtual zero.
    Description: Check inner remaining constructors and virtual zero.
    Expectation: Internal constructors retain expected output names and alignment metadata, with unresolved
        format values left unresolved.
    """
    assert inner.NonZeroWithValue(True).attrs['output_names'] == ['value', 'index', 'count']
    assert inner.NonZeroWithValueShape().attrs['output_names'] == ['out_value', 'out_index']
    assert inner.DynamicResizeNearestNeighbor().attrs['align_corners'] is False
    with pytest.raises(TypeError, match='align_corners'):
        inner.DynamicResizeNearestNeighbor(1)
    assert inner.Format().__infer__(abstract([], dt.string, None)) == abstract([], dt.string)


def test_cell_backward_hook_infer_and_pre_hook_protocol(monkeypatch):
    """
    Feature: Cell backward hook infer and pre hook protocol.
    Description: Check cell backward hook infer and pre hook protocol.
    Expectation: Hook inference preserves single/multiple input metadata, registers one callback, and rejects a
        pre-hook with the wrong return arity.
    """
    hooks = OrderedDict()
    op = inner.CellBackwardHook('cell', cell='example-cell', hook_dict=hooks)
    assert op.infer_shape([2, 3]) == [2, 3]
    assert op.infer_shape([2], [3]) == ([2], [3])
    assert op.infer_dtype(F32) == F32
    assert op.infer_dtype(F32, F16) == (F32, F16)
    registered = []
    # Capture only native hook registration; execute the real Python hook closure.
    monkeypatch.setattr(type(op), 'set_hook_fn', lambda self, fn, kind: registered.append((fn, kind)))
    hooks[1] = lambda cell, grad_value: ('too', 'many')
    op.register_backward_pre_hook()
    assert len(registered) == 1
    with pytest.raises(TypeError, match='return value size'):
        registered[0][0]((1,))


@pytest.mark.xfail(strict=True, reason=(
    'CusTranspose02314 bprop references undefined C and lacks the transpose '
    'gradient'
))
def test_thor_backward_preserves_transpose_gradient():
    """
    Feature: Thor backward preserves transpose gradient.
    Description: Check thor backward preserves transpose gradient.
    Expectation: Backward returns one tensor with sensitivity transposed by the independently specified axis
        permutation.
    """
    backward = thor.CusTranspose02314().get_bprop()
    data = Tensor(np.zeros((1, 1, 2, 3, 16), np.float32))
    sensitivity = np.arange(96, dtype=np.float32).reshape(1, 2, 3, 1, 16)
    result = backward(data, None, Tensor(sensitivity))
    assert isinstance(result, tuple) and len(result) == 1
    np.testing.assert_array_equal(result[0].asnumpy(), np.transpose(sensitivity, (0, 3, 1, 2, 4)))


def test_manual_control_flow_fallbacks():
    """
    Feature: Manual control flow fallbacks.
    Description: Check manual control flow fallbacks.
    Expectation: Python loop fallbacks produce the literal carries and collected outputs, handle empty inputs,
        and propagate body exceptions.
    """
    assert manual.WhileLoop()(lambda value: value < 4, lambda value: value+1, 0) == 4
    assert manual.WhileLoop()(lambda value: False, lambda value: value+1, 7) == 7
    assert manual.Scan()(lambda carry, x: (carry+x, carry+x), 0, [1, 2, 3]) == (6, [1, 3, 6])
    assert manual.Scan()(lambda carry, x: (carry, x), 4, []) == (4, [])
    assert manual.ForiLoop()(0, 4, lambda index, carry: carry+index, 0) == 6
    def broken(*args):
        raise RuntimeError('body failure')
    with pytest.raises(ValueError, match='body failure'):
        manual.WhileLoop()(lambda value: True, broken, 0)
    with pytest.raises(ValueError, match='body failure'):
        manual.Scan()(broken, 0, [1])
    with pytest.raises(ValueError, match='body failure'):
        manual.ForiLoop()(0, 1, broken, 0)


def test_manual_shape_constants_and_bfloat16():
    """
    Feature: Manual shape constants and bfloat16.
    Description: Check manual shape constants and bfloat16.
    Expectation: Constant shape operations preserve values and bfloat16 where supported, return None for
        unresolved data, and reject invalid reshape dimensions.
    """
    x = Tensor([[1, 2], [3, 4]], dt.float32)
    assert manual.infer_value_for_Cast(7, 43).item() == 7.0
    for fn in (manual.infer_value_for_BroadcastTo, manual.infer_value_for_Reshape):
        assert fn(x, None) is None
        assert fn(x, (None,)) is None
        assert fn(None, (2, 2)) is None
        np.testing.assert_array_equal(fn(x, Tensor([2, 2], dt.int32)).asnumpy(), [[1, 2], [3, 4]])
    np.testing.assert_array_equal(manual.infer_value_for_BroadcastTo(x, (-1, 2)).asnumpy(), [[1, 2], [3, 4]])
    with pytest.raises(ValueError, match='negative'):
        manual.infer_value_for_BroadcastTo(x, (-1, 2, 2))
    with pytest.raises(ValueError, match='at most one'):
        manual.infer_value_for_Reshape(x, (-1, -1))
    with pytest.raises(ValueError, match='product'):
        manual.infer_value_for_Reshape(x, (-2,))
    with pytest.raises(ValueError, match='product'):
        manual.infer_value_for_Reshape(x, (3,))
    bfloat = Tensor([1., 2.], dt.bfloat16)
    assert manual.infer_value_for_Concat((bfloat, bfloat), 0).dtype == dt.bfloat16
    assert manual.infer_value_for_DivMod(bfloat, bfloat, 1) is None
    assert manual.infer_value_for_DivMods(bfloat, 2, 1) is None


def test_manual_batch_norm_cpu_values():
    """
    Feature: Manual batch norm cpu values.
    Description: Check manual batch norm cpu values.
    Expectation: CPU batch-normalization folding produces the independently specified values within the stated
        numerical tolerance.
    """
    x = Tensor([[[[1., 2.], [3., 4.]]]], dt.float32)
    result = manual.batch_norm_(x, Tensor([1.]), Tensor([0.]), Tensor([0.]), Tensor([1.]))
    np.testing.assert_allclose(result[0].asnumpy(), [[[[0.999995, 1.99999], [2.999985, 3.99998]]]], rtol=1e-5)


def test_custom_ninja_unavailable_boundary(monkeypatch):
    """
    Feature: Custom ninja unavailable boundary.
    Description: Check custom ninja unavailable boundary.
    Expectation: The version query uses ninja --version and reports the documented error when Ninja is
        unavailable.
    """
    def missing_ninja(args):
        assert args == ['ninja', '--version']
        raise FileNotFoundError('ninja')
    monkeypatch.setattr(custom.subprocess, 'check_output', missing_ninja)
    with pytest.raises(RuntimeError, match='Ninja is required'):
        custom.ExtensionBuilder._verify_ninja_availability()


def test_custom_ninja_failure_cleans_artifacts(tmp_path, monkeypatch):
    """
    Feature: Custom ninja failure cleans artifacts.
    Description: Check custom ninja failure cleans artifacts.
    Expectation: A failed verbose Ninja build retains compiler diagnostics and removes both final and temporary
        shared-library artifacts.
    """
    for name in ('module.so', 'module.so.tmp'):
        (tmp_path / name).write_bytes(b'partial')
    def failed_process(cmd, **kwargs):
        assert cmd == ['ninja', '-v']
        assert kwargs['cwd'] == str(tmp_path)
        assert kwargs['check'] is True
        assert kwargs['stderr'] is kwargs['stdout']
        kwargs['stdout'].write('compiler diagnostic: invalid C++\n')
        raise subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(custom.subprocess, 'run', failed_process)
    with pytest.raises(RuntimeError, match='compiler diagnostic'):
        custom.ExtensionBuilder(str(tmp_path))._run_ninja_build('module')
    assert not (tmp_path / 'module.so').exists()
    assert not (tmp_path / 'module.so.tmp').exists()
    assert 'invalid C++' in (tmp_path / '.build_log.txt').read_text()


def test_custom_compile_unchanged_and_completed_contender(tmp_path, monkeypatch):
    """
    Feature: Custom compile unchanged and completed contender.
    Description: Check custom compile unchanged and completed contender.
    Expectation: Unchanged builds are skipped and a completed competing build is reused without overwriting its
        shared-library bytes.
    """
    manager = custom.VersionManager()
    monkeypatch.setattr(custom, 'version_manager', manager)
    source = tmp_path / 'module.cc'
    source.write_text('int module = 0;')
    args = ['module', [str(source)], [], [], []]
    assert manager.check_version(*args, str(tmp_path)) is True
    assert custom.ExtensionBuilder(str(tmp_path))._compile(*args) is None
    source.write_text('int module = 1;')
    (tmp_path / 'module.so').write_bytes(b'\x7fELFcompleted')
    locker = custom.FileLocker(str(tmp_path))
    assert locker.try_lock()
    timer = threading.Timer(0.02, locker.release_lock)
    timer.start()
    try:
        assert custom.ExtensionBuilder(str(tmp_path))._compile(*args) is None
    finally:
        timer.join()
        locker.release_lock()
    assert (tmp_path / 'module.so').read_bytes() == b'\x7fELFcompleted'


def test_custom_vendor_discovery_and_missing_attribute(tmp_path, monkeypatch):
    """
    Feature: Custom vendor discovery and missing attribute.
    Description: Check custom vendor discovery and missing attribute.
    Expectation: Vendor discovery returns the literal ABI type and reports missing operator attributes and
        absent vendor operators.
    """
    monkeypatch.delenv('ASCEND_CUSTOM_OPP_PATH', raising=False)
    monkeypatch.setenv('ASCEND_OPP_PATH', str(tmp_path))
    vendor = tmp_path / 'vendors/vendor'
    include = vendor / 'op_api/include'
    include.mkdir(parents=True)
    (tmp_path / 'vendors/config.ini').write_text('load_priority=missing,vendor\n')
    (include / 'aclnn_test.h').write_text('aclnnStatus aclnnTestGetWorkspaceSize(aclTensor *x);')
    info = custom.CustomInfoGenerator('Test')
    assert info.get_aclnn_api_types() == ['aclTensor*']
    soc = custom.MSContext.get_instance().get_ascend_soc_version()
    relative = 'op_impl/ai_core/tbe/config/' + soc
    directory = vendor / relative
    directory.mkdir(parents=True)
    filename = directory / ('aic-' + soc + '-ops-info.json')
    filename.write_text(json.dumps(
        {'Test': {'input0': {'name': 'x'}, 'output0': {'name': 'y'}, 'attr': {'list': 'axis'}}}))
    with pytest.raises(KeyError, match='attr_axis'):
        info._generate_reg_info()
    filename.write_text(json.dumps({'Other': {}}))
    (tmp_path / 'built-in' / relative).mkdir(parents=True)
    with pytest.raises(RuntimeError, match='Cannot find operator'):
        info._get_op_json()


@pytest.mark.parametrize('name', ['Cast', 'Tile'])
@pytest.mark.parametrize('compiled', [False, True])
def test_manual_jit_dispatch_protocol(name, compiled, monkeypatch):
    """
    Feature: Manual jit dispatch protocol.
    Description: Check manual jit dispatch protocol.
    Expectation: Manual operations preserve argument identity through JIT/native dispatch and return the
        expected result for elimination and dispatch paths.
    """
    x = Tensor([1., 2.], dt.float32)
    native_result = Tensor([1., 2.], dt.float16)
    marker = object()
    calls = []
    def run_op(op, result, *args):
        calls.append((op, result, args))
        return marker
    jit = pytypes.SimpleNamespace(compiled=compiled, default_output=lambda: marker, run_op=run_op)
    monkeypatch.setattr(manual, 'jit_context', lambda: jit)
    dispatched = []
    def native(op, args):
        dispatched.append((op, args))
        # A native asynchronous result exposes this two-member protocol.
        return pytypes.SimpleNamespace(stub=object(), stub_sync=lambda: native_result)
    monkeypatch.setattr(manual, 'pyboost_cast' if name == 'Cast' else 'pyboost_tile', native)
    op = getattr(manual, name)()
    argument = dt.float16 if name == 'Cast' else (2,)
    assert op(x, argument) is marker
    if compiled:
        assert not calls and not dispatched
    else:
        assert len(dispatched) == 1
        assert dispatched[0][0] is op
        assert dispatched[0][1][0] is x
        assert len(calls) == 1
        assert calls[0][0] is op
        assert calls[0][1] is native_result
        assert calls[0][2][0] is x
        assert calls[0][2][1] == argument


def test_custom_file_locker_windows_system_errors(tmp_path, monkeypatch):
    """
    Feature: Custom file locker windows system errors.
    Description: Check custom file locker windows system errors.
    Expectation: Windows locking errors propagate with the expected diagnostics and clear the stored descriptor
        during cleanup.
    """
    calls = []
    def lock_failure(fd, mode, count):
        calls.append((fd, mode, count))
        raise OSError(errno.EIO, 'lock I/O error')
    monkeypatch.setattr(custom.platform, 'system', lambda: 'Windows')
    monkeypatch.setattr(custom, 'msvcrt', pytypes.SimpleNamespace(locking=lock_failure, LK_NBLCK=1, LK_UNLCK=2))
    locker = custom.FileLocker(str(tmp_path))
    with pytest.raises(OSError, match='lock I/O error'):
        locker.try_lock()
    assert locker.lock_fd is None
    fd = calls[0][0]
    with pytest.raises(OSError, match='Bad file descriptor'):
        os.fstat(fd)
    # Release tolerates the native unlock failure, and still closes its fd.
    locker.lock_fd = os.open(str(tmp_path / 'build.lock'), os.O_RDWR)
    locker.release_lock()
    assert locker.lock_fd is None
    assert calls[-1][1:] == (2, 1)


def test_custom_file_locker_close_failure(tmp_path, monkeypatch):
    """
    Feature: Custom file locker close failure.
    Description: Check custom file locker close failure.
    Expectation: A close failure clears the stored descriptor, and a later release is a harmless no-op.
    """
    locker = custom.FileLocker(str(tmp_path))
    assert locker.try_lock()
    target_fd = locker.lock_fd
    real_close = os.close
    def close_and_report_error(fd):
        real_close(fd)
        if fd == target_fd:
            raise OSError(errno.EIO, 'close diagnostic')
    with monkeypatch.context() as patch:
        patch.setattr(custom.os, 'close', close_and_report_error)
        assert locker.release_lock() is None
    assert locker.lock_fd is None


def test_maxpool_argmax_subclass_preserves_data_format():
    """
    Feature: Maxpool argmax subclass preserves data format.
    Description: Check maxpool argmax subclass preserves data format.
    Expectation: A derived argmax primitive retains NCHW metadata, while unsupported NHWC device use raises
        ValueError.
    """
    class SpecializedArgmax(grad.MaxPoolGradWithArgmax):
        """Inherited constructors must keep normal primitive naming semantics."""
    assert SpecializedArgmax().attrs['data_format'] == 'NCHW'
    with pytest.raises(ValueError, match='GPU'):
        grad.MaxPoolGradWithArgmax(data_format='NHWC')


@pytest.mark.xfail(strict=True, reason='MaxPoolGradV1 calls len() on a valid integer kernel_size')
def test_maxpoolgrad_v1_integer_kernel_normalization():
    """
    Feature: Maxpoolgrad v1 integer kernel normalization.
    Description: Check maxpoolgrad v1 integer kernel normalization.
    Expectation: Integer kernel and stride values normalize to the explicit (1, 1, 2, 2) tuples.
    """
    op = grad.MaxPoolGradV1(kernel_size=2, strides=2)
    assert op.attrs['kernel_size'] == (1, 1, 2, 2)
    assert op.attrs['strides'] == (1, 1, 2, 2)


@pytest.mark.xfail(strict=True, reason='VirtualConverterBegin returns ValueError instead of raising it')
def test_virtual_converter_rejects_zero_output_count():
    """
    Feature: Virtual converter rejects zero output count.
    Description: Check virtual converter rejects zero output count.
    Expectation: Invalid virtual converter zero output count raise ValueError. Diagnostics identify output_nums
        can't be zero.
    """
    with pytest.raises(ValueError, match="output_nums can't be zero"):
        inner._VirtualConverterBegin(0).infer_shape([2, 3])


@pytest.mark.xfail(strict=True, reason='Scan validates xs against None rather than type(None)')
def test_scan_supports_none_inputs_with_explicit_length():
    """
    Feature: Scan supports none inputs with explicit length.
    Description: Check scan supports none inputs with explicit length.
    Expectation: A scan of length three with no input sequence returns carry five and collected values [2, 3,
        4].
    """
    assert manual.Scan()(lambda carry, x: (carry + 1, carry), 2, None, length=3) == (5, [2, 3, 4])


@pytest.fixture(autouse=True, name='isolated_pyfunc_registry')
def fixture_isolated_pyfunc_registry(monkeypatch):
    """Preserve actual registration behavior without leaking callbacks across tests."""
    # Import at the test boundary so fixture isolation is established first.
    # pylint: disable-next=import-outside-toplevel
    from mindspore.ops.operations._pyfunc_registry import CustomPyFuncRegistry
    monkeypatch.setattr(CustomPyFuncRegistry, '_instance', CustomPyFuncRegistry(), raising=False)
