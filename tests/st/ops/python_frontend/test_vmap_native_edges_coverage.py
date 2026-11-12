# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Native CPU oracles for previously unverified vmap normalization/deformation edges."""
import importlib
import os
import random

import numpy as np
import pytest
import mindspore as ms
from mindspore import dataset
from mindspore import log as ms_log
from mindspore.common.generator import default_generator
from mindspore._c_expression import FormatEnum, ReductionEnum
from mindspore.ops.primitive import Primitive
from mindspore.ops.operations import nn_ops as NN, _grad_ops as G
from mindspore.ops._vmap import vmap_base as B, vmap_nn_ops as N, vmap_grad_nn_ops as D
from tests.mark_utils import arg_mark


GRAPH_PREPROCESS = B.vmap_general_preprocess


def tensor(value):
    return ms.Tensor(np.asarray(value, np.float32))


def check(value, expected, atol=2e-5, rtol=2e-5):
    expected = np.asarray(expected)
    assert isinstance(value, ms.Tensor) and value.dtype == ms.float32
    assert value.shape == expected.shape
    np.testing.assert_allclose(value.asnumpy(), expected, atol=atol, rtol=rtol)


def eager_preprocess(prim, *arguments):
    """Bridge only graph argument metadata; execute the original native primitive."""
    if not callable(prim):
        raise TypeError('preprocess requires the real callable primitive')
    assert all(isinstance(value, tuple) and len(value) == 2 for value in arguments)
    if all(axis is None for _, axis in arguments):
        return True, B.vmap_bind_all_none(prim(*(value for value, _ in arguments)))
    return False, None


@pytest.fixture(name='isolated_cpu', autouse=True)
def fixture_isolated_cpu(monkeypatch):
    # Snapshot private RNG and logging state to fully isolate native test execution.
    # pylint: disable=protected-access
    context = {key: ms.get_context(key) for key in ('mode', 'device_target', 'pynative_synchronize')}
    python_state, numpy_state = random.getstate(), np.random.get_state()
    dataset_seed, generator_state, global_seed = dataset.config.get_seed(), ms.get_rng_state(), ms.get_seed()
    manual = default_generator._generator.attrs['manual_seed']
    seed_module = importlib.import_module('mindspore.common.seed')
    op_seeds = seed_module._KERNEL_SEED.copy()
    monkeypatch.setattr(ms_log._LogActionOnce, 'is_logged', ms_log._LogActionOnce.is_logged.copy())
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU', pynative_synchronize=True)
    for module in (N, D):
        monkeypatch.setattr(module, 'vmap_general_preprocess', eager_preprocess)
    yield
    ms.set_rng_state(generator_state)
    default_generator._generator.add_prim_attr('manual_seed', manual)
    dataset.config.set_seed(dataset_seed)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    seed_module._GLOBAL_SEED = global_seed
    seed_module._KERNEL_SEED.clear()
    seed_module._KERNEL_SEED.update(op_seeds)
    ms.set_context(**context)


def batchnorm_data(training, layout):
    x = np.array([[[[1., 2.]], [[-1., 3.]], [[2., 5.]]],
                  [[[3., 7.]], [[2., 8.]], [[-2., 4.]]]], np.float64)
    dy = np.arange(1, 13, dtype=np.float64).reshape(x.shape)/5-1
    gamma = np.array([.8, 1.2, -.6], np.float64)
    if layout == FormatEnum.NHWC:
        x, dy = x.transpose(0, 2, 3, 1), dy.transpose(0, 2, 3, 1)
    axis = 1 if layout == FormatEnum.NCHW else 3
    reduce_axes = tuple(i for i in range(4) if i != axis)
    if training:
        mean, variance = x.mean(axis=reduce_axes), x.var(axis=reduce_axes)
    else:
        mean, variance = np.array([.2, -.3, .5]), np.array([1.1, 2.3, .7])
    return x, dy, gamma, mean, variance, axis, reduce_axes


def numpy_bn_gradient(x, dy, gamma, mean, variance, axis, training, epsilon):
    reduce_axes = tuple(i for i in range(x.ndim) if i != axis)
    shape = [1]*x.ndim
    shape[axis] = x.shape[axis]
    inverse = 1/np.sqrt(variance.reshape(shape)+epsilon)
    normalized = (x-mean.reshape(shape))*inverse
    dgamma = (dy*normalized).sum(axis=reduce_axes)
    dbeta = dy.sum(axis=reduce_axes)
    if training:
        dx = (dy-dy.mean(axis=reduce_axes, keepdims=True)
              -normalized*(dy*normalized).mean(axis=reduce_axes, keepdims=True))*gamma.reshape(shape)*inverse
    else:
        dx = dy*gamma.reshape(shape)*inverse
    return dx, dgamma, dbeta


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('layout', [FormatEnum.NCHW,
    pytest.param(FormatEnum.NHWC, marks=pytest.mark.xfail(strict=True, raises=RuntimeError,
        reason='CPU oneDNN BatchNormGrad NHWC fails its channel-buffer memcpy'))])
def test_unmapped_batchnorm_gradient_native_cpu(training, layout):
    """
    Feature: Native normalization and deformation vmap edges
    Description: Check unmapped batchnorm gradient native CPU.
    Expectation: All three gradients retain None axes and match NumPy; strict NHWC failure remains limited to the
        known native memcpy defect.
    """
    x, dy, gamma, mean, variance, axis, _ = batchnorm_data(training, layout)
    args = [tensor(dy), tensor(x), tensor(gamma), tensor(mean), tensor(variance), tensor([0.]),
            training, .25, int(layout)]
    prim = Primitive('BatchNormGrad')
    try:
        result = D.get_batchnorm_grad_vmap_rule(prim, 2)(*((arg, None) for arg in args))
    except RuntimeError as exc:
        if layout == FormatEnum.NHWC and 'Memcpy_s error.' in str(exc):
            raise
        raise AssertionError('unexpected BatchNormGrad native failure: '+str(exc)) from exc
    expected = numpy_bn_gradient(x, dy, gamma, mean, variance, axis, training, .25)
    assert len(result) == 3 and all(dim is None for _, dim in result)
    for (actual, _), oracle in zip(result, expected):
        check(actual, oracle)


def finite_gradient(function, values, step=1e-4):
    result = np.empty_like(values, np.float64)
    for position in np.ndindex(values.shape):
        plus, minus = values.copy(), values.copy()
        plus[position] += step
        minus[position] -= step
        result[position] = (function(plus)-function(minus))/(2*step)
    return result


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('layout', [FormatEnum.NCHW, FormatEnum.NHWC])
def test_unmapped_batchnorm_second_gradient_independent_finite_difference(training, layout):
    """
    Feature: Native normalization and deformation vmap edges
    Description: Check unmapped batchnorm second gradient independent finite difference.
    Expectation: Three unmapped second derivatives match independent finite differences in both training modes and
        layouts.
    """
    x, dy, gamma, mean, variance, axis, reduce_axes = batchnorm_data(training, layout)
    u = np.arange(x.size, dtype=np.float64).reshape(x.shape)/7-.4
    v, w = np.array([.3, -.5, .7]), np.array([-.2, .8, .4])
    def objective(input_x, input_dy, input_gamma):
        current_mean = input_x.mean(axis=reduce_axes) if training else mean
        current_variance = input_x.var(axis=reduce_axes) if training else variance
        first = numpy_bn_gradient(input_x, input_dy, input_gamma, current_mean, current_variance,
                                  axis, training, .25)
        return sum(np.sum(derivative*sensitivity) for derivative, sensitivity in zip(first, (u, v, w)))
    expected = [finite_gradient(lambda current: objective(current, dy, gamma), x),
                finite_gradient(lambda current: objective(x, current, gamma), dy),
                finite_gradient(lambda current: objective(x, dy, current), gamma)]
    args = [tensor(x), tensor(dy), tensor(gamma), tensor(mean), tensor(variance), tensor(u), tensor(v), tensor(w),
            training, .25, int(layout)]
    result = D.get_batchnorm_grad_grad_vmap_rule(Primitive('BatchNormGradGrad'), 2)(*((arg, None) for arg in args))
    assert len(result) == 3 and all(dim is None for _, dim in result)
    for (actual, _), oracle in zip(result, expected):
        check(actual, oracle, atol=8e-5, rtol=8e-5)


def bilinear_deformation(x, offsets):
    # The independent oracle explicitly enumerates batch, pixels and four interpolation neighbors.
    # pylint: disable=too-many-nested-blocks
    """Independent 1x1 sampling with x/y offsets, mask, and zero outside the image."""
    batch, channels, height, width = x.shape
    oh, ow = offsets.shape[-2:]
    output = np.zeros((batch, channels, oh, ow), np.float64)
    for n in range(batch):
        for i in range(oh):
            for j in range(ow):
                row, col = i+offsets[n, 1, i, j], j+offsets[n, 0, i, j]
                lower_row, lower_col = int(np.floor(row)), int(np.floor(col))
                mask = offsets[n, 2, i, j]
                for r in (lower_row, lower_row+1):
                    for c in (lower_col, lower_col+1):
                        if 0 <= r < height and 0 <= c < width:
                            weight = (1-abs(row-r))*(1-abs(col-c))*mask
                            output[n, :, i, j] += x[n, :, r, c]*weight
    return output


def deformable_inputs():
    x = (np.arange(18, dtype=np.float64).reshape(1, 2, 3, 3)/4)**2-1
    offsets = np.array([[[[.2, .35, .4], [.25, .3, .2], [.2, .35, .25]],
                         [[.3, .2, .25], [.4, .2, .3], [.2, .25, .4]],
                         [[.7, 1.1, .5], [1.3, .8, .9], [1.2, .6, 1.]]]], np.float64)
    return x, offsets


def deformation_primitive(gradient=False):
    kwargs = {'strides': (1, 1, 1, 1), 'pads': (0, 0, 0, 0), 'dilations': (1, 1, 1, 1),
              'data_format': 'NCHW', 'deformable_groups': 1, 'modulated': True}
    return G.DeformableOffsetsGrad(kernel_size=(
        1, 1), **kwargs) if gradient else NN.DeformableOffsets(ksize=(1, 1), **kwargs)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_unmapped_deformable_offsets_real_bilinear_cpu(monkeypatch):
    """
    Feature: Native normalization and deformation vmap edges
    Description: Check unmapped deformable offsets real bilinear CPU.
    Expectation: One unchanged native call preserves operand identity/attributes and matches independent masked
        bilinear sampling.
    """
    x, offsets = deformable_inputs()
    prim = deformation_primitive()
    operands = [tensor(x), tensor(offsets)]
    original, calls, attrs = type(prim).__call__, [], prim.attrs.copy()
    def observe(actual_prim, *args):
        assert actual_prim is prim and actual_prim.attrs == attrs
        assert len(args) == 2 and all(actual is expected for actual, expected in zip(args, operands))
        calls.append(args)
        return original(actual_prim, *args)
    monkeypatch.setattr(type(prim), '__call__', observe)
    value, axis = N.get_matmul_vmap_rule(prim, 2)(*((arg, None) for arg in operands))
    assert axis is None and len(calls) == 1
    check(value, bilinear_deformation(x, offsets))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_unmapped_deformable_offsets_gradient_finite_difference_cpu(monkeypatch):
    """
    Feature: Native normalization and deformation vmap edges
    Description: Check unmapped deformable offsets gradient finite difference CPU.
    Expectation: Input and offset derivatives retain None axes and match finite differences with nonuniform
        sensitivity.
    """
    x, offsets = deformable_inputs()
    sensitivity = np.arange(18, dtype=np.float64).reshape(1, 2, 3, 3)/7-1.1
    def objective_x(current):
        return np.sum(bilinear_deformation(current, offsets)*sensitivity)
    def objective_offsets(current):
        return np.sum(bilinear_deformation(x, current)*sensitivity)
    expected = (finite_gradient(objective_x, x), finite_gradient(objective_offsets, offsets))
    prim = deformation_primitive(gradient=True)
    operands = [tensor(sensitivity), tensor(x), tensor(offsets)]
    original, calls, attrs = type(prim).__call__, [], prim.attrs.copy()
    def observe(actual_prim, *args):
        assert actual_prim is prim and actual_prim.attrs == attrs
        assert len(args) == 3 and all(actual is expected for actual, expected in zip(args, operands))
        calls.append(args)
        return original(actual_prim, *args)
    monkeypatch.setattr(type(prim), '__call__', observe)
    result = D.get_deformable_offsets_vmap_rule(prim, 2)(*((arg, None) for arg in operands))
    assert len(calls) == 1 and len(result) == 2
    for (actual, axis), oracle in zip(result, expected):
        assert axis is None
        check(actual, oracle, atol=8e-5, rtol=8e-5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('reduction', [
    pytest.param(ReductionEnum.NONE, marks=pytest.mark.xfail(strict=True, raises=AssertionError,
        reason='Unmapped SmoothL1LossGrad swaps prediction/target/dout before native dispatch')),
    pytest.param(ReductionEnum.SUM, marks=pytest.mark.xfail(strict=True, raises=ValueError,
        reason='Unmapped SmoothL1LossGrad sends scalar dout as prediction')),
    pytest.param(ReductionEnum.MEAN, marks=pytest.mark.xfail(strict=True, raises=ValueError,
        reason='Unmapped SmoothL1LossGrad sends scalar dout as prediction')),
])
def test_unmapped_smooth_l1_gradient_nonuniform_sensitivity(reduction):
    """
    Feature: Native normalization and deformation vmap edges
    Description: Check unmapped smooth l1 gradient nonuniform sensitivity.
    Expectation: The piecewise gradient should respect sensitivity and reduction; strict xfails retain the known
        unmapped argument-order defects.
    """
    prediction, target = np.array([-2., .2, 3.]), np.array([0., 0., 1.])
    sensitivity = np.array([2., 3., 4.]) if reduction == ReductionEnum.NONE else np.array(2.)
    delta = prediction-target
    expected = np.where(np.abs(delta) < 1, delta, np.sign(delta))*sensitivity
    if reduction == ReductionEnum.MEAN:
        expected /= prediction.size
    args = [tensor(prediction), tensor(target), tensor(sensitivity), 1., int(reduction)]
    value, axis = N.get_smooth_l1_loss_grad_vmap_rule(Primitive('SmoothL1LossGrad'), 2)(*((arg, None) for arg in args))
    assert axis is None
    check(value, expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('reduction', [ReductionEnum.NONE, ReductionEnum.SUM, ReductionEnum.MEAN])
def test_smooth_l1_direct_native_cpu_confirms_oracle_and_argument_order(reduction):
    """
    Feature: Native normalization and deformation vmap edges
    Description: Check smooth l1 direct native CPU confirms oracle and argument order.
    Expectation: Direct native SmoothL1 gradients match the same piecewise oracle for none, sum and mean reductions.
    """
    prediction, target = np.array([-2., .2, 3.]), np.array([0., 0., 1.])
    sensitivity = np.array([2., 3., 4.]) if reduction == ReductionEnum.NONE else np.array(2.)
    expected = np.where(np.abs(prediction-target) < 1, prediction-target, np.sign(prediction-target))*sensitivity
    if reduction == ReductionEnum.MEAN:
        expected /= prediction.size
    actual = Primitive('SmoothL1LossGrad')(tensor(prediction), tensor(target), tensor(sensitivity), 1., int(reduction))
    check(actual, expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('gradient', [False, True])
def test_original_lrn_primitive_rejects_rank_three_samples(gradient):
    """
    Feature: Native normalization and deformation vmap edges
    Description: The residual rank<=4 mapped path requires samples outside LRN's 4D native contract.
    Expectation: Forward and gradient primitives reject rank-three samples with the native rank-four diagnostic.
    """
    x = tensor(np.arange(12).reshape(3, 2, 2))
    prim = G.LRNGrad() if gradient else NN.LRN()
    with pytest.raises(ValueError, match='rank.*equal to 4.*got 3'):
        if gradient:
            prim(x, x, x)
        else:
            prim(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=KeyError,
                   reason='Real graph _VmapGeneralRule construction loses __input_0__ binding')
def test_batchnorm_nhwc_real_graph_general_rule(monkeypatch):
    """
    Feature: Native normalization and deformation vmap edges
    Description: Attempt actual graph lowering of the native metagraph, without a replacement object.
    Expectation: Graph gradients should match per-sample NumPy values; strict xfail verifies the lost __input_0__
        binding.
    """
    monkeypatch.setattr(D, 'vmap_general_preprocess', GRAPH_PREPROCESS)
    first = batchnorm_data(False, FormatEnum.NHWC)
    x, dy, gamma, mean, variance, axis, _ = first
    x = np.stack((x, x+.7))
    dy = np.stack((dy, dy*1.3))
    gamma = np.stack((gamma, gamma*.8))
    mean = np.stack((mean, mean-.2))
    variance = np.stack((variance, variance+.4))
    prim = G.BatchNormGrad(is_training=False, epsilon=.25, data_format='NHWC')
    rule = D.get_batchnorm_grad_vmap_rule(prim, 2)
    nhwc = int(FormatEnum.NHWC)
    class GeneralNHWC(ms.nn.Cell):
        def construct(self, dout, values, scale, saved_mean, saved_variance, reserve):
            return rule((dout, 0), (values, 0), (scale, 0), (saved_mean, 0), (saved_variance, 0),
                        (reserve, 0), (False, None), (.25, None), (nhwc, None))
    ms.set_context(mode=ms.GRAPH_MODE)
    try:
        result = GeneralNHWC()(tensor(dy), tensor(x), tensor(gamma), tensor(mean),
                               tensor(variance), tensor(np.zeros((2, 1))))
    except KeyError as exc:
        assert exc.args == ('__input_0__',)
        raise
    expected = [numpy_bn_gradient(x[i], dy[i], gamma[i], mean[i], variance[i], axis, False, .25) for i in range(2)]
    assert len(result) == 3
    for index, (actual, dim) in enumerate(result):
        assert dim == 0
        check(actual, np.stack([item[index] for item in expected]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_batchnorm_nhwc_general_executor_complete_protocol(monkeypatch):
    """
    Feature: Native normalization and deformation vmap edges
    Description: Protocol only: construct the real metagraph; replace its unavailable eager execution boundary.
    Expectation: The real metagraph is constructed once with preserved attributes; its executor receives six
        original packed tensors once.
    """
    x, dy, gamma, mean, variance, _, _ = batchnorm_data(False, FormatEnum.NHWC)
    arrays = [np.stack((dy, dy*1.3)), np.stack((x, x+.7)), np.stack((gamma, gamma*.8)),
              np.stack((mean, mean-.2)), np.stack((variance, variance+.4)), np.zeros((2, 1))]
    operands = [tensor(array) for array in arrays]
    packed = tuple((operand, 0) for operand in operands)
    prim = G.BatchNormGrad(is_training=False, epsilon=.25, data_format='NHWC')
    attrs = prim.attrs.copy()
    # Inspect the actual native metagraph constructor used by this fallback.
    native_factory = D._VmapGeneralRule  # pylint: disable=protected-access
    constructions, executions = [], []
    outputs = tuple((object(), 0) for _ in range(3))
    def constructor(actual_prim, axis_size):
        assert actual_prim is prim and actual_prim.attrs == attrs and axis_size == 2
        assert actual_prim.is_training is False and actual_prim.epsilon == .25
        assert actual_prim.data_format == int(FormatEnum.NHWC)
        real_metagraph = native_factory(actual_prim, axis_size)
        assert isinstance(real_metagraph, native_factory)
        assert not callable(real_metagraph) and callable(real_metagraph.set_signatures)
        constructions.append(real_metagraph)
        def execute(*received):
            assert constructions == [real_metagraph] and len(received) == 6
            for actual, original, expected in zip(received, packed, arrays):
                assert actual is original and actual[1] == 0
                check(actual[0], expected)
            executions.append(received)
            return outputs
        return execute
    monkeypatch.setattr(D, '_VmapGeneralRule', constructor)
    result = D.get_batchnorm_grad_vmap_rule(prim, 2)(*packed, (False, None), (.25, None),
                                                    (int(FormatEnum.NHWC), None))
    assert result is outputs and len(constructions) == len(executions) == 1
    assert all(result[index][1] == 0 for index in range(3))


@pytest.fixture
def ascend_context(isolated_cpu):
    """Only the explicit fresh launcher may opt into the allocated device."""
    expected_device = os.environ.get('MS_VMAP_NATIVE_DEVICE')
    if expected_device is not None:
        assert ms.get_context('device_id') == int(expected_device)
    previous = {key: ms.get_context(key) for key in ('mode', 'device_target')}
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='Ascend')
    yield
    ms.set_context(**previous)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level2', card_mark='onecard')
@pytest.mark.skipif(os.environ.get('MS_VMAP_NATIVE_TARGET') != 'Ascend', reason='Explicit Ascend allocation required')
@pytest.mark.usefixtures('ascend_context')
@pytest.mark.parametrize('training', [
    pytest.param(False, marks=pytest.mark.xfail(strict=True, raises=AssertionError,
        reason='Ascend NHWC inference BatchNormGrad uses the first channel statistics for other channels')),
    True,
])
def test_unmapped_batchnorm_gradient_native_ascend(training):
    """
    Feature: Native normalization and deformation vmap edges
    Description: Check unmapped batchnorm gradient native ascend.
    Expectation: Allocated Ascend execution retains None axes and matches NumPy; inference keeps its strict known
        channel-statistics failure.
    """
    x, dy, gamma, mean, variance, axis, _ = batchnorm_data(training, FormatEnum.NHWC)
    args = [tensor(dy), tensor(x), tensor(gamma), tensor(mean), tensor(variance), tensor([0.]),
            training, .25, int(FormatEnum.NHWC)]
    result = D.get_batchnorm_grad_vmap_rule(Primitive('BatchNormGrad'), 2)(*((arg, None) for arg in args))
    expected = numpy_bn_gradient(x, dy, gamma, mean, variance, axis, training, .25)
    assert len(result) == 3 and all(dim is None for _, dim in result)
    for (actual, _), oracle in zip(result, expected):
        check(actual, oracle, atol=5e-5, rtol=5e-5)
