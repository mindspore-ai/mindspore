# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Remaining Python bprop contracts: real CPU arithmetic and narrow native ABI spies.

A native spy only returns a labelled cotangent at an unavailable device kernel
boundary. It checks the actual primitive, attributes, ordered arguments and call
count. Such cases validate Python routing, not the native kernel's numerics.
"""
# Direct private-helper calls intentionally cover the internal rules under test.
# pylint: disable=protected-access
from types import SimpleNamespace

import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, CSRTensor, COOTensor, context
from mindspore.ops import operations as P
from mindspore.ops.primitive import Primitive
from mindspore.ops.operations import _quant_ops as Q
from mindspore.ops.operations import array_ops as A, math_ops as M, sparse_ops as S, nn_ops as N
from mindspore.ops._grad_experimental import grad_quant_ops as gq, grad_base as gb
from mindspore.ops._grad_experimental import grad_sparse as gc, grad_sparse_ops as gs
from mindspore.ops._grad_experimental import grad_array_ops as ga, grad_math_ops as gm
from mindspore.ops._grad_experimental import grad_nn_ops as gn, grad_inner_ops as ge
from mindspore.ops._grad_experimental import grad_debug_ops as gd, grad_implementations as gi
from tests.mark_utils import arg_mark


@pytest.fixture(autouse=True)
def cpu_context():
    previous = {key: context.get_context(key) for key in ('mode', 'device_target', 'pynative_synchronize')}
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU', pynative_synchronize=True)
    yield
    context.set_context(**previous)


def t(value, dtype=ms.float32):
    return Tensor(value, dtype)


def check(actual, expected, dtype=ms.float32, atol=2e-5):
    assert isinstance(actual, Tensor)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual.asnumpy(), expected, rtol=2e-5, atol=atol)


def equal(actual, expected):
    if isinstance(expected, Tensor):
        check(actual, expected.asnumpy(), expected.dtype)
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            equal(left, right)
    else:
        assert actual == expected


@pytest.fixture(name='native_boundary')
def native_boundary_fixture(monkeypatch):
    """Keep constructors and every other CPU primitive real; intercept named native kernels only."""
    installed = []

    def install(specifications):
        assert not installed, 'Use one ordered native call plan per test'
        pending = list(specifications)
        installed.append(pending)
        intercepted = {entry[0] for entry in pending}
        original = Primitive.__call__

        def invoke(primitive, *args):
            if primitive.name not in intercepted:
                return original(primitive, *args)
            assert pending, 'Unexpected additional native call'
            name, expected_args, result, attributes = pending.pop(0)
            assert primitive.name == name
            equal(args, expected_args)
            for key, value in attributes.items():
                assert primitive.attrs[key] == value, (name, key, primitive.attrs)
            return result

        monkeypatch.setattr(Primitive, '__call__', invoke)
        if 'TensorDump' in intercepted:
            monkeypatch.setattr(P.TensorDump, '__call__', invoke)
        return pending

    yield install
    for pending in installed:
        assert not pending, "Expected native calls were not consumed"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind', ['layer', 'vars', 'vars_channel', 'channel', 'learned', 'learned_channel'])
def test_quantization_native_gradient_abi(native_boundary, kind):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Quantization native gradient ABI.
    Expectation: Each native gradient receives the ordered tensors and independent quantization attributes, and
        nondifferentiable bounds receive zeros.
    """
    x, low, high, dout = t([[-2.0, 1.0], [2.0, 3.0]]), t([0.0, 0.0]), t([4.0, 4.0]), t([[2.0, 3.0], [5.0, 7.0]])
    dx, da = t([[11.0, 13.0], [17.0, 19.0]]), t([23.0, 29.0])
    if kind == 'layer':
        primitive = Q.FakeQuantPerLayer(num_bits=4, quant_delay=7)
        getter, name = gq.get_bprop_fakequant_with_minmax, 'FakeQuantPerLayerGrad'
        attributes = {'num_bits': 4, 'quant_delay': 7}
    elif kind == 'vars':
        primitive = Q.FakeQuantWithMinMaxVars(num_bits=4, narrow_range=True)
        getter, name = gq.get_bprop_fakequant_with_minmax_vars, 'FakeQuantWithMinMaxVarsGradient'
        attributes = {'num_bits': 4, 'narrow_range': True}
    elif kind == 'vars_channel':
        primitive = Q.FakeQuantWithMinMaxVarsPerChannel(num_bits=4, narrow_range=True)
        getter, name = gq.get_bprop_fakequant_with_minmax_vars_perchannel, 'FakeQuantWithMinMaxVarsPerChannelGradient'
        attributes = {'num_bits': 4, 'narrow_range': True}
    elif kind == 'channel':
        primitive = Q.FakeQuantPerChannel(num_bits=4, quant_delay=7, symmetric=True, narrow_range=True, channel_axis=0)
        getter, name = gq.get_bprop_fakequant_with_minmax_perchannel, 'FakeQuantPerChannelGrad'
        attributes = {'num_bits': 4, 'quant_delay': 7, 'symmetric': True, 'narrow_range': True, 'channel_axis': 0}
    else:
        channel = kind == 'learned_channel'
        primitive = (
            Q.FakeLearnedScaleQuantPerChannel(quant_delay=7, neg_trunc=True, channel_axis=0)
            if channel
            else Q.FakeLearnedScaleQuantPerLayer(quant_delay=7, neg_trunc=True)
        )
        getter = (
            gq.get_bprop_fakequant_with_learned_scale_perchannel
            if channel
            else gq.get_bprop_fakequant_with_learned_scale_perlayer
        )
        name = 'FakeLearnedScaleQuantPerChannelGrad' if channel else 'FakeLearnedScaleQuantPerLayerGrad'
        attributes = {'quant_delay': 7, 'neg_trunc': True}
        if channel:
            attributes['channel_axis'] = 0
    learned = kind.startswith('learned')
    if kind in ('layer', 'vars'):
        low, high = t([0.0]), t([4.0])
    elif learned:
        low = t([2.0, 3.0]) if kind == 'learned_channel' else t([2.0])
        high = t([7.0])
        da = t(np.full(low.shape, 23.0))
    pending = native_boundary([(name, (dout, x, low, high), (dx, da) if learned else dx, attributes)])
    result = getter(primitive)(x, low, high, None, dout)
    assert result[0] is dx
    if learned:
        assert result[1] is da
    else:
        check(result[1], np.zeros(low.shape))
    check(result[2], np.zeros(high.shape))
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=AssertionError, reason='Per-channel rule copies symmetric into narrow_range')
def test_quantization_gradient_preserves_independent_narrow_range(native_boundary):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Quantization gradient preserves independent narrow range.
    Expectation: The native call retains both the primitive and negative-truncation narrow-range settings and
        returns the supplied sensitivity.
    """
    primitive = Q.FakeQuantPerChannel(symmetric=False, narrow_range=True, channel_axis=0)
    x, low, high, dout = t([[1.0]]), t([0.0]), t([2.0]), t([[3.0]])
    pending = native_boundary(
        [
            (
                'FakeQuantPerChannelGrad',
                (dout, x, low, high),
                dout,
                {'symmetric': False, 'narrow_range': True, 'channel_axis': 0},
            )
        ]
    )
    result = gq.get_bprop_fakequant_with_minmax_perchannel(primitive)(x, low, high, None, dout)
    assert result[0] is dout and not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind', ['fold', 'fold_d', 'fold2', 'fold2_d', 'correction', 'acts'])
def test_quantized_training_composite_native_protocol(native_boundary, kind):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Quantized training composite native protocol.
    Expectation: Every scheduled native gradient call consumes its expected ABI and Python assembly preserves
        supplied gradients and zero metadata.
    """
    x = t(np.arange(8).reshape(1, 2, 2, 2))
    mean, variance, step = t([1.0, 2.0]), t([3.0, 4.0]), t([5], ms.int32)
    out = tuple(t([float(i), float(i + 1)]) for i in range(1, 5))
    dout = tuple(t([float(i), float(i + 2)]) for i in range(6, 10))
    dx = t(np.full(x.shape, 31.0))
    full_dout = t(np.full(x.shape, 6.0))
    tokens = tuple(t([float(i), float(i + 1)]) for i in range(11, 17))
    if kind == 'fold':
        primitive = Q.BatchNormFold(epsilon=0.01, is_training=False, freeze_bn=9)
        pending = native_boundary(
            [
                (
                    'BatchNormFoldGrad',
                    (dout[0], dout[1], x, out[0], out[1], step),
                    dx,
                    {'epsilon': 0.01, 'is_training': False, 'freeze_bn': 9},
                )
            ]
        )
        got = gq.get_bprop_batchnorm_fold(primitive)(x, mean, variance, step, out, dout)
        assert got[0] is dx
        equal(got[1:], (t([0.0, 0.0]), t([0.0, 0.0]), t([0], ms.int32)))
    elif kind == 'fold_d':
        primitive = Q.BatchNormFoldD(epsilon=0.01, is_training=False, freeze_bn=9)
        pending = native_boundary(
            [
                (
                    'BatchNormFoldGradD',
                    (dout[1], dout[2], x, out[1], out[2]),
                    dx,
                    {'epsilon': 0.01, 'is_training': False, 'freeze_bn': 9},
                )
            ]
        )
        got = gq.get_bprop_batchnormfold(primitive)(x, mean, variance, mean, variance, out, dout)
        assert got[0] is dx
        for value in got[1:]:
            check(value, [0.0, 0.0])
    elif kind == 'fold2':
        pending = native_boundary(
            [
                (
                    'BatchNormFold2Grad',
                    (full_dout, x, tokens[1], tokens[2], tokens[3], mean, variance, step),
                    tokens[:4] + (dx,),
                    {'freeze_bn': 9},
                )
            ]
        )
        got = gq.get_bprop_batchnorm_fold2(Q.BatchNormFold2(freeze_bn=9))(
            x, tokens[0], tokens[1], tokens[2], tokens[3], mean, variance, step, None, full_dout
        )
        equal(got, (dx, tokens[2], tokens[3], tokens[0], tokens[1], t([0.0, 0.0]), t([0.0, 0.0]), t([0], ms.int32)))
    elif kind == 'fold2_d':
        pending = native_boundary(
            [
                ('BatchNormFold2GradReduce', (full_dout, x), tokens[:2], {'freeze_bn': 9}),
                (
                    'BatchNormFold2GradD',
                    (full_dout, tokens[0], tokens[1], tokens[3], tokens[4], tokens[5], mean),
                    (tokens[2], tokens[3], tokens[4], dx),
                    {'freeze_bn': 9},
                ),
            ]
        )
        got = gq.get_bprop_batchnorm_fold2_(Q.BatchNormFold2D(freeze_bn=9))(
            x, tokens[2], tokens[3], tokens[4], tokens[5], mean, None, full_dout
        )
        equal(got, (dx, tokens[0], tokens[4], tokens[2], tokens[3], t([0.0, 0.0])))
    elif kind == 'correction':
        pending = native_boundary(
            [('CorrectionMulGrad', (full_dout, x, mean, variance), (dx, tokens[0]), {'channel_axis': 1})]
        )
        got = gq.get_bprop_correction_mul(Q.CorrectionMul(channel_axis=1))(x, mean, variance, None, full_dout)
        equal(got, (dx, tokens[0], t([0.0, 0.0])))
    else:
        lower, upper = t(np.full((1, 1, 1, 1), -1.0)), t(np.full((1, 1, 1, 1), 9.0))
        min_mask, max_mask = t(np.ones(x.shape), ms.bool_), t(np.zeros(x.shape), ms.bool_)
        loss = t(np.full(x.shape, 13.0))
        dmin, dmax = t(np.full(lower.shape, 17.0)), t(np.full(upper.shape, 19.0))
        pending = native_boundary(
            [
                ('ActsULQInputGrad', (full_dout, min_mask, max_mask), dx, {}),
                ('ActULQClampMinGrad', (full_dout, min_mask, loss), dmin, {}),
                ('ActULQClampMaxGrad', (full_dout, max_mask, loss), dmax, {}),
            ]
        )
        got = gq.get_bprop_acts_ulq(Q.ActsULQ())(
            x, lower, upper, (x, min_mask, max_mask, loss), (full_dout, None, None, None)
        )
        equal(got, (dx, dmin, dmax))
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_correction_ascend_branch_uses_reduce_without_selecting_a_device(monkeypatch, native_boundary):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Correction ascend branch uses reduce without selecting a device.
    Expectation: The Ascend frontend branch consumes the expected reduction protocol while the runtime remains
        on CPU.
    """
    x, dout, batch, running = t(np.ones((1, 2, 2, 2))), t(np.full((1, 2, 2, 2), 3.0)), t([2.0, 4.0]), t([5.0, 7.0])
    dx, products, reduced = t(np.full(x.shape, 11.0)), t(np.full(x.shape, 13.0)), t([17.0, 19.0])
    pending = native_boundary(
        [
            ('CorrectionMulGrad', (dout, x, batch, running), (dx, products), {'channel_axis': 1}),
            ('CorrectionMulGradReduce', (products,), reduced, {'channel_axis': 1}),
        ]
    )
    original = gq.context.get_context
    monkeypatch.setattr(gq.context, 'get_context', lambda key: 'Ascend' if key == 'device_target' else original(key))
    closure = gq.get_bprop_correction_mul(Q.CorrectionMul(channel_axis=1))
    monkeypatch.setattr(gq.context, 'get_context', original)
    equal(closure(x, batch, running, None, dout), (dx, reduced, t([0.0, 0.0])))
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind', ['minmax', 'minmax_channel', 'reduce', 'weights'])
def test_quantization_nondifferentiable_inputs_and_straight_through(kind):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Quantization nondifferentiable inputs and straight through.
    Expectation: Min/max update inputs receive zero cotangents, while the weight approximation passes the
        upstream sensitivity through.
    """
    x, low, high, dout = t([[2.0, 3.0], [5.0, 7.0]]), t([1.0, 2.0]), t([4.0, 8.0]), t([[11.0, 13.0], [17.0, 19.0]])
    if kind == 'reduce':
        x = t(np.arange(8).reshape(1, 2, 2, 2))
        got = gq.get_bprop_bn_training_reduce(None)(x, None, None)
        equal(got, (t(np.zeros(x.shape)),))
    else:
        if kind == 'minmax':
            low, high = t([1.0]), t([4.0])
        elif kind == 'weights':
            low, high = t([[1.0]]), t([[4.0]])
        getter = {
            'minmax': gq.get_bprop_fakequant_with_minmax_per_layer_update,
            'minmax_channel': gq.get_bprop_fakequant_with_minmax_per_channel_update,
            'weights': gq.get_bprop_wts_arq,
        }[kind]
        got = getter(None)(x, low, high, None, dout)
        equal(
            got, (dout if kind == 'weights' else t(np.zeros(x.shape)), t(np.zeros(low.shape)), t(np.zeros(high.shape)))
        )


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_base_registry_fallbacks_do_not_mutate_registries():
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Base registry fallbacks do not mutate registries.
    Expectation: Unknown registry names return None, TensorAdd resolves its real closure, and existing tensors
        retain identity.
    """
    assert gb.get_bprop_fn('DefinitelyNotAGradientPrimitive') is None
    assert gb.get_taylor_fprop_fn('DefinitelyNotATaylorPrimitive') is None
    assert gb.get_bprop_fn('TensorAdd', get_closure=True) is gm.get_bprop_tensor_add
    value = t([1.0, 2.0])
    assert gb.create_tensor_by_element(value) is value


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('value', [[1, 2], (1, 2), 3, 4.0, 'unchanged'])
def test_base_convert_constant_values_keeps_identity(value):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Base convert constant values keeps identity.
    Expectation: Python constant inputs return a false conversion flag and the original object.
    """
    flag, actual = gb.convert_to_tensor(value)
    assert flag is False and actual is value


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize(
    'value,dtype,expected',
    [([1, 2], ms.int64, [1, 2]), ((3, 4), ms.int64, [3, 4]), (5, ms.int64, 5), (6.0, ms.float32, 6.0)],
)
def test_base_convert_dynamic_metadata_uses_real_cpu_conversion(monkeypatch, value, dtype, expected):
    # Pynative reports all Python values constant. Only frontend abstract metadata
    # is selected here; actual List/Tuple/ScalarToTensor kernels remain real.
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Base convert dynamic metadata uses real CPU conversion.
    Expectation: Dynamic metadata selects real CPU conversion with the expected int64 or float32 values and a
        true conversion flag.
    """
    monkeypatch.setattr(gb.F, 'is_sequence_value_unknown', lambda _: True)
    monkeypatch.setattr(gb.F, 'isconstant', lambda _: False)
    flag, actual = gb.convert_to_tensor(value)
    assert flag is True
    check(actual, expected, dtype)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_base_existing_tensor_remains_unchanged():
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Base existing tensor remains unchanged.
    Expectation: An existing Tensor returns a true conversion flag and retains object identity.
    """
    value = t([2.0, 3.0])
    flag, result = gb.convert_to_tensor(value)
    assert flag is True and result is value


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_debug_callback_none_preserves_cotangent_and_tensor_dump_zeros():
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Debug callback none preserves cotangent and tensor dump zeros.
    Expectation: A callback returning None preserves the sensitivity; tensor-dump backpropagation preserves the
        path and zeros the tensor.
    """
    calls = []

    def record(value):
        calls.append(value)

    closure = gd.get_bprop_insert_gradient_of(SimpleNamespace(f=record))
    d = t([2.0, 3.0])
    (result,) = closure(t([1.0, 1.0]), None, d)
    assert result is d and calls == [d]
    path, zero = gd.bprop_tensor_dump('unused_dump_path', d, None, d)
    assert path == 'unused_dump_path'
    check(zero, [0.0, 0.0])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_dump_gradient_native_side_effect_protocol(native_boundary):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Dump gradient native side effect protocol.
    Expectation: TensorDump receives the expected path, sensitivity and side-effect attributes, while only the
        tensor retains a cotangent.
    """
    x, dout = t([1.0, 2.0]), t([3.0, 5.0])
    pending = native_boundary(
        [('TensorDump', ('unused_dump_path', dout), None, {'side_effect_io': False, 'td_flag': True})]
    )
    got = gd.get_bprop_dump_gradient(P.DumpGradient())('unused_dump_path', x, 'out', None, dout)
    assert got[0] == 0 and got[1] is dout and got[2] == 0
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_inner_psroi_pooling_passes_spatial_shape_and_mapping(native_boundary):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Inner PSROI pooling passes spatial shape and mapping.
    Expectation: PsROIPoolingGrad receives the full spatial attributes and mapping tensor; ROI coordinates
        receive zeros.
    """
    x, rois = t(np.ones((2, 8, 4, 5))), t(np.ones((2, 5)))
    mapping, dout, dx = t(np.ones((2, 2, 2, 2)), ms.int32), t(np.ones((2, 2, 2, 2))), t(np.full(x.shape, 7.0))
    attrs = {
        'batch_size': 2,
        'channels': 8,
        'height': 4,
        'width': 5,
        'num_rois': 2,
        'pooled_height': 2,
        'pooled_width': 2,
        'spatial_scale': 0.5,
        'out_dim': 2,
    }
    pending = native_boundary([('PsROIPoolingGrad', (dout, rois, mapping), dx, attrs)])
    primitive = SimpleNamespace(pooled_height=2, pooled_width=2, spatial_scale=0.5, out_dim=2, num_rois=2)
    got = ge.get_bprop_ps_roi_pooling(primitive)(x, rois, (dout, mapping), (dout, None))
    assert got[0] is dx
    check(got[1], np.zeros(rois.shape))
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason='GenerateEodMask bprop returns two gradients for one input'
)
def test_eod_mask_gradient_has_one_cotangent_per_input():
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Eod mask gradient has one cotangent per input.
    Expectation: The intended single input receives one cotangent; the documented extra-gradient defect remains
        a strict expected failure.
    """
    value = t([[1, 0], [2, 3]], ms.int32)
    result = ge.get_bprop_generate_eod_mask(None)(value, value, value)
    assert len(result) == 1
    assert result[0] is value


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind', ['gru', 'lstm', 'wkv'])
def test_recurrent_gradient_native_order_and_real_postprocessing(native_boundary, kind):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Recurrent gradient native order and real postprocessing.
    Expectation: GRU and LSTM derivatives preserve native argument order, while WKV sums parameter gradients and
        zeros recurrent states.
    """
    if kind == 'gru':
        x, hx, weight = t(np.ones((2, 1, 3))), t(np.full((1, 1, 2), 2.0)), t(np.full((42, 1, 1), 3.0))
        y, reserve, state = t(np.full((2, 1, 2), 5.0)), t([7.0, 11.0]), t([13.0, 17.0])
        dy, dhy = t(np.full(y.shape, 19.0)), t(np.full(hx.shape, 23.0))
        dx, dhx, dw = t(np.full(x.shape, 29.0)), t(np.full(hx.shape, 31.0)), t(np.full(weight.shape, 37.0))
        attrs = {
            'input_size': 3,
            'hidden_size': 2,
            'num_layers': 1,
            'has_bias': True,
            'bidirectional': False,
            'dropout': 0.0,
        }
        pending = native_boundary(
            [
                ('GruGradData', (y, dy, dhy, weight, hx, reserve, state), (dx, dhx), attrs),
                ('GruGradWeight', (x, hx, y, reserve, state), dw, attrs),
            ]
        )
        got = gn.get_bprop_gru(gn.rl_ops.CudnnGRU(**attrs))(
            x, hx, weight, (y, hx, reserve, state), (dy, dhy, None, None)
        )
        equal(got, (dx, dhx, dw))
    elif kind == 'lstm':
        x, h, c, weight, bias = (
            t(np.full(shape, i + 1.0)) for i, shape in enumerate([(2, 3), (2, 2), (2, 2), (5, 8), (8,)])
        )
        gates = tuple(t(np.full((2, 2), i + 7.0)) for i in range(7))
        dout = tuple(t(np.full((2, 2), i + 17.0)) for i in range(7))
        dgate, dc = t(np.full((2, 8), 29.0)), t(np.full(c.shape, 31.0))
        dx, dh, dw, db = (t(np.full(v.shape, i + 37.0)) for i, v in enumerate((x, h, weight, bias)))
        pending = native_boundary(
            [
                (
                    'BasicLSTMCellCStateGrad',
                    (c, dout[1], dout[0], *gates[2:]),
                    (dgate, dc),
                    {'forget_bias': 1.0, 'activation': 'tanh'},
                ),
                ('BasicLSTMCellInputGrad', (dgate, weight), (dx, dh), {'keep_prob': 1.0}),
                ('BasicLSTMCellWeightGrad', (x, h, dgate), (dw, db), {}),
            ]
        )
        got = gn.get_bprop_basic_lstm_cell(P.BasicLSTMCell())(x, h, c, weight, bias, gates, dout)
        equal(got, (dx, dh, dc, dw, db))
    else:
        w, u, k, v = t([1.0, 2.0, 3.0]), t([4.0, 5.0, 6.0]), t(np.ones((2, 4, 3))), t(np.full((2, 4, 3), 7.0))
        states = tuple(t(np.full((2, 3), i + 11.0)) for i in range(3))
        dout = t(np.full(k.shape, 17.0))
        gw, gu, gk, gv = t([[1.0, 2.0, 3.0], [5.0, 7.0, 11.0]]), t([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]]), k, v
        pending = native_boundary([('WKVGrad', (w, u, k, v, dout), (gw, gu, gk, gv), {})])
        got = gn.get_bprop_wkv(N.WKV())(w, u, k, v, *states, None, (dout, None, None, None))
        equal(got, (t([6.0, 9.0, 14.0]), t([9.0, 14.0, 18.0]), gk, gv, *(t(np.zeros(s.shape)) for s in states)))
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind', ['avg', 'fractional'])
def test_pooling_gradient_native_shape_and_indices_contract(native_boundary, kind):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Pooling gradient native shape and indices contract.
    Expectation: Pooling gradients receive actual input-shape or maximum-index metadata and preserve the
        supplied input cotangent.
    """
    x = t(np.arange(16).reshape(1, 1, 4, 4))
    dout, dx = t(np.full((1, 1, 2, 2), 3.0)), t(np.full(x.shape, 5.0))
    if kind == 'avg':
        primitive = N.AvgPoolV1(kernel_size=2, strides=2, pad_mode='valid', data_format='NCHW')
        pending = native_boundary(
            [
                (
                    'AvgPoolGradV1',
                    (t([1, 1, 4, 4], ms.int32), dout),
                    dx,
                    {'kernel_size': (1, 1, 2, 2), 'strides': (1, 1, 2, 2), 'pad_mode': 'VALID', 'data_format': 'NCHW'},
                )
            ]
        )
        equal(gn.get_bprop_avg_pool_v1_grad(primitive)(x, None, dout), (dx,))
    else:
        random_samples, indices = t([[[0.2, 0.7]]]), t(np.zeros(dout.shape), ms.int64)
        pending = native_boundary(
            [('FractionalMaxPoolGradWithFixedKsize', (x, dout, indices), dx, {'data_format': 'NCHW'})]
        )
        primitive = N.FractionalMaxPoolWithFixedKsize(ksize=2, output_shape=(2, 2))
        got = gn.get_bprop_fractional_max_pool_with_fixed_ksize(primitive)(
            x, random_samples, (dout, indices), (dout, None)
        )
        equal(got, (dx, t(np.zeros(random_samples.shape))))
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('is_grad', [False, True])
def test_sparse_cross_entropy_rule_preserves_loss_scaling(native_boundary, is_grad):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Sparse cross entropy rule preserves loss scaling.
    Expectation: Loss mode multiplies the native logits gradient by seven; gradient mode retains it, and labels
        receive int32 zeros.
    """
    logits, labels, dout = t([[1.0, 2.0, 3.0], [2.0, 4.0, 1.0]]), t([1, 0], ms.int32), t(7.0)
    grad = t([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
    pending = native_boundary(
        [] if is_grad else [('SparseSoftmaxCrossEntropyWithLogits', (logits, labels), grad, {'is_grad': True})]
    )
    got = gn.get_bprop_sparse_softmax_cross_entropy_with_logits(P.SparseSoftmaxCrossEntropyWithLogits(is_grad=is_grad))(
        logits, labels, (grad,), dout
    )
    check(got[0], grad.asnumpy() if is_grad else 7 * grad.asnumpy())
    check(got[1], [0, 0], ms.int32)
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('divide', [False, True])
@pytest.mark.parametrize('column', [False, True])
def test_csr_elementwise_rule_native_values_and_real_dense_scatter(monkeypatch, native_boundary, divide, column):
    # Graph-only constructors are adapted to the actual eager sparse constructor.
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: CSR elementwise rule native values and real dense scatter.
    Expectation: Supplied native value cotangents and real dense scatter or reduction match the independent
        product and quotient derivatives.
    """
    monkeypatch.setattr(gc.F, 'make_csr_tensor', CSRTensor)
    ptr, indices, values, dout = (
        t([0, 2, 3], ms.int32),
        t([0, 2, 1], ms.int32),
        t([2.0, 3.0, 5.0]),
        t([7.0, 11.0, 13.0]),
    )
    dense_np = np.array([[2.0], [4.0]]) if column else np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    dense = t(dense_np)
    sampled = dense_np[[0, 0, 1], [0, 0, 0] if column else [0, 2, 1]]
    out = t(values.asnumpy() / sampled if divide else values.asnumpy() * sampled)
    native_result = t(dout.asnumpy() / sampled if divide else dout.asnumpy() * sampled)
    dense_values = -dout.asnumpy() * values.asnumpy() / sampled**2 if divide else dout.asnumpy() * values.asnumpy()
    expected = np.zeros_like(dense_np)
    np.add.at(expected, ([0, 0, 1], [0, 0, 0] if column else [0, 2, 1]), dense_values)
    specifications = [('CSRDiv' if divide else 'CSRMul', (ptr, indices, dout, (2, 3), dense), native_result, {})]
    if column:
        specifications.append(('CSRReduceSum', (ptr, indices, t(dense_values), (2, 3), 1), t(expected), {}))
    else:
        specifications.append(('CSR2COO', (ptr, 3), t([0, 0, 1], ms.int32), {}))
    pending = native_boundary(specifications)
    getter = gc.get_bprop_csr_div if divide else gc.get_bprop_csr_mul
    got = getter(None)(ptr, indices, values, (2, 3), dense, out, dout)
    # An empty tuple is part of the gradient return contract.
    # pylint: disable-next=use-implicit-booleaness-not-comparison
    assert got[0] is ptr and got[1] is indices and got[3] == ()
    check(got[2], native_result.asnumpy())
    check(got[4], expected)
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('divide', [False, True])
def test_csr_explicit_first_axis_broadcast_rejection(monkeypatch, native_boundary, divide):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: CSR explicit first axis broadcast rejection.
    Expectation: CSR derivatives reject unsupported first-axis broadcasting after the specified native
        value-gradient request.
    """
    monkeypatch.setattr(gc.F, 'make_csr_tensor', CSRTensor)
    ptr, idx, value, dense, dout = (
        t([0, 1, 2], ms.int32),
        t([0, 1], ms.int32),
        t([2.0, 3.0]),
        t([[2.0, 4.0]]),
        t([5.0, 7.0]),
    )
    native_result = t([2.5, 1.75] if divide else [10.0, 28.0])
    pending = native_boundary([('CSRDiv' if divide else 'CSRMul', (ptr, idx, dout, (2, 2), dense), native_result, {})])
    getter = gc.get_bprop_csr_div if divide else gc.get_bprop_csr_mul
    with pytest.raises(ValueError, match='broadcast for the first dimension is not supported'):
        getter(None)(ptr, idx, value, (2, 2), dense, t([1.0, 0.75] if divide else [4.0, 12.0]), dout)
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_csr_reduction_gradient_tiles_then_gathers_real_cpu_values(native_boundary):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: CSR reduction gradient tiles then gathers real CPU values.
    Expectation: Tiled row sensitivities reach CSRGather in the expected order and produce stored-value
        cotangents [7, 7, 11].
    """
    ptr, idx, values, dout = t([0, 2, 3], ms.int32), t([0, 2, 1], ms.int32), t([2.0, 3.0, 5.0]), t([[7.0], [11.0]])
    pending = native_boundary(
        [('CSRGather', (ptr, idx, t([[7.0, 7.0, 7.0], [11.0, 11.0, 11.0]]), (2, 3)), t([7.0, 7.0, 11.0]), {})]
    )
    got = gc.get_bprop_csr_reduce_sum(None)(ptr, idx, values, (2, 3), 1, None, dout)
    equal(got, (ptr, idx, t([7.0, 7.0, 11.0]), (), 0))
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_csr_matvec_gradient_transposes_and_preserves_cotangent_protocol(monkeypatch, native_boundary):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: CSR matvec gradient transposes and preserves cotangent protocol.
    Expectation: The transposed CSR protocol preserves ordering and yields value cotangents [14, 28, 33] with
        the supplied dense cotangent.
    """
    monkeypatch.setattr(gc.F, 'make_csr_tensor', CSRTensor)
    ptr, idx, values = t([0, 2, 3], ms.int32), t([0, 2, 1], ms.int32), t([2.0, 3.0, 5.0])
    dense, dout = t([[2.0], [3.0], [4.0]]), t([[7.0], [11.0]])
    pending = native_boundary(
        [
            ('CSR2COO', (ptr, 3), t([0, 0, 1], ms.int32), {}),
            ('COO2CSR', (t([0, 1, 2], ms.int32), 3), t([0, 1, 2, 3], ms.int32), {}),
            (
                'CSRMV',
                (t([0, 1, 2, 3], ms.int32), t([0, 1, 0], ms.int32), t([2.0, 5.0, 3.0]), (3, 2), dout),
                t([[14.0], [55.0], [21.0]]),
                {},
            ),
        ]
    )
    got = gc.get_bprop_csr_mv(None)(ptr, idx, values, (2, 3), dense, None, dout)
    equal(got, (ptr, idx, t([14.0, 28.0, 33.0]), (), t([[14.0], [55.0], [21.0]])))
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('with_segments', [False, True])
def test_sparse_segment_gpu_native_protocol_stays_on_cpu(monkeypatch, native_boundary, with_segments):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Sparse segment GPU native protocol stays on CPU.
    Expectation: The GPU frontend rule sends int32 index metadata to the native boundary and returns its
        cotangent without changing the runtime device.
    """
    x, indices, segments = t(np.arange(8).reshape(4, 2)), t([0, 2, 2], ms.int64), t([0, 0, 1], ms.int64)
    dout, dx = t([[3.0, 5.0], [7.0, 11.0]]), t([[3.0, 5.0], [0.0, 0.0], [10.0, 16.0], [0.0, 0.0]])
    pending = native_boundary(
        [('SparseSegmentSumGrad', (dout, t([0, 2, 2], ms.int32), t([0, 0, 1], ms.int32), t(4, ms.int32)), dx, {})]
    )
    original = gs.context.get_context
    monkeypatch.setattr(gs.context, 'get_context', lambda key: 'GPU' if key == 'device_target' else original(key))
    closure = (
        gs.get_bprop_sparse_segment_sum_with_num_segments(None)
        if with_segments
        else gs.get_bprop_sparse_segment_sum(None)
    )
    monkeypatch.setattr(gs.context, 'get_context', original)
    arguments = (x, indices, segments, t(2, ms.int32)) if with_segments else (x, indices, segments)
    got = closure(*arguments, None, dout)
    assert got[0] is dx
    check(got[1], [0, 0, 0], ms.int32)
    check(got[2], [0, 0, 0], ms.int32)
    if with_segments:
        check(got[3], 0, ms.int32)
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_sparse_segment_dynamic_dim_metadata_keeps_real_mean_gradient(monkeypatch):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Sparse segment dynamic dim metadata keeps real mean gradient.
    Expectation: Dynamic dimension metadata preserves real CPU segment-mean accumulation and zero integer
        metadata cotangents.
    """
    monkeypatch.setattr(gs.F, 'is_sequence_value_unknown', lambda _: True)
    x, indices, segments = t(np.arange(8).reshape(4, 2)), t([0, 2, 2], ms.int32), t([0, 0, 1], ms.int32)
    got = gs.get_bprop_sparse_segment_mean_with_num_segments(None)(
        x, indices, segments, t(2, ms.int32), None, t([[2.0, 4.0], [3.0, 5.0]])
    )
    check(got[0], [[1.0, 2.0], [0.0, 0.0], [4.0, 7.0], [0.0, 0.0]])
    check(got[1], [0, 0, 0], ms.int32)
    check(got[2], [0, 0, 0], ms.int32)
    check(got[3], 0, ms.int32)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('container', [tuple, list])
@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason='Ragged gradient zeros enumerate pairs instead of split tensors'
)
def test_ragged_gradient_structure_matches_input_splits(container):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Ragged gradient structure matches input splits.
    Expectation: Value sensitivities are gathered in order and split cotangents preserve the tuple or list
        container with int64 zeros.
    """
    splits = container([t([0, 2], ms.int64), t([0, 1, 3], ms.int64)])
    values = t([[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]])
    dout = t([17.0, 19.0, 23.0, 29.0, 31.0, 37.0])
    got = gs.get_bprop_ragged_tensor_to_sparse(None)(splits, values, None, (None, dout, None))
    check(got[1], [[17.0, 19.0], [23.0, 29.0], [31.0, 37.0]])
    # Exact Python result types are part of this conversion contract.
    # pylint: disable-next=unidiomatic-typecheck
    assert type(got[0]) is container and len(got[0]) == 2
    check(got[0][0], [0, 0], ms.int64)
    check(got[0][1], [0, 0, 0], ms.int64)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_map_tensor_gradient_routes_actual_primitive_arguments(native_boundary):
    # The map handle is opaque to Python; only the native map-gradient kernel
    # consumes it. No communication, persistent map or device allocation is used.
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Map tensor gradient routes actual primitive arguments.
    Expectation: The map gradient handle is returned from the expected native request and lookup keys receive
        int32 zeros.
    """
    handle = object()
    keys, dout, gradient_handle = t([3, 7], ms.int32), t([[2.0, 3.0], [5.0, 7.0]]), object()
    pending = native_boundary([('MapTensorGetGrad', (handle, keys, dout), gradient_handle, {})])
    got = gs.get_bprop_map_tensor_get(None)(handle, keys, None, dout)
    assert got[0] is gradient_handle
    check(got[1], [0, 0], ms.int32)
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind', ['diag_part', 'set_diag', 'resize', 'index_fill'])
def test_array_dynamic_shape_metadata_uses_real_cpu_gradient_math(monkeypatch, kind):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Array dynamic shape metadata uses real CPU gradient math.
    Expectation: Dynamic shape metadata retains real CPU resize-gradient values [[10, 18], [42, 50]] and zero
        size cotangent.
    """
    monkeypatch.setattr(ga.F, 'is_sequence_value_unknown', lambda _: True)
    if kind == 'diag_part':
        x, k, pad, dout = t(np.ones((2, 3))), t(0, ms.int32), t(0.0), t([2.0, 3.0])
        got = ga.get_bprop_matrix_diag_part_v3(A.MatrixDiagPartV3())(x, k, pad, None, dout)
        equal(got, (t([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]), t(0, ms.int32), t(0.0)))
    elif kind == 'set_diag':
        x, diag, k, dout = t(np.ones((2, 3))), t([2.0, 3.0]), t(0, ms.int32), t([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
        got = ga.get_bprop_matrix_set_diag_v3(A.MatrixSetDiagV3())(x, diag, k, None, dout)
        equal(got, (t([[0.0, 3.0, 5.0], [7.0, 0.0, 13.0]]), t([2.0, 11.0]), t(0, ms.int32)))
    elif kind == 'resize':
        x, dout = t(np.ones((1, 1, 2, 2))), t(np.arange(16).reshape(1, 1, 4, 4))
        got = ga.get_bprop_resize_nearest_neighbor_v2(A.ResizeNearestNeighborV2())(x, t([4, 4], ms.int32), None, dout)
        check(got[0], [[[[10.0, 18.0], [42.0, 50.0]]]])
        assert got[1] == 0
    else:
        x, indices, dout = t([[1.0, 2.0], [3.0, 4.0]]), t([1], ms.int32), t([[2.0, 3.0], [5.0, 7.0]])
        got = ga.get_bprop_index_fill(A.IndexFill())(x, 0, indices, t(9.0), None, dout)
        equal(got, (t([[2.0, 3.0], [0.0, 0.0]]), 0, t([0], ms.int32), t(12.0)))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_embedding_dynamic_metadata_has_explicit_error(monkeypatch):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Embedding dynamic metadata has explicit error.
    Expectation: Unsupported dynamic embedding sensitivity raises the documented RuntimeError.
    """
    monkeypatch.setattr(ga.F, 'is_sequence_value_unknown', lambda _: True)
    with pytest.raises(RuntimeError, match="EmbeddingLookup op's grad don't support Dynamic Sense"):
        ga.get_bprop_embedding_lookup(None)(t(np.ones((3, 2))), t([1], ms.int32), 0, None, t([[2.0, 3.0]]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_unique_gradient_native_tuple_protocol(native_boundary):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Unique gradient native tuple protocol.
    Expectation: UniqueGrad receives the original result tuple and gradient tuple, and its supplied input
        cotangent is preserved.
    """
    x, values, inverse, dout = t([2, 1, 2], ms.int32), t([2, 1], ms.int32), t([0, 1, 0], ms.int32), t([3, 5], ms.int32)
    cotangents = (dout, t([0, 0, 0], ms.int32))
    pending = native_boundary([('UniqueGrad', (cotangents, (values, inverse)), t([3, 5, 3], ms.int32), {})])
    got = ga.get_bprop_unique(P.Unique())(x, (values, inverse), cotangents)
    equal(got, (t([3, 5, 3], ms.int32),))
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('adjoint', [False, True])
@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason='MatrixSolve bprop downcasts float64 gradients to float32'
)
def test_matrix_solve_float64_gradient_preserves_input_precision(adjoint):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Matrix solve float64 gradient preserves input precision.
    Expectation: Intended cotangents match NumPy in float64; the documented float32 downcast remains a strict
        expected failure.
    """
    a, b, d = np.array([[3.0, 1.0], [0.0, 2.0]]), np.array([[1.0], [4.0]]), np.array([[2.0], [-1.0]])
    y = np.linalg.solve(a.T if adjoint else a, b)
    db = np.linalg.solve(a if adjoint else a.T, d)
    da = -(y @ db.T if adjoint else db @ y.T)
    got = gm.get_bprop_matrix_solve(M.MatrixSolve(adjoint=adjoint))(
        t(a, ms.float64), t(b, ms.float64), t(y, ms.float64), t(d, ms.float64)
    )
    check(got[0], da, ms.float64)
    check(got[1], db, ms.float64)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('condition', [True, False])
def test_switch_gradient_native_control_flow_protocol(native_boundary, condition):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Switch gradient native control flow protocol.
    Expectation: The selected branch receives the sensitivity, the other branch receives zeros, and the
        condition cotangent is zero.
    """
    true_branch, false_branch, dout = t([1.0, 2.0]), t([3.0, 4.0]), t([5.0, 7.0])
    zero = t([0.0, 0.0])
    pending = native_boundary(
        [
            ('Switch', (condition, dout, zero), dout if condition else zero, {}),
            ('Switch', (condition, zero, dout), zero if condition else dout, {}),
        ]
    )
    got = gi.bprop_switch(condition, true_branch, false_branch, None, dout)
    equal(got, (0, dout if condition else zero, zero if condition else dout))
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_switch_layer_native_selection_and_gradient_structure(native_boundary):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Switch layer native selection and gradient structure.
    Expectation: Index one selects the second real callable; backpropagation retains sensitivity and zeros index
        metadata.
    """
    index, dout = t(1, ms.int32), t([3.0, 5.0])

    def first(value):
        return value + 1

    def second(value):
        return value * 2

    layers = (first, second)
    pending = native_boundary([('switch_layer', (index, layers), second, {})])
    chosen, backward = gi._fprop_switch_layer(index, layers)
    assert chosen is second
    check(chosen(t([2.0, 3.0])), [4.0, 6.0])
    equal(backward(dout), (dout, t(0, ms.int32), ()))
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize(
    'name',
    [
        pytest.param(
            'bprop_tuple_getitem',
            marks=pytest.mark.xfail(
                strict=True, raises=TypeError, reason='Eager zeros_like tuple becomes a scalar before tuple_setitem'
            ),
        ),
        pytest.param(
            'bprop_list_getitem',
            marks=pytest.mark.xfail(strict=True, raises=ValueError, reason='Eager zeros_like has no list signature'),
        ),
        pytest.param(
            'bprop_dict_getitem',
            marks=pytest.mark.xfail(
                strict=True, raises=NotImplementedError, reason='Eager zeros_like cannot convert a dictionary type'
            ),
        ),
    ],
)
def test_legacy_structure_getitem_gradient_correct_container(name):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Legacy structure getitem gradient correct container.
    Expectation: Intended tuple, list and dictionary cotangents preserve their containers and route sensitivity
        only to the selected element.
    """
    first, second, dout = t([1.0, 2.0]), t([3.0, 4.0]), t([5.0, 7.0])
    if name == 'bprop_tuple_getitem':
        got = gi.bprop_tuple_getitem((first, second), 1, None, dout)
        equal(got, ((t([0.0, 0.0]), dout), 0))
    elif name == 'bprop_list_getitem':
        got = gi.bprop_list_getitem([first, second], 0, None, dout)
        equal(got, ([dout, t([0.0, 0.0])], 0))
    else:
        got = gi.bprop_dict_getitem({'first': first, 'second': second}, 'first', None, dout)
        assert got[1] == 'first'
        equal(got[0]['first'], dout)
        check(got[0]['second'], [0.0, 0.0])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(
    condition=np.lib.NumpyVersion(np.__version__) >= '2.0.0',
    strict=True,
    raises=AttributeError,
    reason='Legacy ScalarCast eager implementation uses removed numpy.cast',
)
def test_legacy_scalar_cast_returns_original_scalar_type():
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Legacy scalar cast returns original scalar type.
    Expectation: The scalar cotangent restores integer type; the removed NumPy cast API remains a
        version-specific strict failure.
    """
    try:
        got = gi.bprop_scalar_cast(2, ms.float32, 2.0, 7.0)
    except AttributeError as error:
        assert 'cast' in str(error)
        raise
    assert got == (7, ms.float32) and isinstance(got[0], int)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_distribute_and_array_reduce_native_shape_protocol(native_boundary):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Distribute and array reduce native shape protocol.
    Expectation: Reduction and distribution receive the expected source shapes and return complementary summed
        or repeated cotangents.
    """
    x, dout = t([[1.0], [2.0]]), t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    reduced = t([[6.0], [15.0]])
    pending = native_boundary(
        [
            ('array_reduce', (gi.F.scalar_add, dout, (2, 1)), reduced, {}),
            ('distribute', (reduced, (2, 3)), t([[6.0, 6.0, 6.0], [15.0, 15.0, 15.0]]), {}),
        ]
    )
    equal(gi.bprop_distribute(x, (2, 3), None, dout), (reduced, 0))
    got = gi.bprop_array_reduce(gi.F.scalar_add, dout, (2, 1), None, reduced)
    equal(got, (t([[6.0, 6.0, 6.0], [15.0, 15.0, 15.0]]), 0))
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind', ['csr', 'coo'])
@pytest.mark.xfail(strict=True, raises=TypeError, reason='Eager ZerosLike does not accept a sparse tensor')
def test_sparse_shape_accessor_gradient_is_zero_sparse_tensor(kind):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Sparse shape accessor gradient is zero sparse tensor.
    Expectation: The intended result retains sparse type and shape with zero values; eager sparse ZerosLike
        rejection remains expected.
    """
    values = t([2.0, 3.0])
    if kind == 'csr':
        obj = CSRTensor(t([0, 1, 2], ms.int32), t([0, 1], ms.int32), values, (2, 2))
        getter = gc.bprop_csr_tensor_get_dense_shape
    else:
        obj = COOTensor(t([[0, 0], [1, 1]], ms.int32), values, (2, 2))
        getter = gc.bprop_coo_tensor_get_dense_shape
    (result,) = getter(obj, None, None)
    assert type(result) is type(obj) and result.shape == obj.shape
    check(result.values, [0.0, 0.0])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('adj_s,adj_d', [(False, True), (True, False)])
def test_sparse_matmul_half_precision_uses_real_cpu_cast_and_gather(native_boundary, adj_s, adj_d):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Sparse matmul half precision uses real CPU cast and gather.
    Expectation: Sparse and dense derivatives match independent matrix products in float16, with int64 zeros for
        index and shape metadata.
    """
    indices = t([[0, 0], [0, 2], [1, 1]], ms.int64)
    values, shape = t([2.0, 3.0, 5.0], ms.float16), t([2, 3], ms.int64)
    sparse = np.array([[2.0, 0.0, 3.0], [0.0, 5.0, 0.0]])
    operator = sparse.T if adj_s else sparse
    dense_matrix = np.arange(operator.shape[1] * 2, dtype=np.float32).reshape(operator.shape[1], 2) + 1
    dense_input = dense_matrix.T if adj_d else dense_matrix
    dout = np.arange(operator.shape[0] * 2, dtype=np.float32).reshape(operator.shape[0], 2) + 2
    raw_dense_grad = operator.T @ dout
    expected_dense = raw_dense_grad.T if adj_d else raw_dense_grad
    expected_sparse = dout @ dense_matrix.T
    if adj_s:
        expected_sparse = expected_sparse.T
    pending = native_boundary(
        [
            (
                'SparseTensorDenseMatmul',
                (indices, values, shape, t(dout, ms.float16)),
                t(raw_dense_grad, ms.float16),
                {'adjoint_st': not adj_s},
            )
        ]
    )
    got = gc.get_bprop_sparse_tensor_dense_matmul(P.SparseTensorDenseMatmul(adjoint_st=adj_s, adjoint_dt=adj_d))(
        indices, values, shape, t(dense_input, ms.float16), None, t(dout, ms.float16)
    )
    check(got[0], np.zeros((3, 2)), ms.int64)
    check(got[1], expected_sparse[[0, 0, 1], [0, 2, 1]], ms.float16)
    check(got[2], [0, 0], ms.int64)
    check(got[3], expected_dense, ms.float16)
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('shift', [None, 1])
def test_broadcast_gradient_dynamic_metadata_dispatches_to_real_cpu_reductions(monkeypatch, shift):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Broadcast gradient dynamic metadata dispatches to real CPU reductions.
    Expectation: Dynamic shape dispatch sums each sensitivity along its operand's broadcast axes while
        preserving shifted trailing dimensions.
    """
    monkeypatch.setattr(gm.F, 'is_sequence_value_unknown', lambda _: True)
    x_shape, y_shape = ((2, 1), (1, 3)) if shift is None else ((2, 1, 2), (1, 3, 2))
    d_shape = (2, 3) if shift is None else (2, 3, 2)
    sensitivity = np.arange(np.prod(d_shape), dtype=np.float32).reshape(d_shape) + 1
    x, y, dout = t(np.ones(x_shape)), t(np.ones(y_shape)), t(sensitivity)
    got = (
        gm.binop_grad_common(x, y, dout, dout)
        if shift is None
        else gm.binop_grad_common_with_shift(x, y, dout, dout, shift)
    )
    check(got[0], sensitivity.sum(axis=1, keepdims=True))
    check(got[1], sensitivity.sum(axis=0, keepdims=True))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('scalar_left', [True, False])
@pytest.mark.parametrize('shift', [None, 0])
def test_broadcast_scalar_abstract_shape_protocol_keeps_real_reduction(monkeypatch, scalar_left, shift):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Broadcast scalar abstract shape protocol keeps real reduction.
    Expectation: The scalar operand receives the total sensitivity 21 and the full-shaped operand retains the
        original tensor.
    """
    scalar, full, dout = t(2.0), t(np.ones((2, 3))), t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    original_shape = gm.shape_op
    monkeypatch.setattr(gm, 'shape_op', lambda value: t([], ms.int64) if value is scalar else original_shape(value))
    monkeypatch.setattr(gm.F, 'is_sequence_value_unknown', lambda _: True)
    x, y = (scalar, full) if scalar_left else (full, scalar)
    got = (
        gm.binop_grad_common(x, y, dout, dout)
        if shift is None
        else gm.binop_grad_common_with_shift(x, y, dout, dout, shift)
    )
    equal(got, (t(21.0), dout) if scalar_left else (dout, t(21.0)))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason='Eager zeros_like cannot convert a dictionary type')
def test_dictionary_inplace_gradient_zeros_each_input():
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Dictionary inplace gradient zeros each input.
    Expectation: Intended dictionary values and assigned input receive zero cotangents; eager dictionary
        ZerosLike rejection remains expected.
    """
    x, target, dout = {'key': t([2.0, 3.0])}, t([5.0, 7.0]), {'key': t([11.0, 13.0])}
    got = gi.get_bprop_dict_inplace_setitem(None)(x, 'key', target, None, dout)
    assert isinstance(got[0], dict)
    check(got[0]['key'], [0.0, 0.0])
    check(got[2], [0.0, 0.0])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dynamic', [False, True])
def test_index_fill_scalar_sensitivity_uses_real_cpu_rule(monkeypatch, dynamic):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Index fill scalar sensitivity uses real CPU rule.
    Expectation: The scalar replacement receives sensitivity eleven while the overwritten input and index
        metadata receive zeros.
    """
    if dynamic:
        monkeypatch.setattr(ga.F, 'is_sequence_value_unknown', lambda _: True)
    got = ga.get_bprop_index_fill(A.IndexFill())(t(3.0), 0, t([0], ms.int32), t(7.0), None, t(11.0))
    equal(got, (t(0.0), 0, t([0], ms.int32), t(11.0)))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_sparse_add_dynamic_shape_metadata_keeps_real_cpu_native_gradient(monkeypatch):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Sparse add dynamic shape metadata keeps real CPU native gradient.
    Expectation: Merged sparse sensitivities gather to [11, 17] and [11, 13], while structural inputs receive
        typed zeros.
    """
    monkeypatch.setattr(gc.F, 'is_sequence_value_unknown', lambda _: True)
    i1, i2, shape = t([[0, 0], [1, 1]], ms.int64), t([[0, 0], [1, 0]], ms.int64), t([2, 2], ms.int64)
    v1, v2 = t([2.0, 3.0]), t([5.0, 7.0])
    out = S.SparseAdd()(i1, v1, shape, i2, v2, shape, t(0.0))
    got = gc.get_bprop_sparse_add(None)(i1, v1, shape, i2, v2, shape, t(0.0), out, (None, t([11.0, 13.0, 17.0]), None))
    equal(
        got,
        (
            t(np.zeros(i1.shape), ms.int64),
            t([11.0, 17.0]),
            t([0, 0], ms.int64),
            t(np.zeros(i2.shape), ms.int64),
            t([11.0, 13.0]),
            t([0, 0], ms.int64),
            t(0.0),
        ),
    )


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('under', [False, True])
@pytest.mark.parametrize('batch', [False, True])
@pytest.mark.parametrize('dtype', [ms.complex64, ms.complex128])
def test_complex_least_squares_native_factorization_protocol_and_cpu_assembly(native_boundary, under, batch, dtype):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Unavailable complex Cholesky kernels are isolated; all Python gradient assembly runs on CPU.
        The real-valued complex fixtures independently check the native requests and the assembled gradients.
        They     do not claim native complex solver execution.
    Expectation: The native factorization requests retain their exact inputs and Python assembly matches
        independent complex matrix and right-hand-side cotangents.
    """
    np_dtype = np.complex64 if dtype == ms.complex64 else np.complex128
    a = np.array([[2.0, 1.0], [1.0, 3.0], [1.0, -1.0]], np_dtype)
    if under:
        a = a.T
    b = np.arange(1, a.shape[0] + 1, dtype=np.float64).reshape(-1, 1).astype(np_dtype)
    d = np.arange(1, a.shape[1] + 1, dtype=np.float64).reshape(-1, 1).astype(np_dtype) / 3
    if batch:
        a, b, d = a[None], b[None], d[None]
    ah = np.swapaxes(a.conj(), -1, -2)
    l2 = 0.25
    gram = (a @ ah if under else ah @ a) + l2 * np.eye(a.shape[-2] if under else a.shape[-1])
    gram = gram.astype(np_dtype)
    chol = np.linalg.cholesky(gram).astype(np_dtype)
    out = (ah @ np.linalg.solve(gram, b) if under else np.linalg.solve(gram, ah @ b)).astype(np_dtype)
    specifications = [('Cholesky', (t(gram, dtype), False), t(chol, dtype), {})]
    if under:
        db = np.linalg.solve(gram, a @ d).astype(np_dtype)
        tmp = np.linalg.solve(gram, b).astype(np_dtype)
        specifications += [
            ('CholeskySolve', (t(a @ d, dtype), t(chol, dtype)), t(db, dtype), {'upper': False}),
            ('CholeskySolve', (t(b, dtype), t(chol, dtype)), t(tmp, dtype), {'upper': False}),
        ]
        da = -db @ np.swapaxes(tmp.conj(), -1, -2) @ a + tmp @ np.swapaxes(d - ah @ db, -1, -2)
    else:
        z = np.linalg.solve(gram, d).astype(np_dtype)
        specifications += [('CholeskySolve', (t(d, dtype), t(chol, dtype)), t(z, dtype), {'upper': False})]
        zh = np.swapaxes(z.conj(), -1, -2)
        da, db = -a @ (out @ zh + z @ np.swapaxes(out.conj(), -1, -2)) + b @ zh, a @ z
    pending = native_boundary(specifications)
    got = gm.get_bprop_matrix_solve_ls(M.MatrixSolveLs())(
        t(a, dtype), t(b, dtype), t(l2, ms.float64), t(out, dtype), t(d, dtype)
    )
    check(got[0], da, dtype, atol=1e-4)
    check(got[1], db, dtype, atol=1e-4)
    check(got[2], 0.0, ms.float64)
    assert not pending


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(
    strict=True, raises=TypeError, reason='Dynamic MaskedFill supplies a range axis unsupported by eager ReduceSum'
)
def test_masked_fill_dynamic_rank_path_preserves_all_input_dtypes(monkeypatch):
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Masked fill dynamic rank path preserves all input dtypes.
    Expectation: Dynamic-rank routing zeros filled input positions, sums fill sensitivity to nine and retains
        boolean mask cotangents.
    """
    monkeypatch.setattr(ga.F, 'is_sequence_shape_unknown', lambda _: True)
    x, mask, dout = (
        t([[1.0, 2.0], [3.0, 4.0]]),
        t([[True, False], [False, True]], ms.bool_),
        t([[2.0, 3.0], [5.0, 7.0]]),
    )
    got = ga.get_bprop_masked_select(P.MaskedFill())(x, mask, t(9.0), None, dout)
    check(got[0], [[0.0, 3.0], [5.0, 0.0]])
    check(got[2], 9.0)
    check(got[1], np.zeros(mask.shape, dtype=bool), ms.bool_)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(
    strict=True, raises=ValueError, reason='EmbeddingLookup empty-indices rule reshapes empty dout to nonempty row'
)
def test_embedding_lookup_empty_indices_preserves_empty_row_gradient():
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: Embedding lookup empty indices preserves empty row gradient.
    Expectation: Empty indices retain empty row-gradient storage when supported; the current zero-size reshape
        defect remains a strict failure.
    """
    try:
        got = ga.get_bprop_embedding_lookup(None)(t(np.ones((3, 2))), t([], ms.int32), 0, None, t(np.empty((0, 2))))
    except ValueError as error:
        assert 'proposed_shape' in str(error) and 'size 0' in str(error)
        raise
    assert got[0].indices.shape == (0,)
    check(got[0].values, np.empty((0, 2)))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=AssertionError, reason='Sspaddmm bprop zeros differentiable sparse input values')
def test_sspaddmm_sparse_values_gradient_matches_linear_map():
    """Feature: Python gradient routing, metadata and native ABI contracts.
    Description: SSPAddmm sparse values gradient matches linear map.
    Expectation: The nine-input gradient contract returns sparse-value cotangents [69, 111] from the independent
        linear map.
    """
    indices = t([[0, 1], [0, 1]], ms.int64)
    shape = t([2, 2], ms.int64)
    x1, x2, dense = t([2.0, 3.0]), t([5.0, 7.0]), t([[11.0, 13.0], [17.0, 19.0]])
    alpha, beta = t(2.0), t(3.0)
    # d(sum(D * (beta*x1 + alpha*x2@dense)))/d x1_values.
    out_indices = t([[0, 0, 1, 1], [0, 1, 0, 1]], ms.int64)
    sensitivity = (out_indices, t([23.0, 29.0, 31.0, 37.0]), shape)
    got = gs.get_bprop_sspaddmm(S.Sspaddmm())(
        indices, x1, shape, indices, x2, shape, dense, alpha, beta, None, sensitivity
    )
    assert len(got) == 9
    check(got[1], [69.0, 111.0])
