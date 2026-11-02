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
"""Execute real TE graphs on LLVM, isolating binary emission and one unsupported tuple scheduler."""
import importlib
import itertools

import numpy as np
import pytest
import mindspore as ms

te = pytest.importorskip('te', reason='Legacy compute graph tests require the optional CANN TE SDK')
from te import tvm
import te.lang.cce

_SERIAL = itertools.count()


@pytest.fixture(scope='module', autouse=True)
def host_graph_context():
    """Keep MindSpore on CPU; TVM executes the same TE graph through its LLVM backend."""
    if not tvm.runtime.enabled('llvm'):
        pytest.skip('TE SDK was built without the LLVM host backend')
    previous = {key: ms.get_context(key) for key in ('mode', 'device_target')}
    ms.set_context(device_target='CPU', mode=ms.PYNATIVE_MODE)
    yield
    ms.set_context(**previous)


def legacy(name):
    """Import the unmodified kernel module, including its real compute implementation."""
    return importlib.import_module('mindspore.ops._op_impl._custom_op.' + name)


def evaluate_graph(inputs, outputs, arrays):
    """Use native LLVM execution, rather than replacing any arithmetic primitive."""
    outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
    schedule = tvm.create_schedule([tensor.op for tensor in outputs])
    module = tvm.build(schedule, list(inputs) + outputs, target='llvm', name=f'coverage_te_{next(_SERIAL)}')
    arguments = [tvm.nd.array(np.asarray(array, dtype=str(tensor.dtype))) for tensor, array in zip(inputs, arrays)]
    results = [tvm.nd.empty(tuple(int(dimension) for dimension in tensor.shape), dtype=str(tensor.dtype))
               for tensor in outputs]
    module(*(arguments + results))
    return [value.numpy() for value in results]


def evaluate(compute, arrays, *constants):
    arrays = [np.asarray(array, np.float32) for array in arrays]
    inputs = [tvm.placeholder(array.shape, name=f'input_{index}', dtype='float32')
              for index, array in enumerate(arrays)]
    return evaluate_graph(inputs, compute(*inputs, *constants), arrays)


@pytest.mark.parametrize('symmetric,narrow,expected', [
    (False, False, [-1, -1, 0, 1, 2, 2]),
    (False, True, [-1.5, 0, 0, 0, 1.5, 1.5]),
    (True, False, [-8 / 3, 0, 0, 0, 4 / 3, 4 / 3]),
    (True, True, [-2, 0, 0, 0, 2, 2]),
])
def test_legacy_fake_quantization_real_llvm_graph(symmetric, narrow, expected):
    """Clamping, nudged zero point and half-up quantization must agree with literal fixtures."""
    graph = legacy('fake_quant_perlayer').fake_quant_per_layer_compute
    actual, = evaluate(graph, [[-4, -0.6, 0, 0.6, 1.6, 4], [-1], [2]], {}, int(narrow), 3, symmetric)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('symmetric,narrow,expected', [
    (False, False, [0, 2, 3, 4, 5, 0]),
    (False, True, [0, 2, 3, 4, 0, 0]),
    (True, False, [0, 2, 3, 4, 0, 0]),
    (True, True, [0, 2, 3, 4, 5, 0]),
])
def test_legacy_quantization_gradient_real_llvm_graph(symmetric, narrow, expected):
    """Only sensitivities strictly within the nudged clipping interval pass through."""
    graph = legacy('fake_quant_perlayer_grad').fake_quant_per_layer_grad_compute
    actual, = evaluate(graph, [[1, 2, 3, 4, 5, 6], [-4, -0.6, 0, 0.6, 1.6, 4], [-1], [2]],
                        int(narrow), 3, symmetric)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('negative_truncation,expected', [
    pytest.param(False, [-2, -1, 0, 1, 1, 2], marks=pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason='CANN TE floor is an int32 Cast on generic LLVM; negative values truncate')),
                                                        (True, [0, 0, 0, 1, 1, 2])])
def test_legacy_learned_scale_quantization_real_llvm_graph(negative_truncation, expected):
    """Learned-scale clipping must retain or truncate negative values as requested."""
    graph = legacy('fake_learned_scale_quant_perlayer').fake_learned_scale_quant_perlayer_compute
    actual, = evaluate(graph, [[-3, -1, 0, 0.5, 1, 3], [2], [2]], negative_truncation)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('negative_truncation', [False, True])
@pytest.mark.parametrize('module', ['fake_learned_scale_quant_perlayer', 'fake_learned_scale_quant_perchannel'])
def test_learned_scale_nonnegative_domain_real_llvm_graph(module, negative_truncation):
    """Nonnegative floor arguments have identical LLVM and CCE semantics."""
    compute = getattr(legacy(module), module + '_compute')
    actual, = evaluate(compute, [[0, 0.5, 1, 3], [2], [2]], negative_truncation)
    np.testing.assert_allclose(actual, [0, 1, 1, 2], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('module', ['fake_learned_scale_quant_perlayer', 'fake_learned_scale_quant_perchannel'])
def test_learned_scale_zero_alpha_is_finite(module):
    """A zero learned scale is guarded by epsilon before division."""
    actual, = evaluate(getattr(legacy(module), module + '_compute'), [[0, 0.5, 1], [0], [2]], True)
    np.testing.assert_allclose(actual, [0, 1e-6, 1e-6], rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize('ema,expected', [(False, ([-2], [4])), (True, ([-3], [6]))])
def test_legacy_minmax_ema_real_llvm_graph(ema, expected):
    """EMA combines old and current extrema before keeping zero inside the range."""
    graph = legacy('minmax_update_perlayer').minmax_update_perlayer_compute
    result = evaluate(graph, [[-2, -1, 0.5, 1, 2, 4], [-4], [8]], ema, 0.5)
    for actual, value in zip(result, expected):
        np.testing.assert_allclose(actual, value, rtol=1e-6)


def test_legacy_correction_multiplication_real_llvm_graph():
    """Each channel uses its own batch/running-standard-deviation ratio."""
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    batch = np.array([1, 2, 3], np.float32).reshape(1, 3, 1, 1)
    running = np.array([2, 2, 1], np.float32).reshape(1, 3, 1, 1)
    actual, = evaluate(legacy('correction_mul').correction_mul_compute, [data, batch, running])
    np.testing.assert_allclose(actual, data * np.array([0.5, 1, 3]).reshape(1, 3, 1, 1), rtol=1e-6)


def test_legacy_batchnorm_fold_gradient_real_llvm_graph():
    """The batch-statistic gradient must normalize over batch and spatial dimensions."""
    data = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    arrays = [np.array([1, 2], np.float32).reshape(1, 2, 1, 1),
              np.array([2, 3], np.float32).reshape(1, 2, 1, 1), data,
              np.array([2, 4], np.float32).reshape(1, 2, 1, 1),
              np.array([2, 4], np.float32).reshape(1, 2, 1, 1)]
    actual, = evaluate(legacy('batchnorm_fold_grad')._batchnorm_fold_grad_compute, arrays)
    expected = ((data - arrays[3]) * arrays[1] / arrays[4] + arrays[0]) / 8
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_legacy_second_fold_real_llvm_graph():
    """Folded normalization combines the channel scale and offset in the correct order."""
    data = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
    arrays = [data] + [np.array(values, np.float32).reshape(1, 2, 1, 1)
                       for values in ([1, -1], [2, 3], [2, 4], [1, 2], [3, 2])]
    actual, = evaluate(legacy('batchnorm_fold2').batchnorm_fold2_compute, arrays)
    expected = data * arrays[5] / arrays[3] + arrays[1] - arrays[4] * arrays[2] / arrays[3]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def descriptor(shape, dtype='float32'):
    return {'shape': tuple(shape), 'ori_shape': tuple(shape), 'dtype': dtype,
            'format': 'NCHW', 'ori_format': 'NCHW'}


@pytest.mark.parametrize('ema', [False, True])
def test_minmax_entry_validates_and_emits_the_same_executable_graph(monkeypatch, ema):
    """The CCE binary-emission boundary receives the real graph, compiled here to host LLVM."""
    calls = []
    def host_compile(schedule, config):
        assert schedule is not None
        assert config['name'] == 'coverage_minmax'
        tensors = config['tensor_list']
        assert len(tensors) == 5
        assert all(tuple(int(dim) for dim in item.shape) == (1,) and str(item.dtype) == 'float32'
                   for item in tensors[3:])
        calls.append(evaluate_graph(tensors[:3], tensors[3:],
                                    [[-2, -1, 0.5, 1, 2, 4], [-4], [8]]))
    monkeypatch.setattr(te.lang.cce, 'cce_build_code', host_compile)
    legacy('minmax_update_perlayer').minmax_update_perlayer(
        descriptor((6,)), descriptor((1,)), descriptor((1,)), descriptor((1,)), descriptor((1,)),
        ema, 0.5, kernel_name='coverage_minmax')
    assert len(calls) == 1
    for actual, expected in zip(calls[0], ([-3], [6]) if ema else ([-2], [4])):
        np.testing.assert_allclose(actual, expected, rtol=1e-6)


@pytest.mark.parametrize('dtype', ['int32', 'bool'])
def test_minmax_entry_rejects_unsupported_storage_dtype(dtype):
    """Invalid storage types must be rejected before any CCE code emission."""
    with pytest.raises(RuntimeError):
        legacy('minmax_update_perlayer').minmax_update_perlayer(
            descriptor((6,), dtype), descriptor((1,)), descriptor((1,)), descriptor((1,)), descriptor((1,)),
            True, 0.5)


def capture_emitted_graph(monkeypatch, arrays, expected):
    """Verify the entire input/output ABI and execute the emitted, unmodified compute graph."""
    calls = []

    def compile_on_host(schedule, config):
        assert schedule is not None
        assert config['name'] == 'coverage_legacy'
        tensors = config['tensor_list']
        assert len(tensors) == len(arrays) + len(expected)
        for tensor, array in zip(tensors, arrays):
            assert tuple(int(dim) for dim in tensor.shape) == np.asarray(array).shape
            assert str(tensor.dtype) == 'float32'
        for tensor, value in zip(tensors[len(arrays):], expected):
            assert tuple(int(dim) for dim in tensor.shape) == np.asarray(value).shape
            assert str(tensor.dtype) == 'float32'
        results = evaluate_graph(tensors[:len(arrays)], tensors[len(arrays):], arrays)
        for actual, value in zip(results, expected):
            np.testing.assert_allclose(actual, value, rtol=3e-5, atol=3e-6)
        calls.append(config['name'])

    monkeypatch.setattr(te.lang.cce, 'cce_build_code', compile_on_host)
    return calls


@pytest.mark.parametrize('perchannel', [False, True])
@pytest.mark.parametrize('narrow', [False, True])
@pytest.mark.parametrize('symmetric', [False, True])
def test_quantization_entry_real_scheduler_and_host_execution(monkeypatch, perchannel, narrow, symmetric):
    """Both registrations must emit numerically equivalent quantization for replicated ranges."""
    shape = (2, 3) if perchannel else (6,)
    data = np.array([-4, -0.6, 0, 0.6, 1.6, 4], np.float32).reshape(shape)
    stat_shape = (2, 1) if perchannel else (1,)
    minimum, maximum = np.full(stat_shape, -1, np.float32), np.full(stat_shape, 2, np.float32)
    fixtures = {(False, False): [-1, -1, 0, 1, 2, 2],
                (False, True): [-1.5, 0, 0, 0, 1.5, 1.5],
                (True, False): [-8 / 3, 0, 0, 0, 4 / 3, 4 / 3],
                (True, True): [-2, 0, 0, 0, 2, 2]}
    calls = capture_emitted_graph(monkeypatch, [data, minimum, maximum],
                                  [np.array(fixtures[symmetric, narrow], np.float32).reshape(shape)])
    name = 'fake_quant_perchannel' if perchannel else 'fake_quant_perlayer'
    entry = getattr(legacy(name), 'fake_quant_perchannel' if perchannel else 'fake_quant_per_layer')
    args = [descriptor(shape), descriptor((2,) if perchannel else (1,)),
            descriptor((2,) if perchannel else (1,)), descriptor(shape), symmetric, narrow, 2]
    if perchannel:
        args.append(0)
    entry(*args, kernel_name='coverage_legacy')
    assert calls == ['coverage_legacy']


@pytest.mark.parametrize('perchannel', [False, True])
@pytest.mark.parametrize('narrow', [False, True])
def test_quantization_gradient_entry_real_scheduler(monkeypatch, perchannel, narrow):
    """Nonuniform upstream gradients and both clipping endpoints exercise the emitted gradient."""
    shape = (2, 3) if perchannel else (6,)
    data = np.array([-4, -0.6, 0, 0.6, 1.6, 4], np.float32).reshape(shape)
    dout = np.arange(1, 7, dtype=np.float32).reshape(shape)
    stat_shape = (2, 1) if perchannel else (1,)
    arrays = [dout, data, np.full(stat_shape, -1, np.float32), np.full(stat_shape, 2, np.float32)]
    expected = np.array([0, 2, 3, 4, 0 if narrow else 5, 0], np.float32).reshape(shape)
    calls = capture_emitted_graph(monkeypatch, arrays, [expected])
    name = 'fake_quant_perchannel_grad' if perchannel else 'fake_quant_perlayer_grad'
    entry = getattr(legacy(name), 'fake_quant_perchannel_grad' if perchannel else 'fake_quant_per_layer_grad')
    args = [descriptor(shape), descriptor(shape), descriptor((2,) if perchannel else (1,)),
            descriptor((2,) if perchannel else (1,)), descriptor(shape)]
    args.extend([False, narrow, 2, 0] if perchannel else [2, False, narrow])
    entry(*args, kernel_name='coverage_legacy')
    assert calls == ['coverage_legacy']


@pytest.mark.parametrize('perchannel', [False, True])
@pytest.mark.parametrize('negative_truncation', [False, True])
def test_learned_scale_entry_real_scheduler(monkeypatch, perchannel, negative_truncation):
    """The parameter helper must preserve flattening and channel broadcasting at the emission ABI."""
    shape = (2, 2) if perchannel else (4,)
    data = np.array([0, 0.5, 1, 3], np.float32).reshape(shape)
    alpha = np.full((2, 1) if perchannel else (1,), 2, np.float32)
    calls = capture_emitted_graph(monkeypatch, [data, alpha, np.array([2], np.float32)],
                                  [np.array([0, 1, 1, 2], np.float32).reshape(shape)])
    name = 'fake_learned_scale_quant_perchannel' if perchannel else 'fake_learned_scale_quant_perlayer'
    args = [descriptor(shape), descriptor((2,) if perchannel else (1,)), descriptor((1,)),
            descriptor(shape), negative_truncation]
    if perchannel:
        args.append(0)
    getattr(legacy(name), name)(*args, kernel_name='coverage_legacy')
    assert calls == ['coverage_legacy']


@pytest.mark.parametrize('perchannel', [False, True])
@pytest.mark.parametrize('negative_truncation', [False, True])
def test_learned_scale_gradient_entry_real_scheduler(monkeypatch, perchannel, negative_truncation):
    """Interior STE gradients and saturation scale gradients use nonuniform sensitivities."""
    shape = (2, 2) if perchannel else (4,)
    data = np.array([0, 0.5, 1, 3], np.float32).reshape(shape)
    dout = np.arange(1, 5, dtype=np.float32).reshape(shape)
    alpha = np.full((2, 1) if perchannel else (1,), 2, np.float32)
    expected = [np.array([1, 2, 3, 0], np.float32).reshape(shape),
                np.array([0, 0.5, 0, 4], np.float32).reshape(shape)]
    calls = capture_emitted_graph(monkeypatch, [dout, data, alpha, np.array([2], np.float32)], expected)
    name = 'fake_learned_scale_quant_perchannel_grad' if perchannel else 'fake_learned_scale_quant_perlayer_grad'
    entry_name = name.replace('_grad', '_grad_d')
    args = [descriptor(shape), descriptor(shape), descriptor((2,) if perchannel else (1,)),
            descriptor((1,)), descriptor(shape), descriptor(shape), negative_truncation]
    if perchannel:
        args.append(0)
    getattr(legacy(name), entry_name)(*args, kernel_name='coverage_legacy')
    assert calls == ['coverage_legacy']


@pytest.mark.parametrize('samples', [1, 8])
def test_batchnorm_statistics_real_llvm_graph(samples):
    """Running variance uses the unbiased estimator; singleton batches retain zero sample variance."""
    data = np.arange(samples * 2, dtype=np.float32).reshape(samples, 2, 1, 1)
    axes = (0, 2, 3)
    mean = np.array([2, 4], np.float32).reshape(1, 2, 1, 1)
    variance = np.array([4, 9], np.float32).reshape(1, 2, 1, 1)
    arrays = [data, data.sum(axes, keepdims=True), (data ** 2).sum(axes, keepdims=True), mean, variance]
    results = evaluate(legacy('batchnorm_fold')._batchnorm_fold_compute, arrays, 0.75, 0.25)
    sample_mean, sample_var = data.mean(axes, keepdims=True), data.var(axes, keepdims=True)
    unbiased = sample_var * samples / (samples - 1) if samples > 1 else sample_var * 0
    expected = [data, sample_mean, np.sqrt(sample_var + 0.25), mean, np.sqrt(variance + 0.25),
                sample_mean * 0.25 + mean * 0.75, unbiased * 0.25 + variance * 0.75]
    for actual, value in zip(results, expected):
        np.testing.assert_allclose(actual, value, rtol=2e-6, atol=2e-6)


def test_second_fold_gradient_real_llvm_graph():
    """Analytic channel derivatives and input derivative must be emitted in the documented order."""
    data = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
    arrays = [data] + [np.array(values, np.float32).reshape(1, 2, 1, 1)
                       for values in ([1, 2], [3, 4], [2, 3], [2, 4], [1, 2], [3, 2])]
    result = evaluate(legacy('batchnorm_fold2_grad').batchnorm_fold2_grad_compute, arrays)
    dout, reduced, x_reduced, gamma, std, mean, running = arrays
    expected = [(reduced * mean * gamma - x_reduced * running) / std ** 2,
                -reduced * gamma / std, -reduced * mean / std, dout * running / std]
    for actual, value in zip(result, expected):
        np.testing.assert_allclose(actual, value, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('shape,channel,data_format,axes', [
    ((2, 3, 2, 2), 0, 'NCHW', (1, 2, 3)),
    ((2, 3, 2, 2, 2), 0, 'NC1HWC0', (1, 2, 3, 4)),
    ((2, 3, 2, 2), 1, 'NCHW', (2, 3)),
])
def test_correction_gradient_reduction_real_llvm_graph(shape, channel, data_format, axes):
    """Both supported storage layouts reduce exactly the documented nonchannel axes."""
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    actual, = evaluate(legacy('correction_mul_grad').correction_mul_grad_reduce_compute,
                        [data], channel, data_format)
    np.testing.assert_allclose(actual, data.sum(axes, keepdims=True), rtol=1e-6)


@pytest.mark.parametrize('blocked', [False, True])
@pytest.mark.parametrize('operation', ['correction_mul', 'correction_mul_grad', 'correction_mul_grad_reduce',
                                      'batchnorm_fold', 'batchnorm_fold_grad', 'batchnorm_fold2',
                                      'batchnorm_fold2_grad'])
def test_normalization_entry_real_scheduler(monkeypatch, blocked, operation):
    """Unmodified CCE scheduling accepts both NCHW and blocked layouts; LLVM verifies all outputs."""
    shape = (2, 1, 2, 2, 16) if blocked else (2, 2, 2, 2)
    stat_shape = (1, 1, 1, 1, 16) if blocked else (1, 2, 1, 1)
    channels = 16 if blocked else 2
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 16
    channel = np.arange(1, channels + 1, dtype=np.float32).reshape(stat_shape)
    mean, variance = channel, channel + 1
    batch_std, running_std, gamma, beta = channel + 2, channel + 3, channel + 4, -channel
    x_desc = descriptor(shape)
    stat_desc = descriptor(stat_shape if blocked else (channels,))
    if blocked:
        x_desc['format'] = stat_desc['format'] = 'NC1HWC0'
        stat_desc['ori_shape'] = (channels,)
    args = []
    if operation == 'correction_mul':
        arrays = [data, batch_std, running_std]
        expected = [data * batch_std / running_std]
        args = [x_desc, stat_desc, stat_desc, x_desc, 1]
    elif operation == 'correction_mul_grad':
        arrays = [data + 1, data, batch_std, running_std]
        expected = [(data + 1) * batch_std / running_std, (data + 1) * data / running_std]
        args = [x_desc, x_desc, stat_desc, stat_desc, x_desc, x_desc, 1]
    elif operation == 'correction_mul_grad_reduce':
        arrays = [data]
        expected = [data.sum((2, 3), keepdims=True)]
        args = [x_desc, stat_desc, 1]
    elif operation == 'batchnorm_fold':
        arrays = [data, data.sum((0, 2, 3), keepdims=True),
                  (data ** 2).sum((0, 2, 3), keepdims=True), mean, variance]
        sample_mean, sample_var = data.mean((0, 2, 3), keepdims=True), data.var((0, 2, 3), keepdims=True)
        expected = [data, sample_mean, np.sqrt(sample_var + 0.25), mean, np.sqrt(variance + 0.25),
                    sample_mean * 0.25 + mean * 0.75, sample_var * 8 / 7 * 0.25 + variance * 0.75]
        args = [x_desc, stat_desc, stat_desc, stat_desc, stat_desc] + [stat_desc] * 7 + [0.75, 0.25]
    elif operation == 'batchnorm_fold_grad':
        arrays = [channel, channel + 1, data, mean, batch_std]
        expected = [((data - mean) * (channel + 1) / batch_std + channel) / 8]
        args = [stat_desc, stat_desc, x_desc, stat_desc, stat_desc, x_desc]
    elif operation == 'batchnorm_fold2':
        arrays = [data, beta, gamma, batch_std, mean, running_std]
        expected = [data * running_std / batch_std + beta - mean * gamma / batch_std]
        args = [x_desc] + [stat_desc] * 5 + [x_desc]
    else:
        arrays = [data, channel, channel + 1, gamma, batch_std, mean, running_std]
        expected = [(channel * mean * gamma - (channel + 1) * running_std) / batch_std ** 2,
                    -channel * gamma / batch_std, -channel * mean / batch_std,
                    data * running_std / batch_std]
        args = [x_desc] + [stat_desc] * 6 + [stat_desc] * 3 + [x_desc]
    calls = capture_emitted_graph(monkeypatch, arrays, expected)
    module = 'correction_mul_grad' if operation == 'correction_mul_grad_reduce' else operation
    getattr(legacy(module), operation)(*args, kernel_name='coverage_legacy')
    assert calls == ['coverage_legacy']


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('ema', [False, True])
def test_channel_minmax_entry_real_scheduler(monkeypatch, axis, ema):
    """Channel statistics match independent extrema in blocked activation and weight layouts."""
    shape = (2, 1, 2, 2, 16)
    data = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 64) / 16
    stat_shape = (2,) if axis == 0 else (1, 16)
    minimum, maximum = np.full(stat_shape, -8, np.float32), np.full(stat_shape, 8, np.float32)
    axes = (1, 2, 3, 4) if axis == 0 else (0, 2, 3)
    weight = 0.5 if ema else 0
    expected = [np.minimum(minimum * weight + data.min(axes) * (1 - weight), 0),
                np.maximum(maximum * weight + data.max(axes) * (1 - weight), 0)]
    calls = capture_emitted_graph(monkeypatch, [data, minimum, maximum], expected)
    x_desc = descriptor(shape)
    x_desc['format'] = 'NC1HWC0'
    x_desc['ori_shape'] = (2, 16, 2, 2)
    stat_desc = descriptor((2,) if axis == 0 else (1, 1, 1, 1, 16))
    stat_desc['ori_shape'] = (2,) if axis == 0 else (16,)
    legacy('minmax_update_perchannel').minmax_update_perchannel(
        x_desc, stat_desc, stat_desc, stat_desc, stat_desc, ema, 0.5, 0, kernel_name='coverage_legacy')
    assert calls == ['coverage_legacy']


@pytest.mark.parametrize('blocked', [False, True])
def test_second_fold_gradient_reduce_real_llvm_graph(blocked):
    """Paired reductions sum the sensitivity and sensitivity-input product over sample axes."""
    shape = (2, 1, 2, 2, 16) if blocked else (2, 2, 2, 2)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 16
    dout = data + 0.5
    desc = descriptor(shape)
    if blocked:
        desc['format'] = 'NC1HWC0'
    arrays = [tvm.placeholder(shape, name=name, dtype='float32') for name in ('sensitivity', 'input')]
    outputs = legacy('batchnorm_fold2_grad_reduce').batchnorm_fold2_grad_reduce_compute(*arrays, desc)
    result = evaluate_graph(arrays, outputs, [dout, data])
    expected = [dout.sum((0, 2, 3), keepdims=blocked), (dout * data).sum((0, 2, 3), keepdims=blocked)]
    for actual, value in zip(result, expected):
        np.testing.assert_allclose(actual, value, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('symmetric,narrow,expected', [
    (False, False, [-1, -1, 0, 1, 2, 2]),
    (False, True, [-1.5, 0, 0, 0, 1.5, 1.5]),
    (True, False, [-8 / 3, 0, 0, 0, 4 / 3, 4 / 3]),
    (True, True, [-2, 0, 0, 0, 2, 2]),
])
def test_channel_quantization_preserves_distinct_ranges(symmetric, narrow, expected):
    """Distinct channel ranges detect broadcasting either range to the wrong channel."""
    data = np.array([[-4, -0.6, 0, 0.6, 1.6, 4], [-8, -1.2, 0, 1.2, 3.2, 8]], np.float32)
    actual, = evaluate(legacy('fake_quant_perchannel').fake_quant_perchannel_compute,
                        [data, [[-1], [-2]], [[2], [4]]], {}, int(narrow), 3, symmetric)
    np.testing.assert_allclose(actual, [expected, np.array(expected) * 2], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('negative_truncation', [False, True])
def test_channel_learned_scale_preserves_distinct_scales(monkeypatch, negative_truncation):
    """Separate channel scales survive the complete parameter-to-emission path."""
    data = np.array([[0, 0.25, 0.5, 1.5], [0, 0.5, 1, 3]], np.float32)
    arrays = [data, np.array([[1], [2]], np.float32), np.array([2], np.float32)]
    expected = [np.array([[0, 0.5, 0.5, 1], [0, 1, 1, 2]], np.float32)]
    calls = capture_emitted_graph(monkeypatch, arrays, expected)
    legacy('fake_learned_scale_quant_perchannel').fake_learned_scale_quant_perchannel(
        descriptor((2, 4)), descriptor((2,)), descriptor((1,)), descriptor((2, 4)),
        negative_truncation, 0, kernel_name='coverage_legacy')
    assert calls == ['coverage_legacy']


def test_batchnorm_zero_sample_count_is_rejected():
    """A zero sample count must raise before dividing the supplied statistics."""
    data = tvm.placeholder((0, 2, 1, 1), name='empty_batch', dtype='float32')
    statistic = tvm.placeholder((1, 2, 1, 1), name='statistic', dtype='float32')
    with pytest.raises(ValueError, match='num.*zero'):
        legacy('batchnorm_fold')._batchnorm_fold_compute(data, statistic, statistic, statistic, statistic, 0.9, 1e-5)


@pytest.mark.parametrize('module,helper', [
    ('fake_quant_perchannel', 'fake_quant_perchannel_param'),
    ('fake_quant_perchannel_grad', 'fake_quant_perchannel_grad_param'),
    ('fake_learned_scale_quant_perchannel', 'fake_learned_scale_quant_perchannel_param'),
    ('fake_learned_scale_quant_perchannel_grad', 'fake_learned_scale_quant_perchannel_grad_d_param'),
])
@pytest.mark.parametrize('blocked', [False, True])
def test_channel_parameter_layout_and_dense_weight_axis_remap(module, helper, blocked):
    """The Dense rewrite remaps channel axis zero to one; blocked statistics keep C0 storage."""
    x = descriptor((1, 1, 2, 2, 16) if blocked else (1, 3, 2, 1))
    x['ori_shape'] = (1, 16, 2, 2) if blocked else (1, 3, 2, 1)
    x['format'] = 'NC1HWC0' if blocked else 'NCHW'
    alpha = descriptor((1, 1, 1, 1, 16) if blocked else (3,))
    alpha['ori_shape'] = (16,) if blocked else (3,)
    max_desc = descriptor((1,)) if 'learned' in module else alpha
    result = getattr(legacy(module), helper)(x, alpha, max_desc, 0)
    if 'learned' in module:
        tensors = result
        assert tuple(int(dim) for dim in tensors[-2].shape) == ((1, 1, 1, 1, 16) if blocked else (1, 3, 1, 1))
    else:
        shape, statistic_shape, dtype = result
        assert tuple(shape) == x['shape']
        assert tuple(statistic_shape) == ((1, 1, 1, 1, 16) if blocked else (1, 3, 1, 1))
        assert dtype == 'float32'


@pytest.mark.parametrize('channel_axis', [None, 0, 1, 2])
def test_learned_scale_gradient_reduce_entry_real_scheduler(monkeypatch, channel_axis):
    """A layer scale sums every element; channel scales exclude exactly their channel axis."""
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 10
    if channel_axis is None:
        arrays, expected = [data.ravel()], [np.array([data.sum()], np.float32)]
        name = 'fake_learned_scale_quant_perlayer_grad_reduce'
        args = [descriptor(data.shape), descriptor((1,))]
    else:
        arrays = [data]
        expected = [data.sum(tuple(axis for axis in range(3) if axis != channel_axis))]
        name = 'fake_learned_scale_quant_perchannel_grad_reduce'
        args = [descriptor(data.shape), descriptor((data.shape[channel_axis],)), channel_axis]
    calls = capture_emitted_graph(monkeypatch, arrays, expected)
    entry_name = name.replace('_grad_reduce', '_grad_d_reduce')
    getattr(legacy(name), entry_name)(*args, kernel_name='coverage_legacy')
    assert calls == ['coverage_legacy']


@pytest.mark.parametrize('module', ['fake_learned_scale_quant_perlayer_grad',
                                   'fake_learned_scale_quant_perchannel_grad'])
@pytest.mark.parametrize('dtype', ['float16', 'float32'])
def test_learned_scale_sign_real_llvm_graph(module, dtype):
    """Both floating implementations preserve negative, zero and positive signs."""
    data = np.array([-3, -0.25, 0, 0.25, 3], dtype=dtype)
    tensor = tvm.placeholder(data.shape, name='signed_values', dtype=dtype)
    result = legacy(module)._sign_function(dtype, tensor)
    actual, = evaluate_graph([tensor], [result], [data])
    np.testing.assert_array_equal(actual, [-1, -1, 0, 1, 1])


@pytest.mark.parametrize('operation', ['batchnorm_fold', 'batchnorm_fold_grad', 'batchnorm_fold2',
                                      'batchnorm_fold2_grad', 'correction_mul_grad',
                                      'correction_mul_grad_reduce'])
def test_normalization_entries_reject_unsupported_layout_before_emission(monkeypatch, operation):
    """Layout mistakes fail at the Python boundary and never reach a compiler."""
    x = descriptor((2, 2, 2, 2))
    x['format'] = 'NHWC'
    stat = descriptor((2,))
    calls = []
    monkeypatch.setattr(te.lang.cce, 'cce_build_code', lambda *args: calls.append(args))
    if operation == 'batchnorm_fold':
        args = [x] + [stat] * 11
    elif operation == 'batchnorm_fold_grad':
        args = [stat, stat, x, stat, stat, x]
    elif operation == 'batchnorm_fold2':
        args = [x] + [stat] * 5 + [x]
    elif operation == 'batchnorm_fold2_grad':
        args = [x] + [stat] * 6 + [stat] * 3 + [x]
    elif operation == 'correction_mul_grad':
        args = [x, x, stat, stat, x, x, 1]
    else:
        args = [x, stat, 1]
    module = 'correction_mul_grad' if operation == 'correction_mul_grad_reduce' else operation
    with pytest.raises(RuntimeError, match='[Ff]ormat'):
        getattr(legacy(module), operation)(*args)
    assert not calls


@pytest.mark.parametrize('operation', ['batchnorm_fold', 'batchnorm_fold_grad'])
@pytest.mark.parametrize('case,reason', [('blocked_rank', '5D'), ('nchw_rank', '2D to 4D'),
                                      ('channels', 'second axis')])
def test_fold_rejects_invalid_statistics_layout(operation, case, reason):
    x, statistic = descriptor((2, 2, 2, 2)), descriptor((2,))
    if case == 'blocked_rank':
        x['format'] = 'NC1HWC0'
    elif case == 'nchw_rank':
        x = descriptor((2,))
    else:
        statistic = descriptor((3,))
    args = [x] + [statistic] * 11 if operation == 'batchnorm_fold' else [statistic, statistic, x, statistic, statistic, x]
    with pytest.raises(RuntimeError, match=reason):
        getattr(legacy(operation), operation)(*args)


def test_fold_rejects_unsupported_logical_data_format():
    desc = descriptor((2, 2, 2, 2))
    with pytest.raises(RuntimeError, match='data_format only support NCHW'):
        legacy('batchnorm_fold').batchnorm_fold(*([desc] * 12), data_format='NHWC')


@pytest.mark.parametrize('operation', ['batchnorm_fold2', 'batchnorm_fold2_grad'])
@pytest.mark.parametrize('case', ['dtype', 'original_format'])
def test_second_fold_rejects_invalid_dtype_or_original_format(operation, case):
    x, statistic = descriptor((2, 2, 2, 2)), descriptor((2,))
    x.update({'dtype': 'int32'} if case == 'dtype' else {'ori_format': 'NHWC'})
    args = [x] + [statistic] * 5 + [x] if operation == 'batchnorm_fold2' else [x] + [statistic] * 9 + [x]
    with pytest.raises(RuntimeError, match='Dtype' if case == 'dtype' else 'ori_format'):
        getattr(legacy(operation), operation)(*args)


def test_correction_mul_rejects_integer_storage():
    with pytest.raises(RuntimeError, match='Dtype'):
        legacy('correction_mul').correction_mul(descriptor((2, 2, 2, 2), 'int32'),
                                                descriptor((2,)), descriptor((2,)), descriptor((2, 2, 2, 2)), 1)


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize('scheduler', ['llvm_boundary', pytest.param('cce', marks=pytest.mark.xfail(
    strict=True, raises=RuntimeError, reason='CANN CCE auto_schedule has no tuple_reduce_sum template'))])
def test_second_fold_gradient_reduce_entry_real_scheduler(monkeypatch, dtype, scheduler):
    """Blocked reduction promotes fp16 inputs before multiplying and accumulating in fp32."""
    shape = (2, 1, 2, 2, 16)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape).astype(dtype) / 16
    dout = data + np.array(0.5, dtype=dtype)
    expected = [dout.astype(np.float32).sum((0, 2, 3), keepdims=True),
                (dout.astype(np.float32) * data.astype(np.float32)).sum((0, 2, 3), keepdims=True)]
    calls, schedules = [], []
    module = legacy('batchnorm_fold2_grad_reduce')
    if scheduler == 'llvm_boundary':
        # This SDK lacks the CCE tuple-sum template. Verify the Python scheduling/emission protocol
        # with a real host schedule; the separate CCE case preserves the native capability failure.
        def host_schedule(outputs):
            assert len(outputs) == 2
            assert all(str(item.dtype) == 'float32' for item in outputs)
            schedule = tvm.create_schedule([item.op for item in outputs])
            schedules.append(schedule)
            return schedule
        monkeypatch.setattr(module.generic, 'auto_schedule', host_schedule)
    def compile_on_host(schedule, config):
        assert schedule is not None and config['name'] == 'coverage_legacy'
        if scheduler == 'llvm_boundary':
            assert schedules == [schedule]
        tensors = config['tensor_list']
        assert len(tensors) == 4
        assert all(str(item.dtype) == dtype and tuple(int(dim) for dim in item.shape) == shape
                   for item in tensors[:2])
        for item, value in zip(tensors[2:], expected):
            assert str(item.dtype) == 'float32'
            assert tuple(int(dim) for dim in item.shape) == value.shape
        actual = evaluate_graph(tensors[:2], tensors[2:], [dout, data])
        for result, value in zip(actual, expected):
            np.testing.assert_allclose(result, value, rtol=1e-6, atol=1e-6)
        calls.append(True)
    monkeypatch.setattr(te.lang.cce, 'cce_build_code', compile_on_host)
    desc = descriptor(shape, dtype)
    desc['format'] = 'NC1HWC0'
    try:
        module.batchnorm_fold2_grad_reduce(
            desc, desc, descriptor(expected[0].shape), descriptor(expected[1].shape), kernel_name='coverage_legacy')
    except RuntimeError as error:
        assert scheduler == 'cce' and 'tuple_reduce_sum not support' in str(error)
        raise
    assert calls == [True]


@pytest.mark.parametrize('narrow,expected', [(False, [0, 2, 3, 4, 0, 0]), (True, [0, 0, 3, 4, 0, 0])])
def test_symmetric_channel_quantization_gradient_entry(monkeypatch, narrow, expected):
    """Signed quantization ranges and narrow-range clipping must retain the correct sensitivity mask."""
    data = np.array([[-2, -1.1, 0, .5, 1.1, 2]], np.float32)
    dout = np.arange(1, 7, dtype=np.float32).reshape(1, 6)
    arrays = [dout, data, np.array([[-1]], np.float32), np.array([[1]], np.float32)]
    calls = capture_emitted_graph(monkeypatch, arrays, [np.array([expected], np.float32)])
    x_desc, stat = descriptor((1, 6)), descriptor((1,))
    legacy('fake_quant_perchannel_grad').fake_quant_perchannel_grad(
        x_desc, x_desc, stat, stat, x_desc, True, narrow, 2, 0, kernel_name='coverage_legacy')
    assert calls == ['coverage_legacy']
