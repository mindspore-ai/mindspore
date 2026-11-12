# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Real CPU numerical and validation contracts for remaining NN/image functions."""
# Unit coverage intentionally exercises internal Python inference and wrapper helpers.
# pylint: disable=protected-access
import itertools
import os
import importlib
import random
import numpy as np
import pytest
import mindspore as ms
from mindspore import dataset
from mindspore.ops.function import random_func
from mindspore.ops.function import nn_func as nn
from mindspore.ops.function import image_func as image
from tests.mark_utils import arg_mark


@pytest.fixture(scope='module', autouse=True, name='cpu_context')
def fixture_cpu_context():
    previous = {key: ms.get_context(key) for key in ('mode', 'device_target')}
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
    yield
    ms.set_context(**previous)


@pytest.fixture(autouse=True, name='preserve_random_state')
def fixture_preserve_random_state(cpu_context):
    """Pool/dropout helpers advance both generator state and per-operation seed counters."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    dataset_seed = dataset.config.get_seed()
    generator_state, global_seed = ms.get_rng_state(), ms.get_seed()
    manual_seed = random_func.default_generator._generator.attrs['manual_seed']
    seed_module = importlib.import_module('mindspore.common.seed')
    op_seeds = seed_module._KERNEL_SEED.copy()
    yield
    ms.set_rng_state(generator_state)
    random_func.default_generator._generator.add_prim_attr('manual_seed', manual_seed)
    dataset.config.set_seed(dataset_seed)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    seed_module._GLOBAL_SEED = global_seed
    seed_module._KERNEL_SEED.clear()
    seed_module._KERNEL_SEED.update(op_seeds)


def tensor(value, dtype=ms.float32):
    return ms.Tensor(value, dtype)


def adaptive_reference(array, size, maximum=False):
    """Bin boundaries follow the mathematical floor/ceil adaptive definition."""
    spatial = array.shape[-len(size):]
    output = np.empty(array.shape[:-len(size)] + tuple(size), array.dtype)
    for position in np.ndindex(*size):
        bins = tuple(slice(int(np.floor(i * n / m)), int(np.ceil((i + 1) * n / m)))
                     for i, n, m in zip(position, spatial, size))
        block = array[(Ellipsis,) + bins]
        axes = tuple(range(block.ndim - len(size), block.ndim))
        output[(Ellipsis,) + position] = block.max(axis=axes) if maximum else block.mean(axis=axes)
    return output


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('maximum', [False, True])
def test_adaptive_pooling_numpy_bins(dim, maximum):
    """
    Feature: Adaptive pooling numpy bins.
    Description: Check adaptive pooling numpy bins.
    Expectation: Adaptive average and maximum pooling match independently calculated NumPy bins for each spatial
        rank.
    """
    a = np.arange(4 ** dim, dtype=np.float32).reshape((1, 1) + (4,) * dim)
    fn = getattr(nn, 'adaptive_' + ('max' if maximum else 'avg') + '_pool' + str(dim) + 'd')
    result = fn(tensor(a), 2 if dim == 1 else (2,) * dim)
    if isinstance(result, tuple):
        result = result[0]
    np.testing.assert_allclose(result.asnumpy(), adaptive_reference(a, (2,) * dim, maximum))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('maximum', [False, True])
@pytest.mark.parametrize('case,error,reason', [
    ('rank', ValueError, '3 dim'), ('small', ValueError, 'greater or equal'),
    ('indivisible', ValueError, 'divisible'), ('dtype', TypeError, 'dtype'),
    ('output_zero', ValueError, 'output_size'), ('output_float', TypeError, 'output_size')])
def test_adaptive_1d_rejects_invalid_contract(maximum, case, error, reason):
    """
    Feature: Adaptive 1d rejects invalid contract.
    Description: Check adaptive 1d rejects invalid contract.
    Expectation: Invalid adaptive 1d contract raise the parameterized exception type.
    """
    fn = nn.adaptive_max_pool1d if maximum else nn.adaptive_avg_pool1d
    shape, dtype, size = (1, 1, 6), ms.float32, 2
    if case == 'rank':
        shape = (1, 6)
    elif case == 'small':
        size = 7
    elif case == 'indivisible':
        size = 4
    elif case == 'dtype':
        dtype = ms.int32
    elif case == 'output_zero':
        size = 0
    elif case == 'output_float':
        size = 2.0
    with pytest.raises(error, match=reason):
        fn(tensor(np.ones(shape), dtype), size)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('include_pad', [False, True])
def test_avg_pool_explicit_tuple_padding(dim, include_pad):
    """
    Feature: Avg pool explicit tuple padding.
    Description: Check avg pool explicit tuple padding.
    Expectation: Explicit tuple padding produces the independently summed average-pooling windows within the
        stated tolerance.
    """
    a = np.arange(1, 1 + 3 ** dim, dtype=np.float32).reshape((1, 1) + (3,) * dim)
    padding = (1,) if dim == 1 else (1,) * (2 * dim)
    kwargs = {'kernel_size': 2, 'stride': (1,) * dim, 'padding': padding, 'count_include_pad': include_pad}
    if dim in (1, 3):
        kwargs['stride'] = 1
    result = getattr(nn, 'avg_pool' + str(dim) + 'd')(tensor(a), **kwargs)
    out = np.empty((1, 1) + (4,) * dim, np.float32)
    for p in np.ndindex(*(4,) * dim):
        values = []
        for offset in itertools.product(range(2), repeat=dim):
            index = tuple(x + y - 1 for x, y in zip(p, offset))
            if all(0 <= x < 3 for x in index):
                values.append(a[(0, 0) + index])
            elif include_pad:
                values.append(0)
        out[(0, 0) + p] = np.mean(values)
    np.testing.assert_allclose(result.asnumpy(), out, atol=1e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dim', [1, 2, 3])
def test_avg_pool_rejects_non_tensor(dim):
    """
    Feature: Avg pool rejects non tensor.
    Description: Check avg pool rejects non tensor.
    Expectation: Invalid avg pool non tensor raise TypeError. Diagnostics identify tensor.
    """
    with pytest.raises(TypeError, match='tensor'):
        getattr(nn, 'avg_pool' + str(dim) + 'd')([1., 2.], 2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('fn,value,expected', [
    ('_check_avgpool_2d_kernel_size', 2, (1, 2, 2)),
    ('_check_avgpool_2d_kernel_size', (2, 3), (1, 2, 3)),
    ('_check_avgpool_2d_stride', (2, 3), (1, 2, 3)),
    ('_check_avgpool_2d_padding', (1, 2, 3, 4), (0, 0, 1, 2, 3, 4)),
    ('_check_avg_pool3d_padding', (0, 1, 2, 3, 4, 5), None)])
def test_average_pool_parameter_normalization(fn, value, expected):
    """
    Feature: Average pool parameter normalization.
    Description: Check average pool parameter normalization.
    Expectation: Scalar and tuple pooling parameters normalize to their independently specified dimensional
        tuples.
    """
    assert getattr(nn, fn)(value) == expected


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('fn', ['_check_avgpool_2d_kernel_size', '_check_avgpool_2d_stride',
                               '_check_avgpool_2d_padding', '_check_avg_pool3d_padding'])
@pytest.mark.parametrize('value,error', [('bad', TypeError), ((1,), ValueError)])
def test_average_pool_invalid_parameter(fn, value, error):
    """
    Feature: Average pool invalid parameter.
    Description: Check average pool invalid parameter.
    Expectation: Invalid average pool parameter raise the parameterized exception type. Diagnostics identify
        should be int or tuple.
    """
    with pytest.raises(error, match='should be int or tuple'):
        getattr(nn, fn)(value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kwargs,error,reason', [
    ({'padding': ()}, ValueError, 'padding'), ({'padding': []}, TypeError, 'padding'),
    pytest.param({'stride': (1, 2)}, ValueError, 'stride', marks=pytest.mark.xfail(
        strict=True, raises=TypeError, reason='stride type guard runs before documented tuple length validation'))])
def test_average_pool1d_invalid_padding_stride(kwargs, error, reason):
    """
    Feature: Average pool1d invalid padding stride.
    Description: Check average pool1d invalid padding stride.
    Expectation: Invalid average pool1d padding stride raise the parameterized exception type.
    """
    with pytest.raises(error, match=reason):
        nn.avg_pool1d(tensor(np.ones((1, 1, 4))), 2, **kwargs)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('explicit', ['none', 'empty', 'spatial', 'full'])
def test_unpool_scatter_matches_literal_indices(dim, batched, explicit):
    """
    Feature: Unpool scatter matches literal indices.
    Description: Check unpool scatter matches literal indices.
    Expectation: Unpooling scatters each value into its literal flattened output index and leaves other entries
        zero.
    """
    shape = ((1, 1) if batched else (1,)) + (2,) * dim
    values = np.arange(1, 1 + 2 ** dim, dtype=np.float32).reshape(shape)
    positions = np.arange(0, 2 * 2 ** dim, 2, dtype=np.int64).reshape(shape)
    size = (4,) * dim
    output_size = {'none': None, 'empty': (), 'spatial': size, 'full': shape[:-dim] + size}[explicit]
    kwargs = {'kernel_size': (2,) * dim, 'padding': (0,) * dim, 'output_size': output_size}
    result = getattr(nn, 'max_unpool' + str(dim) + 'd')(tensor(values), tensor(positions, ms.int64), **kwargs)
    expected = np.zeros(shape[:-dim] + size, np.float32)
    expected.reshape(-1)[positions.reshape(-1)] = values.reshape(-1)
    np.testing.assert_array_equal(result.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dim,value', [(1, []), (1, (1, 2, 3, 4)), (3, []), (3, (1, 2))])
def test_unpool_rejects_output_size_container_and_rank(dim, value):
    """
    Feature: Unpool rejects output size container and rank.
    Description: Check unpool rejects output size container and rank.
    Expectation: Invalid unpool output size container and rank raise ValueError. Diagnostics identify
        output_size.
    """
    shape = (1, 1) + (2,) * dim
    with pytest.raises(ValueError, match='output_size'):
        getattr(nn, 'max_unpool' + str(dim) + 'd')(tensor(np.ones(shape)),
                                                tensor(np.zeros(shape), ms.int64), 2, output_size=value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('return_indices', [False, True])
def test_max_pool_values_and_flattened_argmax(dim, return_indices):
    """
    Feature: Max pool values and flattened argmax.
    Description: Check max pool values and flattened argmax.
    Expectation: Max pooling returns the expected maxima and, when requested, their flattened int64 indices.
    """
    a = np.arange(4 ** dim, dtype=np.float32).reshape((1, 1) + (4,) * dim)
    result = getattr(nn, 'max_pool' + str(dim) + 'd')(tensor(a), 2, return_indices=return_indices)
    expected = adaptive_reference(a, (2,) * dim, True)
    if return_indices:
        np.testing.assert_array_equal(result[0].asnumpy(), expected)
        np.testing.assert_array_equal(result[1].asnumpy(), expected.astype(np.int64))
    else:
        np.testing.assert_array_equal(result.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('name,shape', [('dropout', (2, 3)), ('dropout1d', (3, 4)),
                                     ('dropout1d', (2, 3, 4)), ('dropout2d', (2, 3, 2, 2)),
                                     ('dropout3d', (2, 3, 2, 2, 2))])
@pytest.mark.parametrize('p,training', [(0., True), (0.5, False)])
def test_dropout_endpoints_and_inference(name, shape, p, training):
    """
    Feature: Dropout endpoints and inference.
    Description: Check dropout endpoints and inference.
    Expectation: Zero dropout and inference retain input values, full training dropout returns zeros, and the
        identity path returns the original tensor.
    """
    a = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
    x = tensor(a)
    result = getattr(nn, name)(x, p=p, training=training)
    np.testing.assert_array_equal(result.asnumpy(), np.zeros_like(a) if p == 1 and training else a)
    if not training:
        assert result is x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('case,error,reason', [('p_type', TypeError, 'p'), ('p_range', ValueError, 'p'),
                                            ('input', TypeError, 'input'), ('rank', ValueError, 'shape'),
                                            ('training', TypeError, 'training')])
def test_dropout1d_invalid_arguments(case, error, reason):
    """
    Feature: Dropout1d invalid arguments.
    Description: Check dropout1d invalid arguments.
    Expectation: Invalid dropout1d arguments raise the parameterized exception type.
    """
    args = {'input': tensor(np.ones((2, 3))), 'p': 0.5, 'training': True}
    args.update({'p_type': {'p': 1}, 'p_range': {'p': -0.1}, 'input': {'input': []},
                 'rank': {'input': tensor([1., 2.])}, 'training': {'training': 1}}[case])
    with pytest.raises(error, match=reason):
        nn.dropout1d(**args)


def multilinear_reference(a, size, align):
    """Independent separable linear interpolation with clamped pixel coordinates."""
    output = a.astype(np.float64)
    for axis, new_size in enumerate(size, start=2):
        old_size = output.shape[axis]
        position = (np.linspace(0, old_size - 1, new_size) if align else
                    (np.arange(new_size) + .5) * old_size / new_size - .5)
        position = np.maximum(position, 0)
        left = np.minimum(np.floor(position).astype(int), old_size - 1)
        right = np.minimum(left + 1, old_size - 1)
        weight_shape = [1] * output.ndim
        weight_shape[axis] = new_size
        weight = (position - left).reshape(weight_shape)
        output = np.take(output, left, axis=axis) * (1 - weight) + np.take(output, right, axis=axis) * weight
    return output


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dim,mode', [(1, 'nearest'), (2, 'nearest'), (3, 'nearest'),
                                    (1, 'nearest-exact'), (2, 'nearest-exact')])
@pytest.mark.parametrize('size_kind', ['tuple', 'scalar', 'list'])
def test_interpolate_nearest_coordinate_selection(dim, mode, size_kind):
    """
    Feature: Interpolate nearest coordinate selection.
    Description: Check interpolate nearest coordinate selection.
    Expectation: Nearest and nearest-exact modes select the explicit expected source coordinates for each output
        size.
    """
    a = np.arange(3 ** dim, dtype=np.float32).reshape((1, 1) + (3,) * dim)
    size = {'tuple': (5,) * dim, 'scalar': 5, 'list': [5] * dim}[size_kind]
    expected = a
    coords = np.floor((np.arange(5) + (.5 if mode == 'nearest-exact' else 0)) * 3 / 5).astype(int)
    for axis in range(2, 2 + dim):
        expected = np.take(expected, coords, axis=axis)
    np.testing.assert_array_equal(nn.interpolate(tensor(a), size=size, mode=mode).asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dim,mode', [(1, 'linear'), (2, 'bilinear'), (3, 'trilinear')])
@pytest.mark.parametrize('align', [False, True])
def test_interpolate_linear_separable_numpy(dim, mode, align):
    """
    Feature: Interpolate linear separable numpy.
    Description: Check interpolate linear separable numpy.
    Expectation: Linear interpolation in each spatial rank matches the independent separable NumPy reference for
        both corner policies.
    """
    a = np.arange(3 ** dim, dtype=np.float32).reshape((1, 1) + (3,) * dim)
    result = nn.interpolate(tensor(a), size=(5,) * dim, mode=mode, align_corners=align)
    np.testing.assert_allclose(result.asnumpy(), multilinear_reference(a, (5,) * dim, align), atol=3e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('scale', [False, True])
def test_interpolate_area_uses_mean_bins(dim, scale):
    """
    Feature: Interpolate area uses mean bins.
    Description: Check interpolate area uses mean bins.
    Expectation: Area interpolation equals the independent adaptive mean of each output bin.
    """
    a = np.arange(4 ** dim, dtype=np.float32).reshape((1, 1) + (4,) * dim)
    kwargs = {'scale_factor': [0.5] * dim} if scale else {'size': (2,) * dim}
    result = nn.interpolate(tensor(a), mode='area', **kwargs)
    np.testing.assert_allclose(result.asnumpy(), adaptive_reference(a, (2,) * dim))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('align', [False, True])
def test_interpolate_bicubic_preserves_constant_channels(align):
    """
    Feature: Interpolate bicubic preserves constant channels.
    Description: Check interpolate bicubic preserves constant channels.
    Expectation: Bicubic resizing preserves each constant channel across the complete requested output shape.
    """
    # NumPy dynamically supplies these ndarray/array-function signatures.
    # pylint: disable=too-many-function-args
    a = np.broadcast_to(np.array([2., -3.], np.float32).reshape(1, 2, 1, 1), (1, 2, 3, 3)).copy()
    result = nn.interpolate(tensor(a), size=(5, 4), mode='bicubic', align_corners=align)
    np.testing.assert_allclose(result.asnumpy(), np.broadcast_to(a[:, :, :1, :1], (1, 2, 5, 4)), atol=2e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode,dim,recompute', [('nearest', 1, True), ('nearest', 2, True),
                                             ('nearest', 3, False), ('trilinear', 3, False)])
def test_interpolate_scale_factor_matches_integer_repeat(mode, dim, recompute):
    """
    Feature: Interpolate scale factor matches integer repeat.
    Description: Check interpolate scale factor matches integer repeat.
    Expectation: Integral nearest-neighbor scale factors reproduce the expected repeated samples.
    """
    a = np.arange(2 ** dim, dtype=np.float32).reshape((1, 1) + (2,) * dim)
    kwargs = {'scale_factor': 2., 'mode': mode, 'recompute_scale_factor': recompute}
    result = nn.interpolate(tensor(a), **kwargs)
    expected = a
    if mode == 'nearest':
        for axis in range(2, 2 + dim):
            expected = np.repeat(expected, 2, axis=axis)
    else:
        expected = multilinear_reference(a, (4,) * dim, False)
    np.testing.assert_allclose(result.asnumpy(), expected, atol=1e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kwargs,error,reason', [
    ({'input': []}, TypeError, 'tensor'), ({'size': None}, ValueError, 'both empty'),
    ({'scale_factor': 2.}, ValueError, 'simultaneously'), ({'size': (2,)}, ValueError, 'spatial dimensions'),
    ({'size': None, 'scale_factor': (2.,)}, ValueError, 'spatial dimensions'),
    ({'mode': []}, ValueError, 'mode'), ({'mode': 'linear'}, ValueError, 'support'),
    ({'size': None, 'scale_factor': 2.}, ValueError, 'scale_factor.*cannot'),
    ({'align_corners': False}, ValueError, 'align_corners'),
    ({'recompute_scale_factor': True}, ValueError, 'explicit.*size')])
def test_interpolate_rejects_conflicting_options(kwargs, error, reason):
    """
    Feature: Interpolate rejects conflicting options.
    Description: Check interpolate rejects conflicting options.
    Expectation: Invalid interpolate conflicting options raise the parameterized exception type.
    """
    args = {'input': tensor(np.ones((1, 1, 2, 2))), 'size': (4, 4), 'mode': 'nearest'}
    args.update(kwargs)
    with pytest.raises(error, match=reason):
        nn.interpolate(**args)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('rank,value,expected', [(2, 3, (3, 3)), (3, .5, (.5, .5, .5))])
def test_interpolate_dynamic_tuple_builder(rank, value, expected):
    """
    Feature: Interpolate dynamic tuple builder.
    Description: Check interpolate dynamic tuple builder.
    Expectation: Dynamic interpolation parameters expand into the expected tuple for the given rank.
    """
    assert nn._interploate_make_tuple(rank, value) == expected


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_interpolate_scale_size_conversion_floors_fractional_size():
    """
    Feature: Interpolate scale size conversion floors fractional size.
    Description: Check interpolate scale size conversion floors fractional size.
    Expectation: Fractional spatial scale factors floor the derived output size to (7, 3).
    """
    assert nn._interpolate_scale_factor_convert_size((2, 3, 5, 7), (1.5, .5)) == (7, 3)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean', 'batchmean'])
def test_kl_div_reductions_follow_numpy_definition(reduction):
    """
    Feature: Kl div reductions follow numpy definition.
    Description: Check kl div reductions follow numpy definition.
    Expectation: Each KL reduction agrees with the independently calculated elementwise divergence and reduction
        formula.
    """
    logits = np.log(np.array([[.2, .8], [.6, .4]], np.float32))
    labels = np.array([[.3, .7], [0., 1.]], np.float32)
    expected = np.zeros_like(labels)
    positive = labels > 0
    expected[positive] = labels[positive] * (np.log(labels[positive]) - logits[positive])
    if reduction == 'sum':
        expected = expected.sum()
    elif reduction == 'mean':
        expected = expected.mean()
    elif reduction == 'batchmean':
        expected = expected.sum() / 2
    np.testing.assert_allclose(nn.kl_div(tensor(logits), tensor(labels), reduction).asnumpy(), expected, atol=2e-7)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_kl_div_non_string_reduction_is_rejected():
    """
    Feature: Kl div non string reduction is rejected.
    Description: Check kl div non string reduction is rejected.
    Expectation: Invalid kl div non string reduction is rejected raise ValueError. Diagnostics identify
        reduction.*must be str.
    """
    with pytest.raises(ValueError, match='reduction.*must be str'):
        nn.kl_div(tensor([1.]), tensor([1.]), 0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('method', ['bilinear', 'nearest'])
def test_crop_resize_full_and_extrapolated_images(method):
    """
    Feature: Crop resize full and extrapolated images.
    Description: Check crop resize full and extrapolated images.
    Expectation: A full-image crop preserves the image, while an out-of-bounds crop fills every element with the
        extrapolation value.
    """
    a = np.array([[[[1.], [3.]], [[5.], [7.]]]], np.float32)
    boxes = tensor([[0., 0., 1., 1.], [-2., -2., -1., -1.]])
    result = image.crop_and_resize(tensor(a), boxes, tensor([0, 0], ms.int32), (2, 2),
                                   method=method, extrapolation_value=-9.)
    np.testing.assert_array_equal(result.asnumpy(), np.concatenate([a, np.full_like(a, -9.)]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('case,error,reason', [
    ('image_type', TypeError, 'image.*tensor'), ('boxes_type', TypeError, 'boxes.*tensor'),
    ('indices_type', TypeError, 'box_indices.*tensor'), ('size_type', TypeError, 'tuple'),
    ('size_length', ValueError, 'length must be 2'), ('image_rank', ValueError, '4D'),
    ('boxes_dtype', TypeError, 'boxes must be'), ('boxes_rank', ValueError, '2D'),
    ('boxes_width', ValueError, 'second-dim'), ('indices_dtype', TypeError, 'box_indices must be'),
    ('indices_rank', ValueError, '1D'), ('indices_count', ValueError, 'first dim'),
    ('size_float', TypeError, 'value must be int'), ('size_zero', ValueError, 'positive')])
def test_crop_resize_rejects_each_invalid_argument(case, error, reason):
    """
    Feature: Crop resize rejects each invalid argument.
    Description: Check crop resize rejects each invalid argument.
    Expectation: Invalid crop resize each argument raise the parameterized exception type.
    """
    args = {'image': tensor(np.ones((1, 2, 2, 1))), 'boxes': tensor(
        [[0., 0., 1., 1.]]), 'box_indices': tensor([0], ms.int32), 'crop_size': (2, 2)}
    args.update({'image_type': {'image': []}, 'boxes_type': {'boxes': []},
                 'indices_type': {'box_indices': []}, 'size_type': {'crop_size': [2, 2]},
                 'size_length': {'crop_size': (2,)}, 'image_rank': {'image': tensor(np.ones((2, 2, 1)))},
                 'boxes_dtype': {'boxes': tensor([[0., 0., 1., 1.]], ms.float64)},
                 'boxes_rank': {'boxes': tensor([0., 0., 1., 1.])},
                 'boxes_width': {'boxes': tensor([[0., 1.]])},
                 'indices_dtype': {'box_indices': tensor([0], ms.int64)},
                 'indices_rank': {'box_indices': tensor([[0]], ms.int32)},
                 'indices_count': {'box_indices': tensor([0, 0], ms.int32)},
                 'size_float': {'crop_size': (2., 2)}, 'size_zero': {'crop_size': (2, 0)}}[case])
    with pytest.raises(error, match=reason):
        image.crop_and_resize(**args)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('return_indices', [False, True])
@pytest.mark.parametrize('ratio', [None, .5, 'tuple'])
def test_fractional_pool_single_window_exact_maximum(dim, batched, return_indices, ratio):
    """
    Feature: Fractional pool single window exact maximum.
    Description: Check fractional pool single window exact maximum.
    Expectation: A single fractional pooling window returns the literal maximum and, when requested, its
        flattened index.
    """
    extent = 3 if dim == 3 else 2
    shape = ((1, 1) if batched else (1,)) + (extent,) * dim
    a = np.arange(extent ** dim, dtype=np.float32).reshape(shape)
    kwargs = {'kernel_size': 2, 'return_indices': return_indices}
    if ratio is None:
        kwargs['output_size'] = (1,) * dim
        kwargs['_random_samples'] = tensor(np.full((1, 1, dim), .3))
    else:
        kwargs['output_ratio'] = (.5,) * dim if ratio == 'tuple' else ratio
    result = getattr(nn, 'fractional_max_pool' + str(dim) + 'd')(tensor(a), **kwargs)
    expected = np.full(shape[:-dim] + (1,) * dim, extent ** dim - 1, np.float32)
    if return_indices:
        np.testing.assert_array_equal(result[0].asnumpy(), expected)
        np.testing.assert_array_equal(result[1].asnumpy(), expected.astype(np.int64))
    else:
        np.testing.assert_array_equal(result.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('kwargs,error,reason', [
    ({}, ValueError, 'output_size.*output_ratio'),
    ({'output_ratio': 1}, TypeError, 'output_ratio'),
    ({'output_ratio': (0., 0., 0.)}, ValueError, 'output_ratio'),
    ({'output_size': 1, 'return_indices': 1}, TypeError, 'return_indices')])
def test_fractional_pool_invalid_configuration(dim, kwargs, error, reason):
    """
    Feature: Fractional pool invalid configuration.
    Description: Check fractional pool invalid configuration.
    Expectation: Invalid fractional pool configuration raise the parameterized exception type.
    """
    with pytest.raises(error, match=reason):
        getattr(nn, 'fractional_max_pool' + str(dim) + 'd')(tensor(np.ones((1, 1) + (2,) * dim)),
                                                        kernel_size=2, **kwargs)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_fractional_pool3d_rejects_mismatched_random_sample_dtype():
    """
    Feature: Fractional pool3d rejects mismatched random sample dtype.
    Description: Check fractional pool3d rejects mismatched random sample dtype.
    Expectation: Invalid fractional pool3d mismatched random sample dtype raise TypeError. Diagnostics identify
        same dtype.
    """
    with pytest.raises(TypeError, match='same dtype'):
        nn.fractional_max_pool3d(tensor(np.ones((1, 1, 2, 2, 2))), 2, output_size=1,
                                _random_samples=tensor([[[.2, .4, .6]]], ms.float64))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('rank', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
@pytest.mark.parametrize('weighted', [False, True])
def test_nll_loss_nd_weighted_smoothed_numpy(rank, reduction, weighted):
    """
    Feature: Nll loss nd weighted smoothed numpy.
    Description: Check nll loss nd weighted smoothed numpy.
    Expectation: N-dimensional NLL values, weighting, smoothing, ignored labels, and reduction match the
        independent NumPy loss.
    """
    shape = (3,) if rank == 1 else (2, 3) + (2,) * (rank - 2)
    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 10 - 3
    axis = 0 if rank == 1 else 1
    target_shape = shape[:axis] + shape[axis + 1:]
    target = np.arange(np.prod(target_shape), dtype=np.int32).reshape(target_shape) % 3
    if rank != 1:
        target.reshape(-1)[0] = -100
    weight = np.array([1., 2., .5], np.float32) if weighted else np.ones(3, np.float32)
    valid = target != -100
    safe_target = np.where(valid, target, 0)
    selected = np.take_along_axis(a, np.expand_dims(safe_target, axis), axis=axis).squeeze(axis)
    broadcast_shape = [1] * rank
    broadcast_shape[axis] = 3
    smooth = -(a * weight.reshape(broadcast_shape)).sum(axis=axis) / 3
    expected = np.where(valid, -.8 * selected * weight[safe_target] + .2 * smooth, 0)
    if reduction == 'sum':
        expected = expected.sum()
    elif reduction == 'mean':
        expected = expected.sum() / np.where(valid, weight[safe_target], 0).sum()
    result = nn.nll_loss(tensor(a), tensor(target, ms.int32), weight=tensor(weight) if weighted else None,
                         reduction=reduction, label_smoothing=.2)
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=2e-6, atol=5e-7)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('rank', [1, 2, 4])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
def test_cross_entropy_probability_targets_numpy(rank, weighted, reduction):
    """
    Feature: Cross entropy probability targets numpy.
    Description: Check cross entropy probability targets numpy.
    Expectation: Probability-target cross entropy agrees with the independently calculated weighted and smoothed
        NumPy expression.
    """
    shape = (3,) if rank == 1 else (2, 3) + (2,) * (rank - 2)
    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 10 - 1
    axis = 0 if rank == 1 else 1
    probability = np.broadcast_to(np.array([.2, .3, .5], np.float32).reshape(
        (3,) if rank == 1 else (1, 3) + (1,) * (rank - 2)), shape).copy()
    weight = np.array([1., .5, 2.], np.float32)
    log_probability = a - np.log(np.exp(a).sum(axis=axis, keepdims=True))
    smoothed = probability * .8 + .2 / 3
    wshape = (3,) if rank == 1 else (1, 3) + (1,) * (rank - 2)
    weighted_log = log_probability * (weight.reshape(wshape) if weighted else 1)
    expected = -(weighted_log * smoothed).sum(axis=axis)
    if reduction == 'sum':
        expected = expected.sum()
    elif reduction == 'mean':
        expected = expected.mean()
    result = nn.cross_entropy(tensor(a), tensor(probability), weight=tensor(weight) if weighted else None,
                              reduction=reduction, label_smoothing=.2)
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=2e-6, atol=2e-7)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
def test_huber_and_l1_loss_broadcasting_numpy(reduction):
    """
    Feature: Huber and l1 loss broadcasting numpy.
    Description: Check huber and l1 loss broadcasting numpy.
    Expectation: Broadcasted Huber and L1 losses agree with their independent piecewise NumPy definitions and
        selected reductions.
    """
    a = np.array([[-3., -.2, .5], [2., 4., 7.]], np.float32)
    target = np.array([[0., 1., 2.]], np.float32)
    diff = np.abs(a - target)
    expected = np.where(diff < 2, .5 * diff ** 2, 2 * (diff - 1))
    l1 = diff
    if reduction == 'mean':
        expected, l1 = expected.mean(), l1.mean()
    elif reduction == 'sum':
        expected, l1 = expected.sum(), l1.sum()
    np.testing.assert_allclose(nn.huber_loss(tensor(a), tensor(target), reduction, delta=2).asnumpy(), expected)
    np.testing.assert_allclose(nn.l1_loss(tensor(a), tensor(target), reduction).asnumpy(), l1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('p', [1, 2])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
def test_multi_margin_loss_numpy_class_pairs(p, weighted, reduction):
    """
    Feature: Multi margin loss numpy class pairs.
    Description: Check multi margin loss numpy class pairs.
    Expectation: Class-pair margins, p powers, weights, and reductions match the explicit NumPy loss formula.
    """
    a = np.array([[.1, 1.2, -.2], [1.3, .4, .7]], np.float32)
    target = np.array([1, 0], np.int64)
    weight = np.array([.5, 1.5, 2.], np.float32)
    expected = []
    for row, index in zip(a, target):
        value = sum(max(0., 2 - row[index] + x) ** p for j, x in enumerate(row) if j != index) / 3
        expected.append(value * (weight[index] if weighted else 1))
    expected = np.array(expected)
    if reduction == 'mean':
        expected = expected.mean()
    elif reduction == 'sum':
        expected = expected.sum()
    result = nn.multi_margin_loss(tensor(a), tensor(target, ms.int64), p=p, margin=2,
                                  weight=tensor(weight) if weighted else None, reduction=reduction)
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=2e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('p', [1, 2])
@pytest.mark.parametrize('swap', [False, True])
@pytest.mark.parametrize('tensor_margin', [False, True])
def test_triplet_margin_native_boundary_protocol(p, swap, tensor_margin, monkeypatch):
    """
    Feature: Triplet margin native boundary protocol.
    Description: CPU has no TripletMarginLoss kernel; verify construction/forwarding only.
    Expectation: CPU has no TripletMarginLoss kernel; verify construction/forwarding only.
    """
    anchor = tensor([[.3, .7], [.5, .5]])
    positive = tensor([[.4, .6], [.4, .6]])
    negative = tensor([[.2, .9], [.3, .7]])
    margin = tensor(.8) if tensor_margin else .8
    sentinel = object()
    calls = []

    def native_boundary(primitive, a, pos, neg, actual_margin):
        assert primitive.get_attr_dict()['p'] == p
        assert primitive.get_attr_dict()['eps'] == 1e-6
        assert primitive.get_attr_dict()['swap'] is swap
        assert primitive.get_attr_dict()['reduction'] == 'none'
        assert a is anchor and pos is positive and neg is negative
        assert actual_margin.dtype == ms.float32 and actual_margin.shape == ()
        np.testing.assert_allclose(actual_margin.asnumpy(), .8)
        if tensor_margin:
            assert actual_margin is margin
        calls.append(True)
        return sentinel

    monkeypatch.setattr(nn.TripletMarginLoss, '__call__', native_boundary)
    result = nn.triplet_margin_loss(anchor, positive, negative, margin=margin, p=p, swap=swap, reduction='none')
    assert result is sentinel and calls == [True]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('which', ['w_q', 'w_k', 'w_v', 'b_q', 'b_k', 'b_v'])
def test_in_projection_rejects_wrong_weight_and_bias_shapes(which):
    """
    Feature: In projection rejects wrong weight and bias shapes.
    Description: Check in projection rejects wrong weight and bias shapes.
    Expectation: Invalid in projection wrong weight and bias shapes raise ValueError.
    """
    args = {
    'q': tensor(
        np.ones(
            (2, 3))), 'k': tensor(
                np.ones(
                    (2, 4))), 'v': tensor(
                        np.ones(
                            (2, 5))), 'w_q': tensor(
                                np.ones(
                                    (3, 3))), 'w_k': tensor(
                                        np.ones(
                                            (3, 4))), 'w_v': tensor(
                                                np.ones(
                                                    (3, 5))), 'b_q': tensor(
                                                        np.zeros(3)), 'b_k': tensor(
                                                            np.zeros(3)), 'b_v': tensor(
                                                                np.zeros(3))}
    args[which] = tensor(np.ones((2, 2)))
    with pytest.raises(ValueError, match='weights shape' if which.startswith('w') else 'bias shape'):
        nn._in_projection(**args)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('route', ['self', 'kv', 'separate'])
def test_packed_projection_with_bias_matches_numpy(route):
    """
    Feature: Packed projection with bias matches numpy.
    Description: Check packed projection with bias matches numpy.
    Expectation: Packed query/key/value projections and biases match the independently split NumPy matrix
        products.
    """
    q = np.array([[1., 2.], [-1., 3.]], np.float32)
    k = q if route == 'self' else q / 2
    v = k if route in ('self', 'kv') else -q
    w = np.arange(12, dtype=np.float32).reshape(6, 2) / 5
    b = np.arange(6, dtype=np.float32) / 4
    results = nn._in_projection_packed(tensor(q), tensor(k), tensor(v), tensor(w), tensor(b),
                                       route in ('self', 'kv'), route == 'self')
    for result, a, weight, bias in zip(results, (q, k, v), np.split(w, 3), np.split(b, 3)):
        np.testing.assert_allclose(result.asnumpy(), a @ weight.T + bias, atol=1e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('rank', [2, 3])
def test_key_padding_mask_rank_validation(rank):
    """
    Feature: Key padding mask rank validation.
    Description: Check key padding mask rank validation.
    Expectation: Invalid key padding mask rank arguments raise ValueError. Diagnostics identify
        key_padding_mask.
    """
    assert nn._check_kpm_shape(rank, rank - 1) is None
    with pytest.raises(ValueError, match='key_padding_mask'):
        nn._check_kpm_shape(rank, rank)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('query_rank', [2, 3])
def test_attention_mask_rank_and_unbatched_head_shape(query_rank):
    """
    Feature: Attention mask rank and unbatched head shape.
    Description: Check attention mask rank and unbatched head shape.
    Expectation: Invalid attention mask rank and unbatched head shape raise ValueError. Diagnostics identify
        attn_mask, shape of.*attn_mask.
    """
    with pytest.raises(ValueError, match='attn_mask'):
        nn._check_attn_mask_shape(query_rank, (3, 4), (5, 4), 1, (3,), 2)
    if query_rank == 2:
        with pytest.raises(ValueError, match='shape of.*attn_mask'):
            nn._check_attn_mask_shape(2, (3, 4), (5, 4), 3, (1, 3, 5), 2)
        assert nn._check_attn_mask_shape(2, (3, 4), (5, 4), 3, (2, 3, 5), 2) is None


def attention_extension_inputs():
    q = np.array([[[1., 0.]], [[0., 1.]]], np.float32)
    k = q * .5
    v = np.array([[[2., 3.]], [[5., -1.]]], np.float32)
    params = {
        'query': tensor(q),
        'key': tensor(k),
        'value': tensor(v),
        'embed_dim_to_check': 2,
        'num_heads': 1,
        'in_proj_weight': tensor(np.tile(np.eye(2, dtype=np.float32), (3, 1))),
        'in_proj_bias': tensor(np.array([0.1, -0.1, 0.2, -0.2, 0.3, -0.3], np.float32)),
        'bias_k': None,
        'bias_v': None,
        'add_zero_attn': False,
        'dropout_p': 0.0,
        'out_proj_weight': tensor(np.eye(2)),
        'out_proj_bias': None,
        'training': False,
        'k_is_v': False,
        'q_is_k': False,
    }
    return params, (q[:, 0] + [.1, -.1], k[:, 0] + [.2, -.2], v[:, 0] + [.3, -.3])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('extension', ['bias', 'zero', 'static'])
@pytest.mark.parametrize('mask', ['none', 'byte', 'float', 'padding'])
@pytest.mark.parametrize('separate', [False, True])
def test_attention_bias_zero_static_extensions_numpy(extension, mask, separate):
    """
    Feature: Attention bias zero static extensions numpy.
    Description: Check attention bias zero static extensions numpy.
    Expectation: Bias, zero-attention, and static key/value extensions preserve the independent attention output
        and weight references.
    """
    args, (q, k, v) = attention_extension_inputs()
    if separate:
        args.update(use_separate_proj_weight=True, q_proj_weight=tensor(np.eye(2)),
                    k_proj_weight=tensor(np.eye(2)), v_proj_weight=tensor(np.eye(2)))
    additive = np.zeros((2, 2), np.float32)
    if mask == 'byte':
        args['attn_mask'] = tensor([[0, 1], [0, 0]], ms.uint8)
        additive[0, 1] = -np.inf
    elif mask == 'float':
        args['attn_mask'] = tensor([[0., -.7], [0., 0.]])
        additive[0, 1] = -.7
    elif mask == 'padding':
        args['key_padding_mask'] = tensor([[False, True]], ms.bool_)
        additive[:, 1] = -np.inf
    if extension == 'bias':
        args.update(bias_k=tensor([[[.7, .4]]]), bias_v=tensor([[[4., 2.]]]))
        k, v = np.concatenate([k, [[.7, .4]]]), np.concatenate([v, [[4., 2.]]])
        additive = np.pad(additive, ((0, 0), (0, 1)))
    elif extension == 'zero':
        args['add_zero_attn'] = True
        k, v = np.concatenate([k, [[0., 0.]]]), np.concatenate([v, [[0., 0.]]])
        additive = np.pad(additive, ((0, 0), (0, 1)))
    else:
        k, v = k * 2, v - 1
        args.update(static_k=tensor(k[None]), static_v=tensor(v[None]))
    scores = q @ k.T / np.sqrt(2) + additive
    probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probability /= probability.sum(axis=-1, keepdims=True)
    output, weights = nn.multi_head_attention_forward(**args)
    np.testing.assert_allclose(output.asnumpy()[:, 0], probability @ v, atol=2e-6)
    np.testing.assert_allclose(weights.asnumpy()[0], probability, atol=2e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('case,reason', [('static_k_width', 'static_k'), ('static_v_batch', 'static_v'),
                                     ('bias_static_k', 'bias_k.*static_k'), ('bias_static_v', 'bias_v.*static_v'),
                                     ('separate_kv', 'sequence length'), ('3d_mask', 'shape of.*attn_mask')])
def test_attention_extension_rejects_incompatible_shapes(case, reason):
    """
    Feature: Attention extension rejects incompatible shapes.
    Description: Check attention extension rejects incompatible shapes.
    Expectation: Invalid attention extension incompatible shapes raise ValueError.
    """
    args, _ = attention_extension_inputs()
    if case == 'static_k_width':
        args['static_k'] = tensor(np.ones((1, 2, 3)))
    elif case == 'static_v_batch':
        args['static_v'] = tensor(np.ones((2, 2, 2)))
    elif case.startswith('bias_static'):
        args.update(bias_k=tensor(np.zeros((1, 1, 2))), bias_v=tensor(np.zeros((1, 1, 2))))
        args[case[5:]] = tensor(np.ones((1, 2, 2)))
    elif case == 'separate_kv':
        args.update(use_separate_proj_weight=True, key=tensor(np.ones((3, 1, 2))))
    else:
        args['attn_mask'] = tensor(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match=reason):
        nn.multi_head_attention_forward(**args)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=RuntimeError, reason='CPU Dropout rejects documented p=1 endpoint')
def test_dropout_one_probability_documented_zero_output():
    """
    Feature: Dropout one probability documented zero output.
    Description: Check dropout one probability documented zero output.
    Expectation: Full training dropout returns a tensor filled with zeros.
    """
    np.testing.assert_array_equal(nn.dropout(tensor([1., 2.]), p=1.).asnumpy(), [0., 0.])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=TypeError, reason='avg_pool1d validates stride as int before its tuple branch')
def test_average_pool1d_documented_tuple_stride():
    """
    Feature: Average pool1d documented tuple stride.
    Description: Check average pool1d documented tuple stride.
    Expectation: A single-element tuple stride behaves as documented for one-dimensional average pooling.
    """
    np.testing.assert_array_equal(nn.avg_pool1d(tensor([[[1., 3., 5., 7.]]]), 2, stride=(2,)).asnumpy(), [[[2., 6.]]])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('name,shape', [('dropout1d', (2, 3)), ('dropout1d', (2, 3, 4)),
                                     ('dropout2d', (2, 3, 2, 2)), ('dropout3d', (2, 3, 2, 2, 2))])
def test_channel_dropout_one_probability_clears_input(name, shape):
    """
    Feature: Channel dropout one probability clears input.
    Description: Check channel dropout one probability clears input.
    Expectation: Full channel dropout clears every element of the input.
    """
    np.testing.assert_array_equal(getattr(nn, name)(tensor(np.ones(shape)), p=1.).asnumpy(), np.zeros(shape))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('name,shape', [('dropout1d', (64, 10)), ('dropout1d', (2, 64, 10)),
                                     ('dropout2d', (2, 64, 2, 5)), ('dropout3d', (2, 64, 2, 2, 5))])
def test_channel_dropout_keeps_whole_channels_with_inverse_scaling(name, shape):
    """
    Feature: Channel dropout keeps whole channels with inverse scaling.
    Description: Check channel dropout keeps whole channels with inverse scaling.
    Expectation: Each channel is entirely dropped or retained, and retained channels use
        inverse-keep-probability scaling.
    """
    ms.set_seed(123)
    a = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
    result = getattr(nn, name)(tensor(a), p=.5).asnumpy()
    ratios = result / a
    channels = ratios.reshape((-1, np.prod(shape[2:]))) if name != 'dropout1d' or len(shape) == 3 else ratios
    for channel in channels:
        assert np.all(channel == 0) or np.all(channel == 2)
    # A constant-zero or identity kernel must fail this fixed-seed, nontrivial mask check.
    assert set(np.unique(ratios)) == {0, 2}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', ['constant', 'reflect', 'replicate', 'circular'])
@pytest.mark.parametrize('container', ['tuple', 'list', 'tensor'])
def test_pad_modes_match_numpy(mode, container):
    """
    Feature: Pad modes match numpy.
    Description: Check pad modes match numpy.
    Expectation: Supported padding modes preserve the same values and spatial layout as the independent NumPy
        padding reference.
    """
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    padding = (1, 1)
    padding = list(padding) if container == 'list' else tensor(padding, ms.int64) if container == 'tensor' else padding
    kwargs = {'value': -2.} if mode == 'constant' else {}
    np_mode = {'constant': 'constant', 'reflect': 'reflect', 'replicate': 'edge', 'circular': 'wrap'}[mode]
    np_kwargs = {'constant_values': -2.} if mode == 'constant' else {}
    expected = np.pad(a, ((0, 0), (1, 1)), mode=np_mode, **np_kwargs)
    np.testing.assert_array_equal(nn.pad(tensor(a), padding, mode=mode, **kwargs).asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('padding', [(), [], ms.Tensor([], ms.int64)])
def test_pad_empty_padding_returns_original(padding):
    """
    Feature: Pad empty padding returns original.
    Description: Check pad empty padding returns original.
    Expectation: An empty padding tuple returns the original tensor by identity.
    """
    x = tensor([1., 2.])
    assert nn.pad(x, padding) is x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kwargs,error,reason', [
    ({'input_x': []}, TypeError, 'input_x'), ({'padding': (1,)}, ValueError, 'divisible'),
    ({'padding': 'aa'}, TypeError, 'paddings'), ({'padding': (1., 1)}, TypeError, 'value'),
    ({'padding': (1,) * 8, 'mode': 'reflect'}, ValueError, 'less than or equal'),
    ({'mode': 'reflect', 'value': 1.}, ValueError, 'can not set value')])
def test_pad_rejects_invalid_configuration(kwargs, error, reason):
    """
    Feature: Pad rejects invalid configuration.
    Description: Check pad rejects invalid configuration.
    Expectation: Invalid pad configuration raise the parameterized exception type.
    """
    args = {'input_x': tensor([[1., 2.]]), 'padding': (1, 1)}
    args.update(kwargs)
    with pytest.raises(error, match=reason):
        nn.pad(**args)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('shape,container', [((2, 3), 'tuple'), ((1, 2, 3), 'list'), ((2, 3), 'tensor')])
def test_internal_circular_pad_wraps_and_restores_rank(shape, container):
    """
    Feature: Internal circular pad wraps and restores rank.
    Description: Check internal circular pad wraps and restores rank.
    Expectation: Circular padding wraps the expected values and restores the original batch/channel rank.
    """
    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    padding = (1, 2)
    padding = list(padding) if container == 'list' else tensor(padding, ms.int64) if container == 'tensor' else padding
    expected = np.pad(a, ((0, 0),) * (len(shape) - 1) + ((1, 2),), mode='wrap')
    np.testing.assert_array_equal(nn._circular_pad(tensor(a), padding).asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('axis,expected', [(-1, 2), (0, 0), (2, 2)])
def test_flip_axis_normalization(axis, expected):
    """
    Feature: Flip axis normalization.
    Description: Check flip axis normalization.
    Expectation: Negative and positive flip axes normalize to the explicitly expected dimension.
    """
    assert nn._check_axis_in_range(axis, 3) == expected


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('axis,error', [('0', TypeError), (3, ValueError), (-4, ValueError)])
def test_flip_axis_rejects_non_integer_or_out_of_range(axis, error):
    """
    Feature: Flip axis rejects non integer or out of range.
    Description: Check flip axis rejects non integer or out of range.
    Expectation: Invalid flip axis non integer or out of range raise the parameterized exception type.
        Diagnostics identify dims|axis.
    """
    with pytest.raises(error, match='dims|axis'):
        nn._check_axis_in_range(axis, 3)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('axes', [None, 1, (0, -3)])
def test_flip_axes_rejects_missing_nonsequence_or_duplicate(axes):
    """
    Feature: Flip axes rejects missing nonsequence or duplicate.
    Description: Check flip axes rejects missing nonsequence or duplicate.
    Expectation: Invalid flip axes missing nonsequence or duplicate raise ValueError. Diagnostics identify
        dims|duplicate.
    """
    with pytest.raises(ValueError, match='dims|duplicate'):
        nn._check_axis_valid(axes, 3)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_flip_slice_helpers_and_tensor_validation():
    """
    Feature: Flip slice helpers and tensor validation.
    Description: Check flip slice helpers and tensor validation.
    Expectation: Slice helpers produce the expected reversed ranges and reject invalid tensor inputs.
    """
    assert nn._check_axis_valid([-1, 0], 3) == (2, 0)
    assert nn._get_flip_start(3, (2, 3, 4), (0, 2)) == (1, 0, 3)
    assert nn._get_flip_end(3, (2, 3, 4), (0, 2)) == (-3, 4, -5)
    assert nn._get_flip_strides(3, (0, 2)) == (-1, 1, -1)
    assert nn._is_shape_empty(0) and nn._is_shape_empty((2, 0, 3))
    assert not nn._is_shape_empty((2, 3))
    assert nn._check_input_tensor('flip', tensor([1.]))
    with pytest.raises(TypeError, match='must be Tensor'):
        nn._check_input_tensor('flip', tensor([1.]), [])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('name', ['softmax', 'softmin'])
def test_softmax_and_softmin_cast_dtype_numpy(name):
    """
    Feature: Softmax and softmin cast dtype numpy.
    Description: Check softmax and softmin cast dtype numpy.
    Expectation: Softmax and softmin match independent stable NumPy exponentials and honor the requested output
        dtype.
    """
    # NumPy dynamically supplies these ndarray/array-function signatures.
    # pylint: disable=unexpected-keyword-arg
    a = np.array([[1., 2., 3.], [-1., 0., 1.]], np.float32)
    values = -a if name == 'softmin' else a
    expected = np.exp(values - values.max(axis=-1, keepdims=True))
    expected /= expected.sum(axis=-1, keepdims=True)
    result = getattr(nn, name)(tensor(a), dtype=ms.float64)
    assert result.dtype == ms.float64
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=2e-7)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_softmax_rejects_non_integer_axis():
    """
    Feature: Softmax rejects non integer axis.
    Description: Check softmax rejects non integer axis.
    Expectation: Invalid softmax non integer axis raise TypeError. Diagnostics identify axis.*int.
    """
    with pytest.raises(TypeError, match='axis.*int'):
        nn.softmax(tensor([1., 2.]), axis='0')


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('beta,threshold', [(1, 20), (2, 1)])
def test_softplus_beta_and_linear_threshold_numpy(beta, threshold):
    """
    Feature: Softplus beta and linear threshold numpy.
    Description: Check softplus beta and linear threshold numpy.
    Expectation: Softplus matches the independent beta-scaled logarithm below the threshold and the linear
        branch above it.
    """
    a = np.array([-2., -.1, 0., 1., 5.], np.float32)
    expected = np.where(a * beta > threshold, a, np.logaddexp(0, a * beta) / beta)
    np.testing.assert_allclose(nn.softplus(tensor(a), beta, threshold).asnumpy(), expected, rtol=8e-7, atol=2e-7)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_softplus_rejects_zero_beta():
    """
    Feature: Softplus rejects zero beta.
    Description: Check softplus rejects zero beta.
    Expectation: Invalid softplus zero beta raise ValueError. Diagnostics identify beta.*cannot be 0.
    """
    with pytest.raises(ValueError, match='beta.*cannot be 0'):
        nn.softplus(tensor([1.]), beta=0)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_activation_wrappers_independent_elementwise_oracles():
    """
    Feature: Activation wrappers independent elementwise oracles.
    Description: Check activation wrappers independent elementwise oracles.
    Expectation: Each activation agrees with its independently written elementwise NumPy formula.
    """
    a = np.array([-3., -.5, 0., .5, 3.], np.float32)
    x = tensor(a)
    expected_shrink = np.where(a > .5, a - .5, np.where(a < -.5, a + .5, 0))
    np.testing.assert_allclose(nn.soft_shrink(x).asnumpy(), expected_shrink)
    np.testing.assert_allclose(nn.hardtanh(x, -.75, 1.5).asnumpy(), np.clip(a, -.75, 1.5))
    np.testing.assert_allclose(nn.threshold(x, .5, -7).asnumpy(), np.where(a > .5, a, -7))
    np.testing.assert_allclose(nn.logsigmoid(x).asnumpy(), -np.logaddexp(0, -a), atol=2e-7)
    np.testing.assert_allclose(nn.mish(x).asnumpy(), a * np.tanh(np.logaddexp(0, a)), atol=2e-7)
    expected_selu = 1.0507009873554805 * np.where(a > 0, a, 1.6732632423543772 * np.expm1(a))
    np.testing.assert_allclose(nn.selu(x).asnumpy(), expected_selu, atol=2e-7)
    np.testing.assert_array_equal(nn.relu(x).asnumpy(), np.maximum(a, 0))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('name,kwargs,error,reason', [
    ('rrelu', {'upper': 'x'}, TypeError, 'upper'), ('rrelu', {'lower': 'x'}, TypeError, 'lower'),
    ('rrelu', {'lower': .5, 'upper': .1}, ValueError, 'upper'), ('rrelu', {'input': []}, TypeError, 'Tensor'),
    ('gelu', {'approximate': 'x'}, ValueError, 'approximate'),
    ('gelu', {'input': tensor([1], ms.int32)}, TypeError, 'dtype'),
    ('multi_margin_loss', {'target': tensor([0], ms.int64), 'margin': 1.5}, TypeError, 'margin')])
def test_remaining_activation_and_margin_argument_checks(name, kwargs, error, reason):
    """
    Feature: Remaining activation and margin argument checks.
    Description: Check remaining activation and margin argument checks.
    Expectation: Invalid remaining activation and margin argument checks raise the parameterized exception type.
    """
    args = {'input': tensor([-1., 2.])}
    args.update(kwargs)
    with pytest.raises(error, match=reason):
        getattr(nn, name)(**args)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('axis', [-1, 0])
def test_glu_numpy_sigmoid_gate(axis):
    """
    Feature: Glu numpy sigmoid gate.
    Description: Check glu numpy sigmoid gate.
    Expectation: GLU splits the selected axis and multiplies the first half by the sigmoid of the second half.
    """
    a = np.arange(8, dtype=np.float32).reshape(2, 4) / 4 - 1
    first, second = np.split(a, 2, axis=axis)
    np.testing.assert_allclose(nn.glu(tensor(a), axis).asnumpy(), first / (1 + np.exp(-second)), atol=1e-7)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_channel_shuffle_literal_channel_permutation():
    """
    Feature: Channel shuffle literal channel permutation.
    Description: Check channel shuffle literal channel permutation.
    Expectation: Channel shuffling returns the explicit expected channel permutation without changing the
        values.
    """
    a = np.arange(16, dtype=np.float32).reshape(1, 4, 2, 2)
    np.testing.assert_array_equal(nn.channel_shuffle(tensor(a), 2).asnumpy(), a[:, [0, 2, 1, 3]])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dim', [1, 2])
@pytest.mark.parametrize('norm', [1, 2])
def test_lp_pool_numpy_window_norm(dim, norm):
    """
    Feature: Lp pool numpy window norm.
    Description: Check lp pool numpy window norm.
    Expectation: Lp pooling matches independently calculated p-power window sums followed by the
        reciprocal-power transform.
    """
    a = np.arange(1, 1 + 4 ** dim, dtype=np.float32).reshape((1, 1) + (4,) * dim)
    result = getattr(nn, 'lp_pool' + str(dim) + 'd')(tensor(a), norm, 2)
    expected = adaptive_reference(a ** norm, (2,) * dim) * (2 ** dim)
    np.testing.assert_allclose(result.asnumpy(), expected ** (1 / norm), rtol=1e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dim', [1, 2])
@pytest.mark.parametrize('norm,error', [('x', TypeError), (0, ValueError)])
def test_lp_pool_rejects_invalid_norm(dim, norm, error):
    """
    Feature: Lp pool rejects invalid norm.
    Description: Check lp pool rejects invalid norm.
    Expectation: Invalid lp pool norm raise the parameterized exception type. Diagnostics identify norm_type.
    """
    with pytest.raises(error, match='norm_type'):
        getattr(nn, 'lp_pool' + str(dim) + 'd')(tensor(np.ones((1, 1) + (4,) * dim)), norm, 2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kwargs,error,reason', [
    ({'mode': 'area'}, ValueError, 'mode'), ({'size': None}, ValueError, 'both empty'),
    ({'recompute_scale_factor': True}, ValueError, 'explicit.*size')])
def test_interpolate_ext_invalid_options_before_dispatch(kwargs, error, reason):
    """
    Feature: Interpolate ext invalid options before dispatch.
    Description: Check interpolate ext invalid options before dispatch.
    Expectation: Invalid interpolate ext options before dispatch raise the parameterized exception type.
    """
    args = {'input': tensor(np.ones((1, 1, 2, 2))), 'size': (4, 4)}
    args.update(kwargs)
    with pytest.raises(error, match=reason):
        nn.interpolate_ext(**args)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('value', [2, (2, 3), [2, 3]])
def test_interpolate_ext_tuple_and_floor_size_helpers(value):
    """
    Feature: Interpolate ext tuple and floor size helpers.
    Description: Check interpolate ext tuple and floor size helpers.
    Expectation: Extended interpolation helpers return the expected tuple and floor fractional output dimensions
        correctly.
    """
    x = tensor(np.ones((1, 1, 3, 5)))
    expected = (2, 2) if isinstance(value, int) else value
    assert nn._interploate_ext_make_tuple(x, value) == expected
    assert nn._interpolate_ext_scale_factor_convert_size(x, (.5, 1.5)) == (1, 7)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('fn', ['interpolate_ext', 'upsample'])
def test_upsample_alias_and_ext_nearest_real_cpu(fn):
    """
    Feature: Upsample alias and ext nearest real cpu.
    Description: Check upsample alias and ext nearest real cpu.
    Expectation: Upsample aliases and extended nearest interpolation preserve the expected repeated CPU values.
    """
    a = np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)
    kwargs = {'scale_factor': 2., 'recompute_scale_factor': True}
    expected = np.repeat(np.repeat(a, 2, axis=2), 2, axis=3)
    np.testing.assert_array_equal(getattr(nn, fn)(tensor(a), **kwargs).asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_softmax_ext_dtype_and_axis_contract():
    """
    Feature: Softmax ext dtype and axis contract.
    Description: Check softmax ext dtype and axis contract.
    Expectation: Extended softmax honors its dtype and axis, and unsupported axis types raise TypeError.
    """
    a = np.array([[1., 2., 3.]], np.float32)
    expected = np.exp(a) / np.exp(a).sum(axis=-1, keepdims=True)
    result = nn.softmax_ext(tensor(a), dim=-1, dtype=ms.float64)
    assert result.dtype == ms.float64
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=2e-7)
    with pytest.raises(TypeError, match='dim|axis|int'):
        nn.softmax_ext(tensor(a), dim='x')


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_dense_shape_checks_reject_bias_broadcast_and_mismatched_batches():
    """
    Feature: Dense shape checks reject bias broadcast and mismatched batches.
    Description: Check dense shape checks reject bias broadcast and mismatched batches.
    Expectation: Invalid dense shape checks reject bias broadcast and mismatched batches raise ValueError.
        Diagnostics identify bias shape, dimensions except the last.
    """
    assert nn._check_dense_add_bias_shape((2, 3), (2, 3), (3,)) is None
    with pytest.raises(ValueError, match='bias shape'):
        nn._check_dense_add_bias_shape((2, 3), (2, 2, 3), (2, 1, 1))
    with pytest.raises(ValueError, match='dimensions except the last'):
        nn.check_dense_inputs_same_shape((2, 3), (3, 4))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_bidense_rejects_differing_bias_dtype():
    """
    Feature: Bidense rejects differing bias dtype.
    Description: Check bidense rejects differing bias dtype.
    Expectation: Invalid bidense differing bias dtype raise TypeError. Diagnostics identify dtype.*bias.
    """
    with pytest.raises(TypeError, match='dtype.*bias'):
        nn.bidense(tensor([[1., 2.]]), tensor([[3., 4., 5.]]), tensor(np.ones((2, 2, 3))),
                   tensor([1., 2.], ms.float64))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('fn,kwargs,reason', [
    ('pad_ext', {'mode': 'wrong'}, 'mode'), ('pad_ext', {'mode': 'reflect', 'value': 2}, 'value')])
def test_pad_ext_rejects_invalid_nonconstant_mode_and_value(fn, kwargs, reason):
    """
    Feature: Pad ext rejects invalid nonconstant mode and value.
    Description: Check pad ext rejects invalid nonconstant mode and value.
    Expectation: Invalid pad ext nonconstant mode and value raise ValueError.
    """
    with pytest.raises(ValueError, match=reason):
        getattr(nn, fn)(tensor([[1., 2.]]), (1, 1), **kwargs)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_pad_ext_circular_and_mirror_pad_numpy():
    """
    Feature: Pad ext circular and mirror pad numpy.
    Description: Check pad ext circular and mirror pad numpy.
    Expectation: Extended circular, reflect, and symmetric padding agree with NumPy's independently padded
        arrays.
    """
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(nn.pad_ext(tensor(a), (1, 1), mode='circular').asnumpy(),
                                  np.pad(a, ((0, 0), (1, 1)), mode='wrap'))
    mirrored = nn.mirror_pad(tensor(a), tensor([[1, 1], [1, 1]], ms.int64), mode='REFLECT')
    np.testing.assert_array_equal(mirrored.asnumpy(), np.pad(a, ((1, 1), (1, 1)), mode='reflect'))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('fn', ['l1_loss', 'huber_loss'])
def test_elementwise_loss_rejects_invalid_reduction(fn):
    """
    Feature: Elementwise loss rejects invalid reduction.
    Description: Check elementwise loss rejects invalid reduction.
    Expectation: Invalid elementwise loss reduction raise ValueError. Diagnostics identify reduction.
    """
    with pytest.raises(ValueError, match='reduction'):
        getattr(nn, fn)(tensor([1.]), tensor([0.]), reduction='bad')


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('case', ['input', 'target', 'reduction'])
def test_mse_loss_invalid_inputs_and_reduction(case):
    """
    Feature: Mse loss invalid inputs and reduction.
    Description: Check mse loss invalid inputs and reduction.
    Expectation: Invalid mse loss inputs and reduction raise ValueError if case == 'reduction' else TypeError.
    """
    args = {'input': tensor([1.]), 'target': tensor([0.]), 'reduction': 'mean'}
    args[case] = 'bad' if case == 'reduction' else []
    with pytest.raises(ValueError if case == 'reduction' else TypeError, match=case):
        nn.mse_loss(**args)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_loss_dtype_check_rejects_inconsistent_pair():
    """
    Feature: Loss dtype check rejects inconsistent pair.
    Description: Check loss dtype check rejects inconsistent pair.
    Expectation: Invalid loss dtype check inconsistent pair raise TypeError. Diagnostics identify dtype should
        be equal.
    """
    with pytest.raises(TypeError, match='dtype should be equal'):
        nn.check_input_dtype('prediction', tensor([1.]), 'target', tensor([1.], ms.float64), 'example')


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('case,error,reason', [
    ('margin', TypeError, 'margin'), ('reduction', ValueError, 'reduction'),
    ('input', TypeError, 'first input'), ('target', TypeError, 'second input'),
    ('input_dtype', TypeError, 'first input.*float')])
def test_hinge_embedding_validator_contract(case, error, reason):
    """
    Feature: Hinge embedding validator contract.
    Description: Check hinge embedding validator contract.
    Expectation: The hinge validator accepts supported loss arguments and rejects the specified incompatible
        types and reductions.
    """
    args = {'inputs_dtype': ms.float32, 'targets_dtype': ms.float32, 'inputs': tensor(
        [1.]), 'targets': tensor([1.]), 'margin': 1., 'reduction': 'none'}
    args.update({'margin': {'margin': 'bad'}, 'reduction': {'reduction': 'bad'},
                 'input': {'inputs': []}, 'target': {'targets': []},
                 'input_dtype': {'inputs_dtype': ms.int32}}[case])
    with pytest.raises(error, match=reason):
        nn._check_hinge_embedding_loss_type(**args)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('case,error,reason', [('groups', ValueError, 'groups'), ('bias', TypeError, 'bias')])
def test_convolution_rejects_groups_and_bias_contract(dim, case, error, reason):
    """
    Feature: Convolution rejects groups and bias contract.
    Description: Check convolution rejects groups and bias contract.
    Expectation: Invalid convolution groups and bias contract raise the parameterized exception type.
    """
    args = {'input': tensor(np.ones((1, 2) + (3,) * dim)), 'weight': tensor(np.ones((2, 2) + (1,) * dim))}
    args.update({'groups': 3} if case == 'groups' else {'bias': [1., 2.]})
    with pytest.raises(error, match=reason):
        getattr(nn, 'conv' + str(dim) + 'd')(**args)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dim', [1, 2])
def test_convolution_rejects_wrong_bias_size(dim):
    """
    Feature: Convolution rejects wrong bias size.
    Description: Check convolution rejects wrong bias size.
    Expectation: Invalid convolution wrong bias size raise ValueError. Diagnostics identify bias.
    """
    with pytest.raises(ValueError, match='bias'):
        getattr(nn, 'conv' + str(dim) + 'd')(tensor(np.ones((1, 2) + (3,) * dim)),
                                           tensor(np.ones((2, 2) + (1,) * dim)), bias=tensor([1.]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('case,error,reason', [
    ('input', ValueError, 'input.*3D'), ('weight', ValueError, 'weight.*3D'),
    ('padding_length', ValueError, 'padding'), ('padding_type', TypeError, 'padding')])
def test_conv1d_rejects_ranks_and_padding(case, error, reason):
    """
    Feature: Conv1d rejects ranks and padding.
    Description: Check conv1d rejects ranks and padding.
    Expectation: Invalid conv1d ranks and padding raise the parameterized exception type.
    """
    args = {'input': tensor(np.ones((1, 1, 3))), 'weight': tensor(np.ones((1, 1, 1)))}
    args.update({'input': {'input': tensor(np.ones((1, 3)))}, 'weight': {'weight': tensor(np.ones((1, 1)))},
                 'padding_length': {'padding': (0, 0)}, 'padding_type': {'padding': 'x'}}[case])
    with pytest.raises(error, match=reason):
        nn.conv1d(**args)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dim', [1, 3])
def test_convolution_without_bias_preserves_identity_kernel(dim):
    """
    Feature: Convolution without bias preserves identity kernel.
    Description: Check convolution without bias preserves identity kernel.
    Expectation: A unit identity kernel without bias leaves the convolution input values unchanged.
    """
    a = np.arange(8, dtype=np.float32).reshape((1, 1) + ((8,) if dim == 1 else (2, 2, 2)))
    result = getattr(nn, 'conv' + str(dim) + 'd')(tensor(a), tensor(np.ones((1, 1) + (1,) * dim)))
    np.testing.assert_array_equal(result.asnumpy(), a)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_convolution_iterable_parameter_rejects_incorrect_length():
    """
    Feature: Convolution iterable parameter rejects incorrect length.
    Description: Check convolution iterable parameter rejects incorrect length.
    Expectation: Invalid convolution iterable parameter incorrect length raise ValueError. Diagnostics identify
        stride.
    """
    with pytest.raises(ValueError, match='stride'):
        nn._check_conv_iterable_lengths((1,), 2, 'stride')


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('which', ['inputs', 'weight'])
def test_conv3d_transpose_rejects_wrong_rank(which):
    """
    Feature: Conv3d transpose rejects wrong rank.
    Description: Check conv3d transpose rejects wrong rank.
    Expectation: Invalid conv3d transpose wrong rank raise ValueError.
    """
    args = {'inputs': tensor(np.ones((1, 1, 2, 2, 2))), 'weight': tensor(np.ones((1, 1, 1, 1, 1)))}
    args[which] = tensor(np.ones((1, 1, 2, 2)))
    with pytest.raises(ValueError, match='rank of ' + which):
        nn.conv3d_transpose(**args)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dtype', [ms.float16, ms.float32])
def test_conv3d_transpose_singleton_kernel_scale(dtype):
    """
    Feature: Conv3d transpose singleton kernel scale.
    Description: Check conv3d transpose singleton kernel scale.
    Expectation: A singleton transpose-convolution kernel scales each input element by the literal kernel value.
    """
    a = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 2, 2)
    result = nn.conv3d_transpose(tensor(a, dtype), tensor(np.full((1, 1, 1, 1, 1), 2.), dtype))
    np.testing.assert_array_equal(result.asnumpy(), a * 2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level2', card_mark='onecard')
@pytest.mark.skipif(os.environ.get('MS_NN_NATIVE_TARGET') != 'Ascend', reason='MultilabelMarginLoss has no CPU kernel')
@pytest.mark.usefixtures('ascend_context')
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
def test_multilabel_margin_loss_literal_pairwise_margins(reduction):
    """
    Feature: Multilabel margin loss literal pairwise margins.
    Description: Check multilabel margin loss literal pairwise margins.
    Expectation: Multilabel margin loss matches the explicitly enumerated positive/negative class margins for
        each reduction.
    """
    a = np.array([[.1, .2, .4, .8], [.7, .3, .1, .2]], np.float32)
    target = np.array([[0, 2, -1, -1], [1, -1, -1, -1]], np.int32)
    expected = []
    for row, labels in zip(a, target):
        classes = labels[labels >= 0]
        expected.append(sum(max(0., 1 - row[c] + row[j]) for c in classes for j in range(4) if j not in classes) / 4)
    expected = np.array(expected)
    if reduction == 'mean':
        expected = expected.mean()
    elif reduction == 'sum':
        expected = expected.sum()
    result = nn.multilabel_margin_loss(tensor(a), tensor(target, ms.int32), reduction)
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=2e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kwargs,error,reason', [
    ({'input': []}, TypeError, 'Tensor'), ({'target': []}, TypeError, 'Tensor'),
    ({'weight': []}, TypeError, 'weight'), ({'label_smoothing': -1.}, ValueError, 'label_smoothing'),
    ({'input': tensor(1.)}, ValueError, '0-dim'), ({'input': tensor(np.zeros((0, 3)))}, ValueError, 'shape')])
def test_cross_entropy_ext_validation_before_unavailable_native_logsoftmax(kwargs, error, reason):
    """
    Feature: Cross entropy ext validation before unavailable native logsoftmax.
    Description: Check cross entropy ext validation before unavailable native logsoftmax.
    Expectation: Invalid cross entropy ext arguments before unavailable native logsoftmax raise the
        parameterized exception type.
    """
    args = {'input': tensor([[1., 2., 3.]]), 'target': tensor([0], ms.int32)}
    args.update(kwargs)
    with pytest.raises(error, match=reason):
        nn.cross_entropy_ext(**args)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
@pytest.mark.parametrize('weighted', [False, True])
def test_cross_entropy_probability_helper_uses_real_cpu_math(reduction, weighted):
    """
    Feature: Cross entropy probability helper uses real cpu math.
    Description: Check cross entropy probability helper uses real cpu math.
    Expectation: The probability-target helper matches the independent smoothed and weighted CPU cross-entropy
        formula.
    """
    # NumPy dynamically supplies these ndarray/array-function signatures.
    # pylint: disable=too-many-function-args
    logp = np.log(np.array([[[.2, .4], [.3, .1], [.5, .5]]], np.float32))
    target = np.array([[[.1, .2], [.2, .3], [.7, .5]]], np.float32)
    weight = np.array([1., .5, 2.], np.float32)
    loss = -logp * (target * .8 + .2 / 3)
    if weighted:
        loss *= weight.reshape(1, 3, 1)
    expected = loss.sum(axis=1)
    if reduction == 'sum':
        expected = expected.sum()
    elif reduction == 'mean':
        expected = expected.mean()
    result = nn._cross_entropy_for_probabilities(tensor(logp), tensor(target), tensor(weight) if weighted else None,
                                                 reduction, .2, 1, 3)
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=2e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('case', ['shape', 'reduction'])
def test_cross_entropy_probability_helper_rejects_shape_and_reduction(case):
    """
    Feature: Cross entropy probability helper rejects shape and reduction.
    Description: Check cross entropy probability helper rejects shape and reduction.
    Expectation: Invalid cross entropy probability helper shape and reduction raise ValueError.
    """
    with pytest.raises(ValueError, match='shape' if case == 'shape' else 'reduction'):
        nn._cross_entropy_for_probabilities(tensor([[1., 2.]]), tensor([[.5, .5]] if case != 'shape' else [[1.]]),
                                            None, 'bad' if case == 'reduction' else 'none', 0., 1, 2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason='CPU MaskedFill cannot combine float input and integer scalar fill in CE smoothing')
def test_cross_entropy_class_helper_smoothed_expected_loss():
    """
    Feature: Cross entropy class helper smoothed expected loss.
    Description: Check cross entropy class helper smoothed expected loss.
    Expectation: The class-target helper returns the independently calculated label-smoothed loss.
    """
    result = nn._cross_entropy_for_class_indices(tensor([[-1., -2., -3.]]), tensor([1], ms.int32),
                                                 None, -100, 'mean', .2, 1, 3)
    np.testing.assert_allclose(result.asnumpy(), 2.)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_nllloss_nd_rejects_mismatched_batch_size():
    """
    Feature: Nllloss nd rejects mismatched batch size.
    Description: Check nllloss nd rejects mismatched batch size.
    Expectation: Invalid nllloss nd mismatched batch size raise ValueError. Diagnostics identify batch_size.
    """
    with pytest.raises(ValueError, match='batch_size'):
        nn._nllloss_nd(tensor(np.ones((2, 3))), tensor([0], ms.int32))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_nll_loss_explicit_none_ignore_index_numpy():
    """
    Feature: Nll loss explicit none ignore index numpy.
    Description: Check nll loss explicit none ignore index numpy.
    Expectation: An explicit None ignore index retains all target classes and agrees with the NumPy NLL
        reference.
    """
    a = np.array([[-1., -2., -3.], [-4., -5., -6.]], np.float32)
    result = nn.nll_loss(tensor(a), tensor([2, 0], ms.int32), ignore_index=None, reduction='none')
    np.testing.assert_array_equal(result.asnumpy(), [3., 4.])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_intopk_literal_membership_and_lrn_channel_window():
    """
    Feature: Intopk literal membership and lrn channel window.
    Description: Check intopk literal membership and lrn channel window.
    Expectation: Top-k membership matches the literal labels, and local response normalization matches the
        explicit channel-window formula.
    """
    # NumPy dynamically supplies these ndarray/array-function signatures.
    # pylint: disable=too-many-function-args
    a = np.array([[1., 8., 5., 2., 7.], [4., 9., 1., 3., 5.]], np.float32)
    np.testing.assert_array_equal(nn.intopk(tensor(a), tensor([1, 3], ms.int32), 3).asnumpy(), [True, False])
    x = np.array([1., 2., 3.], np.float32).reshape(1, 3, 1, 1)
    expected = x.copy()
    for c in range(3):
        expected[:, c] /= (2 + .5 * (x[:, max(0, c - 1):min(3, c + 2)] ** 2).sum(axis=1)) ** .75
    result = nn.lrn(tensor(x), depth_radius=1, bias=2., alpha=.5, beta=.75)
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=2e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
def test_ctc_loss_enumerated_two_timestep_alignments(reduction):
    """
    Feature: Ctc loss enumerated two timestep alignments.
    Description: Check ctc loss enumerated two timestep alignments.
    Expectation: CTC loss equals the negative log probability from the independently enumerated two-timestep
        alignments.
    """
    probabilities = np.array([[[.6, .4]], [[.7, .3]]], np.float32)
    expected = -np.log(.4 * .7 + .6 * .3 + .4 * .3)
    loss, log_alpha = nn.ctc_loss(tensor(np.log(probabilities)), tensor([[1]], ms.int32),
                                 tensor([2], ms.int32), tensor([1], ms.int32), reduction=reduction)
    np.testing.assert_allclose(loss.asnumpy(), [expected] if reduction == 'none' else expected, atol=2e-7)
    assert log_alpha.shape == (1, 2, 3)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('merge', [False, True])
def test_ctc_greedy_decoder_exact_path(merge):
    """
    Feature: Ctc greedy decoder exact path.
    Description: Check ctc greedy decoder exact path.
    Expectation: Greedy CTC decoding returns the literal sparse path, shape, and negative log-probability sum.
    """
    a = np.array([[[.1, .8, .2]], [[.2, .7, .1]], [[.9, .1, .2]]], np.float32)
    indices, values, shape, log_probability = nn.ctc_greedy_decoder(
        tensor(a), tensor([3], ms.int32), merge_repeated=merge)
    expected_values = [1, 0] if merge else [1, 1, 0]
    np.testing.assert_array_equal(values.asnumpy(), expected_values)
    np.testing.assert_array_equal(indices.asnumpy(), [[0, i] for i in range(len(expected_values))])
    np.testing.assert_array_equal(shape.asnumpy(), [1, len(expected_values)])
    np.testing.assert_allclose(log_probability.asnumpy(), [[-2.4]], atol=2e-7)


@pytest.fixture(name='ascend_context')
def fixture_ascend_context(cpu_context, preserve_random_state):
    """Native-only cases run in a fresh process with an explicitly selected device."""
    if 'MS_NN_ASCEND_DEVICE' in os.environ:
        assert ms.get_context('device_id') == int(os.environ['MS_NN_ASCEND_DEVICE'])
    previous = {key: ms.get_context(key) for key in ('mode', 'device_target')}
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='Ascend')
    yield
    ms.set_context(**previous)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level2', card_mark='onecard')
@pytest.mark.skipif(os.environ.get('MS_NN_NATIVE_TARGET') != 'Ascend', reason='TripletMarginLoss has no CPU kernel')
@pytest.mark.usefixtures('ascend_context')
@pytest.mark.parametrize('p', [1, 2])
@pytest.mark.parametrize('swap', [False, True])
@pytest.mark.parametrize('tensor_margin', [False, True])
@pytest.mark.xfail(strict=True, raises=RuntimeError,
                   reason='Ascend TripletMarginLoss OpAdapter is missing in this runtime')
def test_ascend_triplet_margin_distance_and_swap_numpy(p, swap, tensor_margin):
    """
    Feature: Ascend triplet margin distance and swap numpy.
    Description: Check ascend triplet margin distance and swap numpy.
    Expectation: Ascend triplet losses match the independent p-distance formula, including distance swapping and
        each reduction.
    """
    anchor = np.array([[.3, .7], [.5, .5]], np.float32)
    positive = np.array([[.4, .6], [.4, .6]], np.float32)
    negative = np.array([[.2, .9], [.3, .7]], np.float32)
    def norm(x, y):
        return (np.abs(x - y + 1e-6) ** p).sum(axis=1) ** (1 / p)
    negative_distance = norm(anchor, negative)
    if swap:
        negative_distance = np.minimum(negative_distance, norm(positive, negative))
    expected = np.maximum(norm(anchor, positive) - negative_distance + .8, 0)
    result = nn.triplet_margin_loss(tensor(anchor), tensor(positive), tensor(negative),
                                    margin=tensor(.8) if tensor_margin else .8, p=p, swap=swap, reduction='none')
    np.testing.assert_allclose(result.asnumpy(), expected, atol=3e-6)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level2', card_mark='onecard')
@pytest.mark.skipif(os.environ.get('MS_NN_NATIVE_TARGET') != 'Ascend',
                   reason='LogSoftmaxExt has no registered CPU kernel')
@pytest.mark.usefixtures('ascend_context')
@pytest.mark.parametrize('rank', [2, 3, 4])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('probability_target', [False, True])
def test_ascend_cross_entropy_ext_weighted_smoothing_numpy(rank, weighted, reduction, probability_target):
    """
    Feature: Ascend cross entropy ext weighted smoothing numpy.
    Description: Check ascend cross entropy ext weighted smoothing numpy.
    Expectation: Ascend extended cross entropy matches the independently calculated class-weighted, smoothed
        loss and reduction.
    """
    shape = (2, 3) + (2,) * (rank - 2)
    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 10 - 1
    logp = a - np.log(np.exp(a).sum(axis=1, keepdims=True))
    weights = np.array([1., .5, 2.], np.float32) if weighted else np.ones(3, np.float32)
    wshape = (1, 3) + (1,) * (rank - 2)
    if probability_target:
        target = np.broadcast_to(np.array([.2, .3, .5], np.float32).reshape(wshape), shape).copy()
        expected = -(logp * (target * .8 + .2 / 3) * weights.reshape(wshape)).sum(axis=1)
        denominator = expected.size
    else:
        target = np.arange(np.prod(shape[:1] + shape[2:]), dtype=np.int32).reshape(shape[:1] + shape[2:]) % 3
        target.reshape(-1)[0] = -100
        valid = target != -100
        safe = np.where(valid, target, 0)
        nll = -np.take_along_axis(logp, safe[:, None], axis=1).squeeze(axis=1) * weights[safe]
        smooth = -(logp * weights.reshape(wshape)).sum(axis=1) / 3
        expected = np.where(valid, .8 * nll + .2 * smooth, 0)
        denominator = np.where(valid, weights[safe], 0).sum()
    if reduction == 'mean':
        expected = expected.sum() / denominator
    elif reduction == 'sum':
        expected = expected.sum()
    result = nn.cross_entropy_ext(tensor(a), tensor(target, ms.float32 if probability_target else ms.int32),
                                   weight=tensor(weights) if weighted else None, reduction=reduction,
                                   label_smoothing=.2)
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=3e-5, atol=3e-6)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level2', card_mark='onecard')
@pytest.mark.skipif(os.environ.get('MS_NN_NATIVE_TARGET') != 'Ascend',
                   reason='LogSoftmaxExt has no registered CPU kernel')
@pytest.mark.usefixtures('ascend_context')
@pytest.mark.parametrize('weighted', [False, True])
def test_ascend_cross_entropy_all_ignored_mean_is_nan(weighted):
    """
    Feature: Ascend cross entropy all ignored mean is nan.
    Description: Check ascend cross entropy all ignored mean is nan.
    Expectation: Mean cross entropy with every target ignored returns NaN.
    """
    result = nn.cross_entropy_ext(tensor([[1., 2., 3.]]), tensor([-100], ms.int32),
                                   weight=tensor([1., 2., 3.]) if weighted else None, label_smoothing=.2)
    assert np.isnan(result.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
def test_multilabel_margin_native_boundary_protocol(reduction, monkeypatch):
    """
    Feature: Multilabel margin native boundary protocol.
    Description: Verify wrapper tuple selection without replacing a numerical result.
    Expectation: Verify wrapper tuple selection without replacing a numerical result.
    """
    x = tensor([[.1, .2, .3]])
    target = tensor([[1, -1, -1]], ms.int32)
    sentinel, ignored = object(), object()
    calls = []

    def native_boundary(primitive, actual_x, actual_target):
        assert primitive.get_attr_dict()['reduction'] == reduction
        assert actual_x is x and actual_target is target
        calls.append(True)
        return sentinel, ignored

    monkeypatch.setattr(nn.P.MultilabelMarginLoss, '__call__', native_boundary)
    assert nn.multilabel_margin_loss(x, target, reduction) is sentinel
    assert calls == [True]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('name,arguments,reason', [
    ('_check_pxiel_shuffle_valid', (3, 2), 'not divisible'),
    ('_check_pxiel_unshuffle_valid', (3, 4, 2), 'divisible'),
    ('_check_pixel_shuffle_unshuffle_input_shape', (ms.Tensor([1.]), 'pixel_shuffle'), 'dimension')])
def test_pixel_reshape_validators_reject_incompatible_dimensions(name, arguments, reason):
    """
    Feature: Pixel reshape validators reject incompatible dimensions.
    Description: Check pixel reshape validators reject incompatible dimensions.
    Expectation: Invalid pixel reshape validators reject incompatible dimensions raise ValueError.
    """
    with pytest.raises(ValueError, match=reason):
        getattr(nn, name)(*arguments)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level2', card_mark='onecard')
@pytest.mark.skipif(os.environ.get('MS_NN_NATIVE_TARGET') != 'Ascend',
                   reason='InplaceReLU has no registered CPU kernel')
@pytest.mark.usefixtures('ascend_context')
def test_relu_inplace_mutates_tensor_and_returns_relu_values():
    """
    Feature: Relu inplace mutates tensor and returns relu values.
    Description: Check relu inplace mutates tensor and returns relu values.
    Expectation: In-place ReLU updates the original input tensor and returns the expected nonnegative values.
    """
    x = tensor([-2., 0., 3.])
    result = nn.relu(x, inplace=True)
    np.testing.assert_array_equal(result.asnumpy(), [0., 0., 3.])
    np.testing.assert_array_equal(x.asnumpy(), [0., 0., 3.])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_attention_dropout_scales_uniform_probabilities():
    """
    Feature: Attention dropout scales uniform probabilities.
    Description: Check attention dropout scales uniform probabilities.
    Expectation: Attention dropout uses the expected mask and inverse-keep scaling on uniform probabilities.
    """
    ms.set_seed(123)
    q = tensor(np.zeros((1, 1, 16, 16)))
    value = tensor(np.eye(16).reshape(1, 1, 16, 16))
    output, attention = nn._scaled_dot_product_attention(q, q, value, None, .5, False, True, ms.float32)
    probabilities = attention.asnumpy()
    assert set(np.unique(probabilities)) == {0, .125}
    np.testing.assert_array_equal(output.asnumpy(), probabilities)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='causal attention adds a bool triangular mask rather than masking future logits')
def test_causal_attention_excludes_future_values():
    """
    Feature: Causal attention excludes future values.
    Description: Check causal attention excludes future values.
    Expectation: Causal attention assigns no weight to future positions and returns the explicit prefix-weighted
        outputs.
    """
    q = tensor(np.zeros((1, 1, 2, 2)))
    value = tensor(np.eye(2).reshape(1, 1, 2, 2))
    output, attention = nn._scaled_dot_product_attention(q, q, value, None, 0., True, False, ms.float32)
    expected = np.array([[[[1., 0.], [.5, .5]]]], np.float32)
    np.testing.assert_allclose(attention.asnumpy(), expected, atol=1e-7)
    np.testing.assert_allclose(output.asnumpy(), expected, atol=1e-7)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='float attention mask adds bool padding as +1 rather than excluding padded positions')
def test_float_attention_mask_combined_with_bool_padding_excludes_padding():
    """
    Feature: Float attention mask combined with bool padding excludes padding.
    Description: Check float attention mask combined with bool padding excludes padding.
    Expectation: Combining a floating attention mask and boolean padding mask fully excludes the padded
        positions.
    """
    args, (_, _, v) = attention_extension_inputs()
    args['attn_mask'] = tensor(np.zeros((2, 2)))
    args['key_padding_mask'] = tensor([[False, True]], ms.bool_)
    output, attention = nn.multi_head_attention_forward(**args)
    np.testing.assert_allclose(attention.asnumpy(), [[[1., 0.], [1., 0.]]], atol=1e-7)
    np.testing.assert_allclose(output.asnumpy()[:, 0], np.broadcast_to(v[0], (2, 2)), atol=1e-7)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dim', [2, 3])
def test_fractional_pool_integer_input_contract(dim):
    """
    Feature: Fractional pool integer input contract.
    Description: Check fractional pool integer input contract.
    Expectation: Invalid fractional pool integer input contract raise TypeError. Diagnostics identify same
        dtype.
    """
    if dim == 2:
        result = nn.fractional_max_pool2d(tensor(np.arange(4).reshape(1, 1, 2, 2), ms.int32),
                                          2, output_size=1)
        np.testing.assert_array_equal(result.asnumpy(), [[[[3]]]])
    else:
        with pytest.raises(TypeError, match='same dtype'):
            nn.fractional_max_pool3d(tensor(np.ones((1, 1, 3, 3, 3)), ms.int32), 2, output_size=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_adaptive_max_pool1d_rejects_non_tensor():
    """
    Feature: Adaptive max pool1d rejects non tensor.
    Description: Check adaptive max pool1d rejects non tensor.
    Expectation: Invalid adaptive max pool1d non tensor raise TypeError. Diagnostics identify must be tensor.
    """
    with pytest.raises(TypeError, match='must be tensor'):
        nn.adaptive_max_pool1d([], 1)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level2', card_mark='onecard')
@pytest.mark.skipif(os.environ.get('MS_NN_NATIVE_TARGET') != 'Ascend', reason='Ascend dtype-specific validation')
@pytest.mark.usefixtures('ascend_context')
def test_ascend_adaptive_max_pool1d_requires_float16():
    """
    Feature: Ascend adaptive max pool1d requires float16.
    Description: Check ascend adaptive max pool1d requires float16.
    Expectation: Invalid ascend adaptive max pool1d requires float16 raise TypeError. Diagnostics identify
        Ascend.*float16.
    """
    with pytest.raises(TypeError, match='Ascend.*float16'):
        nn.adaptive_max_pool1d(tensor(np.ones((1, 1, 4))), 2)
    result = nn.adaptive_max_pool1d(tensor([[[1., 3., 2., 5.]]], ms.float16), 2)
    np.testing.assert_array_equal(result.asnumpy(), [[[3., 5.]]])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('fn',
    [ pytest.param('interpolate',
    marks=pytest.mark.xfail( strict=True,
    raises=TypeError,
    reason='dynamic rank interpolate cannot join scalar and dynamic tuple branches')),
     'interpolate_ext'])
@pytest.mark.parametrize('use_scale', [False, True])
def test_dynamic_rank_resize_uses_runtime_spatial_dimensions(fn, use_scale):
    """
    Feature: Dynamic rank resize uses runtime spatial dimensions.
    Description: Check dynamic rank resize uses runtime spatial dimensions.
    Expectation: A dynamic-rank resize derives the requested spatial dimensions at runtime and preserves the
        expected sample values.
    """
    class Resize(ms.nn.Cell):
        def construct(self, x):
            if use_scale:
                return getattr(nn, fn)(x, scale_factor=2., recompute_scale_factor=True)
            return getattr(nn, fn)(x, size=4)

    previous = ms.get_context('mode')
    try:
        ms.set_context(mode=ms.GRAPH_MODE)
        net = Resize()
        net.set_inputs(ms.Tensor(shape=None, dtype=ms.float32))
        result = net(tensor([[[1., 2.]]]))
        np.testing.assert_array_equal(result.asnumpy(), [[[1., 1., 2., 2.]]])
    finally:
        ms.set_context(mode=previous)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_pad_ext_empty_tuple_returns_original():
    """
    Feature: Pad ext empty tuple returns original.
    Description: Check pad ext empty tuple returns original.
    Expectation: Extended padding with an empty tuple returns the original input tensor.
    """
    x = tensor([1., 2.])
    assert nn.pad_ext(x, ()) is x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_interpolate_ext_mutable_parameters_preserve_exact_sizes():
    """
    Feature: Interpolate ext mutable parameters preserve exact sizes.
    Description: Check interpolate ext mutable parameters preserve exact sizes.
    Expectation: Mutable size and scale parameters preserve the exact requested output sizes and
        nearest-neighbor values.
    """
    x = tensor(np.ones((1, 1, 3, 5)))
    assert nn._interploate_ext_make_tuple(x, ms.mutable(2)) == (2, 2)
    assert nn._interpolate_ext_scale_factor_convert_size(x, ms.mutable((.5, 1.5))) == (1, 7)
    result = nn.interpolate(x, scale_factor=ms.mutable((2., 2.)), recompute_scale_factor=True)
    np.testing.assert_array_equal(result.asnumpy(), np.ones((1, 1, 6, 10)))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=IndexError,
                   reason='NLLLoss scalar guard indexes shape[1] before checking rank')
def test_nllloss_scalar_input_reports_value_error():
    """
    Feature: Nllloss scalar input reports value error.
    Description: Check nllloss scalar input reports value error.
    Expectation: Invalid nllloss scalar input reports value error raise ValueError. Diagnostics identify dim.
    """
    with pytest.raises(ValueError, match='dim'):
        nn._nllloss_nd(tensor(1.), tensor(0, ms.int32))
