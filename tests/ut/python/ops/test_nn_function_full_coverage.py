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
"""Attention projection, mask validation and convolution padding contracts."""
import numpy as np
import pytest

import mindspore as ms
from mindspore.ops.function import nn_func as nn


@pytest.fixture(scope='module', autouse=True)
def cpu_context():
    """Use real CPU kernels and restore context after the module."""
    previous = {key: ms.get_context(key) for key in ('mode', 'device_target')}
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
    yield
    ms.set_context(**previous)


def attention_inputs(projection='qkv', unbatched=False):
    """Use distinct projections and values so incorrect Q/K/V routing is observable."""
    q = np.arange(24, dtype=np.float32).reshape(3, 2, 4) / 10 - 0.5
    k = q[::-1].copy() / 2 + 0.2
    v = np.cos(q).astype(np.float32)
    if projection == 'self':
        k, v = q, q
    elif projection == 'kv':
        v = k
    if unbatched:
        q, k, v = q[:, 0], k[:, 0], v[:, 0]
    weights = [np.eye(4, dtype=np.float32) * scale for scale in (1, 0.5, 2)]
    params = dict(query=ms.Tensor(q.copy()), key=ms.Tensor(k.copy()), value=ms.Tensor(v.copy()),
                  embed_dim_to_check=4, num_heads=2, in_proj_weight=ms.Tensor(np.concatenate(weights)),
                  in_proj_bias=None, bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0.0,
                  out_proj_weight=ms.Tensor(np.diag([1., -2., 0.5, 1.5]).astype(np.float32)),
                  out_proj_bias=ms.Tensor([0.1, -0.2, 0.3, -0.4], ms.float32), training=False,
                  k_is_v=projection in ('self', 'kv'), q_is_k=projection == 'self')
    if projection == 'separate':
        params.update(use_separate_proj_weight=True, q_proj_weight=ms.Tensor(weights[0]),
                      k_proj_weight=ms.Tensor(weights[1]), v_proj_weight=ms.Tensor(weights[2]))
    return params, (q, k, v)


@pytest.mark.parametrize('projection', ['self', 'kv', 'qkv', 'separate'])
@pytest.mark.parametrize('unbatched', [False, True])
@pytest.mark.parametrize('mask_kind', ['none', 'bool', 'float', 'padding', 'combined'])
def test_multihead_attention_matches_independent_scaled_dot_product(projection, unbatched, mask_kind):
    """All projection routes and mask combinations must retain head/sample alignment."""
    params, (query, key, value) = attention_inputs(projection, unbatched)
    batch = 1 if unbatched else 2
    params['average_attn_weights'] = mask_kind != 'float'
    logits_mask = np.zeros((3, 3), np.float32)
    padding_mask = np.zeros((batch, 3), bool)
    if mask_kind in ('bool', 'combined'):
        bool_mask = np.zeros((3, 3), bool)
        bool_mask[0, 2] = True
        params['attn_mask'] = ms.Tensor(bool_mask)
        logits_mask[bool_mask] = -np.inf
    elif mask_kind == 'float':
        logits_mask[0, 2] = -1.25
        params['attn_mask'] = ms.Tensor(logits_mask.copy())
    if mask_kind in ('padding', 'combined'):
        padding_mask[:, 1] = True
        params['key_padding_mask'] = ms.Tensor(padding_mask[0] if unbatched else padding_mask)
    result, result_weights = nn.multi_head_attention_forward(**params)
    if unbatched:
        query, key, value = query[:, None], key[:, None], value[:, None]
    q = query.reshape(3, batch, 2, 2).transpose(1, 2, 0, 3)
    k = (key * 0.5).reshape(3, batch, 2, 2).transpose(1, 2, 0, 3)
    v = (value * 2).reshape(3, batch, 2, 2).transpose(1, 2, 0, 3)
    scores = np.matmul(q, k.swapaxes(-1, -2)) / np.sqrt(2)
    scores = scores + logits_mask
    scores = np.where(padding_mask[:, None, None, :], -np.inf, scores)
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = np.matmul(probabilities, v).transpose(2, 0, 1, 3).reshape(3, batch, 4)
    expected = expected * np.array([1., -2., 0.5, 1.5]) + np.array([0.1, -0.2, 0.3, -0.4])
    expected_weights = probabilities.mean(axis=1) if params['average_attn_weights'] else probabilities
    if unbatched:
        expected, expected_weights = expected[:, 0], expected_weights[0]
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(result_weights.asnumpy(), expected_weights, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('case,reason', [
    ('embed', 'embed_dim'), ('heads', 'num_heads'), ('missing_packed', 'in_proj_weight'),
    ('missing_q', 'q_proj_weight'), ('missing_k', 'k_proj_weight'), ('missing_v', 'v_proj_weight'),
    ('kv_shape', 'shape of `key`'), ('padding_dtype', 'key_padding_mask'),
    ('mask_dtype', 'attn_mask'), ('mask_size', 'shape of the `attn_mask`'),
    ('mask_rank', 'attn_mask'), ('padding_size', 'shape of `key_padding_mask`'),
    ('bias_pair', 'bias_k and bias_v'), ('static_k_batch', 'static_k'), ('static_v_width', 'static_v'),
])
def test_multihead_attention_rejects_inconsistent_configuration(case, reason):
    """Each invalid configuration must fail at its corresponding attention contract."""
    params, _ = attention_inputs('separate' if case.startswith('missing_') and case != 'missing_packed' else 'qkv')
    if case == 'embed':
        params['embed_dim_to_check'] = 5
    elif case == 'heads':
        params['num_heads'] = 3
    elif case == 'missing_packed':
        params['in_proj_weight'] = None
    elif case in ('missing_q', 'missing_k', 'missing_v'):
        params[case[-1] + '_proj_weight'] = None
    elif case == 'kv_shape':
        params['key'] = ms.Tensor(np.zeros((2, 2, 4), np.float32))
    elif case == 'padding_dtype':
        params['key_padding_mask'] = ms.Tensor(np.zeros((2, 3), np.int32))
    elif case == 'padding_size':
        params['key_padding_mask'] = ms.Tensor(np.zeros((2, 4), bool))
    elif case == 'mask_dtype':
        params['attn_mask'] = ms.Tensor(np.zeros((3, 3), np.int32))
    elif case == 'mask_size':
        params['attn_mask'] = ms.Tensor(np.zeros((4, 3), bool))
    elif case == 'mask_rank':
        params['attn_mask'] = ms.Tensor(np.zeros((3,), bool))
    elif case == 'bias_pair':
        params['bias_k'] = ms.Tensor(np.zeros((1, 1, 4), np.float32))
    elif case == 'static_k_batch':
        params['static_k'] = ms.Tensor(np.zeros((3, 3, 2), np.float32))
    elif case == 'static_v_width':
        params['static_v'] = ms.Tensor(np.zeros((4, 3, 3), np.float32))
    with pytest.raises(ValueError, match=reason):
        nn.multi_head_attention_forward(**params)


@pytest.mark.parametrize('ranks,expected', [((3, 3, 3), True), ((2, 2, 2), False)])
def test_attention_rank_validation_accepts_consistent_inputs(ranks, expected):
    """Batched and unbatched attention must select distinct layouts."""
    assert nn._check_qkv_shape(*ranks) is expected


@pytest.mark.parametrize('ranks', [(3, 2, 3), (2, 2, 3), (1, 1, 1)])
def test_attention_rank_validation_rejects_inconsistent_inputs(ranks):
    """Query rank determines the required key/value ranks."""
    with pytest.raises(ValueError):
        nn._check_qkv_shape(*ranks)


@pytest.mark.parametrize('padding,dim,expected', [((1,), 1, (1, 1)),
                                                ([1, 2], 2, (1, 1, 2, 2)),
                                                ((1, 2, 3), 3, (1, 1, 2, 2, 3, 3))])
def test_convolution_padding_duplicates_each_spatial_dimension(padding, dim, expected):
    """Convolution padding retains the spatial-axis ordering."""
    assert nn._manipulate_padding(padding, dim) == expected


@pytest.mark.parametrize('padding,dim,error', [('bad', 2, TypeError), ((1,), 2, ValueError)])
def test_convolution_padding_rejects_wrong_container_and_length(padding, dim, error):
    """Invalid padding must not be silently extended to missing spatial axes."""
    with pytest.raises(error):
        nn._manipulate_padding(padding, dim)


@pytest.mark.parametrize('value,expected', [(2, (1, 2)), ((3,), (1, 3)), ([4], (1, 4))])
def test_conv1d_parameter_normalization(value, expected):
    """A 1D convolution parameter is promoted to the expected 2D shape."""
    assert nn._dim_manipulation(value, 'stride') == expected


@pytest.mark.parametrize('value', [0, (0,), (1, 2), 'bad'])
def test_conv1d_parameter_normalization_rejects_invalid_values(value):
    """Zero, malformed sequences and strings are rejected independently."""
    with pytest.raises(ValueError, match='stride'):
        nn._dim_manipulation(value, 'stride')


@pytest.mark.parametrize('value,error', [(2, ValueError), ((1, 2), ValueError), ((1,), ValueError), ('1', TypeError)])
def test_same_padding_rejects_strided_convolution(value, error):
    """Same padding requires a unit stride along every spatial dimension."""
    with pytest.raises(error):
        nn._check_stride_when_same_mode(value)


@pytest.mark.parametrize('shape,padding,expected', [((1, 2, 3), None, ()),
                                                  ((2, 3, 4), (6, 7), (1, 2, 6, 7)),
                                                  ((1, 2, 3, 4), (1, 2, 6, 7), (1, 2, 6, 7))])
def test_max_unpool_output_shape_normalization(shape, padding, expected):
    """Unpool output sizes preserve batch/channel dimensions."""
    assert nn._normalize_max_unpool2d_output_size(shape, padding) == expected


@pytest.mark.parametrize('value', [[6, 7], (6,)])
def test_max_unpool_output_shape_rejects_invalid_size(value):
    """Unsupported output-size container and rank must be rejected."""
    with pytest.raises(ValueError):
        nn._normalize_max_unpool2d_output_size((1, 2, 3, 4), value)


@pytest.mark.parametrize('dilation,kernel,expected', [((1, 1), (3, 3), (False, (1, 1), (1, 1))),
                                                    ((1, 1), (2, 4), (True, (0, 1), (1, 2))),
                                                    ((2, 1), (2, 3), (False, (1, 1), (1, 1)))])
def test_same_padding_computation_handles_even_kernels(dilation, kernel, expected):
    """Even kernels must place the extra padding element on the correct side."""
    weight = ms.Tensor(np.ones((1, 1) + kernel, np.float32))
    assert nn._get_pad_info(dilation, weight) == expected


@pytest.mark.parametrize('left,right,expected', [((0, 1), (1, 2), ((0, 1, 0, 1), (0, 1))),
                                               ((2, 1), (1, 1), ((0, 0, -1, 0), (1, 1)))])
def test_asymmetric_padding_separates_crop_and_symmetric_components(left, right, expected):
    """Negative asymmetric deltas must preserve the smaller symmetric padding."""
    assert nn._get_pad_nd_info(left, right) == expected


@pytest.mark.parametrize('alpha', [0.2, 1, 2])
def test_leaky_relu_handles_slopes_above_and_below_one(alpha):
    """A slope above one must still apply only to the negative half-line."""
    data = np.array([-2., -0.5, 0., 1.5], np.float32)
    result = nn.leaky_relu(ms.Tensor(data), alpha)
    np.testing.assert_allclose(result.asnumpy(), np.where(data < 0, alpha * data, data), rtol=1e-6)


def test_threshold_and_hardtanh_boundaries():
    """Threshold equality and asymmetric clipping bounds must retain their exact semantics."""
    data = np.array([-2., -0.5, 0.5, 2.], np.float32)
    np.testing.assert_array_equal(nn.threshold(ms.Tensor(data), 0.5, -3).asnumpy(), [-3, -3, -3, 2])
    np.testing.assert_array_equal(nn.hardtanh(ms.Tensor(data), -0.5, 1.5).asnumpy(), [-0.5, -0.5, 0.5, 1.5])


@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('margin', [0, 0.5])
@pytest.mark.parametrize('integer_inputs', [False, True])
def test_cosine_embedding_loss_matches_geometric_definition(reduction, margin, integer_inputs):
    """Positive and negative pair labels select different cosine-loss branches."""
    dtype = np.int32 if integer_inputs else np.float32
    left = np.array([[1, 0], [1, 0], [0, 1]], dtype)
    right = np.array([[0, 1], [1, 0], [1, 0]], dtype)
    target = ms.Tensor([1, -1, -1], ms.int32)
    result = nn.cosine_embedding_loss(ms.Tensor(left), ms.Tensor(right), target, margin, reduction)
    expected = np.array([1., max(1 - margin, 0), 0.], np.float32)
    if reduction == 'mean':
        expected = expected.mean()
    elif reduction == 'sum':
        expected = expected.sum()
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=1e-6)


@pytest.mark.parametrize('margin,error', [(-1.5, ValueError), (1.5, ValueError), ('bad', TypeError)])
def test_cosine_embedding_loss_rejects_invalid_margin(margin, error):
    """Cosine margins must be numeric and remain within the cosine range."""
    with pytest.raises(error, match='margin'):
        nn.cosine_embedding_loss(ms.Tensor([[1., 0.]]), ms.Tensor([[1., 0.]]), ms.Tensor([1]), margin)


@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
def test_hinge_embedding_loss_uses_label_sign(reduction):
    """Only negative pairs are clipped against the margin."""
    result = nn.hinge_embedding_loss(ms.Tensor([-2., 0.25, 2.]), ms.Tensor([1, -1, -1], ms.float32),
                                     margin=1.0, reduction=reduction)
    expected = np.array([-2., 0.75, 0.], np.float32)
    if reduction == 'mean':
        expected = expected.mean()
    elif reduction == 'sum':
        expected = expected.sum()
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=1e-6)


@pytest.mark.parametrize('full', [False, True])
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
def test_gaussian_nll_loss_broadcasts_row_variance(full, reduction):
    """One variance per row must broadcast across predictions, with the optional Gaussian constant."""
    data = np.array([[1., 2.], [3., 4.]], np.float32)
    target = np.array([[0., 1.], [2., 2.]], np.float32)
    variance = np.array([1., 2.], np.float32)
    result = nn.gaussian_nll_loss(ms.Tensor(data), ms.Tensor(target), ms.Tensor(variance),
                                  full=full, reduction=reduction)
    expected = 0.5 * (np.log(variance[:, None]) + (data - target) ** 2 / variance[:, None])
    if full:
        expected += 0.5 * np.log(2 * np.pi)
    if reduction == 'mean':
        expected = expected.mean()
    elif reduction == 'sum':
        expected = expected.sum()
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('argument,value,error', [('x', [1], TypeError), ('target', [1], TypeError),
                                                ('var', [1], TypeError), ('full', 1, TypeError),
                                                ('eps', 0.0, ValueError), ('reduction', 'bad', ValueError)])
def test_gaussian_nll_loss_rejects_invalid_argument_contracts(argument, value, error):
    """All public Gaussian-loss arguments are independently validated."""
    params = dict(x=ms.Tensor([1.]), target=ms.Tensor([0.]), var=ms.Tensor([1.]))
    params[argument] = value
    with pytest.raises(error, match=argument):
        nn.gaussian_nll_loss(**params)


@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('weighted', [False, True])
def test_multilabel_soft_margin_loss_matches_binary_cross_entropy(reduction, weighted):
    """Class weights apply before per-sample class averaging."""
    data = np.array([[-1., 0., 1.], [2., -2., 0.5]], np.float32)
    target = np.array([[0., 1., 1.], [1., 0., 1.]], np.float32)
    weights = np.array([1., 2., 0.5], np.float32)
    result = nn.multilabel_soft_margin_loss(ms.Tensor(data), ms.Tensor(target),
                                            weight=ms.Tensor(weights) if weighted else None, reduction=reduction)
    expected = np.logaddexp(0, data) - target * data
    if weighted:
        expected *= weights
    expected = expected.mean(axis=-1)
    if reduction == 'mean':
        expected = expected.mean()
    elif reduction == 'sum':
        expected = expected.sum()
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('dtype', [np.float32, np.int32])
def test_mse_loss_casts_integer_inputs_and_applies_reduction(reduction, dtype):
    """Integer squared errors must produce a floating loss with the requested reduction."""
    left, right = np.array([1, 3, -2], dtype), np.array([0, 1, 1], dtype)
    result = nn.mse_loss(ms.Tensor(left), ms.Tensor(right), reduction)
    expected = np.array([1, 4, 9], np.float32)
    if reduction == 'mean':
        expected = expected.mean()
    elif reduction == 'sum':
        expected = expected.sum()
    assert result.dtype == ms.float32
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=1e-6)


@pytest.mark.parametrize('dimension', [1, 2, 3])
@pytest.mark.parametrize('groups', [1, 2])
@pytest.mark.parametrize('padding,stride', [(0, 1), (1, 2)])
def test_convolution_wrappers_match_direct_cross_correlation(dimension, groups, padding, stride):
    """Dimension promotion, grouping and bias placement must preserve correlation values."""
    shape = (1, 2) + (4,) * dimension
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 10
    weight_shape = (2, 2 // groups) + (2,) * dimension
    weight = np.arange(np.prod(weight_shape), dtype=np.float32).reshape(weight_shape) / 20 + 0.1
    bias = np.array([0.25, -0.5], np.float32)
    result = getattr(nn, f'conv{dimension}d')(ms.Tensor(data), ms.Tensor(weight), ms.Tensor(bias),
                                            stride=(stride,) * dimension, pad_mode='pad',
                                            padding=(padding,) * dimension, dilation=(1,) * dimension, groups=groups)
    spatial = ((4 + 2 * padding - 2) // stride + 1,) * dimension
    padded = np.pad(data, [(0, 0), (0, 0)] + [(padding, padding)] * dimension)
    expected = np.zeros((1, 2) + spatial, np.float32)
    channels_per_group = 2 // groups
    for channel in range(2):
        begin = (channel // channels_per_group) * channels_per_group
        for position in np.ndindex(spatial):
            window = tuple(slice(index * stride, index * stride + 2) for index in position)
            patch = padded[(0, slice(begin, begin + channels_per_group)) + window]
            expected[(0, channel) + position] = np.sum(patch * weight[channel]) + bias[channel]
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize('name,factor', [('pixel_shuffle', 2), ('pixel_unshuffle', 2)])
def test_pixel_rearrangement_preserves_element_positions(name, factor):
    """Channel-to-space and space-to-channel moves must keep pixel coordinates distinct."""
    if name == 'pixel_shuffle':
        data = np.arange(16, dtype=np.float32).reshape(1, 4, 2, 2)
        expected = np.array([[[[0, 4, 1, 5], [8, 12, 9, 13], [2, 6, 3, 7], [10, 14, 11, 15]]]], np.float32)
    else:
        data = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        expected = np.array([[[[0, 2], [8, 10]], [[1, 3], [9, 11]],
                              [[4, 6], [12, 14]], [[5, 7], [13, 15]]]], np.float32)
    np.testing.assert_array_equal(getattr(nn, name)(ms.Tensor(data), factor).asnumpy(), expected)


def test_hinge_embedding_loss_requires_floating_target_dtype():
    """The target's numeric representation must satisfy the documented dtype contract."""
    with pytest.raises(TypeError, match='second input must be float'):
        nn.hinge_embedding_loss(ms.Tensor([1.]), ms.Tensor([1], ms.int32))
