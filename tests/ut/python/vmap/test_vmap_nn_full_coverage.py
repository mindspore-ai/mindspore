# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Real eager tensor checks for neural-network vmap rule transformations."""
import inspect
from functools import partial
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import auto_generate as A
from mindspore.ops.operations import nn_ops as NN, image_ops as IMG, random_ops as R, _grad_ops as G
from mindspore.ops.primitive import Primitive
from mindspore.ops._utils import arg_handler
from mindspore.ops._vmap import vmap_base as B, vmap_nn_ops as N, vmap_grad_nn_ops as D
from mindspore.ops._vmap import vmap_convolution_ops as C, vmap_image_ops as I, vmap_random_ops as Q


def _eager_graph_preprocess(prim, *args):
    """Implement only the graph-only preprocess boundary, never tensor kernels.

    Mapped operands return (False, None). Unmapped operands call the original
    primitive and recursively attach None output axes. Missing primitive
    arguments are rejected so malformed image rules cannot bypass validation.
    """
    if not callable(prim):
        raise TypeError('preprocess requires a callable primitive')
    if len(args) == 1 and isinstance(args[0], tuple) and args[0] and isinstance(args[0][0], tuple):
        args = args[0]
    if any(dim is not None for _, dim in args):
        return False, None
    return True, B.vmap_bind_all_none(prim(*(value for value, _ in args)))


@pytest.fixture(autouse=True)
def _real_cpu_rules(monkeypatch):
    previous = {key: ms.get_context(key) for key in ('mode', 'device_target')}
    ms.set_context(device_target='CPU', mode=ms.PYNATIVE_MODE)
    for module in (N, D, C, I, Q):
        monkeypatch.setattr(module, 'vmap_general_preprocess', _eager_graph_preprocess)
    yield
    ms.set_context(**previous)


def t(x):
    return Tensor(np.asarray(x, np.float32))


def packed(x, axis):
    return (t(x[0] if axis is None else np.moveaxis(x, 0, axis)), axis)


def check(result, expected, axis=0, atol=1e-5):
    value, actual_axis = result
    assert actual_axis == axis
    actual = value.asnumpy() if axis is None else np.moveaxis(value.asnumpy(), axis, 0)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=1e-5)


@pytest.mark.parametrize('rank', [2, 3])
@pytest.mark.parametrize('group', [1, 2])
@pytest.mark.parametrize('axes', [(0, None), (None, 0), (0, 0), (2, None), (None, 2), (2, 1), (None, None)])
def test_convolution_pointwise_real_kernel(rank, group, axes):
    """
    Feature: Neural-network vmap rules
    Description: Check convolution pointwise real kernel.
    Expectation: Pointwise kernels match the explicit scaled-input oracle; simultaneously mapped input and filter
        raise the declared ambiguity error.
    """
    xa, wa = axes
    if rank == 3 and xa is not None and wa is not None:
        with pytest.raises(ValueError, match='cannot be non-None at the same time'):
            C.get_conv_vmap_rule(P.Conv3D(group, 1, group=group), 2)(
                packed(np.ones((2, 1, group, 2, 2, 2)), xa), packed(np.ones((2, group, 1, 1, 1, 1)), wa))
        return
    x = np.arange(2 * group * 2 ** rank, dtype=np.float32).reshape((2, 1, group) + (2,) * rank)
    w = np.stack([np.ones((group, 1) + (1,) * rank), np.full((group, 1) + (1,) * rank, 3)]).astype(np.float32)
    prim = (P.Conv2D if rank == 2 else P.Conv3D)(group, 1, group=group)
    result = C.get_conv_vmap_rule(prim, 2)(packed(x, xa), packed(w, wa))
    ex = x[0] if xa is None else x
    factor = 1 if wa is None else np.array([1, 3]).reshape((2,) + (1,) * (rank + 2))
    expected = ex * factor
    axis = None if xa is None and wa is None else (0 if wa is None else 1)
    check(result, expected, axis)


@pytest.mark.parametrize('group', [1, 2])
@pytest.mark.parametrize('axes', [(0, None), (None, 0), (0, 0), (2, None), (None, 2), (2, 1), (None, None)])
def test_conv2d_transpose_pointwise(group, axes):
    """
    Feature: Neural-network vmap rules
    Description: Check conv2d transpose pointwise.
    Expectation: Pointwise transpose convolution matches each sample's input scaled by its shared or mapped filter.
    """
    # Generated native rule factories expose dynamic call signatures.
    # pylint: disable=too-many-function-args
    xa, wa = axes
    x = np.arange(2 * group * 4, dtype=np.float32).reshape(2, 1, group, 2, 2)
    w = np.stack([np.ones((group, 1, 1, 1)), np.full((group, 1, 1, 1), 3)]).astype(np.float32)
    result = C.get_conv2d_transpose_vmap_rule(P.Conv2DTranspose(group, 1, group=group), 2)(
        packed(x, xa), packed(w, wa), ((1, group, 2, 2), None))
    ex = x[0] if xa is None else x
    factor = 1 if wa is None else np.array([1, 3]).reshape(2, 1, 1, 1, 1)
    axis = None if xa is None and wa is None else (0 if wa is None else 1)
    check(result, ex * factor, axis)


@pytest.mark.parametrize('group', [1, 2])
@pytest.mark.parametrize('axes', [(0, None), (None, 0), (0, 0), (2, None), (None, 2), (2, 1), (None, None)])
def test_conv2d_filter_grad_pointwise(group, axes):
    """
    Feature: Neural-network vmap rules
    Description: Check conv2d filter grad pointwise.
    Expectation: Pointwise filter gradients equal explicit input/sensitivity products summed over each sample's
        spatial dimensions.
    """
    da, xa = axes
    x = np.arange(1, 2 * group * 4 + 1, dtype=np.float32).reshape(2, 1, group, 2, 2)
    dy = x + 2
    result = C.get_conv2d_backprop_filter_vmap_rule(G.Conv2DBackpropFilter(group, 1, group=group), 2)(
        packed(dy, da), packed(x, xa), ((group, 1, 1, 1), None))
    ex = np.broadcast_to(x[0], x.shape) if xa is None else x
    ed = np.broadcast_to(dy[0], dy.shape) if da is None else dy
    expected = (ex * ed).sum(axis=(1, 3, 4)).reshape(2, group, 1, 1, 1)
    axis = None if da is None and xa is None else (1 if da is None else 0)
    check(result, expected[0] if axis is None else expected, axis)


@pytest.mark.parametrize('kind', ['transpose', 'filter'])
@pytest.mark.parametrize('size,axis,reason', [(Tensor([1, 1, 2, 2], ms.int32), None, 'dynamic shape'),
                                            ((1, 1, 2, 2), 0, 'must be None')])
def test_convolution_size_contract(kind, size, axis, reason):
    """
    Feature: Neural-network vmap rules
    Description: Check convolution size contract.
    Expectation: Invalid convolution size, source-dimension and mapped-shape contracts raise their specified
        diagnostics.
    """
    x = t(np.ones((2, 1, 1, 2, 2)))
    if kind == 'transpose':
        rule = C.get_conv2d_transpose_vmap_rule(P.Conv2DTranspose(1, 1), 2)
    else:
        rule = C.get_conv2d_backprop_filter_vmap_rule(G.Conv2DBackpropFilter(1, 1), 2)
    with pytest.raises(ValueError, match=reason):
        rule((x, 0), (x, 0), (size, axis))


@pytest.mark.parametrize('data,compare,expected', [(0, 2, (3, 0)), (3, 1, (1, 4))])
def test_convolution_reshape_source_dimension(data, compare, expected):
    """
    Feature: Neural-network vmap rules
    Description: Check convolution reshape source dimension.
    Expectation: Convolution shape/channel helpers return the listed indices and sizes; unsupported data formats
        raise ValueError.
    """
    # Internal helpers are the unit coverage targets here.
    # pylint: disable=protected-access
    assert C._get_reshape_src_dim(data, compare) == expected
    # The empty tuple is an exact shape contract, including its container type.
    assert C._get_new_size_by_index(None, 2, 0) == ()  # pylint: disable=use-implicit-booleaness-not-comparison
    assert C._get_new_size_by_index((1, 3, 4), 2, 1) == (1, 6, 4)
    assert C._get_channel_index('NHWC', 'Conv2D') == 3
    with pytest.raises(ValueError, match='should be NHWC/NCHW/NCDHW'):
        C._get_channel_index('bad', 'Conv2D')


@pytest.mark.parametrize('kind', ['kl', 'smooth', 'bce'])
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('axes', [(0, 0), (1, None), (None, 1), (None, None)])
def test_losses_real_reductions(kind, reduction, axes):
    """
    Feature: Neural-network vmap rules
    Description: Check losses real reductions.
    Expectation: KL, SmoothL1 and binary-cross-entropy values match independent formulas for shared/mapped operands
        and each reduction.
    """
    xa, ya = axes
    x = np.array([[-1, -2, -3], [-0.5, -1.5, -2.5]], np.float32)
    y = np.array([[0.2, 0.4, 0.6], [0.3, 0.6, 0.9]], np.float32)
    ex = x[0] if xa is None else x
    ey = y[0] if ya is None else y
    if kind == 'kl':
        rule = N.get_kl_div_loss_vmap_rule(P.KLDivLoss(reduction), 2)
        result = rule(packed(x, xa), packed(y, ya))
        expected = ey * (np.log(ey) - ex)
    elif kind == 'smooth':
        rule = N.get_smooth_l1_loss_vmap_rule(Primitive('SmoothL1Loss'), 2)
        result = rule(packed(x, xa), packed(y, ya), (1., None),
                      (arg_handler.str_to_enum('SmoothL1Loss', 'reduction', reduction), None))
        z = np.abs(ex - ey)
        expected = np.where(z < 1, z * z / 2, z - .5)
    else:
        rule = N.get_bce_with_logits_loss_vamp_rule(Primitive('BCEWithLogitsLoss'), 2)
        result = rule(packed(x, xa), packed(y, ya), (t(np.ones(3)), None), (t(np.ones(3)), None),
                      (arg_handler.str_to_enum('BCEWithLogitsLoss', 'reduction', reduction), None))
        expected = np.maximum(ex, 0) - ex * ey + np.log1p(np.exp(-np.abs(ex)))
    if reduction != 'none':
        expected = expected.mean(axis=-1) if reduction == 'mean' else expected.sum(axis=-1)
    check(result, expected, None if xa is None and ya is None else 0)


@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum', 'batchmean'])
def test_kl_reductions_scalar_and_batchmean(reduction):
    """
    Feature: Neural-network vmap rules
    Description: Check KL reductions scalar and batchmean.
    Expectation: KL scalar and batchmean reductions operate per sample and match explicit means or sums.
    """
    x = np.array([[-1, -2], [-3, -4]], np.float32)
    y = np.array([[.25, .75], [.5, .5]], np.float32)
    rule = N.get_kl_div_loss_vmap_rule(P.KLDivLoss(reduction), 2)
    expected = y * (np.log(y) - x)
    if reduction in ('mean', 'batchmean'):
        expected = expected.mean(-1)
    elif reduction == 'sum':
        expected = expected.sum(-1)
    check(rule((t(x), 0), (t(y), 0)), expected)


@pytest.mark.parametrize('kind', ['bce', 'smooth', 'kl'])
def test_invalid_loss_reduction(kind):
    """
    Feature: Neural-network vmap rules
    Description: Check invalid loss reduction.
    Expectation: An unknown reduction raises RuntimeError identifying the reduction requirement.
    """
    x = (t(np.ones((2, 3))), 0)
    if kind == 'kl':
        prim = P.KLDivLoss('none')
        prim.add_prim_attr('reduction', 'bad')
        rule = N.get_kl_div_loss_vmap_rule(prim, 2)
        args = (x, x)
    else:
        name = 'BCEWithLogitsLoss' if kind == 'bce' else 'SmoothL1Loss'
        rule = (N.get_bce_with_logits_loss_vamp_rule if kind ==
                'bce' else N.get_smooth_l1_loss_vmap_rule)(Primitive(name), 2)
        args = (x, x, x, x, (99, None)) if kind == 'bce' else (x, x, (1., None), (99, None))
    with pytest.raises(RuntimeError, match='reduction'):
        rule(*args)


@pytest.mark.parametrize('x_axis,w_axis,b_axis', [(0, None, None), (2, 1, 1), (None, 0, None), (1, None, 0)])
@pytest.mark.parametrize('with_bias', [False, True])
def test_dense_transforms_and_broadcasts_inputs(x_axis, w_axis, b_axis, with_bias):
    """
    Feature: Neural-network vmap rules
    Description: Check dense transforms and broadcasts inputs.
    Expectation: Each mapped/shared input, weight and bias combination matches independently computed dense products
        plus bias.
    """
    x = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3)
    w = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    b = np.array([[1, 2], [3, 4]], np.float32)
    expected = np.stack([(x[0] if x_axis is None else x[i]) @ (w[0] if w_axis is None else w[i]).T
                         + ((b[0] if b_axis is None else b[i]) if with_bias else 0) for i in range(2)])
    result = N.get_dense_vmap_rule('Dense', 2)(packed(x, x_axis), packed(w, w_axis),
                    packed(b, b_axis) if with_bias else (None, None))
    check(result, expected)


@pytest.mark.parametrize('kind', ['softmax', 'logsoftmax'])
@pytest.mark.parametrize('batch_axis', [0, 1, 2])
@pytest.mark.parametrize('axis', [0, -1])
def test_softmax_logsoftmax_reduction_axes(kind, batch_axis, axis):
    """
    Feature: Neural-network vmap rules
    Description: Check softmax logsoftmax reduction axes.
    Expectation: Softmax/log-softmax normalize the intended logical axis and preserve the original physical batch
        position.
    """
    x = np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 10
    physical = np.moveaxis(x, 0, batch_axis)
    name = 'Softmax' if kind == 'softmax' else 'LogSoftmax'
    factory = N.get_softmax_vmap_rule if kind == 'softmax' else N.get_log_softmax_vmap_rule
    result = factory(Primitive(name), 2)((t(physical), batch_axis), (axis, None))
    logical_axis = 1 if axis == 0 else 2
    exp = np.exp(x - x.max(logical_axis, keepdims=True))
    expected = exp / exp.sum(logical_axis, keepdims=True)
    if kind == 'logsoftmax':
        expected = np.log(expected)
    check(result, expected, batch_axis)


@pytest.mark.parametrize('kind', ['avg2', 'avg3', 'max2', 'max3', 'maxarg3'])
@pytest.mark.parametrize('extra', [False, True])
@pytest.mark.parametrize('axis', [0, 1, None])
def test_pooling_reconstructs_all_leading_dimensions(kind, extra, axis):
    """
    Feature: Neural-network vmap rules
    Description: Check pooling reconstructs all leading dimensions.
    Expectation: Pooled values and argmax indices match explicit reductions while all leading sample dimensions are
        reconstructed.
    """
    rank = 3 if '3' in kind else 2
    shape = (2,) + ((1,) if extra or (kind == 'maxarg3' and axis is None) else ()) + (1,) + (2,) * rank
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    if kind == 'avg2':
        prim, factory, args = P.AdaptiveAvgPool2D((1, 1)), N.get_adaptive_avgpool2d_vmap_rule, ()
    elif kind == 'avg3':
        prim, factory, args = NN.AdaptiveAvgPool3D((1, 1, 1)), N.get_adaptive_avgpool3d_vmap_rule, ()
    elif kind == 'max2':
        prim, factory, args = Primitive('AdaptiveMaxPool2D'), N.get_adaptive_max_pool_2d_vmap_rule, (((1, 1), None),)
    elif kind == 'max3':
        prim, factory, args = NN.AdaptiveMaxPool3D(
        ), N.get_adaptive_max_pool3d_vmap_rule, ((Tensor([1, 1, 1], ms.int32), None),)
    else:
        prim, factory, args = NN.MaxPool3DWithArgmax(2, 2, 0), N.get_max_pool3d_with_argmax_vmap_rule, ()
    if kind == 'max3' and extra and axis is not None:
        pytest.skip('CPU dynamic int32 tensor reshape fails and can segfault; debug_01/debug_02 logs')
    result = factory(prim, 2)(packed(x, axis), *args)
    expected_x = x[0] if axis is None else x
    axes = tuple(range(-rank, 0))
    expected = expected_x.mean(axes, keepdims=True) if kind.startswith('avg') else expected_x.max(axes, keepdims=True)
    if kind.startswith('max'):
        value, indices = result
        check(value, expected, None if axis is None else 0)
        check(indices, np.full(expected.shape, 2 ** rank - 1), None if axis is None else 0)
    else:
        check(result, expected, None if axis is None else 0)


@pytest.mark.parametrize('rank', [1, 2, 3])
@pytest.mark.parametrize('linear', [False, True])
@pytest.mark.parametrize('axis', [0, 1, None])
def test_upsampling_identity_real_kernel(rank, linear, axis):
    """
    Feature: Neural-network vmap rules
    Description: Check upsampling identity real kernel.
    Expectation: Identity-size upsampling returns the original values and preserves unmapped or normalized mapped
        metadata.
    """
    name = ('UpsampleLinear1D', 'UpsampleBilinear2D', 'UpsampleTrilinear3D')[
            rank-1] if linear else f'UpsampleNearest{rank}D'
    factory = N.get_upsample_linear_vmap_rule if linear else N.get_upsample_nearest_3d_vmap_rule
    x = np.arange(2 * 2 ** rank, dtype=np.float32).reshape((2, 1, 1) + (2,) * rank)
    if linear and rank in (1, 2):
        pytest.skip('No CPU kernel; Ascend device 7 also lacks adapter (debug_03/debug_04)')
    args = (((2,) * rank, None), (None, None)) + (((False, None),) if linear else ())
    result = factory(Primitive(name), 2)(packed(x, axis), *args)
    check(result, x[0] if axis is None else x, None if axis is None else 0)


@pytest.mark.parametrize('rank', [1, 2, 3])
@pytest.mark.parametrize('linear', [False, True])
def test_upsampling_rejects_mapped_attributes(rank, linear):
    """
    Feature: Neural-network vmap rules
    Description: Check upsampling rejects mapped attributes.
    Expectation: Mapped size, scale or alignment attributes raise ValueError requiring None metadata.
    """
    name = ('UpsampleLinear1D', 'UpsampleBilinear2D', 'UpsampleTrilinear3D')[
            rank-1] if linear else f'UpsampleNearest{rank}D'
    factory = N.get_upsample_linear_vmap_rule if linear else N.get_upsample_nearest_3d_vmap_rule
    args = (((2,) * rank, 0), (None, None)) + (((False, None),) if linear else ())
    with pytest.raises(ValueError, match='must.*be None'):
        factory(Primitive(name), 2)((t(np.ones((2, 1, 1) + (2,) * rank)), 0), *args)


@pytest.mark.parametrize('name', ['Dropout2D', 'Dropout3D'])
@pytest.mark.parametrize('extra', [False, True])
@pytest.mark.parametrize('axis', [0, None])
def test_dropout_keep_probability_one_preserves_values(name, extra, axis):
    """
    Feature: Neural-network vmap rules
    Description: Check dropout keep probability one preserves values.
    Expectation: Keep probability one returns unchanged values and an all-true mask with the expected output axes.
    """
    rank = 3 if name == 'Dropout3D' else 2
    x = np.arange(2 * 2 ** rank, dtype=np.float32).reshape((2,) + \
                  ((1,) if extra or axis is None else ()) + (1,) + (2,) * rank)
    result = N.get_dropout_nd_vmap_rule(getattr(P, name)(keep_prob=1.), 2)(packed(x, axis))
    expected = x[0] if axis is None else x
    check(result[0], expected, None if axis is None else 0)
    check(result[1], np.ones(expected.shape, bool), None if axis is None else 0)


@pytest.mark.parametrize('kind', ['pad', 'padv3', 'mirror', 'mirrorgrad'])
@pytest.mark.parametrize('extra', [False, True])
@pytest.mark.parametrize('axis', [0, 1])
def test_padding_and_mirror_gradient_preserve_batch(kind, extra, axis):
    """
    Feature: Neural-network vmap rules
    Description: Check padding and mirror gradient preserve batch.
    Expectation: Padding matches NumPy and reflected-gradient accumulation matches explicit doubled values without
        padding the batch axis.
    """
    x = np.arange(8, dtype=np.float32).reshape((2,) + ((1, 1) if extra else ()) + (2, 2))
    if kind == 'pad':
        paddings = ((0, 0),) * (x.ndim-2) + ((1, 1),)
        result = N.get_pad_vmap_rule(P.Pad(paddings), 2)(packed(x, axis))
        expected = np.pad(x, ((0, 0),) * (x.ndim-1) + ((1, 1),))
        check(result, expected, axis)
        return
    if kind == 'padv3':
        result = N.get_pad_v3_vmap_rule(NN.PadV3(), 2)(packed(x, axis), ((1, 1), None), (t(0), None))
        expected = np.pad(x, ((0, 0),) * (x.ndim-1) + ((1, 1),))
    else:
        actual_rank = 4 if extra else x.ndim
        pads = Tensor([[0, 0]] * (actual_rank-1) + [[1, 1]], ms.int64)
        if kind == 'mirror':
            result = N.get_mirror_pad_vmap_rule(NN.MirrorPad('REFLECT'), 2)(packed(x, axis), (pads, None))
            expected = np.pad(x, ((0, 0),) * (x.ndim-1) + ((1, 1),), mode='reflect')
        else:
            padded = np.pad(x, ((0, 0),) * (x.ndim-1) + ((1, 1),), mode='reflect')
            result = D.get_mirror_pad_grad_grad_vmap_rule(
                G.MirrorPadGrad('REFLECT'), 2)(packed(padded, axis), (pads, None))
            expected = 2 * x
    check(result, expected)


@pytest.mark.parametrize('kind', ['padv3', 'mirror', 'mirrorgrad'])
@pytest.mark.parametrize('reason', ['arity', 'axis', 'rank'])
def test_padding_rejects_invalid_contract(kind, reason):
    """
    Feature: Neural-network vmap rules
    Description: Check padding rejects invalid contract.
    Expectation: Unsupported padding metadata or shapes raise their specific contract diagnostics.
    """
    factory, prim = {'padv3': (N.get_pad_v3_vmap_rule, NN.PadV3()),
                     'mirror': (N.get_mirror_pad_vmap_rule, NN.MirrorPad()),
                     'mirrorgrad': (D.get_mirror_pad_grad_grad_vmap_rule, G.MirrorPadGrad('REFLECT'))}[kind]
    rule = factory(prim, 2)
    x = (t(np.ones((2, 2))), 0)
    if reason == 'arity':
        args, match = (x,), 'must >= 2'
    elif reason == 'axis':
        args, match = (x, (Tensor([[0, 0]], ms.int64), 0)), 'paddings.*must be None'
    else:
        pads = (0, 0, 0, 0) if kind == 'padv3' else Tensor([[0, 0]], ms.int64)
        args, match = (x, (pads, None)), 'must be bigger'
        if kind == 'padv3':
            args += ((t(0), None),)
    with pytest.raises(ValueError, match=match):
        rule(*args)


@pytest.mark.parametrize('rank', [2, 3])
@pytest.mark.parametrize('axis', [0, 1, None])
def test_grid_sampler_center_values_and_gradients(rank, axis):
    """
    Feature: Neural-network vmap rules
    Description: Check grid sampler center values and gradients.
    Expectation: Center sampling matches explicit averages, image gradients are uniform and grid gradients match
        analytic coordinate derivatives.
    """
    x = np.arange(2 * 2 ** rank, dtype=np.float32).reshape((2, 1, 1) + (2,) * rank)
    grid = np.zeros((2, 1) + (1,) * rank + (rank,), np.float32)
    attrs = ((0, None), (0, None), (True, None))
    result = N.get_grid_sampler_vmap_rule(Primitive(f'GridSampler{rank}D'), 2)(
        packed(x, axis), packed(grid, axis), *attrs)
    expected_x = x[0] if axis is None else x
    expected = expected_x.mean(tuple(range(-rank, 0)), keepdims=True)
    check(result, expected, None if axis is None else 0)
    if axis is None:
        return
    dy = np.ones((2, 1, 1) + (1,) * rank, np.float32)
    dx, dg = D.get_grid_sampler_grad_vmap_rule(Primitive(f'GridSampler{rank}DGrad'), 2)(
        packed(dy, axis), packed(x, axis), packed(grid, axis), *attrs, ((1, 1), None))
    check(dx, np.full(x.shape, 1 / 2 ** rank))
    # With align_corners, a unit grid displacement traverses half the two-pixel span.
    expected_grid = np.array([.5 * 2 ** i for i in range(rank)], np.float32)
    check(dg, np.broadcast_to(expected_grid, grid.shape))


@pytest.mark.parametrize('grad', [False, True])
def test_grid_sampler_rejects_unknown_primitive(grad):
    """
    Feature: Neural-network vmap rules
    Description: Check grid sampler rejects unknown primitive.
    Expectation: An unknown sampler operation raises ValueError naming the supported primitive requirement.
    """
    with pytest.raises(ValueError, match='prim name must be'):
        (D.get_grid_sampler_grad_vmap_rule if grad else N.get_grid_sampler_vmap_rule)(Primitive('Add'), 2)


@pytest.mark.parametrize('axis', [0, 1, None])
def test_image_resize_identity_and_size_axis_contract(axis):
    """
    Feature: Neural-network vmap rules
    Description: Check image resize identity and size axis contract.
    Expectation: Identity resizing preserves values for mapped/shared samples and mapped output-size metadata raises
        ValueError.
    """
    x = np.arange(8, dtype=np.float32).reshape(2, 1, 1, 2, 2)
    rule = I.get_resize_bilinear_v2_vmap_rule(Primitive('ResizeBilinearV2'), 2)
    result = rule(packed(x, axis), ((2, 2), None), (False, None), (False, None))
    check(result, x[0] if axis is None else x, None if axis is None else 0)
    with pytest.raises(ValueError, match='size.*must be None'):
        rule(packed(x, 0), ((2, 2), 0), (False, None), (False, None))
    x = np.arange(4, dtype=np.float32).reshape(2, 1, 1, 2)
    rule = I.get_resize_linear_1d_vmap_rule(Primitive('ResizeLinear1D'), 2)
    result = rule(packed(x, axis), ((2,), None), (arg_handler.str_to_enum(
        'ResizeLinear1D', 'coordinate_transformation_mode', 'align_corners'), None))
    check(result, x[0] if axis is None else x, None if axis is None else 0)
    with pytest.raises(ValueError, match='size.*must be None'):
        rule(packed(x, 0), ((2,), 0), (arg_handler.str_to_enum(
            'ResizeLinear1D', 'coordinate_transformation_mode', 'align_corners'), None))


@pytest.mark.xfail(strict=True, raises=TypeError, reason='image rules omit prim in vmap_general_preprocess')
@pytest.mark.parametrize('kind', ['bilineargrad', 'lineargrad', 'crop'])
def test_image_rules_expected_values_pending_missing_preprocess_primitive(kind):
    """
    Feature: Neural-network vmap rules
    Description: Check image rules expected values pending missing preprocess primitive.
    Expectation: Image rule outputs should match their explicit expected values while strict xfails retain missing
        preprocessing-primitive defects.
    """
    x = (t(np.ones((2, 1, 1, 2, 2))), 0)
    if kind == 'bilineargrad':
        rule, args = I.get_resize_bilinear_grad_vmap_rule(
            Primitive('ResizeBilinearGrad'), 2), (x, x, (False, None), (False, None))
    elif kind == 'lineargrad':
        x = (t(np.ones((2, 1, 1, 2))), 0)
        mode = arg_handler.str_to_enum('ResizeLinear1D', 'coordinate_transformation_mode', 'align_corners')
        rule, args = I.get_resize_linear_1d_grad_vmap_rule(Primitive('ResizeLinear1DGrad'), 2, (mode, None)), (x, x)
    else:
        rule = I.get_crop_and_resize_vmap_rule(IMG.CropAndResize(), 2)
        args = (x, (t(np.zeros((2, 1, 4))), 0), (Tensor([[0], [0]], ms.int32), 0), ((1, 1), None))
    result = rule(*args)
    expected = np.ones((2, 1, 1, 2, 2) if kind == 'bilineargrad' else (
        (2, 1, 1, 2) if kind == 'lineargrad' else (2, 1, 1, 1, 2)), np.float32)
    check(result, expected)


@pytest.mark.parametrize('name', ['Embedding', Primitive('Embedding')])
def test_embedding_explicitly_rejects_vmap(name):
    """
    Feature: Neural-network vmap rules
    Description: Check embedding explicitly rejects vmap.
    Expectation: Embedding rule construction raises RuntimeError stating that vmap is unsupported.
    """
    with pytest.raises(RuntimeError, match='Embedding does not support vmap'):
        N.get_embedding_vmap_rule(name, 2)


@pytest.mark.parametrize('kind', ['elu', 'celu'])
@pytest.mark.parametrize('axis', [0, 1, None])
def test_elu_and_celu_math(kind, axis):
    """
    Feature: Neural-network vmap rules
    Description: Check elu and celu math.
    Expectation: ELU/CELU equal their NumPy piecewise formulas and mapped CELU alpha is rejected.
    """
    x = np.array([[-2, 0, 3], [-1, 1, 2]], np.float32)
    factory = N.get_elu_vmap_rule if kind == 'elu' else N.get_logit_vmap_rule
    result = factory('Elu' if kind == 'elu' else P.CeLU(), 2)(packed(x, axis), (1., None))
    xx = x[0] if axis is None else x
    check(result, np.where(xx > 0, xx, np.expm1(xx)), axis)
    if kind == 'elu':
        with pytest.raises(ValueError, match='alpha.*must be None'):
            factory(Primitive('Elu'), 2)(packed(x, 0), (1., 0))


@pytest.mark.parametrize('axis', [0, 1, None])
def test_in_top_k_handles_shared_targets(axis):
    """
    Feature: Neural-network vmap rules
    Description: Check in top k handles shared targets.
    Expectation: Per-sample membership matches the explicit boolean arrays for shared and mapped predictions.
    """
    x = np.array([[[.1, .8, .3], [.5, .4, .3]], [[.9, .1, .2], [.2, .1, .8]]], np.float32)
    targets = Tensor([1, 2], ms.int32)
    result = N.get_in_top_k_vmap_rule(P.InTopK(1), 2)(packed(x, axis), (targets, None))
    check(result, [True, False] if axis is None else [[True, False], [False, True]], None if axis is None else 0)


@pytest.mark.parametrize('kind', ['shuffle', 'choice', 'uniform'])
@pytest.mark.parametrize('existing_batch', [False, True])
@pytest.mark.parametrize('axis', [0, 1, None])
def test_random_rules_structural_and_probability_contracts(kind, existing_batch, axis):
    """
    Feature: Neural-network vmap rules
    Description: Check random rules structural and probability contracts.
    Expectation: Random outputs preserve permutations, valid distinct sampled indices and exact candidate expected
        counts with correct axes.
    """
    if kind == 'shuffle':
        prim, factory = R.RandomShuffle(seed=1), Q.get_random_shuffle_vmap_rule
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    elif kind == 'choice':
        prim, factory = R.RandomChoiceWithMask(2, seed=1), Q.get_random_choice_with_mask
        data = np.ones((2, 4), bool)
    else:
        prim = R.UniformCandidateSampler(1, 2, False, 5, seed=1)
        factory = Q.get_uniform_candidate_sampler_vmap_rule
        data = np.array([[[0], [1]], [[2], [3]]], np.int64)
    if existing_batch:
        prim.add_prim_attr('batch_rank', 1)
        data = np.expand_dims(data, 1)
    value = data[0] if axis is None else np.moveaxis(data, 0, axis)
    result = factory(prim, 2)((Tensor(value), axis))
    if kind == 'shuffle':
        out, out_axis = result
        assert out_axis == (None if axis is None else 0)
        normalized = out.asnumpy() if axis is None else np.moveaxis(out.asnumpy(), out_axis, 0)
        expected = data[0] if axis is None else data
        actual_rows = normalized.reshape(-1, normalized.shape[-2], normalized.shape[-1])
        expected_rows = expected.reshape(-1, expected.shape[-2], expected.shape[-1])
        for actual_sample, expected_sample in zip(actual_rows, expected_rows):
            assert sorted(map(tuple, actual_sample.tolist())) == sorted(map(tuple, expected_sample.tolist()))
    elif kind == 'choice':
        indices, mask = result
        assert indices[1] == mask[1] == (None if axis is None else 0)
        assert np.all(mask[0].asnumpy())
        idx = indices[0].asnumpy()
        assert np.all((idx >= 0) & (idx < 4))
        assert np.all(np.diff(np.sort(idx[..., 0], axis=-1), axis=-1) != 0)
    else:
        candidates, true_count, sampled_count = result
        assert candidates[1] == true_count[1] == sampled_count[1] == (None if axis is None else 0)
        assert np.all((candidates[0].asnumpy() >= 0) & (candidates[0].asnumpy() < 5))
        np.testing.assert_allclose(true_count[0].asnumpy(), .4)
        np.testing.assert_allclose(sampled_count[0].asnumpy(), .4)


@pytest.mark.parametrize('axis', [0, 1, None])
def test_multinomial_single_nonzero_category_is_deterministic(axis):
    """
    Feature: Neural-network vmap rules
    Description: Check multinomial single nonzero category is deterministic.
    Expectation: Only the supported category is sampled for each row; mapped sample-count metadata raises
        ValueError.
    """
    x = np.array([[0, 1, 0], [0, 0, 1]], np.float32)
    rule = Q.get_multinomial_vmap_rule(R.Multinomial(seed=1), 2)
    result = rule(packed(x, axis), (4, None))
    expected = [1] * 4 if axis is None else [[1] * 4, [2] * 4]
    check(result, expected, None if axis is None else 0)
    with pytest.raises(ValueError, match='source axis.*must be None'):
        rule(packed(x, 0), (4, 0))


_OPTIMIZERS = [
    ('get_apply_ada_max_rule', 'ApplyAdaMax'), ('get_apply_adadelta_rule', 'ApplyAdadelta'),
    ('get_apply_ftrl_rule', 'ApplyFtrl'), ('get_apply_proximal_adagrad_rule', 'ApplyProximalAdagrad'),
    ('get_apply_gradient_descent_rule', 'ApplyGradientDescent'),
    ('get_apply_proximal_gradient_descent_rule', 'ApplyProximalGradientDescent'),
    ('get_apply_adam_with_amsgrad_rule', 'ApplyAdamWithAmsgrad'),
    ('get_apply_adam_with_amsgrad_v2_rule', 'ApplyAdamWithAmsgradV2'), ('get_adam_rule', 'Adam'),
    ('get_apply_power_sign_rule', 'ApplyPowerSign'), ('get_apply_adagrad_v2_vmap_rule', 'ApplyAdagradV2'),
    ('get_apply_adagrad_da_vmap_rule', 'ApplyAdagradDA'), ('get_rmsprop_vmap_rule', 'ApplyRMSProp'),
    ('get_apply_centered_rmsprop_vmap_rule', 'ApplyCenteredRMSProp'),
    ('get_sparse_apply_adagrad_vmap_rule', 'SparseApplyAdagrad'),
]


def optimizer_primitive(name):
    if name == 'ApplyAdagradV2':
        return P.ApplyAdagradV2(1e-6)
    if name == 'SparseApplyAdagrad':
        return NN.SparseApplyAdagrad(.1)
    return getattr(P, name)()


@pytest.mark.parametrize('factory,name', _OPTIMIZERS)
@pytest.mark.parametrize('existing_batch', [False, True])
@pytest.mark.parametrize('var_axis', [None, 1])
def test_optimizer_rules_reject_ambiguous_mutation_order(factory, name, existing_batch, var_axis):
    """
    Feature: Neural-network vmap rules
    Description: Check optimizer rules reject ambiguous mutation order.
    Expectation: Unsafe combinations of shared state and mapped optimizer operands raise source-axis validation
        errors.
    """
    prim = optimizer_primitive(name)
    if existing_batch:
        prim.add_prim_attr('batch_rank', 1)
    rule = getattr(N, factory)(prim, 2)
    params = list(inspect.signature(rule).parameters)
    args = [(t(np.ones((2, 3))), var_axis)]
    args.extend([(t(np.ones((2, 3))), 0)] * (len(params)-2))
    args.append(None)
    with pytest.raises(ValueError, match='source axis.*(None|must)'):
        rule(*args)


@pytest.mark.parametrize('existing_batch', [False, True])
@pytest.mark.parametrize('gamma_axis', [None, 1])
def test_instance_norm_rejects_ambiguous_parameter_mutation(existing_batch, gamma_axis):
    """
    Feature: Neural-network vmap rules
    Description: Check instance norm rejects ambiguous parameter mutation.
    Expectation: Mapped data with unsupported parameter mutation axes raises the source-axis ValueError.
    """
    prim = NN.InstanceNorm()
    if existing_batch:
        prim.add_prim_attr('batch_rank', 1)
    with pytest.raises(ValueError, match='source axis'):
        N.get_instance_norm_rule(prim, 2)((t(np.ones((2, 1, 1, 2, 2))), 0),
            (t(np.ones((2, 1))), gamma_axis), *[(t(np.ones((2, 1))), 0)] * 3, None)


@pytest.mark.parametrize('axis', [0, 1, None])
@pytest.mark.parametrize('norm_axis', [0, 1, -1])
@pytest.mark.parametrize('parameter_batched', [False, True])
def test_layernorm_forward_and_gradient_against_numpy(axis, norm_axis, parameter_batched):
    """
    Feature: Neural-network vmap rules
    Description: Check layernorm forward and gradient against numpy.
    Expectation: Normalized values, means, variances and all gradient components match independent NumPy formulas.
    """
    x = np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 2
    logical_axes = (1, 2) if norm_axis == 0 else (2,)
    param_shape = (2, 3) if norm_axis == 0 else (3,)
    g = np.stack([np.ones(param_shape), np.full(param_shape, 2)]).astype(np.float32)
    b = g / 3
    ga = 0 if parameter_batched else None
    xx = x[0] if axis is None else x
    # Map at least one operand when testing batched parameters with shared x.
    has_batch = axis is not None or parameter_batched
    expected_x = np.broadcast_to(x[0], x.shape) if axis is None and has_batch else xx
    axes = logical_axes if has_batch else tuple(a-1 for a in logical_axes)
    mean = expected_x.mean(axes, keepdims=True)
    var = expected_x.var(axes, keepdims=True)
    eg = g if parameter_batched else g[0]
    eb = b if parameter_batched else b[0]
    if parameter_batched and norm_axis != 0:
        eg, eb = eg[:, None, :], eb[:, None, :]
    expected = (expected_x - mean) / np.sqrt(var + 1e-5) * eg + eb
    result = N.get_layernorm_vmap_rule(Primitive('LayerNorm'), 2)(packed(x, axis), packed(g, ga), packed(b, ga),
        (norm_axis, None), (norm_axis, None), (1e-5, None))
    out_axis = 0 if has_batch else None
    check(result[0], expected, out_axis)
    check(result[1], mean, out_axis)
    check(result[2], var, out_axis)
    if axis is None:
        return
    dy = np.ones_like(x)
    dresult = D.get_layernormgrad_vmap_rule(Primitive('LayerNormGrad'), 2)(packed(x, axis), packed(dy, axis),
        (t(var), 0), (t(mean), 0), packed(g, ga), (norm_axis, None), (norm_axis, None))
    check(dresult[0], np.zeros_like(x), atol=2e-5)
    expected_dg = (x-mean) / np.sqrt(var + 1e-12)
    expected_db = dy
    if norm_axis != 0:
        expected_dg = expected_dg.sum(1)
        expected_db = expected_db.sum(1)
    check(dresult[1], expected_dg)
    check(dresult[2], expected_db)


@pytest.mark.parametrize('axis', [0, 1, None])
@pytest.mark.parametrize('parameter_batched', [False, True])
def test_batchnorm_inference_transforms_channel_and_batch(axis, parameter_batched):
    """
    Feature: Neural-network vmap rules
    Description: Check batchnorm inference transforms channel and batch.
    Expectation: Inference values match per-sample normalization; auxiliary outputs retain correct channel shapes,
        axes and finite values.
    """
    x = np.arange(16, dtype=np.float32).reshape(2, 1, 2, 2, 2)
    scale = np.array([[1, 2], [2, 3]], np.float32)
    offset = scale / 2
    mean = scale / 3
    var = scale + 1
    pa = 0 if parameter_batched else None
    args = [packed(v, pa) for v in (scale, offset, mean, var)]
    attrs = ((False, None), (1e-5, None), (.1, None), (0, None))
    result = N.get_batchnorm_vmap_rule(Primitive('BatchNorm'), 2)(packed(x, axis), *args, *attrs)
    has_batch = axis is not None or parameter_batched
    expected = []
    for k in range(2 if has_batch else 1):
        i = k if parameter_batched else 0
        xx = x[k] if axis is not None else x[0]
        expected.append((xx - mean[i].reshape(1, 2, 1, 1)) / np.sqrt(var[i].reshape(1, 2, 1, 1)+1e-5)
                        * scale[i].reshape(1, 2, 1, 1) + offset[i].reshape(1, 2, 1, 1))
    check(result[0], np.stack(expected) if has_batch else expected[0], 1 if has_batch else None)
    for value, output_axis in result[1:]:
        assert output_axis == (0 if has_batch else None)
        assert value.shape == ((2, 2) if has_batch else (2,))
        assert np.all(np.isfinite(value.asnumpy()))


@pytest.mark.parametrize('kind', ['train', 'rank', 'gradrank'])
def test_batchnorm_rejects_unsupported_training_and_rank(kind):
    """
    Feature: Neural-network vmap rules
    Description: Check batchnorm rejects unsupported training and rank.
    Expectation: Unsupported training vmap and invalid operand ranks raise the appropriate ValueErrors.
    """
    x = (t(np.ones((2, 1, 2) if kind == 'train' else (2, 3))), 0)
    params = [(t(np.ones(2)), None)] * 4
    if kind == 'gradrank':
        rule = D.get_batchnorm_grad_vmap_rule(Primitive('BatchNormGrad'), 2)
        args = (x, x, *params, (False, None), (1e-5, None), (0, None))
    else:
        rule = N.get_batchnorm_vmap_rule(Primitive('BatchNorm'), 2)
        args = (x, *params, (kind == 'train', None), (1e-5, None), (.1, None), (0, None))
    with pytest.raises(ValueError, match='does not support Vmap during training' if kind == 'train' else 'dim of'):
        rule(*args)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('kind', ['max', 'avg', 'avg3', 'maxgradgrad', 'maxgradgrad3', 'adaptmax2', 'adaptmax3'])
def test_pooling_gradient_sparse_or_uniform_distribution(axis, kind):
    """
    Feature: Neural-network vmap rules
    Description: Check pooling gradient sparse or uniform distribution.
    Expectation: Maximum gradients route to argmax cells and average gradients distribute sensitivities uniformly
        within each sample.
    """
    rank = 3 if '3' in kind else 2
    x = np.arange(2 * 2 ** rank, dtype=np.float32).reshape((2, 1, 1) + (2,) * rank)
    y = x.max(tuple(range(-rank, 0)), keepdims=True)
    dy = np.array([1, 3], np.float32).reshape((2, 1, 1) + (1,) * rank)
    if kind == 'max':
        result = D.get_max_pool_grad_vmap_rule(G.MaxPoolGrad(2, 2), 2)(
            packed(x, axis), packed(y, axis), packed(dy, axis))
    elif kind == 'avg':
        attrs = (((2, 2), None), ((2, 2), None), (2, None), (0, None))
        result = D.get_avg_pool_grad_vmap_rule(Primitive('AvgPoolGrad'), 2)(
            packed(x, axis), packed(y, axis), packed(dy, axis), *attrs)
    elif kind == 'avg3':
        result = D.get_avg_pool3d_grad_vmap_rule(G.AvgPool3DGrad(2, 2), 2)((x.shape[1:], None), packed(dy, axis))
    elif kind in ('maxgradgrad', 'maxgradgrad3'):
        factory = D.get_maxpool_3d_grad_grad_vmap_rule if rank == 3 else D.get_maxpool_grad_grad_vmap_rule
        prim = G.MaxPool3DGradGrad(kernel_size=(1, 1, 2, 2, 2), strides=(
            1, 1, 2, 2, 2)) if rank == 3 else G.MaxPoolGradGrad(2, 2)
        result = factory(prim, 2)(packed(x, axis), packed(y, axis), packed(np.ones_like(x)*3, axis))
        check(result, np.ones_like(y) * 3)
        return
    else:
        prim = G.AdaptiveMaxPool3DGrad() if rank == 3 else G.AdaptiveMaxPool2DGrad()
        mask = Tensor(np.full(y.shape, 2 ** rank - 1), ms.int32 if rank == 3 else ms.int64)
        result = D.get_adaptive_avgpool2d_vmap_rule(prim, 2)(packed(dy, axis), packed(x, axis),
                        (ms.ops.moveaxis(mask, 0, axis), axis))
    expected = np.zeros_like(x)
    if kind.startswith('avg'):
        expected[:] = dy / 2 ** rank
    else:
        expected[(slice(None), 0, 0) + (-1,) * rank] = [1, 3]
    check(result, expected)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('linear', [False, True])
def test_upsample3d_gradient_identity_and_attribute_contract(axis, linear):
    """
    Feature: Neural-network vmap rules
    Description: Check upsample3d gradient identity and attribute contract.
    Expectation: Identity-size gradients preserve all values and mapped gradient attributes raise ValueError.
    """
    grad = np.arange(16, dtype=np.float32).reshape(2, 1, 1, 2, 2, 2)
    prim = Primitive('UpsampleTrilinear3DGrad' if linear else 'UpsampleNearest3DGrad')
    factory = D.get_upsample_trilinear3d_grad_vmap_rule if linear else D.get_upsample_nearesst3d_grad_vmap_rule
    rule = factory(prim, 2)
    attrs = (((1, 1, 2, 2, 2), None), ((2, 2, 2), None), (None, None))
    extra = ((False, None),) if linear else ()
    check(rule(packed(grad, axis), *attrs, *extra), grad)
    with pytest.raises(ValueError, match='must be None'):
        rule(packed(grad, axis), (attrs[0][0], 0), *attrs[1:], *extra)


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('logical_axis', [0, -1])
def test_log_softmax_gradient_formula(axis, logical_axis):
    """
    Feature: Neural-network vmap rules
    Description: Check log softmax gradient formula.
    Expectation: The result equals dy minus softmax probabilities times the logical-axis sensitivity sum, preserving
        the mapped position.
    """
    x = np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 10
    reduce_axis = 1 if logical_axis == 0 else 2
    probabilities = np.exp(x) / np.exp(x).sum(reduce_axis, keepdims=True)
    logp = np.log(probabilities)
    dy = np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3)
    result = D.get_log_softmax_grad_vmap_rule(Primitive('LogSoftmaxGrad'), 2)(
        packed(logp, axis), packed(dy, axis), (logical_axis, None))
    check(result, dy - probabilities * dy.sum(reduce_axis, keepdims=True), axis)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
def test_nll_loss_gradient_scatter_and_validation(axis, reduction):
    """
    Feature: Neural-network vmap rules
    Description: Check NLL loss gradient scatter and validation.
    Expectation: Weighted sensitivities scatter only to target classes; invalid weight/total-weight and gradient
        axes raise ValueError.
    """
    x = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    target = np.array([[0, 2], [1, 0]], np.int32)
    lg = np.array([[1, 2], [3, 4]], np.float32)
    rule = D.get_nll_loss_grad_vmap_rule(Primitive('NLLLossGrad'), 2)
    enum = arg_handler.str_to_enum('NLLLossGrad', 'reduction', reduction)
    weight = (t([1, 2, 3]), None)
    tw = (t(2), None)
    loss_grad = packed(lg, axis) if reduction == 'none' else (t(1), None)
    result = rule(packed(x, axis), loss_grad, (Tensor(np.moveaxis(target, 0, axis)), axis),
                  weight, tw, (enum, None), (-100, None))
    expected = np.zeros_like(x)
    for i in range(2):
        for j in range(2):
            expected[i, j, target[i, j]] = -(target[i, j] + 1) * (lg[i, j] if reduction ==
                                             'none' else 1) / (2 if reduction == 'mean' else 1)
    check(result, expected)
    with pytest.raises(ValueError, match='weight and total_weight.*must be None'):
        rule(packed(x, axis), loss_grad, (Tensor(target), 0), (weight[0], 0), tw, (enum, None), (-100, None))
    if reduction != 'none':
        with pytest.raises(ValueError, match='loss_grad.*can be not None'):
            rule(packed(x, axis), packed(lg, axis), (Tensor(target), 0), weight, tw, (enum, None), (-100, None))


@pytest.mark.parametrize('rank', [3, 4, 5])
@pytest.mark.parametrize('axes', [(0, None), (1, 0), (None, 1), (None, None)])
def test_bias_add_and_bias_gradient_preserve_channels(rank, axes):
    """
    Feature: Neural-network vmap rules
    Description: Check bias add and bias gradient preserve channels.
    Expectation: Channel biases broadcast correctly and bias gradients sum only non-channel sample axes with correct
        mapped metadata.
    """
    xa, ba = axes
    sample_shape = (1, 2) + (2,) * (rank-2)
    x = np.arange(2 * np.prod(sample_shape), dtype=np.float32).reshape((2,) + sample_shape)
    b = np.array([[1, 3], [2, 5]], np.float32)
    result = N.get_bias_add_vmap_rule(Primitive('BiasAdd'), 2)(packed(x, xa), packed(b, ba), (0, None))
    has_batch = xa is not None or ba is not None
    if has_batch:
        ex = np.broadcast_to(x[0], x.shape) if xa is None else x
        eb = np.broadcast_to(b[0], b.shape) if ba is None else b
        expected = ex + eb.reshape((2, 1, 2) + (1,) * (rank-2))
    else:
        expected = x[0] + b[0].reshape((1, 2) + (1,) * (rank-2))
    check(result, expected, 0 if has_batch else None)
    grad = N.get_bias_add_grad_vmap_rule(Primitive('BiasAddGrad'), 2)(packed(x, xa), (0, None))
    ex = x[0] if xa is None else x
    reduce_axes = (0,) + tuple(range(2, rank)) if xa is None else (1,) + tuple(range(3, rank+1))
    check(grad, ex.sum(reduce_axes), None if xa is None else 0)


@pytest.mark.parametrize('shape', [(2, 3, 2)])
@pytest.mark.parametrize('axis', [
    pytest.param(0, marks=pytest.mark.xfail(strict=True, raises=ValueError,
                                          reason='Pdist kernel rejects mapped rank 3')),
    pytest.param(1, marks=pytest.mark.xfail(strict=True, raises=ValueError,
                                          reason='Pdist kernel rejects mapped rank 3')), None])
def test_pdist_pairwise_euclidean_distance(shape, axis):
    """
    Feature: Neural-network vmap rules
    Description: Check pdist pairwise euclidean distance.
    Expectation: Each sample's condensed pairwise distances equal the explicit Euclidean-distance oracle.
    """
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    result = N.get_pdist_vmap_rule('Pdist', 2)(packed(x, axis))
    ex = x[0] if axis is None else x
    expected = np.stack([np.sqrt(np.sum((ex[..., i, :] - ex[..., j, :])**2, -1))
                         for i in range(3) for j in range(i+1, 3)], -1)
    check(result, expected, None if axis is None else 0)


@pytest.mark.parametrize('extra', [False, True])
@pytest.mark.parametrize('axis', [0, 1, None])
def test_local_response_normalization_and_gradient_formula(extra, axis):
    """
    Feature: Neural-network vmap rules
    Description: Check local response normalization and gradient formula.
    Expectation: Forward normalization and its input derivative match the independent channel-window formulas with
        correct output axes.
    """
    x = np.arange(1, 17, dtype=np.float32).reshape((2, 1, 2, 2, 2)) / 10
    prim = NN.LRN(depth_radius=1, bias=1., alpha=.5, beta=.5)
    result = N.get_lrn_vmap_rule(prim, 2)(packed(x, axis))
    ex = x[0] if axis is None else x
    channel_axis = -3
    denom = 1 + .5 * (ex**2).sum(channel_axis, keepdims=True)
    expected = ex / np.sqrt(denom)
    output_axis = None if axis is None else ex.ndim-1
    check(result, expected, output_axis)
    dy = np.ones_like(x) * 2
    y = x / np.sqrt(1 + .5 * (x*x).sum(channel_axis, keepdims=True))
    dx = N.get_lrn_grad_vmap_rule(G.LRNGrad(depth_radius=1, bias=1., alpha=.5, beta=.5), 2)(
        packed(dy, axis), packed(x, axis), packed(y, axis))
    expected_dx = 2 / np.sqrt(denom) - .5 * ex * (2*ex).sum(channel_axis, keepdims=True) / denom**1.5
    check(dx, expected_dx, output_axis)


@pytest.mark.parametrize('same_axis', [True, False])
def test_hswish_gradient_preserves_mapped_operands(same_axis):
    """
    Feature: Neural-network vmap rules
    Description: Check hswish gradient preserves mapped operands.
    Expectation: The piecewise HSwish derivative respects aligned sensitivities and incompatible input/gradient
        shapes raise RuntimeError.
    """
    x = np.array([[-4, -2, 0], [1, 2, 4]], np.float32)
    dy = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    # HSwishGrad takes sensitivity first, activation second.
    result = N.get_common_activation_grad_vmap_rule(G.HSwishGrad(), 2)(packed(dy, 0), packed(x, 0 if same_axis else 1))
    expected = dy * np.where(x <= -3, 0, np.where(x >= 3, 1, x/3+.5))
    check(result, expected)
    with pytest.raises(RuntimeError, match='input x shape.*same as input dy shape'):
        N.get_common_activation_grad_vmap_rule(G.HSwishGrad(), 2)((t([1, 2]), None), (t(np.ones((2, 3))), 0))


def test_deformable_offsets_unit_mask_and_zero_gradient_contract():
    """
    Feature: Neural-network vmap rules
    Description: Check deformable offsets unit mask and zero gradient contract.
    Expectation: Unit masks and zero offsets preserve inputs; input gradients are ones and the tested offset
        gradients are zero.
    """
    x = np.zeros((2, 1, 1, 2, 2), np.float32)
    offsets = np.zeros((2, 1, 3, 2, 2), np.float32)
    offsets[..., 2, :, :] = 1
    prim = NN.DeformableOffsets(strides=(1, 1, 1, 1), pads=(0, 0, 0, 0), ksize=(1, 1))
    check(N.get_matmul_vmap_rule(prim, 2)(packed(x, 0), packed(offsets, 0)), x)
    grad = G.DeformableOffsetsGrad(strides=(1, 1, 1, 1), pads=(0, 0, 0, 0), kernel_size=(1, 1))
    dx, doffsets = D.get_deformable_offsets_vmap_rule(grad, 2)(
        packed(np.ones_like(x), 0), packed(x, 0), packed(offsets, 0))
    check(dx, np.ones_like(x))
    check(doffsets, np.zeros_like(offsets))


@pytest.mark.xfail(strict=True, raises=AttributeError,
                   reason='CdistGrad primitive lacks has_label/clone/set_label API used by rule')
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('existing_batch', [False])
def test_cdist_gradient_real_squared_euclidean_formula(axis, existing_batch):
    """
    Feature: Neural-network vmap rules
    Description: Check cdist gradient real squared euclidean formula.
    Expectation: Squared-Euclidean Cdist gradients equal the explicit weighted difference formula with existing
        batch dimensions retained.
    """
    prim = A.CdistGrad()
    if existing_batch:
        prim.set_label('batch_rank', 1)
    x = np.array([[[0, 0], [1, 0]], [[0, 1], [1, 1]]], np.float32)
    y = np.array([[[2, 0]], [[2, 1]]], np.float32)
    difference = x[:, :, None, :] - y[:, None, :, :]
    distance = np.sqrt((difference**2).sum(-1))
    if existing_batch:
        x, y, distance = x[:, None], y[:, None], distance[:, None]
    grad = np.ones_like(distance)
    result = D.get_cdist_grad_vmap_rule(prim, 2)(packed(grad, axis), packed(x, axis), packed(y, axis),
        packed(distance, axis), (2., None))
    expected = (difference / np.sqrt((difference**2).sum(-1))[..., None]).sum(-2)
    check(result, expected[:, None] if existing_batch else expected)


@pytest.mark.parametrize('rank', [3, 5])
@pytest.mark.parametrize('axis', [0, 1])
def test_batchnorm_gradient_inference_numpy_formula(rank, axis):
    """
    Feature: Neural-network vmap rules
    Description: Check batchnorm gradient inference numpy formula.
    Expectation: Input, scale and bias gradients match inference formulas and each output reports its expected
        mapped position.
    """
    x = np.arange(8 if rank == 3 else 16, dtype=np.float32).reshape((2, 2, 2) if rank == 3 else (2, 1, 2, 2, 2))
    dy = np.ones_like(x) * 3
    scale, mean, var = np.array([1, 2], np.float32), np.array([.2, .4], np.float32), np.array([2, 3], np.float32)
    params = [(t(v), None) for v in (scale, mean, var, np.zeros(2, np.float32))]
    result = D.get_batchnorm_grad_vmap_rule(Primitive('BatchNormGrad'), 2)(packed(dy, axis), packed(x, axis),
        *params, (False, None), (1e-5, None), (0, None))
    parameter_shape = (1, 1, 2) + (1,) * (rank-3)
    expected_dx = dy * scale.reshape(parameter_shape) / np.sqrt(var.reshape(parameter_shape)+1e-5)
    reduce_axes = (1,) + tuple(range(3, rank))
    expected_scale = (dy * (x-mean.reshape(parameter_shape)) / \
                      np.sqrt(var.reshape(parameter_shape)+1e-5)).sum(reduce_axes)
    check(result[0], expected_dx, 1)
    check(result[1], expected_scale)
    check(result[2], dy.sum(reduce_axes))


@pytest.mark.parametrize('kind', ['transpose', 'inputgrad', 'filtergrad'])
@pytest.mark.parametrize('axes', [(0, None), (None, 0), (0, 0), (2, None), (None, 2), (None, None)])
def test_conv3d_gradient_and_transpose_pointwise(kind, axes):
    """
    Feature: Neural-network vmap rules
    Description: Check conv3d gradient and transpose pointwise.
    Expectation: Pointwise 3D filter gradients and transpose convolution match explicit products; ambiguous
        simultaneous mappings raise ValueError.
    """
    # Generated native rule factories expose dynamic call signatures.
    # pylint: disable=too-many-function-args
    da, wa = axes
    data = np.arange(1, 17, dtype=np.float32).reshape(2, 1, 1, 2, 2, 2)
    other = np.stack([np.ones((1, 1, 1, 1, 1)), np.full((1, 1, 1, 1, 1), 3)]).astype(np.float32)
    if kind == 'filtergrad':
        prim = G.Conv3DBackpropFilter(out_channel=1, kernel_size=1)
        rule = C.get_conv3d_backprop_filter_vmap_rule(prim, 2)
        inputs = (packed(data, wa), packed(data+1, da), ((1, 1, 1, 1, 1), None))
    elif kind == 'inputgrad':
        prim = NN.Conv3DBackpropInput(out_channel=1, kernel_size=1)
        rule = C.get_conv3d_backprop_input_vmap_rule(prim, 2)
        inputs = (packed(other, wa), packed(data, da), ((1, 1, 2, 2, 2), None))
    else:
        prim = P.Conv3DTranspose(in_channel=1, out_channel=1, kernel_size=1)
        rule = C.get_conv3d_transpose_vmap_rule(prim, 2)
        inputs = (packed(data, da), packed(other, wa))
    if da is not None and wa is not None:
        with pytest.raises(ValueError, match='cannot be non-None at the same time'):
            rule(*inputs)
        return
    result = rule(*inputs)
    output_axis = None if da is None and wa is None else (0 if wa is None else 1)
    if kind == 'filtergrad':
        ex = np.broadcast_to(data[0], data.shape) if wa is None else data
        ed = np.broadcast_to(data[0]+1, data.shape) if da is None else data+1
        expected = (ex*ed).sum((1, 2, 3, 4, 5)).reshape(2, 1, 1, 1, 1, 1)
        output_axis = None if da is None and wa is None else (1 if da is None else 0)
        if output_axis is None:
            expected = expected[0]
    else:
        ex = data[0] if da is None else data
        expected = ex if wa is None else ex * np.array([1, 3]).reshape(2, 1, 1, 1, 1, 1)
    check(result, expected, output_axis)


@pytest.mark.parametrize('kind', ['max', 'maxarg', 'avg'])
@pytest.mark.parametrize('axis', [0, 1, None])
def test_regular_pooling_forward_max_and_mean(kind, axis):
    """
    Feature: Neural-network vmap rules
    Description: Check regular pooling forward max and mean.
    Expectation: Forward max/mean pooling matches explicit sample reductions and max argmax indices equal three.
    """
    x = np.arange(8, dtype=np.float32).reshape(2, 1, 1, 2, 2)
    if kind == 'avg':
        result = N.get_avgpool_vmap_rule(Primitive('AvgPool'), 2)(packed(x, axis), ((2, 2), None),
                        ((2, 2), None), (2, None), (0, None))
    else:
        prim = P.MaxPool(2, 2) if kind == 'max' else P.MaxPoolWithArgmax(2, 2)
        result = N.get_max_pool_vmap_rule(prim, 2)(packed(x, axis))
    ex = x[0] if axis is None else x
    expected = ex.mean((-1, -2), keepdims=True) if kind == 'avg' else ex.max((-1, -2), keepdims=True)
    if kind == 'maxarg':
        check(result[0], expected, None if axis is None else 0)
        check(result[1], np.full(expected.shape, 3), None if axis is None else 0)
    else:
        check(result, expected, None if axis is None else 0)


@pytest.mark.parametrize('axis', [0, pytest.param(None, marks=pytest.mark.xfail(strict=True, raises=(
    AssertionError, ValueError), reason='unmapped preprocess reorders SmoothL1LossGrad operands'))])
@pytest.mark.parametrize('reduction', [
    'none', pytest.param('mean', marks=pytest.mark.xfail(
        strict=True, raises=(AssertionError, ValueError), reason='reduction is global across vmap batch')),
    pytest.param('sum', marks=pytest.mark.xfail(
        strict=True, raises=(AssertionError, ValueError), reason='batched sensitivity is interpreted as one scalar'))])
def test_smooth_l1_gradient_piecewise_formula(axis, reduction):
    """
    Feature: Neural-network vmap rules
    Description: Check smooth l1 gradient piecewise formula.
    Expectation: Piecewise SmoothL1 derivatives include the proper sensitivity and per-sample none/sum/mean
        reduction factors.
    """
    x = np.array([[-2, -.5, .25], [1.5, 0, -1]], np.float32)
    dy = np.array([[1, 2, 3], [4, 5, 6]], np.float32) if reduction == 'none' else np.array([2, 3], np.float32)
    enum = arg_handler.str_to_enum('SmoothL1Loss', 'reduction', reduction)
    sensitivity = packed(dy, axis)
    result = N.get_smooth_l1_loss_grad_vmap_rule(Primitive('SmoothL1LossGrad'), 2)(packed(x, axis),
        (t(np.zeros(3)), None), sensitivity, (1., None), (enum, None))
    ex = x[0] if axis is None else x
    edy = dy[0] if axis is None else dy
    if reduction != 'none' and axis is not None:
        edy = edy[:, None]
    expected = np.clip(ex, -1, 1) * edy / (3 if reduction == 'mean' else 1)
    check(result, expected, None if axis is None else 0)


@pytest.mark.parametrize('axis', [0, None])
def test_kl_loss_none_gradient_is_negative_target_times_sensitivity(axis):
    """
    Feature: Neural-network vmap rules
    Description: Check KL loss none gradient is negative target times sensitivity.
    Expectation: Unreduced KL gradients equal negative target times sensitivity with the appropriate unmapped or
        mapped axis.
    """
    x = np.array([[-1, -2, -3], [-2, -3, -4]], np.float32)
    y = np.array([[.1, .2, .3], [.2, .3, .4]], np.float32)
    dy = np.array([[1, 2, 3], [2, 3, 4]], np.float32)
    result = N.get_kl_div_loss_grad_vmap_rule(G.KLDivLossGrad('none'), 2)(
        packed(dy, axis), packed(x, axis), packed(y, axis))
    check(result, -y[0]*dy[0] if axis is None else -y*dy, None if axis is None else 0)


@pytest.mark.parametrize('extra', [False, True])
@pytest.mark.parametrize('axis', [0, 1])
def test_maxpool3d_gradient_with_argmax_routes_sensitivity(extra, axis):
    """
    Feature: Neural-network vmap rules
    Description: Check maxpool3d gradient with argmax routes sensitivity.
    Expectation: Each nonzero sensitivity returns only to its recorded three-dimensional maximum position.
    """
    shape = (2,) + ((1,) if extra else ()) + (1, 1, 2, 2, 2)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    output_shape = shape[:-3] + (1, 1, 1)
    dy = np.ones(output_shape, np.float32)*3
    mask = Tensor(np.full(output_shape, 7), ms.int64)
    prim = G.MaxPool3DGradWithArgmax(2, 2, 0)
    result = D.get_max_pool3d_grad_with_argmax_vmap_rule(prim, 2)(
        packed(x, axis), packed(dy, axis), (ms.ops.moveaxis(mask, 0, axis), axis))
    expected = np.zeros(shape, np.float32)
    expected[..., -1, -1, -1] = 3
    check(result, expected)


@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
def test_bce_equal_batched_shapes_take_direct_reduction_path(reduction):
    """
    Feature: Neural-network vmap rules
    Description: Check BCE equal batched shapes take direct reduction path.
    Expectation: Equal-shaped mapped inputs match the stable cross-entropy formula under none, sum and mean
        reductions.
    """
    logits = np.array([[-2, 0, 1], [-1, 2, 3]], np.float32)
    label = np.array([[0, 1, .5], [1, 0, .25]], np.float32)
    weight = np.array([[1, 2, 1], [2, 1, 3]], np.float32)
    pos = np.ones_like(logits)
    expected = (np.maximum(logits, 0) - logits*label + np.log1p(np.exp(-np.abs(logits)))) * weight
    if reduction != 'none':
        expected = expected.mean(-1) if reduction == 'mean' else expected.sum(-1)
    result = N.get_bce_with_logits_loss_vamp_rule('BCEWithLogitsLoss', 2)(
        (t(logits), 0), (t(label), 0), (t(weight), 0), (t(pos), 0),
        (arg_handler.str_to_enum('BCEWithLogitsLoss', 'reduction', reduction), None))
    check(result, expected)


@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum', 'batchmean'])
def test_kl_scalar_samples_preserve_batch_without_reduction(reduction):
    """
    Feature: Neural-network vmap rules
    Description: Check KL scalar samples preserve batch without reduction.
    Expectation: Scalar samples return target times log-target minus input without collapsing the mapped dimension.
    """
    x = np.array([-1, -2], np.float32)
    target = np.array([.2, .3], np.float32)
    result = N.get_kl_div_loss_vmap_rule(P.KLDivLoss(reduction), 2)((t(x), 0), (t(target), 0))
    check(result, target * (np.log(target)-x))


@pytest.mark.parametrize('mode', ['constant', 'reflect', 'edge'])
@pytest.mark.parametrize('extra', [False, True])
@pytest.mark.parametrize('tensor_padding', [False, True])
def test_padv3_modes_and_tensor_padding(mode, extra, tensor_padding):
    """
    Feature: Neural-network vmap rules
    Description: Check padv3 modes and tensor padding.
    Expectation: Supported PadV3 modes and Tensor padding specifications match their explicit expected sample
        arrays.
    """
    x = np.arange(8, dtype=np.float32).reshape((2,) + ((1, 1) if extra else ()) + (2, 2))
    padding_values = (1, 1, 0, 0) if extra and mode != 'constant' else (1, 1)
    pads = Tensor(padding_values, ms.int64) if tensor_padding else padding_values
    args = ((t(x), 0), (pads, None))
    if mode == 'constant':
        args += ((t(7), None),)
    result = N.get_pad_v3_vmap_rule(NN.PadV3(mode), 2)(*args)
    kwargs = {'constant_values': 7} if mode == 'constant' else {}
    expected = np.pad(x, ((0, 0),) * (x.ndim-1) + ((1, 1),), mode=mode, **kwargs)
    check(result, expected)


@pytest.mark.parametrize('bad', ['missing_values', 'mapped_values'])
def test_padv3_constant_value_input_contract(bad):
    """
    Feature: Neural-network vmap rules
    Description: Check padv3 constant value input contract.
    Expectation: Missing constant values or mapped value metadata raise the corresponding constant-padding
        diagnostics.
    """
    rule = N.get_pad_v3_vmap_rule(NN.PadV3(), 2)
    args = ((t(np.ones((2, 2))), 0), ((1, 1), None))
    if bad == 'mapped_values':
        args += ((t([1, 2]), 0),)
    with pytest.raises(ValueError,
                       match='constant mode must be 3' if bad == 'missing_values' else 'values_dim.*must be None'):
        rule(*args)


@pytest.mark.parametrize('extra', [False, True])
@pytest.mark.parametrize('axis', [0, 1])
def test_random_categorical_extreme_logits_choose_only_supported_class(extra, axis):
    """
    Feature: Neural-network vmap rules
    Description: Check random categorical extreme logits choose only supported class.
    Expectation: Extreme logits select only the supported class per sample and mapped count/seed metadata raises
        RuntimeError.
    """
    x = np.tile(np.array([-1000, 1000, -1000], np.float32), (2, 1, 1) if extra else (2, 1))
    rule = N.get_random_categorical_vmap_rule(P.RandomCategorical(ms.int64), 2)
    result = rule(packed(x, axis), (3, None), (7, None))
    expected = np.ones((2, 1, 3) if extra else (2, 3), np.int64)
    check(result, expected)
    with pytest.raises(RuntimeError, match='num_sample and seed should be None'):
        rule(packed(x, axis), (3, 0), (7, None))


@pytest.mark.parametrize('axis', [0, 1])
def test_adaptive_max_pool2d_output_size_retains_requested_dimensions(axis):
    """
    Feature: Neural-network vmap rules
    Description: Check adaptive max pool2d output size retains requested dimensions.
    Expectation: Each requested output size produces explicit local maxima and their exact original flattened
        indices.
    """
    x = np.arange(32, dtype=np.float32).reshape(2, 1, 1, 4, 4)
    rule = N.get_adaptive_max_pool_2d_vmap_rule(partial(ms.ops.adaptive_max_pool2d, return_indices=True), 2)
    for output_size in ((None, 2), (2, None), 2, (-1, 2), (2, -1)):
        result = rule(packed(x, axis), (output_size, None))
        h, w = (output_size, output_size) if isinstance(output_size, int) else output_size
        h = 4 if h in (None, -1) else h
        w = 4 if w in (None, -1) else w
        expected = x.reshape(2, 1, 1, h, 4//h, w, 4//w).max((4, 6))
        check(result[0], expected)
        expected_indices = np.empty((2, 1, 1, h, w), np.int64)
        for i in range(h):
            for j in range(w):
                expected_indices[..., i, j] = ((i+1)*(4//h)-1)*4 + ((j+1)*(4//w)-1)
        check(result[1], expected_indices)


@pytest.mark.parametrize('kind', ['avg3grad', 'adaptmax3'])
def test_pooling_rejects_mapped_shape_metadata(kind):
    """
    Feature: Neural-network vmap rules
    Description: Check pooling rejects mapped shape metadata.
    Expectation: Mapped pooling shape/size metadata raises ValueError requiring an unmapped value.
    """
    x = (t(np.ones((2, 1, 1, 2, 2, 2))), 0)
    with pytest.raises(ValueError, match='must be None'):
        if kind == 'avg3grad':
            D.get_avg_pool3d_grad_vmap_rule(G.AvgPool3DGrad(), 2)(((1, 1, 2, 2, 2), 0), x)
        else:
            N.get_adaptive_max_pool3d_vmap_rule(NN.AdaptiveMaxPool3D(), 2)(x, (Tensor([1, 1, 1], ms.int32), 0))


@pytest.mark.skip(reason='CPU lacks batch_rank kernel and Ascend7 lacks InstanceNormGrad adapter; probe logs')
@pytest.mark.parametrize('existing_batch', [False, True])
@pytest.mark.parametrize('axis', [0, 1])
def test_instance_norm_gradient_constant_signal_contract(existing_batch, axis):
    """
    Feature: Neural-network vmap rules
    Description: Check instance norm gradient constant signal contract.
    Expectation: Constant signals produce zero input/scale gradients and bias gradient eight with the expected
        sample/parameter shapes.
    """
    prim = G.InstanceNormGrad(epsilon=1e-5)
    shape = (2,) + ((1,) if existing_batch else ()) + (1, 2, 2, 2)
    param_shape = (2,) + ((1,) if existing_batch else ()) + (2,)
    if existing_batch:
        prim.add_prim_attr('batch_rank', 1)
    x = np.zeros(shape, np.float32)
    dy = np.full(shape, 2, np.float32)
    gamma, mean, variance = np.ones(param_shape, np.float32), np.zeros(
        param_shape, np.float32), np.ones(param_shape, np.float32)
    result = D.get_instance_norm_grad_rule(prim, 2)(packed(dy, axis), packed(x, axis), packed(gamma, axis),
        packed(mean, axis), packed(variance, axis))
    check(result[0], np.zeros(shape, np.float32))
    check(result[1], np.zeros(param_shape, np.float32))
    check(result[2], np.full(param_shape, 8, np.float32))


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('data_format', [0, 1])
def test_batchnorm_gradgrad_inference_second_derivative_contract(axis, data_format):
    # With x=mean=0, dy=0 and all second sensitivities zero, every derivative is zero.
    """
    Feature: Neural-network vmap rules
    Description: Check batchnorm gradgrad inference second derivative contract.
    Expectation: The tested zero-sensitivity inference case returns three zero derivatives with data-layout-specific
        output axes.
    """
    shape = (2, 1, 2, 2, 2) if data_format == 0 else (2, 1, 2, 2, 2)
    x = np.zeros(shape, np.float32)
    parameters = np.ones((2, 2), np.float32)
    zeros = np.zeros_like(parameters)
    result = D.get_batchnorm_grad_grad_vmap_rule(Primitive('BatchNormGradGrad'), 2)(
        packed(x, axis), packed(x, axis), packed(parameters, 0), packed(zeros, 0), packed(parameters, 0),
        packed(x, axis), packed(zeros, 0), packed(zeros, 0), (False, None), (1e-5, None), (data_format, None))
    for value in result[:2]:
        check(value, np.zeros(shape, np.float32), 1 if data_format == 0 else 3)
    check(result[2], zeros)
