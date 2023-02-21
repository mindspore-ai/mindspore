# Copyright 2022 Huawei Technologies Co., Ltd
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

"""The vmap implement of grad operator corresponding to nn_ops."""
from __future__ import absolute_import

from __future__ import division
from functools import reduce
import mindspore.numpy as mnp
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops import functional as F
from mindspore.ops import constexpr
from mindspore.ops.primitive import _primexpr
from mindspore.ops.primitive import Primitive
from mindspore.ops.function import _VmapGeneralRule
from mindspore.ops._vmap.vmap_base import vmap_rules_getters, vmap_general_preprocess, _raise_value_error, \
    _bdim_at_front, _vmap_clone_prim, _vmap_update_prim_attr, _bdim_at_any, _handle_broadcasting


@vmap_rules_getters.register(G.NLLLossGrad)
def get_nll_loss_grad_vmap_rule(prim, axis_size):
    r"""
    VmapRule for NLLLossGrad operations.

    Limited by current kernel capability:
    1. Only support one dim batch for x, loss_grad and target.
    2. And weight only support shape as (C,), while total_weight should be a scalar.
    """

    @_primexpr
    def _get_reshape_shape(shape, keep_dim=0):
        new_batch_size = reduce(
            lambda x, y: x * y, shape if keep_dim == 0 else shape[:-keep_dim])
        return (new_batch_size,) if keep_dim == 0 else (new_batch_size, *shape[-keep_dim:])

    if isinstance(prim, str):
        prim = Primitive(prim)
        reduction_type = "none"
    else:
        reduction_type = prim.reduction

    def vmap_rule(x_bdim, loss_grad_bdim, target_bdim, weight_bdim, total_weight_bdim):
        is_all_none, result = vmap_general_preprocess(
            prim, x_bdim, loss_grad_bdim, target_bdim, weight_bdim, total_weight_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        loss_grad, lg_dim = loss_grad_bdim
        target, target_dim = target_bdim
        weight, w_dim = weight_bdim
        total_weight, tw_dim = total_weight_bdim

        x_shape = F.shape(x)
        loss_grad_shape = F.shape(loss_grad)
        target_shape = F.shape(target)
        base_x_len = len(x_shape) - (1 if x_dim is not None else 0)

        if w_dim is not None or tw_dim is not None:
            _raise_value_error("The source axis of weight and total_weight in `NLLLossGrad` must be None for now, "
                               "but got {} and {}.".format(w_dim, tw_dim))
        if lg_dim is not None and reduction_type != "none":
            _raise_value_error("The source axis of loss_grad in `NLLLossGrad` can be not None "
                               "just when reduction type is none for vmap, "
                               "but reduction type is {}.".format(reduction_type))

        # If stacked, move vmap_dim to first dim and reshape to right shape.
        if x_dim is not None:
            if x_dim != 0:
                x = mnp.moveaxis(x, x_dim, 0)
                x_shape = F.shape(x)
            if base_x_len == 2:
                x = F.reshape(x, _get_reshape_shape(x_shape, 1))

        if lg_dim is not None:
            if lg_dim != 0:
                loss_grad = mnp.moveaxis(loss_grad, lg_dim, 0)
                loss_grad_shape = F.shape(loss_grad)
            loss_grad = F.reshape(
                loss_grad, _get_reshape_shape(loss_grad_shape))

        if target_dim is not None:
            if target_dim != 0:
                target = mnp.moveaxis(target, target_dim, 0)
                target_shape = F.shape(target)
            target = F.reshape(target, _get_reshape_shape(target_shape))

        out = prim(x, loss_grad, target, weight, total_weight)
        output = F.reshape(out, x_shape)
        out_dim = 0
        return output, out_dim

    return vmap_rule


@vmap_rules_getters.register(G.MaxPoolGrad)
@vmap_rules_getters.register(G.AvgPoolGrad)
def get_avg_pool_grad_vmap_rule(prim, axis_size):
    """VmapRule for `AvgPoolGrad`."""
    chw_reverse_index = -3

    def vmap_rule(x_bdim, y_bdim, dy_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, y_bdim, dy_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        y, y_dim = y_bdim
        dy, dy_dim = dy_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        y = _bdim_at_front(y, y_dim, axis_size)
        dy = _bdim_at_front(dy, dy_dim, axis_size)
        x_shape = F.shape(x)
        y_shape = F.shape(y)
        dy_shape = F.shape(dy)
        x = F.reshape(x, (-1,) + x_shape[chw_reverse_index:])
        y = F.reshape(y, (-1,) + y_shape[chw_reverse_index:])
        dy = F.reshape(dy, (-1,) + dy_shape[chw_reverse_index:])
        out = prim(x, y, dy)
        out = F.reshape(out, x_shape)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(G.AvgPool3DGrad)
def get_avg_pool3d_grad_vmap_rule(prim, axis_size):
    """VmapRule for `AvgPool3DGrad`."""
    cdhw_reverse_index = -4

    def vmap_rule(shape_bdim, dy_bdim):
        is_all_none, result = vmap_general_preprocess(prim, shape_bdim, dy_bdim)
        if is_all_none:
            return result

        shape, shape_dim = shape_bdim
        dy, dy_dim = dy_bdim
        if shape_dim is not None:
            _raise_value_error("The source axis of 'origin_input_shape' in 'AvgPool3DGrad' must be None, "
                               "but got {}.".format(shape_dim))
        dy = _bdim_at_front(dy, dy_dim, axis_size)
        dy_shape = F.shape(dy)
        dy = F.reshape(dy, (-1,) + dy_shape[cdhw_reverse_index:])
        input_shape = (F.shape(dy)[0],) + shape[cdhw_reverse_index:]
        out = prim(input_shape, dy)
        out_shape = F.shape(out)
        return_shape = dy_shape[:cdhw_reverse_index] + out_shape[cdhw_reverse_index:]
        out = F.reshape(out, return_shape)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(G.MaxPool3DGradWithArgmax)
def get_max_pool3d_grad_with_argmax_vmap_rule(prim, axis_size):
    """VmapRule for `MaxPool3DGradWithArgmax`."""
    cdhw_reverse_index = -4

    def vmap_rule(x_bdim, dy_bdim, mask_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, dy_bdim, mask_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        dy, dy_dim = dy_bdim
        mask, mask_dim = mask_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        dy = _bdim_at_front(dy, dy_dim, axis_size)
        mask = _bdim_at_front(mask, mask_dim, axis_size)
        x_shape = F.shape(x)
        dy_shape = F.shape(dy)
        mask_shape = F.shape(mask)
        x_in_shape = (-1,) + x_shape[cdhw_reverse_index:]
        dy_in_shape = (-1,) + dy_shape[cdhw_reverse_index:]
        mask_in_shape = (-1,) + mask_shape[cdhw_reverse_index:]
        input_x = F.reshape(x, x_in_shape)
        input_dy = F.reshape(dy, dy_in_shape)
        input_mask = F.reshape(mask, mask_in_shape)
        out = prim(input_x, input_dy, input_mask)
        out = F.reshape(out, x_shape)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(G.CdistGrad)
def get_cdist_grad_vmap_rule(prim, axis_size):
    """VmapRule for `cdist grad` operation."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr("batch_rank", batch_rank)

    def vmap_rule(grad_bdim, x_bdim, y_bdim, cdist_bdim):
        is_all_none, result = vmap_general_preprocess(prim,
                                                      grad_bdim, x_bdim, y_bdim, cdist_bdim)
        if is_all_none:
            return result
        grad, grad_dim = grad_bdim
        x, x_dim = x_bdim
        y, y_dim = y_bdim
        cdist, cdist_dim = cdist_bdim

        grad = _bdim_at_front(grad, grad_dim, axis_size)
        x = _bdim_at_front(x, x_dim, axis_size)
        y = _bdim_at_front(y, y_dim, axis_size)
        cdist = _bdim_at_front(cdist, cdist_dim, axis_size)

        out = batch_prim(grad, x, y, cdist)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(G.AdaptiveMaxPool3DGrad)
@vmap_rules_getters.register(G.AdaptiveMaxPool2DGrad)
def get_adaptive_avgpool2d_vmap_rule(prim, axis_size):
    """VmapRule for `AdaptiveMaxPool2DGrad` and `AdaptiveMaxPool3DGrad` operation."""
    chw_reverse_index = -3
    if prim.name == "AdaptiveMaxPool2DGrad":
        hw_reverse_index = -2
    else:
        hw_reverse_index = -3

    def vmap_rule(ygrad_bdim, x_bdim, max_index_bdim):
        is_all_none, result = vmap_general_preprocess(prim, ygrad_bdim, x_bdim, max_index_bdim)
        if is_all_none:
            return result

        dy, dy_dim = ygrad_bdim
        in_x, in_x_dim = x_bdim
        max_idx, max_idx_dim = max_index_bdim

        dy = _bdim_at_front(dy, dy_dim, axis_size)
        in_x = _bdim_at_front(in_x, in_x_dim, axis_size)
        max_idx = _bdim_at_front(max_idx, max_idx_dim, axis_size)

        # expand out dim
        dy_shape = F.shape(dy)
        dy_shape_tmp = (-1,) + dy_shape[chw_reverse_index:]
        dy = F.reshape(dy, dy_shape_tmp)

        in_x_shape = F.shape(in_x)
        in_x_tmp_shape = (-1,) + in_x_shape[chw_reverse_index:]
        in_x = F.reshape(in_x, in_x_tmp_shape)

        max_idx_shape = F.shape(max_idx)
        max_idx_tmp_shape = (-1,) + max_idx_shape[chw_reverse_index:]
        max_idx = F.reshape(max_idx, max_idx_tmp_shape)

        # cal
        out = prim(dy, in_x, max_idx)
        out_shape = F.shape(out)
        real_out_shape = dy_shape[:hw_reverse_index] + out_shape[hw_reverse_index:]
        out = F.reshape(out, real_out_shape)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(G.BatchNormGradGrad)
def get_batchnorm_grad_grad_vmap_rule(prim, axis_size):
    """VmapRule for `BatchNormGradGrad` operation."""
    data_format = prim.format

    def vmap_rule(x_bdim, dy_bdim, scale_bdim, mean_bdim, variance_bdim, dout_dx_bdim,
                  dout_dscale_bdim, dout_dbias_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, dy_bdim, scale_bdim, mean_bdim, variance_bdim,
                                                      dout_dx_bdim, dout_dscale_bdim, dout_dbias_bdim)
        if is_all_none:
            return result

        dst_dim = 1 if data_format == "NCHW" else 3
        x = _bdim_at_any(*x_bdim, dst_dim, axis_size)
        dy = _bdim_at_any(*dy_bdim, dst_dim, axis_size)
        dout_dx = _bdim_at_any(*dout_dx_bdim, dst_dim, axis_size)

        scale = _bdim_at_front(*scale_bdim, axis_size)
        mean = _bdim_at_front(*mean_bdim, axis_size)
        variance = _bdim_at_front(*variance_bdim, axis_size)
        dout_dscale = _bdim_at_front(*dout_dscale_bdim, axis_size)
        dout_dbias = _bdim_at_front(*dout_dbias_bdim, axis_size)

        x_shape = x.shape
        scale_shape = scale.shape
        shape = (x_shape[0], -1,) + x_shape[3:] if data_format == "NCHW" else x_shape[:-2] + (-1,)
        dx, ddy, dscale = prim(x.reshape(shape), dy.reshape(shape), scale.flatten(), mean.flatten(),
                               variance.flatten(), dout_dx.reshape(shape), dout_dscale.flatten(),
                               dout_dbias.flatten())
        pos = 1 if data_format == "NCHW" else 3
        return (dx.reshape(x_shape), pos), (ddy.reshape(x_shape), pos), (dscale.reshape(scale_shape), 0)
    return vmap_rule


@vmap_rules_getters.register(G.DeformableOffsetsGrad)
def get_deformable_offsets_vmap_rule(prim, axis_size):
    """VmapRule for `DeformableOffsetsGrad` operation."""
    chw_reverse_index = -3

    def vmap_rule(dout_bdim, x_bdim, offsets_bdim):
        is_all_none, result = vmap_general_preprocess(prim, dout_bdim, x_bdim, offsets_bdim)
        if is_all_none:
            return result

        dout, dout_dim = dout_bdim
        x, x_dim = x_bdim
        offsets, offsets_dim = offsets_bdim

        dout = _bdim_at_front(dout, dout_dim, axis_size)
        dout_origin_shape = F.shape(dout)

        x = _bdim_at_front(x, x_dim, axis_size)
        x_origin_shape = F.shape(x)

        offsets = _bdim_at_front(offsets, offsets_dim, axis_size)
        offsets_origin_shape = F.shape(offsets)

        dout = F.reshape(dout, (-1,) + dout_origin_shape[chw_reverse_index:])
        x = F.reshape(x, (-1,) + x_origin_shape[chw_reverse_index:])
        offsets = F.reshape(offsets, (-1,) + offsets_origin_shape[chw_reverse_index:])

        dx, d_offsets = prim(dout, x, offsets)
        dx = F.reshape(dx, x_origin_shape)
        d_offsets = F.reshape(d_offsets, offsets_origin_shape)
        return (dx, 0), (d_offsets, 0)

    return vmap_rule


@vmap_rules_getters.register(G.BatchNormGrad)
def get_batchnorm_grad_vmap_rule(prim, axis_size):
    """VmapRule for `BatchNormGrad` operation."""
    bn_min_dim = 3
    bn_max_dim = 5
    data_format = prim.data_format
    prim_name = prim.name
    if data_format == "NHWC":
        batchnorm_grad_nhwc_vmap = _VmapGeneralRule(prim, axis_size)

    def vmap_rule(grad_bdim, x_bdim, scale_bdim, rsv_1_bdim, rsv_2_bdim, rsv_3_bdim):
        is_all_none, result = \
            vmap_general_preprocess(prim, grad_bdim, x_bdim, scale_bdim, rsv_1_bdim, rsv_2_bdim, rsv_3_bdim)
        if is_all_none:
            return result
        if data_format == "NHWC":
            # BatchNormGrad with NHWC format is a GPU backend operation and not supported for now.
            return batchnorm_grad_nhwc_vmap(grad_bdim, x_bdim, scale_bdim, rsv_1_bdim, rsv_2_bdim, rsv_3_bdim)
        grad, grad_dim = grad_bdim
        input_x, input_x_dim = x_bdim
        scale, scale_dim = scale_bdim
        rsv_1, rsv_1_dim = rsv_1_bdim
        rsv_2, rsv_2_dim = rsv_2_bdim
        rsv_3, rsv_3_dim = rsv_3_bdim
        x_ndim = F.rank(input_x)
        if x_ndim not in (bn_min_dim, bn_max_dim):
            raise ValueError("The dim of `input_x` in `{}` must be equal to {} or {}, "
                             "but got {}.".format(prim_name, bn_min_dim, bn_max_dim, x_ndim))
        # Move input_x and grad axis to the dim front of C
        out_axis = 1
        grad = _bdim_at_any(grad, grad_dim, out_axis, axis_size)
        input_x = _bdim_at_any(input_x, input_x_dim, out_axis, axis_size)
        scale = _bdim_at_front(scale, scale_dim, axis_size)
        rsv_1 = _bdim_at_front(rsv_1, rsv_1_dim, axis_size)
        rsv_2 = _bdim_at_front(rsv_2, rsv_2_dim, axis_size)
        rsv_3 = _bdim_at_front(rsv_3, rsv_3_dim, axis_size)
        x_shape = input_x.shape
        other_shape = scale.shape
        vmap_shape = (x_shape[0], -1,) + x_shape[3:]
        grad = F.reshape(grad, vmap_shape)
        input_x = F.reshape(input_x, vmap_shape)
        scale = scale.flatten()
        rsv_1 = rsv_1.flatten()
        rsv_2 = rsv_2.flatten()
        rsv_3 = rsv_3.flatten()
        grad_x, grad_scale, grad_offset = prim(grad, input_x, scale, rsv_1, rsv_2, rsv_3)
        grad_x = F.reshape(grad_x, x_shape)
        grad_scale = F.reshape(grad_scale, other_shape)
        grad_offset = F.reshape(grad_offset, other_shape)
        return (grad_x, out_axis), (grad_scale, 0), (grad_offset, 0)

    return vmap_rule


@vmap_rules_getters.register(G.MaxPoolGradGrad)
@vmap_rules_getters.register(G.MaxPoolGradGradWithArgmax)
def get_maxpool_grad_grad_vmap_rule(prim, axis_size):
    """VmapRule for `MaxPoolGradGrad` and `MaxPoolGradGradWithArgmax`."""
    chw_reverse_index = -3

    def vmap_rule(in0_bdim, in1_bdim, in2_bdim):
        is_all_none, result = vmap_general_preprocess(
            prim, in0_bdim, in1_bdim, in2_bdim)
        if is_all_none:
            return result

        in0, in0_dim = in0_bdim
        in0 = _bdim_at_front(in0, in0_dim, axis_size)
        in0_shape = F.shape(in0)
        input_shape = (-1,) + in0_shape[chw_reverse_index:]
        in0 = F.reshape(in0, input_shape)

        in1, in1_dim = in1_bdim
        in1 = _bdim_at_front(in1, in1_dim, axis_size)
        in1_shape = F.shape(in1)
        in1_shape = (-1,) + in1_shape[chw_reverse_index:]
        in1 = F.reshape(in1, in1_shape)

        in2, in2_dim = in2_bdim
        in2 = _bdim_at_front(in2, in2_dim, axis_size)
        in2_shape = F.shape(in2)
        in2_shape = (-1,) + in2_shape[chw_reverse_index:]
        in2 = F.reshape(in2, in2_shape)

        out = prim(in0, in1, in2)
        out_shape = F.shape(out)
        real_out_shape = in0_shape[:chw_reverse_index] + \
            out_shape[chw_reverse_index:]
        out = F.reshape(out, real_out_shape)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(G.MaxPool3DGradGrad)
def get_maxpool_3d_grad_grad_vmap_rule(prim, axis_size):
    """VmapRule for `MaxPool3DGradGrad`."""
    cdhw_reverse_index = -4

    def vmap_rule(x_bdim, y_bdim, grad_bdim):
        is_all_none, result = vmap_general_preprocess(prim, y_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        x_shape = F.shape(x)
        input_shape = (-1,) + x_shape[cdhw_reverse_index:]
        x = F.reshape(x, input_shape)

        y, y_dim = y_bdim
        y = _bdim_at_front(y, y_dim, axis_size)
        y_shape = F.shape(y)
        y_shape = (-1,) + y_shape[cdhw_reverse_index:]
        y = F.reshape(y, y_shape)

        grad, grad_dim = grad_bdim
        grad = _bdim_at_front(grad, grad_dim, axis_size)
        grad_shape = F.shape(grad)
        grad_shape = (-1,) + grad_shape[cdhw_reverse_index:]
        grad = F.reshape(grad, grad_shape)

        out = prim(x, y, grad)
        out_shape = F.shape(out)
        real_out_shape = x_shape[:cdhw_reverse_index] + \
            out_shape[cdhw_reverse_index:]
        out = F.reshape(out, real_out_shape)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(G.InstanceNormGrad)
def get_instance_norm_grad_rule(prim, axis_size):
    """VmapRule for `InstanceNormGrad` operation."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr('batch_rank', batch_rank)

    def vmap_rule(dy_bdim, x_bdim, gamma_bdim, mean_bdim, variance_bdim):
        dy, dy_dim = dy_bdim
        x, x_dim = x_bdim
        gamma, gamma_dim = gamma_bdim
        mean, mean_dim = mean_bdim
        variance, variance_dim = variance_bdim

        dy = _bdim_at_front(dy, dy_dim, axis_size)
        x = _bdim_at_front(x, x_dim, axis_size)
        gamma = _bdim_at_front(gamma, gamma_dim, axis_size)
        mean = _bdim_at_front(mean, mean_dim, axis_size)
        variance = _bdim_at_front(variance, variance_dim, axis_size)

        output_x, updated_moving_mean, updated_moving_variance = batch_prim(dy, x, gamma, mean, variance)
        return (output_x, 0), (updated_moving_mean, 0), (updated_moving_variance, 0)

    return vmap_rule


@vmap_rules_getters.register(G.MirrorPadGrad)
def get_mirror_pad_grad_grad_vmap_rule(prim, axis_size):
    """VmapRule for `MirrorPadGrad` operation."""
    input_max_dim = 4

    def vmap_rule(*params_bdim):
        is_all_none, result = vmap_general_preprocess(prim, params_bdim)
        if is_all_none:
            return result
        if len(params_bdim) < 2:
            _raise_value_error("The input params in `{}` must >= 2, but got {}.".format(prim.name, len(params_bdim)))
        input_x, input_x_dim = params_bdim[0]
        paddings, paddings_dim = params_bdim[1]

        out = None
        x = _bdim_at_front(input_x, input_x_dim, axis_size)
        if paddings_dim is not None:
            _raise_value_error(
                "The source axis of `paddings` in `{}` must be None, but got {}.".format(prim.name, paddings_dim))
        pad_dim = F.shape(paddings)[0]
        x_ndim = F.rank(x)

        if pad_dim == x_ndim and x_ndim <= input_max_dim:
            out = prim(x, paddings)
        elif x_ndim > input_max_dim:
            # reshape to 4 dims
            x_shape = F.shape(x)
            diff_dim = x_ndim - input_max_dim
            first_shape = 1
            for i in range(diff_dim + 1):
                first_shape *= x_shape[i]
            input_shape = (first_shape,) + x_shape[(-input_max_dim + 1):]
            x = F.reshape(x, input_shape)
            out = prim(x, paddings)
            out_shape = F.shape(out)
            real_out_shape = x_shape[:diff_dim + 1] + out_shape[1:]
            out = F.reshape(out, real_out_shape)
        else:
            _raise_value_error("The dim of `input_x` in `{}` must be bigger than {}, "
                               "but got {}.".format(prim.name, pad_dim, x_ndim))
        return out, 0

    return vmap_rule


@vmap_rules_getters.register('LayerNormGrad')
def get_layernormgrad_vmap_rule(prim, axis_size):
    """VmapRule for `LayerNormGrad` operation."""
    @constexpr
    def process_attr_axis(prim_attr_axis):
        if prim_attr_axis < 0:
            return prim_attr_axis
        return prim_attr_axis + 1

    @_primexpr
    def get_batch_params_reduce_axes(begin_params_axis, x_shape):
        if begin_params_axis < 0:
            x_rank = len(x_shape)
            begin_params_axis += x_rank
        batch_params_reduce_axes = tuple(range(1, begin_params_axis))
        return batch_params_reduce_axes

    @_primexpr
    def get_logical_shape(var_shape):
        return var_shape[1:]

    norm_axis = process_attr_axis(prim.begin_norm_axis)
    params_axis = process_attr_axis(prim.begin_params_axis)
    batch_prim = G.LayerNormGrad(norm_axis, params_axis)
    eps = 1e-12

    def vmap_rule(x_bdim, dy_bdim, var_bdim, mean_bdim, gamma_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, dy_bdim, var_bdim, mean_bdim, gamma_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        dy, dy_dim = dy_bdim
        var, var_dim = var_bdim
        mean, mean_dim = mean_bdim
        gamma, gamma_dim = gamma_bdim

        x = _bdim_at_front(x, x_dim, axis_size)
        dy = _bdim_at_front(dy, dy_dim, axis_size)
        var = _bdim_at_front(var, var_dim, axis_size)
        mean = _bdim_at_front(mean, mean_dim, axis_size)
        gamma = _bdim_at_front(gamma, gamma_dim, axis_size)

        dy_shape = F.shape(dy)
        batch_params_reduce_axes = get_batch_params_reduce_axes(params_axis, dy_shape)

        if not batch_params_reduce_axes:
            d_beta = dy
        else:
            d_beta = F.reduce_sum(dy, batch_params_reduce_axes)

        d_gamma_tmp = dy * (x - mean) / F.sqrt(var + eps)
        if not batch_params_reduce_axes:
            d_gamma = d_gamma_tmp
        else:
            d_gamma = F.reduce_sum(d_gamma_tmp, batch_params_reduce_axes)

        gamma_shape = F.shape(gamma)
        gamma = _handle_broadcasting(gamma, gamma_shape, dy_shape)
        dy = dy * gamma
        gamma_logical_shape = get_logical_shape(gamma_shape)
        ones_like_gamma = F.ones(gamma_logical_shape, F.dtype(gamma))
        dx, _, _ = batch_prim(x, dy, var, mean, ones_like_gamma)

        return (dx, 0), (d_gamma, 0), (d_beta, 0)
    return vmap_rule


@vmap_rules_getters.register(G.GridSampler2DGrad)
@vmap_rules_getters.register(G.GridSampler3DGrad)
def get_grid_sampler_grad_vmap_rule(prim, axis_size):
    """VmapRule for `GridSampler2DGrad` and `GridSampler3DGrad`."""
    prim_name = prim.name
    if prim_name == "GridSampler2DGrad":
        non_batch_dim_index = -3
    else:
        non_batch_dim_index = -4

    def vmap_rule(grad_bdim, input_x_bdim, grid_bdim):
        is_all_none, result = vmap_general_preprocess(prim, grad_bdim, input_x_bdim, grid_bdim)
        if is_all_none:
            return result

        grad, grad_dim = grad_bdim
        input_x, input_x_dim = input_x_bdim
        grid, grid_dim = grid_bdim

        grad = _bdim_at_front(grad, grad_dim, axis_size)
        grad_shape = F.shape(grad)
        grad = F.reshape(grad, (-1,) + grad_shape[non_batch_dim_index:])

        input_x = _bdim_at_front(input_x, input_x_dim, axis_size)
        input_x_shape = F.shape(input_x)
        input_x = F.reshape(input_x, (-1,) + input_x_shape[non_batch_dim_index:])

        grid = _bdim_at_front(grid, grid_dim, axis_size)
        grid_shape = F.shape(grid)
        grid = F.reshape(grid, (-1,) + grid_shape[non_batch_dim_index:])

        dx, dgrid = prim(grad, input_x, grid)
        dx_shape = F.shape(dx)
        dx_return_shape = input_x_shape[:non_batch_dim_index] + dx_shape[non_batch_dim_index:]
        dx = F.reshape(dx, dx_return_shape)
        dgrid_shape = F.shape(dgrid)
        dgrid_return_shape = input_x_shape[:non_batch_dim_index] + dgrid_shape[non_batch_dim_index:]
        dgrid = F.reshape(dgrid, dgrid_return_shape)
        return (dx, 0), (dgrid, 0)
    return vmap_rule


@vmap_rules_getters.register(G.UpsampleNearest3DGrad)
@vmap_rules_getters.register(G.UpsampleTrilinear3DGrad)
def get_upsample_grad_vmap_rule(prim, axis_size):
    """VmapRule for `UpsampleNearest3DGrad` and `UpsampleTrilinear3DGrad`."""
    cdhw_reverse_index = -4
    input_size = prim.input_size

    def vmap_rule(grad_bdim):
        is_all_none, result = vmap_general_preprocess(prim, grad_bdim)
        if is_all_none:
            return result

        grad, grad_dim = grad_bdim
        grad = _bdim_at_front(grad, grad_dim, axis_size)
        grad_shape = F.shape(grad)
        input_shape = (-1,) + grad_shape[cdhw_reverse_index:]
        grad = F.reshape(grad, input_shape)
        real_in_shape = F.shape(grad)

        # update batch dimension of input_size
        new_input_size = (real_in_shape[0],) + input_size[1:]
        _vmap_update_prim_attr(prim, 'input_size', new_input_size)

        out = prim(grad)
        out_shape = F.shape(out)
        real_out_shape = grad_shape[:cdhw_reverse_index] + out_shape[cdhw_reverse_index:]
        out = F.reshape(out, real_out_shape)
        return out, 0
    return vmap_rule


@vmap_rules_getters.register(G.LogSoftmaxGrad)
def get_log_softmax_vmap_rule(prim, axis_size):
    """VmapRule for 'LogSoftmaxGrad' operation."""
    if isinstance(prim, str):
        axis = -1
    else:
        axis = prim.axis

    def vmap_rule(x_bdim, grad_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result
        x, x_dim = x_bdim
        grad, _ = grad_bdim
        x_ndim = F.rank(x) - 1

        batch_axis = axis + x_ndim if axis < 0 else axis
        batch_axis = batch_axis if batch_axis < x_dim else batch_axis + 1

        dx = G.LogSoftmaxGrad(axis=batch_axis)(x, grad)
        return dx, x_dim

    return vmap_rule
