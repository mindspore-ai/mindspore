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

"""nn_ops vmap impl."""

from functools import reduce
import mindspore.numpy as mnp
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops import functional as F
from mindspore.ops import constexpr
from ..primitive import Primitive
from .._vmap.vmap_base import vmap_rules_getters, vmap_general_preprocess, _raise_value_error, _bdim_at_front,\
     _vmap_clone_prim, _bdim_at_any


@vmap_rules_getters.register(G.NLLLossGrad)
def get_nll_loss_grad_vmap_rule(prim, axis_size):
    r"""
    VmapRule for NLLLossGrad operations.

    Limited by current kernel capability:
    1. Only support one dim batch for x, loss_grad and target.
    2. And weight only support shape as (C,), while total_weight should be a scalar.
    """

    @constexpr
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
        if lg_dim is not None and (base_x_len != 2 or reduction_type != "none"):
            _raise_value_error("The source axis of loss_grad in `NLLLossGrad` can be not None "
                               "just when x is 2d and reduction type is none, "
                               "but x is {}d and reduction type is {}.".format(base_x_len, reduction_type))

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
        return (output, out_dim)

    return vmap_rule


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
        return (out, 0)

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
        return (out, 0)

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
        return (out, 0)

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
        return (out, 0)

    return vmap_rule


@vmap_rules_getters.register(G.AdaptiveMaxPool2DGrad)
def get_adaptive_avgpool2d_vmap_rule(prim, axis_size):
    """VmapRule for `AdaptiveMaxPool2DGrad` operation."""
    chw_reverse_index = -3
    hw_reverse_index = -2

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
        return (out, 0)

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
        return (out, 0)

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
        return (out, 0)

    return vmap_rule
