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
from .._vmap.vmap_base import vmap_rules_getters, vmap_general_preprocess, _raise_value_error, _bdim_at_front


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
        new_batch_size = reduce(lambda x, y: x * y, shape if keep_dim == 0 else shape[:-keep_dim])
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
            loss_grad = F.reshape(loss_grad, _get_reshape_shape(loss_grad_shape))

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
