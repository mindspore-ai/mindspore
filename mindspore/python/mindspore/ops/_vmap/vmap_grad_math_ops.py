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

"""The vmap implement of grad operator corresponding to math_ops."""
from __future__ import absolute_import

from mindspore.ops import functional as F
from mindspore.ops.primitive import _primexpr
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.function import _VmapGeneralRule
from mindspore.ops._vmap.vmap_base import vmap_rules_getters, vmap_general_preprocess, _bdim_at_front, \
    _handle_broadcasting, get_unary_grad_vmap_rule, _get_broadcasting_with_front_axis_additional_axis


@vmap_rules_getters.register('MaximumGrad')
@vmap_rules_getters.register('MinimumGrad')
def get_broadcast_binary_op_grad_vmap_rule(prim, axis_size):
    """VmapRule for grad of binary operations with broadcasting"""
    broadcast_binary_op_grad_map = {
        "MinimumGrad": G.MinimumGrad,
        "MaximumGrad": G.MaximumGrad
    }

    if isinstance(prim, str):
        prim = broadcast_binary_op_grad_map.get(prim)()

    @_primexpr
    def get_longest_shape(x_shape, y_shape, g_shape):
        x_rank = len(x_shape)
        y_rank = len(y_shape)
        g_rank = len(g_shape)
        if x_rank > y_rank:
            if x_rank > g_rank:
                return x_shape
        else:
            if y_rank > g_rank:
                return y_shape
        return g_shape

    def vmap_rule(x_bdim, y_bdim, grad_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, y_bdim, grad_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        y, y_dim = y_bdim
        g, g_dim = grad_bdim

        x_shape = F.shape(x)
        y_shape = F.shape(y)
        g_shape = F.shape(g)

        if x_dim == y_dim and x_dim == g_dim and \
            x_shape == y_shape and x_shape == g_shape:
            dx, dy = prim(x, y, g)
            return (dx, x_dim), (dy, y_dim)

        x = _bdim_at_front(x, x_dim, axis_size)
        y = _bdim_at_front(y, y_dim, axis_size)
        g = _bdim_at_front(g, g_dim, axis_size)

        x_shape = F.shape(x)
        y_shape = F.shape(y)
        g_shape = F.shape(g)

        longest_shape = get_longest_shape(x_shape, y_shape, g_shape)
        x = _handle_broadcasting(x, x_shape, longest_shape)
        y = _handle_broadcasting(y, y_shape, longest_shape)
        g = _handle_broadcasting(g, g_shape, longest_shape)

        x_axis_for_reduce = _get_broadcasting_with_front_axis_additional_axis(x_shape, longest_shape)
        y_axis_for_reduce = _get_broadcasting_with_front_axis_additional_axis(y_shape, longest_shape)

        dx, dy = prim(x, y, g)
        if x_axis_for_reduce:
            dx = F.reduce_sum(dx, x_axis_for_reduce)

        if y_axis_for_reduce:
            dy = F.reduce_sum(dy, y_axis_for_reduce)

        return (dx, 0), (dy, 0)
    return vmap_rule


@vmap_rules_getters.register(G.MaximumGradGrad)
@vmap_rules_getters.register(G.MinimumGradGrad)
def get_broadcast_grad_grad_vmap_rule(prim, axis_size):
    """VmapRule for GradGrad operations with broadcasting."""

    def vmap_rule(x1_bdim, x2_bdim, dx1_bdim, dx2_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x1_bdim, x2_bdim, dx1_bdim, dx2_bdim)
        if is_all_none:
            return result

        x1, x1_dim = x1_bdim
        x2, x2_dim = x2_bdim
        dx1, dx1_dim = dx1_bdim
        dx2, dx2_dim = dx2_bdim
        x1_shape = F.shape(x1)
        x2_shape = F.shape(x2)
        dx1_shape = F.shape(dx1)
        dx2_shape = F.shape(dx2)

        if x1_dim == x2_dim and dx1_dim == dx2_dim and x1_dim == dx1_dim \
                and x1_shape == x2_shape and dx1_shape == dx2_shape:
            sopd_x1, sopd_x2, sopd_grad = prim(x1, x2, dx1, dx2)
            return (sopd_x1, x1_dim), (sopd_x2, x1_dim), (sopd_grad, x1_dim)

        if F.rank(x1):
            x1 = _bdim_at_front(x1, x1_dim, 1)
        if F.rank(x2):
            x2 = _bdim_at_front(x2, x2_dim, 1)
        if F.rank(dx1):
            dx1 = _bdim_at_front(dx1, dx2_dim, 1)
        if F.rank(dx2):
            dx2 = _bdim_at_front(dx2, dx2_dim, 1)
        x1_shape = F.shape(x1)
        x2_shape = F.shape(x2)
        dx1_shape = F.shape(dx1)
        dx2_shape = F.shape(dx2)
        x1 = _handle_broadcasting(x1, x1_shape, x2_shape)
        x2 = _handle_broadcasting(x2, x2_shape, x1_shape)
        dx1 = _handle_broadcasting(dx1, dx1_shape, dx2_shape)
        dx2 = _handle_broadcasting(dx2, dx2_shape, dx1_shape)
        sopd_x1, sopd_x2, sopd_grad = prim(x1, x2, dx1, dx2)
        return (sopd_x1, 0), (sopd_x2, 0), (sopd_grad, 0)

    return vmap_rule


@vmap_rules_getters.register(G.MedianGrad)
def get_median_grad_vmap_rule(prim, axis_size):
    """VmapRule for MedianGrad."""
    prim_vmap = _VmapGeneralRule(prim, axis_size)
    global_median = prim.global_median
    axis = prim.axis
    keep_dims = prim.keep_dims

    @_primexpr
    def trans_grad_axis(axis, rank, dim, keep_dims):
        if axis < 0:
            axis += rank - 1
        axis_new = axis + 1 if dim <= axis else axis
        if keep_dims:
            dim_new = dim
        else:
            dim_new = dim - 1 if dim > axis_new else dim
        return dim_new

    def vmap_rule(dy_bdim, x_bdim, y_bdim, indices_bdim):
        if global_median is True:
            return prim_vmap(dy_bdim, x_bdim, y_bdim, indices_bdim)
        dy, dy_dim = dy_bdim
        x, x_dim = x_bdim
        y, y_dim = y_bdim
        indices, indices_dim = indices_bdim
        rank = len(x.shape)
        dim_new = trans_grad_axis(axis, rank, x_dim, keep_dims)

        dy = _bdim_at_front(dy, dy_dim, axis_size)
        x = _bdim_at_front(x, x_dim, axis_size)
        y = _bdim_at_front(y, y_dim, axis_size)
        indices = _bdim_at_front(indices, indices_dim, axis_size)
        x_grad = G.MedianGrad(global_median, axis, keep_dims)(dy, x, y, indices)
        return x_grad, dim_new
    return vmap_rule


# UnaryGrad vmap
get_unary_grad_vmap_rule = vmap_rules_getters.register(G.InvGrad)(get_unary_grad_vmap_rule)
get_unary_grad_vmap_rule = vmap_rules_getters.register(G.LogitGrad)(get_unary_grad_vmap_rule)
get_unary_grad_vmap_rule = vmap_rules_getters.register('AbsGrad')(get_unary_grad_vmap_rule)
get_unary_grad_vmap_rule = vmap_rules_getters.register('ReciprocalGrad')(get_unary_grad_vmap_rule)
get_unary_grad_vmap_rule = vmap_rules_getters.register('SqrtGrad')(get_unary_grad_vmap_rule)
