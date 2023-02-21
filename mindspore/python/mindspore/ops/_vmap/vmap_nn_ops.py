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
from __future__ import absolute_import

import mindspore
from mindspore.common import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.operations import nn_ops as NN
from mindspore.ops import functional as F
from mindspore.ops import constexpr
from mindspore.ops.primitive import _primexpr
from mindspore.ops._vmap.vmap_base import vmap_rules_getters, vmap_general_preprocess, get_unop_vmap_rule, \
    _bdim_at_any, _bdim_at_front, _bdim_at_back, _handle_broadcasting, get_unary_grad_vmap_rule, _raise_value_error, \
    _vmap_clone_prim, _get_reduce_batch_axis
from mindspore.ops.primitive import Primitive


@vmap_rules_getters.register(P.ApplyAdaMax)
def get_apply_ada_max_rule(prim, axis_size):
    """VmapRule for `ApplyAdaMax` operation."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1
    prim_name = prim.name
    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr("batch_rank", batch_rank)

    def vmap_rule(var_bdim, m_bdim, v_bdim, beta1_power_bdim, lr_bdim, beta1_bdim, beta2_bdim,
                  epsilon_bdim, grad_bdim, u_monad):
        var, var_dim = var_bdim
        m, m_dim = m_bdim
        v, v_dim = v_bdim
        lr, lr_dim = lr_bdim
        beta1_power, beta1_power_dim = beta1_power_bdim
        beta1, beta1_dim = beta1_bdim
        beta2, beta2_dim = beta2_bdim
        epsilon, epsilon_dim = epsilon_bdim
        grad, grad_dim = grad_bdim

        if var_dim is None:
            if any(dim is not None for dim in [m_bdim, v_bdim, beta1_power_bdim, lr_bdim, beta1_bdim, beta2_bdim,
                                               epsilon_bdim, grad_bdim]):
                raise ValueError("The source axis of `var` is None, but the source "
                                 "axis of `accum/lr/beta1/beta1_power/beta2/epsilon/grad` is not None. "
                                 "The execution order of operator `{}` cannot be guaranteed.".format(prim_name))
            var, m, v = prim(var, m, v, beta1_power, lr, beta1, beta2, epsilon, grad, u_monad)
            return (var, None), (m, None), (v, None)
        if var_dim != 0 or m_dim != var_dim or var_dim != v_dim:
            raise ValueError("For `{}`, the source axis of `var` must be equal to `accum`, and not equal to 0, "
                             "but got the source axis of `var`: {}, `accum`: {}.".format(prim_name, var_dim, m_dim))

        lr = _bdim_at_front(lr, lr_dim, axis_size)
        beta1_power = _bdim_at_front(beta1_power, beta1_power_dim, axis_size)
        beta1 = _bdim_at_front(beta1, beta1_dim, axis_size)
        beta2 = _bdim_at_front(beta2, beta2_dim, axis_size)
        epsilon = _bdim_at_front(epsilon, epsilon_dim, axis_size)
        grad = _bdim_at_front(grad, grad_dim, axis_size)
        var, m, v = batch_prim(var, m, v, beta1_power, lr, beta1, beta2, epsilon, grad, u_monad)
        return (var, 0), (m, 0), (v, 0)

    return vmap_rule


@vmap_rules_getters.register(P.ApplyAdadelta)
def get_apply_adadelta_rule(prim, axis_size):
    """VmapRule for `ApplyAdadelta` operation."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    prim_name = prim.name
    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr('batch_rank', batch_rank)

    def vmap_rule(var_bdim, accum_bdim, accum_update_bdim, lr_bdim, rho_bdim, epsilon_bdim, grad_bdim, u_monad):
        var, var_dim = var_bdim
        accum, accum_dim = accum_bdim
        accum_update, accum_update_dim = accum_update_bdim
        lr, lr_dim = lr_bdim
        rho, rho_dim = rho_bdim
        epsilon, epsilon_dim = epsilon_bdim
        grad, grad_dim = grad_bdim

        if var_dim is None:
            if any(dim is not None for dim in [accum, accum_dim, lr_dim, rho_dim, epsilon_dim, grad_dim]):
                raise ValueError("The source axis of `var` is None, but the source "
                                 "axis of `accum/accum_dim/lr/rho/epsilon/grad` is not None. The execution order of "
                                 "operator `{}` cannot be guaranteed.".format(prim_name))
            var, accum, accum_update = prim(var, accum, accum_update, lr, rho, epsilon, grad, u_monad)
            return (var, None), (accum, None), (accum_update, None)
        if var_dim != 0 or accum_dim != var_dim or accum_update_dim != var_dim:
            raise ValueError(
                "For `{}`, the source axis of `var` must be equal to `accum` and `accum_update`, and not equal to 0, "
                "but got the source axis of `var`: {}, `accum`: {}, `accum_update`: {}.".format(
                    prim_name, var_dim, accum_dim, accum_update_dim))

        lr = _bdim_at_front(lr, lr_dim, axis_size)
        rho = _bdim_at_front(rho, rho_dim, axis_size)
        epsilon = _bdim_at_front(epsilon, epsilon_dim, axis_size)
        grad = _bdim_at_front(grad, grad_dim, axis_size)

        var, accum, accum_update = batch_prim(var, accum, accum_update, lr, rho, epsilon, grad, u_monad)
        return (var, 0), (accum, 0), (accum_update, 0)

    return vmap_rule


@vmap_rules_getters.register(P.ApplyFtrl)
def get_apply_ftrl_rule(prim, axis_size):
    """VmapRule for `ApplyFtrl` operation"""
    if hasattr(prim, "batch_rank"):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1
    prim_name = prim.name
    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr('batch_rank', batch_rank)

    def vmap_rule(var_bdim, accum_bdim, linear_bdim, grad_bdim, lr_bdim, l1_bdim, l2_bdim, lr_power_bdim, u_monad):
        var, var_dim = var_bdim
        accum, accum_dim = accum_bdim
        linear, linear_dim = linear_bdim
        grad, grad_dim = grad_bdim
        lr, lr_dim = lr_bdim
        l1, l1_dim = l1_bdim
        l2, l2_dim = l2_bdim
        lr_power, lr_power_dim = lr_power_bdim

        if var_dim is None:
            if any(dim is not None for dim in [accum_dim, linear_dim, grad_dim, lr_dim, l1_dim, l2_dim, lr_power_dim]):
                raise ValueError("The source axis of `var` is None, "
                                 "but the source axis of `accum/linear/grad/lr/l1/l1/lr_power` is not None. "
                                 "The execution order of operator `{}` cannot be guaranteed.".format(prim_name))
            var = prim(var, accum, linear, grad, lr, l1, l2, lr_power, u_monad)
            return var, None
        if var_dim != 0 or accum_dim != var_dim or linear_dim != var_dim:
            raise ValueError("For `{}`, the source axis of `var/accum/linear` must be 0, "
                             "but get `var`: {}, `accum`: {}, `linear`: {}.".format(prim_name, var_dim, accum_dim,
                                                                                    linear_dim))
        grad = _bdim_at_front(grad, grad_dim, axis_size)
        lr = _bdim_at_front(lr, lr_dim, axis_size)
        l1 = _bdim_at_front(l1, l1_dim, axis_size)
        l2 = _bdim_at_front(l2, l2_dim, axis_size)
        lr_power = _bdim_at_front(lr_power, lr_power_dim, axis_size)

        var = batch_prim(var, accum, linear, grad, lr, l1, l2, lr_power, u_monad)
        return var, 0

    return vmap_rule


@vmap_rules_getters.register(P.ApplyProximalAdagrad)
def get_apply_proximal_adagrad_rule(prim, axis_size):
    """VmapRule for `ApplyProximalAdagrad` operation."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    prim_name = prim.name
    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr('batch_rank', batch_rank)

    def vmap_rule(var_bdim, accum_bdim, lr_bdim, l1_bdim, l2_bdim, grad_bdim, u_monad):
        var, var_dim = var_bdim
        accum, accum_dim = accum_bdim
        lr, lr_dim = lr_bdim
        l1, l1_dim = l1_bdim
        l2, l2_dim = l2_bdim
        grad, grad_dim = grad_bdim

        if var_dim is None:
            if any(dim is not None for dim in [accum_dim, lr_dim, l1_dim, l2_dim, grad_dim]):
                raise ValueError("The source axis of `var` is None, but the source "
                                 "axis of `accum/lr/l1/l2/grad` is not None. The execution order of "
                                 "operator `{}` cannot be guaranteed.".format(prim_name))
            var, accum = prim(var, accum, lr, l1, l2, grad, u_monad)
            return (var, None), (accum, None)

        if var_dim != 0 or accum_dim != var_dim:
            raise ValueError("For `{}`, the source axis of `var` must be equal to `accum`, and not equal to 0, "
                             "but got the source axis of `var`: {}, `accum`: {}.".format(prim_name, var_dim, accum_dim))

        lr = _bdim_at_front(lr, lr_dim, axis_size)
        l1 = _bdim_at_front(l1, l1_dim, axis_size)
        l2 = _bdim_at_front(l2, l2_dim, axis_size)
        grad = _bdim_at_front(grad, grad_dim, axis_size)

        var, accum = batch_prim(var, accum, lr, l1, l2, grad, u_monad)
        return (var, 0), (accum, 0)

    return vmap_rule


@vmap_rules_getters.register(P.ApplyGradientDescent)
def get_apply_gradient_descent_rule(prim, axis_size):
    """VmapRule for `ApplyGradientDescent` operation."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    prim_name = prim.name
    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr('batch_rank', batch_rank)

    def vmap_rule(var_bdim, alpha_bdim, delta_bdim, u_monad):
        var, var_dim = var_bdim
        alpha, alpha_dim = alpha_bdim
        delta, delta_dim = delta_bdim

        if var_dim is None:
            if any(dim is not None for dim in [alpha_dim, delta_dim]):
                raise ValueError("The source axis of `var` is None, but the source "
                                 "axis of `alpha/delta` is not None. The execution order of "
                                 "operator `{}` cannot be guaranteed.".format(prim_name))
            var = prim(var, alpha, delta, u_monad)
            return var, None

        if var_dim != 0:
            raise ValueError("For `{}`, the source axis of `var` must not equal to 0, "
                             "but got the source axis of `var`: {}.".format(prim_name, var_dim))

        alpha = _bdim_at_front(alpha, alpha_dim, axis_size)
        delta = _bdim_at_front(delta, delta_dim, axis_size)

        var = batch_prim(var, alpha, delta, u_monad)
        return var, 0

    return vmap_rule


@vmap_rules_getters.register(P.ApplyProximalGradientDescent)
def get_apply_proximal_gradient_descent_rule(prim, axis_size):
    """VmapRule for `ApplyProximalGradientDescent` operation."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    prim_name = prim.name
    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr('batch_rank', batch_rank)

    def vmap_rule(var_bdim, alpha_bdim, l1_bdim, l2_bdim, delta_bdim, u_monad):
        var, var_dim = var_bdim
        alpha, alpha_dim = alpha_bdim
        l1, l1_dim = l1_bdim
        l2, l2_dim = l2_bdim
        delta, delta_dim = delta_bdim

        if var_dim is None:
            if any(dim is not None for dim in [alpha_dim, l1_dim, l2_dim, delta_dim]):
                raise ValueError("The source axis of `var` is None, but the source "
                                 "axis of `alpha/l1/l2/delta` is not None. The execution order of "
                                 "operator `{}` cannot be guaranteed.".format(prim_name))
            var = prim(var, alpha, l1, l2, delta, u_monad)
            return var, None

        if var_dim != 0:
            raise ValueError("For `{}`, the source axis of `var` must not equal to 0, "
                             "but got the source axis of `var`: {}.".format(prim_name, var_dim))

        alpha = _bdim_at_front(alpha, alpha_dim, axis_size)
        l1 = _bdim_at_front(l1, l1_dim, axis_size)
        l2 = _bdim_at_front(l2, l2_dim, axis_size)
        delta = _bdim_at_front(delta, delta_dim, axis_size)

        var = batch_prim(var, alpha, l1, l2, delta, u_monad)
        return var, 0

    return vmap_rule


@vmap_rules_getters.register(NN.BCEWithLogitsLoss)
def get_bce_with_logits_loss_vamp_rule(prim, axis_size):
    """VmapRule for 'BCEWithLogitsLoss' ."""

    if isinstance(prim, str):
        prim = Primitive(prim)
        prim_reduction = 'none'
    else:
        prim_reduction = prim.reduction
    prim_name = prim.name
    bce_logits_with_loss_op = NN.BCEWithLogitsLoss('none')
    if prim_reduction == 'mean':
        reduce_op = P.ReduceMean()
    elif prim_reduction == "sum":
        reduce_op = P.ReduceSum()

    def vmap_rule(logits_bdim, label_bdim, weight_bdim, pos_weight_bdim):
        is_all_none, result = vmap_general_preprocess(prim, logits_bdim, label_bdim,
                                                      weight_bdim, pos_weight_bdim)
        if is_all_none:
            return result
        logits, logits_dim = logits_bdim
        label, label_dim = label_bdim
        weight, weight_dim = weight_bdim
        pos_weight, pos_weight_dim = pos_weight_bdim
        logits_rank = F.rank(logits)
        label_rank = F.rank(label)
        weight_rank = F.rank(weight)
        pos_weight_rank = F.rank(pos_weight)
        max_rank = max(logits_rank, label_rank)
        max_rank = max(max_rank, weight_rank)
        max_rank = max(max_rank, pos_weight_rank)
        reduce_indexes = None
        # If rank is larger than 1, we need to reduce result when reduction != 'none'
        if max_rank > 1:
            reduce_indexes = tuple(range(1, max_rank))
        if logits_dim == label_dim and F.shape(logits) == F.shape(label) \
                and logits_dim == weight_dim and F.shape(logits) == F.shape(weight) \
                and logits_dim == pos_weight_dim and F.shape(logits) == F.shape(pos_weight):
            if prim_reduction == 'none':
                output = prim(logits, label, weight, pos_weight)
            elif prim_reduction in ('mean', 'sum'):
                out = bce_logits_with_loss_op(logits, label, weight, pos_weight)
                output = reduce_op(out, reduce_indexes)
            else:
                raise RuntimeError("For {} vmap, the attribute of reduction must in "
                                   "('none', 'mean', 'sum'), but got {}."
                                   .format(prim_name, prim_reduction))
            return output, logits_dim

        logits = _bdim_at_front(logits, logits_dim, axis_size)
        label = _bdim_at_front(label, label_dim, axis_size)
        weight = _bdim_at_front(weight, weight_dim, axis_size)
        pos_weight = _bdim_at_front(pos_weight, pos_weight_dim, axis_size)
        logits_shape = F.shape(logits)
        weight_shape = F.shape(weight)
        pos_weight_shape = F.shape(pos_weight)
        weight = _handle_broadcasting(weight, weight_shape, logits_shape)
        pos_weight = _handle_broadcasting(pos_weight, pos_weight_shape, logits_shape)
        if prim_reduction == 'none':
            output = prim(logits, label, weight, pos_weight)
        elif prim_reduction in ('mean', 'sum'):
            out = bce_logits_with_loss_op(logits, label, weight, pos_weight)
            output = reduce_op(out, reduce_indexes)
        else:
            raise RuntimeError("For {} vmap, the attribute of reduction must in "
                               "('none', 'mean', 'sum'), but got {}."
                               .format(prim_name, prim_reduction))
        return output, 0

    return vmap_rule


@vmap_rules_getters.register(P.BiasAdd)
def get_bias_add_vmap_rule(prim, axis_size):
    """VmapRule for `BiasAdd` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)
        data_format = "NCHW"
    else:
        data_format = prim.data_format
    add_op = P.Add()

    @constexpr
    def get_channal_pos_in_x(d_format):
        return d_format.find('C') + 1

    @_primexpr
    def get_bias_dst_shape(x_shape, n_dims, d_format, has_b_dim: bool):
        pos = get_channal_pos_in_x(d_format)

        bias_shape = []
        for i in range(n_dims):
            if i != pos:
                bias_shape.append(1)
            else:
                bias_shape.append(x_shape[i])

        if has_b_dim:
            bias_shape[0] = axis_size

        return tuple(bias_shape)

    def vmap_rule(x_bdim, bias_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, bias_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        b, b_dim = bias_bdim

        x = _bdim_at_front(x, x_dim, axis_size)
        has_b_dim = False
        if b_dim is not None:
            b = _bdim_at_front(b, b_dim, axis_size)
            has_b_dim = True

        x_shape = x.shape
        n_dims = len(x_shape)
        b_shape = get_bias_dst_shape(x_shape, n_dims, data_format, has_b_dim)

        b = b.reshape(b_shape)
        result = add_op(x, b)

        return (result, 0)

    return vmap_rule


@vmap_rules_getters.register(G.BiasAddGrad)
def get_bias_add_grad_vmap_rule(prim, axis_size):
    """VmapRule for `BiasAddGrad` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)
        data_format = "NCHW"
    else:
        data_format = prim.data_format

    @constexpr
    def get_channal_pos(d_format):
        return d_format.find('C') + 1

    @_primexpr
    def get_axis_for_reduce(x_shape_rank, data_format):
        channal_pos = get_channal_pos(data_format)
        axis_list = ()
        for i in range(1, x_shape_rank):
            if channal_pos == i:
                continue
            axis_list += (i,)

        return axis_list

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        x_shape_rank = len(x.shape)

        axis_for_reduce = get_axis_for_reduce(x_shape_rank, data_format)

        result = x.sum(axis=axis_for_reduce)
        return (result, 0)

    return vmap_rule


@vmap_rules_getters.register(P.Dropout)
@vmap_rules_getters.register(P.Dropout2D)
@vmap_rules_getters.register(P.Dropout3D)
def get_dropout_nd_vmap_rule(prim, axis_size):
    """VmapRule for 'DropoutND' operation."""
    prim_name = prim.name
    dropout_nd_dim = 4
    if prim_name == "Dropout3D":
        dropout_nd_dim = 5

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        x_ndim = F.rank(x)
        if x_ndim > dropout_nd_dim:
            x_ori_shape = F.shape(x)
            x = F.reshape(x, (-1,) + x_ori_shape[2:x_ndim])
            output, mask = prim(x)
            output = F.reshape(output, x_ori_shape)
            mask = F.reshape(mask, x_ori_shape)
        else:
            output, mask = prim(x)

        return (output, 0), (mask, 0)

    return vmap_rule


@vmap_rules_getters.register(P.InTopK)
def get_in_top_k_vmap_rule(prim, axis_size):
    """VmapRule for `InTopK`."""

    def vmap_rule(x1_bdim, x2_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x1_bdim, x2_bdim)
        if is_all_none:
            return result

        x1, x1_dim = x1_bdim
        x2, x2_dim = x2_bdim
        x1 = _bdim_at_front(x1, x1_dim, axis_size)
        x2 = _bdim_at_front(x2, x2_dim, axis_size)
        x1_shape = F.shape(x1)
        x2_shape = F.shape(x2)
        x1 = F.reshape(x1, (-1, x1_shape[-1]))
        x2 = F.reshape(x2, (-1,))
        output = prim(x1, x2)
        output = F.reshape(output, x2_shape)
        return output, 0

    return vmap_rule


@vmap_rules_getters.register(G.FastGeLUGrad)
@vmap_rules_getters.register(G.HShrinkGrad)
@vmap_rules_getters.register(G.HSwishGrad)
@vmap_rules_getters.register(G.SoftShrinkGrad)
def get_common_activation_grad_vmap_rule(prim, axis_size):
    """VmapRule for common activation grad operation."""
    if isinstance(prim, str):
        prim_name = prim
        prim = Primitive(prim)
    else:
        prim_name = prim.name

    def vmap_rule(x_bdim, dy_bdim):
        x, x_dim = x_bdim
        dy, dy_dim = dy_bdim
        x_shape = F.shape(x)
        dy_shape = F.shape(dy)
        if x_dim == dy_dim and x_shape == dy_shape:
            out = prim(x, dy)
            return out, x_dim

        if F.rank(x):
            x = _bdim_at_front(x, x_dim, 1)
        if F.rank(dy):
            dy = _bdim_at_front(dy, dy_dim, 1)
        x_shape = F.shape(x)
        dy_shape = F.shape(dy)
        if x_shape != dy_shape:
            raise RuntimeError("For {} vmap, input x shape is supposed to be the same as input dy shape "
                               "after batch transforming, but got x_shape {}, dy_shape {}"
                               .format(prim_name, x_shape, dy_shape))
        out = prim(x, dy)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.Pad)
def get_pad_vmap_rule(prim, axis_size):
    """VmapRule for `Pad`"""
    paddings = prim.paddings

    @constexpr
    def _get_paddings(cur_paddings, x_dim):
        """get paddings."""
        new_paddings = list(cur_paddings)
        new_paddings.insert(x_dim, (0, 0))
        return tuple(new_paddings)

    def vmap_rule(x_bdim):
        x, x_dim = x_bdim
        if x_dim is None:
            # case1: batch not exists
            out = prim(x)
        else:
            # case2: batch exists
            new_paddings = _get_paddings(paddings, x_dim)
            op = P.Pad(new_paddings)
            out = op(x)
        return out, x_dim

    return vmap_rule


@vmap_rules_getters.register(NN.Pdist)
def get_pdist_vmap_rule(prim, axis_size):
    """VmapRule for `Pdist`"""
    if isinstance(prim, str):
        prim = Primitive(prim)
        prim.add_prim_attr('p', 2.0)

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result
        x, x_dim = x_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        out = prim(x)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(NN.DeformableOffsets)
def get_matmul_vmap_rule(prim, axis_size):
    """VmapRule for `DeformableOffsets` operation."""
    nchw_size = 4
    chw_size = 3
    chw_reverse_index = -chw_size

    def vmap_rule(x_bdim, offsets_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, offsets_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        offsets, offsets_dim = offsets_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        x_ndim = F.rank(x)
        x_origin_shape = F.shape(x)

        offsets = _bdim_at_front(offsets, offsets_dim, axis_size)
        offsets_ndim = F.rank(offsets)
        offsets_origin_shape = F.shape(offsets)

        batch_origin_shape = x_origin_shape
        if x_ndim > nchw_size:
            x = F.reshape(x, (-1,) + x_origin_shape[chw_reverse_index:])
        if offsets_ndim > nchw_size:
            offsets = F.reshape(offsets, (-1,) + offsets_origin_shape[chw_reverse_index:])
            batch_origin_shape = offsets_origin_shape

        out = prim(x, offsets)
        out_shape = F.shape(out)
        out = F.reshape(out, batch_origin_shape[:(nchw_size + 1 - chw_size)] + out_shape[chw_reverse_index:])
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.Softmax)
def get_softmax_vmap_rule(prim, axis_size):
    """VmapRule for `Softmax`"""
    axis = prim.axis[0]
    if isinstance(axis, tuple):
        axis = axis[0]

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result
        x, x_dim = x_bdim
        x_ndim = F.rank(x)
        batch_axis = _get_reduce_batch_axis(axis, x_dim, x_ndim)
        out = P.Softmax(batch_axis)(x)
        return out, x_dim

    return vmap_rule


@vmap_rules_getters.register(P.AdaptiveAvgPool2D)
def get_adaptive_avgpool2d_vmap_rule(prim, axis_size):
    """VmapRule for `AdaptiveAvgPool2D` operation."""
    chw_reverse_index = -3
    hw_reverse_index = -2

    def vmap_rule(input_bdim):
        is_all_none, result = vmap_general_preprocess(prim, input_bdim)
        if is_all_none:
            return result

        input_x, x_dim = input_bdim
        input_x = _bdim_at_front(input_x, x_dim, axis_size)
        x_shape = F.shape(input_x)
        input_shape = (-1,) + x_shape[chw_reverse_index:]
        input_x = F.reshape(input_x, input_shape)
        out = prim(input_x)
        out_shape = F.shape(out)
        real_out_shape = x_shape[:hw_reverse_index] + out_shape[hw_reverse_index:]
        out = F.reshape(out, real_out_shape)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(NN.AdaptiveAvgPool3D)
def get_adaptive_avgpool3d_vmap_rule(prim, axis_size):
    """VmapRule for `AdaptiveAvgPool3D` operation."""
    dhw_reverse_index = -3
    max_dims = 5

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        if F.rank(x) == max_dims:
            out = prim(x)
            return out, 0

        x_shape = F.shape(x)
        shape = (-1,) + x_shape[dhw_reverse_index:]
        x = F.reshape(x, shape)
        out = prim(x)
        out_shape = F.shape(out)
        real_out_shape = x_shape[:dhw_reverse_index] + out_shape[dhw_reverse_index:]
        out = F.reshape(out, real_out_shape)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.AvgPool)
def get_avgpool_vmap_rule(prim, axis_size):
    """VmapRule for `AvgPool`."""
    chw_reverse_index = -3

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        x_shape = F.shape(x)
        input_shape = (-1,) + x_shape[chw_reverse_index:]
        x = F.reshape(x, input_shape)
        out = prim(x)
        out_shape = F.shape(out)
        real_out_shape = x_shape[:chw_reverse_index] + out_shape[chw_reverse_index:]
        out = F.reshape(out, real_out_shape)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(NN.AdaptiveMaxPool3D)
def get_adaptive_max_pool3d_vmap_rule(prim, axis_size):
    """VmapRule for `AdaptiveMaxPool3D`."""
    dhw_reverse_index = -3
    max_dims = 5

    @constexpr
    def convert_shape_to_tensor(shape):
        return Tensor(shape, dtype=mindspore.int32)

    def vmap_rule(x_bdim, out_size_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, out_size_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        out_size, out_size_dim = out_size_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        if out_size_dim is not None:
            _raise_value_error("The source axis of `output_size` in `AdaptiveMaxPool3D` must be None, "
                               "but got {}.".format(out_size_dim))
        if F.rank(x) == max_dims:
            out, indices = prim(x, out_size)
            return (out, 0), (indices, 0)

        x_shape = F.shape(x)
        shape = (-1,) + x_shape[dhw_reverse_index:]
        x = F.reshape(x, shape)
        out, indices = prim(x, out_size)
        # AdaptiveMaxPool3D is a dynamic op, the 'shape' of reshape should be a tensor
        front_shape = convert_shape_to_tensor(x_shape[:dhw_reverse_index])
        output_shape = F.concat((front_shape, out_size))
        out = F.reshape(out, output_shape)
        indices = F.reshape(indices, output_shape)
        return (out, 0), (indices, 0)

    return vmap_rule


@vmap_rules_getters.register(NN.InstanceNorm)
def get_instance_norm_rule(prim, axis_size):
    """VmapRule for `InstanceNorm` operation."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    prim_name = prim.name
    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr('batch_rank', batch_rank)

    def vmap_rule(input_x_bdim, gamma_bdim, beta_bdim, mean_bdim, variance_bdim, u_monad):
        input_x, input_x_dim = input_x_bdim
        gamma, gamma_dim = gamma_bdim
        beta, beta_dim = beta_bdim
        mean, mean_dim = mean_bdim
        variance, variance_dim = variance_bdim
        if gamma_dim is None:
            if any(dim is not None for dim in [input_x_dim, beta_dim, mean_dim, variance_dim]):
                raise ValueError("The source axis of `gamma` is None, but the source "
                                 "axis of `input_x/beta/mean/variance` is not None. The execution order of "
                                 "operator `{}` cannot be guaranteed.".format(prim_name))
            output_x, updated_moving_mean, updated_moving_variance = prim(input_x, gamma, beta, mean, variance, u_monad)
            return (output_x, None), (updated_moving_mean, None), (updated_moving_variance, None)

        if gamma_dim != 0 or beta_dim != gamma_dim or mean_dim != gamma_dim or variance_dim != gamma_dim:
            # pylint: disable=too-many-format-args
            raise ValueError(
                "For `{}`, the source axis of `var` must be equal to `accum` and `accum_update`, and not equal to 0, "
                "but got the source axis of `var`: {}, `accum`: {}, `accum_update`: {}.".format(
                    prim_name, gamma_dim, beta_dim, mean_dim, variance_dim))
        input_x = _bdim_at_front(input_x, input_x_dim, axis_size)
        output_x, updated_moving_mean, updated_moving_variance = batch_prim(input_x, gamma, beta, mean, variance,
                                                                            u_monad)
        return (output_x, 0), (updated_moving_mean, 0), (updated_moving_variance, 0)

    return vmap_rule


@vmap_rules_getters.register(P.KLDivLoss)
def get_kl_div_loss_vmap_rule(prim, axis_size):
    """VmapRule for `KLDivLoss` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    prim_reduction = prim.reduction
    if prim_reduction == "mean":
        kl_div_loss_op = P.KLDivLoss("none")
        reduce_op = P.ReduceMean()
    elif prim_reduction == "sum":
        kl_div_loss_op = P.KLDivLoss("none")
        reduce_op = P.ReduceSum()
    elif prim_reduction == "batchmean":
        kl_div_loss_op = P.KLDivLoss("none")
        reduce_op = P.ReduceSum()
        factor_op = P.Div()

    def vmap_rule(x_bdim, target_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, target_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        target, target_dim = target_bdim
        x_ndim = F.rank(x)
        target_ndim = F.rank(target)
        max_rank = max(x_ndim, target_ndim)
        x = _bdim_at_front(x, x_dim, axis_size)
        target = _bdim_at_front(target, target_dim, axis_size)
        reduce_indexes = None
        factor = 1
        # if rank is larger than 1, we need to reduce result when reduction != 'none'
        if max_rank > 1:
            reduce_indexes = tuple(range(1, max_rank))
            factor = F.shape(x)[1]

        # elementwise style when reduction='none', otherwise reduce style
        if prim_reduction == "none":
            out = prim(x, target)
        elif prim_reduction in ("mean", "sum"):
            out = kl_div_loss_op(x, target)
            if reduce_indexes is not None:
                out = reduce_op(out, reduce_indexes)
        elif prim_reduction == "batchmean":
            out = kl_div_loss_op(x, target)
            if reduce_indexes is not None:
                out = reduce_op(out, reduce_indexes)
                out = factor_op(out, factor)
        else:
            raise RuntimeError("For KLDivLoss vmap, reduction should be one of "
                               "['none', 'mean', 'batchmean', 'sum'], but got '{}'".format(prim_reduction))
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(G.KLDivLossGrad)
def get_kl_div_loss_grad_vmap_rule(prim, axis_size):
    """VmapRule for `KLDivLossGrad`."""
    if isinstance(prim, str):
        prim = Primitive(prim)
        reduction = "mean"
    else:
        reduction = prim.reduction

    kldivloss_grad = G.KLDivLossGrad(reduction=reduction)

    def vmap_rule(dy_bdim, x_bdim, target_bdim):
        is_all_none, result = vmap_general_preprocess(prim, dy_bdim, x_bdim, target_bdim)
        if is_all_none:
            return result

        dy, dy_dim = dy_bdim
        x, x_dim = x_bdim
        target, target_dim = target_bdim
        dy = _bdim_at_front(dy, dy_dim, axis_size)
        x = _bdim_at_front(x, x_dim, axis_size)
        target = _bdim_at_front(target, target_dim, axis_size)

        out = kldivloss_grad(dy, x, target)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.SmoothL1Loss)
def get_smooth_l1_loss_vmap_rule(prim, axis_size):
    """VmapRule for `SmoothL1Loss` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)
        prim_beta = 1.0
        prim_reduction = 'none'
    else:
        prim_reduction = prim.reduction
        prim_beta = prim.beta

    smooth_l1_loss_op = P.SmoothL1Loss(prim_beta, 'none')
    if prim_reduction == 'mean':
        reduce_op = P.ReduceMean()
    elif prim_reduction == "sum":
        reduce_op = P.ReduceSum()

    def vmap_rule(x_bdim, target_bdim):
        is_all_none, result = vmap_general_preprocess(
            prim, x_bdim, target_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        target, target_dim = target_bdim
        x_ndim = F.rank(x)
        target_ndim = F.rank(target)
        max_rank = max(x_ndim, target_ndim)
        x = _bdim_at_front(x, x_dim, axis_size)
        target = _bdim_at_front(target, target_dim, axis_size)
        reduce_indexes = None
        # if rank is larger than 1, we need to reduce result when reduction != 'none'
        if max_rank > 1:
            reduce_indexes = tuple(range(1, max_rank))

        # elementwise style when reduction='none', otherwise reduce style
        if prim_reduction == "none":
            out = prim(x, target)
        elif prim_reduction in ("mean", "sum"):
            out = smooth_l1_loss_op(x, target)
            if reduce_indexes is not None:
                out = reduce_op(out, reduce_indexes)
        else:
            raise RuntimeError("For SmoothL1Loss vmap, reduction should be one of "
                               "['none', 'mean', 'sum'], but got '{}'".format(prim_reduction))
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(G.SmoothL1LossGrad)
def get_smooth_l1_loss_grad_vmap_rule(prim, axis_size):
    """VmapRule for `SmoothL1LossGrad`."""
    if isinstance(prim, str):
        prim = Primitive(prim)
        reduction = "none"
        beta = 1.0
    else:
        reduction = prim.reduction
        beta = prim.beta
    smooth_l1_loss_grad = G.SmoothL1LossGrad(beta, reduction)

    def vmap_rule(x_bdim, target_bdim, dy_bdim):
        is_all_none, result = vmap_general_preprocess(
            prim, dy_bdim, x_bdim, target_bdim)
        if is_all_none:
            return result

        dy, dy_dim = dy_bdim
        x, x_dim = x_bdim
        target, target_dim = target_bdim
        dy = _bdim_at_front(dy, dy_dim, axis_size)
        x = _bdim_at_front(x, x_dim, axis_size)
        target = _bdim_at_front(target, target_dim, axis_size)

        out = smooth_l1_loss_grad(x, target, dy)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.nn_ops.LogSoftmax)
def get_log_softmax_vmap_rule(prim, axis_size):
    """VmapRule for 'LogSoftmax' operation."""
    if isinstance(prim, str):
        axis = -1
    else:
        axis = prim.axis

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result
        x, x_dim = x_bdim
        x_ndim = F.rank(x) - 1

        batch_axis = axis + x_ndim if axis < 0 else axis
        batch_axis = batch_axis if batch_axis < x_dim else batch_axis + 1

        out = F.log_softmax(x, batch_axis)
        return out, x_dim

    return vmap_rule


@vmap_rules_getters.register(P.RandomCategorical)
def get_random_categorical_vmap_rule(prim, axis_size):
    """VmapRule for `RandomCategorical` operation."""

    default_dim = 2

    def vmap_rule(logits_bdim, num_sample_bdim, seed_bdim):
        is_all_none, result = vmap_general_preprocess(prim, logits_bdim, num_sample_bdim, seed_bdim)
        if is_all_none:
            return result
        logits, logits_dim = logits_bdim
        num_sample, num_sample_dim = num_sample_bdim
        seed, seed_dim = seed_bdim
        if num_sample_dim is not None or seed_dim is not None:
            raise RuntimeError("For RandomCategorical vmap, num_sample and seed should be None.")
        # Move axis to first dim
        logits = _bdim_at_front(logits, logits_dim, axis_size)
        x_ndim = F.rank(logits)
        if x_ndim > default_dim:
            x_ori_shape = F.shape(logits)
            logits = F.reshape(logits, (-1, x_ori_shape[-1]))
            dx = prim(logits, num_sample, seed)
            new_output_shape = (x_ori_shape[0], x_ori_shape[1], num_sample)
            dx = F.reshape(dx, new_output_shape)
        else:
            dx = prim(logits, num_sample, seed)
        return dx, 0

    return vmap_rule


@vmap_rules_getters.register(NN.LRN)
def get_lrn_vmap_rule(prim, axis_size):
    """VmapRule for `LRN` operation."""
    lrn_default_dim = 4
    lrn_pre_remain_dim = 3

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result
        input_x, input_x_dim = x_bdim
        # Move axis to last dim
        x = _bdim_at_back(input_x, input_x_dim, axis_size)
        x_ndim = F.rank(x)
        if x_ndim > lrn_default_dim:
            x_ori_shape = F.shape(x)
            x = F.reshape(x, x_ori_shape[:lrn_pre_remain_dim] + (-1,))
            out = prim(x)
            out = F.reshape(out, x_ori_shape)
        else:
            out = prim(x)
        return out, x_ndim - 1

    return vmap_rule


@vmap_rules_getters.register(NN.PadV3)
def get_pad_v3_vmap_rule(prim, axis_size):
    """VmapRule for `PadV3` operation."""
    pad_pair = 2
    input_max_dim = 4
    mode = prim.mode

    def vmap_rule(*params_bdim):
        is_all_none, result = vmap_general_preprocess(
            prim, params_bdim)
        if is_all_none:
            return result
        if len(params_bdim) < 2:
            _raise_value_error("The input params in `PadV3` must >= 2, "
                               "but got {}.".format(len(params_bdim)))
        input_x, input_x_dim = params_bdim[0]
        paddings, paddings_dim = params_bdim[1]
        values = None
        out = None
        x = _bdim_at_front(input_x, input_x_dim, axis_size)
        if paddings_dim is not None:
            _raise_value_error("The source axis of `paddings` in `PadV3` must be None, "
                               "but got {}.".format(paddings_dim))
        if mode == "constant":
            if len(params_bdim) != 3:
                _raise_value_error("The input params in `PadV3` of constant mode must be 3, "
                                   "but got {}.".format(len(params_bdim)))
            values, values_dim = params_bdim[2]
            if values_dim is not None:
                _raise_value_error("The source axis of `values_dim` in `PadV3` must be None, "
                                   "but got {}.".format(values_dim))
        if isinstance(paddings, Tensor):
            pad_dim = F.shape(paddings)[0] / pad_pair
        else:
            pad_dim = len(paddings) / pad_pair
        x_ndim = F.rank(x)
        # pylint: disable=chained-comparison
        if pad_dim < x_ndim and x_ndim < input_max_dim:
            if mode == "constant":
                out = prim(x, paddings, values)
            else:
                out = prim(x, paddings)
        elif x_ndim >= input_max_dim:
            # reshape to 4 dims
            x_shape = F.shape(x)
            diff_dim = x_ndim - input_max_dim
            first_shape = 1
            for i in range(diff_dim + 1):
                first_shape *= x_shape[i]
            input_shape = (first_shape,) + x_shape[(-input_max_dim + 1):]
            x = F.reshape(x, input_shape)
            if mode == "constant":
                out = prim(x, paddings, values)
            else:
                out = prim(x, paddings)
            out_shape = F.shape(out)
            real_out_shape = x_shape[:diff_dim + 1] + out_shape[1:]
            out = F.reshape(out, real_out_shape)
        else:
            _raise_value_error("The dim of `input_x` in `PadV3` must be bigger than {}, "
                               "but got {}.".format(pad_dim, x_ndim))
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(NN.MirrorPad)
def get_mirror_pad_vmap_rule(prim, axis_size):
    """VmapRule for `MirrorPad` operation."""
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


@vmap_rules_getters.register(G.LRNGrad)
def get_lrn_grad_vmap_rule(prim, axis_size):
    """VmapRule for `LRNGrad` operation."""
    lrn_default_dim = 4
    lrn_pre_remain_dim = 3

    def vmap_rule(dout_bdim, x_bdim, out_bdim):
        is_all_none, result = vmap_general_preprocess(prim, dout_bdim, x_bdim, out_bdim)
        if is_all_none:
            return result
        x, x_dim = x_bdim
        dy, dy_dim = dout_bdim
        y, y_dim = out_bdim
        # Move axis to last dim
        x = _bdim_at_back(x, x_dim, axis_size)
        dy = _bdim_at_back(dy, dy_dim, axis_size)
        y = _bdim_at_back(y, y_dim, axis_size)
        x_ndim = F.rank(x)
        if x_ndim > lrn_default_dim:
            x_ori_shape = F.shape(x)
            dy_ori_shape = F.shape(dy)
            y_ori_shape = F.shape(y)
            x = F.reshape(x, x_ori_shape[:lrn_pre_remain_dim] + (-1,))
            dy = F.reshape(dy, dy_ori_shape[:lrn_pre_remain_dim] + (-1,))
            y = F.reshape(y, y_ori_shape[:lrn_pre_remain_dim] + (-1,))
            dx = prim(dy, x, y)
            dx = F.reshape(dx, x_ori_shape)
        else:
            dx = prim(dy, x, y)
        return dx, x_ndim - 1

    return vmap_rule


@vmap_rules_getters.register(P.BatchNorm)
def get_batchnorm_vmap_rule(prim, axis_size):
    """VmapRule for `BatchNorm` operation."""
    bn_min_dim = 3
    bn_max_dim = 5
    is_training = prim.is_training
    prim_name = prim.name
    data_format = prim.data_format

    def vmap_rule(*inputs):
        x_bdim = inputs[0]
        scale_bdim = inputs[1]
        offset_bdim = inputs[2]
        mean_bdim = inputs[3]
        var_bdim = inputs[4]
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, scale_bdim, offset_bdim, mean_bdim, var_bdim)
        if is_all_none:
            return result
        if is_training:
            raise ValueError("Operator {} does not support Vmap during training, since the input `scale, offset, mean, "
                             "var of BatchNorm are parameters when is_training = true. If multiple batches of input "
                             "data share the same parameters, please stack batches to the N dimension manually."
                             .format(prim_name))
        input_x, input_x_dim = x_bdim
        scale, scale_dim = scale_bdim
        offset, offset_dim = offset_bdim
        mean, mean_dim = mean_bdim
        var, var_dim = var_bdim
        x_ndim = F.rank(input_x)
        if x_ndim < bn_min_dim or x_ndim > bn_max_dim:
            raise ValueError("The dim of `input_x` in `{}` must be larger than {} and less than {}, "
                             "but got {}.".format(prim_name, bn_min_dim - 1, bn_max_dim + 1, x_ndim))
        # Move input_x axis to the dim front of C
        out_axis = 1 if data_format == "NCHW" else x_ndim - 2
        input_x = _bdim_at_any(input_x, input_x_dim, out_axis, axis_size)
        scale = _bdim_at_front(scale, scale_dim, axis_size)
        offset = _bdim_at_front(offset, offset_dim, axis_size)
        mean = _bdim_at_front(mean, mean_dim, axis_size)
        var = _bdim_at_front(var, var_dim, axis_size)
        x_shape = input_x.shape
        other_shape = scale.shape
        vmap_shape = (x_shape[0], -1,) + x_shape[3:] if data_format == "NCHW" else x_shape[:-2] + (-1,)
        input_x = F.reshape(input_x, vmap_shape)
        scale = scale.flatten()
        offset = offset.flatten()
        mean = mean.flatten()
        var = var.flatten()
        out, batch_mean, batch_var, rsv_1, rsv_2 = prim(input_x, scale, offset, mean, var)
        out = F.reshape(out, x_shape)
        batch_mean = F.reshape(batch_mean, other_shape)
        batch_var = F.reshape(batch_var, other_shape)
        rsv_1 = F.reshape(rsv_1, other_shape)
        rsv_2 = F.reshape(rsv_2, other_shape)
        return (out, out_axis), (batch_mean, 0), (batch_var, 0), (rsv_1, 0), (rsv_2, 0)

    return vmap_rule


@vmap_rules_getters.register(P.ApplyAdamWithAmsgrad)
def get_apply_adam_with_amsgrad_rule(prim, axis_size):
    """VmapRule for `ApplyAdamWithAmsgrad` operation"""
    if hasattr(prim, "batch_rank"):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1
    prim_name = prim.name
    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr("batch_rank", batch_rank)

    def vmap_rule(var_bdim, m_bdim, v_bdim, vhat_bdim, beta1_power_bdim, beta2_power_bdim, lr_bdim, grad_bdim, u_monad):
        var, var_dim = var_bdim
        m, m_dim = m_bdim
        v, v_dim = v_bdim
        vhat, vhat_dim = vhat_bdim
        beta1_power, beta1_power_dim = beta1_power_bdim
        beta2_power, beta2_power_dim = beta2_power_bdim
        lr, lr_dim = lr_bdim
        grad, grad_dim = grad_bdim

        if var_dim is None:
            if any(dim is not None for dim in [m_dim, v_dim, vhat_dim, beta1_power_dim,
                                               beta2_power_dim, lr_dim, grad_dim]):
                raise ValueError("The source axis of `var` is None, "
                                 "but the source axis of `m/v/vhat/beta1_power/beta2_power/lr/grad` is not None. "
                                 "The execution of operator `{}` cannot be guaranteed.".format(prim_name))
            out_var, out_m, out_v, out_vhat = prim(var, m, v, vhat, beta1_power, beta2_power, lr, grad, u_monad)
            return (out_var, None), (out_m, None), (out_v, None), (out_vhat, None)

        if any(dim != 0 for dim in [var_dim, m_dim, v_dim, vhat_dim]):
            raise ValueError("For `{}`, the source axis of `var/m/v/vhat` must be 0, "
                             "but get `var`: {}, `m`: {}, `v`: {}, `vhat`: {}".format(prim_name, var_dim,
                                                                                      m_dim, v_dim, vhat_dim))

        beta1_power = _bdim_at_front(beta1_power, beta1_power_dim, axis_size)
        beta2_power = _bdim_at_front(beta2_power, beta2_power_dim, axis_size)
        lr = _bdim_at_front(lr, lr_dim, axis_size)
        grad = _bdim_at_front(grad, grad_dim, axis_size)

        out_var, out_m, out_v, out_vhat = batch_prim(var, m, v, vhat, beta1_power, beta2_power, lr, grad, u_monad)
        return (out_var, 0), (out_m, 0), (out_v, 0), (out_vhat, 0)

    return vmap_rule


@vmap_rules_getters.register(P.Adam)
def get_adam_rule(prim, axis_size):
    """VmapRule for `Adam` operation"""
    if hasattr(prim, "batch_rank"):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1
    prim_name = prim.name
    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr("batch_rank", batch_rank)

    def vmap_rule(var_bdim, m_bdim, v_bdim, beta1_power_bdim, beta2_power_bdim, lr_bdim, beta1_bdim,
                  beta2_bdim, epsilon_bdim, grad_bdim, u_monad):
        var, var_dim = var_bdim
        m, m_dim = m_bdim
        v, v_dim = v_bdim
        beta1_power, beta1_power_dim = beta1_power_bdim
        beta2_power, beta2_power_dim = beta2_power_bdim
        lr, lr_dim = lr_bdim
        beta1, beta1_dim = beta1_bdim
        beta2, beta2_dim = beta2_bdim
        epsilon, epsilon_dim = epsilon_bdim
        grad, grad_dim = grad_bdim

        if var_dim is None:
            if any(dim is not None for dim in [m_dim, v_dim, beta1_power_dim,
                                               beta2_power_dim, lr_dim, beta1_dim,
                                               beta2_dim, epsilon_dim, grad_dim]):
                raise ValueError("The source axis of `var` is None, "
                                 "but the source axis of `m/v/vhat/beta1_power/beta2_power/lr/beta1/beta2/epsilon grad"
                                 " is not None. The execution of operator `{}` cannot be guaranteed.".format(prim_name))
            out_var, out_m, out_v = prim(
                var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, u_monad)
            return ((out_var, None), (out_m, None), (out_v, None))

        if any(dim != 0 for dim in [var_dim, m_dim, v_dim]):
            raise ValueError("For `{}`, the source axis of `var/m/v` must be 0, "
                             "but get `var`: {}, `m`: {}, `v`: {}".format(prim_name, var_dim,
                                                                          m_dim, v_dim))

        beta1_power = _bdim_at_front(beta1_power, beta1_power_dim, axis_size)
        beta2_power = _bdim_at_front(beta2_power, beta2_power_dim, axis_size)
        lr = _bdim_at_front(lr, lr_dim, axis_size)
        beta1 = _bdim_at_front(beta1, beta1_dim, axis_size)
        beta2 = _bdim_at_front(beta2, beta2_dim, axis_size)
        epsilon = _bdim_at_front(epsilon, epsilon_dim, axis_size)
        grad = _bdim_at_front(grad, grad_dim, axis_size)

        out_var, out_m, out_v = batch_prim(
            var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, u_monad)
        return ((out_var, 0), (out_m, 0), (out_v, 0))

    return vmap_rule


@vmap_rules_getters.register(P.ApplyPowerSign)
def get_apply_power_sign_rule(prim, axis_size):
    """VmapRule for `ApplyPowerSign` operation."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    prim_name = prim.name
    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr("batch_rank", batch_rank)

    def vmap_rule(var_bdim, m_bdim, lr_bdim, logbase_bdim, sign_decay_bdim, beta_bdim, grad_bdim, u_monad):
        var, var_dim = var_bdim
        m, m_dim = m_bdim
        lr, lr_dim = lr_bdim
        logbase, logbase_dim = logbase_bdim
        sign_decay, sign_decay_dim = sign_decay_bdim
        beta, beta_dim = beta_bdim
        grad, grad_dim = grad_bdim

        if var_dim is None:
            if any(dim is not None for dim in [m_bdim, lr_bdim, logbase_bdim, sign_decay_bdim, beta_bdim, grad_bdim]):
                raise ValueError("The source axis of `var` is None, but the source "
                                 "axis of `m/lr/logbase/sign_decay/beta/grad` is not None. The execution order of "
                                 "operator `{}` cannot be guaranteed.".format(prim_name))
            var, m = prim(var, m, lr, logbase, sign_decay, beta, grad, u_monad)
            return (var, None), (m, None)
        if var_dim != 0 or m_dim != var_dim:
            raise ValueError("For `{}`, the source axis of `var` must be equal to `m`, and not equal to 0, "
                             "but got the source axis of `var`: {}, `m`: {}.".format(prim_name, var_dim, m_dim))

        lr = _bdim_at_front(lr, lr_dim, axis_size)
        logbase = _bdim_at_front(logbase, logbase_dim, axis_size)
        sign_decay = _bdim_at_front(sign_decay, sign_decay_dim, axis_size)
        beta = _bdim_at_front(beta, beta_dim, axis_size)
        grad = _bdim_at_front(grad, grad_dim, axis_size)
        var, m = batch_prim(var, m, lr, logbase, sign_decay, beta, grad, u_monad)
        return (var, 0), (m, 0)

    return vmap_rule


@vmap_rules_getters.register(P.ApplyAdagradV2)
def get_apply_adagrad_v2_vmap_rule(prim, axis_size):
    """VmapRule for `ApplyAdagradV2` operation."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr('batch_rank', batch_rank)
    prim_name = prim.name

    def vmap_rule(var_bdim, accum_bdim, lr_bdim, grad_bdim, u_monad):
        var, var_dim = var_bdim
        accum, accum_dim = accum_bdim
        lr, lr_dim = lr_bdim
        grad, grad_dim = grad_bdim

        if var_dim is None:
            if any(dim is not None for dim in
                   [accum_bdim, lr_dim, grad_bdim]):
                raise ValueError("The source axis of 'var' is None, but the source "
                                 "axis of 'accum/lr/grad'"
                                 " is not None. The execution order of "
                                 "operator '{}' cannot be guaranteed.".format(prim_name))
            var, accum = prim(var, accum, lr, grad, u_monad)
            return (var, None), (accum, None)
        if var_dim != 0 or var_dim != accum_dim:
            raise ValueError(
                f"For '{prim_name}', the source axis of 'var' must be equal to 'accum_dim' "
                f"and not equal to 0, but got the source axis of 'var': {var_dim}, "
                f"'accum_dim': {accum_dim}")

        lr = _bdim_at_front(lr, lr_dim, axis_size)
        grad = _bdim_at_front(grad, grad_dim, axis_size)

        var, accum = batch_prim(var, accum, lr, grad, u_monad)
        return (var, 0), (accum, 0)

    return vmap_rule


@vmap_rules_getters.register(P.ApplyAdagradDA)
def get_apply_adagrad_da_vmap_rule(prim, axis_size):
    """VmapRule for `ApplyAdagradDA` operation."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    attr = prim.init_attrs
    batch_prim = P.ApplyAdagradDA(**attr)
    batch_prim.add_prim_attr('batch_rank', batch_rank)
    prim_name = prim.name

    def vmap_rule(var_bdim, gradient_accumulator_bdim, gradient_squared_accumulator_bdim, grad_bdim, lr_bdim, l1_bdim,
                  l2_bdim, global_step_bdim, u_monad):
        var, var_dim = var_bdim
        gradient_accumulator, gradient_accumulator_dim = gradient_accumulator_bdim
        gradient_squared_accumulator, gradient_squared_accumulator_dim = gradient_squared_accumulator_bdim
        grad, grad_dim = grad_bdim
        lr, lr_dim = lr_bdim
        l1, l1_dim = l1_bdim
        l2, l2_dim = l2_bdim
        global_step, global_step_dim = global_step_bdim

        if var_dim is None:
            if any(dim is not None for dim in
                   [gradient_accumulator_bdim, gradient_squared_accumulator_bdim, grad_bdim, lr_bdim, l1_bdim, l2_bdim,
                    global_step_bdim]):
                raise ValueError("The source axis of 'var' is None, but the source "
                                 "axis of 'gradient_accumulator/gradient_squared_accumulator/grad/lr/l1/l2/global_step'"
                                 " is not None. The execution order of "
                                 "operator '{}' cannot be guaranteed.".format(prim_name))
            var, gradient_accumulator, gradient_squared_accumulator = prim(var, gradient_accumulator,
                                                                           gradient_squared_accumulator, grad, lr, l1,
                                                                           l2,
                                                                           global_step,
                                                                           u_monad)  # Low dimensional operator
            return (var, None), (gradient_accumulator, None), (gradient_squared_accumulator, None)
        if var_dim != 0 or var_dim != gradient_accumulator_dim or var_dim != gradient_squared_accumulator_dim:
            raise ValueError(
                f"For '{prim_name}', the source axis of 'var' must be equal to 'gradient_accumulator_dim' "
                f"and 'gradient_squared_accumulator_dim' and not equal to 0, "
                f"but got the source axis of 'var': {var_dim}, "
                f"'gradient_accumulator_dim': {gradient_accumulator_dim}, "
                f"'gradient_squared_accumulator_dim': {gradient_squared_accumulator_dim}")

        grad = _bdim_at_front(grad, grad_dim, axis_size)
        lr = _bdim_at_front(lr, lr_dim, axis_size)
        l1 = _bdim_at_front(l1, l1_dim, axis_size)
        l2 = _bdim_at_front(l2, l2_dim, axis_size)
        global_step = _bdim_at_front(global_step, global_step_dim, axis_size)

        var, gradient_accumulator, gradient_squared_accumulator = batch_prim(var, gradient_accumulator,
                                                                             gradient_squared_accumulator, grad, lr, l1,
                                                                             l2,
                                                                             global_step,
                                                                             u_monad)  # High dimensional operator;
        return (var, 0), (gradient_accumulator, 0), (gradient_squared_accumulator, 0)

    return vmap_rule


@vmap_rules_getters.register(NN.AdaptiveMaxPool2D)
def get_adaptive_max_pool_2d_vmap_rule(prim, axis_size):
    """VmapRule for `AdaptiveMaxPool2D`."""
    nchw_index = 4
    chw_reverse_index = -3
    hw_size = 2
    output_size = prim.output_size

    @_primexpr
    def get_output_shape(x_ori_shape, output_size):
        if isinstance(output_size, tuple):
            h_out, w_out = output_size
        else:
            h_out = output_size
            w_out = output_size

        rank = len(x_ori_shape)
        output_shape = x_ori_shape[:rank - hw_size]
        if h_out is None or h_out == -1:
            output_shape += (x_ori_shape[-2],)
        else:
            output_shape += (h_out,)

        if w_out is None or w_out == -1:
            output_shape += (x_ori_shape[-1],)
        else:
            output_shape += (w_out,)
        return output_shape

    def vmap_rule(input_x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, input_x_bdim)
        if is_all_none:
            return result

        input_x, input_x_dim = input_x_bdim
        x = _bdim_at_front(input_x, input_x_dim, axis_size)
        x_ndim = F.rank(x)

        if x_ndim > nchw_index:
            # for the case of NCHW
            x_ori_shape = F.shape(x)
            x = F.reshape(x, (-1,) + x_ori_shape[chw_reverse_index:])
            output_shape = get_output_shape(x_ori_shape, output_size)
            out, indices = prim(x)
            out = F.reshape(out, output_shape)
            indices = F.reshape(indices, output_shape)
            return (out, 0), (indices, 0)

        # for the case of CHW
        out, indices = prim(x)
        return (out, 0), (indices, 0)

    return vmap_rule


@vmap_rules_getters.register(NN.MaxPool3DWithArgmax)
def get_max_pool3d_with_argmax_vmap_rule(prim, axis_size):
    """VmapRule for `MaxPool3DWithArgmax`."""
    cdhw_reverse_index = -4

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        x_shape = F.shape(x)
        input_shape = (-1,) + x_shape[cdhw_reverse_index:]
        x = F.reshape(x, input_shape)
        out, indices = prim(x)
        out_shape = F.shape(out)
        return_shape = x_shape[:cdhw_reverse_index] + out_shape[cdhw_reverse_index:]
        out = F.reshape(out, return_shape)
        indices = F.reshape(indices, return_shape)
        return (out, 0), (indices, 0)

    return vmap_rule


@vmap_rules_getters.register(P.ApplyRMSProp)
def get_rmsprop_vmap_rule(prim, axis_size):
    """VmapRule for `ApplyRMSProp` operation."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr('batch_rank', batch_rank)
    prim_name = prim.name

    def vmap_rule(var_bdim, mean_square_bdim, moment_bdim, lr_bdim, grad_bdim, decay_bdim, momentum_bdim,
                  epsilon_bdim, u_monad):
        var, var_dim = var_bdim
        mean_square, mean_square_dim = mean_square_bdim
        moment, moment_dim = moment_bdim
        grad, grad_dim = grad_bdim
        lr, lr_dim = lr_bdim
        decay, decay_dim = decay_bdim
        momentum, momentum_dim = momentum_bdim
        epsilon, epsilon_dim = epsilon_bdim

        if var_dim is None:
            if any(dim is not None for dim in
                   [mean_square_dim, moment_dim, grad_dim, lr_dim, decay_dim, momentum_dim, epsilon_dim]):
                raise ValueError("The source axis of 'var' is None, but the source "
                                 "axis of 'mean_square/moment/lr/grad/decay/momentum/epsilon'"
                                 " is not None. The execution order of "
                                 "operator '{}' cannot be guaranteed.".format(prim_name))

            res = prim(var, mean_square, moment, lr, grad, decay, momentum, epsilon,
                       u_monad)  # low dimensional operator;
            return (res, None)
        if var_dim != 0 or var_dim != mean_square_dim or var_dim != moment_dim or var_dim != grad_dim:
            raise ValueError(
                f"For '{prim_name}', the source axis of 'var' must be equal to 'mean_square_dim' "
                f"and 'moment_dim' and 'grad_dim' and not equal to 0, "
                f"but got the source axis of 'var': {var_dim}, "
                f"'mean_square_dim': {mean_square_dim}, "
                f"'moment_dim': {moment_dim},"
                f"'gradient_dim':{grad_dim}.")

        mean_square = _bdim_at_front(mean_square, mean_square_dim, axis_size)
        moment = _bdim_at_front(moment, moment_dim, axis_size)
        grad = _bdim_at_front(grad, grad_dim, axis_size)
        lr = _bdim_at_front(lr, lr_dim, axis_size)

        res = batch_prim(var, mean_square, moment, lr, grad, decay, momentum, epsilon,
                         u_monad)  # High dimensional operator;

        return res, 0

    return vmap_rule


@vmap_rules_getters.register(P.ApplyCenteredRMSProp)
def get_apply_centered_rmsprop_vmap_rule(prim, axis_size):
    """VmapRule for `ApplyCenteredRMSProp` operation."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1
    prim_name = prim.name
    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr("batch_rank", batch_rank)

    def vmap_rule(var_bdim, mean_grad_bdim, mean_square_bdim, mom_bdim, grad_bdim, lr_bdim, rho_bdim,
                  momentum_bdim, eps_bdim, u_monad):
        var, var_dim = var_bdim
        mean_grad, mean_grad_dim = mean_grad_bdim
        mean_square, mean_square_dim = mean_square_bdim
        mom, mom_dim = mom_bdim
        grad, grad_dim = grad_bdim
        lr, lr_dim = lr_bdim
        rho, rho_dim = rho_bdim
        momentum, momentum_dim = momentum_bdim
        eps, eps_dim = eps_bdim

        if var_dim is None:
            if any(dim is not None for dim in
                   [mean_grad_dim, mean_square_dim, mom_dim, grad_dim, lr_dim, rho_dim,
                    momentum_dim, eps_dim]):
                raise ValueError("The source axis of 'var' is None, but the source "
                                 "axis of 'mean_gradient/mean_square/mom/grad/lr/rho/momentum/eps'"
                                 " is not None. The execution order of "
                                 "operator '{}' cannot be guaranteed.".format(prim_name))
            var = prim(var, mean_grad, mean_square,
                       mom, grad, lr, rho, momentum, eps, u_monad)
            return (var, None)

        if var_dim != 0 or var_dim != mean_grad_dim or var_dim != mean_square_dim or var_dim != mom_dim:
            raise ValueError(
                f"For '{prim_name}', the source axis of 'var' must be equal to 'mean_grad_dim' "
                f"and 'mean_square_dim' and 'mom_dim' and not equal to 0, "
                f"but got the source axis of 'var': {var_dim}, "
                f"'mean_grad_dim': {mean_grad_dim}, "
                f"'mean_square_dim': {mean_square_dim},"
                f"'mom_dim': {mom_dim}.")

        grad = _bdim_at_front(grad, grad_dim, axis_size)
        lr = _bdim_at_front(lr, lr_dim, axis_size)
        rho = _bdim_at_front(rho, rho_dim, axis_size)
        momentum = _bdim_at_front(momentum, momentum_dim, axis_size)
        eps = _bdim_at_front(eps, eps_dim, axis_size)

        var = batch_prim(var, mean_grad, mean_square,
                         mom, grad, lr, rho, momentum, eps, u_monad)
        return var, 0

    return vmap_rule


@vmap_rules_getters.register(P.MaxPool)
@vmap_rules_getters.register(P.MaxPoolWithArgmax)
@vmap_rules_getters.register(P.MaxPoolWithArgmaxV2)
def get_max_pool_vmap_rule(prim, axis_size):
    """VmapRule for `MaxPool` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    prim_name = prim.name

    @_primexpr
    def get_original_shape(x_shape, out_shape):
        h_new = out_shape[2]
        w_new = out_shape[3]
        original_shape = x_shape[:3] + (h_new,) + (w_new,)
        return original_shape

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result
        x, x_dim = x_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        x_shape = x.shape
        x_new_shape = (-1,) + x_shape[2:]
        x = x.reshape(x_new_shape)
        if prim_name == "MaxPool":
            out = prim(x)
            out_shape = out.shape
            original_shape = get_original_shape(x_shape, out_shape)
            out = out.reshape(original_shape)
            return out, 0
        out, indices = prim(x)
        out_shape = out.shape
        original_shape = get_original_shape(x_shape, out_shape)
        out = out.reshape(original_shape)
        indices = indices.reshape(original_shape)
        return (out, 0), (indices, 0)

    return vmap_rule


@vmap_rules_getters.register(P.LayerNorm)
def get_layernorm_vmap_rule(prim, axis_size):
    """VmapRule for `LayerNorm` operation."""
    @constexpr
    def process_attr_axis(prim_attr_axis):
        if prim_attr_axis < 0:
            return prim_attr_axis
        return prim_attr_axis + 1

    norm_axis = process_attr_axis(prim.begin_norm_axis)
    params_axis = process_attr_axis(prim.begin_params_axis)
    batch_prim = P.LayerNorm(norm_axis, params_axis, prim.epsilon)

    @_primexpr
    def get_logical_shape(var_shape):
        return var_shape[1:]

    def vmap_rule(x_bdim, gamma_bdim, beta_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, gamma_bdim, beta_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        g, g_dim = gamma_bdim
        b, b_dim = beta_bdim

        x = _bdim_at_front(x, x_dim, axis_size)

        if g_dim is None and b_dim is None:
            output, mean, var = batch_prim(x, g, b)
            return (output, 0), (mean, 0), (var, 0)

        g = _bdim_at_front(g, g_dim, axis_size)
        b = _bdim_at_front(b, b_dim, axis_size)
        g_logical_shape = get_logical_shape(F.shape(g))
        b_logical_shape = get_logical_shape(F.shape(b))

        ones_like_g = F.ones(g_logical_shape, F.dtype(g))
        zeros_like_b = F.zeros(b_logical_shape, F.dtype(b))
        output_tmp, mean, var = batch_prim(x, ones_like_g, zeros_like_b)

        x_shape = F.shape(x)
        g_shape = F.shape(g)
        b_shape = F.shape(b)
        g = _handle_broadcasting(g, g_shape, x_shape)
        b = _handle_broadcasting(b, b_shape, x_shape)
        output = F.add(F.mul(output_tmp, g), b)

        return (output, 0), (mean, 0), (var, 0)
    return vmap_rule


@vmap_rules_getters.register(NN.GridSampler2D)
@vmap_rules_getters.register(NN.GridSampler3D)
def get_grid_sampler_vmap_rule(prim, axis_size):
    """VmapRule for `GridSampler2D` and `GridSampler3D`."""
    prim_name = prim.name
    if prim_name == "GridSampler2D":
        non_batch_dim_index = -3
    else:
        non_batch_dim_index = -4

    def vmap_rule(input_x_bdim, grid_bdim):
        is_all_none, result = vmap_general_preprocess(prim, input_x_bdim, grid_bdim)
        if is_all_none:
            return result

        input_x, input_x_dim = input_x_bdim
        grid, grid_dim = grid_bdim

        input_x = _bdim_at_front(input_x, input_x_dim, axis_size)
        input_x_shape = F.shape(input_x)
        input_x = F.reshape(input_x, (-1,) + input_x_shape[non_batch_dim_index:])

        grid = _bdim_at_front(grid, grid_dim, axis_size)
        grid_shape = F.shape(grid)
        grid = F.reshape(grid, (-1,) + grid_shape[non_batch_dim_index:])

        out = prim(input_x, grid)
        out_shape = F.shape(out)
        return_shape = input_x_shape[:non_batch_dim_index] + out_shape[non_batch_dim_index:]
        out = F.reshape(out, return_shape)
        return out, 0
    return vmap_rule


@vmap_rules_getters.register(NN.UpsampleNearest3D)
@vmap_rules_getters.register(NN.UpsampleTrilinear3D)
def get_upsample_nearest_3d_vmap_rule(prim, axis_size):
    """VmapRule for `UpsampleNearest3D` and `UpsampleTrilinear3D`."""
    cdhw_reverse_index = -4

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        x_shape = F.shape(x)
        input_shape = (-1,) + x_shape[cdhw_reverse_index:]
        x = F.reshape(x, input_shape)
        out = prim(x)
        out_shape = F.shape(out)
        return_shape = x_shape[:cdhw_reverse_index] + out_shape[cdhw_reverse_index:]
        out = F.reshape(out, return_shape)
        return out, 0
    return vmap_rule


@vmap_rules_getters.register(NN.SparseApplyAdagrad)
@vmap_rules_getters.register(NN.SparseApplyAdagradV2)
def get_sparse_apply_adagrad_vmap_rule(prim, axis_size):
    """VmapRule for `SparseApplyAdagrad`."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    prim_name = prim.name
    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr('batch_rank', batch_rank)

    def vmap_rule(var_bdim, accum_bdim, grad_bdim, indices_bdim, u_monad):
        var, var_dim = var_bdim
        accum, accum_dim = accum_bdim
        grad, grad_dim = grad_bdim
        indices, indices_dim = indices_bdim
        if var_dim is None:
            if any(dim is not None for dim in [accum_dim, grad_dim, indices_dim]):
                ValueError("The source axis of `var` is None, but the source "
                           "axis of `accum/grad/indices` is not None. The execution order of "
                           "operator `{}` cannot be guaranteed.".format(prim_name))
            var, accum = prim(var, accum, grad, indices, u_monad)
            return (var, None), (accum, None)
        if var_dim != 0 or accum_dim != var_dim:
            ValueError("For `{}`, the source axis of `var` must be equal to `accum`, and not equal to 0, "
                       "but got the source axis of `var`: {}, `accum`: {}.".format(prim_name, var_dim, accum_dim))

        grad = _bdim_at_front(grad, grad_dim, axis_size)
        indices = _bdim_at_front(indices, indices_dim, axis_size)

        var, accum = batch_prim(var, accum, grad, indices, u_monad)
        return (var, 0), (accum, 0)
    return vmap_rule


@vmap_rules_getters.register(NN.SparseApplyFtrl)
def get_sparse_apply_ftrl_vmap_rule(prim, axis_size):
    """VmapRule for `SparseApplyFtrl`."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    prim_name = prim.name
    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr('batch_rank', batch_rank)

    def vmap_rule(var_bdim, accum_bdim, linear_bdim, grad_bdim, indices_bdim, u_monad):
        var, var_dim = var_bdim
        accum, accum_dim = accum_bdim
        linear, linear_dim = linear_bdim
        grad, grad_dim = grad_bdim
        indices, indices_dim = indices_bdim
        if var_dim is None:
            if any(dim is not None for dim in [accum_dim, linear_dim, grad_dim, indices_dim]):
                ValueError("The source axis of `var` is None, but the source "
                           "axis of `accum/linear/grad/indices` is not None. The execution order of "
                           "operator `{}` cannot be guaranteed.".format(prim_name))
            var, accum, linear = prim(var, accum, linear, grad, indices, u_monad)
            return (var, None), (accum, None), (linear, None)
        if var_dim != 0 or accum_dim != var_dim or linear_dim != var_dim:
            ValueError("For `{}`, the source axis of `var`, `accum` and `linear` must be equal, and "
                       "not equal to 0, but got the source axis of `var`: {}, `accum`: {}, "
                       "`linear`:{}.".format(prim_name, var_dim, accum_dim, linear_dim))

        grad = _bdim_at_front(grad, grad_dim, axis_size)
        indices = _bdim_at_front(indices, indices_dim, axis_size)

        var, accum, linear = batch_prim(var, accum, linear, grad, indices, u_monad)
        return (var, 0), (accum, 0), (linear, 0)
    return vmap_rule


# Unary vmap
get_unop_vmap_rule = vmap_rules_getters.register(P.Elu)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.ReLU)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.ReLU6)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.CeLU)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.SeLU)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.HSigmoid)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Softplus)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Softsign)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.SoftShrink)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.HShrink)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.GeLU)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.FastGeLU)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.HSwish)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Tanh)(get_unop_vmap_rule)
# UnaryGrad vmap
get_unary_grad_vmap_rule = vmap_rules_getters.register(G.TanhGrad)(get_unary_grad_vmap_rule)
get_unary_grad_vmap_rule = vmap_rules_getters.register(G.SoftplusGrad)(get_unary_grad_vmap_rule)
get_unary_grad_vmap_rule = vmap_rules_getters.register('ReluGrad')(get_unary_grad_vmap_rule)
