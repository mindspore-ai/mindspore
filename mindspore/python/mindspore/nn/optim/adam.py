# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""adam"""
from __future__ import absolute_import, division

import numpy as np

from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register
from mindspore.nn.optim._dist_optimizer_registry import _register_dist_optimizer
from mindspore.common._decorator import deprecated

_adam_opt = C.MultitypeFuncGraph("adam_opt")
_fused_adam_weight_decay = C.MultitypeFuncGraph("fused_adam_weight_decay")
_lazy_adam_opt = C.MultitypeFuncGraph("lazy_adam_opt")
_scaler_one = Tensor(1, mstype.int32)
_scaler_ten = Tensor(10, mstype.float32)


@_lazy_adam_opt.register("Function", "Function", "Function", "Function", "Bool", "Bool", "Bool", "Tensor", "Tensor",
                         "Tensor", "Tensor", "Tensor", "Tensor", "RowTensor", "Tensor", "Tensor", "Tensor", "Bool",
                         "Bool", "Function", "Bool", "Function", "Bool")
def _run_lazy_opt_with_sparse_dist(opt, sparse_opt, push, pull, use_locking, use_nesterov, target, beta1_power,
                                   beta2_power, beta1, beta2, eps, lr, gradient, params, m, v, ps_parameter,
                                   cache_enable, distributed_opt, use_flag, distributed_sparse_opt, use_sparse_flag):
    """Apply sparse lazy adam optimizer to the weight parameter when the gradient is sparse."""
    success = True
    indices = gradient.indices
    values = gradient.values
    if use_sparse_flag:
        success = F.depend(success, distributed_sparse_opt(params, m, v, beta1_power, beta2_power, lr, beta1, beta2,
                                                           eps, values, indices))
        return success
    if ps_parameter and not cache_enable:
        op_shape = P.Shape()
        shapes = (op_shape(params), op_shape(m), op_shape(v),
                  op_shape(beta1_power), op_shape(beta2_power), op_shape(lr), op_shape(beta1),
                  op_shape(beta2), op_shape(eps), op_shape(values), op_shape(indices))
        success = F.depend(success, pull(push((beta1_power, beta2_power, lr, beta1, beta2,
                                               eps, values, indices), shapes), params))
        return success

    if not target:
        success = F.depend(success, sparse_opt(params, m, v, beta1_power, beta2_power, lr, beta1, beta2,
                                               eps, values, indices))
    else:
        op_gather = P.Gather()
        op_sqrt = P.Sqrt()
        scatter_add = P.ScatterAdd(use_locking)
        scatter_update = P.ScatterUpdate(use_locking)

        m_slice = op_gather(m, indices, 0)
        v_slice = op_gather(v, indices, 0)

        next_m = m_slice * beta1 + values * (1 - beta1)
        next_v = v_slice * beta2 + values * values * (1 - beta2)

        lr_t = lr * op_sqrt(1 - beta2_power) / (1 - beta1_power)

        if use_nesterov:
            m_temp = beta1 * next_m + values * (1 - beta1)
            param_update = m_temp / (op_sqrt(next_v) + eps)
        else:
            param_update = next_m / (op_sqrt(next_v) + eps)

        success = F.depend(success, scatter_add(params, indices, - lr_t * param_update))
        success = F.depend(success, scatter_update(m, indices, next_m))
        success = F.depend(success, scatter_update(v, indices, next_v))

    return success


@_lazy_adam_opt.register("Function", "Function", "Function", "Function", "Bool", "Bool", "Bool", "Tensor", "Tensor",
                         "Tensor", "Tensor", "Tensor", "Tensor", "MapTensor", "MapTensor", "MapTensor", "MapTensor",
                         "Bool", "Bool", "Function", "Bool", "Function", "Bool")
def _run_map_tensor_lazy_opt_with_sparse_dist(opt, sparse_opt, push, pull, use_locking, use_nesterov, target,
                                              beta1_power, beta2_power, beta1, beta2, eps, lr, gradient, params, m, v,
                                              ps_parameter, cache_enable, distributed_opt, use_flag,
                                              distributed_sparse_opt, use_sparse_flag):
    """Apply sparse lazy adam optimizer to the weight parameter when the gradient is sparse."""
    success = True
    indices, values = gradient.get_data()
    if use_sparse_flag:
        # PS Mode.
        success = F.depend(success, distributed_sparse_opt(params, m, v, beta1_power, beta2_power, lr, beta1, beta2,
                                                           eps, values, indices))
    else:
        # PS Cache mode.
        op_sqrt = P.Sqrt()

        m_slice = m.get(indices)
        v_slice = v.get(indices)

        next_m = m_slice * beta1 + values * (1 - beta1)
        next_v = v_slice * beta2 + values * values * (1 - beta2)

        lr_t = lr * op_sqrt(1 - beta2_power) / (1 - beta1_power)

        if use_nesterov:
            m_temp = beta1 * next_m + values * (1 - beta1)
            param_update = m_temp / (op_sqrt(next_v) + eps)
        else:
            param_update = next_m / (op_sqrt(next_v) + eps)

        params_need_update = params.get(indices)
        params.put(indices, params_need_update - lr_t * param_update)
        m.put(indices, next_m)
        v.put(indices, next_v)

    return success


@_lazy_adam_opt.register("Function", "Function", "Function", "Function", "Bool", "Bool", "Bool", "Tensor", "Tensor",
                         "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool",
                         "Function", "Bool", "Function", "Bool")
def _run_lazy_opt_with_one_number_dist(opt, sparse_opt, push, pull, use_locking, use_nesterov, target,
                                       beta1_power, beta2_power, beta1, beta2, eps, lr, gradient, params, moment1,
                                       moment2, ps_parameter, cache_enable, distributed_opt, use_flag,
                                       distributed_sparse_opt, use_sparse_flag):
    """Apply lazy adam optimizer to the weight parameter using Tensor."""
    success = True
    if use_flag:
        success = F.depend(success, distributed_opt(params, moment1, moment2, beta1_power, beta2_power, lr, beta1,
                                                    beta2, eps, gradient))
    elif ps_parameter and not cache_enable:
        op_shape = P.Shape()
        success = F.depend(success, pull(push((beta1_power, beta2_power, lr, beta1, beta2, eps, gradient),
                                              (op_shape(params), op_shape(moment1), op_shape(moment2))), params))
    else:
        success = F.depend(success, opt(params, moment1, moment2, beta1_power, beta2_power, lr, beta1, beta2,
                                        eps, gradient))
    return success


@_lazy_adam_opt.register("Function", "Function", "Function", "Function", "Bool", "Bool", "Bool", "Tensor", "Tensor",
                         "Tensor", "Tensor", "Tensor", "Tensor", "RowTensor", "Tensor", "Tensor", "Tensor", "Bool",
                         "Bool")
def _run_lazy_opt_with_sparse(opt, sparse_opt, push, pull, use_locking, use_nesterov, target, beta1_power, beta2_power,
                              beta1, beta2, eps, lr, gradient, params, m, v, ps_parameter, cache_enable):
    """Apply sparse lazy adam optimizer to the weight parameter when the gradient is sparse."""
    success = True
    indices = gradient.indices
    values = gradient.values
    if ps_parameter and not cache_enable:
        op_shape = P.Shape()
        shapes = (op_shape(params), op_shape(m), op_shape(v),
                  op_shape(beta1_power), op_shape(beta2_power), op_shape(lr), op_shape(beta1),
                  op_shape(beta2), op_shape(eps), op_shape(values), op_shape(indices))
        success = F.depend(success, pull(push((beta1_power, beta2_power, lr, beta1, beta2,
                                               eps, values, indices), shapes), params))
        return success

    if not target:
        success = F.depend(success, sparse_opt(params, m, v, beta1_power, beta2_power, lr, beta1, beta2,
                                               eps, values, indices))
    else:
        op_gather = P.Gather()
        op_sqrt = P.Sqrt()
        scatter_add = P.ScatterAdd(use_locking)
        scatter_update = P.ScatterUpdate(use_locking)

        m_slice = op_gather(m, indices, 0)
        v_slice = op_gather(v, indices, 0)

        next_m = m_slice * beta1 + values * (1 - beta1)
        next_v = v_slice * beta2 + values * values * (1 - beta2)

        lr_t = lr * op_sqrt(1 - beta2_power) / (1 - beta1_power)

        if use_nesterov:
            m_temp = beta1 * next_m + values * (1 - beta1)
            param_update = m_temp / (op_sqrt(next_v) + eps)
        else:
            param_update = next_m / (op_sqrt(next_v) + eps)

        success = F.depend(success, scatter_add(params, indices, - lr_t * param_update))
        success = F.depend(success, scatter_update(m, indices, next_m))
        success = F.depend(success, scatter_update(v, indices, next_v))

    return success


@_lazy_adam_opt.register("Function", "Function", "Function", "Function", "Bool", "Bool", "Bool", "Tensor", "Tensor",
                         "Tensor", "Tensor", "Tensor", "Tensor", "MapTensor", "MapTensor", "MapTensor", "MapTensor",
                         "Bool", "Bool")
def _run_map_tensor_lazy_opt_with_sparse(opt, sparse_opt, push, pull, use_locking, use_nesterov, target, beta1_power,
                                         beta2_power, beta1, beta2, eps, lr, gradient, params, m, v, ps_parameter,
                                         cache_enable):
    """Apply sparse lazy adam optimizer to the weight parameter when the gradient is sparse(MapTensor)."""
    success = True
    indices, values = gradient.get_data()

    op_sqrt = P.Sqrt()

    m_slice = m.get(indices)
    v_slice = v.get(indices)

    next_m = m_slice * beta1 + values * (1 - beta1)
    next_v = v_slice * beta2 + values * values * (1 - beta2)

    lr_t = lr * op_sqrt(1 - beta2_power) / (1 - beta1_power)

    if use_nesterov:
        m_temp = beta1 * next_m + values * (1 - beta1)
        param_update = m_temp / (op_sqrt(next_v) + eps)
    else:
        param_update = next_m / (op_sqrt(next_v) + eps)

    params_need_update = params.get(indices)
    params.put(indices, params_need_update - lr_t * param_update)
    m.put(indices, next_m)
    v.put(indices, next_v)

    return success


@_lazy_adam_opt.register("Function", "Function", "Function", "Function", "Bool", "Bool", "Bool", "Tensor", "Tensor",
                         "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool")
def _run_lazy_opt_with_one_number(opt, sparse_opt, push, pull, use_locking, use_nesterov, target, beta1_power,
                                  beta2_power, beta1, beta2, eps, lr, gradient, params, moment1, moment2, ps_parameter,
                                  cache_enable):
    """Apply lazy adam optimizer to the weight parameter using Tensor."""
    success = True
    if ps_parameter and not cache_enable:
        op_shape = P.Shape()
        success = F.depend(success, pull(push((beta1_power, beta2_power, lr, beta1, beta2, eps, gradient),
                                              (op_shape(params), op_shape(moment1), op_shape(moment2))), params))
    else:
        success = F.depend(success, opt(params, moment1, moment2, beta1_power, beta2_power, lr, beta1, beta2,
                                        eps, gradient))
    return success


@_adam_opt.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Bool", "Bool")
def _update_run_op(beta1, beta2, eps, lr, weight_decay, param, m, v, gradient, decay_flag, optim_filter):
    """
    Update parameters.

    Args:
        beta1 (Tensor): The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).
        beta2 (Tensor): The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).
        eps (Tensor): Term added to the denominator to improve numerical stability. Should be greater than 0.
        lr (Tensor): Learning rate.
        weight_decay (numbers.Number): Weight decay. Should be equal to or greater than 0.
        param (Tensor): Parameters.
        m (Tensor): m value of parameters.
        v (Tensor): v value of parameters.
        gradient (Tensor): Gradient of parameters.
        decay_flag (bool): Applies weight decay or not.
        optim_filter (bool): Applies parameter update or not.

    Returns:
        Tensor, the new value of v after updating.
    """
    op_cast = P.Cast()
    if optim_filter:
        op_mul = P.Mul()
        op_square = P.Square()
        op_sqrt = P.Sqrt()
        op_cast = P.Cast()
        op_reshape = P.Reshape()
        op_shape = P.Shape()
        param_fp32 = op_cast(param, mstype.float32)
        m_fp32 = op_cast(m, mstype.float32)
        v_fp32 = op_cast(v, mstype.float32)
        gradient_fp32 = op_cast(gradient, mstype.float32)

        next_m = op_mul(beta1, m_fp32) + op_mul(op_cast(F.tuple_to_array((1.0,)), mstype.float32)
                                                - beta1, gradient_fp32)

        next_v = op_mul(beta2, v_fp32) + op_mul(op_cast(F.tuple_to_array((1.0,)), mstype.float32)
                                                - beta2, op_square(gradient_fp32))

        update = next_m / (eps + op_sqrt(next_v))
        if decay_flag:
            update = op_mul(weight_decay, param_fp32) + update

        update_with_lr = op_mul(lr, update)
        next_param = param_fp32 - op_reshape(update_with_lr, op_shape(param_fp32))

        next_param = F.depend(next_param, F.assign(param, op_cast(next_param, F.dtype(param))))
        next_param = F.depend(next_param, F.assign(m, op_cast(next_m, F.dtype(m))))
        next_param = F.depend(next_param, F.assign(v, op_cast(next_v, F.dtype(v))))

        return op_cast(next_param, F.dtype(param))
    return op_cast(gradient, F.dtype(param))


@_adam_opt.register("Function", "Function", "Function", "Function", "Bool", "Bool", "Bool", "Tensor", "Tensor",
                    "Tensor", "Tensor", "Tensor", "Tensor", "RowTensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool",
                    "Function", "Bool", "Function", "Bool")
def _run_opt_with_sparse_dist(opt, sparse_opt, push, pull, use_locking, use_nesterov, target, beta1_power,
                              beta2_power, beta1, beta2, eps, lr, gradient, param, m, v, ps_parameter, cache_enable,
                              distributed_opt, use_flag, distributed_sparse_opt, use_sparse_flag):
    """Apply sparse adam optimizer to the weight parameter when the gradient is sparse."""
    success = True
    indices = gradient.indices
    values = gradient.values
    if use_sparse_flag:
        success = F.depend(success, distributed_sparse_opt(param, m, v, beta1_power, beta2_power, lr, beta1, beta2,
                                                           eps, values, indices))
        return success
    if ps_parameter and not cache_enable:
        op_shape = P.Shape()
        shapes = (op_shape(param), op_shape(m), op_shape(v),
                  op_shape(beta1_power), op_shape(beta2_power), op_shape(lr), op_shape(beta1),
                  op_shape(beta2), op_shape(eps), op_shape(values), op_shape(indices))
        success = F.depend(success, pull(push((beta1_power, beta2_power, lr, beta1, beta2,
                                               eps, values, indices), shapes), param))
        return success

    if not target:
        success = F.depend(success, sparse_opt(param, m, v, beta1_power, beta2_power, lr, beta1, beta2,
                                               eps, values, indices))
    else:
        op_mul = P.Mul()
        op_square = P.Square()
        op_sqrt = P.Sqrt()
        scatter_add = P.ScatterAdd(use_locking)

        success = F.depend(success, F.assign(m, op_mul(beta1, m)))
        success = F.depend(success, F.assign(v, op_mul(beta2, v)))

        grad_indices = gradient.indices
        grad_value = gradient.values

        next_m = scatter_add(m,
                             grad_indices,
                             op_mul(F.tuple_to_array((1.0,)) - beta1, grad_value))

        next_v = scatter_add(v,
                             grad_indices,
                             op_mul(F.tuple_to_array((1.0,)) - beta2, op_square(grad_value)))

        if use_nesterov:
            m_temp = next_m * _scaler_ten
            F.assign(m, op_mul(beta1, next_m))
            div_value = scatter_add(m,
                                    op_mul(grad_indices, _scaler_one),
                                    op_mul(F.tuple_to_array((1.0,)) - beta1, grad_value))
            param_update = div_value / (op_sqrt(next_v) + eps)
            F.assign(m, m_temp / _scaler_ten)
        else:
            param_update = next_m / (op_sqrt(next_v) + eps)

        lr_t = lr * op_sqrt(1 - beta2_power) / (1 - beta1_power)
        next_param = param - lr_t * param_update

        success = F.depend(success, F.assign(param, next_param))
        success = F.depend(success, F.assign(m, next_m))
        success = F.depend(success, F.assign(v, next_v))

    return success


@_adam_opt.register("Function", "Function", "Function", "Function", "Bool", "Bool", "Bool", "Tensor", "Tensor",
                    "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool",
                    "Function", "Bool", "Function", "Bool")
def _run_opt_with_one_number_dist(opt, sparse_opt, push, pull, use_locking, use_nesterov, target,
                                  beta1_power, beta2_power, beta1, beta2, eps, lr, gradient, param,
                                  moment1, moment2, ps_parameter, cache_enable,
                                  distributed_opt, use_flag, distributed_sparse_opt, use_sparse_flag):
    """Apply adam optimizer to the weight parameter using Tensor."""
    success = True
    if use_flag:
        success = F.depend(success, distributed_opt(param, moment1, moment2, beta1_power, beta2_power, lr, beta1, beta2,
                                                    eps, gradient))
    elif ps_parameter and not cache_enable:
        op_shape = P.Shape()
        success = F.depend(success, pull(push((beta1_power, beta2_power, lr, beta1, beta2, eps, gradient),
                                              (op_shape(param), op_shape(moment1), op_shape(moment2))), param))
    else:
        success = F.depend(success, opt(param, moment1, moment2, beta1_power, beta2_power, lr, beta1, beta2,
                                        eps, gradient))
    return success


@_adam_opt.register("Function", "Function", "Function", "Function",
                    "Bool", "Bool", "Bool",
                    "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                    "RowTensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool")
def _run_opt_with_sparse(opt, sparse_opt, push, pull,
                         use_locking, use_nesterov, target,
                         beta1_power, beta2_power, beta1, beta2, eps, lr,
                         gradient, param, m, v, ps_parameter, cache_enable):
    """Apply sparse adam optimizer to the weight parameter when the gradient is sparse."""
    success = True
    indices = gradient.indices
    values = gradient.values
    if ps_parameter and not cache_enable:
        op_shape = P.Shape()
        shapes = (op_shape(param), op_shape(m), op_shape(v),
                  op_shape(beta1_power), op_shape(beta2_power), op_shape(lr), op_shape(beta1),
                  op_shape(beta2), op_shape(eps), op_shape(values), op_shape(indices))
        success = F.depend(success, pull(push((beta1_power, beta2_power, lr, beta1, beta2,
                                               eps, values, indices), shapes), param))
        return success

    if not target:
        success = F.depend(success, sparse_opt(param, m, v, beta1_power, beta2_power, lr, beta1, beta2,
                                               eps, values, indices))
    else:
        op_mul = P.Mul()
        op_square = P.Square()
        op_sqrt = P.Sqrt()
        scatter_add = P.ScatterAdd(use_locking)

        success = F.depend(success, F.assign(m, op_mul(beta1, m)))
        success = F.depend(success, F.assign(v, op_mul(beta2, v)))

        grad_indices = gradient.indices
        grad_value = gradient.values

        next_m = scatter_add(m,
                             grad_indices,
                             op_mul(F.tuple_to_array((1.0,)) - beta1, grad_value))

        next_v = scatter_add(v,
                             grad_indices,
                             op_mul(F.tuple_to_array((1.0,)) - beta2, op_square(grad_value)))

        if use_nesterov:
            m_temp = next_m * _scaler_ten
            F.assign(m, op_mul(beta1, next_m))
            div_value = scatter_add(m,
                                    op_mul(grad_indices, _scaler_one),
                                    op_mul(F.tuple_to_array((1.0,)) - beta1, grad_value))
            param_update = div_value / (op_sqrt(next_v) + eps)
            F.assign(m, m_temp / _scaler_ten)
        else:
            param_update = next_m / (op_sqrt(next_v) + eps)

        lr_t = lr * op_sqrt(1 - beta2_power) / (1 - beta1_power)
        next_param = param - lr_t * param_update

        success = F.depend(success, F.assign(param, next_param))
        success = F.depend(success, F.assign(m, next_m))
        success = F.depend(success, F.assign(v, next_v))

    return success


@_adam_opt.register("Function", "Function", "Function", "Function",
                    "Bool", "Bool", "Bool",
                    "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool")
def _run_opt_with_one_number(opt, sparse_opt, push, pull,
                             use_locking, use_nesterov, target,
                             beta1_power, beta2_power, beta1, beta2, eps, lr,
                             gradient, param, moment1, moment2, ps_parameter, cache_enable):
    """Apply adam optimizer to the weight parameter using Tensor."""
    success = True
    if ps_parameter and not cache_enable:
        op_shape = P.Shape()
        success = F.depend(success, pull(push((beta1_power, beta2_power, lr, beta1, beta2, eps, gradient),
                                              (op_shape(param), op_shape(moment1), op_shape(moment2))), param))
    else:
        success = F.depend(success, opt(param, moment1, moment2, beta1_power, beta2_power, lr, beta1, beta2,
                                        eps, gradient))
    return success


@_adam_opt.register("Function", "Function", "Function", "Function",
                    "Bool", "Bool", "Bool",
                    "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool")
def _run_opt_with_one_number_use_amsgrad(opt, sparse_opt, push, pull,
                                         use_locking, use_nesterov, target,
                                         beta1_power, beta2_power, beta1, beta2, eps, lr,
                                         gradient, param, moment1, moment2, vhat, ps_parameter, cache_enable):
    """Apply adam optimizer to the weight parameter using Tensor and use amsgrad."""
    success = True
    if ps_parameter and not cache_enable:
        op_shape = P.Shape()
        success = F.depend(success, pull(push((beta1_power, beta2_power, lr, gradient),
                                              (op_shape(param), op_shape(moment1), op_shape(moment2),
                                               op_shape(vhat))), param))
    else:
        success = F.depend(success, opt(param, moment1, moment2, vhat, beta1_power, beta2_power, lr, gradient))
    return success


@_adam_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Tensor")
def _run_off_load_opt(opt, beta1_power, beta2_power, beta1, beta2, eps, lr, gradient, param, moment1, moment2):
    """Apply AdamOffload optimizer to the weight parameter using Tensor."""
    success = True
    delat_param = opt(moment1, moment2, beta1_power, beta2_power, lr, beta1, beta2, eps, gradient)
    success = F.depend(success, F.assign_add(param, delat_param))
    return success


@_fused_adam_weight_decay.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                                   "Tensor", "Tensor", "Bool", "Bool")
def _run_fused_adam_weight_decay_opt(opt, beta1, beta2, eps, lr, weight_decay, param, moment1, moment2, gradient,
                                     decay_flags, optim_filter):
    """Apply FusedAdamWeightDecay optimizer to the weight parameter using Tensor."""
    if optim_filter:
        if decay_flags:
            opt(param, moment1, moment2, lr, beta1, beta2, eps, weight_decay, P.Cast()(gradient, F.dtype(param)))
        else:
            opt(param, moment1, moment2, lr, beta1, beta2, eps, 0.0, P.Cast()(gradient, F.dtype(param)))
    return True


def _check_param_value(beta1, beta2, eps, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_float_range(beta1, 0.0, 1.0, Rel.INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(beta2, 0.0, 1.0, Rel.INC_NEITHER, "beta2", prim_name)
    validator.check_positive_float(eps, "eps", prim_name)


class Adam(Optimizer):
    r"""
    Implements the Adaptive Moment Estimation (Adam) algorithm.

    The Adam optimizer can dynamically adjust the learning rate of each parameter using the first-order
    moment estimation and the second-order moment estimation of the gradient.
    The Adam algorithm is proposed in `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_.

    The updating formulas are as follows:

    .. math::
        \begin{array}{l}
            &\newline
            &\hline \\
            &\textbf{Parameters}: \: 1^{\text {st }}\text {moment vector} \: m , \: 2^{\text {nd}} \:
             \text{moment vector} \: v , \\
            &\:\text{gradients } g, \: \text{learning rate} \: \gamma, \text
             { exponential decay rates for the moment estimates} \: \beta_{1} \: \beta_{2} , \\
            &\:\text {parameter vector} \: w_{0}, \:\text{timestep} \: t , \text{ weight decay } \lambda \\
            &\textbf{Init}: m_{0} \leftarrow 0, \: v_{0} \leftarrow 0, \: t \leftarrow 0, \:
             \text{init parameter vector} \: w_{0} \\[-1.ex]
            &\newline
            &\hline \\
            &\textbf{while} \: w_{t} \: \text{not converged} \: \textbf{do} \\
            &\hspace{5mm}\boldsymbol{g}_{t} \leftarrow \nabla_{w} \boldsymbol{f}_{t}\left(\boldsymbol{w}_{t-1}\right) \\
            &\hspace{5mm}\textbf {if } \lambda \neq 0 \\
            &\hspace{10mm}\boldsymbol{g}_{t} \leftarrow \boldsymbol{g}_{t}+\lambda \boldsymbol{w}_{t-1} \\
            &\hspace{5mm}\boldsymbol{m}_{t} \leftarrow \beta_{1} \boldsymbol{m}_{t-1}+\left(1-\beta_{1}\right)
             \boldsymbol{g}_{t} \\
            &\hspace{5mm}\boldsymbol{v}_{t} \leftarrow \beta_{2} \boldsymbol{v}_{t-1}+\left(1-\beta_{2}\right)
             \boldsymbol{g}_{t}^{2} \\
            &\hspace{5mm}\hat{\boldsymbol{m}}_{t} \leftarrow \boldsymbol{m}_{t} /\left(1-\beta_{1}^{t}\right) \\
            &\hspace{5mm}\hat{\boldsymbol{v}}_{t} \leftarrow \boldsymbol{v}_{t} /\left(1-\beta_{2}^{t}\right) \\
            &\hspace{5mm}\boldsymbol{w}_{t} \leftarrow \boldsymbol{w}_{t-1}-\gamma \hat{\boldsymbol{m}}_{t}
             /(\sqrt{\hat{\boldsymbol{v}}_{t}}+\epsilon) \\
            &\textbf{end while} \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
            &\textbf{return} \:  \boldsymbol{w}_{t} \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
        \end{array}

    :math:`m` represents the 1st moment vector, :math:`v` represents the 2nd moment vector,
    :math:`g` represents `gradients`, :math:`\beta_1, \beta_2` represent `beta1` and `beta2`,
    :math:`t` represents the current step while :math:`beta_1^t` and :math:`beta_2^t` represent
    `beta1_power` and `beta2_power`, :math:`\gamma` represents `learning_rate`, :math:`w` represents `params`,
    :math:`\epsilon` represents `eps`.

    Note:
        On Ascend, when `use_amsgrad` is set to True, it might have slightly larger accuracy error.

        The sparse strategy is applied while the SparseGatherV2 operator is used for forward network. If the sparse
        strategy wants to be executed on the host, set the target to the CPU.
        The sparse feature is under continuous development.

        If parameters are not grouped, the `weight_decay` in optimizer will be applied on the network parameters without
        'beta' or 'gamma' in their names. Users can group parameters to change the strategy of decaying weight. When
        parameters are grouped, each group can set `weight_decay`. If not, the `weight_decay` in optimizer will be
        applied.

        When using Adam with use_lazy=True:

        Please note, the optimizer only updates the current index position of the network parameters
        when the gradient is sparse. The sparse behavior is not equivalent to the original Adam algorithm.
        If you want to execute a sparse policy, target needs to be set to CPU.

        When using Adam with use_offload=True:

        This optimizer only supports `GRAPH_MODE`.

    Args:
        params (Union[list[Parameter], list[dict]]): Must be list of `Parameter` or list of `dict`. When the
            `params` is a list of `dict`, the string "params", "lr", "weight_decay", "grad_centralization" and
            "order_params" are the keys can be parsed.

            - params: Required. Parameters in current group. The value must be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in optimizer will be used. Fixed and dynamic learning rate are supported.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the optimizer will be used. It should be noted that weight
              decay can be a constant value or a Cell. It is a Cell only when dynamic weight decay is applied. Dynamic
              weight decay is similar to dynamic learning rate, users need to customize a weight decay schedule only
              with global step as input, and during training, the optimizer calls the instance of WeightDecaySchedule
              to get the weight decay value of current step.

            - grad_centralization: Optional. Must be Boolean. If "grad_centralization" is in the keys, the set value
              will be used. If not, the `grad_centralization` is False by default. This configuration only works on the
              convolution layer.

            - order_params: Optional. When parameters is grouped, this usually is used to maintain the order of
              parameters that appeared in the network to improve performance. The value should be parameters whose
              order will be followed in optimizer.
              If `order_params` in the keys, other keys will be ignored and the element of 'order_params' must be in
              one group of `params`.

        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule]): Default: 1e-3.

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        beta1 (float): The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).
                       Default: 0.9.
        beta2 (float): The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).
                       Default: 0.999.
        eps (float): Term added to the denominator to improve numerical stability. Should be greater than 0. Default:
                     1e-8.
        use_locking (bool): Whether to enable a lock to protect the updating process of variable tensors.
            If true, updates of the `w`, `m`, and `v` tensors will be protected by a lock.
            If false, the result is unpredictable. Default: False.
        use_nesterov (bool): Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients.
            If true, update the gradients using NAG.
            If false, update the gradients without using NAG. Default: False.
        use_amsgrad (bool): Whether to use Amsgrad algorithm to update the gradients.
            If true, update the gradients using Amsgrad.
            If false, update the gradients without using Amsgrad. Default: False.

        weight_decay (Union[float, int, Cell]): Weight decay (L2 penalty). Default: 0.0.

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

        loss_scale (float): A floating point value for the loss scale. Should be greater than 0. In general, use the
            default value. Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to False, then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.amp.FixedLossScaleManager` for more details.
            Default: 1.0.

        kwargs:

            - use_lazy (bool): Whether to use Lazy Adam algorithm. Default: False.
              If true, apply lazy adam algorithm.
              If false, apply normal adam algorithm.

            - use_offload (bool): Whether to offload adam optimizer to host CPU and keep parameters being updated on
              the device in order to minimize the memory cost. Default: False.
              If true, apply offload adam.
              If false, apply normal adam.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        Tensor[bool], the value is True.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `beta1`, `beta2`, `eps` or `loss_scale` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        TypeError: If `use_locking`, `use_nesterov`, `use_amsgrad`, `use_lazy` or `use_offload` is not a bool.
        ValueError: If `loss_scale` or `eps` is less than or equal to 0.
        ValueError: If `beta1`, `beta2` is not in range (0.0, 1.0).
        ValueError: If `weight_decay` is less than 0.
        ValueError: If `use_lazy` and `use_offload` are both true.
        ValueError: If `use_amsgrad` is true and (`use_lazy` or `use_offload` is true).
        ValueError: If `use_amsgrad` while using distributed training.

    Supported Platforms:
        ``Ascend`` ``GPU``  ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>>
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.Adam(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = nn.Adam(group_params, learning_rate=0.1, weight_decay=0.0, use_lazy=False, use_offload=False)
        >>> # The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01 and grad
        >>> # centralization of True.
        >>> # The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0 and grad
        >>> # centralization of False.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = ms.Model(net, loss_fn=loss, optimizer=optim)
    """

    @opt_init_args_register
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, use_locking=False,
                 use_nesterov=False, weight_decay=0.0, loss_scale=1.0, use_amsgrad=False, **kwargs):
        super(Adam, self).__init__(learning_rate, params, weight_decay, loss_scale)
        use_lazy = kwargs.get('use_lazy', False)
        use_offload = kwargs.get('use_offload', False)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        validator.check_value_type("use_locking", use_locking, [bool], self.cls_name)
        validator.check_value_type("use_nesterov", use_nesterov, [bool], self.cls_name)
        validator.check_value_type("use_amsgrad", use_amsgrad, [bool], self.cls_name)
        validator.check_value_type("use_lazy", use_lazy, [bool], self.cls_name)
        validator.check_value_type("use_offload", use_offload, [bool], self.cls_name)

        if use_lazy and use_offload:
            raise ValueError(f"For 'Adam', 'use_lazy' and 'use_offload' can not both be True."
                             f"But got use_lazy={use_lazy}, use_offload={use_offload}.")

        if use_amsgrad and (use_lazy or use_offload):
            raise ValueError(f"For lazy Adam and Adam with offload, there is no parameter named 'use_amsgrad'."
                             f"but got 'use_amsgrad'={use_amsgrad}.")

        self.beta1 = Tensor(beta1, mstype.float32)
        self.beta2 = Tensor(beta2, mstype.float32)
        self.beta1_power = Parameter(initializer(1, (), mstype.float32), name="beta1_power")
        self.beta2_power = Parameter(initializer(1, (), mstype.float32), name="beta2_power")
        self.eps = Tensor(eps, mstype.float32)
        self.use_nesterov = use_nesterov
        self.use_locking = use_locking
        self.use_amsgrad = use_amsgrad
        self.use_lazy = use_lazy
        self.use_offload = use_offload
        self.moment1 = self._parameters.clone(prefix="moment1", init='zeros')
        self.moment2 = self._parameters.clone(prefix="moment2", init='zeros')
        if use_amsgrad:
            self.vhat = self._parameters.clone(prefix="vhat", init='zeros')

        if use_offload:
            self.opt = P.AdamNoUpdateParam(use_locking, use_nesterov)
            self.opt.set_device("CPU")

        elif use_lazy:
            self._is_device = True
            self.opt = P.Adam(use_locking, use_nesterov)
            self.sparse_opt = P.FusedSparseLazyAdam(use_locking, use_nesterov)
            self.sparse_opt.set_device("CPU")
            self._ps_pull = P.Pull()
            self._ps_push = P.Push("Adam", [0, 1, 2])
            self._ps_push.add_prim_attr("use_nesterov", use_nesterov)
            self._init_distributed_opts(use_locking, use_nesterov)

        else:
            self._is_device = True
            if use_amsgrad:
                self.opt = P.ApplyAdamWithAmsgrad(beta1, beta2, eps, use_locking)
            else:
                self.opt = P.Adam(use_locking, use_nesterov)
            self.sparse_opt = P.FusedSparseAdam(use_locking, use_nesterov)
            self.sparse_opt.set_device("CPU")
            self._ps_pull = P.Pull()
            if use_amsgrad:
                self._ps_push = P.Push("ApplyAdamWithAmsgrad", [0, 1, 2, 3])
            else:
                self._ps_push = P.Push("Adam", [0, 1, 2])
                self._ps_push.add_prim_attr("use_nesterov", use_nesterov)

            self._init_distributed_opts(use_locking, use_nesterov)

    def _apply_adam(self, params, beta1_power, beta2_power, moment1, moment2, lr, gradients):
        """Execute Adam optimizer and its variants."""
        if self.use_offload:
            if self.is_group_lr:
                success = self.map_reverse(F.partial(_adam_opt, self.opt, beta1_power, beta2_power, self.beta1,
                                                     self.beta2, self.eps), lr, gradients, params, moment1, moment2)
            else:
                success = self.map_reverse(F.partial(_adam_opt, self.opt, beta1_power, beta2_power, self.beta1,
                                                     self.beta2, self.eps, lr), gradients, params, moment1, moment2)
        # Lazy adam or normal adam
        else:
            if self.use_dist_optimizer:
                if self.use_dist_optimizer and self.use_amsgrad:
                    raise ValueError(f"Adam with amsgrad is currently not supporting distributed training!"
                                     f"Please set use_amsgrad=False for distributed training.")
                if self.is_group_lr:
                    if self.use_lazy:
                        success = self.map_reverse(F.partial(_lazy_adam_opt, self.opt, self.sparse_opt,
                                                             self._ps_push, self._ps_pull, self.use_locking,
                                                             self.use_nesterov,
                                                             self._is_device, beta1_power, beta2_power,
                                                             self.beta1, self.beta2, self.eps),
                                                   lr, gradients, self._parameters, self.moment1, self.moment2,
                                                   self.ps_parameters, self.cache_enable, self.dense_lazyadam_opts,
                                                   self.use_dense_opt_flags, self.sparse_lazyadam_opts,
                                                   self.use_sparse_opt_flags)
                    # Normal Adam
                    else:
                        success = self.map_(F.partial(_adam_opt, self.opt, self.sparse_opt, self._ps_push,
                                                      self._ps_pull, self.use_locking, self.use_nesterov,
                                                      self._is_device, beta1_power, beta2_power, self.beta1, self.beta2,
                                                      self.eps), lr, gradients, params, moment1, moment2,
                                            self.ps_parameters, self.cache_enable, self.dense_adam_opts,
                                            self.use_dense_opt_flags, self.sparse_adam_opts, self.use_sparse_opt_flags)
                else:
                    if self.use_lazy:
                        success = self.map_reverse(F.partial(_lazy_adam_opt, self.opt, self.sparse_opt, self._ps_push,
                                                             self._ps_pull, self.use_locking, self.use_nesterov,
                                                             self._is_device, beta1_power, beta2_power, self.beta1,
                                                             self.beta2, self.eps, lr), gradients, self._parameters,
                                                   self.moment1, self.moment2, self.ps_parameters, self.cache_enable,
                                                   self.dense_lazyadam_opts, self.use_dense_opt_flags,
                                                   self.sparse_lazyadam_opts, self.use_sparse_opt_flags)
                    else:
                        success = self.map_(F.partial(_adam_opt, self.opt, self.sparse_opt, self._ps_push,
                                                      self._ps_pull, self.use_locking, self.use_nesterov,
                                                      self._is_device, beta1_power, beta2_power, self.beta1, self.beta2,
                                                      self.eps, lr), gradients, params, moment1, moment2,
                                            self.ps_parameters, self.cache_enable, self.dense_adam_opts,
                                            self.use_dense_opt_flags, self.sparse_adam_opts, self.use_sparse_opt_flags)
            else:
                if self.is_group_lr:
                    if self.use_lazy:
                        success = self.map_(F.partial(_lazy_adam_opt, self.opt, self.sparse_opt, self._ps_push,
                                                      self._ps_pull, self.use_locking, self.use_nesterov,
                                                      self._is_device, beta1_power, beta2_power, self.beta1, self.beta2,
                                                      self.eps), lr, gradients, params, moment1, moment2,
                                            self.ps_parameters, self.cache_enable)
                    else:
                        if self.use_amsgrad:
                            success = self.map_(F.partial(_adam_opt, self.opt, self.sparse_opt, self._ps_push,
                                                          self._ps_pull, self.use_locking, self.use_nesterov,
                                                          self._is_device, beta1_power, beta2_power,
                                                          self.beta1, self.beta2, self.eps), lr, gradients, params,
                                                moment1, moment2, self.vhat, self.ps_parameters, self.cache_enable)
                        else:
                            success = self.map_(F.partial(_adam_opt, self.opt, self.sparse_opt, self._ps_push,
                                                          self._ps_pull, self.use_locking, self.use_nesterov,
                                                          self._is_device, beta1_power, beta2_power,
                                                          self.beta1, self.beta2, self.eps), lr, gradients, params,
                                                moment1, moment2, self.ps_parameters, self.cache_enable)
                else:
                    if self.use_lazy:
                        success = self.map_(F.partial(_lazy_adam_opt, self.opt, self.sparse_opt, self._ps_push,
                                                      self._ps_pull, self.use_locking, self.use_nesterov,
                                                      self._is_device, beta1_power, beta2_power, self.beta1, self.beta2,
                                                      self.eps, lr), gradients, params, moment1, moment2,
                                            self.ps_parameters, self.cache_enable)
                    else:
                        if self.use_amsgrad:
                            success = self.map_(F.partial(_adam_opt, self.opt, self.sparse_opt, self._ps_push,
                                                          self._ps_pull, self.use_locking, self.use_nesterov,
                                                          self._is_device, beta1_power, beta2_power,
                                                          self.beta1, self.beta2, self.eps, lr), gradients, params,
                                                moment1, moment2, self.vhat, self.ps_parameters, self.cache_enable)
                        else:
                            success = self.map_(F.partial(_adam_opt, self.opt, self.sparse_opt, self._ps_push,
                                                          self._ps_pull, self.use_locking, self.use_nesterov,
                                                          self._is_device, beta1_power, beta2_power,
                                                          self.beta1, self.beta2, self.eps, lr), gradients, params,
                                                moment1, moment2, self.ps_parameters, self.cache_enable)

        return success

    @jit
    def construct(self, gradients):
        params = self._parameters
        moment1 = self.moment1
        moment2 = self.moment2
        gradients = self.flatten_gradients(gradients)
        gradients = self.decay_weight(gradients)
        if not self.use_offload:
            gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)
        gradients = self._grad_sparse_indices_deduplicate(gradients)
        lr = self.get_lr()

        beta1_power = self.beta1_power * self.beta1
        self.beta1_power = beta1_power
        beta2_power = self.beta2_power * self.beta2
        self.beta2_power = beta2_power

        return self._apply_adam(params, beta1_power, beta2_power, moment1, moment2, lr, gradients)

    @Optimizer.target.setter
    def target(self, value):
        """
        If the input value is set to "CPU", the parameters will be updated on the host using the Fused
        optimizer operation.
        """
        self._set_base_target(value)

    def _init_distributed_opts(self, use_locking, use_nesterov):
        self.use_dist_optimizer = self._use_distibuted_optimizer()
        self.dense_adam_opts, self.use_dense_opt_flags = \
            self._get_distributed_optimizer_list("adam", use_locking, use_nesterov)
        self.sparse_adam_opts, self.use_sparse_opt_flags = \
            self._get_distributed_optimizer_list("fused_sparse_adam", use_locking, use_nesterov)


class AdamWeightDecay(Optimizer):
    r"""
    Implements the Adam algorithm with weight decay.

    .. math::
        \begin{array}{l}
            &\newline
            &\hline \\
            &\textbf{Parameters}: \: 1^{\text {st }}\text {moment vector} \: m , \: 2^{\text {nd}} \:
             \text{moment vector} \: v , \\
            &\: gradients \: g, \: \text{learning rate} \: \gamma,
             \text {exponential decay rates for the moment estimates} \: \beta_{1} \: \beta_{2} , \\
            &\:\text {parameter vector} \: w_{0}, \:\text{timestep} \: t, \: \text{weight decay} \: \lambda \\
            &\textbf{Init}:  m_{0} \leftarrow 0, \: v_{0} \leftarrow 0, \: t \leftarrow 0, \:
             \text{init parameter vector} \: w_{0} \\[-1.ex]
            &\newline
            &\hline \\
            &\textbf{repeat} \\
            &\hspace{5mm} t \leftarrow t+1 \\
            &\hspace{5mm}\boldsymbol{g}_{t} \leftarrow \nabla f_{t}\left(\boldsymbol{w}_{t-1}\right) \\
            &\hspace{5mm}\boldsymbol{m}_{t} \leftarrow \beta_{1} \boldsymbol{m}_{t-1}+\left(1-\beta_{1}\right)
             \boldsymbol{g}_{t} \\
            &\hspace{5mm}\boldsymbol{v}_{t} \leftarrow \beta_{2} \boldsymbol{v}_{t-1}+\left(1-\beta_{2}\right)
             \boldsymbol{g}_{t}^{2} \\
            &\hspace{5mm}\boldsymbol{w}_{t} \leftarrow \boldsymbol{w}_{t-1}-\left(\gamma \hat{\boldsymbol{m}}_{t}
             /\left(\sqrt{\hat{\boldsymbol{v}}_{t}}+\epsilon\right)+\lambda \boldsymbol{w}_{t-1}\right) \\
            &\textbf{until}\text { stopping criterion is met } \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
            &\textbf{return} \: \boldsymbol{w}_{t} \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
        \end{array}

    :math:`m` represents the 1st moment vector `moment1`, :math:`v` represents the 2nd moment vector `moment2`,
    :math:`g` represents `gradients`, :math:`\gamma` represents `learning_rate`,
    :math:`\beta_1, \beta_2` represent `beta1` and `beta2`, :math:`t` represents the current step,
    :math:`w` represents `params`, :math:`\lambda` represents `weight_decay`.

    Note:
        There is usually no connection between a optimizer and mixed precision. But when `FixedLossScaleManager` is used
        and `drop_overflow_update` in `FixedLossScaleManager` is set to False, optimizer needs to set the 'loss_scale'.
        As this optimizer has no argument of `loss_scale`, so `loss_scale` needs to be processed by other means, refer
        document `LossScale <https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html>`_ to
        process `loss_scale` correctly.

        If parameters are not grouped, the `weight_decay` in optimizer will be applied on the network parameters without
        'beta' or 'gamma' in their names. Users can group parameters to change the strategy of decaying weight. When
        parameters are grouped, each group can set `weight_decay`. If not, the `weight_decay` in optimizer will be
        applied.

    Args:
        params (Union[list[Parameter], list[dict]]): Must be list of `Parameter` or list of `dict`. When the
            `params` is a list of `dict`, the string "params", "lr", "weight_decay", and "order_params"
            are the keys can be parsed.

            - params: Required. Parameters in current group. The value must be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in optimizer will be used. Fixed and dynamic learning rate are supported.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the optimizer will be used. It should be noted that weight
              decay can be a constant value or a Cell. It is a Cell only when dynamic weight decay is applied. Dynamic
              weight decay is similar to dynamic learning rate, users need to customize a weight decay schedule only
              with global step as input, and during training, the optimizer calls the instance of WeightDecaySchedule
              to get the weight decay value of current step.

            - order_params: Optional. When parameters is grouped, this usually is used to maintain the order of
              parameters that appeared in the network to improve performance. The value should be parameters whose
              order will be followed in optimizer.
              If `order_params` in the keys, other keys will be ignored and the element of 'order_params' must be in
              one group of `params`.

        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule]): Default: 1e-3.

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        beta1 (float): The exponential decay rate for the 1st moment estimations. Default: 0.9.
            Should be in range (0.0, 1.0).
        beta2 (float): The exponential decay rate for the 2nd moment estimations. Default: 0.999.
            Should be in range (0.0, 1.0).
        eps (float): Term added to the denominator to improve numerical stability. Default: 1e-6.
            Should be greater than 0.

        weight_decay (Union[float, int, Cell]): Weight decay (L2 penalty). Default: 0.0.

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[bool], all elements are True.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `beta1`, `beta2` or `eps` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        ValueError: If `eps` is less than or equal to 0.
        ValueError: If `beta1`, `beta2` is not in range (0.0, 1.0).
        ValueError: If `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>>
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.AdamWeightDecay(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = nn.AdamWeightDecay(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01.
        >>> # The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = ms.Model(net, loss_fn=loss, optimizer=optim)
   """
    _support_parallel_optimizer = True

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(AdamWeightDecay, self).__init__(learning_rate, params, weight_decay)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self._parameters.clone(prefix="adam_m", init='zeros')
        self.moments2 = self._parameters.clone(prefix="adam_v", init='zeros')
        self.fused_opt = P.AdamWeightDecay()
        if context.get_context("device_target") == "CPU":
            self.use_fused_opt = True
        else:
            self.use_fused_opt = False

    @jit
    def construct(self, gradients):
        gradients = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()

        if self.use_fused_opt:
            if self.is_group:
                if self.is_group_lr:
                    optim_result = self.hyper_map(
                        F.partial(_fused_adam_weight_decay, self.fused_opt, self.beta1, self.beta2, self.eps),
                        lr, weight_decay, self._parameters, self.moments1,
                        self.moments2, gradients, self.decay_flags, self.optim_filter)
                else:
                    optim_result = self.hyper_map(
                        F.partial(_fused_adam_weight_decay, self.fused_opt, self.beta1, self.beta2, self.eps, lr),
                        weight_decay, self._parameters, self.moments1, self.moments2,
                        gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.hyper_map(
                    F.partial(_fused_adam_weight_decay, self.fused_opt, self.beta1, self.beta2, self.eps, lr,
                              weight_decay),
                    self._parameters, self.moments1, self.moments2,
                    gradients, self.decay_flags, self.optim_filter)
        else:
            if self.is_group:
                if self.is_group_lr:
                    optim_result = self.hyper_map(F.partial(_adam_opt, self.beta1, self.beta2, self.eps),
                                                  lr, weight_decay, self._parameters, self.moments1,
                                                  self.moments2, gradients, self.decay_flags, self.optim_filter)
                else:
                    optim_result = self.hyper_map(F.partial(_adam_opt, self.beta1, self.beta2, self.eps, lr),
                                                  weight_decay, self._parameters, self.moments1, self.moments2,
                                                  gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.hyper_map(F.partial(_adam_opt, self.beta1, self.beta2, self.eps, lr, weight_decay),
                                              self._parameters, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
        if self.use_parallel:
            self.broadcast_params(optim_result)

        return optim_result

    @Optimizer.target.setter
    def target(self, value):
        """
        If the input value is set to "CPU", the parameters will be updated on the host using the Fused
        optimizer operation.
        """
        self._set_base_target(value)
        if value == 'CPU':
            self.fused_opt.set_device("CPU")
            self.use_fused_opt = True
        else:
            self.use_fused_opt = False


class AdamOffload(Optimizer):
    r"""
    This optimizer will offload Adam optimizer to host CPU and keep parameters being updated on the device,
    to minimize the memory cost. Although that would bring about an increase of performance overhead,
    the optimizer could be used to run a larger model.

    The Adam algorithm is proposed in `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_.

    The updating formulas are as follows,

    .. math::
        \begin{array}{ll} \\
            m_{t+1} = \beta_1 * m_{t} + (1 - \beta_1) * g \\
            v_{t+1} = \beta_2 * v_{t} + (1 - \beta_2) * g * g \\
            l = \alpha * \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \\
            w_{t+1} = w_{t} - l * \frac{m_{t+1}}{\sqrt{v_{t+1}} + \epsilon}
        \end{array}

    :math:`m` represents the 1st moment vector `moment1`, :math:`v` represents the 2nd moment vector `moment2`,
    :math:`g` represents `gradients`, :math:`l` represents scaling factor, :math:`\beta_1, \beta_2` represent
    `beta1` and `beta2`, :math:`t` represents the current step while :math:`beta_1^t` and :math:`beta_2^t` represent
    `beta1_power` and `beta2_power`, :math:`\alpha` represents `learning_rate`, :math:`w` represents `params`,
    :math:`\epsilon` represents `eps`.

    Note:
        This optimizer only supports `GRAPH_MODE` currently.

        If parameters are not grouped, the `weight_decay` in optimizer will be applied on the network parameters without
        'beta' or 'gamma' in their names. Users can group parameters to change the strategy of decaying weight. When
        parameters are grouped, each group can set `weight_decay`. If not, the `weight_decay` in optimizer will be
        applied.

    Args:
        params (Union[list[Parameter], list[dict]]): Must be list of `Parameter` or list of `dict`. When the
            `params` is a list of `dict`, the string "params", "lr", "weight_decay", and "order_params"
            are the keys can be parsed.

            - params: Required. Parameters in current group. The value must be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in optimizer will be used. Fixed and dynamic learning rate are supported.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the optimizer will be used. It should be noted that weight
              decay can be a constant value or a Cell. It is a Cell only when dynamic weight decay is applied. Dynamic
              weight decay is similar to dynamic learning rate, users need to customize a weight decay schedule only
              with global step as input, and during training, the optimizer calls the instance of WeightDecaySchedule
              to get the weight decay value of current step.

            - order_params: Optional. When parameters is grouped, this usually is used to maintain the order of
              parameters that appeared in the network to improve performance. The value should be parameters whose
              order will be followed in optimizer.
              If `order_params` in the keys, other keys will be ignored and the element of 'order_params' must be in
              one group of `params`.

        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule]): Default: 1e-3.

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        beta1 (float): The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).
                       Default: 0.9.
        beta2 (float): The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).
                       Default: 0.999.
        eps (float): Term added to the denominator to improve numerical stability. Should be greater than 0. Default:
                     1e-8.
        use_locking (bool): Whether to enable a lock to protect the updating process of variable tensors.
            If true, updates of the `w`, `m`, and `v` tensors will be protected by a lock.
            If false, the result is unpredictable. Default: False.
        use_nesterov (bool): Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients.
            If true, update the gradients using NAG.
            If false, update the gradients without using NAG. Default: False.

        weight_decay (Union[float, int, Cell]): Weight decay (L2 penalty). Default: 0.0.

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

        loss_scale (float): A floating point value for the loss scale. Should be greater than 0. In general, use the
            default value. Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to False, then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.amp.FixedLossScaleManager` for more details.
            Default: 1.0.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        Tensor[bool], the value is True.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `beta1`, `beta2`, `eps` or `loss_scale` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        TypeError: If `use_locking` or `use_nesterov` is not a bool.
        ValueError: If `loss_scale` or `eps` is less than or equal to 0.
        ValueError: If `beta1`, `beta2` is not in range (0.0, 1.0).
        ValueError: If `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>>
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.AdamOffload(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = nn.AdamOffload(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01.
        >>> # The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = ms.Model(net, loss_fn=loss, optimizer=optim)
    """

    @deprecated("2.0", "Adam", False)
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, use_locking=False,
                 use_nesterov=False, weight_decay=0.0, loss_scale=1.0):
        super(AdamOffload, self).__init__(learning_rate, params, weight_decay, loss_scale)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        validator.check_value_type("use_locking", use_locking, [bool], self.cls_name)
        validator.check_value_type("use_nesterov", use_nesterov, [bool], self.cls_name)

        self.params = self.parameters
        self.beta1 = Tensor(beta1, mstype.float32)
        self.beta2 = Tensor(beta2, mstype.float32)
        self.beta1_power = Parameter(initializer(1, [1], mstype.float32), name="beta1_power")
        self.beta2_power = Parameter(initializer(1, [1], mstype.float32), name="beta2_power")
        self.eps = Tensor(eps, mstype.float32)
        self.moment1 = self._parameters.clone(prefix="moment1", init='zeros')
        self.moment2 = self._parameters.clone(prefix="moment2", init='zeros')
        self.opt = P.AdamNoUpdateParam(use_locking, use_nesterov)
        self.opt.set_device("CPU")

    @jit
    def construct(self, gradients):
        params = self._parameters
        moment1 = self.moment1
        moment2 = self.moment2
        gradients = self.flatten_gradients(gradients)
        gradients = self.decay_weight(gradients)
        gradients = self.scale_grad(gradients)
        lr = self.get_lr()

        beta1_power = self.beta1_power * self.beta1
        self.beta1_power = beta1_power
        beta2_power = self.beta2_power * self.beta2
        self.beta2_power = beta2_power
        if self.is_group_lr:
            success = self.map_reverse(F.partial(_adam_opt, self.opt,
                                                 beta1_power, beta2_power, self.beta1, self.beta2, self.eps),
                                       lr, gradients, params, moment1, moment2)
        else:
            success = self.map_reverse(F.partial(_adam_opt, self.opt,
                                                 beta1_power, beta2_power, self.beta1, self.beta2, self.eps, lr),
                                       gradients, params, moment1, moment2)
        return success


def create_distributed_adam(*args, **kwargs):
    """
    Create the distributed Adam op.
    """
    adam = P.Adam(*args, **kwargs)
    adam.add_prim_attr("gradient_type", "dense_gradient")
    adam.add_prim_attr("parameter_input_index", 0)
    adam.add_prim_attr("gradient_input_index", 9)
    return adam


def create_distributed_fused_sparse_adam(*args, **kwargs):
    """
    Create the distributed FusedSparseAdam op.
    """
    sparse_adam = P.FusedSparseAdam(*args, **kwargs)
    sparse_adam.add_prim_attr("gradient_type", "sparse_gradient")
    sparse_adam.add_prim_attr("parameter_input_index", 0)
    sparse_adam.add_prim_attr("gradient_input_index", 9)
    sparse_adam.add_prim_attr("indices_input_index", 10)
    return sparse_adam


_register_dist_optimizer("adam", create_distributed_adam)
_register_dist_optimizer("fused_sparse_adam", create_distributed_fused_sparse_adam)
