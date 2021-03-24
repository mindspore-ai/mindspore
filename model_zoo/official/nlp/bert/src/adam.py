# Copyright 2021 Huawei Technologies Co., Ltd
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
"""AdamWeightDecayForBert, a customized Adam for bert. Input: gradient, overflow flag."""
import numpy as np

from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.nn.optim.optimizer import Optimizer

_adam_opt = C.MultitypeFuncGraph("adam_opt")
_scaler_one = Tensor(1, mstype.int32)
_scaler_ten = Tensor(10, mstype.float32)


@_adam_opt.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Number", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Bool", "Bool")
def _update_run_op(beta1, beta2, eps, lr, overflow, weight_decay, param, m, v, gradient, decay_flag, optim_filter):
    """
    Update parameters.

    Args:
        beta1 (Tensor): The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).
        beta2 (Tensor): The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).
        eps (Tensor): Term added to the denominator to improve numerical stability. Should be greater than 0.
        lr (Tensor): Learning rate.
        overflow (Tensor): Whether overflow occurs.
        weight_decay (Number): Weight decay. Should be equal to or greater than 0.
        param (Tensor): Parameters.
        m (Tensor): m value of parameters.
        v (Tensor): v value of parameters.
        gradient (Tensor): Gradient of parameters.
        decay_flag (bool): Applies weight decay or not.
        optim_filter (bool): Applies parameter update or not.

    Returns:
        Tensor, the new value of v after updating.
    """
    if optim_filter:
        op_mul = P.Mul()
        op_square = P.Square()
        op_sqrt = P.Sqrt()
        op_cast = P.Cast()
        op_reshape = P.Reshape()
        op_shape = P.Shape()
        op_select = P.Select()

        param_fp32 = op_cast(param, mstype.float32)
        m_fp32 = op_cast(m, mstype.float32)
        v_fp32 = op_cast(v, mstype.float32)
        gradient_fp32 = op_cast(gradient, mstype.float32)

        cond = op_cast(F.fill(mstype.int32, op_shape(m_fp32), 1) * op_reshape(overflow, (())), mstype.bool_)
        next_m = op_mul(beta1, m_fp32) + op_select(cond, m_fp32,\
                op_mul(op_cast(F.tuple_to_array((1.0,)), mstype.float32) - beta1, gradient_fp32))

        next_v = op_mul(beta2, v_fp32) + op_select(cond, v_fp32,\
                op_mul(op_cast(F.tuple_to_array((1.0,)), mstype.float32) - beta2, op_square(gradient_fp32)))

        update = next_m / (eps + op_sqrt(next_v))
        if decay_flag:
            update = op_mul(weight_decay, param_fp32) + update

        update_with_lr = op_mul(lr, update)
        zeros = F.fill(mstype.float32, op_shape(param_fp32), 0)
        next_param = param_fp32 - op_select(cond, zeros, op_reshape(update_with_lr, op_shape(param_fp32)))

        next_param = F.depend(next_param, F.assign(param, op_cast(next_param, F.dtype(param))))
        next_param = F.depend(next_param, F.assign(m, op_cast(next_m, F.dtype(m))))
        next_param = F.depend(next_param, F.assign(v, op_cast(next_v, F.dtype(v))))

        return op_cast(next_param, F.dtype(param))
    return gradient


@_adam_opt.register("Function", "Function", "Function", "Function", "Bool", "Bool", "Bool", "Tensor", "Tensor",
                    "Tensor", "Tensor", "Tensor", "Tensor", "RowTensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool")
def _run_opt_with_sparse(opt, sparse_opt, push, pull, use_locking, use_nesterov, target, beta1_power,
                         beta2_power, beta1, beta2, eps, lr, gradient, param, m, v, ps_parameter, cache_enable):
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


@_adam_opt.register("Function", "Function", "Function", "Function", "Bool", "Bool", "Bool", "Tensor", "Tensor",
                    "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool")
def _run_opt_with_one_number(opt, sparse_opt, push, pull, use_locking, use_nesterov, target,
                             beta1_power, beta2_power, beta1, beta2, eps, lr, gradient, param,
                             moment1, moment2, ps_parameter, cache_enable):
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


@_adam_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Tensor")
def _run_off_load_opt(opt, beta1_power, beta2_power, beta1, beta2, eps, lr, gradient, param, moment1, moment2):
    """Apply AdamOffload optimizer to the weight parameter using Tensor."""
    success = True
    delat_param = opt(moment1, moment2, beta1_power, beta2_power, lr, beta1, beta2, eps, gradient)
    success = F.depend(success, F.assign_add(param, delat_param))
    return success


def _check_param_value(beta1, beta2, eps, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_float_range(beta1, 0.0, 1.0, Rel.INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(beta2, 0.0, 1.0, Rel.INC_NEITHER, "beta2", prim_name)
    validator.check_positive_float(eps, "eps", prim_name)

class AdamWeightDecayForBert(Optimizer):
    """
    Implements the Adam algorithm to fix the weight decay.

    Note:
        When separating parameter groups, the weight decay in each group will be applied on the parameters if the
        weight decay is positive. When not separating parameter groups, the `weight_decay` in the API will be applied
        on the parameters without 'beta' or 'gamma' in their names if `weight_decay` is positive.

        To improve parameter groups performance, the customized order of parameters can be supported.

    Args:
        params (Union[list[Parameter], list[dict]]): When the `params` is a list of `Parameter` which will be updated,
            the element in `params` must be class `Parameter`. When the `params` is a list of `dict`, the "params",
            "lr", "weight_decay" and "order_params" are the keys can be parsed.

            - params: Required. The value must be a list of `Parameter`.

            - lr: Optional. If "lr" is in the keys, the value of the corresponding learning rate will be used.
              If not, the `learning_rate` in the API will be used.

            - weight_decay: Optional. If "weight_decay" is in the keys, the value of the corresponding weight decay
              will be used. If not, the `weight_decay` in the API will be used.

            - order_params: Optional. If "order_params" is in the keys, the value must be the order of parameters and
              the order will be followed in the optimizer. There are no other keys in the `dict` and the parameters
              which in the 'order_params' must be in one of group parameters.

        learning_rate (Union[float, Tensor, Iterable, LearningRateSchedule]): A value or a graph for the learning rate.
            When the learning_rate is an Iterable or a Tensor in a 1D dimension, use the dynamic learning rate, then
            the i-th step will take the i-th value as the learning rate. When the learning_rate is LearningRateSchedule,
            use dynamic learning rate, the i-th learning rate will be calculated during the process of training
            according to the formula of LearningRateSchedule. When the learning_rate is a float or a Tensor in a zero
            dimension, use fixed learning rate. Other cases are not supported. The float learning rate must be
            equal to or greater than 0. If the type of `learning_rate` is int, it will be converted to float.
            Default: 1e-3.
        beta1 (float): The exponential decay rate for the 1st moment estimations. Default: 0.9.
            Should be in range (0.0, 1.0).
        beta2 (float): The exponential decay rate for the 2nd moment estimations. Default: 0.999.
            Should be in range (0.0, 1.0).
        eps (float): Term added to the denominator to improve numerical stability. Default: 1e-6.
            Should be greater than 0.
        weight_decay (float): Weight decay (L2 penalty). It must be equal to or greater than 0. Default: 0.0.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.
        - **overflow** (tuple[Tensor]) - The overflow flag in dynamiclossscale.

    Outputs:
        tuple[bool], all elements are True.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
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
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
   """
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(AdamWeightDecayForBert, self).__init__(learning_rate, params, weight_decay)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self.parameters.clone(prefix="adam_m", init='zeros')
        self.moments2 = self.parameters.clone(prefix="adam_v", init='zeros')
        self.hyper_map = C.HyperMap()
        self.op_select = P.Select()
        self.op_cast = P.Cast()
        self.op_reshape = P.Reshape()
        self.op_shape = P.Shape()

    def construct(self, gradients, overflow):
        """AdamWeightDecayForBert"""
        lr = self.get_lr()
        cond = self.op_cast(F.fill(mstype.int32, self.op_shape(self.beta1), 1) *\
                            self.op_reshape(overflow, (())), mstype.bool_)
        beta1 = self.op_select(cond, self.op_cast(F.tuple_to_array((1.0,)), mstype.float32), self.beta1)
        beta2 = self.op_select(cond, self.op_cast(F.tuple_to_array((1.0,)), mstype.float32), self.beta2)
        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(F.partial(_adam_opt, self.beta1, self.beta2, self.eps),
                                              lr, self.weight_decay, self.parameters, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.hyper_map(F.partial(_adam_opt, beta1, beta2, self.eps, lr, overflow),
                                              self.weight_decay, self.parameters, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
        else:
            optim_result = self.hyper_map(F.partial(_adam_opt, self.beta1, self.beta2, self.eps, lr, self.weight_decay),
                                          self.parameters, self.moments1, self.moments2,
                                          gradients, self.decay_flags, self.optim_filter)
        if self.use_parallel:
            self.broadcast_params(optim_result)
        return optim_result
