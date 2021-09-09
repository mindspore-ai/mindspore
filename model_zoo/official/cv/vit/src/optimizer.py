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
"""Gradient clipping wrapper for optimizers."""

import numpy as np

from mindspore._checkparam import Validator as validator
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops import composite as C

from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Rel
from mindspore.nn.optim import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register


def _check_param_value(beta1, beta2, eps, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_float_range(beta1, 0.0, 1.0, Rel.INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(beta2, 0.0, 1.0, Rel.INC_NEITHER, "beta2", prim_name)
    validator.check_positive_float(eps, "eps", prim_name)


_grad_scale = C.MultitypeFuncGraph("grad_scale")
op_mul = P.Mul()
map_ = C.Map()


@_grad_scale.register("Number", "Tensor")
def tensor_grad_scale(scale, grad):
    """Get grad with scale."""
    if scale == 1.0:
        return grad
    return op_mul(grad, F.cast(scale, F.dtype(grad)))


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale_with_tensor(scale, grad):
    """Get grad with scale."""
    return op_mul(grad, F.cast(scale, F.dtype(grad)))


def scale_grad(gradients, reciprocal_scale):
    gradients = map_(F.partial(_grad_scale, reciprocal_scale), gradients)
    return gradients


_adam_opt = C.MultitypeFuncGraph("adam_opt")
_scaler_one = Tensor(1, mstype.int32)


@_adam_opt.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Number", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Bool", "Bool")
def _update_run_op(beta1_power, beta2_power, beta1, beta2, eps, lr, weight_decay, param, \
                   m, v, gradient, decay_flag, optim_filter):
    """
    Update parameters.

    Args:
        beta1 (Tensor): The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).
        beta2 (Tensor): The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).
        eps (Tensor): Term added to the denominator to improve numerical stability. Should be greater than 0.
        lr (Tensor): Learning rate.
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
        # op_mul = P.Mul(), defined output
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

        regulate_m = next_m / (_scaler_one - beta1_power)
        regulate_v = next_v / (_scaler_one - beta2_power)

        update = regulate_m / (eps + op_sqrt(regulate_v))
        if decay_flag:
            update = op_mul(weight_decay, param_fp32) + update

        update_with_lr = op_mul(lr, update)
        next_param = param_fp32 - op_reshape(update_with_lr, op_shape(param_fp32))

        next_param = F.depend(next_param, F.assign(param, op_cast(next_param, F.dtype(param))))
        next_param = F.depend(next_param, F.assign(m, op_cast(next_m, F.dtype(m))))
        next_param = F.depend(next_param, F.assign(v, op_cast(next_v, F.dtype(v))))

        return op_cast(next_param, F.dtype(param))
    return gradient


class AdamW(Optimizer):
    """
    Implements the gradient clipping by norm for a AdamWeightDecay optimizer.
    """
    @opt_init_args_register
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, \
                 weight_decay=0.0, loss_scale=1.0, clip=False):
        super(AdamW, self).__init__(learning_rate, params, weight_decay)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self.parameters.clone(prefix="adam_m", init='zeros')
        self.moments2 = self.parameters.clone(prefix="adam_v", init='zeros')
        self.hyper_map = C.HyperMap()
        self.beta1_power = Parameter(initializer(1, [1], mstype.float32), name="beta1_power")
        self.beta2_power = Parameter(initializer(1, [1], mstype.float32), name="beta2_power")

        self.reciprocal_scale = Tensor(1.0 / loss_scale, mstype.float32)
        self.clip = clip

    def construct(self, gradients):
        lr = self.get_lr()
        gradients = scale_grad(gradients, self.reciprocal_scale)
        if self.clip:
            gradients = C.clip_by_global_norm(gradients, 5.0, None)

        beta1_power = self.beta1_power * self.beta1
        self.beta1_power = beta1_power
        beta2_power = self.beta2_power * self.beta2
        self.beta2_power = beta2_power

        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(F.partial(_adam_opt, beta1_power, beta2_power, \
                                              self.beta1, self.beta2, self.eps),
                                              lr, self.weight_decay, self.parameters, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.hyper_map(F.partial(_adam_opt, beta1_power, beta2_power, \
                                              self.beta1, self.beta2, self.eps, lr),
                                              self.weight_decay, self.parameters, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
        else:
            optim_result = self.hyper_map(F.partial(_adam_opt, beta1_power, beta2_power, self.beta1, self.beta2, \
                                          self.eps, lr, self.weight_decay),
                                          self.parameters, self.moments1, self.moments2,
                                          gradients, self.decay_flags, self.optim_filter)
        if self.use_parallel:
            self.broadcast_params(optim_result)
        return optim_result

def paramter_group(network, weight_decay, no_weight_decay_filter, gc_flag):
    """paramter_group"""
    filter_len = len(no_weight_decay_filter)
    if filter_len > 0:
        decayed_params = []
        no_decayed_params = []
        for param in network.trainable_params():
            if all([key not in param.name for key in no_weight_decay_filter]):
                decayed_params.append(param)
            else:
                no_decayed_params.append(param)

        group_params = [{'params': decayed_params, 'weight_decay': weight_decay, 'grad_centralization': gc_flag},
                        {'params': no_decayed_params},
                        {'order_params': network.trainable_params()}]
    else:
        group_params = [{'params': network.trainable_params(), \
                        'weight_decay': weight_decay, 'grad_centralization': gc_flag},
                        {'order_params': network.trainable_params()}]

    return group_params

def get_optimizer(optimizer_name, network, lrs, args):
    no_weight_decay_filter = [x for x in args.no_weight_decay_filter.split(",") if len(x) > 0]
    group_params = paramter_group(network, args.weight_decay, no_weight_decay_filter, bool(args.gc_flag))

    if optimizer_name == 'adamw':
        opt = AdamW(group_params, lrs, args.beta1, args.beta2, loss_scale=args.loss_scale)
    else:
        raise NotImplementedError

    return opt, group_params
