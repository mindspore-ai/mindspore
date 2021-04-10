# Copyright 2020 Huawei Technologies Co., Ltd
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

import numpy as np

from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.nn import Optimizer
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel

_learning_rate_update_func = ['linear', 'cos', 'sin']

adam_opt = C.MultitypeFuncGraph("adam_opt")


@adam_opt.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool")
def _update_run_op(beta1, beta2, eps, lr, weight_decay_tensor, param, m, v, gradient, decay_flag):
    """
    Update parameters.

    Args:
        beta1 (Tensor): The exponential decay rate for the 1st moment estimates. Should be in range (0.0, 1.0).
        beta2 (Tensor): The exponential decay rate for the 2nd moment estimates. Should be in range (0.0, 1.0).
        eps (Tensor): Term added to the denominator to improve numerical stability. Should be greater than 0.
        lr (Tensor): Learning rate.
        weight_decay_tensor (Tensor): Weight decay. Should be equal to or greater than 0.
        param (Tensor): Parameters.
        m (Tensor): m value of parameters.
        v (Tensor): v value of parameters.
        gradient (Tensor): Gradient of parameters.

    Returns:
        Tensor, the new value of v after updating.
    """
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

    next_m = op_mul(beta1, m_fp32) + op_mul(op_cast(F.tuple_to_array((1.0,)), mstype.float32) - beta1, gradient_fp32)

    next_v = op_mul(beta2, v_fp32) + op_mul(op_cast(F.tuple_to_array((1.0,)), mstype.float32)
                                            - beta2, op_square(gradient_fp32))

    update = next_m / (op_sqrt(next_v) + eps)
    if decay_flag:
        update = update + op_mul(weight_decay_tensor, param_fp32)

    update_with_lr = op_mul(lr, update)
    next_param = param_fp32 - op_reshape(update_with_lr, op_shape(param_fp32))

    next_v = F.depend(next_v, F.assign(param, next_param))
    next_v = F.depend(next_v, F.assign(m, next_m))
    next_v = F.depend(next_v, F.assign(v, next_v))
    return next_v


def _check_param_value(beta1, beta2, eps, weight_decay, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_value_type("weight_dacay", weight_decay, [float], prim_name)



def _check_learning_rate_value(learning_rate, end_learning_rate, decay_steps, power, prim_name):
    """Check the type of inputs."""
    validator.check_float_positive('learning_rate', learning_rate, prim_name)
    validator.check_float_legal_value('learning_rate', learning_rate, prim_name)
    validator.check_float_positive('end_learning_rate', end_learning_rate, prim_name)
    validator.check_float_legal_value('end_learning_rate', end_learning_rate, prim_name)
    validator.check_float_positive('power', power, prim_name)
    validator.check_float_legal_value('power', power, prim_name)
    validator.check_integer('decay_steps', decay_steps, 0, Rel.GT, prim_name)


@adam_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Number", "Tensor", "Tensor", "Tensor", "Tensor",
                   "Tensor")
def _run_opt_with_one_number(opt, beta1_power, beta2_power, beta1, beta2, eps, lr, gradient, params, moment1,
                             moment2):
    """Apply adam optimizer to the weight parameter using Tensor."""
    success = True
    success = F.depend(success, opt(params, moment1, moment2, beta1_power, beta2_power, lr, beta1, beta2,
                                    eps, gradient))
    return success


class Adam(Optimizer):
    r"""
    Updates gradients by Adaptive Moment Estimation (Adam) algorithm.

    The Adam algorithm is proposed in `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_.

    The updating formulas are as follows,

    .. math::
        \begin{array}{ll} \\
            m = \beta_1 * m + (1 - \beta_1) * g \\
            v = \beta_2 * v + (1 - \beta_2) * g * g \\
            l = \alpha * \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \\
            w = w - l * \frac{m}{\sqrt{v} + \epsilon}
        \end{array}

    :math:`m` represents the 1st moment vector `moment1`, :math:`v` represents the 2nd moment vector `moment2`,
    :math:`g` represents `gradients`, :math:`l` represents scaling factor `lr`, :math:`\beta_1, \beta_2` represent
    `beta1` and `beta2`, :math:`t` represents updating step while :math:`beta_1^t` and :math:`beta_2^t` represent
    `beta1_power` and `beta2_power`, :math:`\alpha` represents `learning_rate`, :math:`w` represents `params`,
    :math:`\epsilon` represents `eps`.

    Note:
        The Adam optimizer supports separating parameter groups. Different parameter groups can set different
        `learning_rate` and `weight_decay`.

        When separating parameter groups, the weight decay in each group will be applied on the parameters if the
        value of weight_decay > 0. When not separating parameter groups, the `weight_decay` in the API will be
        applied on the parameters if `weight_decay` > 0 and the 'beta' and 'gamma' are not in the name of parameters.

    Args:
        params (Union[list[Parameter], list[dict]]): When the `params` is a list of `Parameter` which will be updated,
            the element in `params` should be class `Parameter`. When the `params` is a list of `dict`, the "params",
            "lr" and "weight_decay" are the keys can be parsed.

            - params: Required. The value should be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in the API will be used.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the API will be used.

        learning_rate (Union[float, Tensor, Iterable]): A value for the learning rate. When the learning_rate is
                                                        Iterable or a Tensor and the dims of the Tensor is 1,
                                                        use dynamic learning rate, then the i-th step will
                                                        take the i-th value as the learning rate.
                                                        When the learning_rate is float or learning_rate is a Tensor
                                                        but the dims of the Tensor is 0, use fixed learning rate.
                                                        Other cases are not supported. Default: 1e-3.
        beta1 (float): The exponential decay rate for the 1st moment estimates. Should be in range (0.0, 1.0). Default:
                       0.9.
        beta2 (float): The exponential decay rate for the 2nd moment estimates. Should be in range (0.0, 1.0). Default:
                       0.999.
        eps (float): Term added to the denominator to improve numerical stability. Should be greater than 0. Default:
                     1e-8.
        use_locking (bool): Whether to enable a lock to protect updating variable tensors.
            If True, updating of the var, m, and v tensors will be protected by a lock.
            If False, the result is unpredictable. Default: False.
        use_nesterov (bool): Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients.
            If True, updates the gradients using NAG.
            If False, updates the gradients without using NAG. Default: False.
        weight_decay (float): Weight decay (L2 penalty). Default: 0.0.
        loss_scale (float): A floating point value for the loss scale. Should be equal to or greater than 1. Default:
                            1.0.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        Tensor[bool], the value is True.

    Examples:
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.Adam(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01, 'lr': 0.01},
        >>>                 {'params': no_conv_params}]
        >>> opt = nn.Adam(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # the conv_params's parameters will use a learning rate of 0.01 and a weight decay of 0.01
        >>> # the no_cov_params's parameters don't set learning and weight decay. So they will use a
        >>> # learning rate of 0.1 and a weight decay of 0.0.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
    """

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, use_locking=False,
                 use_nesterov=False, weight_decay=0.0, loss_scale=1.0):
        super(Adam, self).__init__(learning_rate, params, weight_decay, loss_scale)
        _check_param_value(beta1, beta2, eps, weight_decay, self.cls_name)
        validator.check_value_type("use_locking", use_locking, [bool], self.cls_name)
        validator.check_value_type("use_nesterov", use_nesterov, [bool], self.cls_name)
        validator.check_value_type("loss_scale", loss_scale, [float], self.cls_name)

        self.beta1 = Tensor(beta1, mstype.float32)
        self.beta2 = Tensor(beta2, mstype.float32)
        self.beta1_power = Parameter(initializer(1, [1], mstype.float32))
        self.beta2_power = Parameter(initializer(1, [1], mstype.float32))
        self.eps = eps

        self.moment1 = self.parameters.clone(prefix="moment1", init='zeros')
        self.moment2 = self.parameters.clone(prefix="moment2", init='zeros')

        self.hyper_map = C.HyperMap()
        self.opt = P.Adam(use_locking, use_nesterov)

        self.pow = P.Pow()
        self.sqrt = P.Sqrt()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.realdiv = P.RealDiv()

        self.lr_scalar = P.ScalarSummary()

    def construct(self, gradients):
        """Adam optimizer."""
        params = self.parameters
        moment1 = self.moment1
        moment2 = self.moment2
        gradients = self.decay_weight(gradients)
        gradients = self.scale_grad(gradients)
        lr = self.get_lr()

        self.lr_scalar("learning_rate", lr)

        beta1_power = self.beta1_power * self.beta1
        self.beta1_power = beta1_power
        beta2_power = self.beta2_power * self.beta2
        self.beta2_power = beta2_power
        if self.is_group_lr:
            success = self.hyper_map(F.partial(adam_opt, self.opt, beta1_power, beta2_power, self.beta1,
                                               self.beta2, self.eps),
                                     lr, gradients, params, moment1, moment2)
        else:
            success = self.hyper_map(F.partial(adam_opt, self.opt, beta1_power, beta2_power, self.beta1,
                                               self.beta2, self.eps, lr),
                                     gradients, params, moment1, moment2)
        return success


class AdamWeightDecay(Optimizer):
    """
    Implements Adam algorithm weight decay fix.

    Args:
        params (list[Parameter]): A list of parameter, which will be updated. The element in `params`
                                  should be class mindspore.Parameter.
        learning_rate (Union[float, Tensor, Iterable]): A value for the learning rate. When the learning_rate is
                                                        Iterable or a Tensor and the dims of the Tensor is 1,
                                                        use dynamic learning rate, then the i-th step will
                                                        take the i-th value as the learning rate.
                                                        When the learning_rate is float or learning_rate is a Tensor
                                                        but the dims of the Tensor is 0, use fixed learning rate.
                                                        Other cases are not supported. Default: 1e-3.
        beta1 (float): The exponential decay rate for the 1st moment estimates. Default: 0.9.
            Should be in range (0.0, 1.0).
        beta2 (float): The exponential decay rate for the 2nd moment estimates. Default: 0.999.
            Should be in range (0.0, 1.0).
        eps (float): Term added to the denominator to improve numerical stability. Default: 1e-6.
            Should be greater than 0.
        weight_decay (float): Weight decay (L2 penalty). Default: 0.0.
        decay_filter (Function): A function to determine whether to apply weight decay on parameters. Default:
                                 lambda x: 'LayerNorm' not in x.name and 'bias' not in x.name.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[Parameter], the updated velocity value, the shape is the same as `params`.

    Examples:
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = nn.AdamWeightDecay(params=net.trainable_params())
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
   """

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0,
                 decay_filter=lambda x: 'beta' not in x.name and 'gamma' not in x.name):
        super(AdamWeightDecay, self).__init__(learning_rate, params)
        if self.is_group:
            raise RuntimeError(f"The {self.cls_name} optimizer cannot support group setting.")
        _check_param_value(beta1, beta2, eps, weight_decay, self.cls_name)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.weight_decay_tensor = Tensor(np.array([weight_decay]).astype(np.float32))

        self.params = self.parameters
        self.moments1 = self.params.clone(prefix="adam_m", init='zeros')
        self.moments2 = self.params.clone(prefix="adam_v", init='zeros')
        self.decay_flag = tuple(decay_filter(x) for x in self.params)

        self.hyper_map = C.HyperMap()

    def construct(self, gradients):
        """Adam Weight Decay"""
        lr = self.get_lr()
        updated_velocity = self.hyper_map(F.partial(adam_opt, self.beta1, self.beta2, self.eps, lr,
                                                    self.weight_decay_tensor),
                                          self.params, self.moments1, self.moments2, gradients, self.decay_flag)

        return updated_velocity


class AdamWeightDecayDynamicLR(Optimizer):
    """
    Adam Weight Decay Dynamic Learning Rate (LR).

    Args:
        params (list[Parameter]): A list of parameter, which will be updated. The element in `params`
                                  should be class mindspore.Parameter.
        decay_steps (int): The steps of the decay.
        learning_rate (float): A floating point value for the learning rate. Default: 0.001.
        end_learning_rate (float): A floating point value for the end learning rate. Default: 0.0001.
        power (float): Power. Default: 10.0.
        beta1 (float): The exponential decay rate for the 1st moment estimates. Default: 0.9.
            Should be in range (0.0, 1.0).
        beta2 (float): The exponential decay rate for the 2nd moment estimates. Default: 0.999.
            Should be in range (0.0, 1.0).
        eps (float): Term added to the denominator to improve numerical stability. Default: 1e-6.
            Should be greater than 0.
        weight_decay (float): Weight decay (L2 penalty). Default: 0.0.
        decay_filter (Function): A function to determine whether to apply weight decay on parameters. Default:
                                 lambda x: 'LayerNorm' not in x.name and 'bias' not in x.name.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[Parameter], the updated velocity value, the shape is the same as `params`.

    Examples:
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = nn.AdamWeightDecayDynamicLR(params=net.trainable_params(), decay_steps=10)
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
    """

    def __init__(self,
                 params,
                 decay_steps,
                 learning_rate=0.001,
                 end_learning_rate=0.0001,
                 power=10.0,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-6,
                 weight_decay=0.0,
                 decay_filter=lambda x: 'beta' not in x.name and 'gamma' not in x.name,
                 warmup_steps=0):
        super(AdamWeightDecayDynamicLR, self).__init__(learning_rate, params)
        if self.is_group:
            raise RuntimeError(f"The {self.cls_name} optimizer cannot support group setting.")
        _check_param_value(beta1, beta2, eps, weight_decay, self.cls_name)
        _check_learning_rate_value(learning_rate, end_learning_rate, decay_steps, power, self.cls_name)
        # turn them to scalar when me support scalar/tensor mix operations
        self.global_step = Parameter(initializer(0, [1]))
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
        self.decay_steps = Tensor(np.array([decay_steps]).astype(np.float32))
        self.end_learning_rate = Tensor(np.array([end_learning_rate]).astype(np.float32))
        self.diff_learning_rate = Tensor(np.array([learning_rate - end_learning_rate]).astype(np.float32))
        self.power = power
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.weight_decay_tensor = Tensor(np.array([weight_decay]).astype(np.float32))
        self.params = self.parameters
        self.moments1 = self.params.clone(prefix="adam_m", init='zeros')
        self.moments2 = self.params.clone(prefix="adam_v", init='zeros')
        self.decay_flag = tuple(decay_filter(x) for x in self.params)
        self.hyper_map = C.HyperMap()
        self.min = P.Minimum()
        self.pow = P.Pow()
        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.start_learning_rate = Tensor(np.array([learning_rate]).astype(np.float32))

    def construct(self, gradients):
        """Adam Weight Decay Dynamic LR."""
        step = self.min(self.global_step, self.decay_steps)
        p = step / self.decay_steps
        lr = self.diff_learning_rate * self.pow(self.one - p, self.power) + self.end_learning_rate
        if self.warmup_flag:
            warmup_percent = self.global_step / self.warmup_steps
            warmup_lr = self.start_learning_rate * warmup_percent
            is_warmup = self.cast(self.greater(self.warmup_steps, self.global_step), mstype.float32)
            lr = (self.one - is_warmup) * lr + is_warmup * warmup_lr
        updated_velocity = self.hyper_map(F.partial(adam_opt, self.beta1, self.beta2, self.eps, lr,
                                                    self.weight_decay_tensor),
                                          self.params, self.moments1, self.moments2, gradients, self.decay_flag)

        added_global_step = self.global_step + self.one
        self.global_step = added_global_step
        return updated_velocity
