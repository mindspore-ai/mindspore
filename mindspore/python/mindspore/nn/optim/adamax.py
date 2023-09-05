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
"""adamax"""
from __future__ import absolute_import

from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore import _checkparam as validator
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register

_ada_max_opt = C.MultitypeFuncGraph("ada_max_opt")


@_ada_max_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                       "Tensor", "Tensor")
def _tensor_run_opt(opt, beta1, beta2, beta1_power, eps, learning_rate, weight, moment1, moment2, gradient):
    success = True
    success = F.depend(success, opt(weight, moment1, moment2, beta1_power, learning_rate, beta1, beta2, eps, gradient))
    return success


def _check_param_value(beta1, beta2, eps, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_float_range(beta1, 0.0, 1.0, validator.INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(beta2, 0.0, 1.0, validator.INC_NEITHER, "beta2", prim_name)
    validator.check_positive_float(eps, "eps", prim_name)


class AdaMax(Optimizer):
    r"""
    Implements the AdaMax algorithm, a variant of Adaptive Movement Estimation (Adam) based on the infinity norm.

    The AdaMax algorithm is proposed in `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_.

    The updating formulas are as follows,

    .. math::
        \begin{array}{ll} \\
            m_{t+1} = \beta_1 * m_{t} + (1 - \beta_1) * g \\
            v_{t+1} = \max(\beta_2 * v_{t}, \left| g \right|) \\
            w = w - \frac{l}{1 - \beta_1^{t+1}} * \frac{m_{t+1}}{v_{t+1} + \epsilon}
        \end{array}

    :math:`m` represents the 1st moment vector, :math:`v` represents the 2nd moment vector,
    :math:`g` represents `gradients`, :math:`\beta_1, \beta_2` represent `beta1` and `beta2`,
    :math:`t` represents the current step, :math:`beta_1^t` represent `beta1_power`,
    :math:`l` represents `learning_rate`, :math:`w` represents `params`,
    :math:`\epsilon` represents `eps`.

    Note:
        If parameters are not grouped, the `weight_decay` in optimizer will be applied on the network parameters without
        'beta' or 'gamma' in their names. Users can group parameters to change the strategy of decaying weight. When
        parameters are grouped, each group can set `weight_decay`. If not, the `weight_decay` in optimizer will be
        applied.

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

        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule]): Default: ``0.001`` .

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        beta1 (float): The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).
                       Default: ``0.9`` .
        beta2 (float): The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).
                       Default: ``0.999`` .
        eps (float): Term added to the denominator to improve numerical stability. Should be greater than 0.
                     Default: ``1e-08`` .

        weight_decay (Union[float, int, Cell]): Weight decay (L2 penalty). Default: ``0.0`` .

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

        loss_scale (float): A floating point value for the loss scale. Should be greater than 0. In general, use the
            default value. Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to ``False`` , then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.amp.FixedLossScaleManager` for more details.
            Default: ``1.0`` .

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        Tensor[bool], the value is True.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `beta1`, `beta2`, `eps` or `loss_scale` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        ValueError: If `loss_scale` or `eps` is less than or equal to 0.
        ValueError: If `beta1`, `beta2` is not in range (0.0, 1.0).
        ValueError: If `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.AdaMax(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = nn.AdaMax(group_params, learning_rate=0.1, weight_decay=0.0)
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
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-08,
                 weight_decay=0.0, loss_scale=1.0):
        super(AdaMax, self).__init__(learning_rate, params, weight_decay, loss_scale)
        _check_param_value(beta1, beta2, eps, self.cls_name)

        self.beta1 = Tensor(beta1, mstype.float32)
        self.beta2 = Tensor(beta2, mstype.float32)
        self.beta1_power = Parameter(initializer(1, [1], mstype.float32), name="beta1_power")
        self.eps = Tensor(eps, mstype.float32)
        self.moment1 = self._parameters.clone(prefix="moment1", init='zeros')
        self.moment2 = self._parameters.clone(prefix="moment2", init='zeros')

        self.opt = P.ApplyAdaMax()

    @jit
    def construct(self, gradients):
        gradients = self.flatten_gradients(gradients)
        gradients = self.decay_weight(gradients)
        gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        self.beta1_power *= self.beta1

        if self.is_group_lr:
            success = self.map_(F.partial(_ada_max_opt, self.opt, self.beta1, self.beta2, self.beta1_power, self.eps),
                                lr, self._parameters, self.moment1, self.moment2, gradients)
        else:
            success = self.map_(F.partial(_ada_max_opt, self.opt, self.beta1, self.beta2, self.beta1_power,
                                          self.eps, lr), self._parameters, self.moment1, self.moment2, gradients)

        return success
