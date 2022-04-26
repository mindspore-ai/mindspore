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
"""power_sign"""
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore._checkparam import Validator as validator
from .optimizer import Optimizer
from .optimizer import opt_init_args_register

_power_sign_opt = C.MultitypeFuncGraph("power_sign_opt")


@_power_sign_opt.register("Function", "Number", "Number", "Number", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt(opt, logbase, sign_decay, beta, learning_rate, gradient, weight, m):
    """Apply powersign optimizer to the weight parameter using Tensor."""
    success = F.depend(True, opt(weight, m, learning_rate, logbase, sign_decay, beta, gradient))
    return success


def _check_param(logbase, sign_decay, beta, prim_name=None):
    """Check param."""
    validator.check_value_type("logbase", logbase, [float], prim_name)
    validator.check_value_type("sign_decay", sign_decay, [float], prim_name)
    validator.check_value_type("beta", beta, [float], prim_name)


class PowerSign(Optimizer):
    r"""
    Updates relevant entries according to the AddSign algorithm.

    .. math::
        \begin{array}{ll} \\
            m_{t+1} = \beta * m_{t} + (1 - \beta) * g \\
            \text{update} = \exp(\text{logbase} * \text{sign_decay} * sign(g) * sign(m)) * g \\
            var = var - lr_{t+1} * \text{update}
        \end{array}

    :math:`t` represents updating step while :math:`m` represents the 1st moment vector, :math:`m_{t}`
    is the last moment of :math:`m_{t+1}`, :math:`lr` represents scaling factor `lr`, :math:`g` represents `grad`,
    :math:`\beta` represents `beta`.

    All of inputs comply with the implicit type conversion rules to make the data types consistent.
    If `lr`, `logbase`, `sign_decay` or `beta` is a number, the number is automatically converted to Tensor,
    and the data type is consistent with the Tensor data type involved in the operation.
    If inputs are tensors and have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.

    Notes:
        If parameters are not grouped, the `weight_decay` in optimizer will be applied on the network parameters without
        'beta' or 'gamma' in their names. Users can group parameters to change the strategy of decaying weight. When
        parameters are grouped, each group can set `weight_decay`, if not, the `weight_decay` in optimizer will be
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

        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule]):

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        logbase (Union[float, Tensor]):

            - float: A scalar with float data type.

            - Tensor: A tensor with float32 or float16 data type.

        sign_decay (Union[float, Tensor]):

            - float: A scalar with float data type.

            - Tensor: A tensor with float32 or float16 data type.

        beta (Union[float, Tensor]):

            - float: A scalar with float data type.

            - Tensor: A tensor with float32 or float16 data type.

        weight_decay (Union[float, int, Cell]): Weight decay (L2 penalty). Default: 0.0.

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

        loss_scale (float): A floating point value for the loss scale. It must be greater than 0.0. In general, use the
            default value. Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to False, then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.FixedLossScaleManager` for more details.
            Default: 1.0.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[bool]. All elements are True.

    Raises:
        TypeError: If dtype of `logbase`, `sign_decay`, `beta` is neither float16 nor float32.
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If `logbase`, `sign_decay` or `beta` is neither a Number nor a Tensor.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `loss_scale` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        ValueError: If `loss_scale` is less than or equal to 0.
        ValueError: If `weight_decay` is less than 0.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore import nn, Model
        >>> import numpy as np
        >>>
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.PowerSign(params=net.trainable_params(), learning_rate=0.1, logbase=np.e,
         sign_decay=0.99, beta=0.9)
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = nn.PowerSign(group_params, learning_rate=0.1, logbase=np.e, sign_decay=0.99,
         beta=0.9, weight_decay=0.0)
        >>> # The conv_params's parameters will use a learning rate of default value 0.1 and a weight decay of 0.01 and
        >>> # grad centralization of True.
        >>> # The no_conv_params's parameters will use a learning rate of 0.01 and a weight decay of default value 0.0
        >>> # and grad centralization of False..
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
    """
    @opt_init_args_register
    def __init__(self, params, learning_rate, logbase, sign_decay, beta, weight_decay=0.0, loss_scale=1.0):
        super(PowerSign, self).__init__(learning_rate, params, weight_decay, loss_scale)
        _check_param(logbase, sign_decay, beta, self.cls_name)
        self.logbase = logbase
        self.sign_decay = sign_decay
        self.beta = beta
        self.params = self._parameters
        self.m = self.params.clone(prefix="moving_average", init='zeros')
        self.opt = P.ApplyPowerSign()

    def construct(self, gradients):
        gradients = self.decay_weight(gradients)
        gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)
        lr = self.get_lr()
        if self.is_group_lr:
            success = self.hyper_map(F.partial(_power_sign_opt, self.opt, self.logbase, self.sign_decay, self.beta),
                                     lr, gradients, self.params, self.m)
        else:
            success = self.hyper_map(F.partial(_power_sign_opt, self.opt, self.logbase,
                                               self.sign_decay, self.beta, lr), gradients, self.params, self.m)
        return success
