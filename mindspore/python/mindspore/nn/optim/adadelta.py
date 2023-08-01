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
"""adadelta"""
from __future__ import absolute_import

from mindspore.ops import functional as F, composite as C, operations as P
from mindspore import _checkparam as validator
from mindspore.common.tensor import Tensor
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register

_adadelta_opt = C.MultitypeFuncGraph("adadelta_opt")


@_adadelta_opt.register("Function", "Number", "Number", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt(opt, rho, epsilon, learning_rate, weight, accum, accum_update, gradient):
    """Apply adadelta optimizer to the weight parameter."""
    success = True
    success = F.depend(success, opt(weight, accum, accum_update, learning_rate, rho, epsilon, gradient))
    return success


def _check_param_value(rho, epsilon, prim_name=None):
    """Check inputs param."""
    validator.check_value_type("rho", rho, [float], prim_name)
    validator.check_value_type("epsilon", epsilon, [float], prim_name)
    validator.check_float_range(rho, 0.0, 1.0, validator.INC_BOTH, "rho", prim_name)
    validator.check_non_negative_float(epsilon, "epsilon", prim_name)


class Adadelta(Optimizer):
    r"""
    Implements the Adadelta algorithm.

    Adadelta is an online Learning and Stochastic Optimization.
    Refer to paper `ADADELTA: AN ADAPTIVE LEARNING RATE METHOD <https://arxiv.org/pdf/1212.5701.pdf>`_.

    .. math::
        \begin{array}{ll} \\
            accum_{t} = \rho * accum_{t-1} + (1 - \rho) * g_{t}^2 \\
            update_{t} = \sqrt{accum\_update_{t-1} + \epsilon} * \frac{g_{t}}{\sqrt{accum_{t} + \epsilon}} \\
            accum\_update_{t} = \rho * accum\_update_{t-1} + (1 - \rho) * update_{t}^2 \\
            w_{t} = w_{t-1} - \gamma * update_{t}
        \end{array}

    where :math:`g` represents `grads`, :math:`\gamma` represents `learning_rate`, :math:`p` represents `rho`,
    :math:`\epsilon` represents `epsilon`, :math:`w` represents `params`,
    :math:`accum` represents accumulation, :math:`accum\_update` represents accumulation update,
    :math:`t` represents current step.

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
              will be used. If not, the `grad_centralization` is ``False``  by default. This configuration only works
              on the convolution layer.

            - order_params: Optional. When parameters is grouped, this usually is used to maintain the order of
              parameters that appeared in the network to improve performance. The value should be parameters whose
              order will be followed in optimizer.
              If `order_params` in the keys, other keys will be ignored and the element of 'order_params' must be in
              one group of `params`.

        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule]): Default: ``1.0`` .

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        rho (float): Decay rate, must be in range [0.0, 1.0]. Default: ``0.9`` .
        epsilon (float):  A small value added for numerical stability, must be non-negative. Default: ``1e-6`` .
        loss_scale (float): Value for the loss scale. It must be greater than 0.0. In general, use the default value.
            Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to ``False`` , then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.amp.FixedLossScaleManager` for more details.
            Default: ``1.0`` .
        weight_decay (Union[float, int, Cell]): Weight decay (L2 penalty). Default: ``0.0`` .

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

    Inputs:
        - **grads** (tuple[Tensor]) - The gradients of `params` in the optimizer, has the same shape and data type as
          the `params` in optimizer. With float16 or float32 data type.

    Outputs:
        Tensor[bool], the value is ``True`` .

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `rho`, `epsilon` or `loss_scale` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        ValueError: if `rho` is not in range [0.0, 1.0].
        ValueError: If `loss_scale` is less than or equal to 0.
        ValueError: If `learning_rate`, `epsilon` or `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>>
        >>> class Net(nn.Cell):
        ...    def __init__(self):
        ...        super(Net, self).__init__()
        ...        self.conv = nn.Conv1d(120, 240, 4)
        ...        self.fc = nn.Dense(4, 1)
        ...    def construct(self, x):
        ...         x = self.conv(x)
        ...         x = self.fc(x)
        ...         return x
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.Adadelta(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = nn.Adadelta(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01 and grad
        >>> # centralization of True.
        >>> # The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0 and grad
        >>> # centralization of False.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = ms.train.Model(net, loss_fn=loss, optimizer=optim)
    """

    @opt_init_args_register
    def __init__(self, params, learning_rate=1.0, rho=0.9, epsilon=1e-6, loss_scale=1.0, weight_decay=0.0):
        super(Adadelta, self).__init__(learning_rate, params, weight_decay, loss_scale)
        _check_param_value(rho, epsilon, self.cls_name)
        self.accum = self.parameters.clone(prefix="accum", init=0)
        self.accum_update = self.parameters.clone(prefix="accum_update", init=0)
        self.opt = P.ApplyAdadelta()
        self.rho = rho
        self.epsilon = epsilon

    def construct(self, grads):
        if not isinstance(grads, tuple) or not isinstance(grads[0], Tensor):
            raise TypeError("For 'Adadelta', the 'grads' must be a tuple of Tensor.")
        params = self.parameters
        grads = self.decay_weight(grads)
        grads = self.gradients_centralization(grads)
        grads = self.scale_grad(grads)
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)
        if self.is_group_lr:
            success = self.map_reverse(F.partial(_adadelta_opt, self.opt, self.rho, self.epsilon), lr, params,
                                       self.accum, self.accum_update, grads)
        else:
            success = self.map_reverse(F.partial(_adadelta_opt, self.opt, self.rho, self.epsilon, lr), params,
                                       self.accum, self.accum_update, grads)
        return success
