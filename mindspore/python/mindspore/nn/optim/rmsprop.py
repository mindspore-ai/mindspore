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
"""rmsprop"""
from __future__ import absolute_import

from mindspore.ops import functional as F, composite as C, operations as P
from mindspore import _checkparam as validator
from mindspore.common.api import jit
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register

_rmsprop_opt = C.MultitypeFuncGraph("rmsprop_opt")
_centered_rmsprop_opt = C.MultitypeFuncGraph("rmsprop_opt")


@_rmsprop_opt.register("Function", "Number", "Number", "Number", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _rmsprop_opt_(opt, decay, epsilon, momentum, learning_rate, weight, ms, mom, grad):
    """Apply rmsprop optimizer to the weight parameter using dynamic learning rate."""
    success = True
    success = F.depend(success, opt(weight, ms, mom, learning_rate, grad, decay, momentum, epsilon))
    return success


@_centered_rmsprop_opt.register("Function", "Number", "Number", "Number", "Tensor", "Tensor", "Tensor", "Tensor",
                                "Tensor", "Tensor")
def _centered_rmsprop_opt_(opt, decay, epsilon, momentum, learning_rate, weight, mg, ms, mom, grad):
    """Apply centered rmsprop optimizer to the weight parameter using dynamic learning rate."""
    success = True
    success = F.depend(success, opt(weight, mg, ms, mom, grad, learning_rate, decay, momentum, epsilon))
    return success


class RMSProp(Optimizer):
    """
    Implements Root Mean Squared Propagation (RMSProp) algorithm.

    Update `params` according to the RMSProp algorithm.
    The 29th of the original `presentation slide
    <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_ proposes RMSProp.
    The equation is as follows:

    .. math::
        s_{t+1} = \\rho s_{t} + (1 - \\rho)(\\nabla Q_{i}(w))^2

    .. math::
        m_{t+1} = \\beta m_{t} + \\frac{\\eta} {\\sqrt{s_{t+1} + \\epsilon}} \\nabla Q_{i}(w)

    .. math::
        w = w - m_{t+1}

    The first equation calculates moving average of the squared gradient for
    each weight. Then dividing the gradient by :math:`\\sqrt{ms_{t+1} + \\epsilon}`.

    If centered is True:

    .. math::
        g_{t+1} = \\rho g_{t} + (1 - \\rho)\\nabla Q_{i}(w)

    .. math::
        s_{t+1} = \\rho s_{t} + (1 - \\rho)(\\nabla Q_{i}(w))^2

    .. math::
        m_{t+1} = \\beta m_{t} + \\frac{\\eta} {\\sqrt{s_{t+1} - g_{t+1}^2 + \\epsilon}} \\nabla Q_{i}(w)

    .. math::
        w = w - m_{t+1}

    where :math:`w` represents `params`, which will be updated.
    :math:`g_{t+1}` is mean gradients.
    :math:`s_{t+1}` is the mean square gradients.
    :math:`m_{t+1}` is moment, the delta of `w`.
    :math:`\\rho` represents `decay`. :math:`\\beta` is the momentum term, represents `momentum`.
    :math:`\\epsilon` is a smoothing term to avoid division by zero, represents `epsilon`.
    :math:`\\eta` is learning rate, represents `learning_rate`. :math:`\\nabla Q_{i}(w)` is gradients,
    represents `gradients`.
    :math:`t` represents the current step.

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

            - order_params: Optional. When parameters are grouped, this usually is used to maintain the order of
              parameters that appeared in the network to improve performance. The value should be parameters whose
              order will be followed in optimizer.
              If `order_params` in the keys, other keys will be ignored and the element of 'order_params' must be in
              one group of `params`.

        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule]): Default: ``0.1`` .

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of the current step.

        decay (float): Decay rate. Should be equal to or greater than 0. Default: ``0.9`` .
        momentum (float): Hyperparameter of type float, means momentum for the moving average. Should be equal to or
                          greater than 0. Default: ``0.0`` .
        epsilon (float): Term added to the denominator to improve numerical stability. Should be greater than
                         0. Default: ``1e-10`` .
        use_locking (bool):  Whether to enable a lock to protect the updating process of variable tensors.
            Default: ``False`` .
        centered (bool): If True, gradients are normalized by the estimated variance of the gradient.
            Default: ``False`` .
        loss_scale (float): A floating point value for the loss scale. Should be greater than 0. In general, use the
            default value. Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to ``False`` , then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.amp.FixedLossScaleManager` for more details.
            Default: ``1.0`` .
        weight_decay (Union[float, int, Cell]): Weight decay (L2 penalty). Default: ``0.0`` .

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        Tensor[bool], the value is True.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If `decay`, `momentum`, `epsilon` or `loss_scale` is not a float.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `weight_decay` is neither float nor int.
        TypeError: If `use_locking` or `centered` is not a bool.
        ValueError: If `epsilon` is less than or equal to 0.
        ValueError: If `decay` or `momentum` is less than 0.

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
        >>> optim = nn.RMSProp(params=net.trainable_params(), learning_rate=0.1)
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = nn.RMSProp(group_params, learning_rate=0.1, weight_decay=0.0)
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
    def __init__(self, params, learning_rate=0.1, decay=0.9, momentum=0.0, epsilon=1e-10,
                 use_locking=False, centered=False, loss_scale=1.0, weight_decay=0.0):
        super(RMSProp, self).__init__(learning_rate, params, weight_decay, loss_scale)
        validator.check_value_type("decay", decay, [float], self.cls_name)
        validator.check_non_negative_float(decay, "decay", self.cls_name)
        validator.check_value_type("momentum", momentum, [float], self.cls_name)
        validator.check_non_negative_float(momentum, "momentum", self.cls_name)
        validator.check_value_type("epsilon", epsilon, [float], self.cls_name)
        validator.check_positive_float(epsilon, "epsilon", self.cls_name)
        validator.check_value_type("use_locking", use_locking, [bool], self.cls_name)
        validator.check_value_type("centered", centered, [bool], self.cls_name)

        self.centered = centered
        if centered:
            self.opt = P.ApplyCenteredRMSProp(use_locking)
            self.mg = self._parameters.clone(prefix="mean_grad", init='zeros')
        else:
            self.opt = P.ApplyRMSProp(use_locking)

        self.momentum = momentum
        self.ms = self._parameters.clone(prefix="mean_square", init='ones')
        self.moment = self._parameters.clone(prefix="moment", init='zeros')
        self.epsilon = epsilon
        self.decay = decay

    @jit
    def construct(self, gradients):
        params = self._parameters
        gradients = self.flatten_gradients(gradients)
        gradients = self.decay_weight(gradients)
        gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)
        if self.centered:
            if self.is_group_lr:
                success = self.hyper_map_reverse(F.partial(_centered_rmsprop_opt, self.opt, self.decay, self.epsilon,
                                                           self.momentum),
                                                 lr, params, self.mg, self.ms, self.moment, gradients)
            else:
                success = self.hyper_map_reverse(F.partial(_centered_rmsprop_opt, self.opt, self.decay, self.epsilon,
                                                           self.momentum, lr),
                                                 params, self.mg, self.ms, self.moment, gradients)
        else:
            if self.is_group_lr:
                success = self.hyper_map_reverse(F.partial(_rmsprop_opt, self.opt, self.decay, self.epsilon,
                                                           self.momentum),
                                                 lr, params, self.ms, self.moment, gradients)
            else:
                success = self.hyper_map_reverse(F.partial(_rmsprop_opt, self.opt, self.decay, self.epsilon,
                                                           self.momentum, lr),
                                                 params, self.ms, self.moment, gradients)
        return success
