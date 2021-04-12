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
"""rmsprop"""
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore._checkparam import Validator as validator
from .optimizer import Optimizer

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

    The equation is as follows:

    .. math::
        s_{t} = \\rho s_{t-1} + (1 - \\rho)(\\nabla Q_{i}(w))^2

    .. math::
        m_{t} = \\beta m_{t-1} + \\frac{\\eta} {\\sqrt{s_{t} + \\epsilon}} \\nabla Q_{i}(w)

    .. math::
        w = w - m_{t}

    The first equation calculates moving average of the squared gradient for
    each weight. Then dividing the gradient by :math:`\\sqrt{ms_{t} + \\epsilon}`.

    if centered is True:

    .. math::
        g_{t} = \\rho g_{t-1} + (1 - \\rho)\\nabla Q_{i}(w)

    .. math::
        s_{t} = \\rho s_{t-1} + (1 - \\rho)(\\nabla Q_{i}(w))^2

    .. math::
        m_{t} = \\beta m_{t-1} + \\frac{\\eta} {\\sqrt{s_{t} - g_{t}^2 + \\epsilon}} \\nabla Q_{i}(w)

    .. math::
        w = w - m_{t}

    where :math:`w` represents `params`, which will be updated.
    :math:`g_{t}` is mean gradients, :math:`g_{t-1}` is the last moment of :math:`g_{t}`.
    :math:`s_{t}` is the mean square gradients, :math:`s_{t-1}` is the last moment of :math:`s_{t}`,
    :math:`m_{t}` is moment, the delta of `w`, :math:`m_{t-1}` is the last moment of :math:`m_{t}`.
    :math:`\\rho` represents `decay`. :math:`\\beta` is the momentum term, represents `momentum`.
    :math:`\\epsilon` is a smoothing term to avoid division by zero, represents `epsilon`.
    :math:`\\eta` is learning rate, represents `learning_rate`. :math:`\\nabla Q_{i}(w)` is gradients,
    represents `gradients`.

    Note:
        When separating parameter groups, the weight decay in each group will be applied on the parameters if the
        weight decay is positive. When not separating parameter groups, the `weight_decay` in the API will be applied
        on the parameters without 'beta' or 'gamma' in their names if `weight_decay` is positive.

        When separating parameter groups, if you want to centralize the gradient, set grad_centralization to True,
        but the gradient centralization can only be applied to the parameters of the convolution layer.
        If the parameters of the non convolution layer are set to True, an error will be reported.

        To improve parameter groups performance, the customized order of parameters can be supported.

    Args:
        params (Union[list[Parameter], list[dict]]): When the `params` is a list of `Parameter` which will be updated,
            the element in `params` must be class `Parameter`. When the `params` is a list of `dict`, the "params",
            "lr", "weight_decay" and "order_params" are the keys can be parsed.

            - params: Required. The value must be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in the API will be used.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the API will be used.

            - order_params: Optional. If "order_params" in the keys, the value must be the order of parameters and
              the order will be followed in optimizer. There are no other keys in the `dict` and the parameters which
              in the value of 'order_params' must be in one of group parameters.

            - grad_centralization: Optional. The data type of "grad_centralization" is Bool. If "grad_centralization"
              is in the keys, the set value will be used. If not, the `grad_centralization` is False by default.
              This parameter only works on the convolution layer.

        learning_rate (Union[float, Tensor, Iterable, LearningRateSchedule]): A value or a graph for the learning rate.
            When the learning_rate is an Iterable or a Tensor in a 1D dimension, use dynamic learning rate, then
            the i-th step will take the i-th value as the learning rate. When the learning_rate is LearningRateSchedule,
            use dynamic learning rate, the i-th learning rate will be calculated during the process of training
            according to the formula of LearningRateSchedule. When the learning_rate is a float or a Tensor in a zero
            dimension, use fixed learning rate. Other cases are not supported. The float learning rate must be
            equal to or greater than 0. If the type of `learning_rate` is int, it will be converted to float.
            Default: 0.1.
        decay (float): Decay rate. Should be equal to or greater than 0. Default: 0.9.
        momentum (float): Hyperparameter of type float, means momentum for the moving average. Should be equal to or
                          greater than 0. Default: 0.0.
        epsilon (float): Term added to the denominator to improve numerical stability. Should be greater than
                         0. Default: 1e-10.
        use_locking (bool):  Whether to enable a lock to protect the variable and accumlation tensors from being
                             updated. Default: False.
        centered (bool): If true, gradients are normalized by the estimated variance of the gradient. Default: False.
        loss_scale (float): A floating point value for the loss scale. Should be greater than 0. In general, use the
            default value. Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to False, then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.FixedLossScaleManager` for more details.
            Default: 1.0.
        weight_decay (Union[float, int]): Weight decay (L2 penalty). Should be equal to or greater than 0. Default: 0.0.

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
        >>> net = Net()
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
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
    """
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
            self.mg = self.parameters.clone(prefix="mean_grad", init='zeros')
        else:
            self.opt = P.ApplyRMSProp(use_locking)

        self.momentum = momentum
        self.ms = self.parameters.clone(prefix="mean_square", init='ones')
        self.moment = self.parameters.clone(prefix="moment", init='zeros')
        self.hyper_map = C.HyperMap()
        self.epsilon = epsilon
        self.decay = decay

    def construct(self, gradients):
        params = self.parameters
        gradients = self.decay_weight(gradients)
        gradients = self.scale_grad(gradients)
        gradients = self.gradients_centralization(gradients)
        lr = self.get_lr()
        if self.centered:
            if self.is_group_lr:
                success = self.hyper_map(F.partial(_centered_rmsprop_opt, self.opt, self.decay, self.epsilon,
                                                   self.momentum), lr, params, self.mg, self.ms, self.moment, gradients)
            else:
                success = self.hyper_map(F.partial(_centered_rmsprop_opt, self.opt, self.decay, self.epsilon,
                                                   self.momentum, lr), params, self.mg, self.ms, self.moment, gradients)

        else:
            if self.is_group_lr:
                success = self.hyper_map(F.partial(_rmsprop_opt, self.opt, self.decay, self.epsilon,
                                                   self.momentum), lr, params, self.ms, self.moment, gradients)
            else:
                success = self.hyper_map(F.partial(_rmsprop_opt, self.opt, self.decay, self.epsilon,
                                                   self.momentum, lr), params, self.ms, self.moment, gradients)
        return success
