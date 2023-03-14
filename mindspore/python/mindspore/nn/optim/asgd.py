# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""asgd"""
from __future__ import absolute_import

from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.api import jit
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
import mindspore
from mindspore._checkparam import Validator as validator
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register


class ASGD(Optimizer):
    r"""
    Implements Average Stochastic Gradient Descent.

    Introduction to ASGD can be found at `Acceleration of stochastic approximation by average
    <http://dl.acm.org/citation.cfm?id=131098>`_.

    The updating formulas are as follows:

    .. math::
        \begin{gather*}
            w_{t} = w_{t-1} * (1 - \lambda * \eta_{t-1}) - \eta_{t-1} * g_{t} \\
            ax_{t} = (w_t - ax_{t-1}) * \mu_{t-1} \\
            \eta_{t} = \frac{1.}{(1 + \lambda * lr * t)^\alpha} \\
            \mu_{t} = \frac{1}{\max(1, t - t0)}
        \end{gather*}

    :math:`\lambda` represents the decay term, :math:`\mu` and :math:`\eta` are tracked to
    update :math:`ax` and :math:`w`, :math:`t0` represents the point of starting averaging,
    :math:`\alpha` represents the power for :math:`\eta` update, :math:`ax` represents the averaged
    parameter value, :math:`t` represents the current step, :math:`g` represents `gradients`,
    :math:`w` represents `params`.

    Note:
        If parameters are not grouped, the `weight_decay` in optimizer will be applied on the parameters without 'beta'
        or 'gamma' in their names. Users can group parameters to change the strategy of decaying weight. When parameters
        are grouped, each group can set `weight_decay`, if not, the `weight_decay` in optimizer will be applied.

    Args:
        params (Union[list[Parameter], list[dict]]): Must be list of `Parameter` or list of `dict`. When the
            `parameters` is a list of `dict`, the "params", "lr", "weight_decay", "grad_centralization" and
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

        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule]): learning_rate. Default: 0.1.

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        lambd (float): The decay term. Default: 1e-4.
        alpha (float): The power for :math:`\eta` update. Default: 0.75.
        t0 (float): The point of starting averaging. Default: 1e6.
        weight_decay (Union[float, int, Cell]): Weight decay (L2 penalty). Default: 0.0.

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
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `lambd`, `alpha` or `t0` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        ValueError: If `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>>
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.ASGD(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params,'grad_centralization':True},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = nn.ASGD(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The conv_params's parameters will use default learning rate of 0.1 default weight decay of 0.0 and grad
        >>> # centralization of True.
        >>> # The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0 and grad
        >>> # centralization of False.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = ms.Model(net, loss_fn=loss, optimizer=optim)
    """

    @opt_init_args_register
    def __init__(self, params, learning_rate=0.1, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0.0):

        super(ASGD, self).__init__(learning_rate, params, weight_decay)

        validator.check_value_type("lambd", lambd, [float], self.cls_name)
        validator.check_value_type("alpha", alpha, [float], self.cls_name)
        validator.check_value_type("t0", t0, [float], self.cls_name)

        self.lambd = lambd
        self.alpha = alpha
        self.t0 = Tensor([t0], dtype=mstype.float32)
        mu, eta = [], []
        for param in self._parameters:
            mu.append(Parameter(Tensor(1., dtype=mstype.float32), name='%s%s' % ("mu_", param.name)))
            eta.append(Parameter(Tensor(0., dtype=mstype.float32), name='%s%s' % ("eta_", param.name)))
        self.lens = len(self._parameters)
        self.mu = mindspore.ParameterTuple(mu)
        self.eta = mindspore.ParameterTuple(eta)
        self.ax = self._parameters.clone(prefix="ax_", init='zeros')
        self.pow = P.Pow()
        self.maximum = P.Maximum()
        self.assign = P.Assign()
        self.assignadd = P.AssignAdd()
        self.assignsub = P.AssignSub()
        self.cast = P.Cast()
        self.squeeze = P.Squeeze()

    @jit
    def construct(self, gradients):
        gradients = self.flatten_gradients(gradients)
        gradients = self.decay_weight(gradients)
        gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)
        lrs = self.get_lr()
        if not self._is_dynamic_lr_or_weight_decay():
            self.assignadd(self.global_step, self.global_step_increase_tensor)
        success = True
        params = self._parameters
        for index, (grad, param, mu, eta, ax) in enumerate(zip(gradients, params, self.mu, self.eta, self.ax)):
            lr = lrs[index] if self.is_group_lr else lrs
            lr = self.squeeze(lr)

            if self.squeeze(self.global_step) == 1:
                self.assign(eta, lr)

            param_fp32 = self.cast(param, mstype.float32)
            gradient_fp32 = self.cast(grad, mstype.float32)
            ax_fp32 = self.cast(ax, mstype.float32)
            param_fp32 = param_fp32 * (1. - self.lambd * eta) - eta * gradient_fp32

            self.assign(param, self.cast(param_fp32, param.dtype))

            if mu != 1:
                self.assignadd(ax, self.cast((param_fp32 - ax_fp32) * mu, ax.dtype))
            else:
                self.assign(ax, param)

            self.assign(eta, lr / (self.pow((1. + (self.lambd * lr * self.cast(
                self.squeeze(self.global_step), mstype.float32))), self.alpha)))
            self.assign(mu, 1. / self.squeeze(self.maximum(1., self.cast(
                self.squeeze(self.global_step), mstype.float32) - self.t0)))
        return success
