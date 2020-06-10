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
from mindspore._checkparam import Rel
from .optimizer import Optimizer

rmsprop_opt = C.MultitypeFuncGraph("rmsprop_opt")
centered_rmsprop_opt = C.MultitypeFuncGraph("rmsprop_opt")


@rmsprop_opt.register("Function", "Number", "Number", "Number", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _rmsprop_opt(opt, decay, epsilon, momentum, learning_rate, weight, ms, mom, grad):
    """Apply rmsprop optimizer to the weight parameter using dynamic learning rate."""
    success = True
    success = F.depend(success, opt(weight, ms, mom, learning_rate, grad, decay, momentum, epsilon))
    return success


@centered_rmsprop_opt.register("Function", "Number", "Number", "Number", "Tensor", "Tensor", "Tensor", "Tensor",
                               "Tensor", "Tensor")
def _centered_rmsprop_opt(opt, decay, epsilon, momentum, learning_rate, weight, mg, ms, mom, grad):
    """Apply centered rmsprop optimizer to the weight parameter using dynamic learning rate."""
    success = True
    success = F.depend(success, opt(weight, mg, ms, mom, grad, learning_rate, decay, momentum, epsilon))
    return success


class RMSProp(Optimizer):
    """
    Implements Root Mean Squared Propagation (RMSProp) algorithm.

    Note:
        The RMSProp optimizer supports separating parameter groups. Different parameter groups can set different
        `learning_rate` and `weight_decay`.

        When separating parameter groups, the weight decay in each group will be applied on the parameters if the
        value of weight_decay > 0. When not separating parameter groups, the `weight_decay` in the API will be
        applied on the parameters if `weight_decay` > 0 and the 'beta' and 'gamma' are not in the name of parameters.

        To improve parameter groups performance, the customized order of parameters can be supported.

        Update `params` according to the RMSProp algorithm.

        The equation is as follows:

        ..  math::
            s_{t} = \\rho s_{t-1} + (1 - \\rho)(\\nabla Q_{i}(w))^2

        ..  math::
            m_{t} = \\beta m_{t-1} + \\frac{\\eta} {\\sqrt{s_{t} + \\epsilon}} \\nabla Q_{i}(w)

        ..  math::
            w = w - m_{t}

        The first equation calculates moving average of the squared gradient for
        each weight. Then dividing the gradient by :math:`\\sqrt{ms_{t} + \\epsilon}`.

        if centered is True:

        ..  math::
            g_{t} = \\rho g_{t-1} + (1 - \\rho)\\nabla Q_{i}(w)

        ..  math::
            s_{t} = \\rho s_{t-1} + (1 - \\rho)(\\nabla Q_{i}(w))^2

        ..  math::
            m_{t} = \\beta m_{t-1} + \\frac{\\eta} {\\sqrt{s_{t} - g_{t}^2 + \\epsilon}} \\nabla Q_{i}(w)

        ..  math::
            w = w - m_{t}

        where, :math:`w` represents `params`, which will be updated.
        :math:`g_{t}` is mean gradients, :math:`g_{t-1}` is the last moment of :math:`g_{t}`.
        :math:`s_{t}` is the mean square gradients, :math:`s_{t-1}` is the last moment of :math:`s_{t}`,
        :math:`m_{t}` is moment, the delta of `w`, :math:`m_{t-1}` is the last moment of :math:`m_{t}`.
        :math:`\\rho` represents `decay`. :math:`\\beta` is the momentum term, represents `momentum`.
        :math:`\\epsilon` is a smoothing term to avoid division by zero, represents `epsilon`.
        :math:`\\eta` is learning rate, represents `learning_rate`. :math:`\\nabla Q_{i}(w)` is gradientse,
        represents `gradients`.

    Args:
        params (Union[list[Parameter], list[dict]]): When the `params` is a list of `Parameter` which will be updated,
            the element in `params` should be class `Parameter`. When the `params` is a list of `dict`, the "params",
            "lr", "weight_decay" and "order_params" are the keys can be parsed.

            - params: Required. The value should be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in the API will be used.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the API will be used.

            - order_params: Optional. If "order_params" in the keys, the value should be the order of parameters and
              the order will be followed in optimizer. There are no other keys in the `dict` and the parameters which
              in the value of 'order_params' but not in any group will use default learning rate and default weight
              decay.

        learning_rate (Union[float, Tensor, Iterable]): A value for the learning rate. When the learning_rate is
                                                        Iterable or a Tensor and the dims of the Tensor is 1,
                                                        use dynamic learning rate, then the i-th step will
                                                        take the i-th value as the learning rate.
                                                        When the learning_rate is float or learning_rate is a Tensor
                                                        but the dims of the Tensor is 0, use fixed learning rate.
                                                        Other cases are not supported. Default: 0.1.
        decay (float): Decay rate. Should be equal to or greater than 0. Default: 0.9.
        momentum (float): Hyperparameter of type float, means momentum for the moving average. Should be equal to or
                          greater than 0. Default: 0.0.
        epsilon (float): Term added to the denominator to improve numerical stability. Should be greater than
                         0. Default: 1e-10.
        use_locking (bool): Enable a lock to protect the update of variable and accumlation tensors. Default: False.
        centered (bool): If True, gradients are normalized by the estimated variance of the gradient. Default: False.
        loss_scale (float): A floating point value for the loss scale. Should be greater than 0. Default: 1.0.
        weight_decay (float): Weight decay (L2 penalty). Should be equal to or greater than 0. Default: 0.0.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        Tensor[bool], the value is True.

    Examples:
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.RMSProp(params=net.trainable_params(), learning_rate=lr)
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> bias_params = list(filter(lambda x: 'bias' in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01},
        >>>                 {'params': bias_params, 'lr': 0.01},
        >>>                 {'order_params': net.trainable_params()}]
        >>> opt = nn.RMSProp(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The conv_params's parameters will use a learning rate of default value 0.1 and a weight decay of 0.01.
        >>> # The bias_params's parameters will use a learning rate of 0.01 and a weight decay of default value 0.0.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>> # The parameters which in the value of 'order_params' but not in any group will use a learning rate
        >>> # of default value 0.1 and a weight decay of default value 0.0.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
    """
    def __init__(self, params, learning_rate=0.1, decay=0.9, momentum=0.0, epsilon=1e-10,
                 use_locking=False, centered=False, loss_scale=1.0, weight_decay=0.0):
        super(RMSProp, self).__init__(learning_rate, params, weight_decay, loss_scale)
        validator.check_value_type("decay", decay, [float], self.cls_name)
        validator.check_number_range("decay", decay, 0.0, float("inf"), Rel.INC_LEFT, self.cls_name)
        validator.check_value_type("momentum", momentum, [float], self.cls_name)
        validator.check_number_range("momentum", momentum, 0.0, float("inf"), Rel.INC_LEFT, self.cls_name)
        validator.check_value_type("epsilon", epsilon, [float], self.cls_name)
        validator.check_number_range("epsilon", epsilon, 0.0, float("inf"), Rel.INC_NEITHER, self.cls_name)
        validator.check_value_type("use_locking", use_locking, [bool], self.cls_name)
        validator.check_value_type("centered", centered, [bool], self.cls_name)

        self.centered = centered
        if centered:
            self.opt = P.ApplyCenteredRMSProp(use_locking)
            self.mg = self.parameters.clone(prefix="mean_grad", init='zeros')
        else:
            self.opt = P.ApplyRMSProp(use_locking)

        self.momentum = momentum
        self.ms = self.parameters.clone(prefix="mean_square", init='zeros')
        self.moment = self.parameters.clone(prefix="moment", init='zeros')
        self.hyper_map = C.HyperMap()
        self.epsilon = epsilon
        self.decay = decay

    def construct(self, gradients):
        params = self.parameters
        gradients = self.decay_weight(gradients)
        gradients = self.scale_grad(gradients)
        lr = self.get_lr()
        if self.centered:
            if self.is_group_lr:
                success = self.hyper_map(F.partial(centered_rmsprop_opt, self.opt, self.decay, self.epsilon,
                                                   self.momentum), lr, params, self.mg, self.ms, self.moment, gradients)
            else:
                success = self.hyper_map(F.partial(centered_rmsprop_opt, self.opt, self.decay, self.epsilon,
                                                   self.momentum, lr), params, self.mg, self.ms, self.moment, gradients)

        else:
            if self.is_group_lr:
                success = self.hyper_map(F.partial(rmsprop_opt, self.opt, self.decay, self.epsilon,
                                                   self.momentum), lr, params, self.ms, self.moment, gradients)
            else:
                success = self.hyper_map(F.partial(rmsprop_opt, self.opt, self.decay, self.epsilon,
                                                   self.momentum, lr), params, self.ms, self.moment, gradients)
        return success
