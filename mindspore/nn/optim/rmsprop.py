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
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore._checkparam import ParamValidator as validator
import mindspore.common.dtype as mstype
from mindspore.common import Tensor
from .optimizer import Optimizer, grad_scale, apply_decay

rmsprop_opt = C.MultitypeFuncGraph("rmsprop_opt")
centered_rmsprop_opt = C.MultitypeFuncGraph("rmsprop_opt")


@rmsprop_opt.register("Function", "Number", "Number", "Number", "Number", "Tensor", "Tensor", "Tensor", "Tensor")
def _rmsprop_opt(opt, learning_rate, decay, epsilon, momentum, weight, ms, mom, grad):
    """Apply rmsprop optimizer to the weight parameter."""
    success = True
    success = F.depend(success, opt(weight, ms, mom, grad, learning_rate, decay, momentum, epsilon))
    return success


@rmsprop_opt.register("Function", "Tensor", "Number", "Number", "Number", "Tensor", "Tensor", "Tensor", "Tensor")
def _rmsprop_opt_dynamic_lr(opt, learning_rate, decay, epsilon, momentum, weight, ms, mom, grad):
    """Apply rmsprop optimizer to the weight parameter using dynamic learning rate."""
    success = True
    success = F.depend(success, opt(weight, ms, mom, grad, learning_rate, decay, momentum, epsilon))
    return success


@centered_rmsprop_opt.register("Function", "Number", "Number", "Number", "Number", "Tensor", "Tensor", "Tensor",
                               "Tensor", "Tensor")
def _centered_rmsprop_opt(opt, learning_rate, decay, epsilon, momentum, weight, mg, ms, mom, grad):
    """Apply centered rmsprop optimizer to the weight parameter."""
    success = True
    success = F.depend(success, opt(weight, mg, ms, mom, grad, learning_rate, decay, momentum, epsilon))
    return success


@centered_rmsprop_opt.register("Function", "Tensor", "Number", "Number", "Number", "Tensor", "Tensor", "Tensor",
                               "Tensor", "Tensor")
def _centered_rmsprop_opt_dynamic_lr(opt, learning_rate, decay, epsilon, momentum, weight, mg, ms, mom, grad):
    """Apply centered rmsprop optimizer to the weight parameter using dynamic learning rate."""
    success = True
    success = F.depend(success, opt(weight, mg, ms, mom, grad, learning_rate, decay, momentum, epsilon))
    return success


class RMSProp(Optimizer):
    """
    Implements Root Mean Squared Propagation (RMSProp) algorithm.

    Note:
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
        params (list[Parameter]): A list of parameter, which will be updated. The element in `parameters`
                                  should be class mindspore.Parameter.
        learning_rate (Union[float, Tensor, Iterable]): A value for the learning rate. When the learning_rate is
                                                        Iterable or a Tensor and the dims of the Tensor is 1,
                                                        use dynamic learning rate, then the i-th step will
                                                        take the i-th value as the learning rate.
                                                        When the learning_rate is float or learning_rate is a Tensor
                                                        but the dims of the Tensor is 0, use fixed learning rate.
                                                        Other cases are not supported.
        decay (float): Decay rate.
        momentum (float): Hyperparameter of type float, means momentum for the moving average.
        epsilon (float): Term added to the denominator to improve numerical stability. Should be greater than 0.
        use_locking (bool): Enable a lock to protect the update of variable and accumlation tensors. Default: False.
        centered (bool): If True, gradients are normalized by the estimated variance of the gradient. Default: False
        loss_scale (float): A floating point value for the loss scale. Default: 1.0.
        weight_decay (float): Weight decay (L2 penalty). Default: 0.0.
        decay_filter (Function): A function to determine whether to apply weight decay on parameters. Default:
                                 lambda x: 'beta' not in x.name and 'gamma' not in x.name.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        Tensor[bool], the value is True.

    Examples:
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> opt = nn.RMSProp(params=net.trainable_params(), learning_rate=lr)
        >>> model = Model(net, loss, opt)
    """
    def __init__(self, params, learning_rate=0.1, decay=0.9, momentum=0.0, epsilon=1e-10,
                 use_locking=False, centered=False, loss_scale=1.0, weight_decay=0.0,
                 decay_filter=lambda x: 'beta' not in x.name and 'gamma' not in x.name):
        super(RMSProp, self).__init__(learning_rate, params)

        if isinstance(momentum, float) and momentum < 0.0:
            raise ValueError("momentum should be at least 0.0, but got momentum {}".format(momentum))

        if decay < 0.0:
            raise ValueError("decay should be at least 0.0, but got dampening {}".format(decay))
        self.decay = decay
        self.epsilon = epsilon

        validator.check_type("use_locking", use_locking, [bool])
        validator.check_type("centered", centered, [bool])
        self.centered = centered
        if centered:
            self.opt = P.ApplyCenteredRMSProp(use_locking)
            self.mg = self.parameters.clone(prefix="mean_grad", init='zeros')
        else:
            self.opt = P.ApplyRMSProp(use_locking)

        self.dynamic_lr = False
        if not isinstance(learning_rate, float):
            self.dynamic_lr = True
            self.gather = P.GatherV2()
            self.assignadd = P.AssignAdd()
            self.global_step = Parameter(initializer(0, [1], mstype.int32), name="global_step")
            self.axis = 0
            self.one = Tensor(1, mstype.int32)

        self.momentum = momentum

        self.ms = self.parameters.clone(prefix="mean_square", init='zeros')
        self.moment = self.parameters.clone(prefix="moment", init='zeros')
        self.hyper_map = C.HyperMap()

        self.decay = decay
        self.decay_tf = tuple(decay_filter(x) for x in self.parameters)
        self.reciprocal_scale = 1.0 / loss_scale
        self.weight_decay = weight_decay * loss_scale

    def construct(self, gradients):
        params = self.parameters
        if self.weight_decay > 0:
            gradients = self.hyper_map(F.partial(apply_decay, self.weight_decay), self.decay_tf, params, gradients)
        if self.reciprocal_scale != 1.0:
            gradients = self.hyper_map(F.partial(grad_scale, self.reciprocal_scale), gradients)
        if self.dynamic_lr:
            lr = self.gather(self.learning_rate, self.global_step, self.axis)
            F.control_depend(lr, self.assignadd(self.global_step, self.one))
        else:
            lr = self.learning_rate
        if self.centered:
            success = self.hyper_map(F.partial(centered_rmsprop_opt, self.opt, lr, self.decay, self.epsilon,
                                               self.momentum), params, self.mg, self.ms, self.moment, gradients)
        else:
            success = self.hyper_map(F.partial(rmsprop_opt, self.opt, lr, self.decay, self.epsilon,
                                               self.momentum), params, self.ms, self.moment, gradients)
        return success
