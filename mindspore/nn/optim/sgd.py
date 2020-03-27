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
"""sgd"""
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore._checkparam import ParamValidator as validator
import mindspore.common.dtype as mstype
from .optimizer import Optimizer, grad_scale

sgd_opt = C.MultitypeFuncGraph("sgd_opt")


@sgd_opt.register("Function", "Number", "Number", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt(opt, learning_rate, momentum, gradient, weight, accum, stat):
    """Apply sgd optimizer to the weight parameter."""
    success = True
    success = F.depend(success, opt(weight, gradient, learning_rate, accum, momentum, stat))
    return success


@sgd_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt_ext(opt, learning_rate, momentum, gradient, weight, accum, stat):
    """Apply sgd optimizer to the weight parameter using Tensor."""
    success = True
    success = F.depend(success, opt(weight, gradient, learning_rate, accum, momentum, stat))
    return success


@sgd_opt.register("Function", "Tensor", "Number", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt_dyn(opt, learning_rate, momentum, gradient, weight, accum, stat):
    """Apply sgd optimizer to the weight parameter using dynamic learning rate."""
    success = True
    success = F.depend(success, opt(weight, gradient, learning_rate, accum, momentum, stat))
    return success


class SGD(Optimizer):
    """
    Implements stochastic gradient descent (optionally with momentum).

    Introduction to SGD can be found at https://en.wikipedia.org/wiki/Stochastic_gradient_descent.
    Nesterov momentum is based on the formula from paper `On the importance of initialization and
    momentum in deep learning <http://proceedings.mlr.press/v28/sutskever13.html>`_.

    Args:
        params (list[Parameter]): A list of parameter, which will be updated. The element in `params`
                                  should be class mindspore.Parameter.
        learning_rate (float): A floating point value for the learning rate. Default: 0.1.
        momentum (float): A floating point value the momentum. Default: 0.
        dampening (float): A floating point value of dampening for momentum. Default: 0.
        weight_decay (float): Weight decay (L2 penalty). Default: 0.
        nesterov (bool): Enables the Nesterov momentum. Default: False.
        loss_scale (float): A floating point value for the loss scale. Default: 1.0.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        Tensor[bool], the value is True.

    Raises:
        ValueError: If the momentum, dampening or weight_decay value is less than 0.0.

    Examples:
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = nn.SGD(params=net.trainable_params())
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
    """
    def __init__(self, params, learning_rate=0.1, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False,
                 loss_scale=1.0):

        super(SGD, self).__init__(learning_rate, params)

        if isinstance(momentum, float) and momentum < 0.0:
            raise ValueError("momentum should be at least 0.0, but got momentum {}".format(momentum))

        if dampening < 0.0:
            raise ValueError("dampening should be at least 0.0, but got dampening {}".format(dampening))
        self.dampening = dampening

        if weight_decay < 0.0:
            raise ValueError("weight_decay should be at least 0.0, but got weight_decay {}".format(weight_decay))
        self.weight_decay = weight_decay

        validator.check_type("nesterov", nesterov, [bool])
        self.nesterov = nesterov

        self.opt = P.SGD(dampening, weight_decay, nesterov)

        self.dynamic_lr = False
        self.gather = None
        self.global_step = None
        self.axis = None
        if not isinstance(learning_rate, float):
            self.dynamic_lr = True
            self.gather = P.GatherV2()
            self.assignadd = P.AssignAdd()
            self.global_step = Parameter(initializer(0, [1], mstype.int32), name="global_step")
            self.axis = 0
        self.momentum = Parameter(momentum, name="momentum")
        self.params = self.parameters
        self.accum = self.params.clone(prefix="accum", init='zeros')
        self.stat = self.params.clone(prefix="stat", init='ones')
        self.hyper_map = C.HyperMap()

        self.weight_decay = weight_decay * loss_scale
        self.reciprocal_scale = 1.0 / loss_scale

    def construct(self, gradients):
        params = self.params
        accum = self.accum
        stat = self.stat
        if self.reciprocal_scale != 1.0:
            gradients = self.hyper_map(F.partial(grad_scale, self.reciprocal_scale), gradients)
        if self.dynamic_lr:
            lr = self.gather(self.learning_rate, self.global_step, self.axis)
            F.control_depend(lr, self.assignadd(self.global_step, 1))
        else:
            lr = self.learning_rate
        success = self.hyper_map(F.partial(sgd_opt, self.opt, lr, self.momentum), gradients, params, accum, stat)
        return success
