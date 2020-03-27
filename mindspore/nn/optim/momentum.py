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
"""momentum"""
from typing import Iterable

from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
import mindspore.common.dtype as mstype
from mindspore.common import Tensor
from .optimizer import Optimizer, apply_decay, grad_scale

momentum_opt = C.MultitypeFuncGraph("momentum_opt")


@momentum_opt.register("Function", "Number", "Number", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt(opt, learning_rate, momentum, gradient, weight, moment):
    """Apply momentum optimizer to the weight parameter."""
    success = True
    success = F.depend(success, opt(weight, moment, learning_rate, gradient, momentum))
    return success


@momentum_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt_ext(opt, learning_rate, momentum, gradient, weight, moment):
    """Apply momentum optimizer to the weight parameter using Tensor."""
    success = True
    success = F.depend(success, opt(weight, moment, learning_rate, gradient, momentum))
    return success


@momentum_opt.register("Function", "Tensor", "Number", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt_dyn(opt, learning_rate, momentum, gradient, weight, moment):
    """Apply momentum optimizer to the weight parameter using dynamic learning rate."""
    success = True
    success = F.depend(success, opt(weight, moment, learning_rate, gradient, momentum))
    return success


class Momentum(Optimizer):
    """
    Implements the Momentum algorithm.

    Refer to the paper on the importance of initialization and momentum in deep learning for more details.

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
        momentum (float): Hyperparameter of type float, means momentum for the moving average.
        weight_decay (float): Weight decay (L2 penalty). Default: 0.0.
        loss_scale (float): A floating point value for the loss scale. Default: 1.0.
        decay_filter (Function): A function to determine whether to apply weight decay on parameters. Default:
                                 lambda x: 'beta' not in x.name and 'gamma' not in x.name.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        Tensor[bool], the value is True.

    Raises:
        ValueError: If the momentum is less than 0.0.

    Examples:
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
    """
    def __init__(self, params, learning_rate, momentum, weight_decay=0.0, loss_scale=1.0,
                 decay_filter=lambda x: 'beta' not in x.name and 'gamma' not in x.name):
        super(Momentum, self).__init__(learning_rate, params)
        if isinstance(momentum, float) and momentum < 0.0:
            raise ValueError("momentum should be at least 0.0, but got momentum {}".format(momentum))
        if isinstance(learning_rate, Iterable) or \
                (isinstance(learning_rate, Tensor) and learning_rate.dim() == 1):
            self.dynamic_lr = True
            self.gather = P.GatherV2()
            self.assignadd = P.AssignAdd()
            self.global_step = Parameter(initializer(0, [1], mstype.int32), name="global_step")
            self.axis = 0
        else:
            self.dynamic_lr = False
            self.gather = None
            self.assignadd = None
            self.global_step = None
            self.axis = None
        self.momentum = Parameter(momentum, name="momentum")
        self.params = self.parameters
        self.moments = self.params.clone(prefix="moments", init='zeros')
        self.decay_tf = tuple(decay_filter(x) for x in self.parameters)
        self.hyper_map = C.HyperMap()
        self.opt = P.ApplyMomentum()
        self.weight_decay = weight_decay * loss_scale
        self.reciprocal_scale = 1.0 / loss_scale
        self.one = Tensor(1, mstype.int32)

    def construct(self, gradients):
        params = self.params
        moments = self.moments
        if self.weight_decay > 0:
            gradients = self.hyper_map(F.partial(apply_decay, self.weight_decay), self.decay_tf, params, gradients)
        if self.reciprocal_scale != 1.0:
            gradients = self.hyper_map(F.partial(grad_scale, self.reciprocal_scale), gradients)
        if self.dynamic_lr:
            lr = self.gather(self.learning_rate, self.global_step, self.axis)
            F.control_depend(lr, self.assignadd(self.global_step, self.one))
        else:
            lr = self.learning_rate
        success = self.hyper_map(F.partial(momentum_opt, self.opt, lr, self.momentum), gradients, params, moments)
        return success
