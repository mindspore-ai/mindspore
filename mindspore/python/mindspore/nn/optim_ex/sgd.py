# Copyright 2023 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import

from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore import _checkparam as Validator
from mindspore.nn.optim_ex.optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    .. math::
            v_{t+1} = u \ast v_{t} + gradient \ast (1-dampening)

    If nesterov is True:

    .. math::
            p_{t+1} = p_{t} - lr \ast (gradient + u \ast v_{t+1})

    If nesterov is False:

    .. math::
            p_{t+1} = p_{t} - lr \ast v_{t+1}

    To be noticed, for the first step, :math:`v_{t+1} = gradient`.

    Here : where p, v and u denote the parameters, accum, and momentum respectively.

    Args:
        params (Union[list(Parameter), list(dict)]): list of parameters to optimize or dicts defining
            parameter groups
        lr (Union[int, float, Tensor]): learning rate
        momentum (Union(int, float), optional): momentum factor. Default: 0.
        weight_decay (float, optional): weight decay (L2 penalty). Default: 0.
        dampening (Union(int, float), optional): dampening for momentum. Default: 0.
        nesterov (bool, optional): enables Nesterov momentum. Default: False.
        maximize (bool, optional): maximize the params based on the objective, instead of minimizing.
            Default: False.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`.

    Raises:
        ValueError: If the learning rate is not int, float or Tensor.
        ValueError: If the learning rate is less than 0.
        ValueError: If the momentum or weight_decay value is less than 0.0.
        ValueError: If the momentum, dampening or weight_decay value is not int or float.
        ValueError: If the nesterov and maximize is not bool.
        ValueError: If the nesterov is true, momentum is not positive or dampening is not 0.0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = Net()
        >>> loss_fn = nn.MAELoss()
        >>> optimizer = mindspore.nn.optim_ex.SGD(model.parameters(), lr=0.1, momentum=0.9)

        >>> def forward_fn(data, label):
        ...     logits = net(data)
        ...     loss = loss_fn(logits, label)
        >>> grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        >>> def train_step(data, label):
        ...     (loss, _), grads = grad_fn(data, label)
        ...     optimizer(grads)
        ...     return loss
        >>> loss = train_step(data, label)
    """
    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, *,
                 maximize=False):
        Validator.check_value_type("lr", lr, [float, int, Tensor], self.cls_name)
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        Validator.check_value_type("momentum", momentum, [int, float], self.cls_name)
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        momentum = float(momentum)
        Validator.check_value_type("dampening", dampening, [int, float], self.cls_name)
        dampening = float(dampening)
        Validator.check_value_type("nesterov", nesterov, [bool], self.cls_name)
        Validator.check_value_type("maximize", maximize, [bool], self.cls_name)

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, grad_centralization=False)
        super(SGD, self).__init__(params, defaults)

        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError("For 'SGD', if 'nesterov' is true, 'momentum' must be > 0.0 and 'dampening' must "
                             "equal to 0.0, but got 'momentum' {}, 'dampening' {}".format(momentum, dampening))
        self.accum = self.parameters.clone(prefix="accum", init='zeros')
        self.stat = self.parameters.clone(prefix="stat", init='ones')
        self.op_cast = P.Cast()

    def construct(self, gradients):
        for group_id, group in enumerate(self.param_groups):
            params = []
            grads = []
            accums = []
            stats = []
            params, grads, accums, stats = self._init_group(group, gradients, params, grads,
                                                            accums, stats, group_id)
            opt = P.SGD(group["dampening"], group["weight_decay"], group["nesterov"])
            lr = self.lrs[group_id]
            momentum = self.op_cast(group["momentum"], mstype.float32)
            self.apply_sgd(opt, params, grads, lr, accums, momentum, stats, group["maximize"],
                           group["grad_centralization"])

    def apply_sgd(self, opt, params, grads, lr, accums, momentum, stats, maximize, grad_centralization):
        grads = self._gradients_centralization(grad_centralization, grads)

        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            opt(param, grad, lr, accums[i], momentum, stats[i])

    def _init_group(self, group, gradients, params, accums, grads, stats, group_id):
        p_id = self.group_start_id[group_id]
        for i, param in enumerate(group["params"]):
            params.append(param)
            grads.append(gradients[p_id+i])
            accums.append(self.accum[p_id+i])
            stats.append(self.stat[p_id+i])
        return params, grads, accums, stats
