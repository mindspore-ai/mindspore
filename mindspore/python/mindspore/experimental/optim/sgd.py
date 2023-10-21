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

from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore import _checkparam as Validator
from mindspore.experimental.optim.optimizer import Optimizer

_sgd_opt = C.MultitypeFuncGraph("sgd_opt")


@_sgd_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",)
def _tensor_run_opt_ext(opt, momentum, learning_rate, gradient, weight, accum, stat):
    """Apply sgd optimizer to the weight parameter using Tensor."""
    success = True
    success = F.depend(success, opt(weight, gradient, learning_rate, accum, momentum, stat))
    return success


class SGD(Optimizer):
    r"""
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

    .. warning::
        This is an experimental optimizer API that is subject to change.
        This module must be used with lr scheduler module in `LRScheduler Class
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#lrscheduler-class>`_ .

    Args:
        params (Union[list(Parameter), list(dict)]): list of parameters to optimize or dicts defining
            parameter groups.
        lr (Union[int, float, Tensor]): learning rate.
        momentum (Union[int, float], optional): momentum factor. Default: ``0``.
        weight_decay (float, optional): weight decay (L2 penalty). Default: ``0``.
        dampening (Union[int, float], optional): dampening for momentum. Default: ``0``.
        nesterov (bool, optional): enables Nesterov momentum. Default: ``False``.

    Keyword Args:
        maximize (bool, optional): maximize the params based on the objective, instead of minimizing.
            Default: ``False``.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`.

    Raises:
        ValueError: If the learning rate is not int, float or Tensor.
        ValueError: If the learning rate is less than 0.
        ValueError: If the `momentum` or `weight_decay` value is less than 0.0.
        ValueError: If the `momentum`, `dampening` or `weight_decay` value is not int or float.
        ValueError: If the `nesterov` and `maximize` is not bool.
        ValueError: If the `nesterov` is true, `momentum` is not positive or `dampening` is not 0.0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import nn
        >>> from mindspore.experimental import optim
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        >>> optimizer = optim.SGD(net.trainable_params(), lr=0.1)
        >>> def forward_fn(data, label):
        ...     logits = net(data)
        ...     loss = loss_fn(logits, label)
        ...     return loss, logits
        >>> grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        >>> def train_step(data, label):
        ...     (loss, _), grads = grad_fn(data, label)
        ...     optimizer(grads)
        ...     return loss
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
        Validator.check_value_type("nesterov", nesterov, [bool], self.cls_name)
        Validator.check_value_type("maximize", maximize, [bool], self.cls_name)

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, grad_centralization=False)
        super(SGD, self).__init__(params, defaults)
        for group in self.param_groups:
            Validator.check_value_type("dampening", group.get("dampening"), [int, float], self.cls_name)
            group["dampening"] = float(group.get("dampening"))
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError("For 'SGD', if 'nesterov' is true, 'momentum' must be > 0.0 and 'dampening' must "
                             "equal to 0.0, but got 'momentum' {}, 'dampening' {}".format(momentum, dampening))
        self.accum = self.parameters.clone(prefix="accum", init='zeros')
        self.stat = self.parameters.clone(prefix="stat", init='ones')
        self.op_cast = P.Cast()

    def construct(self, gradients):
        for group_id, group in enumerate(self.param_groups):
            opt = P.SGD(group.get("dampening"), group.get("weight_decay"), group.get("nesterov"))
            lr = group.get("lr")
            if isinstance(lr, float):
                lr = self.op_cast(group.get("lr"), mstype.float32)
            maximize = group.get("maximize")
            momentum = self.op_cast(group.get("momentum"), mstype.float32)
            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id+1]
            grads = gradients[start_id: end_id] if not maximize else -gradients[start_id: end_id]
            self.hyper_map(F.partial(_sgd_opt, opt, momentum, lr), grads,
                           self.parameters[start_id: end_id], self.accum[start_id: end_id],
                           self.stat[start_id: end_id])
        return True
