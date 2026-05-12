# Copyright 2024 Huawei Technologies Co., Ltd
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
"""SGD"""
from __future__ import absolute_import

from mindspore import ops
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.experimental.optim.optimizer import Optimizer
from mindspore import _checkparam as validator
from mindspore import mint
hyper_map = ops.HyperMap()


def _tensor_weight_decay(weight_decay, weight, gradient):
    """Get grad with weight_decay."""
    return mint.add(gradient, weight, alpha=weight_decay)


def _run_optim_sgd_opt(step_t, lr, momentum, dampening, nesterov, params, momentum_buf, d_p):
    """Apply sgd optimizer."""
    success = True
    if momentum != 0:
        buf = momentum_buf
        if step_t == 0:
            buf.copy_(d_p)
            momentum_buf.copy_(buf)
        else:
            buf.mul_(momentum)
            buf.add_(d_p, alpha=1-dampening)
        if nesterov:
            d_p = mint.add(d_p, buf, alpha=momentum)
        else:
            d_p = buf
    params.add_(d_p, alpha=-lr)
    return success


def _check_param_value(lr, momentum, weight_decay, dampening, nesterov, maximize, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("lr", lr, [float, int, bool, Tensor], prim_name)
    validator.check_value_type("momentum", momentum, [float, int, bool], prim_name)
    validator.check_value_type("weight_decay", weight_decay, [float, int, bool], prim_name)
    validator.check_value_type("dampening", dampening, [float, int, bool], prim_name)
    validator.check_value_type("nesterov", nesterov, [bool], prim_name)
    validator.check_value_type("maximize", maximize, [bool], prim_name)


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

    Here : p, v and u denote the parameters, accum, and momentum respectively.

    .. warning::
        This is an experimental optimizer API, which may be modified or removed in the future.
        This module must be used with lr scheduler module in `LRScheduler Class
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#lrscheduler-class>`_ .

    Args:
        params (Union[list(Parameter), list(dict)]): list of parameters to optimize or dicts defining
            parameter groups.
        lr (Union[bool, int, float, Tensor]): learning rate.
        momentum (Union[bool, int, float], optional): momentum factor. Default: ``0``.
        weight_decay (Union[bool, int, float], optional): weight decay (L2 penalty). Must be greater than or equal to 0.
            Default: ``0.``.
        dampening (Union[bool, int, float], optional): dampening for momentum. Default: ``0``.
        nesterov (bool, optional): enable Nesterov momentum. If Nesterov is utilized, the momentum must be positive,
            and the damping must be equal to 0. Default: ``False``.

    Keyword Args:
        maximize (bool, optional): maximize the params based on the objective, instead of minimizing.
            Default: ``False``.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`.

    Raises:
        ValueError: If the learning rate is not bool, int, float or Tensor.
        ValueError: If the learning rate is less than 0.
        ValueError: If the `momentum` or `weight_decay` value is less than 0.0.
        ValueError: If the `momentum`, `dampening` or `weight_decay` value is not bool, int or float.
        ValueError: If the `nesterov` and `maximize` are not bool.
        ValueError: If the `nesterov` is True, `momentum` is not positive or `dampening` is not 0.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import nn
        >>> from mindspore.mint import optim
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://atomgit.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
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

    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False):
        _check_param_value(lr, momentum, weight_decay, dampening, nesterov, maximize, self.cls_name)
        if lr < 0.:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening, nesterov=nesterov,
                        maximize=maximize, lr_float=lr)
        for param in params:
            if isinstance(param, dict) and param.get('lr', None):
                param['lr_float'] = param['lr']
        super(SGD, self).__init__(params, defaults)

        if nesterov and (momentum <= 0. or dampening != 0.):
            raise ValueError("For 'SGD', if 'nesterov' is true, 'momentum' must be > 0.0 and 'dampening' must "
                             "equal to 0.0, but got 'momentum' {}, 'dampening' {}".format(momentum, dampening))
        self.momentum_buf = self.parameters.clone(prefix='momentum_buf', init='zeros')
        self.step_t = Parameter(Tensor(0, mstype.int32), "step_t")
        self.increase_tensor = Tensor(1, mstype.int32)

    def construct(self, gradients):
        for group_id, group in enumerate(self.param_groups):
            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id + 1]
            grads = tuple([grad if not group.get('maximize') else -grad for grad in gradients[start_id: end_id]])
            if group.get('weight_decay') != 0.:
                grads = self.map_(ops.partial(_tensor_weight_decay, group.get('weight_decay')),
                                  self.parameters[start_id: end_id], grads)
            self.hyper_map(ops.partial(_run_optim_sgd_opt, self.step_t.value(), group.get('lr_float'),
                                       group.get('momentum'), group.get('dampening'), group.get('nesterov')),
                           self.parameters[start_id: end_id], self.momentum_buf[start_id: end_id], grads)
        self.step_t.add_(self.increase_tensor)
        return True
