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

from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common import Tensor, Parameter
import mindspore.common.dtype as mstype
from mindspore.experimental.optim.optimizer import Optimizer, check_not_less_than, check_not_less_than_without_equal
from mindspore.common.api import jit

_asgd_opt = C.MultitypeFuncGraph("asgd_opt")

op_cast = P.Cast()
op_pow = P.Pow()
op_maximum = P.Maximum()
op_assign = P.Assign()
op_assignadd = P.AssignAdd()


@_asgd_opt.register("Number", "Number", "Number", "Tensor", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Tensor", "Tensor")
def _run_asgd_opt(lambd, alpha, t0, step, lr, param, grad, eta, mu, ax):
    """Apply asgd optimizer to the weight parameter using dynamic learning rate."""
    if step == 1:
        op_assign(eta, lr)
    next_param = op_cast(param * (1. - lambd * eta) - eta * grad, param.dtype)
    F.assign(param, next_param)

    if mu != 1:
        op_assignadd(ax, op_cast((next_param - ax) * mu, ax.dtype))
    else:
        op_assign(ax, next_param)

    op_assign(eta, lr / (op_pow((1. + lambd * lr * step), alpha)))
    op_assign(mu, 1. / op_maximum(1., step - t0))
    return True


class ASGD(Optimizer):
    r"""
    Implements Averaged Stochastic Gradient Descent algorithm.

    .. warning::
        This is an experimental optimizer API that is subject to change.
        This module must be used with lr scheduler module in `LRScheduler Class
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#lrscheduler-class>`_ .

    Args:
        params (Union[list(Parameter), list(dict)]): list of parameters to optimize or dicts defining
            parameter groups.
        lr (Union[int, float, Tensor], optional): learning rate. Default: ``1e-2``.
        lambd (float, optional): decay term. Default: ``1e-4``.
        alpha (float, optional): power for eta update. Default: ``0.75``.
        t0 (float, optional): point at which to start averaging. Default: ``1e6``.
        weight_decay (float, optional): weight decay (L2 penalty). Default: ``0.``.
        maximize (bool, optional): maximize the params based on the objective, instead of minimizing.
            Default: ``False``.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`.

    Raises:
        ValueError: If the learning rate is not int, float or Tensor.
        ValueError: If the learning rate is less than 0.
        ValueError: If the `weight_decay` is less than 0.

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
        >>> optimizer = optim.ASGD(net.trainable_params(), lr=0.1)
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

    def __init__(self, params, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0.0, maximize=False):
        check_not_less_than_without_equal(lr, "lr", self.cls_name)
        check_not_less_than(weight_decay, "weight_decay", self.cls_name)
        if not isinstance(lambd, float):
            raise TypeError(f"For 'ASGD', the type of lambd must be float, but got {type(lambd)}.")
        if not isinstance(t0, float):
            raise TypeError(f"For 'ASGD', the type of t0 must be float, but got {type(t0)}.")
        defaults = dict(
            lr=lr,
            lambd=lambd,
            alpha=alpha,
            t0=t0,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super(ASGD, self).__init__(params, defaults)
        self.mu = [Parameter(Tensor(1.), "mu_" + param.name) for param in self.parameters]
        self.eta = [Parameter(Tensor(0.), "eta_" + param.name) for param in self.parameters]
        self.ax = self.parameters.clone(prefix="ax", init='zeros')
        self.step_t = Parameter(Tensor(0, mstype.int32), "step_t")
        self.increase_tensor = Tensor(1, mstype.int32)
        self.assignadd = P.AssignAdd()
        self.op_cast = P.Cast()

    @jit
    def implementation(self, lambd, alpha, t0, lr, group_id, maximize, gradients, weight_decay):
        """Extract the common computing part for acceleration"""
        start_id = self.group_start_id[group_id]
        end_id = self.group_start_id[group_id + 1]
        params = self.parameters[start_id: end_id]
        grads = tuple([grad if not maximize else F.neg(grad) for grad in gradients[start_id: end_id]])
        grads = self._decay_weight(weight_decay, params, grads)

        ax = self.ax[start_id: end_id]
        eta = self.eta[start_id: end_id]
        mu = self.mu[start_id: end_id]
        self.hyper_map(F.partial(_asgd_opt, lambd, alpha, t0, self.step_t, lr),
                       params, grads, eta, mu, ax)
        return True

    def construct(self, gradients):
        self.assignadd(self.step_t, self.increase_tensor)
        for group_id, group in enumerate(self.param_groups):
            lr = self.lrs[group_id]
            if isinstance(group.get("lr"), float):
                lr = self.op_cast(group.get("lr"), mstype.float32)
            maximize = group.get("maximize")

            self.implementation(group["lambd"], group["alpha"], group["t0"], lr, group_id, maximize, gradients,
                                group["weight_decay"])
        return True
