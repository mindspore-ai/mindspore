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
"""nadam"""
from __future__ import absolute_import

from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common import Parameter, Tensor
import mindspore.common.dtype as mstype
from mindspore import _checkparam as validator
from mindspore.experimental.optim.optimizer import Optimizer, check_not_less_than, check_not_less_than_without_equal
from mindspore import jit

_nadam_opt = C.MultitypeFuncGraph("nadam_opt")

op_sqrt = P.Sqrt()


@_nadam_opt.register("Number", "Number", "Number", "Number", "Tensor", "Tensor", "Tensor",
                     "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt(beta1, beta2, momentum_decay, eps, step_t, lr, param, grad, exp_avg, exp_avg_sq, mu_product):
    """Apply nadam optimizer to the weight parameter."""
    bias_correction2 = 1 - beta2 ** step_t
    mu = beta1 * (1. - 0.5 * (0.96 ** (step_t * momentum_decay)))
    mu_next = beta1 * (1. - 0.5 * (0.96 ** ((step_t + 1) * momentum_decay)))
    F.assign(mu_product, mu_product * mu)
    F.assign(exp_avg, exp_avg * beta1 + grad * (1 - beta1))
    F.assign(exp_avg_sq, exp_avg_sq * beta2 + grad * grad * (1 - beta2))

    denom = op_sqrt(exp_avg_sq / bias_correction2) + eps

    mu_product_next = mu_product * mu_next
    F.assign(param, param - lr * (1. - mu) / (1. - mu_product) * grad / denom)
    F.assign(param, param - (lr * mu_next) / (1. - mu_product_next) * exp_avg / denom)

    return True


class NAdam(Optimizer):
    r"""
    Implements NAdam algorithm.

    .. _Incorporating Nesterov Momentum into Adam:
        https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ

    .. warning::
        This is an experimental optimizer API that is subject to change.
        This module must be used with lr scheduler module in `LRScheduler Class
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#lrscheduler-class>`_ .

    Args:
        params (Union[list(Parameter), list(dict)]): list of parameters to optimize or dicts defining
            parameter groups.
        lr (Union[int, float, Tensor], optional): learning rate. Default: ``2e-3``.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. Default: ``(0.9, 0.999)``.
        eps (float, optional): term added to the denominator to improve
            numerical stability. Default: ``1e-8``.
        weight_decay (float, optional): weight decay (L2 penalty). Default: ``0.``.
        momentum_decay (float, optional): momentum momentum_decay. Default: ``4e-3``.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`.

    Raises:
        ValueError: If the learning rate is not int, float or Tensor.
        ValueError: If the learning rate is less than 0.
        ValueError: If the `eps` is less than 0.0.
        ValueError: If the `weight_decay` is less than 0.
        ValueError: If the `momentum_decay` is less than 0.
        ValueError: If elements of `betas` not in the range of [0, 1).

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
        >>> optimizer = optim.NAdam(net.trainable_params(), lr=0.1)
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

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, momentum_decay=4e-3):
        check_not_less_than_without_equal(lr, "lr", self.cls_name)
        check_not_less_than_without_equal(eps, "eps", self.cls_name)
        check_not_less_than(weight_decay, "weight_decay", self.cls_name)
        check_not_less_than(momentum_decay, "momentum_decay", self.cls_name)

        validator.check_float_range(betas[0], 0., 1., validator.INC_LEFT, "betas[0]", self.cls_name)
        validator.check_float_range(betas[1], 0., 1., validator.INC_LEFT, "betas[1]", self.cls_name)

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, momentum_decay=momentum_decay)
        super(NAdam, self).__init__(params, defaults)
        self.step_t = Parameter(Tensor(0, mstype.int32), "step_t")
        self.exp_avg = self.parameters.clone(prefix="exp_avg", init='zeros')
        self.exp_avg_sq = self.parameters.clone(prefix="exp_avg_sq", init='zeros')
        self.mu_product = [Parameter(Tensor(1.), "mu_product_" + param.name) for param in self.parameters]

        self.increase_tensor = Tensor(1, mstype.int32)
        self.assignadd = P.AssignAdd()
        self.op_cast = P.Cast()

    @jit
    def implementation(self, lr, beta1, beta2, weight_decay, momentum_decay, eps, start_id, end_id, gradients):
        """Extract the common computing part for acceleration"""
        params = self.parameters[start_id: end_id]
        grads = gradients[start_id: end_id]
        grads = self._decay_weight(weight_decay, params, grads)
        exp_avg = self.exp_avg[start_id: end_id]
        exp_avg_sq = self.exp_avg_sq[start_id: end_id]
        mu_product = self.mu_product[start_id: end_id]

        self.hyper_map(F.partial(_nadam_opt, beta1, beta2, momentum_decay, eps, self.step_t, lr),
                       params, grads, exp_avg, exp_avg_sq, mu_product)
        return True

    def construct(self, gradients):
        self.assignadd(self.step_t, self.increase_tensor)
        for group_id, group in enumerate(self.param_groups):

            lr = self.lrs[group_id]
            if isinstance(group.get("lr"), float):
                lr = self.op_cast(group.get("lr"), mstype.float32)

            beta1, beta2 = group["betas"]
            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id + 1]
            weight_decay = group["weight_decay"]
            momentum_decay = group["momentum_decay"]
            eps = group["eps"]
            self.implementation(lr, beta1, beta2, weight_decay, momentum_decay, eps, start_id, end_id, gradients)
        return True
