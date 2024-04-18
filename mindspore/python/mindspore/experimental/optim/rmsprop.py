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
"""rmsprop"""
from __future__ import absolute_import

from mindspore.ops import functional as F, composite as C, operations as P
import mindspore.common.dtype as mstype
from mindspore.experimental.optim.optimizer import Optimizer, check_not_less_than, check_not_less_than_without_equal
from mindspore import ops
from mindspore import jit

_rmsprop_opt = C.MultitypeFuncGraph("rmsprop_opt")

op_mul = P.Mul()
op_sqrt = P.Sqrt()


@_rmsprop_opt.register("Bool", "Number", "Number", "Number", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _run_rmsprop_opt(centered, alpha, eps, momentum, lr, weight, mean_square, mean_grad, mom, grad):
    """Apply rmsprop optimizer to the weight parameter using dynamic learning rate."""
    F.assign(mean_square, ops.addcmul(op_mul(mean_square, alpha), grad, grad, 1 - alpha))

    if centered:
        F.assign(mean_grad, op_mul(mean_grad, alpha) + op_mul(grad, 1 - alpha))
        avg = op_sqrt(ops.addcmul(mean_square, mean_grad, mean_grad, -1.)) + eps
    else:
        avg = op_sqrt(mean_square) + eps

    if momentum > 0:
        F.assign(mom, op_mul(mom, momentum) + grad / avg)
        F.assign(weight, weight - mom * lr)
    else:
        F.assign(weight, weight - lr * grad / avg)
    return True


class RMSprop(Optimizer):
    r"""
    Implements RMSprop algorithm.

    .. warning::
        This is an experimental optimizer API that is subject to change.
        This module must be used with lr scheduler module in `LRScheduler Class
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#lrscheduler-class>`_ .

    Args:
        params (Union[list(Parameter), list(dict)]): list of parameters to optimize or dicts defining
            parameter groups.
        lr (Union[int, float, Tensor], optional): learning rate. Default: ``1e-2``.
        alpha (float, optional): smoothing constant. Default: ``0.99``.
        eps (float, optional): term added to the denominator to improve numerical stability. Default: ``1e-8``.
        weight_decay (float, optional): weight decay (L2 penalty). Default: ``0.``.
        momentum (float, optional): momentum factor. Default: ``0.``.
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance. Default: ``False``.
        maximize (bool, optional): maximize the params based on the objective, instead of minimizing.
            Default: ``False``.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`.

    Raises:
        ValueError: If the learning rate is not int, float or Tensor.
        ValueError: If the learning rate is less than 0.
        ValueError: If the `momentum` is less than 0.0.
        ValueError: If the `alpha` is less than 0.0.
        ValueError: If the `eps` is less than 0.0.
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
        >>> optimizer = optim.RMSprop(net.trainable_params(), lr=0.1)
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

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0,
                 centered=False, maximize=False):
        check_not_less_than_without_equal(lr, "lr", self.cls_name)
        check_not_less_than(alpha, "alpha", self.cls_name)
        check_not_less_than_without_equal(eps, "eps", self.cls_name)
        check_not_less_than(momentum, "momentum", self.cls_name)
        check_not_less_than(weight_decay, "weight_decay", self.cls_name)

        defaults = dict(
            lr=lr,
            momentum=momentum,
            alpha=alpha,
            eps=eps,
            centered=centered,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super(RMSprop, self).__init__(params, defaults)

        self.mean_grad = self.parameters.clone(prefix="mean_grad", init='zeros')
        self.mean_square = self.parameters.clone(prefix="mean_square", init='zeros')
        self.moment = self.parameters.clone(prefix="moment", init='zeros')
        self.op_cast = P.Cast()

    @jit
    def implementation(self, group_id, lr, gradients, maximize, weight_decay, centered, alpha, eps, momentum):
        """Extract the common computing part for acceleration"""
        start_id = self.group_start_id[group_id]
        end_id = self.group_start_id[group_id + 1]
        params = self.parameters[start_id: end_id]
        grads = tuple([grad if not maximize else F.neg(grad) for grad in gradients[start_id: end_id]])
        grads = self._decay_weight(weight_decay, params, grads)
        mean_grad = self.mean_grad[start_id: end_id]
        mean_square = self.mean_square[start_id: end_id]
        moment = self.moment[start_id: end_id]
        self.hyper_map(F.partial(_rmsprop_opt, centered, alpha, eps, momentum, lr),
                       params, mean_square, mean_grad, moment, grads)
        return True

    def construct(self, gradients):
        for group_id, group in enumerate(self.param_groups):
            lr = self.lrs[group_id]
            if isinstance(group.get("lr"), float):
                lr = self.op_cast(group.get("lr"), mstype.float32)
            maximize = group.get("maximize")

            self.implementation(group_id, lr, gradients, maximize, group["weight_decay"], group["centered"],
                                group["alpha"], group["eps"],
                                group["momentum"])
        return True
