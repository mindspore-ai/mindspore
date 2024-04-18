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
"""adagrad"""
from __future__ import absolute_import

from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common import Tensor, Parameter
import mindspore.common.dtype as mstype
from mindspore.experimental.optim.optimizer import Optimizer, check_not_less_than, check_not_less_than_without_equal
from mindspore import jit

_adagrad_opt = C.MultitypeFuncGraph("adagrad_opt")


@_adagrad_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt(opt, learning_rate, weight, accum, gradient):
    """Apply adagrad optimizer to the weight parameter."""
    success = True
    success = F.depend(success, opt(weight, accum, learning_rate, gradient))
    return success


class Adagrad(Optimizer):
    r"""
    Implements Adagrad algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{12mm}    \tau \text{ (initial accumulator value)}, \: \eta\text{ (lr decay)}\\
            &\textbf{initialize} :  state\_sum_0 \leftarrow 0                             \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \tilde{\gamma}    \leftarrow \gamma / (1 +(t-1) \eta)                  \\
            &\hspace{5mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda \theta_{t-1}                             \\
            &\hspace{5mm}state\_sum_t  \leftarrow  state\_sum_{t-1} + g^2_t                      \\
            &\hspace{5mm}\theta_t \leftarrow
                \theta_{t-1}- \tilde{\gamma} \frac{g_t}{\sqrt{state\_sum_t}+\epsilon}            \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    .. warning::
        This is an experimental optimizer API that is subject to change.
        This module must be used with lr scheduler module in `LRScheduler Class
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#lrscheduler-class>`_ .

    Args:
        params (Union[list(Parameter), list(dict)]): list of parameters to optimize or dicts defining
            parameter groups.
        lr (Union[int, float, Tensor], optional): learning rate. Default: ``1e-2``.
        lr_decay (float, optional): learning rate decay. Default: ``0.``.
        weight_decay (float, optional): weight decay (L2 penalty). Default: ``0.``.
        initial_accumulator_value (float, optional): the initial accumulator value. Default: ``0.``.
        eps (float, optional): term added to the denominator to improve
            numerical stability. Default: ``1e-10``.

    Keyword Args:
        maximize (bool, optional): maximize the params based on the objective, instead of minimizing.
            Default: ``False``.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`.

    Raises:
        ValueError: If the learning rate is not int, float or Tensor.
        ValueError: If the learning rate is less than 0.
        ValueError: If the learning rate decay is less than 0.
        ValueError: If the `weight_decay` is less than 0.
        ValueError: If the `initial_accumulator_value` is less than 0.0.
        ValueError: If the `eps` is less than 0.0.

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
        >>> optimizer = optim.Adagrad(net.trainable_params(), lr=0.1)
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

    def __init__(self, params, lr=1e-2, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0,
                 eps=1e-10, *, maximize=False):
        check_not_less_than_without_equal(lr, "lr", self.cls_name)
        check_not_less_than(lr_decay, "lr_decay", self.cls_name)
        check_not_less_than(weight_decay, "weight_decay", self.cls_name)
        check_not_less_than(initial_accumulator_value, "initial_accumulator_value", self.cls_name)
        check_not_less_than_without_equal(eps, "eps", self.cls_name)

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            maximize=maximize,
        )
        super(Adagrad, self).__init__(params, defaults)

        self.accum = self.parameters.clone(prefix="accum", init=initial_accumulator_value)
        self.op_cast = P.Cast()
        self.step_t = Parameter(Tensor(0, mstype.int32), "step_t")
        self.increase_tensor = Tensor(1, mstype.int32)
        self.assignadd = P.AssignAdd()
        self.assign = P.Assign()

    @jit
    def implementation(self, eps, lr, lr_decay, maximize, weight_decay, start_id, end_id, gradients):
        """Extract the common computing part for acceleration"""
        opt = P.ApplyAdagradV2(epsilon=eps, update_slots=True)
        decay_lr = lr / (1 + self.step_t * lr_decay)
        params = self.parameters[start_id: end_id]
        grads = tuple([grad if not maximize else F.neg(grad) for grad in gradients[start_id: end_id]])
        grads = self._decay_weight(weight_decay, params, grads)
        accum = self.accum[start_id: end_id]
        self.hyper_map(F.partial(_adagrad_opt, opt, decay_lr), params, accum, grads)
        return True

    def construct(self, gradients):
        for group_id, group in enumerate(self.param_groups):

            lr = self.lrs[group_id]
            if isinstance(group.get("lr"), float):
                lr = self.op_cast(group.get("lr"), mstype.float32)

            lr_decay = group["lr_decay"]
            maximize = group.get("maximize")
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id + 1]
            self.implementation(eps, lr, lr_decay, maximize, weight_decay, start_id, end_id, gradients)

        self.assignadd(self.step_t, self.increase_tensor)

        return True
