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
"""adadelta"""
from __future__ import absolute_import

from mindspore.ops import functional as F, composite as C, operations as P
import mindspore.common.dtype as mstype
from mindspore.experimental.optim.optimizer import Optimizer, check_not_less_than, check_not_less_than_without_equal
from mindspore import _checkparam as validator
from mindspore import jit

_adadelta_opt = C.MultitypeFuncGraph("adadelta_opt")


@_adadelta_opt.register("Function", "Number", "Number", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt(opt, rho, epsilon, learning_rate, weight, accum, accum_update, gradient):
    """Apply adadelta optimizer to the weight parameter."""
    success = True
    success = F.depend(success, opt(weight, accum, accum_update, learning_rate, rho, epsilon, gradient))
    return success


class Adadelta(Optimizer):
    r"""
    Implements Adadelta algorithm.

    .. math::
       \begin{aligned}
            &\rule{150mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)},
                \: f(\theta) \text{ (objective)}, \: \rho \text{ (decay)},
                \: \lambda \text{ (weight decay)}                                                \\
            &\textbf{initialize} :  v_0  \leftarrow 0 \: \text{ (square avg)},
                \: u_0 \leftarrow 0 \: \text{ (accumulate variables)}                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm} v_t      \leftarrow v_{t-1} \rho + g^2_t (1 - \rho)                    \\
            &\hspace{5mm}\Delta x_t    \leftarrow   \frac{\sqrt{u_{t-1} +
                \epsilon }}{ \sqrt{v_t + \epsilon}  }g_t \hspace{21mm}                           \\
            &\hspace{5mm} u_t  \leftarrow   u_{t-1}  \rho +
                 \Delta x^2_t  (1 - \rho)                                                        \\
            &\hspace{5mm}\theta_t      \leftarrow   \theta_{t-1} - \gamma  \Delta x_t            \\
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
        lr (Union[int, float, Tensor], optional): learning rate. Default: ``1.0``.
        rho (float, optional): coefficient used for computing a running average
            of squared gradients. :math:`\rho` in the formula above. Default: ``0.9``.
        eps (float, optional): term added to the denominator to improve
            numerical stability. :math:`\epsilon` in the formula above. Default: ``1e-6``.
        weight_decay (float, optional): weight decay (L2 penalty). Default: ``0.``.

    Keyword Args:
        maximize (bool, optional): maximize the params based on the objective, instead of minimizing.
            Default: ``False``.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`.

    Raises:
        ValueError: If the learning rate is not int, float or Tensor.
        ValueError: If the learning rate is less than 0.
        ValueError: If the `eps` is less than or equal to 0.0.
        ValueError: If the `rho` is not in the range of [0, 1].
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
        >>> optimizer = optim.Adadelta(net.trainable_params(), lr=0.1)
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

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0, *, maximize=False):
        check_not_less_than_without_equal(lr, "lr", self.cls_name)
        check_not_less_than_without_equal(eps, "eps", self.cls_name)
        check_not_less_than(weight_decay, "weight_decay", self.cls_name)
        validator.check_float_range(rho, 0., 1., validator.INC_BOTH, "rho", self.cls_name)

        defaults = dict(
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super(Adadelta, self).__init__(params, defaults)

        self.accum = self.parameters.clone(prefix="accum", init=0)
        self.accum_update = self.parameters.clone(prefix="accum_update", init=0)
        self.opt = P.ApplyAdadelta()
        self.op_cast = P.Cast()

    @jit
    def implementation(self, lr, rho, eps, maximize, weight_decay, start_id, end_id, gradients):
        """Extract the common computing part for acceleration"""
        params = self.parameters[start_id: end_id]
        grads = tuple([grad if not maximize else F.neg(grad) for grad in gradients[start_id: end_id]])
        grads = self._decay_weight(weight_decay, params, grads)
        accum = self.accum[start_id: end_id]
        accum_update = self.accum_update[start_id: end_id]
        self.hyper_map(F.partial(_adadelta_opt, self.opt, rho, eps, lr),
                       params, accum, accum_update, grads)
        return True

    def construct(self, gradients):
        for group_id, group in enumerate(self.param_groups):
            lr = self.lrs[group_id]
            if isinstance(group.get("lr"), float):
                lr = self.op_cast(group.get("lr"), mstype.float32)
            maximize = group.get("maximize")
            rho = group["rho"]
            eps = group["eps"]

            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id + 1]
            weight_decay = group["weight_decay"]
            self.implementation(lr, rho, eps, maximize, weight_decay, start_id, end_id, gradients)

        return True
