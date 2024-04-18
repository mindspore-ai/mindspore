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
"""adamax"""
from __future__ import absolute_import

from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common import Tensor, Parameter
import mindspore.common.dtype as mstype
from mindspore import _checkparam as validator
from mindspore.experimental.optim.optimizer import Optimizer, check_not_less_than, check_not_less_than_without_equal
from mindspore import ops
from mindspore import jit

_adamax_opt = C.MultitypeFuncGraph("adamax_opt")


@_adamax_opt.register("Number", "Number", "Number", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt(beta1, beta2, eps, clr, param, grad, exp_avg, exp_inf):
    """Apply adamax optimizer to the weight parameter."""
    F.assign(exp_avg, exp_avg * beta1 + grad * (1-beta1))
    norm_buf = ops.cat([ops.unsqueeze(exp_inf * beta2, 0), ops.unsqueeze(grad.abs().add(eps), 0)], 0)
    F.assign(exp_inf, ops.amax(norm_buf, 0))

    F.assign(param, param - clr * exp_avg / exp_inf)
    return True


class Adamax(Optimizer):
    r"""
    Implements Adamax algorithm (a variant of Adam based on infinity norm).

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)},
                \: \lambda \text{ (weight decay)},                                                \\
            &\hspace{13mm}    \epsilon \text{ (epsilon)}                                          \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                u_0 \leftarrow 0 \text{ ( infinity norm)}                                 \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t      \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t               \\
            &\hspace{5mm}u_t      \leftarrow   \mathrm{max}(\beta_2 u_{t-1}, |g_{t}|+\epsilon)   \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1} - \frac{\gamma m_t}{(1-\beta^t_1) u_t} \\
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
        lr (Union[int, float, Tensor], optional): learning rate. Default: ``2e-3``.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. Default: ``(0.9, 0.999)``.
        eps (float, optional): term added to the denominator to improve
            numerical stability. Default: ``1e-8``.
        weight_decay (float, optional): weight decay (L2 penalty). Default: ``0.``.

    Keyword Args:
        maximize (bool, optional): maximize the params based on the objective, instead of minimizing.
            Default: ``False``.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`.

    Raises:
        ValueError: If the learning rate is not int, float or Tensor.
        ValueError: If the learning rate is less than 0.
        ValueError: If the `eps` is less than 0.0.
        ValueError: If the `weight_decay` is less than 0.
        ValueError: If elements of the `betas` not in the range of [0,1).

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
        >>> optimizer = optim.Adamax(net.trainable_params(), lr=0.1)
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

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, *, maximize=False):
        check_not_less_than_without_equal(lr, "lr", self.cls_name)
        check_not_less_than(weight_decay, "weight_decay", self.cls_name)
        check_not_less_than_without_equal(eps, "eps", self.cls_name)
        validator.check_float_range(betas[0], 0., 1., validator.INC_LEFT, "betas[0]", self.cls_name)
        validator.check_float_range(betas[1], 0., 1., validator.INC_LEFT, "betas[1]", self.cls_name)

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super(Adamax, self).__init__(params, defaults)

        self.step_t = Parameter(Tensor(0, mstype.int32), "step_t")
        self.exp_avg = self.parameters.clone(prefix="exp_avg", init='zeros')
        self.exp_inf = self.parameters.clone(prefix="exp_inf", init='zeros')
        self.increase_tensor = Tensor(1, mstype.int32)
        self.assignadd = P.AssignAdd()
        self.op_cast = P.Cast()

    @jit
    def implementation(self, group_id, lr, gradients, maximize, weight_decay, beta1, beta2, eps):
        """Extract the common computing part for acceleration"""
        start_id = self.group_start_id[group_id]
        end_id = self.group_start_id[group_id + 1]
        params = self.parameters[start_id: end_id]
        grads = tuple([grad if not maximize else F.neg(grad) for grad in gradients[start_id: end_id]])
        grads = self._decay_weight(weight_decay, params, grads)
        exp_avg = self.exp_avg[start_id: end_id]
        exp_inf = self.exp_inf[start_id: end_id]
        bias_correction = 1 - beta1 ** self.step_t
        clr = lr / bias_correction
        self.hyper_map(F.partial(_adamax_opt, beta1, beta2, eps, clr),
                       params, grads, exp_avg, exp_inf)
        return True

    def construct(self, gradients):
        self.assignadd(self.step_t, self.increase_tensor)

        for group_id, group in enumerate(self.param_groups):

            lr = self.lrs[group_id]
            if isinstance(group.get("lr"), float):
                lr = self.op_cast(group.get("lr"), mstype.float32)

            maximize = group.get("maximize")
            beta1, beta2 = group["betas"]

            self.implementation(group_id, lr, gradients, maximize, group["weight_decay"], beta1, beta2, group["eps"])
        return True
