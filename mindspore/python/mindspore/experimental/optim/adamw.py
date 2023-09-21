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
"""adamw"""
from __future__ import absolute_import

from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.experimental.optim.optimizer import Optimizer
from mindspore import ops

_adamw_opt = C.MultitypeFuncGraph("adamw_opt")

op_mul = P.Mul()
op_pow = P.Pow()
op_sqrt = P.Sqrt()
op_maximum = P.Maximum()


@_adamw_opt.register("Float", "Tensor", "Bool", "Float", "Tensor", "Float", "Float", "Tensor", "Tensor",
                     "Tensor", "Tensor", "Tensor")
def _run_adamw_opt(weight_decay, lr, amsgrad, eps, state_step, beta1, beta2, param, grad,
                   exp_avg, exp_avg_sq, max_exp_avg_sq):
    """Apply adamw optimizer to the weight parameter."""
    success = True
    next_param = op_mul(param, 1 - lr * weight_decay)
    F.assign(exp_avg, op_mul(exp_avg, beta1) + op_mul(grad, 1 - beta1))
    F.assign(exp_avg_sq, ops.addcmul(op_mul(exp_avg_sq, beta2), grad, grad, 1 - beta2))
    bias_correction1 = 1 - op_pow(beta1, state_step)
    bias_correction2 = 1 - op_pow(beta2, state_step)
    step_size = lr / bias_correction1
    bias_correction2_sqrt = op_sqrt(bias_correction2)

    if amsgrad:
        next_max_exp_avg = op_maximum(max_exp_avg_sq, exp_avg_sq)
        denom = op_sqrt(next_max_exp_avg) / bias_correction2_sqrt + eps
        F.assign(max_exp_avg_sq, next_max_exp_avg)
    else:
        denom = op_sqrt(exp_avg_sq) / bias_correction2_sqrt + eps

    return_param = next_param - op_mul(exp_avg / denom, step_size)
    F.assign(param, return_param)
    return success


class AdamW(Optimizer):
    r"""
    Implements Adam Weight Decay algorithm.

    .. math::
        \begin{aligned}
            &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
                \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
                \: \epsilon \text{ (epsilon)}                                                    \\
            &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
                \: \textit{maximize}                                                             \\
            &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
                \text{ ( second moment)}, \: \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
       \end{aligned}

    .. warning::
        This is an experimental optimizer API that is subject to change.
        This module must be used with lr scheduler module in `LRScheduler Class
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#lrscheduler-class>`_ .

    Args:
        params (Union[list(Parameter), list(dict)]): list of parameters to optimize or dicts defining
            parameter groups
        lr (Union[int, float, Tensor], optional): learning rate. Default: ``1e-3``.
        betas (Tuple[float, float], optional): The exponential decay rate for the moment estimations.
            Default: ``(0.9, 0.999)``.
        eps (float, optional): term added to the denominator to improve
            numerical stability. Default: ``1e-8``.
        weight_decay (float, optional): weight decay (L2 penalty). Default: ``0``.
        amsgrad (bool, optional): whether to use the AMSGrad algorithm. Default: ``False``.

    Keyword Args:
        maximize (bool, optional): maximize the params based on the objective, instead of minimizing.
            Default: ``False``.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`.

    Raises:
        ValueError: If the learning rate is not int, float or Tensor.
        ValueError: If the learning rate is less than 0.
        ValueError: If the `eps` is less than 0.0.
        ValueError: If the `betas` not in the range of 0-1.
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
        >>> optimizer = optim.AdamW(net.trainable_params(), lr=0.1)
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
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, *, maximize=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        maximize=maximize)
        super(AdamW, self).__init__(params, defaults)

        self.exp_avg = self.parameters.clone(prefix="exp_avg", init='zeros')
        self.exp_avg_sq = self.parameters.clone(prefix="exp_avg_sq", init='zeros')
        self.max_exp_avg_sq = self.parameters.clone(prefix="max_exp_avg_sq", init='zeros')
        self.state_step = Parameter(Tensor(0, mstype.int32), "state_step")
        self.increase_tensor = Tensor(1, mstype.int32)
        self.assignadd = P.AssignAdd()

    def construct(self, gradients):
        self.assignadd(self.state_step, self.increase_tensor)
        for group_id, group in enumerate(self.param_groups):
            beta1, beta2 = group['betas']
            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id + 1]
            grads = gradients[start_id: end_id] if not group.get("maximize") else -gradients[start_id: end_id]
            self.hyper_map(F.partial(_adamw_opt, group.get("weight_decay"), group.get("lr"), group.get("amsgrad"),
                                     group.get("eps"), self.state_step, beta1, beta2),
                           self.parameters[start_id: end_id], grads, self.exp_avg[start_id: end_id],
                           self.exp_avg_sq[start_id: end_id], self.max_exp_avg_sq[start_id: end_id])
        return True
