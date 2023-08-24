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
"""adam"""
from __future__ import absolute_import

from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.experimental.optim.optimizer import Optimizer

_adam_opt = C.MultitypeFuncGraph("adam_opt")


@_adam_opt.register("Function", "Tensor", "Tensor", "Float", "Float", "Float", "Tensor",
                    "Tensor", "Tensor", "Tensor", "Tensor")
def _run_adam_opt(opt, beta1_power, beta2_power, beta1, beta2, eps, lr, gradient, param, moment1, moment2):
    """Apply adam optimizer to the weight parameter."""
    success = True
    success = F.depend(success, opt(param, moment1, moment2, beta1_power, beta2_power, lr, beta1, beta2, eps, gradient))
    return success


@_adam_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _run_adam_with_amsgrad_opt(opt, beta1_power, beta2_power, lr, gradient, param, moment1, moment2, vhat):
    """Apply adam optimizer to the weight parameter with amsgrad."""
    success = True
    success = F.depend(success, opt(param, moment1, moment2, vhat, beta1_power, beta2_power, lr, gradient))
    return success


class Adam(Optimizer):
    r"""
    Implements Adam algorithm..

    The updating formulas are as follows:

    .. math::
        \begin{aligned}
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)}          \\
            &\hspace{13mm}      \lambda \text{ (weight decay)},  \: \textit{amsgrad},
                \:\textit{maximize}                                                              \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0\leftarrow 0 \text{ (second moment)},\: \widehat{v_0}^{max}\leftarrow 0\\[-1.ex]
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
       \end{aligned}

    .. warning::
        This is an experimental optimizer API that is subject to change.
        This module must be used with lr scheduler module in `LRScheduler Class
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#learningrateschedule-class>`_ .

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
        ValueError: If the `lr` is not int, float or Tensor.
        ValueError: If the `lr` is less than 0.
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
        >>> optimizer = optim.Adam(net.trainable_params(), lr=0.1)
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
                 weight_decay=0, amsgrad=False, *, maximize=False):
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
        super(Adam, self).__init__(params, defaults)

        self.exp_avg = self.parameters.clone(prefix="exp_avg", init='zeros')
        self.exp_avg_sq = self.parameters.clone(prefix="exp_avg_sq", init='zeros')
        self.max_exp_avg_sq = self.parameters.clone(prefix="max_exp_avg_sq", init='zeros')
        self.state_step = Parameter(Tensor(0, mstype.int32), "state_step")
        self.increase_tensor = Tensor(1, mstype.int32)
        self.assignadd = P.AssignAdd()
        self.op_add = P.AddN()
        self.op_mul = P.Mul()
        self.op_pow = P.Pow()
        self.adam_opt = P.Adam(False, False)
        self.op_cast = P.Cast()

    def construct(self, gradients):
        self.assignadd(self.state_step, self.increase_tensor)
        for group_id, group in enumerate(self.param_groups):
            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id+1]

            lr = group.get("lr")
            weight_decay = group.get("weight_decay")
            beta1, beta2 = group.get("betas")
            maximize = group.get("maximize")
            eps = group.get("eps")

            beta1_power = self.op_pow(beta1, self.state_step)
            beta2_power = self.op_pow(beta2, self.state_step)
            adam_with_amsgrad_opt = P.ApplyAdamWithAmsgrad(beta1, beta2, eps, False)
            params = self.parameters[start_id: end_id]
            grads = gradients[start_id: end_id]
            grads = grads if not maximize else -grads
            grads = self._decay_weight(weight_decay, params, grads)
            if isinstance(lr, float):
                lr = self.op_cast(group.get("lr"), mstype.float32)
            if group.get("amsgrad"):
                self.hyper_map(F.partial(_adam_opt, adam_with_amsgrad_opt, beta1_power, beta2_power, lr),
                               grads, params,
                               self.exp_avg[start_id: end_id], self.exp_avg_sq[start_id: end_id],
                               self.max_exp_avg_sq[start_id: end_id])
            else:
                self.hyper_map(F.partial(_adam_opt, self.adam_opt, beta1_power, beta2_power, beta1, beta2, eps, lr),
                               grads, params,
                               self.exp_avg[start_id: end_id], self.exp_avg_sq[start_id: end_id])
        return True
