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
"""radam"""
from __future__ import absolute_import

from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common import Tensor, Parameter
import mindspore.common.dtype as mstype
from mindspore import _checkparam as validator
from mindspore.experimental.optim.optimizer import Optimizer, check_not_less_than, check_not_less_than_without_equal
from mindspore import jit

_radam_opt = C.MultitypeFuncGraph("radam_opt")

op_pow = P.Pow()
op_sqrt = P.Sqrt()
op_cast = P.Cast()


@_radam_opt.register("Number", "Number", "Number", "Tensor", "Number", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                     "Tensor", "Tensor")
def _tensor_run_opt(beta1, beta2, eps, lr, rho_inf, rho_t, bias_correction1, bias_correction2, param, grad, exp_avg,
                    exp_avg_sq):
    """Apply radam optimizer to the weight parameter."""

    F.assign(exp_avg, exp_avg * beta1 + grad * (1 - beta1))
    F.assign(exp_avg_sq, exp_avg_sq * beta2 + grad * grad * (1 - beta2))
    bias_corrected_exp_avg = exp_avg / bias_correction1

    if rho_t > 5.0:
        rect = op_sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
        exp_avg_sq_sqrt = op_sqrt(exp_avg_sq) + eps
        adaptive_lr = op_sqrt(bias_correction2) / exp_avg_sq_sqrt
        F.assign(param, param - bias_corrected_exp_avg * lr * adaptive_lr * rect)
    else:
        F.assign(param, param - bias_corrected_exp_avg * lr)

    return True


class RAdam(Optimizer):
    r"""
    Implements RAdam algorithm.

    .. math::
        \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \beta_1, \beta_2
                \text{ (betas)}, \: \theta_0 \text{ (params)}, \:f(\theta) \text{ (objective)}, \:
                \lambda \text{ (weightdecay)},                                                   \\
            &\hspace{13mm} \epsilon \text{ (epsilon)}                                            \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0 \leftarrow 0 \text{ ( second moment)},                                       \\
            &\hspace{18mm} \rho_{\infty} \leftarrow 2/(1-\beta_2) -1                      \\[-1.ex]
            &\rule{110mm}{0.4pt}  \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{6mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda \theta_{t-1}                             \\
            &\hspace{6mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{6mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{6mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{6mm}\rho_t \leftarrow \rho_{\infty} -
                2 t \beta^t_2 /\big(1-\beta_2^t \big)                                     \\[-1.ex]
            &\hspace{6mm}\textbf{if} \: \rho_t > 5                                               \\
            &\hspace{12mm} l_t \leftarrow \frac{\sqrt{ (1-\beta^t_2) }}{ \sqrt{v_t} +\epsilon  } \\
            &\hspace{12mm} r_t \leftarrow
                \sqrt{\frac{(\rho_t-4)(\rho_t-2)\rho_{\infty}}{(\rho_{\infty}-4)(\rho_{\infty}-2) \rho_t}} \\
            &\hspace{12mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t} r_t l_t        \\
            &\hspace{6mm}\textbf{else}                                                           \\
            &\hspace{12mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}                \\
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
        lr (Union[int, float, Tensor], optional): learning rate. Default: ``1e-3``.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. Default: ``(0.9, 0.999)``.
        eps (float, optional): term added to the denominator to improve
            numerical stability. Default: ``1e-8``.
        weight_decay (float, optional): weight decay (L2 penalty). Default: ``0.0``.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`.

    Raises:
        ValueError: If the learning rate is not int, float or Tensor.
        ValueError: If the learning rate is less than 0.
        ValueError: If the `eps` is less than 0.0.
        ValueError: If the `weight_decay` is less than 0.
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
        >>> optimizer = optim.RAdam(net.trainable_params(), lr=0.1)
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

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
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
        )
        super(RAdam, self).__init__(params, defaults)
        self.step_t = Parameter(Tensor(0, mstype.int32), "step_t")
        self.exp_avg = self.parameters.clone(prefix="exp_avg", init='zeros')
        self.exp_avg_sq = self.parameters.clone(prefix="exp_avg_sq", init='zeros')
        self.increase_tensor = Tensor(1, mstype.int32)
        self.assignadd = P.AssignAdd()

    @jit
    def implementation(self, lr, beta1, beta2, weight_decay, eps, start_id, end_id, gradients):
        """Extract the common computing part for acceleration"""
        params = self.parameters[start_id: end_id]
        grads = gradients[start_id: end_id]
        grads = self._decay_weight(weight_decay, params, grads)
        exp_avg = self.exp_avg[start_id: end_id]
        exp_avg_sq = self.exp_avg_sq[start_id: end_id]

        bias_correction1 = 1 - op_pow(beta1, self.step_t.value())
        bias_correction2 = 1 - op_pow(beta2, self.step_t.value())

        rho_inf = 2 / (1 - beta2) - 1
        beta2_pow = op_pow(beta2, self.step_t.value())
        right = 2 * self.step_t.value() * beta2_pow / bias_correction2

        rho_t = rho_inf - right

        self.hyper_map(F.partial(_radam_opt, beta1, beta2, eps, lr, rho_inf, rho_t, bias_correction1, bias_correction2),
                       params, grads, exp_avg, exp_avg_sq)
        return True

    def construct(self, gradients):
        self.assignadd(self.step_t, self.increase_tensor)
        for group_id, group in enumerate(self.param_groups):

            lr = self.lrs[group_id]
            if isinstance(group.get("lr"), float):
                lr = op_cast(group.get("lr"), mstype.float32)

            beta1, beta2 = group["betas"]
            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id + 1]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            self.implementation(lr, beta1, beta2, weight_decay, eps, start_id, end_id, gradients)

        return True
