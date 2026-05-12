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
"""Adam"""
from __future__ import absolute_import

from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.experimental.optim.optimizer import Optimizer
from mindspore import _checkparam as validator
from mindspore import mint
from mindspore import ops


_optim_adamw_opt = ops.MultitypeFuncGraph("optim_adamw_opt")
hyper_map = ops.HyperMap()


@_optim_adamw_opt.register("Float", "Float", "Float", "Tensor", "Tensor", "Tensor", "Tensor",
                           "Tensor", "Tensor", "Tensor")
def _run_optim_adamw_amsgrad_opt(beta1, beta2, eps, neg_step_size, sqrt_bias_correction2, parameters, grads, exp_avg,
                                 exp_avg_sq, max_exp_avg_sq):
    """Apply adam optimizer to the weight parameter when amsgrad is True."""
    success = True
    exp_avg_tmp = mint.add(mint.mul(exp_avg, beta1), grads, alpha=1 - beta1)
    exp_avg_sq_tmp = mint.mul(exp_avg_sq, beta2) + mint.mul(mint.mul(grads, grads), 1 - beta2)

    max_exp_avg_sq = mint.maximum(max_exp_avg_sq, exp_avg_sq_tmp)
    denom = ops.cast(mint.div(mint.sqrt(max_exp_avg_sq), sqrt_bias_correction2), max_exp_avg_sq.dtype)
    denom = mint.add(denom, eps)

    delta_param = mint.mul(ops.cast(neg_step_size, max_exp_avg_sq.dtype), mint.div(exp_avg_tmp, denom))
    exp_avg.copy_(exp_avg_tmp)
    exp_avg_sq.copy_(exp_avg_sq_tmp)
    parameters.add_(delta_param)
    return success


@_optim_adamw_opt.register("Float", "Float", "Float", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _run_optim_adamw_opt(beta1, beta2, eps, neg_step_size, sqrt_bias_correction2, parameters, grads, exp_avg,
                         exp_avg_sq):
    """Apply adam optimizer to the weight parameter when amsgrad is False."""
    success = True
    exp_avg_tmp = mint.add(mint.mul(exp_avg, beta1), grads, alpha=1 - beta1)
    exp_avg_sq_tmp = mint.mul(exp_avg_sq, beta2) + mint.mul(mint.mul(grads, grads), 1 - beta2)

    denom = ops.cast(mint.div(mint.sqrt(exp_avg_sq_tmp), sqrt_bias_correction2), exp_avg_sq_tmp.dtype)
    denom = mint.add(denom, eps)

    delta_param = mint.mul(ops.cast(neg_step_size, exp_avg_sq_tmp.dtype), mint.div(exp_avg_tmp, denom))
    exp_avg.copy_(exp_avg_tmp)
    exp_avg_sq.copy_(exp_avg_sq_tmp)
    parameters.add_(delta_param)
    return success


def _check_param_value(betas, eps, weight_decay, lr, amsgrad, maximize, prim_name):
    """Check the type of inputs."""
    validator.check_value_type('betas', betas, [tuple], prim_name)
    validator.check("betas size", len(betas), "", [2], validator.IN, prim_name)
    validator.check_value_type("betas[0]", betas[0], [float], prim_name)
    validator.check_value_type("betas[1]", betas[1], [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_value_type("weight_decay", weight_decay, [float], prim_name)
    validator.check_value_type("lr", lr, [float], prim_name)
    validator.check_value_type("amsgrad", amsgrad, [bool], prim_name)
    validator.check_value_type("maximize", maximize, [bool], prim_name)


class Adam(Optimizer):
    r"""
    Implements Adaptive Moment Estimation (Adam) algorithm.

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
        This is an experimental API that is subject to change or deletion.

    Args:
        params (Union[list(Parameter), list(dict)]): list of parameters to optimize or dicts defining
            parameter groups
        lr (Union[int, float, Tensor], optional): learning rate. Default: ``1e-3``.
        betas (Tuple[float, float], optional): The exponential decay rate for the moment estimations.
            Should be in range (0.0, 1.0). Default: ``(0.9, 0.999)``.
        eps (float, optional): Term added to the denominator to improve
            numerical stability. Should be greater than 0. Default: ``1e-8``.
        weight_decay (float, optional): Weight decay (L2 penalty). Default: ``0.``.
        amsgrad (bool, optional): Whether to use the AMSGrad algorithm. Default: ``False``.

    Keyword Args:
        maximize (bool, optional): maximize the params based on the objective, instead of minimizing.
            Default: ``False``.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`.

    Raises:
        ValueError: If the `lr` is not int, float or Tensor.
        ValueError: If the `lr` is less than 0.
        ValueError: If the `eps` is less than 0.0.
        ValueError: If the `betas` is not in the range of [0, 1).
        ValueError: If the `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import nn
        >>> from mindspore import mint
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://atomgit.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        >>> optimizer = mint.optim.Adam(net.trainable_params(), lr=0.1)
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
                 weight_decay=0.0, amsgrad=False, *, maximize=False):
        _check_param_value(betas, eps, weight_decay, lr, amsgrad, maximize, self.cls_name)
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

        defaults = {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "amsgrad": amsgrad,
                "maximize": maximize,
            }
        self.max_v_group = True
        super().__init__(params, defaults)

        self.exp_avg = self.parameters.clone(prefix="exp_avg", init='zeros')
        self.exp_avg_sq = self.parameters.clone(prefix="exp_avg_sq", init='zeros')
        self.state_step = Parameter(Tensor([0], mstype.float32), "state_step")
        self.increase_tensor = Tensor(1, mstype.float32)
        self.pow = ops.Pow()


    def construct(self, gradients):
        self.state_step.add_(self.increase_tensor)
        for group_id, group in enumerate(self.param_groups):
            beta1, beta2 = group['betas']
            maximize = group.get("maximize")
            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id + 1]
            lr = group.get("lr")
            grads = tuple(grad if not maximize else mint.neg(grad) for grad in gradients[start_id: end_id])

            bias_correction1 = 1 - beta1 ** self.state_step
            bias_correction2 = 1 - beta2 ** self.state_step
            neg_step_size = -mint.div(lr, bias_correction1)
            sqrt_bias_correction2 = mint.sqrt(bias_correction2)
            grads = self._decay_weight(group.get("weight_decay"), self.parameters[start_id: end_id], grads)

            if group.get("amsgrad"):
                self.hyper_map(ops.partial(_optim_adamw_opt, beta1, beta2, group.get("eps"), neg_step_size,
                                           sqrt_bias_correction2),
                               self.parameters[start_id: end_id], grads, self.exp_avg[start_id: end_id],
                               self.exp_avg_sq[start_id: end_id], group.get("max_exp_avg_sq"))
            else:
                self.hyper_map(ops.partial(_optim_adamw_opt, beta1, beta2, group.get("eps"), neg_step_size,
                                           sqrt_bias_correction2),
                               self.parameters[start_id: end_id], grads, self.exp_avg[start_id: end_id],
                               self.exp_avg_sq[start_id: end_id])
        return True
