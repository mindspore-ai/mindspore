# Copyright 2026 Huawei Technologies Co., Ltd
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
"""FusedAdamW"""
from __future__ import absolute_import
from mindspore.experimental.optim.optimizer import Optimizer
from mindspore import _checkparam as validator
from mindspore.ops import auto_generate as gen
from mindspore import mint
import mindspore as ms


def _check_param_value(betas, eps, weight_decay, lr, amsgrad, maximize, prim_name):
    """
    Validates the types and values of AdamW optimizer parameters.

    Args:
        betas (Union[tuple, list]): Coefficients used for computing running averages of gradient and its square.
        eps (float): Term added to the denominator to improve numerical stability.
        weight_decay (float): Weight decay coefficient.
        lr (float): Learning rate.
        amsgrad (bool): Whether to use the AMSGrad variant of this algorithm.
        maximize (bool): Whether to maximize the parameters based on the objective.
        prim_name (str): Name of the primitive for error message purposes.
    """
    validator.check_value_type("betas", betas, [tuple, list], prim_name)
    validator.check("betas size", len(betas), "", [2], validator.IN, prim_name)
    validator.check_value_type("betas[0]", betas[0], [float], prim_name)
    validator.check_value_type("betas[1]", betas[1], [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_value_type("weight_decay", weight_decay, [float], prim_name)
    validator.check_value_type("lr", lr, [float], prim_name)
    validator.check_value_type("amsgrad", amsgrad, [bool], prim_name)
    validator.check_value_type("maximize", maximize, [bool], prim_name)


class FusedAdamW(Optimizer):
    r"""
    Implements Adam Weight Decay algorithm.

    .. math::
        \begin{array}{l}
            &\newline
            &\hline \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
                \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
                \: \epsilon \text{ (epsilon)}                                                    \\
            &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
                \: \textit{maximize}                                                             \\
            &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
                \text{ ( second moment)}, \: \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
            &\newline
            &\hline \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{6mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{11mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{6mm}\textbf{else}                                                           \\
            &\hspace{11mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{6mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
            &\hspace{6mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{6mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{6mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{6mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{6mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{11mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{11mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{6mm}\textbf{else}                                                           \\
            &\hspace{11mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\newline
            &\hline \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
       \end{array}

    More details of the FusedAdamW algorithm can be found in the paper `Decoupled Weight Decay Regularization
    <https://arxiv.org/abs/1711.05101>`_ and `On the Convergence of Adam and Beyond
    <https://openreview.net/forum?id=ryQu7f-RZ>`_.

    .. warning::
        - This is an experimental optimizer API that is subject to change or deletion.
          This module must be used with lr scheduler module in `LRScheduler Class
          <https://www.mindspore.cn/docs/en/master/api_python/mindspore.experimental.html#lrscheduler-class>`_ .
        - For Ascend, it is only supported on platforms above Atlas A2.

    Args:
        params (Union[tuple, list]): list of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. Default: ``1e-3``.
        betas (Tuple[float, float], optional): The exponential decay rate for the moment estimations.
            Default: ``(0.9, 0.999)``.
        eps (float, optional): term added to the denominator to improve
            numerical stability. Must be greater than 0. Default: ``1e-8``.
        weight_decay (float, optional): weight decay (L2 penalty). Default: ``1e-2``.
        amsgrad (bool, optional): whether to use the AMSGrad algorithm. Default: ``False``.

    Keyword Args:
        maximize (bool, optional): maximize the params based on the objective, instead of minimizing.
            Default: ``False``.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`.

    Raises:
        ValueError: If the learning rate is not float.
        ValueError: If the learning rate is less than 0.
        ValueError: If the `eps` is less than 0.
        ValueError: If the `betas` not in the range of [0, 1).
        ValueError: If the `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import nn
        >>> from mindspore.mint import optim
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        >>> optimizer = optim.FusedAdamW(net.trainable_params(), lr=0.1)
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

    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, *, maximize=False):
        # The API definition and the validation of attributes below refers to PyTorch:
        # https://github.com/pytorch/pytorch/blob/v2.1.0/torch/optim/adamw.py
        # Note: The _check_param_value is a custom validation method.
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

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay,
                    "amsgrad": amsgrad,  "maximize": maximize}
        self.max_v_group = True
        super().__init__(params, defaults)

        self.exp_avg = self.parameters.clone(prefix="exp_avg", init="zeros")
        self.exp_avg_sq = self.parameters.clone(prefix="exp_avg_sq", init="zeros")
        self.state_step = ms.Parameter(ms.Tensor([-1], ms.float32), "state_step")
        self.increase_tensor = ms.Tensor(1, ms.float32)

        self.max_exp_avg_sq = None
        self.continuous_parameters = []
        self.continuous_exp_avg = []
        self.continuous_exp_avg_sq = []
        self.continuous_max_exp_avg_sq = []
        if amsgrad:
            self.max_exp_avg_sq = self.parameters.clone(prefix="max_exp_avg_sq", init="zeros")

        self.adamw_opt = gen.AdamW()
        self.set_contiguous_params()

    def add_param_group(self, param_group):
        """
        Add a param group to the `Optimizer.param_groups`

        Args:
            param_group (dict): Specifies what Parameters should be optimizerd along with group
                                specific optimization options.
        """
        super().add_param_group(param_group)
        if hasattr(self, "exp_avg"):
            params = ms.ParameterTuple(param_group.get("params"))
            self.exp_avg += params.clone(prefix="exp_avg", init="zeros")
            self.exp_avg_sq += params.clone(prefix="exp_avg_sq", init="zeros")
            if "amsgrad" in param_group and param_group.get("amsgrad"):
                self.max_exp_avg_sq += params.clone(prefix="max_exp_avg_sq", init="zeros")
            start_id = self.group_start_id[-2]
            end_id = self.group_start_id[-1]
            self.contiguous_params(start_id, end_id, param_group)

    def contiguous_params(self, start_id, end_id, param_group):
        """
        Concatenates parameters and their associated states (exp_avg, exp_avg_sq) into contiguous memory blocks
        to optimize memory access during optimization steps.

        Args:
            start_id (int): Start index of the parameter group.
            end_id (int): End index of the parameter group.
            param_group (dict): Parameter group containing optimization settings.
        """
        amsgrad = "amsgrad" in param_group and param_group.get("amsgrad") and hasattr(self, "max_v_group")
        max_exp_avg_sq = None

        if end_id - start_id > 1:
            size = [p.size for p in self.parameters[start_id:end_id]]
            cur_tensor = mint.zeros((sum(size),), dtype=self.parameters[start_id].dtype)
            params = ms.Parameter(cur_tensor)
            exp_avg = ms.Parameter(cur_tensor)
            exp_avg_sq = ms.Parameter(cur_tensor)
            if amsgrad:
                max_exp_avg_sq = ms.Parameter(cur_tensor)
            pre_size = 0
            # pylint: disable=protected-access
            for i, s_t in enumerate(size):
                shape = self.parameters[start_id + i].shape
                params[pre_size : pre_size + s_t].copy_(self.parameters[start_id + i].view(-1))
                self.parameters[start_id + i].data = params[pre_size : pre_size + s_t].view(shape)
                self.exp_avg[start_id + i].data = exp_avg[pre_size : pre_size + s_t].view(shape)
                self.exp_avg_sq[start_id + i].data = exp_avg_sq[pre_size : pre_size + s_t].view(shape)
                if amsgrad:
                    param_group["max_exp_avg_sq"][i].data = max_exp_avg_sq[pre_size : pre_size + s_t].view(shape)
                pre_size += s_t
            # pylint: enable=protected-access

        else:
            params = self.parameters[start_id]
            exp_avg = self.exp_avg[start_id]
            exp_avg_sq = self.exp_avg_sq[start_id]
            if amsgrad:
                if self.max_exp_avg_sq is not None:
                    max_exp_avg_sq = self.max_exp_avg_sq[start_id]
                else:
                    max_exp_avg_sq = ms.Parameter(mint.zeros((params.shape), dtype=params.dtype))
        self.continuous_parameters.append(params)
        self.continuous_exp_avg.append(exp_avg)
        self.continuous_exp_avg_sq.append(exp_avg_sq)
        self.continuous_max_exp_avg_sq.append(max_exp_avg_sq)

    def set_contiguous_params(self):
        for group_id, group in enumerate(self.param_groups):
            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id + 1]
            self.contiguous_params(start_id, end_id, group)


    def construct(self, gradients):
        """
        fusedAdamW.
        Args:
            gradients (list[Tensor]): Gradient tensors for all parameters.

        Returns
            bool, True
        """
        self.state_step += self.increase_tensor
        for group_id, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            maximize = group.get("maximize")
            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id + 1]
            if end_id - start_id > 1:
                continuous_grads = mint.cat([g.view(-1).contiguous() for g in gradients[start_id:end_id]], dim=0)
            else:
                continuous_grads = gradients[start_id]

            if group.get("amsgrad"):
                self.adamw_opt(
                    self.continuous_parameters[group_id],
                    self.continuous_exp_avg[group_id],
                    self.continuous_exp_avg_sq[group_id],
                    self.continuous_max_exp_avg_sq[group_id],
                    continuous_grads,
                    self.state_step,
                    float(group.get("lr")),
                    beta1,
                    beta2,
                    group.get("weight_decay"),
                    group.get("eps"),
                    group.get("amsgrad"),
                    maximize,
                )
            else:
                self.adamw_opt(
                    self.continuous_parameters[group_id],
                    self.continuous_exp_avg[group_id],
                    self.continuous_exp_avg_sq[group_id],
                    self.continuous_exp_avg_sq[group_id],
                    continuous_grads,
                    self.state_step,
                    float(group.get("lr")),
                    beta1,
                    beta2,
                    group.get("weight_decay"),
                    group.get("eps"),
                    group.get("amsgrad"),
                    maximize,
                )
        return True
