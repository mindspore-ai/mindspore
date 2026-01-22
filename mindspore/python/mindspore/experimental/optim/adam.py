# The code implementation refers to the following files from pytorch:
# - https://github.com/pytorch/pytorch/blob/v1.13.0/torch/optim/adam.py
# Additional modifications are made by Huawei Technologies Co., Ltd in 2023.
# ============================================================================
"""adam"""
from __future__ import absolute_import

from mindspore import ops
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.experimental.optim.optimizer import Optimizer
from mindspore.graph.api import jit

_adam_opt = ops.MultitypeFuncGraph("adam_opt")
adam_op = ops.Adam(False, False)


@_adam_opt.register("Tensor", "Tensor", "Float", "Float", "Float", "Tensor",
                    "Tensor", "Tensor", "Tensor", "Tensor")
def _run_adam_opt(beta1_power, beta2_power, beta1, beta2, eps, lr, gradient, param, moment1, moment2):
    """Apply adam optimizer to the weight parameter."""
    adam_op(param, moment1, moment2, beta1_power, beta2_power, lr, beta1, beta2, eps, gradient)
    return True


@_adam_opt.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _run_adam_with_amsgrad_opt(beta1_power, beta2_power, lr, gradient, param, moment1, moment2, vhat):
    """Apply adam optimizer to the weight parameter with amsgrad."""
    adam_op(param, moment1, moment2, vhat, beta1_power, beta2_power, lr, gradient)
    return True


class Adam(Optimizer):
    r"""
    Implements Adam algorithm.

    The updating formulas are as follows:

    .. math::
        \begin{aligned}
            &\rule{180mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)}          \\
            &\hspace{13mm}      \lambda \text{ (weight decay)},  \: \textit{amsgrad},
                \:\textit{maximize}                                                              \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0\leftarrow 0 \text{ (second moment)},\: \widehat{v_0}^{max}\leftarrow 0\\[-1.ex]
            &\rule{180mm}{0.4pt}                                                                 \\
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
            &\rule{180mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{180mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For more details about Adam algorithm, please refer to `Adam: A Method for Stochastic Optimization
    <https://arxiv.org/abs/1412.6980>`_.

    .. warning::
        The implementation formula of this optimizer interface is not completely consistent with that in the paper.
        If you want to use an interface that is completely consistent, it is recommended to use
        :class:`mindspore.mint.optim.Adam`, which currently only supports Ascend.
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
        weight_decay (float, optional): weight decay (L2 penalty). Default: ``0.``.
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
        ValueError: If the `betas` not in the range of [0, 1).
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
                 weight_decay=0.0, amsgrad=False, *, maximize=False):
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
        self.assignadd = ops.AssignAdd()
        self.op_add = ops.AddN()
        self.op_mul = ops.Mul()
        self.op_pow = ops.Pow()
        self.adam_opt = ops.Adam(False, False)
        self.op_cast = ops.Cast()

    @jit
    def implementation(self, beta1, beta2, eps, lr, start_id, end_id, gradients, maximize, weight_decay):
        """Extract the common computing part for acceleration"""
        beta1_power = self.op_pow(beta1, self.state_step)
        beta2_power = self.op_pow(beta2, self.state_step)
        params = self.parameters[start_id: end_id]
        grads = tuple([grad if not maximize else ops.neg(grad) for grad in gradients[start_id: end_id]])
        grads = self._decay_weight(weight_decay, params, grads)
        self.hyper_map(ops.partial(_adam_opt, beta1_power, beta2_power, beta1, beta2, eps, lr),
                       grads, params,
                       self.exp_avg[start_id: end_id], self.exp_avg_sq[start_id: end_id])
        return True

    def construct(self, gradients):
        self.assignadd(self.state_step, self.increase_tensor)
        for group_id, group in enumerate(self.param_groups):
            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id + 1]

            lr = self.lrs[group_id]
            weight_decay = group.get("weight_decay")
            beta1, beta2 = group.get("betas")
            maximize = group.get("maximize")
            eps = group.get("eps")

            if isinstance(group.get("lr"), float):
                lr = self.op_cast(group.get("lr"), mstype.float32)

            self.implementation(beta1, beta2, eps, lr, start_id, end_id, gradients, maximize, weight_decay)

        return True
