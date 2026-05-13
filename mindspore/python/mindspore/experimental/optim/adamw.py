# The code implementation refers to the following files from pytorch:
# - https://github.com/pytorch/pytorch/blob/v1.13.0/torch/optim/adamw.py
# Additional modifications are made by Huawei Technologies Co., Ltd in 2023.
# ============================================================================
"""adamw"""
from __future__ import absolute_import

from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.experimental.optim.optimizer import Optimizer
from mindspore import _checkparam as validator
from mindspore.ops import auto_generate as gen
from mindspore import ops
from mindspore import jit

_adamw_opt = ops.MultitypeFuncGraph("adamw_opt")
_speed_adamw_opt = ops.MultitypeFuncGraph("speed_adamw_opt")

op_mul = ops.Mul()
op_pow = ops.Pow()
op_sqrt = ops.Sqrt()
op_maximum = ops.Maximum()
hyper_map = ops.HyperMap()


@_speed_adamw_opt.register("Function", "Float", "Float", "Tensor", "Float", "Float", "Bool", "Bool", "Tensor", "Tensor",
                           "Tensor", "Tensor", "Tensor", "Tensor")
def _run_speed_adamw_opt(opt, beta1, beta2, lr, eps, weight_decay, amsgrad, maximize, bias_correction1,
                         bias_correction2, parameters, grads, exp_avg, exp_avg_sq):
    """Apply adamw optimizer to the weight parameter."""
    success = True
    opt(parameters, exp_avg, exp_avg_sq, bias_correction1, bias_correction2, lr, weight_decay, beta1, beta2, eps,
        grads, None, amsgrad, maximize)
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


@jit
def prepare_func(lr, weight_decay, state_step, beta1, beta2):
    weight_decay_new = 1 - lr * weight_decay
    bias_correction1 = 1 - op_pow(beta1, state_step)
    bias_correction2 = 1 - op_pow(beta2, state_step)
    step_size = lr / bias_correction1
    bias_correction2_sqrt = op_sqrt(bias_correction2)
    return weight_decay_new, step_size, bias_correction2_sqrt


@_adamw_opt.register("Tensor", "Tensor", "Bool", "Float", "Tensor", "Float", "Float", "Tensor", "Tensor",
                     "Tensor", "Tensor", "Tensor")
def _run_adamw_opt(weight_decay_new, step_size, amsgrad, eps, bias_correction2_sqrt, beta1, beta2, param, grad,
                   exp_avg, exp_avg_sq, max_exp_avg_sq):
    """Apply adamw optimizer to the weight parameter."""
    success = True
    next_param = op_mul(param, weight_decay_new)
    ops.assign(exp_avg, op_mul(exp_avg, beta1) + op_mul(grad, 1 - beta1))
    ops.assign(exp_avg_sq, ops.addcmul(op_mul(exp_avg_sq, beta2), grad, grad, 1 - beta2))

    if amsgrad:
        next_max_exp_avg = op_maximum(max_exp_avg_sq, exp_avg_sq)
        denom = op_sqrt(next_max_exp_avg) / bias_correction2_sqrt + eps
        ops.assign(max_exp_avg_sq, next_max_exp_avg)
    else:
        denom = op_sqrt(exp_avg_sq) / bias_correction2_sqrt + eps

    return_param = next_param - op_mul(exp_avg / denom, step_size)
    ops.assign(param, return_param)
    return success


class AdamW(Optimizer):
    r"""
    Implements Adam Weight Decay algorithm.

    .. math::
        \begin{aligned}
            &\rule{180mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
                \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
                \: \epsilon \text{ (epsilon)}                                                    \\
            &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
                \: \textit{maximize}                                                             \\
            &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
                \text{ ( second moment)}, \: \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
            &\rule{180mm}{0.4pt}                                                                 \\
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
            &\rule{180mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{180mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    More details of the AdamW algorithm can be found in the paper `Decoupled Weight Decay Regularization
    <https://arxiv.org/abs/1711.05101>`_ and `On the Convergence of Adam and Beyond
    <https://openreview.net/forum?id=ryQu7f-RZ>`_.

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
        weight_decay (float, optional): weight decay (L2 penalty). Default: ``0.``.
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
        ValueError: If the `betas` not in the range of [0, 1).
        ValueError: If the `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import nn
        >>> from mindspore.experimental import optim
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://atomgit.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
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

        defaults = {"lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "amsgrad": amsgrad,
                "maximize": maximize}
        super().__init__(params, defaults)

        self.exp_avg = self.parameters.clone(prefix="exp_avg", init='zeros')
        self.exp_avg_sq = self.parameters.clone(prefix="exp_avg_sq", init='zeros')
        self.max_exp_avg_sq = self.parameters.clone(prefix="max_exp_avg_sq", init='zeros')
        self.state_step = Parameter(Tensor(0, mstype.int32), "state_step")
        self.increase_tensor = Tensor(1, mstype.int32)
        self.assignadd = ops.AssignAdd()
        self.op_cast = ops.Cast()

    @jit
    def implementation(self, lr, weight_decay, beta1, beta2, amsgrad, eps, grads, start_id, end_id):
        """Extract the common computing part for acceleration"""
        weight_decay_new, step_size, bias_correction2_sqrt = prepare_func(lr, weight_decay,
                                                                          self.state_step, beta1, beta2)
        self.hyper_map(ops.partial(_adamw_opt, weight_decay_new, step_size, amsgrad,
                                   eps, bias_correction2_sqrt, beta1, beta2),
                       self.parameters[start_id: end_id], grads, self.exp_avg[start_id: end_id],
                       self.exp_avg_sq[start_id: end_id], self.max_exp_avg_sq[start_id: end_id])
        return True

    def construct(self, gradients):
        self.assignadd(self.state_step, self.increase_tensor)
        for group_id, group in enumerate(self.param_groups):
            beta1, beta2 = group['betas']
            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id + 1]
            lr = self.lrs[group_id]
            if isinstance(group.get("lr"), float):
                lr = self.op_cast(group.get("lr"), mstype.float32)
            grads = tuple(grad if not group.get("maximize") else ops.neg(grad) \
                          for grad in gradients[start_id:end_id])

            self.implementation(lr, group.get("weight_decay"), beta1, beta2, group.get("amsgrad"), group.get("eps"),
                                grads, start_id, end_id)

        return True


class SpeedAdamW(Optimizer):
    r"""
    Implements Adam Weight Decay algorithm.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, *, maximize=False):
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

        defaults = {"lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "amsgrad": amsgrad,
                "maximize": maximize}
        self.max_v_group = True
        super().__init__(params, defaults)

        self.exp_avg = self.parameters.clone(prefix="exp_avg", init='zeros')
        self.exp_avg_sq = self.parameters.clone(prefix="exp_avg_sq", init='zeros')
        self.state_step = Parameter(Tensor([0], mstype.float32), "state_step")
        self.increase_tensor = Tensor(1, mstype.float32)
        self.assignadd = ops.AssignAdd()
        self.adamw_opt = gen.ApplyAdamW()

    def construct(self, gradients):
        self.assignadd(self.state_step, self.increase_tensor)
        for group_id, group in enumerate(self.param_groups):
            beta1, beta2 = group['betas']
            maximize = group.get("maximize")
            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id + 1]
            lr = group.get("lr")
            grads = tuple(gradients[start_id: end_id])

            bias_correction1 = float(beta1) ** (float(self.state_step) - 1.0)
            bias_correction2 = float(beta2) ** (float(self.state_step) - 1.0)

            # 当前 ApplyAdamW 仅支持 amsgrad 为 False
            if group.get("amsgrad"):
                raise ValueError("For SpeedAdamW, the value of amsgrad can only be False.")

            self.hyper_map(ops.partial(_speed_adamw_opt, self.adamw_opt, beta1, beta2, lr,
                                       group.get("eps"), group.get("weight_decay"),
                                       group.get("amsgrad"), maximize, bias_correction1, bias_correction2),
                           self.parameters[start_id: end_id], grads, self.exp_avg[start_id: end_id],
                           self.exp_avg_sq[start_id: end_id])

        return True
