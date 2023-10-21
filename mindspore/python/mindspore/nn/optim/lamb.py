# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""lamb"""
from __future__ import absolute_import

import numpy as np

from mindspore import context
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner
from mindspore.common.tensor import Tensor
from mindspore.common.api import jit
from mindspore import _checkparam as validator
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register

_lamb_opt = C.MultitypeFuncGraph("lamb_opt")


@_lamb_opt.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Bool", "Bool")
def _update_run_op(beta1, beta2, eps, global_step, lr, weight_decay, param, m, v, gradient, decay_flag, optim_filter):
    """
    Update parameters.

    Args:
        beta1 (Tensor): The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).
        beta2 (Tensor): The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).
        eps (Tensor): Term added to the denominator to improve numerical stability. Should be greater than 0.
        lr (Tensor): Learning rate.
        weight_decay (numbers.Number): Weight decay. Should be equal to or greater than 0.
        global_step (Tensor): Global step.
        param (Tensor): Parameters.
        m (Tensor): m value of parameters.
        v (Tensor): v value of parameters.
        gradient (Tensor): Gradient of parameters.
        decay_flag (bool): Specifies whether param update with weight decay.
        optim_filter(bool): Applies parameter update or not.

    Returns:
        Tensor, the new value of v after updating.
    """
    if optim_filter:
        op_lamb = inner.Lamb()
        if decay_flag:
            ret = op_lamb(param, m, v, lr, beta1, beta2, eps, weight_decay, global_step, gradient)
        else:
            ret = op_lamb(param, m, v, lr, beta1, beta2, eps, 0.0, global_step, gradient)
    else:
        ret = gradient
    return ret


def _check_param_value(beta1, beta2, eps, prim_name):
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_float_range(beta1, 0.0, 1.0, validator.INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(beta2, 0.0, 1.0, validator.INC_NEITHER, "beta2", prim_name)
    validator.check_positive_float(eps, "eps", prim_name)


class Lamb(Optimizer):
    r"""
    Implements the Lamb(Layer-wise Adaptive Moments optimizer for Batching training) algorithm.

    LAMB is an optimization algorithm employing a layerwise adaptive large batch optimization technique.
    Refer to the paper `LARGE BATCH OPTIMIZATION FOR DEEP LEARNING: TRAINING BERT IN 76
    MINUTES <https://arxiv.org/abs/1904.00962>`_.

    The LAMB optimizer aims to increase the training batch size without reducing the accuracy,
    and it supports adaptive element-by-element update and accurate layered correction.

    The updating of parameters follows:

    ..  math::
        \begin{array}{l}
            &\newline
            &\hline \\
            &\textbf{Parameters}:   \: 1^{\text {st }}\text {moment vector} \: m , \: 2^{\text {nd}} \:
             \text{moment vector} \: v , \\
            &\hspace{5mm}\text{learning rate }  \left\{ \gamma_{t}\right\}_{t=1}^{T} , \: \text
             {exponential decay rates for the moment estimates} \: \beta_{1} \: \beta_{2} , \\
            &\hspace{5mm}\text{scaling function } \phi \\
            &\textbf{Init}: \boldsymbol{m}_{0} \leftarrow 0, \: \boldsymbol{v}_{0} \leftarrow 0 \\[-1.ex]
            &\newline
            &\hline \\
            &\textbf{for} \text { t=1  to  T } \textbf{do} \\
            &\hspace{5mm}\text{Draw b samples } S_{t} \text{ from } \mathbb{P} \text{ . } \\
            &\hspace{5mm}\text{Compute } g_{t}=\frac{1}{\left|\mathcal{S}_{t}\right|} \sum_{s_{t} \in \mathcal{S}_{t}}
             \nabla \ell\left(x_{t}, s_{t}\right) . \\
            &\hspace{5mm}\boldsymbol{m}_{t} \leftarrow \beta_{1} \boldsymbol{m}_{t-1}+\left(1-\beta_{1}\right)
             \boldsymbol{g}_{t} \\
            &\hspace{5mm}\boldsymbol{v}_{t} \leftarrow \beta_{2} \boldsymbol{v}_{t-1}+\left(1-\beta_{2}\right)
             \boldsymbol{g}_{t}^{2} \\
            &\hspace{5mm}\hat{\boldsymbol{m}}_{t} \leftarrow \boldsymbol{m}_{t} /\left(1-\beta_{1}^{t}\right) \\
            &\hspace{5mm}\hat{\boldsymbol{v}}_{t} \leftarrow \boldsymbol{v}_{t} /\left(1-\beta_{2}^{t}\right) \\
            &\hspace{5mm}\text{Compute ratio } \boldsymbol{r}_{t}=\hat{\boldsymbol{m}}_{t}
             /(\sqrt{\hat{\boldsymbol{v}}_{t}}+\epsilon) \\
            &\hspace{5mm}\boldsymbol{w}_{t+1}^{(i)}=\boldsymbol{w}_{t}^{(i)}- \gamma_{t}
             \frac{\boldsymbol{\phi}\left(\left\|\boldsymbol{w}_{t}^{(i)}\right\|\right)}
             {\left\|\boldsymbol{r}_{t}^{(i)}+\lambda \boldsymbol{w}_{t}^{(i)}\right\|}\left(\boldsymbol{r}_{t}^{(i)}+
             \lambda \boldsymbol{w}_{t}^{(i)}\right) \\
            &\textbf{end for} \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
            &\textbf{return} \: \boldsymbol{w}_{t+1}\\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
        \end{array}

    :math:`m` represents the 1st moment vector `moment1`, :math:`v` represents the 2nd moment vector `moment2`,
    :math:`g` represents `gradients`, :math:`\beta_1, \beta_2` represent `beta1` and `beta2`,
    :math:`t` represents the current step while :math:`beta_1^t` and :math:`beta_2^t` represent
    `beta1_power` and `beta2_power`, :math:`\gamma` represents `learning_rate`, :math:`w` represents `params`,
    :math:`\epsilon` represents `eps`, :math:`\lambda` represents `weight_decay`.

    Note:
        There is usually no connection between a optimizer and mixed precision. But when `FixedLossScaleManager` is used
        and `drop_overflow_update` in `FixedLossScaleManager` is set to False, optimizer needs to set the 'loss_scale'.
        As this optimizer has no argument of `loss_scale`, so `loss_scale` needs to be processed by other means. Refer
        document `LossScale <https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html>`_ to
        process `loss_scale` correctly.

        If parameters are not grouped, the `weight_decay` in optimizer will be applied on the network parameters without
        'beta' or 'gamma' in their names. Users can group parameters to change the strategy of decaying weight. When
        parameters are grouped, each group can set `weight_decay`. If not, the `weight_decay` in optimizer will be
        applied.

    .. warning::
        The update process of the Lamb optimizer is not completely elementwise, and the sharding of weights in
        distributed parallel may affect the update result.

    Args:
        params (Union[list[Parameter], list[dict]]): Must be list of `Parameter` or list of `dict`. When the
            `params` is a list of `dict`, the string "params", "lr", "weight_decay", "grad_centralization" and
            "order_params" are the keys can be parsed.

            - params: Required. Parameters in current group. The value must be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in optimizer will be used. Fixed and dynamic learning rate are supported.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the optimizer will be used. It should be noted that weight
              decay can be a constant value or a Cell. It is a Cell only when dynamic weight decay is applied. Dynamic
              weight decay is similar to dynamic learning rate, users need to customize a weight decay schedule only
              with global step as input, and during training, the optimizer calls the instance of WeightDecaySchedule
              to get the weight decay value of current step.

            - grad_centralization: Optional. Must be Boolean. If "grad_centralization" is in the keys, the set value
              will be used. If not, the `grad_centralization` is False by default. This configuration only works on the
              convolution layer.

            - order_params: Optional. When parameters is grouped, this usually is used to maintain the order of
              parameters that appeared in the network to improve performance. The value should be parameters whose
              order will be followed in optimizer.
              If `order_params` in the keys, other keys will be ignored and the element of 'order_params' must be in
              one group of `params`.

        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule]):

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        beta1 (float): The exponential decay rate for the 1st moment estimations. Default: ``0.9`` .
            Should be in range (0.0, 1.0).
        beta2 (float): The exponential decay rate for the 2nd moment estimations. Default: ``0.999`` .
            Should be in range (0.0, 1.0).
        eps (float): Term added to the denominator to improve numerical stability. Default: ``1e-6`` .
            Should be greater than 0.

        weight_decay (Union[float, int, Cell]): Weight decay (L2 penalty). Default: ``0.0`` .

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[bool], all elements are ``True`` .

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `beta1`, `beta2` or `eps` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        ValueError: If `eps` is less than or equal to 0.
        ValueError: If `beta1`, `beta2` is not in range (0.0, 1.0).
        ValueError: If `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.Lamb(params=net.trainable_params(), learning_rate=0.1)
        >>>
        >>> #2) Use parameter groups and set different values
        >>> poly_decay_lr = nn.PolynomialDecayLR(learning_rate=0.1, end_learning_rate=0.01,
        ...                                                    decay_steps=4, power = 0.5)
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
        ...                 {'params': no_conv_params, 'lr': poly_decay_lr},
        ...                 {'order_params': net.trainable_params(0.01)}]
        >>> optim = nn.Lamb(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01 and grad
        >>> # centralization of True.
        >>> # The no_conv_params's parameters will use dynamic learning rate of poly decay learning rate and default
        >>> # weight decay of 0.0 and grad centralization of False.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = ms.train.Model(net, loss_fn=loss, optimizer=optim)
    """
    _support_parallel_optimizer = True

    @opt_init_args_register
    def __init__(self, params, learning_rate, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(Lamb, self).__init__(learning_rate, params, weight_decay)
        _check_param_value(beta1, beta2, eps, self.cls_name)

        # turn them to scalar when me support scalar/tensor mix operations
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.params = self._parameters
        self.moments1 = self.params.clone(prefix="lamb_m", init='zeros')
        self.moments2 = self.params.clone(prefix="lamb_v", init='zeros')
        self.device_ascend = context.get_context("device_target") == "Ascend"

    @jit
    def construct(self, gradients):
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)
        lamb_opt = _lamb_opt
        gradients = self.flatten_gradients(gradients)
        gradients = self.gradients_centralization(gradients)
        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(F.partial(lamb_opt, self.beta1, self.beta2, self.eps,
                                                        self.global_step),
                                              lr, weight_decay, self.params, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.hyper_map(F.partial(lamb_opt, self.beta1, self.beta2, self.eps,
                                                        self.global_step, lr),
                                              weight_decay, self.params, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
        else:
            optim_result = self.hyper_map(F.partial(lamb_opt, self.beta1, self.beta2, self.eps,
                                                    self.global_step, lr, weight_decay),
                                          self.params, self.moments1, self.moments2, gradients,
                                          self.decay_flags, self.optim_filter)

        if self.use_parallel:
            optim_result = F.depend(optim_result, self.broadcast_params(optim_result))

        return optim_result
