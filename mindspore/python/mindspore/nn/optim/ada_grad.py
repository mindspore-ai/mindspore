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
"""ADA_GRAD"""
from __future__ import absolute_import

from mindspore.ops import functional as F, composite as C, operations as P
from mindspore._checkparam import Validator as validator
from mindspore.common.api import jit
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register


_ada_grad_opt = C.MultitypeFuncGraph("ada_grad_opt")


@_ada_grad_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt(opt, learning_rate, weight, accum, gradient):
    """Apply ada_grad optimizer to the weight parameter."""
    success = True
    success = F.depend(success, opt(weight, accum, learning_rate, gradient))
    return success


def _check_param_value(accum, update_slots, prim_name=None):
    """Check inputs param."""
    validator.check_value_type("accum", accum, [float], prim_name)
    validator.check_value_type("update_slots", update_slots, [bool], prim_name)
    validator.check_non_negative_float(accum, "accum", prim_name)


class Adagrad(Optimizer):
    r"""
    Implements the Adagrad algorithm.

    Adagrad is an online Learning and Stochastic Optimization.
    Refer to paper `Efficient Learning using Forward-Backward Splitting
    <https://proceedings.neurips.cc/paper/2009/file/621bf66ddb7c962aa0d22ac97d69b793-Paper.pdf>`_.
    Adagrad can adaptively assign different learning rates to each parameter in response to the uneven number of
    samples for different parameters.

    The updating Pseudo codes are as follows:

    .. math::
        \begin{aligned} \\
            &\newline
            &\hline \\
            &\textbf{Parameters}: \text{learning rate } \gamma, \:  \text{ params } w_0, \:
                \: \text{ weight decay } \lambda, \\
            &\hspace{12mm} \text{ initial accumulator value } state\_sum\\
            &\textbf{Init}: state\_sum_0 \leftarrow 0 \\[-1.ex]
            &\newline
            &\hline \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do} \\
            &\hspace{5mm}g_t  \leftarrow  \nabla_{w} f_t (w_{t-1}) \\
            &\hspace{5mm} \textbf{if} \: \lambda \neq 0 \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda w_{t-1} \\
            &\hspace{5mm}state\_sum_t  \leftarrow  state\_sum_{t-1} + g^2_t \\
            &\hspace{5mm}w_t \leftarrow w_{t-1}- \gamma*\frac{g_t}{\sqrt{state\_sum_t} + \epsilon} \\
            &\newline
            &\hline \\
            &\bf{return} \:  w_t \\[-1.ex]
            &\newline
            &\hline \\
        \end{aligned}

    :math:`state\_sum` stands for the accumulated squared sum of the gradients :math:`accum`.
    :math:`g` stands for `grads`, :math:`\lambda` stands for `weight_decay`.
    :math:`\gamma` stands for `learning_rate`, :math:`w` stands for `params`.

    Note:
        If parameters are not grouped, the `weight_decay` in optimizer will be applied on the network parameters without
        'beta' or 'gamma' in their names. Users can group parameters to change the strategy of decaying weight. When
        parameters are grouped, each group can set `weight_decay`. If not, the `weight_decay` in optimizer will be
        applied.

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

        accum (float): The starting value for :math:`h`, must be zero or positive values. Default: 0.1.
        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule]): Default: 0.001.

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        update_slots (bool): Whether the :math:`h` will be updated. Default: True.
        loss_scale (float): Value for the loss scale. It must be greater than 0.0. In general, use the default value.
            Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to False, then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.amp.FixedLossScaleManager` for more details.
            Default: 1.0.
        weight_decay (Union[float, int, Cell]): Weight decay (L2 penalty). Default: 0.0.

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

    Inputs:
        - **grads** (tuple[Tensor]) - The gradients of `params` in the optimizer, the shape is the same as the `params`
          in optimizer.

    Outputs:
        Tensor[bool], the value is True.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `accum` or `loss_scale` is not a float.
        TypeError: If `update_slots` is not a bool.
        TypeError: If `weight_decay` is neither float nor int.
        ValueError: If `loss_scale` is less than or equal to 0.
        ValueError: If `accum` or `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>>
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.Adagrad(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = nn.Adagrad(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01 and grad
        >>> # centralization of True.
        >>> # The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0 and grad
        >>> # centralization of False.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = ms.Model(net, loss_fn=loss, optimizer=optim)
    """

    @opt_init_args_register
    def __init__(self, params, accum=0.1, learning_rate=0.001,
                 update_slots=True, loss_scale=1.0, weight_decay=0.0):
        super(Adagrad, self).__init__(learning_rate, params, weight_decay, loss_scale)
        _check_param_value(accum, update_slots, self.cls_name)
        self.accum = self._parameters.clone(prefix="accum", init=accum)
        self.opt = P.ApplyAdagrad(update_slots=update_slots)

    @jit
    def construct(self, grads):
        params = self._parameters
        accum = self.accum
        grads = self.flatten_gradients(grads)
        grads = self.decay_weight(grads)
        grads = self.gradients_centralization(grads)
        grads = self.scale_grad(grads)
        lr = self.get_lr()
        if self.is_group_lr:
            success = self.map_reverse(F.partial(_ada_grad_opt, self.opt), lr, params, accum,
                                       grads)
        else:
            success = self.map_reverse(F.partial(_ada_grad_opt, self.opt, lr), params, accum,
                                       grads)
        return success
