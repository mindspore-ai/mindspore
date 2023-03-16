# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""rprop"""
from __future__ import absolute_import

from mindspore import ops
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindspore.common.api import jit
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register


class Rprop(Optimizer):
    r"""
    Implements Resilient backpropagation.

    Further information about this implementation can be found at  `A Direct Adaptive Method for Faster Backpropagation
    Learning: The RPROP Algorithm <https://ieeexplore.ieee.org/document/298623>`_.

    The updating formulas are as follows:

    .. math::
        \begin{gather*}
            &\hspace{-10mm}  \textbf{if} \:   g_{t-1} g_t  > 0                                     \\
            &\hspace{25mm}  \Delta_t \leftarrow \mathrm{min}(\Delta_{t-1} \eta_{+}, \Delta_{max})  \\
            &\hspace{0mm}  \textbf{else if}  \:  g_{t-1} g_t < 0                                   \\
            &\hspace{25mm}  \Delta_t \leftarrow \mathrm{max}(\Delta_{t-1} \eta_{-}, \Delta_{min})  \\
            &\hspace{-25mm}  \textbf{else}  \:                                                     \\
            &\hspace{-5mm}  \Delta_t \leftarrow \Delta_{t-1}                                       \\
            &\hspace{15mm} w_{t} \leftarrow w_{t-1}- \Delta_{t} \mathrm{sign}(g_t)                 \\
        \end{gather*}

    :math:`\Delta_{min/max}` represents the min/max step size, :math:`\eta_{+/-}` represents the factors of
    etaminus and etaplus, :math:`g` represents `gradients`, :math:`w` represents `parameters`.

    Note:
        If parameters are not grouped, the `weight_decay` in optimizer will be applied on the parameters without 'beta'
        or 'gamma' in their names.

        Users can group parameters to change the strategy of decaying weight.

        When parameters are grouped, each group can set `weight_decay`. If not, the `weight_decay` in optimizer will be
        applied.

    Args:
        params (Union[list[Parameter], list[dict]]): Must be list of `Parameter` or list of `dict`. When the
            `parameters` is a list of `dict`, the "params", "lr", "weight_decay", "grad_centralization" and
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

        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule]): Learning_rate. Default: 0.1.

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        etas (tuple[float, float]): The factor of multiplicative increasing or
            descreasing(etaminus, etaplus). Default: (0.5, 1.2).
        step_sizes(tuple[float, float]): The allowed minimal and maximal step size(min_step_sizes, max_step_size).
            Default: (1e-6, 50.).
        weight_decay (Union[float, int, Cell]): Weight decay (L2 penalty). Default: 0.0.

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        Tensor[bool], the value is True.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `step_sizes` or `etas` is not a tuple.
        ValueError: If maximal step size is less than minimal step size.
        ValueError: If the length of `step_sizes` or `etas` is not equal to 2.
        TypeError: If  the element in `etas` or `step_sizes` is not a float.
        ValueError: If `etaminus` is not in the range of (0, 1) or `etaplus` is not greater than 1.
        TypeError: If `weight_decay` is neither float nor int.
        ValueError: If `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>>
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.Rprop(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params,'grad_centralization':True},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = nn.Rprop(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The conv_params's parameters will use default learning rate of 0.1 default weight decay of 0.0 and grad
        >>> # centralization of True.
        >>> # The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0 and grad
        >>> # centralization of False.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = ms.Model(net, loss_fn=loss, optimizer=optim)
    """

    @opt_init_args_register
    def __init__(self, params, learning_rate=0.1, etas=(0.5, 1.2), step_sizes=(1e-6, 50.), weight_decay=0.):

        super(Rprop, self).__init__(learning_rate, params, weight_decay)
        if not isinstance(etas, tuple):
            raise TypeError("For Rprop, etas must be a tuple, but got {}.".format(type(etas)))
        if len(etas) != 2:
            raise ValueError("For Rprop, etas must be a tuple with the size of 2, but got {}.".format(len(etas)))

        if not isinstance(step_sizes, tuple):
            raise TypeError("For Rprop, step_sizes must be a tuple, but got {}.".format(type(etas)))
        if len(step_sizes) != 2:
            raise ValueError("For Rprop, step_sizes must be a tuple with the size of 2, "
                             "but got {}.".format(len(step_sizes)))

        if step_sizes[0] > step_sizes[1]:
            raise ValueError("For Rprop, maximal step size should not be less than minimal step size, "
                             "but got {} > {}.".format(step_sizes[0], step_sizes[1]))

        validator.check_float_range(etas[0], 0.0, 1.0, Rel.INC_NEITHER, "etaminus", self.cls_name)
        validator.check_value_type("etaplus", etas[1], [float], self.cls_name)
        if etas[1] <= 1.0:
            raise ValueError("For Rprop, etaplus must be greater than 1.0, but got etaplus {}.".format(etas[1]))

        validator.check_value_type("min_step_sizes", step_sizes[0], [float], self.cls_name)
        validator.check_value_type("max_step_sizes", step_sizes[1], [float], self.cls_name)

        self.etaminus, self.etaplus = etas
        self.step_size_min, self.step_size_max = step_sizes
        self.prev = self._parameters.clone(prefix="prev", init='zeros')
        self.step_size = self._parameters.clone(prefix="step_size", init='zeros')

        self.fill = P.Fill()
        self.sign = P.Sign()
        self.assign = P.Assign()
        self.assignadd = P.AssignAdd()
        self.cast = P.Cast()
        self.select = P.Select()
        self.ones_like = P.OnesLike()

    @jit
    def construct(self, gradients):
        gradients = self.flatten_gradients(gradients)
        gradients = self.decay_weight(gradients)
        gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)
        lrs = self.get_lr()
        if not self._is_dynamic_lr_or_weight_decay():
            self.assignadd(self.global_step, self.global_step_increase_tensor)
        success = True

        for index, (grad, param, prev, step_size) in enumerate(zip(gradients, self._parameters,
                                                                   self.prev, self.step_size)):
            lr = lrs[index] if self.is_group_lr else lrs

            if self.global_step == 1:
                step_size_fp32 = self.ones_like(step_size) * lr
            else:
                step_size_fp32 = self.cast(step_size, mstype.float32)

            gradient_fp32 = self.cast(grad, mstype.float32)
            param_fp32 = self.cast(param, mstype.float32)

            sign = self.sign(gradient_fp32 * prev)
            sign = self.select(sign > 0, self.fill(mstype.float32, sign.shape, self.etaplus), sign)
            sign = self.select(sign < 0, self.fill(mstype.float32, sign.shape, self.etaminus), sign)
            sign = self.select(sign == 0, self.fill(mstype.float32, sign.shape, 1.), sign)

            step_size_fp32 = ops.clip_by_value(step_size_fp32 * sign, self.step_size_min, self.step_size_max)

            gradient_update = self.select(sign == self.etaminus, self.fill(mstype.float32, sign.shape, 0.),
                                          gradient_fp32)
            next_param = param_fp32 - self.sign(gradient_update) * step_size_fp32

            self.assign(param, self.cast(next_param, param.dtype))
            self.assign(prev, self.cast(gradient_update, prev.dtype))
            self.assign(step_size, self.cast(step_size_fp32, step_size.dtype))

        return success
