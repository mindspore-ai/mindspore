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
"""PROXIMAL_ADA_GRAD"""
from __future__ import absolute_import

from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common import Tensor
import mindspore.common.dtype as mstype
from mindspore.common.api import jit
from mindspore._checkparam import Validator as validator
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register

_proximal_ada_grad_opt = C.MultitypeFuncGraph("proximal_ada_grad_opt")

@_proximal_ada_grad_opt.register("Function", "Function", "Tensor", "Tensor", "Tensor", "RowTensor", "Tensor",
                                 "Tensor")
def _tensor_run_opt_with_sparse(opt, sparse_opt, l1, l2, learning_rate, gradient, weight, accum):
    """Apply sparse proximal_ada_grad optimizer to the weight parameter."""
    success = True
    success = F.depend(success, sparse_opt(weight, accum, learning_rate, l1, l2, gradient.values, gradient.indices))
    return success


@_proximal_ada_grad_opt.register("Function", "Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt(opt, sparse_opt, l1, l2, learning_rate, gradient, weight, accum):
    """Apply proximal_ada_grad optimizer to the weight parameter."""
    success = True
    success = F.depend(success, opt(weight, accum, learning_rate, l1, l2, gradient))
    return success


def _check_param_value(accum, l1, l2, use_locking, prim_name=None):
    """Check inputs param."""
    validator.check_value_type("accum", accum, [float], prim_name)
    validator.check_value_type("l1", l1, [float], prim_name)
    validator.check_value_type("l2", l2, [float], prim_name)
    validator.check_value_type("use_locking", use_locking, [bool], prim_name)
    validator.check_non_negative_float(accum, "accum", prim_name)
    validator.check_non_negative_float(l1, "l1", prim_name)
    validator.check_non_negative_float(l2, "l2", prim_name)


class ProximalAdagrad(Optimizer):
    r"""
    Implements the ProximalAdagrad algorithm.

    ProximalAdagrad is an online Learning and Stochastic Optimization.
    Refer to paper `Efficient Learning using Forward-Backward Splitting
    <http://papers.nips.cc//paper/3793-efficient-learning-using-forward-backward-splitting.pdf>`_.

    .. math::
        accum_{t+1} = accum_{t} + g * g

    .. math::
        \text{prox_v} = w_{t} - \gamma * g * \frac{1}{\sqrt{accum_{t+1}}}

    .. math::
        w_{t+1} = \frac{sign(\text{prox_v})}{1 + \gamma * l2} * \max(\left| \text{prox_v} \right| - \gamma * l1, 0)

    Here : where :math:`g` , :math:`\gamma`, :math:`w` , :math:`accum` and :math:`t` denote the `grads`,
    `learning_rate`, `params`, accumulation and current step respectively.

    Note:
        The sparse strategy is applied while the SparseGatherV2 operator is used for forward network. If the sparse
        strategy wants to be executed on the host, set the target to the CPU.
        The sparse feature is under continuous development.

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

            - order_params: Optional. When parameters are grouped, this usually is used to maintain the order of
              parameters that appeared in the network to improve performance. The value should be parameters whose
              order will be followed in optimizer.
              If `order_params` in the keys, other keys will be ignored and the element of 'order_params' must be in
              one group of `params`.

        accum (float): The starting value for accumulators `accum`, must be zero or positive values. Default: 0.1.
        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule]): Default: 0.001.

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of the current step.

        l1 (float): l1 regularization strength, must be greater than or equal to zero. Default: 0.0.
        l2 (float): l2 regularization strength, must be greater than or equal to zero. Default: 0.0.
        use_locking (bool): If true, use locks for updating operation. Default: False.
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
        TypeError: If `accum`, `l1`, `l2` or `loss_scale` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        ValueError: If `loss_scale` is less than or equal to 0.
        ValueError: If `accum`, `l1`, `l2` or `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>>
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.ProximalAdagrad(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = nn.ProximalAdagrad(group_params, learning_rate=0.1, weight_decay=0.0)
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
    def __init__(self, params, accum=0.1, learning_rate=0.001, l1=0.0, l2=0.0,
                 use_locking=False, loss_scale=1.0, weight_decay=0.0):
        super(ProximalAdagrad, self).__init__(learning_rate, params, weight_decay, loss_scale)
        _check_param_value(accum, l1, l2, use_locking, self.cls_name)
        self.accum = self._parameters.clone(prefix="accum", init=accum)
        self.l1 = Tensor(l1, mstype.float32)
        self.l2 = Tensor(l2, mstype.float32)
        self.use_locking = use_locking
        self.opt = P.ApplyProximalAdagrad(use_locking=use_locking)
        self.sparse_opt = P.SparseApplyProximalAdagrad(use_locking=use_locking)

    @jit
    def construct(self, grads):
        params = self._parameters
        accum = self.accum
        grads = self.flatten_gradients(grads)
        grads = self.decay_weight(grads)
        grads = self.gradients_centralization(grads)
        grads = self.scale_grad(grads)
        grads = self._grad_sparse_indices_deduplicate(grads)
        lr = self.get_lr()
        if self.is_group_lr:
            success = self.map_reverse(F.partial(_proximal_ada_grad_opt, self.opt, self.sparse_opt, self.l1, self.l2),
                                       lr, grads, params, accum)
        else:
            success = self.map_reverse(F.partial(_proximal_ada_grad_opt, self.opt, self.sparse_opt, self.l1, self.l2,
                                                 lr),
                                       grads, params, accum)
        return success

    @Optimizer.target.setter
    def target(self, value):
        """
        If the input value is set to "CPU", the parameters will be updated on the host using the Fused
        optimizer operation.
        """
        if not isinstance(value, str):
            raise TypeError("For 'ProximalAdagrad', the property 'target' must be string type, "
                            "but got {}".format(type(value)))

        if value not in ('CPU', 'Ascend', 'GPU'):
            raise ValueError("For 'ProximalAdagrad', the property 'target' must be 'CPU', 'Ascend' or 'GPU', "
                             "but got {}.".format(value))

        if value == 'CPU':
            self.sparse_opt = P.FusedSparseProximalAdagrad(self.use_locking)
            self.sparse_opt.set_device("CPU")
        else:
            self.sparse_opt = P.SparseApplyProximalAdagrad(self.use_locking)

        self._target = value
