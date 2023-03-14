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
"""momentum"""
from __future__ import absolute_import

from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.common.api import jit
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register
from mindspore.nn.optim._dist_optimizer_registry import _register_dist_optimizer


_momentum_opt = C.MultitypeFuncGraph("momentum_opt")


@_momentum_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool")
def _tensor_run_opt_ext(opt, momentum, learning_rate, gradient, weight, moment, ps_parameter, cache_enable):
    """Apply momentum optimizer to the weight parameter using Tensor."""
    if ps_parameter and not cache_enable:
        op_shape = P.Shape()
        _ps_pull = P.Pull()
        _ps_push = P.Push("ApplyMomentum", [])
        shapes = (op_shape(learning_rate), op_shape(gradient), op_shape(momentum))
        success = F.depend(True, _ps_pull(_ps_push((learning_rate, gradient, momentum), shapes), weight))
    else:
        success = F.depend(True, opt(weight, moment, learning_rate, gradient, momentum))
    return success


@_momentum_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool",
                        "Function", "Bool")
def _tensor_run_opt_ext_dist(opt, momentum, learning_rate, gradient, weight, moment, ps_parameter, cache_enable,
                             distributed_opt, use_flag):
    """Apply momentum optimizer to the weight parameter using Tensor."""
    if use_flag:
        success = F.depend(True, distributed_opt(weight, moment, learning_rate, gradient, momentum))
    elif ps_parameter and not cache_enable:
        op_shape = P.Shape()
        _ps_pull = P.Pull()
        _ps_push = P.Push("ApplyMomentum", [])
        shapes = (op_shape(learning_rate), op_shape(gradient), op_shape(momentum))
        success = F.depend(True, _ps_pull(_ps_push((learning_rate, gradient, momentum), shapes), weight))
    else:
        success = F.depend(True, opt(weight, moment, learning_rate, gradient, momentum))
    return success


class Momentum(Optimizer):
    r"""
    Implements the Momentum algorithm.

    Refer to the paper `On the importance of initialization and momentum in deep
    learning <https://dl.acm.org/doi/10.5555/3042817.3043064>`_ for more details.

    .. math::
        v_{t+1} = v_{t} \ast u + grad

    If use_nesterov is True:

    .. math::
        p_{t+1} =  p_{t} - (grad \ast lr + v_{t+1} \ast u \ast lr)

    If use_nesterov is False:

    .. math::
        p_{t+1} = p_{t} - lr \ast v_{t+1}

    Here: where :math:`grad`, :math:`lr`, :math:`p`, :math:`v` and :math:`u` denote the gradients,
    learning_rate, params, moments, and momentum respectively.

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

        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule]):

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        momentum (float): Hyperparameter of type float, means momentum for the moving average.
            It must be at least 0.0.

        weight_decay (Union[float, int, Cell]): Weight decay (L2 penalty). Default: 0.0.

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

        loss_scale (float): A floating point value for the loss scale. It must be greater than 0.0. In general, use the
            default value. Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to False, then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.amp.FixedLossScaleManager` for more details.
            Default: 1.0.
        use_nesterov (bool): Enable Nesterov momentum. Default: False.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[bool]. All elements are True.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `loss_scale` or `momentum` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        TypeError: If `use_nesterov` is not a bool.
        ValueError: If `loss_scale` is less than or equal to 0.
        ValueError: If `weight_decay` or `momentum` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>>
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = nn.Momentum(group_params, learning_rate=0.1, momentum=0.9, weight_decay=0.0)
        >>> # The conv_params's parameters will use a learning rate of default value 0.1 and a weight decay of 0.01 and
        >>> # grad centralization of True.
        >>> # The no_conv_params's parameters will use a learning rate of 0.01 and a weight decay of default value 0.0
        >>> # and grad centralization of False..
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = ms.Model(net, loss_fn=loss, optimizer=optim, metrics=None)
    """
    @opt_init_args_register
    def __init__(self, params, learning_rate, momentum, weight_decay=0.0, loss_scale=1.0, use_nesterov=False):
        super(Momentum, self).__init__(learning_rate, params, weight_decay, loss_scale)
        Validator.check_value_type("momentum", momentum, [float], self.cls_name)
        if isinstance(momentum, float) and momentum < 0.0:
            raise ValueError("For 'Momentum', the argument 'momentum' must be at least 0.0, "
                             "but got {}".format(momentum))
        self.momentum = Parameter(Tensor(momentum, mstype.float32), name="momentum")
        self.params = self._parameters
        self.use_nesterov = Validator.check_bool(use_nesterov)
        self.moments = self.params.clone(prefix="moments", init='zeros')
        self.opt = P.ApplyMomentum(use_nesterov=self.use_nesterov)

        self.distributed_opts, self.use_distributed_opt_flags =\
        self._get_distributed_optimizer_list("momentum", use_nesterov=self.use_nesterov)
        self.use_dist_optimizer = self._use_distibuted_optimizer()

    @jit
    def construct(self, gradients):
        params = self.params
        moments = self.moments
        gradients = self.flatten_gradients(gradients)
        gradients = self.decay_weight(gradients)
        gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)
        lr = self.get_lr()
        if self.use_dist_optimizer:
            if self.is_group_lr:
                success = self.hyper_map_reverse(F.partial(_momentum_opt, self.opt, self.momentum),
                                                 lr, gradients, params, moments, self.ps_parameters, self.cache_enable,
                                                 self.distributed_opts, self.use_distributed_opt_flags)
            else:
                success = self.hyper_map_reverse(F.partial(_momentum_opt, self.opt, self.momentum, lr),
                                                 gradients, params, moments, self.ps_parameters, self.cache_enable,
                                                 self.distributed_opts, self.use_distributed_opt_flags)
        else:
            if self.is_group_lr:
                success = self.hyper_map_reverse(F.partial(_momentum_opt, self.opt, self.momentum),
                                                 lr, gradients, params, moments, self.ps_parameters, self.cache_enable)
            else:
                success = self.hyper_map_reverse(F.partial(_momentum_opt, self.opt, self.momentum, lr),
                                                 gradients, params, moments, self.ps_parameters, self.cache_enable)
        return success


def _create_distributed_momentum(*args, **kwargs):
    momentum = P.ApplyMomentum(*args, **kwargs)
    momentum.add_prim_attr("gradient_type", "dense_gradient")
    momentum.add_prim_attr("parameter_input_index", 0)
    momentum.add_prim_attr("gradient_input_index", 3)
    return momentum


_register_dist_optimizer("momentum", _create_distributed_momentum)
