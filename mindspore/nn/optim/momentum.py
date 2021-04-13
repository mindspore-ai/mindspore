# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.ops import _selected_ops
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator
from .optimizer import Optimizer

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


class Momentum(Optimizer):
    r"""
    Implements the Momentum algorithm.

    Refer to the paper on the importance of initialization and momentum in deep learning for more details.

    .. math::
            v_{t} = v_{t-1} \ast u + gradients

    If use_nesterov is True:

    .. math::
            p_{t} =  p_{t-1} - (grad \ast lr + v_{t} \ast u \ast lr)

    If use_nesterov is False:

    .. math::
            p_{t} = p_{t-1} - lr \ast v_{t}

    Here: where grad, lr, p, v and u denote the gradients, learning_rate, params, moments, and momentum respectively.

    Note:
        When separating parameter groups, the weight decay in each group will be applied on the parameters if the
        weight decay is positive. When not separating parameter groups, the `weight_decay` in the API will be applied
        on the parameters without 'beta' or 'gamma' in their names if `weight_decay` is positive.

        When separating parameter groups, if you want to centralize the gradient, set grad_centralization to True,
        but the gradient centralization can only be applied to the parameters of the convolution layer.
        If the parameters of the non convolution layer are set to True, an error will be reported.

        To improve parameter groups performance, the customized order of parameters can be supported.

    Args:
        params (Union[list[Parameter], list[dict]]): When the `params` is a list of `Parameter` which will be updated,
            the element in `params` must be class `Parameter`. When the `params` is a list of `dict`, the "params",
            "lr", "weight_decay" and "order_params" are the keys can be parsed.

            - params: Required. The value must be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in the API will be used.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the API will be used.

            - order_params: Optional. If "order_params" in the keys, the value must be the order of parameters and
              the order will be followed in optimizer. There are no other keys in the `dict` and the parameters which
              in the value of 'order_params' must be in one of group parameters.

            - grad_centralization: Optional. The data type of "grad_centralization" is Bool. If "grad_centralization"
              is in the keys, the set value will be used. If not, the `grad_centralization` is False by default.
              This parameter only works on the convolution layer.

        learning_rate (Union[float, Tensor, Iterable, LearningRateSchedule]): A value or a graph for the learning rate.
            When the learning_rate is an Iterable or a Tensor in a 1D dimension, use dynamic learning rate, then
            the i-th step will take the i-th value as the learning rate. When the learning_rate is LearningRateSchedule,
            use dynamic learning rate, the i-th learning rate will be calculated during the process of training
            according to the formula of LearningRateSchedule. When the learning_rate is a float or a Tensor in a zero
            dimension, use fixed learning rate. Other cases are not supported. The float learning rate must be
            equal to or greater than 0. If the type of `learning_rate` is int, it will be converted to float.
        momentum (float): Hyperparameter of type float, means momentum for the moving average.
            It must be at least 0.0.
        weight_decay (int, float): Weight decay (L2 penalty). It must be equal to or greater than 0.0. Default: 0.0.
        loss_scale (float): A floating point value for the loss scale. It must be greater than 0.0. In general, use the
            default value. Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to False, then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.FixedLossScaleManager` for more details.
            Default: 1.0.
        use_nesterov (bool): Enable Nesterov momentum. Default: False.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[bool], all elements are True.

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
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
    """
    def __init__(self, params, learning_rate, momentum, weight_decay=0.0, loss_scale=1.0, use_nesterov=False):
        super(Momentum, self).__init__(learning_rate, params, weight_decay, loss_scale)
        Validator.check_value_type("momentum", momentum, [float], self.cls_name)
        if isinstance(momentum, float) and momentum < 0.0:
            raise ValueError("momentum should be at least 0.0, but got momentum {}".format(momentum))
        self.momentum = Parameter(Tensor(momentum, mstype.float32), name="momentum")
        self.params = self.parameters
        self.use_nesterov = Validator.check_bool(use_nesterov)
        self.moments = self.params.clone(prefix="moments", init='zeros')
        self.hyper_map = C.HyperMap()
        self.opt = _selected_ops.ApplyMomentum(use_nesterov=self.use_nesterov)

    def construct(self, gradients):
        params = self.params
        moments = self.moments
        gradients = self.decay_weight(gradients)
        gradients = self.scale_grad(gradients)
        gradients = self.gradients_centralization(gradients)
        lr = self.get_lr()
        if self.is_group_lr:
            success = self.hyper_map(F.partial(_momentum_opt, self.opt, self.momentum), lr, gradients, params, moments,
                                     self.ps_parameters, self.cache_enable)
        else:
            success = self.hyper_map(F.partial(_momentum_opt, self.opt, self.momentum, lr), gradients, params, moments,
                                     self.ps_parameters, self.cache_enable)
        return success
