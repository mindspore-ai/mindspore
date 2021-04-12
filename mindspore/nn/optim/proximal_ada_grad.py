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
"""PROXIMAL_ADA_GRAD"""
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common import Tensor
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator as validator
from .optimizer import Optimizer

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
    """
    Implements the ProximalAdagrad algorithm with ApplyProximalAdagrad Operator.

    ProximalAdagrad is an online Learning and Stochastic Optimization.
    Refer to paper `Efficient Learning using Forward-Backward Splitting
    <http://papers.nips.cc//paper/3793-efficient-learning-using-forward-backward-splitting.pdf>`_.

    Note:
        When separating parameter groups, the weight decay in each group will be applied on the parameters if the
        weight decay is positive. When not separating parameter groups, the `weight_decay` in the API will be applied
        on the parameters without 'beta' or 'gamma' in their names if `weight_decay` is positive.

        When separating parameter groups, if you want to centralize the gradient, set grad_centralization to True,
        but the gradient centralization can only be applied to the parameters of the convolution layer.
        If the parameters of the non convolution layer are set to True, an error will be reported.

        To improve parameter groups performance, the customized order of parameters can be supported.

        The sparse strategy is applied while the SparseGatherV2 operator being used for forward network.
        The sparse feature is under continuous development. If the sparse strategy wants to be executed on the host,
        set the target to the CPU.

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

        accum (float): The starting value for accumulators, must be zero or positive values. Default: 0.1.
        learning_rate (Union[float, Tensor, Iterable, LearningRateSchedule]): A value or a graph for the learning rate.
            When the learning_rate is an Iterable or a Tensor in a 1D dimension, use dynamic learning rate, then
            the i-th step will take the i-th value as the learning rate. When the learning_rate is LearningRateSchedule,
            use dynamic learning rate, the i-th learning rate will be calculated during the process of training
            according to the formula of LearningRateSchedule. When the learning_rate is a float or a Tensor in a zero
            dimension, use fixed learning rate. Other cases are not supported. The float learning rate must be
            equal to or greater than 0. If the type of `learning_rate` is int, it will be converted to float.
            Default: 0.001.
        l1 (float): l1 regularization strength, must be greater than or equal to zero. Default: 0.0.
        l2 (float): l2 regularization strength, must be greater than or equal to zero. Default: 0.0.
        use_locking (bool): If true, use locks for updating operation. Default: False.
        loss_scale (float): Value for the loss scale. It must be greater than 0.0. In general, use the default value.
            Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to False, then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.FixedLossScaleManager` for more details.
            Default: 1.0.
        weight_decay (Union[float, int]): Weight decay value to multiply weight, must be zero or positive value.
            Default: 0.0.

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
        ``Ascend``

    Examples:
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
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
    """

    def __init__(self, params, accum=0.1, learning_rate=0.001, l1=0.0, l2=0.0,
                 use_locking=False, loss_scale=1.0, weight_decay=0.0):
        super(ProximalAdagrad, self).__init__(learning_rate, params, weight_decay, loss_scale)
        _check_param_value(accum, l1, l2, use_locking, self.cls_name)
        self.accum = self.parameters.clone(prefix="accum", init=accum)
        self.l1 = Tensor(l1, mstype.float32)
        self.l2 = Tensor(l2, mstype.float32)
        self.hyper_map = C.HyperMap()
        self.use_locking = use_locking
        self.opt = P.ApplyProximalAdagrad(use_locking=use_locking)
        self.sparse_opt = P.SparseApplyProximalAdagrad(use_locking=use_locking)

    def construct(self, grads):
        params = self.parameters
        accum = self.accum
        grads = self.decay_weight(grads)
        grads = self.scale_grad(grads)
        grads = self._grad_sparse_indices_deduplicate(grads)
        grads = self.gradients_centralization(grads)
        lr = self.get_lr()
        if self.is_group_lr:
            success = self.map_(F.partial(_proximal_ada_grad_opt, self.opt, self.sparse_opt, self.l1, self.l2), lr,
                                grads, params, accum)
        else:
            success = self.map_(F.partial(_proximal_ada_grad_opt, self.opt, self.sparse_opt, self.l1, self.l2, lr),
                                grads, params, accum)
        return success

    @Optimizer.target.setter
    def target(self, value):
        """If the input value is set to "CPU", the parameters will be updated on the host using the Fused
           optimizer operation."""
        if not isinstance(value, str):
            raise TypeError("The value must be str type, but got value type is {}".format(type(value)))

        if value not in ('CPU', 'Ascend', 'GPU'):
            raise ValueError("The value must be 'CPU', 'Ascend' or 'GPU', but got value {}".format(value))

        if value == 'CPU':
            self.sparse_opt = P.FusedSparseProximalAdagrad(self.use_locking).add_prim_attr("primitive_target", "CPU")
        else:
            self.sparse_opt = P.SparseApplyProximalAdagrad(self.use_locking)

        self._target = value
