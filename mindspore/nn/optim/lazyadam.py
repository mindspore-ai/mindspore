# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""lazy adam"""
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from .optimizer import Optimizer

_lazy_adam_opt = C.MultitypeFuncGraph("lazy_adam_opt")


@_lazy_adam_opt.register("Function", "Function", "Function", "Function", "Bool", "Bool", "Bool", "Tensor", "Tensor",
                         "Tensor", "Tensor", "Tensor", "Tensor", "RowTensor", "Tensor", "Tensor", "Tensor", "Bool",
                         "Bool")
def _run_opt_with_sparse(opt, sparse_opt, push, pull, use_locking, use_nesterov, target, beta1_power, beta2_power,
                         beta1, beta2, eps, lr, gradient, params, m, v, ps_parameter, cache_enable):
    """Apply sparse lazy adam optimizer to the weight parameter when the gradient is sparse."""
    success = True
    indices = gradient.indices
    values = gradient.values
    if ps_parameter and not cache_enable:
        op_shape = P.Shape()
        shapes = (op_shape(params), op_shape(m), op_shape(v),
                  op_shape(beta1_power), op_shape(beta2_power), op_shape(lr), op_shape(beta1),
                  op_shape(beta2), op_shape(eps), op_shape(values), op_shape(indices))
        success = F.depend(success, pull(push((beta1_power, beta2_power, lr, beta1, beta2,
                                               eps, values, indices), shapes), params))
        return success

    if not target:
        success = F.depend(success, sparse_opt(params, m, v, beta1_power, beta2_power, lr, beta1, beta2,
                                               eps, values, indices))
    else:
        op_gather = P.Gather()
        op_sqrt = P.Sqrt()
        scatter_add = P.ScatterAdd(use_locking)
        scatter_update = P.ScatterUpdate(use_locking)

        m_slice = op_gather(m, indices, 0)
        v_slice = op_gather(v, indices, 0)

        next_m = m_slice * beta1 + values * (1 - beta1)
        next_v = v_slice * beta2 + values * values * (1 - beta2)

        lr_t = lr * op_sqrt(1 - beta2_power) / (1 - beta1_power)

        if use_nesterov:
            m_temp = beta1 * next_m + values * (1 - beta1)
            param_update = m_temp / (op_sqrt(next_v) + eps)
        else:
            param_update = next_m / (op_sqrt(next_v) + eps)

        success = F.depend(success, scatter_add(params, indices, - lr_t * param_update))
        success = F.depend(success, scatter_update(m, indices, next_m))
        success = F.depend(success, scatter_update(v, indices, next_v))

    return success


@_lazy_adam_opt.register("Function", "Function", "Function", "Function", "Bool", "Bool", "Bool", "Tensor", "Tensor",
                         "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool")
def _run_opt_with_one_number(opt, sparse_opt, push, pull, use_locking, use_nesterov, target, beta1_power, beta2_power,
                             beta1, beta2, eps, lr, gradient, params, moment1, moment2, ps_parameter, cache_enable):
    """Apply lazy adam optimizer to the weight parameter using Tensor."""
    success = True
    if ps_parameter and not cache_enable:
        op_shape = P.Shape()
        success = F.depend(success, pull(push((beta1_power, beta2_power, lr, beta1, beta2, eps, gradient),
                                              (op_shape(params), op_shape(moment1), op_shape(moment2))), params))
    else:
        success = F.depend(success, opt(params, moment1, moment2, beta1_power, beta2_power, lr, beta1, beta2,
                                        eps, gradient))
    return success


def _check_param_value(beta1, beta2, eps, weight_decay, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_value_type("weight_dacay", weight_decay, [float], prim_name)
    validator.check_float_range(beta1, 0.0, 1.0, Rel.INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(beta2, 0.0, 1.0, Rel.INC_NEITHER, "beta2", prim_name)
    validator.check_positive_float(eps, "eps", prim_name)
    validator.check_non_negative_float(weight_decay, "weight_decay", prim_name)


class LazyAdam(Optimizer):
    r"""
    This optimizer will apply a lazy adam algorithm when gradient is sparse.

    The original adam algorithm is proposed in `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_.

    The updating formulas are as follows,

    .. math::
        \begin{array}{ll} \\
            m = \beta_1 * m + (1 - \beta_1) * g \\
            v = \beta_2 * v + (1 - \beta_2) * g * g \\
            l = \alpha * \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \\
            w = w - l * \frac{m}{\sqrt{v} + \epsilon}
        \end{array}

    :math:`m` represents the 1st moment vector `moment1`, :math:`v` represents the 2nd moment vector `moment2`,
    :math:`g` represents `gradients`, :math:`l` represents scaling factor `lr`, :math:`\beta_1, \beta_2` represent
    `beta1` and `beta2`, :math:`t` represents updating step while :math:`beta_1^t` and :math:`beta_2^t` represent
    `beta1_power` and `beta2_power`, :math:`\alpha` represents `learning_rate`, :math:`w` represents `params`,
    :math:`\epsilon` represents `eps`.

    Note:
        When separating parameter groups, the weight decay in each group will be applied on the parameters if the
        weight decay is positive. When not separating parameter groups, the `weight_decay` in the API will be applied
        on the parameters without 'beta' or 'gamma' in their names if `weight_decay` is positive.

        When separating parameter groups, if you want to centralize the gradient, set grad_centralization to True,
        but the gradient centralization can only be applied to the parameters of the convolution layer.
        If the parameters of the non convolution layer are set to True, an error will be reported.

        To improve parameter groups performance, the customized order of parameters can be supported.

        The sparse strategy is applied while the SparseGatherV2 operator being used for forward network.
        The sparse behavior, to be notice, is not equivalent to the
        original Adam algorithm, as only the current indices parames will be updated. The sparse feature is under
        continuous development. If the sparse strategy wants to be executed on the host, set the target to the CPU.

    Args:
        params (Union[list[Parameter], list[dict]]): When the `params` is a list of `Parameter` which will be updated,
            the element in `params` must be class `Parameter`. When the `params` is a list of `dict`, the "params",
            "lr" and "weight_decay" are the keys can be parsed.

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
            Default: 1e-3.
        beta1 (float): The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).
                       Default: 0.9.
        beta2 (float): The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).
                       Default: 0.999.
        eps (float): Term added to the denominator to improve numerical stability. Should be greater than 0. Default:
                     1e-8.
        use_locking (bool): Whether to enable a lock to protect variable tensors from being updated.
            If true, updates of the var, m, and v tensors will be protected by a lock.
            If false, the result is unpredictable. Default: False.
        use_nesterov (bool): Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients.
            If true, update the gradients using NAG.
            If false, update the gradients without using NAG. Default: False.
        weight_decay (Union[float, int]): Weight decay (L2 penalty). Default: 0.0.
        loss_scale (float): A floating point value for the loss scale. Should be equal to or greater than 1. In general,
            use the default value. Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update`
            in `FixedLossScaleManager` is set to False, then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.FixedLossScaleManager` for more details.
            Default: 1.0.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        Tensor[bool], the value is True.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `beta1`, `beta2`, `eps` or `loss_scale` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        TypeError: If `use_locking` or `use_nesterov` is not a bool.
        ValueError: If `loss_scale` or `eps` is less than or equal to 0.
        ValueError: If `beta1`, `beta2` is not in range (0.0, 1.0).
        ValueError: If `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.LazyAdam(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = nn.LazyAdam(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01 and grad
        >>> # centralization of True.
        >>> # The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0 and grad
        >>> # centralization of False.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
    """

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, use_locking=False,
                 use_nesterov=False, weight_decay=0.0, loss_scale=1.0):
        super(LazyAdam, self).__init__(learning_rate, params, weight_decay, loss_scale)
        _check_param_value(beta1, beta2, eps, weight_decay, self.cls_name)
        validator.check_value_type("use_locking", use_locking, [bool], self.cls_name)
        validator.check_value_type("use_nesterov", use_nesterov, [bool], self.cls_name)

        self.beta1 = Tensor(beta1, mstype.float32)
        self.beta2 = Tensor(beta2, mstype.float32)
        self.beta1_power = Parameter(initializer(1, [1], mstype.float32), name="beta1_power")
        self.beta2_power = Parameter(initializer(1, [1], mstype.float32), name="beta2_power")
        self.eps = Tensor(eps, mstype.float32)
        self.use_nesterov = use_nesterov
        self.use_locking = use_locking
        self._is_device = True
        self.moment1 = self.parameters.clone(prefix="moment1", init='zeros')
        self.moment2 = self.parameters.clone(prefix="moment2", init='zeros')

        self.hyper_map = C.HyperMap()
        self.opt = P.Adam(use_locking, use_nesterov)
        self.sparse_opt = P.FusedSparseLazyAdam(use_locking, use_nesterov)
        self.sparse_opt.add_prim_attr("primitive_target", "CPU")
        self._ps_pull = P.Pull()
        self._ps_push = P.Push("Adam", [0, 1, 2])
        self._ps_push.add_prim_attr("use_nesterov", use_nesterov)

    def construct(self, gradients):
        gradients = self.decay_weight(gradients)
        gradients = self.scale_grad(gradients)
        gradients = self._grad_sparse_indices_deduplicate(gradients)
        gradients = self.gradients_centralization(gradients)
        lr = self.get_lr()

        self.beta1_power = self.beta1_power * self.beta1
        self.beta2_power = self.beta2_power * self.beta2

        if self.is_group_lr:
            success = self.map_(F.partial(_lazy_adam_opt, self.opt, self.sparse_opt, self._ps_push, self._ps_pull,
                                          self.use_locking, self.use_nesterov, self._is_device,
                                          self.beta1_power, self.beta2_power, self.beta1, self.beta2, self.eps),
                                lr, gradients, self.parameters, self.moment1, self.moment2, self.ps_parameters,
                                self.cache_enable)
        else:
            success = self.map_(F.partial(_lazy_adam_opt, self.opt, self.sparse_opt, self._ps_push, self._ps_pull,
                                          self.use_locking, self.use_nesterov, self._is_device,
                                          self.beta1_power, self.beta2_power, self.beta1, self.beta2, self.eps, lr),
                                gradients, self.parameters, self.moment1, self.moment2, self.ps_parameters,
                                self.cache_enable)
        return success

    @Optimizer.target.setter
    def target(self, value):
        """If the input value is set to "CPU", the parameters will be updated on the host using the Fused
           optimizer operation."""
        if not isinstance(value, str):
            raise TypeError("The value must be str type, but got value type is {}".format(type(value)))

        if value not in ('CPU', 'Ascend', 'GPU'):
            raise ValueError("The value must be 'CPU', 'Ascend' or 'GPU', but got value {}".format(value))

        if self._target == "CPU" and value in('Ascend', 'GPU'):
            raise ValueError("In the CPU environment, target cannot be set to 'GPU' and 'Ascend'.")

        if self._target == "Ascend" and value == 'GPU':
            raise ValueError("In the Ascend environment, target cannot be set to 'GPU'.")

        self._is_device = (value != 'CPU')
        self._target = value
