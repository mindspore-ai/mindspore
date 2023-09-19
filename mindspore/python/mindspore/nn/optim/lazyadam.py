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
"""lazy adam"""
from __future__ import absolute_import

from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore import _checkparam as validator
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register
from mindspore.nn.optim._dist_optimizer_registry import _register_dist_optimizer
from mindspore.common._decorator import deprecated

_lazy_adam_opt = C.MultitypeFuncGraph("lazy_adam_opt")


@_lazy_adam_opt.register("Function", "Function", "Function", "Function", "Bool", "Bool", "Bool", "Tensor", "Tensor",
                         "Tensor", "Tensor", "Tensor", "Tensor", "RowTensor", "Tensor", "Tensor", "Tensor", "Bool",
                         "Bool", "Function", "Bool", "Function", "Bool")
def _run_opt_with_sparse_dist(opt, sparse_opt, push, pull, use_locking, use_nesterov, target, beta1_power, beta2_power,
                              beta1, beta2, eps, lr, gradient, params, m, v, ps_parameter, cache_enable,
                              distributed_opt, use_flag, distributed_sparse_opt, use_sparse_flag):
    """Apply sparse lazy adam optimizer to the weight parameter when the gradient is sparse."""
    success = True
    indices = gradient.indices
    values = gradient.values
    if use_sparse_flag:
        success = F.depend(success, distributed_sparse_opt(params, m, v, beta1_power, beta2_power, lr, beta1, beta2,
                                                           eps, values, indices))
        return success
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
                         "Tensor", "Tensor", "Tensor", "Tensor", "MapTensor", "MapTensor", "MapTensor", "MapTensor",
                         "Bool", "Bool", "Function", "Bool", "Function", "Bool")
def _run_map_tensor_opt_with_sparse_dist(opt, sparse_opt, push, pull, use_locking, use_nesterov, target, beta1_power,
                                         beta2_power, beta1, beta2, eps, lr, gradient, params, m, v,
                                         ps_parameter, cache_enable, distributed_opt, use_flag, distributed_sparse_opt,
                                         use_sparse_flag):
    """Apply sparse lazy adam optimizer to the weight parameter when the gradient is sparse."""
    success = True
    indices, values = gradient.get_data()
    if use_sparse_flag:
        # PS Mode.
        success = F.depend(success, distributed_sparse_opt(params, m, v, beta1_power, beta2_power, lr, beta1, beta2,
                                                           eps, values, indices))
    else:
        # PS Cache mode.
        op_sqrt = P.Sqrt()

        m_slice = m.get(indices)
        v_slice = v.get(indices)

        next_m = m_slice * beta1 + values * (1 - beta1)
        next_v = v_slice * beta2 + values * values * (1 - beta2)

        lr_t = lr * op_sqrt(1 - beta2_power) / (1 - beta1_power)

        if use_nesterov:
            m_temp = beta1 * next_m + values * (1 - beta1)
            param_update = m_temp / (op_sqrt(next_v) + eps)
        else:
            param_update = next_m / (op_sqrt(next_v) + eps)

        params_need_update = params.get(indices)
        params.put(indices, params_need_update - lr_t * param_update)
        m.put(indices, next_m)
        v.put(indices, next_v)

    return success


@_lazy_adam_opt.register("Function", "Function", "Function", "Function", "Bool", "Bool", "Bool", "Tensor", "Tensor",
                         "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool",
                         "Function", "Bool", "Function", "Bool")
def _run_opt_with_one_number_dist(opt, sparse_opt, push, pull, use_locking, use_nesterov, target,
                                  beta1_power, beta2_power,
                                  beta1, beta2, eps, lr, gradient, params, moment1, moment2, ps_parameter, cache_enable,
                                  distributed_opt, use_flag, distributed_sparse_opt, use_sparse_flag):
    """Apply lazy adam optimizer to the weight parameter using Tensor."""
    success = True
    if use_flag:
        success = F.depend(success, distributed_opt(params, moment1, moment2, beta1_power, beta2_power, lr, beta1,
                                                    beta2, eps, gradient))
    elif ps_parameter and not cache_enable:
        op_shape = P.Shape()
        success = F.depend(success, pull(push((beta1_power, beta2_power, lr, beta1, beta2, eps, gradient),
                                              (op_shape(params), op_shape(moment1), op_shape(moment2))), params))
    else:
        success = F.depend(success, opt(params, moment1, moment2, beta1_power, beta2_power, lr, beta1, beta2,
                                        eps, gradient))
    return success


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
                         "Tensor", "Tensor", "Tensor", "Tensor", "MapTensor", "MapTensor", "MapTensor", "MapTensor",
                         "Bool", "Bool")
def _run_map_tensor_opt_with_sparse(opt, sparse_opt, push, pull, use_locking, use_nesterov, target, beta1_power,
                                    beta2_power, beta1, beta2, eps, lr, gradient, params, m, v, ps_parameter,
                                    cache_enable):
    """Apply sparse lazy adam optimizer to the weight parameter when the gradient is sparse(MapTensor)."""
    success = True
    indices, values = gradient.get_data()

    op_sqrt = P.Sqrt()

    m_slice = m.get(indices)
    v_slice = v.get(indices)

    next_m = m_slice * beta1 + values * (1 - beta1)
    next_v = v_slice * beta2 + values * values * (1 - beta2)

    lr_t = lr * op_sqrt(1 - beta2_power) / (1 - beta1_power)

    if use_nesterov:
        m_temp = beta1 * next_m + values * (1 - beta1)
        param_update = m_temp / (op_sqrt(next_v) + eps)
    else:
        param_update = next_m / (op_sqrt(next_v) + eps)

    params_need_update = params.get(indices)
    params.put(indices, params_need_update - lr_t * param_update)
    m.put(indices, next_m)
    v.put(indices, next_v)

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
    validator.check_float_range(beta1, 0.0, 1.0, validator.INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(beta2, 0.0, 1.0, validator.INC_NEITHER, "beta2", prim_name)
    validator.check_positive_float(eps, "eps", prim_name)
    validator.check_non_negative_float(weight_decay, "weight_decay", prim_name)


class LazyAdam(Optimizer):
    r"""
    Implements the Adaptive Moment Estimation (Adam) algorithm. The Adam algorithm is proposed
    in `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_.

    This optimizer will apply a lazy adam algorithm when gradient is sparse.

    The updating formulas are as follows,

    .. math::
        \begin{array}{ll} \\
            m_{t+1} = \beta_1 * m_{t} + (1 - \beta_1) * g \\
            v_{t+1} = \beta_2 * v_{t} + (1 - \beta_2) * g * g \\
            \widehat{m_{t+1}} = \frac{m_{t+1}}{1-\beta_1^t} \\
            \widehat{v_{t+1}} = \frac{v_{t+1}}{1-\beta_2^t} \\
            w_{t+1} = w_{t} - \gamma * \frac{\widehat{m_{t+1}}}{\sqrt{\widehat{v_{t+1}}} + \epsilon}
        \end{array}

    :math:`m` represents the 1st moment vector `moment1`, :math:`v` represents the 2nd moment vector `moment2`,
    :math:`g` represents `gradients`, :math:`\gamma` represents `learning_rate`, :math:`\beta_1, \beta_2` represent
    `beta1` and `beta2`, :math:`t` represents the current step while :math:`beta_1^t` and :math:`beta_2^t` represent
    `beta1_power` and `beta2_power`, :math:`w` represents `params`,
    :math:`\epsilon` represents `eps`.

    Note:
        The sparse strategy is applied while the SparseGatherV2 operator is used for forward network. If the sparse
        strategy wants to be executed on the host, set the target to the CPU.
        Please note, the optimizer only updates the current index position of the network parameters
        when the gradient is sparse.
        The sparse behavior is not equivalent to the original Adam algorithm.
        If you want to execute a sparse policy, target needs to be set to CPU.

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

        learning_rate (Union[float, int, Tensor, Iterable, :class:`~.train.LearningRateScheduler`]): Default: ``1e-3`` .

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        beta1 (float): The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).
                       Default: ``0.9`` .
        beta2 (float): The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).
                       Default: ``0.999`` .
        eps (float): Term added to the denominator to improve numerical stability. Should be greater than 0.
                     Default: ``1e-8`` .
        use_locking (bool): Whether to enable a lock to protect the updating process of variable tensors.
            If ``true`` , updates of the `w`, `m`, and `v` tensors will be protected by a lock.
            If ``false`` , the result is unpredictable. Default: ``False`` .
        use_nesterov (bool): Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients.
            If ``true`` , update the gradients using NAG.
            If ``false`` , update the gradients without using NAG. Default: ``False`` .

        weight_decay (Union[float, int, Cell]): Weight decay (L2 penalty). Default: ``0.0`` .

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

        loss_scale (float): A floating point value for the loss scale. Should be equal to or greater than 1. In general,
            use the default value. Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update`
            in `FixedLossScaleManager` is set to ``False`` , then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.amp.FixedLossScaleManager` for more details.
            Default: ``1.0`` .

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        Tensor[bool], the value is ``True`` .

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable,
            :class:`~.train.LearningRateScheduler`.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `beta1`, `beta2`, `eps` or `loss_scale` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        TypeError: If `use_locking` or `use_nesterov` is not a bool.
        ValueError: If `loss_scale` or `eps` is less than or equal to 0.
        ValueError: If `beta1`, `beta2` is not in range (0.0, 1.0).
        ValueError: If `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
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
        >>> model = ms.train.Model(net, loss_fn=loss, optimizer=optim)
    """

    @deprecated("2.0", "Adam", False)
    @opt_init_args_register
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
        self.moment1 = self._parameters.clone(prefix="moment1", init='zeros')
        self.moment2 = self._parameters.clone(prefix="moment2", init='zeros')
        self.opt = P.Adam(use_locking, use_nesterov)
        self.sparse_opt = P.FusedSparseLazyAdam(use_locking, use_nesterov)
        self.sparse_opt.set_device("CPU")
        self._ps_pull = P.Pull()
        self._ps_push = P.Push("Adam", [0, 1, 2])
        self._ps_push.add_prim_attr("use_nesterov", use_nesterov)

        self._init_distributed_opts(use_locking, use_nesterov)

    @jit
    def construct(self, gradients):
        gradients = self.flatten_gradients(gradients)
        gradients = self.decay_weight(gradients)
        gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)
        gradients = self._grad_sparse_indices_deduplicate(gradients)
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        beta1_power = self.beta1_power * self.beta1
        self.beta1_power = beta1_power
        beta2_power = self.beta2_power * self.beta2
        self.beta2_power = beta2_power

        if self.use_dist_optimizer:
            if self.is_group_lr:
                success = self.map_reverse(F.partial(_lazy_adam_opt, self.opt, self.sparse_opt,
                                                     self._ps_push, self._ps_pull, self.use_locking, self.use_nesterov,
                                                     self._is_device, beta1_power, beta2_power,
                                                     self.beta1, self.beta2, self.eps),
                                           lr, gradients, self._parameters, self.moment1, self.moment2,
                                           self.ps_parameters, self.cache_enable, self.dense_lazyadam_opts,
                                           self.use_dense_opt_flags, self.sparse_lazyadam_opts,
                                           self.use_sparse_opt_flags)
            else:
                success = self.map_reverse(F.partial(_lazy_adam_opt, self.opt, self.sparse_opt,
                                                     self._ps_push, self._ps_pull, self.use_locking, self.use_nesterov,
                                                     self._is_device, beta1_power, beta2_power,
                                                     self.beta1, self.beta2, self.eps, lr),
                                           gradients, self._parameters, self.moment1, self.moment2,
                                           self.ps_parameters, self.cache_enable, self.dense_lazyadam_opts,
                                           self.use_dense_opt_flags, self.sparse_lazyadam_opts,
                                           self.use_sparse_opt_flags)
        else:
            if self.is_group_lr:
                success = self.map_reverse(F.partial(_lazy_adam_opt, self.opt, self.sparse_opt,
                                                     self._ps_push, self._ps_pull, self.use_locking, self.use_nesterov,
                                                     self._is_device, beta1_power, beta2_power,
                                                     self.beta1, self.beta2, self.eps),
                                           lr, gradients, self._parameters, self.moment1, self.moment2,
                                           self.ps_parameters, self.cache_enable)
            else:
                success = self.map_reverse(F.partial(_lazy_adam_opt, self.opt, self.sparse_opt,
                                                     self._ps_push, self._ps_pull, self.use_locking, self.use_nesterov,
                                                     self._is_device, beta1_power, beta2_power,
                                                     self.beta1, self.beta2, self.eps, lr),
                                           gradients, self._parameters, self.moment1, self.moment2,
                                           self.ps_parameters, self.cache_enable)
        return success

    @Optimizer.target.setter
    def target(self, value):
        """
        If the input value is set to "CPU", the parameters will be updated on the host using the Fused
        optimizer operation.
        """
        self._set_base_target(value)

    def _init_distributed_opts(self, use_locking, use_nesterov):
        self.use_dist_optimizer = self._use_distibuted_optimizer()
        self.dense_lazyadam_opts, self.use_dense_opt_flags =\
        self._get_distributed_optimizer_list("adam", use_locking, use_nesterov)
        self.sparse_lazyadam_opts, self.use_sparse_opt_flags =\
        self._get_distributed_optimizer_list("fused_sparse_lazy_adam", use_locking, use_nesterov)


def create_distributed_adam(*args, **kwargs):
    """
    Create the distributed Adam op.
    """
    adam = P.Adam(*args, **kwargs)
    adam.add_prim_attr("gradient_type", "dense_gradient")
    adam.add_prim_attr("parameter_input_index", 0)
    adam.add_prim_attr("gradient_input_index", 9)
    return adam


def create_distributed_fused_sparse_lazy_adam(*args, **kwargs):
    """
    Create the distributed FusedSparseLazyAdam op.
    """
    sparse_lazy_adam = P.FusedSparseLazyAdam(*args, **kwargs)
    sparse_lazy_adam.add_prim_attr("gradient_type", "sparse_gradient")
    sparse_lazy_adam.add_prim_attr("parameter_input_index", 0)
    sparse_lazy_adam.add_prim_attr("gradient_input_index", 9)
    sparse_lazy_adam.add_prim_attr("indices_input_index", 10)
    return sparse_lazy_adam

_register_dist_optimizer("adam", create_distributed_adam)
_register_dist_optimizer("fused_sparse_lazy_adam", create_distributed_fused_sparse_lazy_adam)
