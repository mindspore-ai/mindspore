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
"""optimizer"""
from typing import Iterable

import numpy as np

import mindspore
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.ops.operations import _inner_ops as inner
from mindspore.nn.cell import Cell
from mindspore.nn.layer.container import CellList
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor, RowTensor
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator as validator
from mindspore import log as logger
from mindspore.parallel._utils import _get_global_rank, _get_device_num, _get_parallel_mode
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.nn.learning_rate_schedule import LearningRateSchedule

__all__ = ['Optimizer']


class Optimizer(Cell):
    """
    Base class for all optimizers.

    Note:
        This class defines the API to add Ops to train a model. Never use
        this class directly, but instead instantiate one of its subclasses.

        Different parameter groups can set different `learning_rate`, `weight_decay` and `grad_centralization`.

        When separating parameter groups, the weight decay in each group will be applied on the parameters if the
        weight_decay is positive. For most optimizer, when not separating parameters, the `weight_decay` in the API will
        be applied on the parameters without 'beta' or 'gamma' in their names if `weight_decay` is positive.

        When separating parameter groups, if you want to centralize the gradient, set grad_centralization to True,
        but the gradient centralization can only be applied to the parameters of the convolution layer.
        If the parameters of the non convolution layer are set to True, an error will be reported.

        To improve parameter groups performance, the customized order of parameters can be supported.

    Args:
        learning_rate (Union[float, Tensor, Iterable, LearningRateSchedule]): A value or a graph for the learning
            rate. When the learning_rate is an Iterable or a Tensor in a 1D dimension, use dynamic learning rate, then
            the i-th step will take the i-th value as the learning rate. When the learning_rate is LearningRateSchedule,
            use dynamic learning rate, the i-th learning rate will be calculated during the process of training
            according to the formula of LearningRateSchedule. When the learning_rate is a float or a Tensor in a zero
            dimension, use fixed learning rate. Other cases are not supported. The float learning rate must be
            equal to or greater than 0. If the type of `learning_rate` is int, it will be converted to float.
        parameters (Union[list[Parameter], list[dict]]): When the `parameters` is a list of `Parameter` which will be
            updated, the element in `parameters` must be class `Parameter`. When the `parameters` is a list of `dict`,
            the "params", "lr", "weight_decay" and "order_params" are the keys can be parsed.

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

        weight_decay (Union[float, int]): An int or a floating point value for the weight decay.
            It must be equal to or greater than 0.
            If the type of `weight_decay` input is int, it will be converted to float. Default: 0.0.
        loss_scale (float): A floating point value for the loss scale. It must be greater than 0. If the
            type of `loss_scale` input is int, it will be converted to float. In general, use the default value. Only
            when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to False, then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.FixedLossScaleManager` for more details.
            Default: 1.0.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `loss_scale` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        ValueError: If `loss_scale` is less than or equal to 0.
        ValueError: If `weight_decay` is less than 0.
        ValueError: If `learning_rate` is a Tensor, but the dimension of tensor is greater than 1.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, learning_rate, parameters, weight_decay=0.0, loss_scale=1.0):
        super(Optimizer, self).__init__(auto_prefix=False)
        if parameters is not None and not isinstance(parameters, list):
            parameters = list(parameters)

        if not parameters:
            raise ValueError("Optimizer got an empty parameter list.")

        if not isinstance(parameters[0], (dict, Parameter)):
            raise TypeError("Only a list of Parameter or dict can be supported.")

        if isinstance(loss_scale, int):
            loss_scale = float(loss_scale)
        validator.check_value_type("loss_scale", loss_scale, [float], self.cls_name)
        validator.check_positive_float(loss_scale, "loss_scale", self.cls_name)
        self.loss_scale = loss_scale

        weight_decay = self._preprocess_weight_decay(weight_decay)
        self.grad_centralization = False

        self._unique = True
        self._target = context.get_context("device_target")
        self.dynamic_lr = False
        self.assignadd = None
        self.global_step = None
        self.is_group = False
        self.is_group_lr = False
        self.is_group_params_ordered = False
        learning_rate = self._preprocess_single_lr(learning_rate)
        if isinstance(parameters[0], dict):
            self.is_group = True
            self.group_params = []
            self.group_lr = []
            self.group_weight_decay = []
            self.group_grad_centralization = []
            self._init_group_params(parameters, learning_rate, weight_decay, self.grad_centralization)

        # The final value of dynamic_lr can be determined after the process of parse_single_lr and init_group_params
        if self.dynamic_lr:
            self.assignadd = P.AssignAdd()
            self.global_step = Parameter(initializer(0, [1], mindspore.int32), name='global_step')

        if self.is_group_lr:
            self.learning_rate = CellList(self.group_lr, auto_prefix=False) if self.dynamic_lr \
                else ParameterTuple(self.group_lr)
        else:
            self.learning_rate = self._build_single_lr(learning_rate, 'learning_rate')

        if self.is_group:
            self.parameters = ParameterTuple(self.group_params)
            self.weight_decay = tuple(self.group_weight_decay)
            self.weight_decay_tensor_tuple = tuple(Tensor(x, mstype.float32) for x in self.group_weight_decay)
            decay_filter = lambda x: x > 0
            self.decay_flags = tuple(decay_filter(x) for x in self.weight_decay)
            self.exec_weight_decay = any(self.decay_flags)
            self.grad_centralization_flags = tuple(self.group_grad_centralization)
        else:
            self.parameters = ParameterTuple(parameters)
            self.weight_decay = weight_decay * loss_scale
            self.weight_decay_tensor = Tensor(self.weight_decay, mstype.float32)
            decay_filter = lambda x: 'beta' not in x.name and 'gamma' not in x.name
            self.decay_flags = tuple(decay_filter(x) for x in self.parameters)
            self.exec_weight_decay = self.weight_decay > 0
        # when a parameter has been unique, there is no need do another unique in optimizer.
        for param in self.parameters:
            if param.unique:
                self._unique = False
                break
        ps_filter = lambda x: x.is_param_ps
        self.ps_parameters = tuple(ps_filter(x) for x in self.parameters)
        cache_filter = lambda x: x.cache_enable
        self.cache_enable = tuple(cache_filter(x) for x in self.parameters)
        self.reciprocal_scale = Tensor(1.0 / loss_scale, mstype.float32)
        self.need_scale = loss_scale != 1.0
        self.global_step_increase_tensor = Tensor(1, mstype.int32)
        self.param_length = len(self.parameters)
        self.map_ = C.Map()
        self._use_parallel_optimizer()

    def _use_parallel_optimizer(self):
        """Indicates whether to use automatic parallelism."""
        if context.get_auto_parallel_context("enable_parallel_optimizer"):
            if _get_parallel_mode() == ParallelMode.DATA_PARALLEL and context.get_context("device_target") == "Ascend":
                self.use_parallel = True
            elif _get_parallel_mode() == ParallelMode.DATA_PARALLEL \
                    and context.get_context("device_target") != "Ascend":
                raise RuntimeError("Parallel optimizer only supports Ascend in data parallel mode.")
            elif _get_parallel_mode() in (ParallelMode.STAND_ALONE, ParallelMode.HYBRID_PARALLEL):
                raise RuntimeError("Parallel optimizer is not supported in {}.".format(_get_parallel_mode()))
            else:
                self.use_parallel = False
        else:
            self.use_parallel = False
        if self.use_parallel:
            if self.cls_name not in ["Lamb", "AdamWeightDecay"]:
                raise RuntimeError("Parallel optimizer does not support optimizer {}".format(self.cls_name))
            self.dev_num = _get_device_num()
            if self.dev_num > self.param_length:
                raise RuntimeError("Parallel optimizer can not be applied when the number of parameters {} is"
                                   " less than the number of devices {}".format(self.param_length, self.dev_num))
            self.param_rank = self._get_parameter_group_id()
            self.optim_filter = tuple(map(lambda x: x == _get_global_rank(), self.param_rank))
            self.param_names = []
            for param in self.parameters:
                self.param_names.append(param.name)
        else:
            self.optim_filter = (True,) * self.param_length

    @property
    def unique(self):
        """The method is to see whether to make unique. The input type is bool. The method is read-only."""
        return self._unique

    @unique.setter
    def unique(self, value):
        """Set whether the input value is unique."""
        if not isinstance(value, bool):
            raise TypeError("The value type must be bool, but got value type is {}".format(type(value)))
        self._unique = value

    @property
    def target(self):
        """The method is used to determine whether the parameter is updated on host or device. The input type is str
           and can only be 'CPU', 'Ascend' or 'GPU'."""
        return self._target

    @target.setter
    def target(self, value):
        """If the input value is set to "CPU", the parameters will be updated on the host using the Fused
           optimizer operation."""
        raise NotImplementedError

    def decay_weight(self, gradients):
        """
        Weight decay.

        An approach to reduce the overfitting of a deep learning neural network model.

        Args:
            gradients (tuple[Tensor]): The gradients of `self.parameters`, and have the same shape as
                `self.parameters`.

        Returns:
            tuple[Tensor], The gradients after weight decay.
        """
        if self.exec_weight_decay:
            params = self.parameters
            if self.is_group:
                gradients = self.map_(F.partial(_apply_decay), self.weight_decay_tensor_tuple, self.decay_flags,
                                      params, gradients)
            else:
                gradients = self.map_(F.partial(_apply_decay, self.weight_decay_tensor), self.decay_flags,
                                      params, gradients)

        return gradients

    def gradients_centralization(self, gradients):
        """
        Gradients centralization.

        A method for optimizing convolutional layer parameters to impore the training speed of a deep learning neural
        network model.

        Args:
            gradients (tuple[Tensor]): The gradients of `self.parameters`, and have the same shape as
                `self.parameters`.

        Returns:
            tuple[Tensor], The gradients after gradients centralization.
        """
        if self.is_group:
            gradients = self.map_(F.partial(_apply_grad_centralization), self.grad_centralization_flags, gradients)

        return gradients

    def scale_grad(self, gradients):
        """
        Loss scale for mixed precision.

        An approach of mixed precision training to improve the speed and energy efficiency of training deep neural
        network.

        Args:
            gradients (tuple[Tensor]): The gradients of `self.parameters`, and have the same shape as
                `self.parameters`.

        Returns:
            tuple[Tensor], The gradients after loss scale.

        """
        if self.need_scale:
            gradients = self.map_(F.partial(_grad_scale, self.reciprocal_scale), gradients)

        return gradients

    def _grad_sparse_indices_deduplicate(self, gradients):
        """ In the case of using big operators, deduplicate the 'indexes' in gradients."""
        if self._target != 'CPU' and self._unique:
            gradients = self.map_(F.partial(_indices_deduplicate), gradients)
        return gradients

    def _preprocess_weight_decay(self, weight_decay):
        """Check weight decay, and convert int to float."""
        if isinstance(weight_decay, (float, int)):
            weight_decay = float(weight_decay)
            validator.check_non_negative_float(weight_decay, "weight_decay", self.cls_name)
            return weight_decay
        raise TypeError("Weight decay should be int or float.")

    def _preprocess_grad_centralization(self, grad_centralization):
        if not isinstance(grad_centralization, bool):
            raise TypeError("The gradients centralization should be bool")
        return grad_centralization

    def _preprocess_single_lr(self, learning_rate):
        """Check lr value, and convert lr to a float, a Tensor or a LearningRateSchedule."""
        if isinstance(learning_rate, (float, int)):
            learning_rate = float(learning_rate)
            validator.check_non_negative_float(learning_rate, "learning rate", self.cls_name)
            return learning_rate
        if isinstance(learning_rate, Tensor) and learning_rate.ndim == 0:
            return learning_rate

        self.dynamic_lr = True
        if isinstance(learning_rate, Iterable):
            return Tensor(np.array(list(learning_rate)).astype(np.float32))
        if isinstance(learning_rate, Tensor):
            if learning_rate.ndim > 1:
                raise ValueError("The dim of `Tensor` type Learning rate should be a 0 or 1,"
                                 f"but got {learning_rate.ndim}.")
            if learning_rate.ndim == 1 and learning_rate.size < 2:
                logger.warning("If use `Tensor` type dynamic learning rate, please make sure that the number"
                               "of elements in the tensor passed is greater than 1.")
            return learning_rate
        if isinstance(learning_rate, LearningRateSchedule):
            return learning_rate
        raise TypeError("Learning rate should be int, float, Tensor, Iterable or LearningRateSchedule.")

    def _build_single_lr(self, learning_rate, name):
        """Build learning rate value, convert learning rate to a Parameter or a LearningRateSchedule."""
        if isinstance(learning_rate, float):
            learning_rate = Parameter(Tensor(learning_rate, mstype.float32), name)
            if self.is_group_lr and self.dynamic_lr:
                learning_rate = _ConvertToCell(learning_rate)
            return learning_rate
        if isinstance(learning_rate, Tensor) and learning_rate.ndim == 0:
            learning_rate = Parameter(learning_rate, name)
            if self.is_group_lr and self.dynamic_lr:
                learning_rate = _ConvertToCell(learning_rate)
            return learning_rate
        if isinstance(learning_rate, Tensor) and learning_rate.ndim == 1:
            return _IteratorLearningRate(learning_rate, name)
        return learning_rate

    def _check_group_params(self, parameters):
        """Check group params."""
        parse_keys = ['params', 'lr', 'weight_decay', 'order_params', 'grad_centralization']
        for group_param in parameters:
            invalid_key = list(filter(lambda x: x not in parse_keys, group_param.keys()))
            if invalid_key:
                raise KeyError(f'The key "{invalid_key}" cannot be recognized in group params.')

            if 'order_params' in group_param.keys():
                if len(group_param.keys()) > 1:
                    raise ValueError("The order params dict in group parameters should "
                                     "only include the 'order_params' key.")
                if not isinstance(group_param['order_params'], Iterable):
                    raise TypeError("The value of 'order_params' should be an Iterable type.")
                continue

            if not group_param['params']:
                raise ValueError("Optimizer got an empty group parameter list.")

            for param in group_param['params']:
                if not isinstance(param, Parameter):
                    raise TypeError("The group param should be an iterator of Parameter type.")

    def _parse_group_params(self, parameters, learning_rate):
        """Parse group params."""
        self._check_group_params(parameters)
        if isinstance(learning_rate, Tensor) and learning_rate.ndim == 1:
            tensor_lr_length = learning_rate.size
        else:
            tensor_lr_length = 0

        for group_param in parameters:
            if 'order_params' in group_param.keys():
                if len(group_param.keys()) > 1:
                    raise ValueError("The order params dict in group parameters should "
                                     "only include the 'order_params' key.")
                if not isinstance(group_param['order_params'], Iterable):
                    raise TypeError("The value of 'order_params' should be an Iterable type.")
                self.is_group_params_ordered = True
                continue

            if 'lr' in group_param.keys():
                self.is_group_lr = True
                group_lr = self._preprocess_single_lr(group_param['lr'])

                if isinstance(group_lr, Tensor) and group_lr.ndim == 1:
                    group_lr_length = group_lr.size
                    if tensor_lr_length == 0:
                        tensor_lr_length = group_lr_length
                    elif group_lr_length != tensor_lr_length:
                        raise ValueError("The Tensor type dynamic learning rate in group should be the same size.")

    def _init_group_params(self, parameters, learning_rate, weight_decay, grad_centralization):
        """Initialize learning rate, weight decay or grad centralization in group params."""
        self._parse_group_params(parameters, learning_rate)
        default_lr = self._build_single_lr(learning_rate, 'learning_rate')

        params_store = []
        for group_num, group_param in enumerate(parameters):
            if 'order_params' in group_param.keys():
                ordered_parameters = group_param['order_params']
                continue

            self.group_params += group_param['params']

            if 'lr' in group_param.keys():
                lr_param_name = 'learning_rate_group_' + str(group_num)
                lr = self._preprocess_single_lr(group_param['lr'])
                lr = self._build_single_lr(lr, lr_param_name)
            else:
                lr = default_lr

            if 'weight_decay' in group_param.keys():
                cur_weight_decay = self._preprocess_weight_decay(group_param['weight_decay'])
                weight_decay_ = cur_weight_decay * self.loss_scale
            else:
                weight_decay_ = weight_decay * self.loss_scale

            if 'grad_centralization' in group_param.keys():
                self.grad_centralization = self._preprocess_grad_centralization(group_param['grad_centralization'])
                for param in group_param['params']:
                    validator.check_value_type("parameter", param, [Parameter], self.cls_name)
                    if "conv" not in param.name and self.grad_centralization is True:
                        raise ValueError("Grad centralization can be perform only on the conv layer. If the parameter"
                                         "is not a convolution layer, this parameter cannot be set to True.")

                    grad_centralization_ = self.grad_centralization
            else:
                grad_centralization_ = grad_centralization

            for key in group_param.keys():
                if key not in ('params', 'lr', 'weight_decay', 'grad_centralization'):
                    logger.warning(f"The optimizer cannot parse '{key}' when setting parameter groups.")

            for param in group_param['params']:
                validator.check_value_type("parameter", param, [Parameter], self.cls_name)
                if param.name in params_store:
                    raise RuntimeError(f"The {param.name} parameter has appeared in parameter groups.")

                params_store.append(param.name)
                self.group_lr.append(lr)
                self.group_weight_decay.append(weight_decay_)
                self.group_grad_centralization.append(grad_centralization_)

        if self.is_group_params_ordered:
            self._order_and_adjust_group_params(ordered_parameters)

    def _order_and_adjust_group_params(self, ordered_parameters):
        """
        Order group parameter, learning rate, weight decay and grad centralization in group params.
        """
        params_length = len(self.group_params)
        if len(ordered_parameters) != len(self.group_params):
            raise ValueError(f"The value of 'order_params' should be same with all group parameters.")

        ordered_params = [None] * params_length
        ordered_learning_rate = [None] * params_length
        ordered_weight_decay = [None] * params_length
        ordered_grad_centralization = [None] * params_length
        params_name = [param.name for param in ordered_parameters]

        for param, lr, wd, gc in zip(self.group_params, self.group_lr, self.group_weight_decay,
                                     self.group_grad_centralization):
            index = params_name.index(param.name)
            ordered_params[index] = param
            ordered_learning_rate[index] = lr
            ordered_weight_decay[index] = wd
            ordered_grad_centralization[index] = gc

        self.group_params = ordered_params
        self.group_lr = ordered_learning_rate
        self.group_weight_decay = ordered_weight_decay
        self.group_grad_centralization = ordered_grad_centralization

    def get_lr(self):
        """
        Get the learning rate of current step.

        Returns:
            float, the learning rate of current step.
        """
        lr = self.learning_rate
        if self.dynamic_lr:
            if self.is_group_lr:
                lr = ()
                for learning_rate in self.learning_rate:
                    current_dynamic_lr = learning_rate(self.global_step)
                    lr += (current_dynamic_lr,)
            else:
                lr = self.learning_rate(self.global_step)

            self.assignadd(self.global_step, self.global_step_increase_tensor)
        return lr

    def get_lr_parameter(self, param):
        """
        Get the learning rate of parameter.

        Args:
            param (Union[Parameter, list[Parameter]]): The `Parameter` or list of `Parameter`.

        Returns:
            Parameter, single `Parameter` or `list[Parameter]` according to the input type.
        """
        def get_lr_value(learning_rate):
            if isinstance(learning_rate, (_ConvertToCell, _IteratorLearningRate)):
                return learning_rate.learning_rate

            return learning_rate

        if isinstance(param, Parameter):
            param_list = [param]
        elif isinstance(param, list):
            param_list = param
        else:
            raise TypeError(f"The parameter only support 'Parameter' or 'list' type.")

        lr = []
        ids = [id(p) for p in self.parameters]
        for p in param_list:
            validator.check_value_type("parameter", p, [Parameter], self.cls_name)
            if id(p) not in ids:
                raise ValueError(f"The parameter {p.name} is not in optimizer.")
            if self.is_group_lr:
                index = ids.index(id(p))
                lr.append(get_lr_value(self.learning_rate[index]))
            else:
                lr.append(get_lr_value(self.learning_rate))

        return lr if isinstance(param, list) else lr[0]

    def _get_parameter_group_id(self):
        """
        Get the parameter partition group id, which is less than the number of devices.

        Returns:
            tuple, the group id tuple of parameters.
        """
        rank_list = ()
        count = 0
        for _ in range(self.param_length):
            rank_list = rank_list + (count,)
            count = count + 1
            if count == self.dev_num:
                count = 0
        return rank_list

    def broadcast_params(self, optim_result):
        """
        Apply Broadcast operations in the sequential order of parameter groups.

        Returns:
             bool, the status flag.
        """
        param_group = []
        key_group = []
        for _ in range(self.dev_num):
            param_group.append(F.make_tuple())
            key_group.append(F.make_tuple())
        for i in range(self.param_length):
            param_group[self.param_rank[i]] = param_group[self.param_rank[i]] + (self.parameters[i],)
            key = P.MakeRefKey(self.param_names[i])()
            key_group[self.param_rank[i]] = key_group[self.param_rank[i]] + (key,)
        new_param_group = []
        for root in range(self.dev_num):
            ops = P.Broadcast(root)
            if root > 0:
                param_group[root] = F.depend(param_group[root], new_param_group[root-1])
            else:
                param_group[root] = F.depend(param_group[root], optim_result)
            next_params = ops(param_group[root])
            new_param_group.append(next_params)
            for i in range(F.tuple_len(next_params)):
                F.assign(key_group[root][i], next_params[i])
        return new_param_group

    def construct(self, *hyper_params):
        raise NotImplementedError


op_add = P.AddN()
op_gather = P.Gather()
op_mul = P.Mul()
op_gc = inner.Centralization()

_apply_decay = C.MultitypeFuncGraph("apply_decay")
_apply_grad_centralization = C.MultitypeFuncGraph("apply_grad_centralization")


@_apply_decay.register("Tensor", "Bool", "Tensor", "RowTensor")
def _tensor_apply_decay_with_sparse(weight_decay, if_apply, weight, gradient):
    """Get grad with weight_decay."""
    if if_apply:
        indices = gradient.indices
        values = op_add((op_gather(weight, indices, 0) * F.cast(weight_decay, F.dtype(weight)), gradient.values))
        shape = gradient.dense_shape
        return RowTensor(indices, values, shape)
    return gradient


@_apply_decay.register("Tensor", "Bool", "Tensor", "Tensor")
def _tensor_apply_decay(weight_decay, if_apply, weight, gradient):
    """Get grad with weight_decay."""
    if if_apply:
        return op_add((op_mul(weight, F.cast(weight_decay, F.dtype(weight))), gradient))
    return gradient


@_apply_grad_centralization.register("Bool", "RowTensor")
def _tensor_apply_grad_centralization_with_sparse(if_apply, gradient):
    """Get grad with grad_centralization."""
    if if_apply:
        indices = gradient.indices
        values = op_gc(gradient.values, -1)
        shape = gradient.dense_shape
        return RowTensor(indices, values, shape)
    return gradient


@_apply_grad_centralization.register("Bool", "Tensor")
def _tensor_apply_grad_centralization(if_apply, gradient):
    """Get grad with grad_centralization."""
    if if_apply:
        return op_gc(gradient, -1)
    return gradient


_grad_scale = C.MultitypeFuncGraph("grad_scale")
_indices_deduplicate = C.MultitypeFuncGraph("indices_deduplicate")


@_grad_scale.register("Number", "Tensor")
def tensor_grad_scale(scale, grad):
    """Get grad with scale."""
    if scale == 1.0:
        return grad
    return op_mul(grad, F.cast(scale, F.dtype(grad)))


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale_with_tensor(scale, grad):
    """Get grad with scale."""
    return op_mul(grad, F.cast(scale, F.dtype(grad)))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_with_sparse(scale, grad):
    """Get grad with scale."""
    return RowTensor(grad.indices, grad.values * F.cast(scale, F.dtype(grad.values)), grad.dense_shape)


@_indices_deduplicate.register("RowTensor")
def rowtensor_deduplicate_indices_slices(grad):
    """Unique the indices and sums the 'values' corresponding to the duplicate indices."""
    indices = grad.indices
    values = grad.values

    unique_indices, index_position = P.Unique()(indices)
    summed_values = P.UnsortedSegmentSum()(values, index_position, P.DynamicShape()(unique_indices)[0])

    return RowTensor(unique_indices, summed_values, grad.dense_shape)


@_indices_deduplicate.register("Tensor")
def tensor_deduplicate_indice_slices(grad):
    """Return the input gradient directly in the dense sences."""
    return grad


class _ConvertToCell(LearningRateSchedule):
    """Inner api, convert learning rate of scalar to LearningRateSchedule."""
    def __init__(self, learning_rate):
        super(_ConvertToCell, self).__init__()
        if not isinstance(learning_rate, Parameter):
            raise TypeError('Learning rate must be Parameter.')
        self.learning_rate = learning_rate

    def construct(self, global_step):
        return self.learning_rate + 1.0 - 1.0


class _IteratorLearningRate(LearningRateSchedule):
    """Inner api, convert learning rate of Tensor(list) to LearningRateSchedule."""
    def __init__(self, learning_rate, name):
        super(_IteratorLearningRate, self).__init__()
        if isinstance(learning_rate, Tensor):
            if learning_rate.ndim != 1:
                raise ValueError("The dim of `Tensor` type dynamic learning rate should be a 1,"
                                 f"but got {learning_rate.ndim}.")
        else:
            raise TypeError("Learning rate should be Tensor.")

        self.learning_rate = Parameter(learning_rate, name)
        self.gather = P.Gather()

    def construct(self, global_step):
        return self.gather(self.learning_rate, global_step, 0)
