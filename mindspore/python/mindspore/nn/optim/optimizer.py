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
"""optimizer"""
from __future__ import absolute_import

import inspect
from typing import Iterable
import numpy as np

import mindspore
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.ops.operations import _inner_ops as inner
from mindspore.nn.cell import Cell
from mindspore.nn.layer.container import CellList
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common.initializer import initializer
from mindspore.common import Tensor
from mindspore.common.sparse_tensor import RowTensorInner
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator as validator
from mindspore import log as logger
from mindspore.parallel._utils import _get_global_rank, _get_device_num, _get_parallel_mode
from mindspore.parallel._ps_context import _is_ps_mode
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.nn.optim._dist_optimizer_registry import generate_dist_optimizer_list

__all__ = ['Optimizer', 'opt_init_args_register']


def opt_init_args_register(fn):
    """Register optimizer init args."""

    def deco(self, *args, **kwargs):
        bound_args = inspect.signature(fn).bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        arguments = bound_args.arguments
        arguments.pop('self')
        if 'params' in arguments.keys():
            setattr(self, 'init_params', dict({"params": arguments['params']}))
            arguments.pop('params')
        if 'optimizer' in arguments.keys():
            setattr(self, 'init_params', dict({"params": arguments['optimizer'].init_params["params"]}))
            arguments.pop('optimizer')
        if 'learning_rate' in arguments.keys():
            if isinstance(arguments['learning_rate'], Tensor):
                arguments['learning_rate'] = arguments['learning_rate'].asnumpy().tolist()
            if isinstance(arguments['learning_rate'], Cell):
                setattr(self, 'init_learning_rate', None)
            else:
                setattr(self, 'init_learning_rate', arguments['learning_rate'])
            arguments.pop('learning_rate')
        setattr(self, 'init_args', arguments)
        fn(self, *args, **kwargs)

    return deco


class Optimizer(Cell):
    """
    Base class for updating parameters. Never use this class directly, but instantiate one of its subclasses instead.

    Grouping parameters is supported. If parameters are grouped, different strategy of `learning_rate`, `weight_decay`
    and `grad_centralization` can be applied to each group.

    Note:
        If parameters are not grouped, the `weight_decay` in optimizer will be applied on the network parameters without
        'beta' or 'gamma' in their names. Users can group parameters to change the strategy of decaying weight. When
        parameters are grouped, each group can set `weight_decay`. If not, the `weight_decay` in optimizer will be
        applied.

    Args:
        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule]):

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        parameters (Union[list[Parameter], list[dict]]): Must be list of `Parameter` or list of `dict`. When the
            `parameters` is a list of `dict`, the string "params", "lr", "weight_decay", "grad_centralization" and
            "order_params" are the keys can be parsed.

            - params: Required. Parameters in current group. The value must be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in optimizer will be used. Fixed and dynamic learning rate are supported.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the optimizer will be used.

            - grad_centralization: Optional. Must be Boolean. If "grad_centralization" is in the keys, the set value
              will be used. If not, the `grad_centralization` is False by default. This configuration only works on the
              convolution layer.

            - order_params: Optional. When parameters is grouped, this usually is used to maintain the order of
              parameters that appeared in the network to improve performance. The value should be parameters whose
              order will be followed in optimizer.
              If `order_params` in the keys, other keys will be ignored and the element of 'order_params' must be in
              one group of `params`.

        weight_decay (Union[float, int]): An int or a floating point value for the weight decay.
            It must be equal to or greater than 0.
            If the type of `weight_decay` input is int, it will be converted to float. Default: 0.0.
        loss_scale (float): A floating point value for the loss scale. It must be greater than 0. If the
            type of `loss_scale` input is int, it will be converted to float. In general, use the default value. Only
            when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to False, this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.amp.FixedLossScaleManager` for more details.
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
        ``Ascend`` ``GPU`` ``CPU``
    """
    _support_parallel_optimizer = False

    def __init__(self, learning_rate, parameters, weight_decay=0.0, loss_scale=1.0):
        super(Optimizer, self).__init__(auto_prefix=False)
        parameters = self._parameters_base_check(parameters, "parameters")
        self.param_rank = None
        self.optim_filter = None
        if not all(isinstance(x, Parameter) for x in parameters) and not all(isinstance(x, dict) for x in parameters):
            raise TypeError("For 'Optimizer', all elements of the argument 'parameters' must be 'Parameter' or 'dict',"
                            " please check the 'parameters'.")

        if isinstance(loss_scale, int):
            loss_scale = float(loss_scale)
        validator.check_value_type("loss_scale", loss_scale, [float], self.cls_name)
        validator.check_positive_float(loss_scale, "loss_scale", self.cls_name)
        self.loss_scale = loss_scale
        self.dynamic_weight_decay = False
        self.grad_centralization = False

        self._unique = True
        self._target = context.get_context("device_target")
        self._use_flattened_params = False
        self._grad_fusion_size = 0
        self.dynamic_lr = False
        self.assignadd = P.AssignAdd()
        self.global_step = Parameter(initializer(0, [1], mindspore.int32), name='global_step')
        self.is_group = False
        self.is_group_lr = False
        self.is_group_params_ordered = False
        self.use_parallel = False
        learning_rate = self._preprocess_single_lr(learning_rate)
        if isinstance(parameters[0], dict):
            self.is_group = True
            self.group_params = []
            self.group_lr = []
            self.group_weight_decay = []
            self.group_grad_centralization = []
            self._init_group_params(parameters, learning_rate, weight_decay, self.grad_centralization)

        self._init_opt_attrs(learning_rate, parameters, weight_decay)
        self.add_flags(skip_auto_parallel_compile=True)

    def _init_opt_attrs(self, learning_rate, parameters, weight_decay):
        """initialize optimizer attributions"""
        weight_decay = self._preprocess_weight_decay(weight_decay)
        if self.is_group_lr:
            if self.dynamic_lr:
                self.learning_rate = CellList(self.group_lr, auto_prefix=False)
            else:
                self.learning_rate = ParameterTuple(self.group_lr)
        else:
            self.learning_rate = self._build_single_lr(learning_rate, 'learning_rate')

        if self.is_group:
            self.parameters = ParameterTuple(self.group_params)
            self._parameters = self.parameters
            decay_filter = lambda x: isinstance(x, Cell) or x > 0
            dynamic_decay_filter = lambda x: isinstance(x, Cell)
            self.decay_flags = tuple(decay_filter(x) for x in self.group_weight_decay)
            self.dynamic_decay_flags = tuple(dynamic_decay_filter(x) for x in self.group_weight_decay)
            self.weight_decay = tuple(x if flag else Tensor(x, mstype.float32)
                                      for x, flag in zip(self.group_weight_decay, self.dynamic_decay_flags))
            self.exec_weight_decay = any(self.decay_flags)
            self.grad_centralization_flags = tuple(self.group_grad_centralization)
        else:
            self.parameters = ParameterTuple(parameters)
            flat_params = self._get_flattened_params(parameters)
            if self._use_flattened_params:
                self._parameters = ParameterTuple(flat_params)
            else:
                self._parameters = self.parameters
            decay_filter = lambda x: 'beta' not in x.name and 'gamma' not in x.name
            self.decay_flags = tuple(decay_filter(x) for x in self._parameters)
            self.dynamic_decay_flags = isinstance(weight_decay, Cell)
            self.exec_weight_decay = isinstance(weight_decay, Cell) or weight_decay > 0
            self.weight_decay = Tensor(weight_decay, mstype.float32) if not self.dynamic_decay_flags else weight_decay
        # when a parameter has been unique, there is no need do another unique in optimizer.
        for param in self._parameters:
            if param.unique:
                self._unique = False
                break
        # set user's parameters as local parameters
        for param in self._parameters:
            self._user_parameters.append(param.name)
        ps_filter = lambda x: x.is_param_ps
        self.ps_parameters = tuple(ps_filter(x) for x in self._parameters)
        cache_filter = lambda x: x.cache_enable
        self.cache_enable = tuple(cache_filter(x) for x in self._parameters)
        self.reciprocal_scale = Tensor(1.0 / self.loss_scale, mstype.float32)
        self.need_scale = self.loss_scale != 1.0
        self.global_step_increase_tensor = Tensor(1, mstype.int32)
        self.param_length = len(self._parameters)
        self.map_ = C.Map()
        self.map_reverse = C.Map(None, True)
        self.hyper_map = C.HyperMap()
        self.hyper_map_reverse = C.HyperMap(None, True)
        self._use_parallel_optimizer()
        self.enable_tuple_broaden = True

    def _get_flattened_params(self, parameters):
        """Get parameters for each contiguous memory chunks used by input parameters if they are flattened."""
        if self.is_group:
            # We don't use flattened parameters when parameters are grouped.
            return parameters
        # Check whether parameters are flattened.
        flattened = Tensor._is_flattened(parameters)  # pylint: disable=W0212
        if not flattened:
            # Parameters are not flattened.
            return parameters
        # Try to get chunk tensors from flattened parameters.
        chunk_tensors = Tensor._get_flattened_tensors(parameters)  # pylint: disable=W0212
        if not chunk_tensors:
            # Failed to get chunk tensors.
            logger.warning("Parameters are not properly flattened, fallback to not flattened parameters.")
            return parameters
        # Convert chunk tensors to parameters.
        self._use_flattened_params = True
        self._grad_fusion_size = Tensor._get_fusion_size(chunk_tensors)  # pylint: disable=W0212
        return [Parameter._from_tensor(t, name='_chunk_param' + str(i) + '_' + str(t.dtype))  # pylint: disable=W0212
                for i, t in enumerate(chunk_tensors)]

    def _use_parallel_optimizer(self):
        """Indicates whether to use automatic parallelism."""
        if context.get_auto_parallel_context("enable_parallel_optimizer"):
            if _get_parallel_mode() == ParallelMode.DATA_PARALLEL and context.get_context("device_target") == "Ascend":
                self.use_parallel = True
            elif _get_parallel_mode() == ParallelMode.DATA_PARALLEL \
                    and context.get_context("device_target") != "Ascend":
                raise RuntimeError(f'For "Optimizer", parallel optimizer only supports "Ascend" in data parallel mode, '
                                   f'but got {context.get_context("device_target")}.')
            elif _get_parallel_mode() in (ParallelMode.STAND_ALONE, ParallelMode.HYBRID_PARALLEL):
                raise RuntimeError("For 'Optimizer', parallel optimizer is not supported in {}, you should set "
                                   "parallel mode to 'data_parallel', 'semi_auto_parallel' or 'auto_parallel'."
                                   .format(_get_parallel_mode()))

        if self.use_parallel:
            if not self._support_parallel_optimizer:
                raise RuntimeError("For 'Optimizer', parallel optimizer only support optimizer 'Lamb' and "
                                   "'AdamWeightDecay' and 'AdaFactor', but got {}.".format(self.cls_name))
            self.dev_num = _get_device_num()
            if self.dev_num > self.param_length:
                raise RuntimeError("Parallel optimizer can not be applied when the number of parameters {} is"
                                   " less than the number of devices {}".format(self.param_length, self.dev_num))
            self.param_rank = self._get_parameter_group_id()
            self.optim_filter = tuple(map(lambda x: x == _get_global_rank(), self.param_rank))
            self.param_names = []
            for param in self._parameters:
                self.param_names.append(param.name)
        else:
            self.optim_filter = (True,) * self.param_length

    @property
    def unique(self):
        """
        Whether to make the gradients unique in optimizer. Generally, it is used in sparse networks. Set to True if the
        gradients of the optimizer are sparse, while set to False if the forward network has made the parameters unique,
        that is, the gradients of the optimizer is no longer sparse.
        The default value is True when it is not set.
        """
        return self._unique

    @unique.setter
    def unique(self, value):
        """Set the `unique` attribute."""
        if not isinstance(value, bool):
            raise TypeError("For 'Optimizer', the property 'unique' must be bool, "
                            "but got {}".format(type(value)))
        self._unique = value

    @property
    def target(self):
        """
        The property is used to determine whether the parameter is updated on host or device. The input type is str
        and can only be 'CPU', 'Ascend' or 'GPU'.
        """
        return self._target

    @target.setter
    def target(self, value):
        """
        If the input value is set to "CPU", the parameters will be updated on the host using the Fused
        optimizer operation.
        """
        raise NotImplementedError

    @staticmethod
    def _preprocess_grad_centralization(grad_centralization):
        if not isinstance(grad_centralization, bool):
            raise TypeError("For 'Optimizer', the 'gradients_centralization' must be bool type, "
                            "but got {}.".format(type(grad_centralization)))
        return grad_centralization

    @staticmethod
    def _parameters_base_check(parameters, param_info):
        """Parameters base check."""
        if parameters is None:
            raise ValueError(f"For 'Optimizer', the argument {param_info} can not be None.")
        if not isinstance(parameters, Iterable):
            raise TypeError(f"For 'Optimizer', the argument {param_info} must be Iterable type, "
                            f"but got {type(parameters)}.")
        parameters = list(parameters)

        if not parameters:
            raise ValueError(f"For 'Optimizer', the argument {param_info} must not be empty.")
        return parameters

    @staticmethod
    def _use_distibuted_optimizer():
        """
        Whether use distributed optimizers.
        """
        return _is_ps_mode()

    def flatten_gradients(self, gradients):
        """
        Flatten gradients into several chunk tensors grouped by data type if network parameters are flattened.

        A method to enable performance improvement by using contiguous memory for parameters and gradients.
        User-defined optimizers based on :class:`mindspore.nn.Optimizer` should call this interface to support
        contiguous memory for network parameters.

        Args:
            gradients (tuple[Tensor]): The gradients of network parameters.

        Returns:
            tuple[Tensor], The gradients after flattened, or the original gradients if parameters are not flattened.
        """
        if self._use_flattened_params:
            flatten_concat = inner.FlattenConcat(fusion_size=self._grad_fusion_size)
            return flatten_concat(gradients)
        return gradients

    def decay_weight(self, gradients):
        """
        Weight decay.

        An approach to reduce the overfitting of a deep learning neural network model. User-defined optimizers based
        on :class:`mindspore.nn.Optimizer` can also call this interface to apply weight decay.

        Args:
            gradients (tuple[Tensor]):The gradients of network parameters, and have the same shape as the parameters.

        Returns:
            tuple[Tensor], The gradients after weight decay.
        """
        if self.exec_weight_decay:
            params = self._parameters
            weight_decay = self.get_weight_decay()
            if self.is_group:
                gradients = self.map_(F.partial(_apply_decay), weight_decay, self.decay_flags, params, gradients)
            else:
                gradients = self.map_(F.partial(_apply_decay, weight_decay), self.decay_flags, params, gradients)

        return gradients

    def gradients_centralization(self, gradients):
        """
        Gradients centralization.

        A method for optimizing convolutional layer parameters to improve the training speed of a deep learning neural
        network model. User-defined optimizers based on :class:`mindspore.nn.Optimizer` can also call this interface to
        centralize gradients.

        Args:
            gradients (tuple[Tensor]): The gradients of network parameters, and have the same shape as the parameters.

        Returns:
            tuple[Tensor], The gradients after gradients centralization.
        """
        if self.is_group:
            gradients = self.map_(F.partial(_apply_grad_centralization), self.grad_centralization_flags, gradients)

        return gradients

    def scale_grad(self, gradients):
        """
        Restore gradients for mixed precision.

        User-defined optimizers based on :class:`mindspore.nn.Optimizer` can also call this interface to restore
        gradients.

        Args:
            gradients (tuple[Tensor]): The gradients of network parameters, and have the same shape as the parameters.

        Returns:
            tuple[Tensor], The gradients after loss scale.

        """
        if self.need_scale:
            gradients = self.map_(F.partial(_grad_scale, self.reciprocal_scale), gradients)

        return gradients

    def _set_base_target(self, value):
        """
        If the input value is set to "CPU", the parameters will be updated on the host using the Fused
        optimizer operation.
        """
        if not isinstance(value, str):
            raise TypeError("For 'Optimizer', the property 'target' must be string, but got {}".format(type(value)))

        if value not in ('CPU', 'Ascend', 'GPU'):
            raise ValueError("For 'Optimizer', the property 'target' must be one of ['CPU', 'Ascend' ,'GPU'], "
                             "but got {}".format(value))

        if self._target == "CPU" and value in ('Ascend', 'GPU'):
            raise ValueError("For 'Optimizer', the property 'target' cannot be set to 'GPU' or 'Ascend' "
                             "in the 'CPU' environment.")

        if self._target == "Ascend" and value == 'GPU':
            raise ValueError("For 'Optimizer', the property 'target' cannot be set to 'GPU' "
                             "in the 'Ascend' environment.")

        if self._target == "GPU" and value == 'Ascend':
            raise ValueError("For 'Optimizer', the property 'target' cannot be set to 'Ascend' "
                             "in the 'GPU' environment.")

        self._is_device = (value != 'CPU')
        self._target = value

    def _grad_sparse_indices_deduplicate(self, gradients):
        """ In the case of using big operators, deduplicate the 'indexes' in gradients."""
        if self._target != 'CPU' and self._unique:
            gradients = self.map_(F.partial(_indices_deduplicate), gradients)
        return gradients

    def _preprocess_weight_decay(self, weight_decay):
        """preprocess weight decay"""
        if isinstance(weight_decay, (float, int)):
            weight_decay = float(weight_decay)
            validator.check_non_negative_float(weight_decay, "weight_decay", self.cls_name)
            weight_decay = weight_decay * self.loss_scale
        elif isinstance(weight_decay, Cell):
            self.dynamic_weight_decay = True
            weight_decay = _WrappedWeightDecay(weight_decay, self.loss_scale)
        else:
            raise TypeError("For 'Optimizer', the argument 'Weight_decay' must be int, "
                            "float or Cell.but got {}".format(type(weight_decay)))
        return weight_decay

    def _preprocess_single_lr(self, learning_rate):
        """Check lr value, and convert lr to a float, a Tensor or a LearningRateSchedule."""
        if isinstance(learning_rate, (float, int)):
            learning_rate = float(learning_rate)
            validator.check_non_negative_float(learning_rate, "learning rate", self.cls_name)
            return learning_rate
        if isinstance(learning_rate, Tensor) and learning_rate.ndim == 0:
            learning_rate = Tensor(learning_rate.asnumpy(), dtype=mstype.float32)
            return learning_rate

        self.dynamic_lr = True
        if isinstance(learning_rate, Iterable):
            return Tensor(np.array(list(learning_rate)).astype(np.float32))
        if isinstance(learning_rate, Tensor):
            if learning_rate.ndim > 1:
                raise ValueError(f"For 'Optimizer', if 'learning_rate' is Tensor type, then the dimension of it should "
                                 f"be 0 or 1, but got {learning_rate.ndim}.")
            if learning_rate.ndim == 1 and learning_rate.size < 2:
                logger.warning("For 'Optimizer', if use 'Tensor' type dynamic learning rate, "
                               "please make sure that the number "
                               "of elements in the tensor is greater than 1, "
                               "but got {}.".format(learning_rate.size))
            learning_rate = Tensor(learning_rate.asnumpy(), dtype=mstype.float32)
            return learning_rate
        if isinstance(learning_rate, LearningRateSchedule):
            return learning_rate
        raise TypeError("For 'Optimizer', the argument 'learning_rate' must be int, float, Tensor, Iterable or "
                        "LearningRateSchedule, but got {}.".format(type(learning_rate)))

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
                raise KeyError(f"For 'Optimizer', the key in group params must be one of in {parse_keys}, "
                               f"but got {invalid_key}.")

            if 'order_params' in group_param.keys():
                if len(group_param.keys()) > 1:
                    raise ValueError(f"For 'Optimizer', the order params dict in group parameters should only "
                                     f"include the 'order_params' key, but got {group_param.keys()}.")
                if not isinstance(group_param['order_params'], Iterable):
                    raise TypeError("For 'Optimizer', the value of 'order_params' in group parameters should "
                                    "be Iterable type, but got {}.".format(type(group_param['order_params'])))
                continue

            parameters = self._parameters_base_check(group_param['params'], "group `params`")
            for index, param in enumerate(parameters):
                if not isinstance(param, Parameter):
                    raise TypeError(f"For 'Optimizer', the element in group parameters must be Parameter type, "
                                    f"but got {type(param)} at index {index}.")

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
                    raise ValueError(f"For 'Optimizer', the order params dict in group parameters should only include "
                                     f"the 'order_params' key, but got {group_param.keys()}.")
                if not isinstance(group_param['order_params'], Iterable):
                    raise TypeError("For 'Optimizer', the value of 'order_params' in group parameters must be "
                                    "Iterable type, but got {}.".format(type(group_param['order_params'])))
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
                        raise ValueError("For 'Optimizer', the Tensor type dynamic learning rate in group must be "
                                         "the same size as the argument 'learning_rate'.")

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
                weight_decay_ = self._preprocess_weight_decay(group_param['weight_decay'])
            else:
                weight_decay_ = self._preprocess_weight_decay(weight_decay)

            if 'grad_centralization' in group_param.keys():
                self.grad_centralization = self._preprocess_grad_centralization(group_param['grad_centralization'])
                for param in group_param['params']:
                    validator.check_value_type("parameter", param, [Parameter], self.cls_name)
                    grad_centralization_ = self.grad_centralization
            else:
                grad_centralization_ = grad_centralization

            for key in group_param.keys():
                if key not in ('params', 'lr', 'weight_decay', 'grad_centralization'):
                    logger.warning(f"The optimizer cannot parse '{key}' when setting parameter groups, "
                                   f"the key should in ['params', 'lr', 'weight_decay', 'grad_centralization']")

            for param in group_param['params']:
                validator.check_value_type("parameter", param, [Parameter], self.cls_name)
                if param.name in params_store:
                    raise RuntimeError(f"For 'Optimizer', the {param.name} parameter already exists, it does not "
                                       f"support repeated setting. Please check whether the optimizer parameter "
                                       f"has been set multiple times.")

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
            raise ValueError(f"For 'Optimizer',"
                             f"the length of order parameters must be the same as the length of group parameters, "
                             f"but got order parameters' length {len(ordered_parameters)}, "
                             f"group parameters' length {len(self.group_params)}.")

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

    def get_weight_decay(self):
        """
        The optimizer calls this interface to get the weight decay value for the current step.
        User-defined optimizers based on :class:`mindspore.nn.Optimizer` can also call this interface
        before updating the parameters.

        Returns:
            float, the weight decay value of current step.
        """
        if self.dynamic_weight_decay:
            if self.is_group:
                weight_decay = ()
                for weight_decay_, flag_ in zip(self.weight_decay, self.dynamic_decay_flags):
                    current_weight_decay = weight_decay_(self.global_step) if flag_ else weight_decay_
                    weight_decay += (current_weight_decay,)
                return weight_decay
            return self.weight_decay(self.global_step)
        return self.weight_decay

    def get_lr(self):
        """
        The optimizer calls this interface to get the learning rate for the current step. User-defined optimizers based
        on :class:`mindspore.nn.Optimizer` can also call this interface before updating the parameters.

        Returns:
            float, the learning rate of current step.
        """
        lr = self.learning_rate
        if self.dynamic_lr:
            if self.is_group_lr:
                lr = ()
                for learning_rate in self.learning_rate:
                    current_dynamic_lr = learning_rate(self.global_step).reshape(())
                    lr += (current_dynamic_lr,)
            else:
                lr = self.learning_rate(self.global_step).reshape(())
        if self._is_dynamic_lr_or_weight_decay():
            self.assignadd(self.global_step, self.global_step_increase_tensor)
        return lr

    def get_lr_parameter(self, param):
        """
        When parameters is grouped and learning rate is different for each group, get the learning rate of the specified
        `param`.

        Args:
            param (Union[Parameter, list[Parameter]]): The `Parameter` or list of `Parameter`.

        Returns:
            Parameter, single `Parameter` or `list[Parameter]` according to the input type. If learning rate is dynamic,
            `LearningRateSchedule` or `list[LearningRateSchedule]` that used to calculate the learning rate will be
            returned.

        Examples:
            >>> from mindspore import nn
            >>> # net = LeNet5()
            >>> net = Net()
            >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
            >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
            >>> group_params = [{'params': conv_params, 'lr': 0.05},
            ...                 {'params': no_conv_params, 'lr': 0.01}]
            >>> optim = nn.Momentum(group_params, learning_rate=0.1, momentum=0.9, weight_decay=0.0)
            >>> conv_lr = optim.get_lr_parameter(conv_params)
            >>> print(conv_lr[0].asnumpy())
            0.05
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
            raise TypeError(f"For 'get_lr_parameter', the 'param' must be 'Parameter' or 'list' type, "
                            f"but got {type(param)}.")

        lr = []
        ids = [id(p) for p in self._parameters]
        for p in param_list:
            validator.check_value_type("parameter", p, [Parameter], self.cls_name)
            if id(p) not in ids:
                raise ValueError(f"For 'get_lr_parameter', the parameter {p.name} is not in optimizer, please check "
                                 f"whether the argument 'param' is correct.")
            if self.is_group_lr:
                index = ids.index(id(p))
                lr.append(get_lr_value(self.learning_rate[index]))
            else:
                lr.append(get_lr_value(self.learning_rate))

        return lr if isinstance(param, list) else lr[0]

    def _is_dynamic_lr_or_weight_decay(self):
        """
        Determine whether the learning rate or weight decay is dynamic.

        Returns:
             bool, represents the learning rate or weight decay is dynamic or not.
        """
        return self.dynamic_lr or self.dynamic_weight_decay

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

        Args:
            optim_result(bool): The results of updating parameters. This input is used to ensure that the parameters are
              updated before they are broadcast.
        Returns:
             bool, the status flag.
        """
        # If rank_id is 0, 1, 2, 3, there are param0 ~ param7,
        # then the value is[(param0, param4), (param1, param5), (param2, param6), (param3, param7)]
        param_group = []
        for _ in range(self.dev_num):
            param_group.append(F.make_tuple())
        for i in range(self.param_length):
            param_group[self.param_rank[i]] = param_group[self.param_rank[i]] + (self._parameters[i],)
        new_param_group = []
        for root in range(self.dev_num):
            if root > 0:
                depend = F.depend(param_group[root], new_param_group[root - 1])
            else:
                depend = F.depend(param_group[root], optim_result)
            next_params = P.Broadcast(root)(depend)
            new_param_group.append(next_params)
            for i in range(F.tuple_len(next_params)):
                F.assign(param_group[root][i], next_params[i])
        return new_param_group

    def _get_distributed_optimizer_list(self, optimizer_type, *args, **kwargs):
        """
        Get the distributed optimizers list in distributed training mode.
        """
        return generate_dist_optimizer_list(optimizer_type, self._parameters, *args, **kwargs)

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
        return RowTensorInner(indices, values, shape)
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
        shape = gradient.dense_shape
        grad_shape = F.shape(gradient)
        axis = []
        for i in range(1, len(grad_shape)):
            axis.append(i)
        if len(axis) >= 1:
            if grad_shape[1] % 16 != 0:
                return gradient
            values = op_gc(gradient.values, axis)
            return RowTensorInner(indices, values, shape)
    return gradient


@_apply_grad_centralization.register("Bool", "Tensor")
def _tensor_apply_grad_centralization(if_apply, gradient):
    """Get grad with grad_centralization."""
    if if_apply:
        axis = []
        grad_shape = F.shape(gradient)
        for i in range(1, len(grad_shape)):
            axis.append(i)
        if len(axis) >= 1:
            if grad_shape[1] % 16 != 0:
                return gradient
            return op_gc(gradient, axis)
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
    return RowTensorInner(grad.indices, grad.values * F.cast(scale, F.dtype(grad.values)), grad.dense_shape)


@_grad_scale.register("Tensor", "MapTensor")
def tensor_grad_scale_with_map_tensor(scale, grad):
    """Get grad with scale."""
    return grad


@_indices_deduplicate.register("RowTensor")
def rowtensor_deduplicate_indices_slices(grad):
    """Unique the indices and sums the 'values' corresponding to the duplicate indices."""
    indices = grad.indices
    values = grad.values

    unique_indices, index_position = P.Unique()(indices)
    summed_values = P.UnsortedSegmentSum()(values, index_position, P.TensorShape()(unique_indices)[0])

    return RowTensorInner(unique_indices, summed_values, grad.dense_shape)


@_indices_deduplicate.register("Tensor")
def tensor_deduplicate_indice_slices(grad):
    """Return the input gradient directly in the dense sences."""
    return grad


class _ConvertToCell(LearningRateSchedule):
    """Inner api, convert learning rate of scalar to LearningRateSchedule."""

    def __init__(self, learning_rate):
        super(_ConvertToCell, self).__init__()
        if not isinstance(learning_rate, Parameter):
            raise TypeError("For 'Optimizer', the argument 'learning_rate' must be Parameter, "
                            "but got {}.".format(type(learning_rate)))
        self.learning_rate = learning_rate

    def construct(self, global_step):
        return self.learning_rate + 1.0 - 1.0


class _IteratorLearningRate(LearningRateSchedule):
    """Inner api, convert learning rate of Tensor(list) to LearningRateSchedule."""

    def __init__(self, learning_rate, name):
        super(_IteratorLearningRate, self).__init__()
        if isinstance(learning_rate, Tensor):
            if learning_rate.ndim != 1:
                raise ValueError(f"For 'Optimizer', the dimension of the argument 'learning_rate' should "
                                 f"be 1, but got {learning_rate.ndim}.")
        else:
            raise TypeError("For 'Optimizer', the argument 'learning_rate' must be Tensor, "
                            "but got {}.".format(type(learning_rate)))

        self.learning_rate = Parameter(learning_rate, name)
        self.gather = P.Gather()

    def construct(self, global_step):
        return self.gather(self.learning_rate, global_step, 0)


class _WrappedWeightDecay(Cell):
    """Inner api, a combination of dynamic or non-dynamic weight decay"""

    def __init__(self, weight_decay, loss_scale=1.0):
        super(_WrappedWeightDecay, self).__init__()
        self.weight_decay = weight_decay
        self.loss_scale = Tensor(loss_scale, mstype.float32)

    def construct(self, global_step):
        return self.weight_decay(global_step) * self.loss_scale
