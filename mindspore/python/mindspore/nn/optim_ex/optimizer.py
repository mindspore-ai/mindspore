# Copyright 2023 Huawei Technologies Co., Ltd
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
from collections import defaultdict
from typing import Iterable
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.ops.operations import _inner_ops as inner
from mindspore.nn.cell import Cell
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common import Tensor
from mindspore.common.sparse_tensor import RowTensorInner
import mindspore.common.dtype as mstype
from mindspore import _checkparam as validator
from mindspore import log as logger


__all__ = ['Optimizer']


class Optimizer(Cell):
    r"""
    Base class for all optimizers.

    .. warning::
        This is an experimental optimizer API that is subject to change.
        This module must be used with lr scheduler module in `LRScheduler Class
        <https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html#lrscheduler>`_ .

    Args:
        params (Union[list(Parameter), list(dict)]): an iterable of :class:`mindspore.Parameter` or
            dict. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `weight_decay` is neither float nor int.
        ValueError: If `weight_decay` is less than 0.
        ValueError: If `learning_rate` is a Tensor, but the dimension of tensor is greater than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    def __init__(self, params, defaults):
        super(Optimizer, self).__init__(auto_prefix=False)

        param_groups = self._parameters_base_check(params, "params")
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []
        self.parameters = []
        self.map_ = C.Map()
        self.group_start_id = [0]
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for i, param_group in enumerate(param_groups):
            self.add_param_group(i, param_group)
            self.group_start_id.append(self.group_start_id[-1] + len(param_group["params"]))
        self.parameters = ParameterTuple(self.parameters)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key].value()) \
                        if key == "lr" and isinstance(group[key], Parameter) \
                        else '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string

    def add_param_group(self, group_id, param_group):
        r"""
        Add a param group to the `Optimizer.param_groups`.

        Args:
            group_id(int): Specifies the group index.
            param_group (dict): Specifies what Parameters should be optimized along with group
                specific optimization options.
        """
        param_group = self._preprocess_param_group(param_group)
        self.parameters += param_group["params"]

        for name, default in self.defaults.items():
            if name not in param_group:
                param_group.setdefault(name, default)

        lr = self._build_single_lr(param_group.get("lr"), 'learning_rate_group_' + str(group_id))
        weight_decay = self._preprocess_weight_decay(param_group.get("weight_decay", 0.0))
        param_group["lr"] = lr
        param_group["weight_decay"] = weight_decay
        param_group["grad_centralization"] = self._preprocess_grad_centralization(
            param_group.get('grad_centralization', False))
        self.param_groups.append(param_group)

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

    def _decay_weight(self, weight_decay, params, gradients):
        """Apply weight decay."""
        if weight_decay != 0.:
            weight_decay = Tensor(weight_decay, mstype.float32)
            gradients = self.map_(F.partial(_apply_decay, weight_decay), params, gradients)
        return gradients

    def _gradients_centralization(self, grad_centralization, gradients):
        """Apply gradients centralization."""
        if grad_centralization:
            return self.map_(_apply_grad_centralization, gradients)
        return gradients

    def _preprocess_param_group(self, param_group):
        """Preprocess param groups."""
        if not isinstance(param_group, dict):
            raise TypeError('Param group must be a dict.')

        params = param_group['params']
        if isinstance(params, Parameter):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('Optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. '
                            'Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, Parameter):
                raise TypeError("Optimizer can only optimize Parameters, but one of the params is " + type(param))

        if len(param_group['params']) != len(set(param_group['params'])):
            logger.warning("Optimizer contains a parameter group with duplicate parameters.")

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))
        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group.")
        return param_group

    def _build_single_lr(self, learning_rate, name):
        """Check lr value, and convert lr to a float or a Tensor."""
        if isinstance(learning_rate, (float, int)):
            learning_rate = float(learning_rate)
            validator.check_non_negative_float(learning_rate, "learning rate", self.cls_name)
            return Parameter(Tensor(learning_rate, mstype.float32), name)

        if isinstance(learning_rate, Tensor):
            if learning_rate.ndim == 0:
                return Parameter(learning_rate.astype(mstype.float32), name)
            raise ValueError(f"For 'Optimizer', if 'learning_rate' is a Tensor, "
                             f"then it should be scalar Tensor")

        raise TypeError("For 'Optimizer', the argument 'learning_rate' must be int, float or Tensor, "
                        "but got {}.".format(type(learning_rate)))

    def _preprocess_weight_decay(self, weight_decay):
        """preprocess weight decay"""
        if isinstance(weight_decay, (float, int)):
            weight_decay = float(weight_decay)
            validator.check_non_negative_float(weight_decay, "weight_decay", self.cls_name)
        else:
            raise TypeError("For 'Optimizer', the argument 'Weight_decay' must be int or "
                            "float.but got {}".format(type(weight_decay)))
        return weight_decay

    @staticmethod
    def _preprocess_grad_centralization(grad_centralization):
        """ Preprocess gradient centralization. """
        if not isinstance(grad_centralization, bool):
            raise TypeError("For 'Optimizer', the 'gradients_centralization' must be bool type, "
                            "but got {}.".format(type(grad_centralization)))
        return grad_centralization

    def construct(self, *hyper_params):
        raise NotImplementedError


op_add = P.AddN()
op_gather = P.Gather()
op_mul = P.Mul()
op_gc = inner.Centralization()

_apply_decay = C.MultitypeFuncGraph("apply_decay")
_apply_grad_centralization = C.MultitypeFuncGraph("apply_grad_centralization")


@_apply_decay.register("Tensor", "Tensor", "RowTensor")
def _tensor_apply_decay_with_sparse(weight_decay, weight, gradient):
    """Get grad with weight_decay."""
    indices = gradient.indices
    values = op_add((op_gather(weight, indices, 0) * F.cast(weight_decay, F.dtype(weight)), gradient.values))
    shape = gradient.dense_shape
    return RowTensorInner(indices, values, shape)


@_apply_decay.register("Tensor", "Tensor", "Tensor")
def _tensor_apply_decay(weight_decay, weight, gradient):
    """Get grad with weight_decay."""
    return op_add((op_mul(weight, F.cast(weight_decay, F.dtype(weight))), gradient))


@_apply_grad_centralization.register("RowTensor")
def _tensor_apply_grad_centralization_with_sparse(gradient):
    """Get grad with grad_centralization."""
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


@_apply_grad_centralization.register("Tensor")
def _tensor_apply_grad_centralization(gradient):
    """Get grad with grad_centralization."""
    axis = []
    grad_shape = F.shape(gradient)
    for i in range(1, len(grad_shape)):
        axis.append(i)
    if len(axis) >= 1:
        if grad_shape[1] % 16 != 0:
            return gradient
        return op_gc(gradient, axis)
    return gradient
