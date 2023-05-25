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
"""Time Distributed."""
from __future__ import absolute_import

from mindspore.ops.primitive import constexpr, Primitive, _primexpr
from mindspore.ops import Reshape, Transpose, Stack, Unstack
from mindspore.common import Tensor
from mindspore import _checkparam as Validator
from mindspore.nn.cell import Cell

__all__ = ['TimeDistributed']


@_primexpr
def _check_reshape_pos(reshape_pos, inputs_shape, outputs_shape, prim_name=None):
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if reshape_pos >= len(outputs_shape) or inputs_shape[reshape_pos] != outputs_shape[reshape_pos]:
        raise ValueError(f"{msg_prefix} 'reshape_with_axis' is invalid in the input and output. "
                         f"The 'reshape_pos' must be less than the length of 'outputs_shape', and the "
                         f"'inputs_shape[reshape_pos]' must be equal to 'outputs_shape[reshape_pos]', but got "
                         f"'reshape_pos': {reshape_pos}, 'inputs_shape': {inputs_shape}, 'outputs_shape': "
                         f"{outputs_shape}. You may try pass parameters without 'reshape_with_axis'.")


@_primexpr
def _check_expand_dims_axis(time_axis, ndim, prim_name=None):
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if time_axis > ndim:
        raise ValueError(f"{msg_prefix} value of 'time_axis' must be in range of [{-ndim - 1}, {ndim}], "
                         f"but got {time_axis}.")


@constexpr
def _generate_perm(axis_a, axis_b, length):
    perm = tuple(range(length))
    axis_a, axis_b = (axis_a, axis_b) if axis_a < axis_b else (axis_b, axis_a)
    return perm[:axis_a] + (perm[axis_b],) + perm[axis_a: axis_b] + perm[axis_b + 1:]


@constexpr
def _check_data(flag, prim_name=None):
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if not flag:
        raise TypeError(f"{msg_prefix} inputs and outputs must be a Tensor.")


@_primexpr
def _check_inputs_dim(shape, prim_name=None):
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if len(shape) < 3:
        raise ValueError(f"{msg_prefix} inputs shape must be at least 3D, but got {len(shape)}.")


class TimeDistributed(Cell):
    r"""
    The time distributed layer.

    Time distributed is a wrapper which allows to apply a layer to every temporal slice of an input.
    And the `x` should be at least 3D.
    There are two cases in the implementation.
    When reshape_with_axis provided, the reshape method will be chosen, which is more efficient;
    otherwise, the method of dividing the inputs along time axis will be used, which is more general.
    For example, reshape_with_axis could not be provided when deal with Batch Normalization.

    Args:
        layer(Union[Cell, Primitive]): The Cell or Primitive which will be wrapped.
        time_axis(int): The axis of time_step.
        reshape_with_axis(int): The axis which will be reshaped with time_axis. Default: ``None`` .

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, T, *)`,
          where :math:`*` means any number of additional dimensions.

    Outputs:
        Tensor of shape :math:`(N, T, *)`

    Raises:
        TypeError: If layer is not a Cell or Primitive.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> x = ms.Tensor(np.random.random([32, 10, 3]), ms.float32)
        >>> dense = ms.nn.Dense(3, 6)
        >>> net = ms.nn.TimeDistributed(dense, time_axis=1, reshape_with_axis=0)
        >>> output = net(x)
        >>> print(output.shape)
        (32, 10, 6)
    """

    def __init__(self, layer, time_axis, reshape_with_axis=None):
        """Initialize TimeDistributed."""
        if not isinstance(layer, (Cell, Primitive)):
            raise TypeError(f"For '{self.cls_name}', the 'layer' must be Cell or Primitive instance, "
                            f"but got type: {type(layer).__name__}.")
        super(TimeDistributed, self).__init__()
        Validator.check_is_int(time_axis, "time_axis", self.cls_name)
        if reshape_with_axis is not None:
            Validator.check_is_int(reshape_with_axis, "reshape_with_axis", self.cls_name)
        self.layer = layer
        self.time_axis = time_axis
        self.reshape_with_axis = reshape_with_axis
        self.transpose = Transpose()
        self.reshape = Reshape()

    def construct(self, inputs):
        _check_data(isinstance(inputs, Tensor), self.cls_name)
        _check_inputs_dim(inputs.shape, self.cls_name)
        time_axis = self.time_axis % len(inputs.shape)
        if self.reshape_with_axis is not None:
            reshape_with_axis = self.reshape_with_axis % len(inputs.shape)
            inputs_shape = inputs.shape
            time_axis_new = len(inputs_shape) - 2 if reshape_with_axis == len(inputs_shape) - 1 \
                else (reshape_with_axis + 1 if time_axis > reshape_with_axis else
                      reshape_with_axis - 1)
            reshape_pos = time_axis_new if time_axis_new < reshape_with_axis else reshape_with_axis
            perm = _generate_perm(time_axis_new, time_axis, len(inputs_shape))
            inputs = self.transpose(inputs, perm)
            inputs_shape_new = inputs.shape
            inputs = self.reshape(inputs, inputs_shape_new[: reshape_pos] + (-1,) + inputs_shape_new[reshape_pos + 2:])
            outputs = self.layer(inputs)
            _check_data(isinstance(outputs, Tensor), self.cls_name)
            _check_reshape_pos(reshape_pos, inputs.shape, outputs.shape, self.cls_name)
            outputs_shape_new = outputs.shape[:reshape_pos] + inputs_shape_new[reshape_pos: reshape_pos + 2]
            if reshape_pos + 1 < len(outputs.shape):
                outputs_shape_new += outputs.shape[reshape_pos + 1:]
            outputs_shape_new = (-1,) + outputs_shape_new[1:]
            return self.reshape(outputs, outputs_shape_new)

        unstack = Unstack(time_axis)
        inputs = unstack(inputs)
        y = ()
        for item in inputs:
            outputs = self.layer(item)
            _check_data(isinstance(outputs, Tensor), self.cls_name)
            _check_expand_dims_axis(time_axis, outputs.ndim, self.cls_name)
            y += (outputs,)
        y = Stack(time_axis)(y)
        return y
