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
"""Time Distributed."""

from mindspore.ops.primitive import constexpr, Primitive
from mindspore.ops import Reshape, Transpose, Stack, Unstack
from mindspore.common import Tensor
from mindspore._checkparam import Validator
from ..cell import Cell

__all__ = ['TimeDistributed']


@constexpr
def _check_reshape_pos(reshape_pos, inputs_shape, outputs_shape):
    if reshape_pos >= len(outputs_shape) or inputs_shape[reshape_pos] != outputs_shape[reshape_pos]:
        raise ValueError("The parameter reshape_with_axis is invalid in the input and output of TimeDistributed. "
                         "You may try pass parameters without reshape_with_axis.")


@constexpr
def _check_expand_dims_axis(time_axis, ndim):
    if time_axis > ndim:
        raise ValueError("The parameter time_axis is invalid in the input. "
                         "The value of time_axis should be in range of [{}, {}].".format(-ndim - 1, ndim))


@constexpr
def _generate_perm(axis_a, axis_b, length):
    perm = tuple(range(length))
    axis_a, axis_b = (axis_a, axis_b) if axis_a < axis_b else (axis_b, axis_a)
    return perm[:axis_a] + (perm[axis_b],) + perm[axis_a: axis_b] + perm[axis_b + 1:]


@constexpr
def _check_data(flag):
    if not flag:
        raise TypeError("The inputs and outputs shuould be a Tensor.")


@constexpr
def _check_inputs_dim(shape):
    if len(shape) < 3:
        raise ValueError("The inputs should be at least 3D.")


class TimeDistributed(Cell):
    r"""
    The time distributed layer.

    Time distributed is a wrapper which allows to apply a layer to every temporal slice of an input.
    And the input should be at least 3D.
    There are two cases in the implementation.
    When reshape_with_axis provided, the reshape method will be chosen, which is more efficient;
    otherwise, the method of dividing the inputs along time axis will be used, which is more general.
    For example, reshape_with_axis could not be provided when deal with batch normal.

    Args:
        layer(Union[Cell, Primitive]): The Cell or Primitive which will be wrapped.
        time_axis(int): The axis of time_step.
        reshape_with_axis(int): The axis which will be reshaped with time_axis. Default: None.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, T, *)`.

    Outputs:
        Tensor of shape :math:`(N, T, *)`

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If layer is not a Cell or Primitive.

    Examples:
        >>> input = Tensor(np.random.random([32, 10, 3]), mindspore.float32)
        >>> dense = nn.Dense(3, 6)
        >>> net = nn.TimeDistributed(dense, time_axis=1, reshape_with_axis=0)
        >>> output = net(input)
        >>> print(output.shape)
        (32, 10, 6)
    """

    def __init__(self, layer, time_axis, reshape_with_axis=None):
        if not isinstance(layer, (Cell, Primitive)):
            raise TypeError("Please initialize TimeDistributed with mindspore.nn.Cell or "
                            "mindspore.ops.Primitive instance. You passed: {input}".format(input=layer))
        super(TimeDistributed, self).__init__()
        Validator.check_is_int(time_axis)
        if reshape_with_axis is not None:
            Validator.check_is_int(reshape_with_axis)
        self.layer = layer
        self.time_axis = time_axis
        self.reshape_with_axis = reshape_with_axis
        self.transpose = Transpose()
        self.reshape = Reshape()

    def construct(self, inputs):
        _check_data(isinstance(inputs, Tensor))
        _check_inputs_dim(inputs.shape)
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
            _check_data(isinstance(outputs, Tensor))
            _check_reshape_pos(reshape_pos, inputs.shape, outputs.shape)
            outputs_shape_new = outputs.shape[:reshape_pos] + inputs_shape_new[reshape_pos: reshape_pos + 2]
            if reshape_pos + 1 < len(outputs.shape):
                outputs_shape_new += outputs.shape[reshape_pos + 1:]
            return self.reshape(outputs, outputs_shape_new)

        unstack = Unstack(time_axis)
        inputs = unstack(inputs)
        y = ()
        for item in inputs:
            outputs = self.layer(item)
            _check_data(isinstance(outputs, Tensor))
            _check_expand_dims_axis(time_axis, outputs.ndim)
            y += (outputs,)
        y = Stack(time_axis)(y)
        return y
