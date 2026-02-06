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
"""Operator argument handle function."""

from mindspore.common import Tensor
from mindspore.common import dtype as mstype
from mindspore import ops
from mindspore.ops.operations._sequence_ops import TensorToScalar

from .arg_dtype_cast import DtypeToEnum, StringToEnum
from . import arg_dtype_cast


def arg_invalid_info(op_name, arg_name, arg_val):
    """
    generate invalid msg.
    """
    return f"For '{op_name}', the value of '{arg_name}' is invalid: '{arg_val}'."


def to_pair(op_name, arg_name, arg_val):
    """
    convert arg_val: int/tuple[int*2] -> tuple[int*2].
    """
    if isinstance(arg_val, (int, float)):
        return (arg_val, arg_val)
    if isinstance(arg_val, (list, tuple)):
        return arg_val
    raise ValueError(arg_invalid_info(op_name, arg_name, arg_val))


def to_kernel_size(op_name, arg_name, kernel_size):
    """
    convert kernel_size: int/tuple[int*4] -> tuple[int*2].
    """
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size)
    if isinstance(kernel_size, (tuple, list)):
        if len(kernel_size) == 4:
            return (kernel_size[2], kernel_size[3])
        return kernel_size
    raise ValueError(arg_invalid_info(op_name, arg_name, kernel_size))


def to_strides(op_name, arg_name, stride):
    """
    convert strides: int/tuple[int*4] -> tuple[int*2].
    """
    if isinstance(stride, int):
        return (stride, stride)
    if isinstance(stride, (tuple, list)):
        if len(stride) == 4:
            return (stride[2], stride[3])
        return stride
    raise ValueError(arg_invalid_info(op_name, arg_name, stride))


def to_rates(op_name, arg_name, rates):
    """
    convert rates: int/tuple[int*4] -> tuple[int*2].
    """
    if isinstance(rates, int):
        return (rates, rates)
    if isinstance(rates, (tuple, list)):
        if len(rates) == 4:
            return (rates[2], rates[3])
        return rates
    raise ValueError(arg_invalid_info(op_name, arg_name, rates))


def to_dilations(op_name, arg_name, dilation):
    """
    convert dilations: int/tuple[int*4] -> tuple[int*2].
    """
    if isinstance(dilation, int):
        return (dilation, dilation)
    if isinstance(dilation, (tuple, list)):
        if len(dilation) == 4:
            return (dilation[2], dilation[3])
        return dilation
    raise ValueError(arg_invalid_info(op_name, arg_name, dilation))


def to_output_padding(op_name, arg_name, output_padding):
    """
    convert output_padding: int/tuple[int*4] -> tuple[int*2].
    """
    if isinstance(output_padding, int):
        return (output_padding, output_padding)
    if isinstance(output_padding, (tuple, list)):
        if len(output_padding) == 4:
            return (output_padding[2], output_padding[3])
        return output_padding
    raise ValueError(arg_invalid_info(op_name, arg_name, output_padding))


def to_2d_paddings(op_name, arg_name, pad):
    """
    convert paddings: int -> tuple[int*2].
    """
    if isinstance(pad, int):
        return (pad,) * 2
    if isinstance(pad, (tuple, list)):
        return pad
    raise ValueError(arg_invalid_info(op_name, arg_name, pad))


def generator_handler(op_name, arg_name, inputs):
    """
    convert constant value in tuple to tensor
    """
    new_inputs = []
    for input_ in inputs:
        if isinstance(input_, int):
            new_inputs.append(Tensor(input_, mstype.int64))
        else:
            new_inputs.append(input_)
    return tuple(new_inputs)


_tensor_to_scalar = TensorToScalar()


def _scalar_tensor_to_scalar(op_name, arg_name, x):
    """convert tensor to scalar"""
    if isinstance(x, Tensor):
        if len(x.shape) != 0:
            raise ValueError(arg_invalid_info(op_name, arg_name, x),
                             "Only support 0-dim tenor to scalar.")
        if arg_dtype_cast.is_integral_mstype(x.dtype):
            target_type = mstype.int64
        elif arg_dtype_cast.is_floating_point_mstype(x.dtype):
            target_type = mstype.float64
        elif x.dtype == mstype.bool_:
            target_type = mstype.bool_
        else:
            raise TypeError(arg_invalid_info(op_name, arg_name, x),
                            "Only support integral, floating-point and bool tenor to scalar.")
        return _tensor_to_scalar(ops.cast(x, target_type))
    return x


def _scalar_tensor_to_int(op_name, arg_name, x):
    """convert tensor to int"""
    # type check is left to arg parser
    return _scalar_tensor_to_scalar(op_name, arg_name, x)


def _scalar_tensor_to_float(op_name, arg_name, x):
    """convert tensor to float"""
    return _scalar_tensor_to_scalar(op_name, arg_name, x)


def _normalize_int_sequence(op_name, arg_name, data):
    """Normalize mixed int sequence."""
    if not isinstance(data, (list, tuple)):
        return data
    is_tuple = isinstance(data, tuple)
    return arg_dtype_cast.normalize_int_sequence(op_name, arg_name, data, is_tuple)

dtype_to_type_id = DtypeToEnum()

# string to enum
# A function for converting str type to enum type are written here,
# but the backend supports str input, and converting str input to enum input is not necessary.
str_to_enum = StringToEnum()

hidden_handlers = ['_scalar_tensor_to_scalar', '_scalar_tensor_to_int', '_scalar_tensor_to_float',
                   '_normalize_int_sequence']
normal_symbols = [name for name in globals() if not name.startswith('_')]
__all__ = hidden_handlers + normal_symbols
