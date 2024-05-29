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

from mindspore.ops_generate.gen_ops_inner_prim import DtypeToEnum, StringToEnum
# Enum Class:
from mindspore._c_expression import FormatEnum as Format
from mindspore._c_expression import ReductionEnum as Reduction
from mindspore.common import Tensor
from mindspore.common import dtype as mstype


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


def to_paddings(op_name, arg_name, pad):
    """
    convert paddings: int -> tuple[int*4].
    """
    if isinstance(pad, int):
        return (pad,) * 4
    if isinstance(pad, (tuple, list)):
        return pad
    raise ValueError(arg_invalid_info(op_name, arg_name, pad))


def to_3d_kernel_size(op_name, arg_name, kernel_size):
    """
    convert 3d kernel_size: int/tuple[int*6] -> tuple[int*3].
    """
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size, kernel_size)
    if isinstance(kernel_size, (tuple, list)):
        if len(kernel_size) == 5:
            return (kernel_size[2], kernel_size[3], kernel_size[4])
        return kernel_size
    raise ValueError(arg_invalid_info(op_name, arg_name, kernel_size))


def to_3d_strides(op_name, arg_name, stride):
    """
    convert 3d stride: int/tuple[int*6] -> tuple[int*3].
    """
    if isinstance(stride, int):
        return (stride, stride, stride)
    if isinstance(stride, (tuple, list)):
        if len(stride) == 5:
            return (stride[2], stride[3], stride[4])
        return stride
    raise ValueError(arg_invalid_info(op_name, arg_name, stride))


def to_3d_dilations(op_name, arg_name, dilation):
    """
    convert 3d dilation: int/tuple[int*6] -> tuple[int*3].
    """
    if isinstance(dilation, int):
        return (dilation, dilation, dilation)
    if isinstance(dilation, (tuple, list)):
        if len(dilation) == 5:
            return (dilation[2], dilation[3], dilation[4])
        return dilation
    raise ValueError(arg_invalid_info(op_name, arg_name, dilation))


def to_3d_paddings(op_name, arg_name, pad):
    """
    convert 3d paddings: int -> tuple[int*6].
    """
    if isinstance(pad, int):
        return (pad,) * 6
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

dtype_to_type_id = DtypeToEnum()

# string to enum
# A function for converting str type to enum type are written here,
# but the backend supports str input, and converting str input to enum input is not necessary.
str_to_enum = StringToEnum()
