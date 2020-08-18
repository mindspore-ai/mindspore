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
"""Quantization utils."""

import numpy as np


def cal_quantization_params(input_min,
                            input_max,
                            data_type,
                            num_bits=8,
                            symmetric=False,
                            narrow_range=False):
    r"""
    Calculate quantization params for scale and zero point.

    Args:
        input_min (numpy.ndarray): The dimension of channel or 1.
        input_max (numpy.ndarray): The dimension of channel or 1.
        data_type (numpy type) : Can ben numpy int8, numpy uint8.
        num_bits (int): Quantization number bit, support 4 and 8bit. Default: 8.
        symmetric (bool): Whether the quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): Whether the quantization algorithm uses narrow range or not. Default: False.

    Returns:
        scale (numpy.ndarray): quantization param.
        zero point (numpy.ndarray): quantization param.
    """
    input_max = np.maximum(0.0, input_max)
    input_min = np.minimum(0.0, input_min)

    if input_min.shape != input_max.shape:
        raise ValueError("input min shape should equal to input max.")
    if len(input_min.shape) > 1:
        raise ValueError("input min and max shape should be one dim.")
    if (input_min > input_max).all():
        raise ValueError("input_min min should less than input max.")
    if (input_max == input_min).all():
        # scale = 1.0, zp = 0.0
        return np.ones(input_min.shape), np.zeros(input_min.shape)

    if data_type == np.int8:
        quant_min = 0 - 2 ** (num_bits - 1)
        quant_max = 2 ** (num_bits - 1)
    else:
        quant_min = 0
        quant_max = 2 ** num_bits - 1
    if narrow_range:
        quant_min = quant_min + 1

    # calculate scale
    if symmetric:
        input_max = np.maximum(-input_min, input_max)
        input_min = -input_max
    scale = (input_max - input_min) / (quant_max - quant_min)

    # calculate zero point
    if symmetric:
        zp = np.zeros(input_min.shape)
    else:
        zp_from_min = quant_min - input_min / scale
        zp_from_max = quant_max - input_max / scale
        zp_from_min_error = np.abs(quant_min) + np.abs(input_min / scale)
        zp_from_max_error = np.abs(quant_max) + np.abs(input_max / scale)
        zp_double = zp_from_min if zp_from_min_error < zp_from_max_error else zp_from_max
        if zp_double < quant_min:
            zp = quant_min
        elif zp_double > quant_max:
            zp = quant_max
        else:
            zp = np.floor(zp_double + 0.5)

    return scale, zp


def weight2int(data, scale, zero_point):
    r"""
    Calculate int8/uint8 weight from fp32. the formula is defined as:

    .. math::
        int8/uint8 = round(float/scale) + offset

    Args:
        data (numpy.ndarray): The dimension of channel or 1. Should be NCHW.
        scale (numpy.ndarray): The dimension of channel or 1.
        zero_point (numpy.ndarray): The dimension of channel or 1.

    Returns:
        weight (numpy.ndarray): The dimension of channel or 1.
    """
    if scale.shape != zero_point.shape:
        raise ValueError("`scale` and `zero_point` should have the same shape.")
    if scale.shape[0] < 0:
        raise ValueError("`scale` and `zero_point` shape should greater than zero.")
    if len(scale.shape) >= 1 and scale.shape[0] > 1:
        # for perchannel
        if scale.shape[0] == data.shape[0]:
            # `Conv2d` or `Dense` op weight
            shape_list = [-1] + [1] * len(data.shape[1:])
            scale = scale.reshape(shape_list)
            zero_point = zero_point.reshape(shape_list)
        elif scale.shape[0] == data.shape[1]:
            # `DepthwiseConv2d` op weight
            shape_list = [1, -1] + [1] * len(data.shape[2:])
            scale = scale.reshape(shape_list)
            zero_point = zero_point.reshape(shape_list)
        else:
            raise ValueError("Unsupported weight shape({})".format(data.shape))

    return np.round((data / scale) + zero_point)


def scale_zp_from_fack_quant_cell(cell, data_type):
    r"""
    Get calculate quantization params for scale and zero point From `FakeQuantWithMinMax`.

    Args:
        cell (Cell): `mindspore.nn.layer.FakeQuantWithMinMax`
        data_type (numpy type): Can ben `numpy.int8` or `numpy.uint8`.

    Returns:
        scale (numpy.ndarray): quantization param.
        zero point (numpy.ndarray): quantization param.
    """
    minq = cell.minq.data.asnumpy()
    maxq = cell.maxq.data.asnumpy()
    op = cell.fake_quant_infer

    scale, zp = cal_quantization_params(
        minq, maxq, data_type,
        num_bits=op.num_bits,
        symmetric=op.symmetric,
        narrow_range=op.narrow_range)
    return scale, zp


def scale_zp_from_data(op, minq, maxq, data_type):
    r"""
    Get calculate quantization params for scale and zero point.

    Calculate from `FakeQuantWithMinMax`'s Parameter or Fake quant primitive.

    Args:
        op (Primitive): Fake quant primitive `mindspore.ops.operation.FakeQuantPerLayer` or
            `mindspore.ops.operation.FakeQuantPerChannel`
        minq (Parameter): Parameter `minq` of `mindspore.nn.layer.FakeQuantWithMinMax`
        maxq (Parameter): Parameter `maxq` of `mindspore.nn.layer.FakeQuantWithMinMax`
        data_type (numpy type): Can ben `numpy.int8` or `numpy.uint8`.

    Returns:
        scale (numpy.ndarray): quantization param.
        zero point (numpy.ndarray): quantization param.
    """
    minq = minq.data.asnumpy()
    maxq = maxq.data.asnumpy()

    scale, zp = cal_quantization_params(
        minq, maxq, data_type,
        num_bits=op.num_bits,
        symmetric=op.symmetric,
        narrow_range=op.narrow_range)
    return scale, zp


def fold_batchnorm(weight, cell_quant):
    r"""
    Fold the batchnorm in `Conv2dBnFoldQuant` to weight.

    Calculate from `FakeQuantWithMinMax`'s Parameter or Fake quant primitive.

    Args:
        weight (numpy.ndarray): Weight of `cell_quant`.
        cell_quant (Cell): Object of `mindspore.nn.layer.Conv2dBnFoldQuant`.

    Returns:
        weight (numpy.ndarray): Folded weight.
        bias (numpy.ndarray): Folded bias.
    """
    variance = cell_quant.moving_variance.data.asnumpy()
    mean = cell_quant.moving_mean.data.asnumpy()
    gamma = cell_quant.gamma.data.asnumpy()
    beta = cell_quant.beta.data.asnumpy()
    epsilon = cell_quant.eps
    sigma = np.sqrt(variance + epsilon)

    if gamma.shape[0] == weight.shape[0]:
        # `Conv2d` or `Dense` op weight
        shape_list = [-1] + [1] * len(weight.shape[1:])
        _gamma = gamma.reshape(shape_list)
        _sigma = sigma.reshape(shape_list)
    elif gamma.shape[0] == weight.shape[1]:
        # `DepthwiseConv2d` op weight
        shape_list = [1, -1] + [1] * len(weight.shape[2:])
        _gamma = gamma.reshape(shape_list)
        _sigma = sigma.reshape(shape_list)
    else:
        raise ValueError("Unsupported weight shape({})".format(weight.shape))

    weight = weight * _gamma / _sigma
    bias = beta - gamma * mean / sigma
    return weight, bias


def without_fold_batchnorm(weight, cell_quant):
    r"""
    Fold the batchnorm in `Conv2dBnWithoutFoldQuant` to weight.

    Calculate from `FakeQuantWithMinMax`'s Parameter or Fake quant primitive.

    Args:
        weight (numpy.ndarray): Weight of `cell_quant`.
        cell_quant (Cell): Object of `mindspore.nn.layer.Conv2dBnWithoutFoldQuant`.

    Returns:
        weight (numpy.ndarray): whihout folded weight.
        bias (numpy.ndarray): without folded bias.
    """
    variance = cell_quant.batchnorm.moving_variance.data.asnumpy()
    mean = cell_quant.batchnorm.moving_mean.data.asnumpy()
    gamma = cell_quant.batchnorm.gamma.data.asnumpy()
    beta = cell_quant.batchnorm.beta.data.asnumpy()
    epsilon = cell_quant.batchnorm.eps
    sigma = np.sqrt(variance + epsilon)

    if gamma.shape[0] == weight.shape[0]:
        # `Conv2d` or `Dense` op weight
        shape_list = [-1] + [1] * len(weight.shape[1:])
        _gamma = gamma.reshape(shape_list)
        _sigma = sigma.reshape(shape_list)
    elif gamma.shape[0] == weight.shape[1]:
        # `DepthwiseConv2d` op weight
        shape_list = [1, -1] + [1] * len(weight.shape[2:])
        _gamma = gamma.reshape(shape_list)
        _sigma = sigma.reshape(shape_list)
    else:
        raise ValueError("Unsupported weight shape({})".format(weight.shape))

    weight = weight * _gamma / _sigma
    bias = beta - gamma * mean / sigma
    return weight, bias
