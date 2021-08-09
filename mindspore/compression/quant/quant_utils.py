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
from mindspore._checkparam import Validator
from ... import nn

__all__ = ["load_nonquant_param_into_quant_net", "query_quant_layers"]


def cal_quantization_params(input_min,
                            input_max,
                            quant_min,
                            quant_max,
                            data_type,
                            symmetric=False):
    r"""
    Calculate quantization params for scale and zero point.

    Args:
        input_min (numpy.ndarray): The dimension of channel or 1.
        input_max (numpy.ndarray): The dimension of channel or 1.
        quant_min (int): The minimum quantization integer.
        quant_max (int): The maximum quantization integer.
        data_type (numpy type) : Can be numpy int8, numpy uint8.
        symmetric (bool): Whether the quantization algorithm is symmetric or not. Default: False.

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
        return np.ones(input_min.shape), np.zeros(input_min.shape)

    # calculate scale
    if symmetric:
        input_max = np.maximum(-input_min, input_max)
        input_min = -input_max
    scale = (input_max - input_min) / (quant_max - quant_min)

    # calculate zero point
    if data_type == np.int8 and symmetric:
        zp = np.zeros(input_min.shape)
    else:
        zp_double = quant_min - input_min / scale
        zp = np.floor(zp_double + 0.5)

    return scale, zp


def get_quant_min_max(data_type, num_bits=8, narrow_range=False):
    """Calculate quantization params for minimum/maximum quantization integer"""
    if data_type == np.int8:
        quant_min = 0 - 2 ** (num_bits - 1)
        quant_max = 2 ** (num_bits - 1) - 1
    elif data_type == np.uint8:
        quant_min = 0
        quant_max = 2 ** num_bits - 1
    else:
        raise ValueError("Unsupported datatype({})".format(data_type))
    if narrow_range:
        quant_min = quant_min + 1
    return quant_min, quant_max


def weight2int(data, scale, zero_point, quant_min, quant_max):
    r"""
    Calculate int8/uint8 weight from fp32. the formula is defined as:

    .. math::
        int8/uint8 = round(float/scale) + offset

    Args:
        data (numpy.ndarray): The dimension of channel or 1. Should be NCHW.
        scale (numpy.ndarray): The dimension of channel or 1.
        zero_point (numpy.ndarray): The dimension of channel or 1.
        quant_min (int): The minimum quantization integer.
        quant_max (int): The maximum quantization integer.

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

    weight_int = np.round((data / scale) + zero_point)
    weight_int[weight_int > quant_max] = quant_max
    weight_int[weight_int < quant_min] = quant_min
    return weight_int


def scale_zp_max_min_from_fake_quant_cell(cell, data_type):
    """Get calculate quantization params for scale, zero point, max and min from `FakeQuantWithMinMaxObserver`."""
    minq = cell.minq.data.asnumpy()
    maxq = cell.maxq.data.asnumpy()
    # make sure maxq > 0 and minq <= 0
    if cell.mode == 'LEARNED_SCALE':
        maxq = np.abs(maxq)
        minq = -np.abs(minq)
    quant_min, quant_max = get_quant_min_max(data_type, num_bits=cell.num_bits, narrow_range=cell.narrow_range)
    symmetric = cell.symmetric and not cell.neg_trunc
    scale, zp = cal_quantization_params(
        minq, maxq,
        quant_min, quant_max, data_type,
        symmetric=symmetric)
    return scale, zp, maxq, minq


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


def compute_kl_threshold(data, bitwidth):
    r"""
    Using KL-J Distance to calculate the clip threshold.

    Args:
        - **data** (NumpyArray) - Data observed to calculate the threshold for quantization,
        - **bitwidth** (QuantDtype) - The datatype of quantization.
    Outputs:
        Tensor with Shape 1. Threshold to calculate the data.
    """
    data_max = np.abs(data).max()
    if data_max < 1e-5:
        return 1e-5
    hist, bin_edges = np.histogram(np.abs(data), bins='sqrt', range=(0, data_max), density=True)
    # For the sake of high efficiency, we limit the maximum number of bins to 1024 in `sqrt` mode, If it exceeds the
    # largest size, turn to use the default bins config.
    largest_bin_size = 1024
    if hist.shape[0] > largest_bin_size:
        hist, bin_edges = np.histogram(np.abs(data), range=(0, data_max), density=True)
    hist = hist / np.sum(hist)
    cumsum = np.cumsum(hist)
    bit_pow_range = pow(2, int(bitwidth.num_bits) - 1)
    threshold = []
    scaling_factor = []
    kl = []
    if bit_pow_range + 1 > len(bin_edges) - 1:
        th_layer_out = bin_edges[-1]
        return float(th_layer_out)
    for i in range(bit_pow_range + 1, len(bin_edges), 1):
        threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
        threshold = np.concatenate((threshold, [threshold_tmp]))
        scaling_factor_tmp = threshold_tmp / (bit_pow_range - 1)
        scaling_factor = np.concatenate((scaling_factor, [scaling_factor_tmp]))
        # forward interpolation
        cumsum_tmp = np.copy(cumsum)
        cumsum_tmp[(i - 1):] = 1
        fwd_x = np.linspace(0.0, 1.0, bit_pow_range)
        fwd_xp = np.linspace(0.0, 1.0, i)
        fwd_fp = cumsum_tmp[:i]
        forward_interp = np.interp(fwd_x, fwd_xp, fwd_fp)
        # backward interpolation
        bwd_x = np.linspace(0.0, 1.0, i)
        bwd_xp = np.linspace(0.0, 1.0, bit_pow_range)
        bwd_fp = forward_interp
        backward_interp = np.interp(bwd_x, bwd_xp, bwd_fp)
        cumsum_tmp[:i] = backward_interp
        kl_tmp = np.sum((cumsum - cumsum_tmp) * np.log2(cumsum / cumsum_tmp))  # Kullback-Leibler-J
        kl = np.concatenate((kl, [kl_tmp]))
    th_layer_out = threshold[np.argmin(kl)]
    threshold = float(th_layer_out)
    if threshold < 1e-5:
        threshold = 1e-5
    return threshold


def query_quant_layers(network):
    r"""
    Query the network's quantization strategy of each quantized layer and print it to the screen, note that all the
    quantization layers are queried before graph compile optimization in the graph mode, thus, some redundant quantized
    layers, which not exist in practical execution, may appear.

    Args:
        network (Cell): input network
    """
    network = Validator.check_isinstance("network", network, nn.Cell)
    tplt = "{0:60}\t{1:10}"
    for cell_and_name in network.cells_and_names():
        cell_name = cell_and_name[0]
        cell = cell_and_name[1]
        if isinstance(cell, nn.FakeQuantWithMinMaxObserver):
            print(tplt.format(cell_name, cell.quant_dtype))


def load_nonquant_param_into_quant_net(quant_model, params_dict, quant_new_params=None):
    r"""
    Load fp32 model parameters into quantization model.

    Args:
        quant_model(Cell): Quantization model.
        params_dict(dict): Parameter dict that stores fp32 parameters.
        quant_new_params(list): Parameters that exist in quantization network but not in non-quantization
            network. Default: None.

    Raises:
        TypeError: If `quant_new_params` is not None and is not list.
        ValueError: If there are parameters in the `quant_model` that are neither in `params_dict`
            nor in `quant_new_params`.
    """
    if quant_new_params is not None and not isinstance(quant_new_params, list):
        raise TypeError("quant_new_params must be list or None.")
    iterable_dict = {
        'minq': iter(list(filter(lambda item: item[0].endswith('minq'), params_dict.items()))),
        'maxq': iter(list(filter(lambda item: item[0].endswith('maxq'), params_dict.items()))),
        'quant_max': iter(list(filter(lambda item: item[0].endswith('quant_max'), params_dict.items())))
    }
    for param in params_dict.items():
        key_name = param[0].split(".")[-1]
        if key_name not in iterable_dict:
            iterable_dict[key_name] = iter(list(filter(lambda item, value=key_name: item[0].endswith(value),
                                                       params_dict.items())))

    for name, param in quant_model.parameters_and_names():
        key_name = name.split(".")[-1]
        if key_name not in iterable_dict.keys():
            if key_name not in quant_new_params:
                raise ValueError(f"Can't find match parameter in ckpt, param name = {name}")
            continue
        value_param = next(iterable_dict[key_name], None)
        if value_param:
            param.set_data(value_param[1].data)
            print(f'init model param {name} with checkpoint param {value_param[0]}')


    # Perform KL_init when learned scale quantization is executed.
    for cell_and_name in quant_model.cells_and_names():
        cell = cell_and_name[1]
        if isinstance(cell, (nn.Conv2dBnFoldQuantOneConv, nn.Conv2dBnFoldQuant, nn.Conv2dBnWithoutFoldQuant,
                             nn.Conv2dQuant, nn.DenseQuant)) and cell.fake_quant_weight.mode == "LEARNED_SCALE":
            subcell_weight_para = cell.weight.data.asnumpy()
            if hasattr(cell, 'gamma'):
                scale_factor = (cell.gamma.data.asnumpy() /
                                np.sqrt(cell.moving_variance.data.asnumpy() + 1e-5))
                subcell_weight_para = subcell_weight_para * scale_factor.reshape(-1, 1, 1, 1)

            if cell.fake_quant_weight.per_channel:
                max_init = [compute_kl_threshold(weight_para_each, cell.fake_quant_weight.quant_dtype)
                            for weight_para_each in subcell_weight_para]
                min_init = [-x for x in max_init]
            else:
                max_init = [compute_kl_threshold(subcell_weight_para, cell.fake_quant_weight.quant_dtype)]
                min_init = [-x for x in max_init]

            cell.fake_quant_weight.reset(quant_dtype=cell.fake_quant_weight.quant_dtype,
                                         min_init=min_init, max_init=max_init)
