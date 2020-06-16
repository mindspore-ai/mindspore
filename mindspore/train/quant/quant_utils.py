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
"""quantization utils."""

import numpy as np


def cal_quantization_params(input_min,
                            input_max,
                            num_bits=8,
                            symmetric=False,
                            narrow_range=False):
    r"""
    calculate quantization params for scale and zero point.

    Args:
        input_min (int, list): The dimension of channel or 1.
        input_max (int, list): The dimension of channel or 1.
        num_bits (int): Quantization number bit, support 4 and 8bit. Default: 8.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

    Outputs:
        scale (int, list): quantization param.
        zero point (int, list): quantization param.

    Examples:
        >>> scale, zp = cal_quantization_params([1, 2, 1], [-2, 0, -1], 8, False, False)
    """
    input_max = np.maximum(0.0, input_max)
    input_min = np.minimum(0.0, input_min)

    if input_min.shape != input_max.shape:
        raise ValueError("input min shape should equal to input max.")
    if len(input_min.shape) > 1:
        raise ValueError("input min and max shape should be one dim.")
    if input_min > input_max:
        raise ValueError("input_min min should less than input max.")
    if (input_max == input_min).all():
        # scale = 1.0, zp = 0.0
        return np.ones(input_min.shape), np.zeros(input_min.shape)

    if symmetric:
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
