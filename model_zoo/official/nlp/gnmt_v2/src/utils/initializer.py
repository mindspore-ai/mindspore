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
"""Initializer."""
import numpy as np

from mindspore import Tensor


def _compute_fans(shape):
    """
    Computes the number of input and output units for a weight shape.

    Args:
        shape (tuple): Integer shape tuple or MS tensor shape.

    Returns:
        tuple, integer scalars (fan_in, fan_out).
    """
    if not shape:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return int(fan_in), int(fan_out)


def weight_variable(shape):
    """
    Generate weight var.

    Args:
        shape (tuple): Shape.

    Returns:
        Tensor, var.
    """
    limit = 0.1
    values = np.random.uniform(-limit, limit, shape)
    return values


def one_weight(shape):
    """
    Generate weight with ones.

    Args:
        shape (tuple): Shape.

    Returns:
        Tensor, var.
    """
    ones = np.ones(shape).astype(np.float32)
    return Tensor(ones)


def zero_weight(shape):
    """
    Generate weight with zeros.

    Args:
        shape (tuple): Shape.

    Returns:
        Tensor, var.
    """
    zeros = np.zeros(shape).astype(np.float32)
    return Tensor(zeros)


def normal_weight(shape, num_units):
    """
    Generate weight with normal dist.

    Args:
        shape (tuple): Shape.
        num_units (int): Dimension.

    Returns:
        Tensor, var.
    """
    norm = np.random.normal(0.0, num_units ** -0.5, shape).astype(np.float32)
    return Tensor(norm)
