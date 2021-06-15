# Copyright 2021 Huawei Technologies Co., Ltd
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
import math
import numpy as np
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype


def _average_units(shape):
    if not shape:
        return 1
    if len(shape) == 1:
        return float(shape[0])
    if len(shape) == 2:
        return float(shape[0] + shape[1]) / 2.
    raise RuntimeError("not support shape.")


def weight_variable(shape):
    scale_shape = shape
    avg_units = _average_units(scale_shape)
    scale = 1.0 / max(1., avg_units)
    limit = math.sqrt(3.0 * scale)
    values = np.random.uniform(-limit, limit, shape).astype(np.float32)
    return Tensor(values)


def one_weight(shape, dtype=mstype.float32):
    ones = np.ones(shape).astype(np.float32)
    return Tensor(ones, dtype=dtype)


def zero_weight(shape, dtype=mstype.float32):
    zeros = np.zeros(shape).astype(np.float32)
    return Tensor(zeros, dtype=dtype)


def zero_weight_fp32(shape, dtype=mstype.float32):
    zeros = np.zeros(shape).astype(np.float32)
    return Tensor(zeros, dtype=dtype)


def normal_weight(shape, num_units, dtype=mstype.float32):
    norm = np.random.normal(0.0, num_units ** -0.5, shape).astype(np.float32)
    return Tensor(norm, dtype=dtype)


def normal_weightfp32(shape, num_units, dtype=mstype.float32):
    norm = np.random.normal(0.0, num_units ** -0.5, shape).astype(np.float32)
    return Tensor(norm, dtype=dtype)
