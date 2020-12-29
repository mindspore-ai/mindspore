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
"""
init weight
"""
import math
import numpy as np

from mindspore.common.tensor import Tensor


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


def one_weight(shape):
    ones = np.ones(shape).astype(np.float32)
    return Tensor(ones)


def zero_weight(shape):
    zeros = np.zeros(shape).astype(np.float32)
    return Tensor(zeros)


def normal_weight(shape, num_units):
    norm = np.random.normal(0.0, num_units ** -0.5, shape).astype(np.float32)
    return Tensor(norm)
