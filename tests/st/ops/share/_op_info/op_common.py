# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Utility helpers for operation testing.

This module provides:
- Common dtype sets used by tests (MindSpore dtypes) and device names.
- Dimension size helpers for generating sample shapes.
- get_default_loss: default numeric tolerances by dtype.
"""
import torch
import numpy as np
import itertools
import mindspore as ms
import math

dtypes_as_torch = (
    ms.bool_, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8,
    ms.float16, ms.float32, ms.float64,
    ms.complex64, ms.complex128,
    ms.bfloat16,
)

dtypes_extra_uint = (
    ms.uint16, ms.uint32, ms.uint64,
)

dtypes_integral = (
    ms.bool_,
    ms.int8, ms.int16, ms.int32, ms.int64,
    ms.uint8, ms.uint16, ms.uint32, ms.uint64,
)

dtype_to_np_dtype = {
                    ms.bool_: np.bool_,
                    ms.int8: np.int8,
                    ms.int16: np.int16,
                    ms.int32: np.int32,
                    ms.int64: np.int64,
                    ms.uint8: np.uint8,
                    ms.float16: np.float16,
                    ms.float32: np.float32,
                    ms.float64: np.float64,
                    ms.bfloat16: np.float32,  # construct in fp32 then to bfloat16 by dtype=
                    ms.complex64: np.complex64,
                    ms.complex128: np.complex128,
                }

dtypes_extra_values = {
    "_uint_small_values": (0, 1, 31, 63, 128, 195, 254),
    "_int_small_values": (0, -1, 1, -63, 63, -127, 127, -128),
    "_float_small_values":  (
            0.0,
            -0.0,
            -1.0,
            1.0,
            -1.25e-1,
            1.25e-1,
            -1e-3,
            1e-3,
            -math.pi / 2,
            math.pi / 2,
            -math.pi + 1e-5,
            math.pi - 1e-5,
            -math.pi,
            math.pi,
            -math.pi - 1e-5,
            math.pi + 1e-5,
        ),
        "_int_large_values": (-1023, 1023, -10922, 10922),
        "_fp16_large_values": (-521, 521, -996.9, 996.9, -13141.5, 13141.5),
        "_fp32_large_values": (-521, 521, -996.9, 996.9, -13141.5, 13141.5, -5062757.7, 5062757.7, -1e20, 1e20)
}
#for opers who has dtype extend-parameter
op_dtype_need_convert ={
    'Tensor.to': {
                  ms.float32: [ms.uint8, ms.int32, ms.float64, ms.complex128],
                  ms.float16: [ms.int8, ms.uint8, ms.float32, ms.bfloat16] ,
                  ms.int32: [ms.int16, ms.float32],
                  ms.int64: [ms.int32, ms.float32, ms.complex64],
                  ms.complex128: ms.int64, ms.uint8: [ms.float32, ms.int32],
                  ms.int16: ms.float64,
                  ms.float64: ms.complex64,
                  ms.int8: ms.complex128,
                  ms.complex64: ms.bool_,
                  ms.bool_: [ms.int16, ms.float32],
                  ms.bfloat16: ms.float32
    },

    'Tensor.sum': {
                  ms.float32: ms.float16,
                  ms.float16: ms.float32,
                  ms.int32: ms.float64,
    },

    'Tensor.mean': {
                  ms.float32: ms.float64,
                  ms.int32: ms.int16,
                  ms.int64: ms.float32,
    },

    'mint.nansum': {
                 ms.float16: ms.float32,
                 ms.int32: ms.int64,
                 ms.int64: ms.int32,
    },
    'mint.mean': {
                  ms.int16:ms.complex128,
                  ms.complex128: ms.complex64,
                  ms.float32:ms.float16,
    },
    'mint.prod':{
                  ms.float64:ms.float32,
                  ms.int32:ms.float64,
                  ms.float32:ms.int16,
    },
    'mint.sum': {
                  ms.int64:ms.float64,
                  ms.complex64:ms.float32,
                  ms.bool:ms.float32,
    },
}

dtypes_common = (
    int, float, complex, bool
)

device_names = [
    'ascend',
    'cpu',
    'gpu',
]

# dim size of tensor shape
LARGE_DIM_SIZE = 20
MEDIUM_DIM_SIZE = 10
SMALL_DIM_SIZE = 5
EXTRA_SMALL_DIM_SIZE = 3

def get_default_loss(dtype):
    """Return default numeric tolerance based on dtype for comparisons.

    Args:
        dtype: MindSpore, NumPy, or PyTorch dtype.

    Returns:
        float: Recommended rtol/atol.
    """
    if dtype in (ms.float16, torch.float16, np.float16):
        return 1e-3
    if dtype in (
            ms.float32, ms.complex64, torch.float32, torch.complex64,
            np.float32, np.complex64):
        return 1e-4
    if dtype in (
            ms.float64, ms.complex128, torch.float64, torch.complex128,
            np.float64, np.complex128, float, complex):
        return 1e-5
    if dtype in (ms.bfloat16, torch.bfloat16):
        return 4e-3
    return 0

def small_large_values(dtype, exclude_values=None):
    values = []
    if dtype == ms.uint8:
        values = dtypes_extra_values['_uint_small_values']
    elif dtype in [ms.int8, ms.int16, ms.int32, ms.int64]:
        values = dtypes_extra_values['_int_small_values'] + dtypes_extra_values['_int_large_values']
    else:
        if dtype in [ms.float16]:
            values = dtypes_extra_values['_float_small_values'] + dtypes_extra_values['_fp16_large_values']
        else:
            float_vals = dtypes_extra_values['_float_small_values'] + dtypes_extra_values['_fp32_large_values']
            if dtype.is_complex:
                values = [complex(r, i) for r, i in itertools.product(float_vals, float_vals)]
            else:
                values = float_vals
    if exclude_values:
        if not isinstance(exclude_values, (tuple, list, set)):
            exclude_values = [exclude_values]
        values = [d for d in values if d not in exclude_values]
    return values
