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
import mindspore as ms

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
