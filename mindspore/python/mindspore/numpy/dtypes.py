# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""Dtypes and utilities"""
from __future__ import absolute_import

from mindspore.common.dtype import (int8, int16, int32, int64, uint8, uint16, uint32, uint64,
                                    float16, float32, float64, bool_)

# original numpy has int->int64, float->float64, uint->uint64 mapping. we map
# them to 32 bit, since 64 bit calculation is not supported from mindspore
# backend for now.

inf = float('inf')
PINF = float('inf')
NINF = float('-inf')
nan = float('nan')
# all three of inf, PINF, and NINF are defined in the original numpy, and as we aim for
# consistency same thing is done here
pi = 3.141592653589793

int_ = int32
uint = uint32
float_ = float32

numeric_types = [
    'int_',
    'int8',
    'int16',
    'int32',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'float_',
    'float16',
    'float32',
    'float64',
    'bool_']

dtype_tuple = (
    int_,
    int8,
    int16,
    int32,
    int64,
    uint,
    uint8,
    uint16,
    uint32,
    uint64,
    float_,
    float16,
    float32,
    float64,
    bool_)

dtype_map = {
    'int': int_,
    'int8': int8,
    'int16': int16,
    'int32': int32,
    'int64': int64,
    'uint': uint,
    'uint8': uint8,
    'uint16': uint16,
    'uint32': uint32,
    'uint64': uint64,
    'float': float_,
    'float16': float16,
    'float32': float32,
    'float64': float64,
    'bool': bool_
}

all_types = [
    'np.int',
    'np.int8',
    'np.int16',
    'np.int32',
    'np.int64',
    'np.uint',
    'np.uint8',
    'np.uint16',
    'np.uint32',
    'np.uint64',
    'np.float',
    'np.float16',
    'np.float32',
    'np.float64',
    'np.bool']

promotion_rule = {
    (uint8, uint16): uint16,
    (uint8, uint32): uint32,
    (uint8, uint64): uint64,
    (uint16, uint32): uint32,
    (uint16, uint64): uint64,
    (uint32, uint64): uint64,
    (uint8, int8): int16,
    (uint8, int16): int16,
    (uint8, int32): int32,
    (uint8, int64): int64,
    (uint16, int8): int32,
    (uint16, int16): int32,
    (uint16, int32): int32,
    (uint16, int64): int64,
    (uint32, int8): int64,
    (uint32, int16): int64,
    (uint32, int32): int64,
    (uint32, int64): int64,
    (uint64, int8): float64,
    (uint64, int16): float64,
    (uint64, int32): float64,
    (uint64, int64): float64,
    (uint8, float16): float16,
    (uint8, float32): float32,
    (uint8, float64): float64,
    (uint16, float16): float16,
    (uint16, float32): float32,
    (uint16, float64): float32,
    (uint32, float16): float16,
    (uint32, float32): float32,
    (uint32, float64): float64,
    (uint64, float16): float16,
    (uint64, float32): float32,
    (uint64, float64): float64,
    (int8, int16): int16,
    (int8, int32): int32,
    (int8, int64): int64,
    (int16, int32): int32,
    (int16, int64): int64,
    (int32, int64): int64,
    (int8, float16): float16,
    (int8, float32): float32,
    (int8, float64): float64,
    (int16, float16): float16,
    (int16, float32): float32,
    (int16, float64): float64,
    (int32, float16): float16,
    (int32, float32): float32,
    (int32, float64): float64,
    (int64, float16): float16,
    (int64, float32): float32,
    (int64, float64): float64,
    (float16, float32): float32,
    (float16, float64): float64,
    (float32, float64): float64,
    (bool_, uint8): uint8,
    (bool_, uint16): uint16,
    (bool_, uint32): uint32,
    (bool_, uint64): uint64,
    (bool_, int8): int8,
    (bool_, int16): int16,
    (bool_, int32): int32,
    (bool_, int64): int64,
    (bool_, float16): float16,
    (bool_, float32): float32,
    (bool_, float64): float64,
}

rule_for_trigonometric = {float16: float16,
                          float32: float32,
                          float64: float64,
                          int8: float16,
                          int16: float32,
                          int32: float32,
                          int64: float32,
                          uint8: float16,
                          uint16: float32,
                          uint32: float32,
                          uint64: float32,
                          bool_: float16}
