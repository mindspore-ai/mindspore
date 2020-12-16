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
"""Dtypes and utilities"""

from ..common.dtype import (int8, int16, int32, int64, uint8, uint16, uint32, uint64, \
    float16, float32, float64, bool_)

# original numpy has int->int64, float->float64, uint->uint64 mapping. we map
# them to 32 bit, since 64 bit calculation is not supported from mindspore
# backend for now.

inf = float('inf')

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
