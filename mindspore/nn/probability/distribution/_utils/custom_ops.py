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
"""Utitly functions to help distribution class."""
import numpy as np
from mindspore.ops import operations as P

def log_by_step(input_x):
    """
    Log op on Ascend is calculated as log(abs(x)).
    Fix this with putting negative values as nan.
    """
    select = P.Select()
    log = P.Log()
    lessequal = P.LessEqual()
    fill = P.Fill()
    dtype = P.DType()
    shape = P.Shape()

    nonpos_x = lessequal(input_x, 0.0)
    log_x = log(input_x)
    nan = fill(dtype(input_x), shape(input_x), np.nan)
    result = select(nonpos_x, nan, log_x)
    return result

def log1p_by_step(x):
    """
    Log1p ops on GPU device or when device_target == GPU.
    """
    return log_by_step(x + 1.0)

def expm1_by_step(input_x):
    """
    Expm1 ops under GPU context.
    """
    exp = P.Exp()
    return exp(input_x) - 1.0
