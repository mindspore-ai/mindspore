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
"""Utility functions to help distribution class."""
import numpy as np
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

def exp_generic(input_x):
    """
    Log op on Ascend doesn't support int types.
    Fix this with casting the type.
    """
    exp = P.Exp()
    cast = P.Cast()
    dtype = P.DType()
    checktype = P.IsSubClass()

    if not checktype(dtype(input_x), mstype.float_):
        input_x = cast(input_x, mstype.float32)
    return exp(input_x)


def log_generic(input_x):
    """
    Log op on Ascend is calculated as log(abs(x)).
    Fix this with putting negative values as nan.
    And log op on Ascend doesn't support int types.
    Fix this with casting the type.
    """
    log = P.Log()
    less = P.Less()
    lessequal = P.LessEqual()
    fill = P.Fill()
    cast = P.Cast()
    dtype = P.DType()
    shape = P.Shape()
    select = P.Select()
    checktype = P.IsSubClass()

    if not checktype(dtype(input_x), mstype.float_):
        input_x = cast(input_x, mstype.float32)
    nan = fill(dtype(input_x), shape(input_x), np.nan)
    inf = fill(dtype(input_x), shape(input_x), np.inf)
    neg_x = less(input_x, 0.0)
    nonpos_x = lessequal(input_x, 0.0)
    log_x = log(input_x)
    result = select(nonpos_x, -inf, log_x)
    return select(neg_x, nan, result)


def log1p_generic(x):
    """
    Log1p ops on GPU device or when device_target == GPU.
    """
    return log_generic(x + 1.0)

def broadcast_to(x, target):
    """
    Broadcast x to the shape of target.
    """
    shape = P.Shape()
    if shape(x) == shape(target):
        return x
    return P.BroadcastTo(shape(target))(x)
