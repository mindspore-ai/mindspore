# Copyright 2019 Huawei Technologies Co., Ltd
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

"""operator dsl function: mean"""
import _akg.topi
import _akg.tvm
from _akg.utils import format_transform as ft_util
from _akg.utils import validation_check as vc_util
from _akg.ops.math import sum


@vc_util.check_input_type(_akg.tvm.tensor.Tensor, (list, tuple, int, type(None)), (bool, type(None)))
def mean(data, axis=None, keepdims=False):
    """
    Computes the mean of the values of a Tensor over the whole dataset.

    Args:
        data (tvm.tensor.Tensor): Tensor.
        axis (Union[list, tuple, int, None]): If the tuple is empty, the axis equal to None.
        keepdims (bool): If keepdims equal to True, the result shape length is same to input shape length.

    Returns:
            tvm.tensor.Tensor, has the same type as data. If keepdims equal to True, all reduced dimensions are
            retained with length 1. else these reduced axis will be eliminate.
    """
    shape = [x.value for x in data.shape]
    vc_util.reduce_axis_check(shape, axis)
    axis = ft_util.refine_reduce_axis(data, axis)

    count = 1
    for i in axis:
        count *= shape[i]
    output, _ = sum.sum_value(data, axis, keepdims)
    res = _akg.topi.divide(output, count)

    return res
