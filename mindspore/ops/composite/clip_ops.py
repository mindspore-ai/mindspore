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

"""Operations for clipping tensors to min/max values."""

from .. import operations as P


def clip_by_value(x, clip_value_min, clip_value_max):
    """
    Clips tensor values to a specified min and max.

    Limits the value of :math:`x` to a range, whose lower limit is 'clip_value_min'
    and upper limit is 'clip_value_max'.

    Note:
        'clip_value_min' needs to be less than or equal to 'clip_value_max'.

    Args:
          x (Tensor): Input data.
          clip_value_min (Tensor): The minimum value.
          clip_value_max (Tensor): The maximum value.

    Returns:
          Tensor, a clipped Tensor.
    """
    min_op = P.Minimum()
    max_op = P.Maximum()
    x_min = min_op(x, clip_value_max)
    x_max = max_op(x_min, clip_value_min)
    return x_max
