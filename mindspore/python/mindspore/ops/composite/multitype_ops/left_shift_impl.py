# Copyright 2022 Huawei Technologies Co., Ltd
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

"""Implementation for internal polymorphism `left_shift` operations."""
from __future__ import absolute_import
from mindspore.ops.composite import base
from mindspore.ops.operations import _inner_ops as inner

# left_shift is a metagraph object which will generate function according to input type
# using ".register" decorator
left_shift = base.MultitypeFuncGraph("left_shift", True)

left_shift.set_need_raise()


@left_shift.register("Number", "Number")
def _left_shift_scalar(x, y):
    """Returns x << y where x and y are all scalars."""
    return inner.bit_left_shift(x, y)
