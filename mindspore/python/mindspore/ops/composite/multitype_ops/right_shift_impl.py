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

"""Implementation for internal polymorphism `right_shift` operations."""
from __future__ import absolute_import
from mindspore.ops.composite import base
from mindspore.ops.operations import _inner_ops as inner

# right_shift is a metagraph object which will generate function according to input type
# using ".register" decorator
right_shift = base.MultitypeFuncGraph("right_shift", True)

right_shift.set_need_raise()


@right_shift.register("Number", "Number")
def _right_shift_scalar(x, y):
    """Returns x >> y where x and y are all scalars."""
    return inner.bit_right_shift(x, y)
