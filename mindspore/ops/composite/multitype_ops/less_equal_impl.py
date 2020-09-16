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

"""Implementation for internal polymorphism `less_equal` operations."""
from mindspore.ops.composite import base
from mindspore.ops import functional as F

# less_equal is a metagraph object which will determine if two objects are less_equal according to input type
# using ".register" decorator
less_equal = base.MultitypeFuncGraph("less_equal", True)


@less_equal.register("Number", "Number")
def _less_equal_scala(x, y):
    """
    Determine whether x is less equal than y.

    Args:
       x(Number): Number.
       y(Number): Number.

    Returns:
       bool, if x <= y return true, x > y return false.
   """
    return F.scalar_le(x, y)

@less_equal.register("Tensor", "Number")
@less_equal.register("Number", "Tensor")
@less_equal.register("Tensor", "Tensor")
def _less_equal_tensor(x, y):
    """
    Determine  whether tensor x is less equal than tensor y elementwise.

    Args:
       x(Tensor): Tensor.
       y(Tensor): Tensor.

    Returns:
       Tensor, return value by operator P.LessEqual.
   """
    return  F.tensor_le(x, y)
