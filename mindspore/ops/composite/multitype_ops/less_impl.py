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

"""Implementation for internal polymorphism `less` operations."""
from mindspore.ops.composite import base
from mindspore.ops import functional as F

# less is a metafuncgraph object which will determine if two objects are less according to input type
# using ".register" decorator
less = base.MultitypeFuncGraph("less", True)


@less.register("Number", "Number")
def _less_scala(x, y):
    """
    Determine whether two numbers are less.

    Args:
       x(Number): Number.
       y(Number): Number.

    Returns:
       bool, if x < y return true, x >= y return false.
   """
    return F.scalar_lt(x, y)

@less.register("Tensor", "Number")
@less.register("Number", "Tensor")
@less.register("Tensor", "Tensor")
def _less_tensor(x, y):
    """
    Determine whether two tensor are less by element.

    Args:
       x(Tensor): Tensor.
       y(Tensor): Tensor.

    Returns:
       Tensor, return value of  x and y by operation P.Less()
   """
    return F.tensor_lt(x, y)
