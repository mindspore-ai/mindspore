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

"""Implementation for internal polymorphism `div` operations."""

from ...composite import base
from ... import functional as F


div = base.MultitypeFuncGraph("div", True)
"""
div is a metafuncgraph object which will div two objects according to input type
using ".register" decorator
"""


@div.register("Number", "Number")
def _div_scalar(x, y):
    """
    Two numbers divide.

    Args:
       x (Number): x
       y (Number): y

    Returns:
       Number, equal to x / y, the type is same as x.
    """
    return F.scalar_div(x, y)


@div.register("Tensor", "Tensor")
def _div_tensor(x, y):
    """
    Two tensors divide by element.

    Args:
        x (Tensor): The first input tensor.
        y (Tensor): The second input tensor.

    Returns:
        Tensor, has the same dtype as x.
    """
    return F.tensor_div(x, y)


@div.register("Number", "Tensor")
def _scalar_div_tensor(x, y):
    """
    Number divided by tensor.

    Args:
        x (Number): x
        y (Tensor): The dtype is same as x.

    Returns:
        Tensor, has the same dtype as x.
    """
    return F.tensor_div(x, y)


@div.register("Tensor", "Number")
def _tensor_div_scalar(x, y):
    """
    Tensor divided by number.

    Args:
        x (Tensor): x
        y (Number): The dtype is same as x.

    Returns:
        Tensor, has the same dtype as x.
    """
    return F.tensor_div(x, y)
