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

"""Implementation for internal polymorphism `mul` operations."""

from ...composite import base
from ... import functional as F


mul = base.MultitypeFuncGraph("mul", True)
"""
`mul` is a metafuncgraph object which will multiply two objects according to input type
using ".register" decorator.
"""


@mul.register("Number", "Number")
def _mul_scalar(x, y):
    """
    Returns x * y where x and y are all scalars.

    Outputs:
       Number, equal to x * y, the type is same as x.
   """
    return F.scalar_mul(x, y)


@mul.register("Tensor", "Tensor")
def _mul_tensor(x, y):
    """
    Returns x * y by element-wise where x and y are all tensors.

    Outputs:
        Tensor, has the same dtype as x.
    """
    return F.tensor_mul(x, y)


@mul.register("Number", "Tensor")
def _scalar_mul_tensor(x, y):
    """
    Returns x * y where x is a scalar and y is a tensor. x and y have same dtype.

    Outputs:
       Tensor, has the same dtype as x.
    """
    return F.tensor_mul(x, y)


@mul.register("Tensor", "Number")
def _tensor_mul_scalar(x, y):
    """
    Returns x * y where x is a tensor and y is a scalar. x and y have same dtype.

    Outputs:
        Tensor, has the same dtype as x.
    """
    return F.tensor_mul(x, y)
