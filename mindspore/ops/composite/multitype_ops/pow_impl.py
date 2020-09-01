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

"""Implementation for internal polymorphism `pow` operations."""

from ...composite import base
from ... import functional as F


pow_ = base.MultitypeFuncGraph("pow", True)
"""
`pow` is a metafuncgraph object which will compute the pow of two objects
using ".register" decorator.
"""


@pow_.register("Number", "Number")
def _pow_scalar(x, y):
    """Returns x ** y where x and y are all scalars."""
    return F.scalar_pow(x, y)


@pow_.register("Tensor", "Tensor")
def _pow_tensor(x, y):
    """Returns x ** y where x and y are all tensors and have save dtype."""
    return F.tensor_pow(x, y)


@pow_.register("Tensor", "Number")
def _tensor_pow_scalar(x, y):
    """Returns x ** y where x is a tensor and y is a scalar. x and y should have same dtype."""
    return F.tensor_pow(x, y)


@pow_.register("Number", "Tensor")
def _scalar_pow_tensor(x, y):
    """Returns x ** y where x is a scalar and y is a tensor. x and y should have same dtype."""
    return F.tensor_pow(x, y)
