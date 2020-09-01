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

"""Implementation for internal polymorphism `sub` operations."""

from ...composite import base
from ... import functional as F


sub = base.MultitypeFuncGraph("sub", True)
"""
`sub` is a metafuncgraph object which will compute the subtraction of two objects
using ".register" decorator.
"""


@sub.register("Number", "Number")
def _sub_scalar(x, y):
    """Returns x - y where x and y are all scalars."""
    return F.scalar_sub(x, y)


@sub.register("Tensor", "Tensor")
def _sub_tensor(x, y):
    """Returns x - y where x and y are all tensors."""
    return F.tensor_sub(x, y)


@sub.register("Number", "Tensor")
def _scalar_sub_tensor(x, y):
    """Returns x - y where x is a scalar and y is a tensor. x and y should have same dtype."""
    return F.tensor_sub(x, y)


@sub.register("Tensor", "Number")
def _tensor_sub_scalar(x, y):
    """Returns x - y where x is a tensor and y is a scalar. x and y should have same dtype."""
    return F.tensor_sub(x, y)
