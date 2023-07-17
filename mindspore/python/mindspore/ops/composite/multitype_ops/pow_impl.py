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

from __future__ import absolute_import
from mindspore.ops.composite.multitype_ops import _compile_utils as utils
from mindspore.ops.composite import base
from mindspore.ops import functional as F


pow_ = base.MultitypeFuncGraph("pow", True)
"""
`pow` is a metafuncgraph object which will compute the pow of two objects
using ".register" decorator.
"""
pow_.set_need_raise()


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


@pow_.register("Tuple", "Tensor")
def _tuple_pow_tensor(x, y):
    """Returns x ** y where x is a tuple and y is a tensor. """
    x = utils.sequence_to_tensor(x, y.dtype)
    return F.tensor_pow(x, y)


@pow_.register("Tensor", "Tuple")
def _tensor_pow_tuple(x, y):
    """Returns x ** y where x is a tensor and y is a tuple. """
    y = utils.sequence_to_tensor(y, x.dtype)
    return F.tensor_pow(x, y)


@pow_.register("List", "Tensor")
def _list_pow_tensor(x, y):
    """Returns x ** y where x is a list and y is a tensor. """
    x = utils.sequence_to_tensor(x, y.dtype)
    return F.tensor_pow(x, y)


@pow_.register("Tensor", "List")
def _tensor_pow_list(x, y):
    """Returns x ** y where x is a tensor and y is a list. """
    y = utils.sequence_to_tensor(y, x.dtype)
    return F.tensor_pow(x, y)
