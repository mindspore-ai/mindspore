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

"""Implementation for internal polymorphism `floordiv` operations."""
from __future__ import absolute_import
from mindspore.ops.composite.multitype_ops import _compile_utils as utils
from mindspore.ops.composite import base
from mindspore.ops import functional as F


floordiv = base.MultitypeFuncGraph("floordiv", True)
"""
`floordiv` is a metafuncgraph object which will compute the floordiv of two objects
using ".register" decorator.
"""
floordiv.set_need_raise()


@floordiv.register("Number", "Number")
def _floordiv_scalar(x, y):
    """Returns x // y where x and y are all scalars."""
    return F.scalar_floordiv(x, y)


@floordiv.register("Tensor", "Tensor")
def _floordiv_tensor(x, y):
    """Returns x // y where x and y are all tensors."""
    return F.tensor_floordiv(x, y)


@floordiv.register("Tensor", "Number")
def _tensor_floordiv_scalar(x, y):
    """Returns x // y where x is a tensor and y is a scalar. x and y should have same dtype."""
    return F.tensor_floordiv(x, y)


@floordiv.register("Number", "Tensor")
def _scalar_floordiv_tensor(x, y):
    """Returns x // y where x is a scalar and y is a tensor. x and y should have same dtype."""
    return F.tensor_floordiv(x, y)


@floordiv.register("Tuple", "Tensor")
def _tuple_floordiv_tensor(x, y):
    """Returns x // y where x is a tuple and y is a tensor. """
    x = utils.sequence_to_tensor(x, y.dtype)
    return F.tensor_floordiv(x, y)


@floordiv.register("Tensor", "Tuple")
def _tensor_floordiv_tuple(x, y):
    """Returns x // y where x is a tensor and y is a tuple. """
    y = utils.sequence_to_tensor(y, x.dtype)
    return F.tensor_floordiv(x, y)


@floordiv.register("List", "Tensor")
def _list_floordiv_tensor(x, y):
    """Returns x // y where x is a list and y is a tensor. """
    x = utils.sequence_to_tensor(x, y.dtype)
    return F.tensor_floordiv(x, y)


@floordiv.register("Tensor", "List")
def _tensor_floordiv_list(x, y):
    """Returns x // y where x is a tensor and y is a list. """
    y = utils.sequence_to_tensor(y, x.dtype)
    return F.tensor_floordiv(x, y)
