# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
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

"""Implementation for internal polymorphism `ones_like_leaf` operations."""

from ...composite import base
from ... import functional as F
from ... import operations as P


ones_like_leaf = base.MultitypeFuncGraph('ones_like_leaf', True)
"""
`ones_like_leaf` is a metafuncgraph object which will generate a tensor filled with one according to its input type
using ".register" decorator.
"""


@ones_like_leaf.register("TypeType")
def _ones_like_type_type(x):
    """Returns x because x is a type. This is usually used in backprop progress."""
    return x


@ones_like_leaf.register("Number")
def _ones_like_scalar(x):
    """Returns 1 which has the same dtype as x where x is a scalar."""
    t = F.typeof(x)
    return F.scalar_cast(1.0, t)


@ones_like_leaf.register("Tensor")
def _ones_like_tensor(x):
    """Returns a tensor with the same shape and dtype as x and all elements ars 1."""
    return P.Fill()(P.DType()(x), P.Shape()(x), 1.0)


@ones_like_leaf.register("SparseTensor")
def _ones_like_sparse_tensor(x):
    """Returns a tensor with the same shape and dtype as x and all elements are 1."""
    values_ = F.sparse_tensor_get_values(x)
    values = P.Fill()(P.DType()(values_),
                      P.Shape()(values_),
                      1.0)
    return F.make_sparse_tensor(F.sparse_tensor_get_indices(x), values, F.sparse_tensor_get_dense_shape(x))


ones_like = base.HyperMap(ones_like_leaf)
"""
`ones_like` is a function which can generate a graph of `ones_like` operation according to input tensor dtype.

Example:
    >>> input = Tensor([2, 3], mindspore.int32)
    >>> ones = ones_like(input) # The dtype of ones is mindspore.int32
    >>> input = Tensor([2, 3], mindspore.float16)
    >>> ones = ones_like(input) # The dtype of ones is mindspore.float16
"""
