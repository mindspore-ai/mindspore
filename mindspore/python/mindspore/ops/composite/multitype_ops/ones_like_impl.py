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

"""Implementation for internal polymorphism `ones_like` operations."""

from __future__ import absolute_import
from mindspore.ops.composite import base
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.function.sparse_func import (
    _coo_tensor_get_dense_shape,
    _coo_tensor_get_indices,
    _coo_tensor_get_values,
    _make_coo_tensor,
    _make_csr_tensor,
)


ones_like_leaf = base.MultitypeFuncGraph('ones_like', True)
"""
`ones_like_leaf` is a metafuncgraph object which will generate a tensor filled with one according to its input type
using ".register" decorator.
"""

_ones_like_leaf_for_grad = base.MultitypeFuncGraph('_ones_like_for_grad', True)


@_ones_like_leaf_for_grad.register("TypeType")
@ones_like_leaf.register("TypeType")
def _ones_like_type_type(x):
    """Returns x because x is a type. This is usually used in backprop progress."""
    return x


@_ones_like_leaf_for_grad.register("Number")
@ones_like_leaf.register("Number")
def _ones_like_scalar(x):
    """Returns 1 which has the same dtype as x where x is a scalar."""
    t = F.typeof(x)
    return F.scalar_cast(1.0, t)


@_ones_like_leaf_for_grad.register("Tensor")
@ones_like_leaf.register("Tensor")
def _ones_like_tensor(x):
    """Returns a tensor with the same shape and dtype as x and all elements are 1."""
    return P.OnesLike()(x)


@_ones_like_leaf_for_grad.register("COOTensor")
@ones_like_leaf.register("COOTensor")
def _ones_like_coo_tensor(x):
    """Returns a tensor with the same shape and dtype as x and all elements are 1."""
    values_ = _coo_tensor_get_values(x)
    values = P.OnesLike()(values_)
    return _make_coo_tensor(_coo_tensor_get_indices(x), values, _coo_tensor_get_dense_shape(x))


@_ones_like_leaf_for_grad.register("CSRTensor")
@ones_like_leaf.register("CSRTensor")
def _ones_like_csr_tensor(x):
    """Returns a tensor with the same shape and dtype as x and all elements are 1."""
    return _make_csr_tensor(x.indptr, x.indices, ones_like(x.values), x.shape)


@_ones_like_leaf_for_grad.register("Function")
@ones_like_leaf.register("Function")
def _ones_like_func(x):
    """
    Derivation of a function.

    Args:
        x (Function): x

    Returns:
        A instance of EnvType.
    """
    # Unused parameters are placeholders.
    return F.environ_create()


@_ones_like_leaf_for_grad.register("None")
@ones_like_leaf.register("None")
def _ones_like_none(x):
    """Returns none"""
    return None


# pylint: disable=protected-access
@_ones_like_leaf_for_grad._register_default(False)
def default_ones_like_leaf_for_grad(x):
    """Default function for _ones_like_leaf_for_grad."""
    return x


ones_like = base.HyperMap(ones_like_leaf)
"""
`ones_like` is a function which can generate a graph of `ones_like` operation according to input tensor dtype.

Example:
    >>> input = Tensor([2, 3], mindspore.int32)
    >>> ones = ones_like(input) # The dtype of ones is mindspore.int32
    >>> input = Tensor([2, 3], mindspore.float16)
    >>> ones = ones_like(input) # The dtype of ones is mindspore.float16
"""

_ones_like_for_grad = base.HyperMap(_ones_like_leaf_for_grad)
