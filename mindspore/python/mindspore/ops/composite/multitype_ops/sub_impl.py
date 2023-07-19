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

from __future__ import absolute_import
from mindspore.ops.composite.multitype_ops import _compile_utils as utils
from mindspore.ops.composite.multitype_ops._constexpr_utils import check_equal, make_tensor
from mindspore.ops.composite import base
from mindspore.ops import functional as F


sub = base.MultitypeFuncGraph("sub", True)
"""
`sub` is a metafuncgraph object which will compute the subtraction of two objects
using ".register" decorator.
"""
sub.set_need_raise()


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


@sub.register("Tuple", "Tensor")
def _tuple_sub_tensor(x, y):
    """Returns x - y where x is a tuple and y is a tensor. """
    x = utils.sequence_to_tensor(x, y.dtype)
    return F.tensor_sub(x, y)


@sub.register("Tensor", "Tuple")
def _tensor_sub_tuple(x, y):
    """Returns x - y where x is a tensor and y is a tuple. """
    y = utils.sequence_to_tensor(y, x.dtype)
    return F.tensor_sub(x, y)


@sub.register("List", "Tensor")
def _list_sub_tensor(x, y):
    """Returns x - y where x is a list and y is a tensor. """
    x = utils.sequence_to_tensor(x, y.dtype)
    return F.tensor_sub(x, y)


@sub.register("Tensor", "List")
def _tensor_sub_list(x, y):
    """Returns x - y where x is a tensor and y is a list. """
    y = utils.sequence_to_tensor(y, x.dtype)
    return F.tensor_sub(x, y)


@sub.register("CSRTensor", "CSRTensor")
def _sub_csrtensor(x, y):
    """Returns x - y where x and y are all CSR tensors."""
    check_equal(x.shape, y.shape, "input1 (shape={}) and input2(shape={}) should be the same shape.")
    return F.csr_add(x, y, make_tensor(1, x.values.dtype), make_tensor(-1, x.values.dtype))


@sub.register("COOTensor", "COOTensor")
def _sub_cootensor(x, y):
    """Returns x - y where x and y are all COO tensors."""
    check_equal(x.shape, y.shape, "input1 (shape={}) and input2(shape={}) should be the same shape.")
    return F.coo_add(x, -y, make_tensor(0, x.values.dtype))


@sub.register("Tensor", "COOTensor")
def _tensor_sub_cootensor(x, y):
    """Returns x - y where x is a tensor and y is a COO tensor."""
    check_equal(x.shape, y.shape, "input1 (shape={}) and input2(shape={}) should be the same shape.")
    return F.tensor_scatter_sub(x, y.indices, y.values)


@sub.register("COOTensor", "Tensor")
def _cootensor_sub_tensor(x, y):
    """Returns x - y where x is a COO tensor and y is a tensor."""
    check_equal(x.shape, y.shape, "input1 (shape={}) and input2(shape={}) should be the same shape.")
    return F.tensor_scatter_add(-y, x.indices, x.values)
