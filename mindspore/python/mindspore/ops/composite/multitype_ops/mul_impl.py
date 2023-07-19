# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.composite.multitype_ops import _compile_utils as utils
from mindspore.ops.composite.multitype_ops._constexpr_utils import check_equal
from mindspore.ops.composite import base
from mindspore.ops import functional as F
from mindspore.common import COOTensor
from ...operations._sequence_ops import SequenceMul


mul = base.MultitypeFuncGraph("mul", True)
"""
`mul` is a metafuncgraph object which will multiply two objects according to input type
using ".register" decorator.
"""
mul.set_need_raise()


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


@mul.register("Number", "String")
def _number_mul_string(x, y):
    """
    Returns x * y where x is a number and y is a string. x must be integer.

    Outputs:
       String.
    """
    return inner.string_mul(x, y)


@mul.register("String", "Number")
def _string_mul_number(x, y):
    """
    Returns x * y where x is a string and y is a number. y must be integer.

    Outputs:
        String.
    """
    return inner.string_mul(x, y)


@mul.register("List", "Number")
def _list_mul_scalar(x, y):
    """
    Returns x * y where x is a list and y is a number. y must be integer.

    Outputs:
        List.
    """
    if not isinstance(y, int):
        raise TypeError(f"can't multiply sequence by non-int of type '{type(y)}'.")
    if F.is_sequence_shape_unknown(x) or not F.isconstant(y):
        return SequenceMul()(x, y)
    res = []
    i = 0
    while i < y:
        res += x
        i += 1
    return res


@mul.register("Number", "List")
def _scalar_mul_list(x, y):
    """
    Returns x * y where x is a number and y is a list. x must be integer.

    Outputs:
        List.
    """
    if not isinstance(x, int):
        raise TypeError(f"can't multiply sequence by non-int of type '{type(x)}'.")
    if not F.isconstant(x) or F.is_sequence_shape_unknown(y):
        return SequenceMul()(y, x)
    res = []
    i = 0
    while i < x:
        res += y
        i += 1
    return res


@mul.register("Tuple", "Number")
def _tuple_mul_scalar(x, y):
    """
    Returns x * y where x is a tuple and y is a number. y must be integer.

    Outputs:
        Tuple.
    """
    if not isinstance(y, int):
        raise TypeError(f"can't multiply sequence by non-int of type '{type(y)}'.")
    if F.is_sequence_shape_unknown(x) or not F.isconstant(y):
        return SequenceMul()(x, y)
    res = ()
    i = 0
    while i < y:
        res += x
        i += 1
    return res


@mul.register("Number", "Tuple")
def _scalar_mul_tuple(x, y):
    """
    Returns x * y where x is a number and y is a tuple. x must be integer.

    Outputs:
        Tuple.
    """
    if not isinstance(x, int):
        raise TypeError(f"can't multiply sequence by non-int of type '{type(x)}'.")
    if not F.isconstant(x) or F.is_sequence_shape_unknown(y):
        return SequenceMul()(y, x)
    res = ()
    i = 0
    while i < x:
        res += y
        i += 1
    return res


@mul.register("Tensor", "Tuple")
def _tensor_mul_tuple(x, y):
    """
    Returns x * y where x is a tensor and y is a tuple.

    Args:
        x (Tensor): x
        y (Tuple): The dtype is same as x.

    Returns:
        Tensor, has the same dtype as x.
    """
    y = utils.sequence_to_tensor(y, x.dtype)
    return F.tensor_mul(x, y)


@mul.register("Tuple", "Tensor")
def _tuple_mul_tensor(x, y):
    """
    Returns x * y where x is a tuple and y is a tensor.

    Args:
        x (Tuple): x
        y (Tensor): The dtype is same as x.

    Returns:
        Tensor, has the same dtype as x.
    """
    x = utils.sequence_to_tensor(x, y.dtype)
    return F.tensor_mul(x, y)


@mul.register("Tensor", "List")
def _tensor_mul_list(x, y):
    """
    Returns x * y where x is a tensor and y is a list.

    Args:
        x (Tensor): x
        y (List): The dtype is same as x.

    Returns:
        Tensor, has the same dtype as x.
    """
    y = utils.sequence_to_tensor(y, x.dtype)
    return F.tensor_mul(x, y)


@mul.register("List", "Tensor")
def _list_mul_tensor(x, y):
    """
    Returns x * y where x is a list and y is a tensor.

    Args:
        x (List): x
        y (Tensor): The dtype is same as x.

    Returns:
        Tensor, has the same dtype as x.
    """
    x = utils.sequence_to_tensor(x, y.dtype)
    return F.tensor_mul(x, y)


@mul.register("CSRTensor", "Tensor")
def _csrtensor_mul_tensor(x, y):
    """
    Returns x * y where x is CSRTensor and y is Tensor.

    Outputs:
       CSRTensor, equal to x * y.
    """
    return F.csr_mul(x, y)


@mul.register("Tensor", "CSRTensor")
def _tensor_mul_csrtensor(x, y):
    """
    Returns x * y where x is Tensor and y is CSRTensor.

    Outputs:
       CSRTensor, equal to x * y.
    """
    return F.csr_mul(y, x)


@mul.register("COOTensor", "Tensor")
def _cootensor_mul_tensor(x, y):
    """
    Returns x * y where x is COOTensor and y is Tensor.

    Outputs:
       COOTensor, equal to x * y.
    """
    check_equal(x.shape, y.shape, "input1 (shape={}) and input2(shape={}) should be the same shape.")
    other_values = F.gather_nd(y, x.indices)
    return COOTensor(x.indices, x.values * other_values, x.shape)


@mul.register("Tensor", "COOTensor")
def _tensor_mul_cootensor(x, y):
    """
    Returns x * y where x is Tensor and y is COOTensor.

    Outputs:
       COOTensor, equal to x * y.
    """
    check_equal(x.shape, y.shape, "input1 (shape={}) and input2(shape={}) should be the same shape.")
    other_values = F.gather_nd(x, y.indices)
    return COOTensor(y.indices, y.values * other_values, y.shape)
