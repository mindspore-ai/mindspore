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
"""math Operations."""
from mindspore.ops.composite.multitype_ops import _constexpr_utils as const_utils
from mindspore.common import dtype as mstype
from mindspore._checkparam import Validator as validator
from mindspore.ops.primitive import constexpr
from mindspore.ops import functional as F
from .. import operations as P


@constexpr
def _check_validate_axis(axis, name):
    if isinstance(axis, (tuple, list)):
        for idx, item in enumerate(axis):
            validator.check_value_type("axis[%d]" % idx, item, [int], name)
    axis = validator.check_value_type('axis', axis, [int, tuple, list], name)
    return axis


@constexpr
def _check_validate_keepdims(keep_dims, name):
    keep_dims = validator.check_value_type('keep_dims', keep_dims, [bool], name)
    return keep_dims


def count_nonzero(x, axis=(), keep_dims=False, dtype=mstype.int32):
    """
    Count number of nonzero elements across axis of input tensor

    Args:
        x (Union(tuple[Tensor], list[Tensor])): Input data is used to count non-zero numbers.
        axis (Union[int, tuple(int), list(int)]): The dimensions to reduce. Only constant value is allowed.
                                                  Default: (), reduce all dimensions.
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions. Default: False.
        dtype (Union[Number, mstype.bool_]): The data type of the output tensor. Only constant value is allowed.
                                             Default: mstype.int32

    Returns:
          Tensor, number of nonzero element. The data type is dtype.

    Examples:
        >>> input_x = Tensor(np.array([[0, 1, 0], [1, 1, 0]]).astype(np.float32))
        >>> nonzero_num = count_nonzero(x=input_x, axis=[0, 1], keep_dims=True, dtype=mstype.int32)
        >>> print(nonzero_num)
        [[3]]
    """

    const_utils.check_valid_type(F.dtype(x), mstype.number_type, 'input x')
    axis = _check_validate_axis(axis, "count_nonzero")
    keep_dims = _check_validate_keepdims(keep_dims, "count_nonzero")
    const_utils.check_valid_type(dtype, mstype.number_type + (mstype.bool_,), 'dtype')

    not_equal = P.NotEqual()
    cast = P.Cast()
    reduce_sum = P.ReduceSum(keep_dims)
    nonzero_bool = not_equal(x, 0)
    # ReduceSum only support float16 or float32 tensor.
    nonzero_val = cast(nonzero_bool, mstype.float16)
    nonzero_num = cast(reduce_sum(nonzero_val, axis), dtype)

    return nonzero_num
