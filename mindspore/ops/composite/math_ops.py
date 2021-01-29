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
import numpy as np
from mindspore.ops.composite.multitype_ops import _constexpr_utils as const_utils
from mindspore.common import dtype as mstype
from mindspore._checkparam import Validator as validator
from mindspore.ops.primitive import constexpr
from mindspore.ops import functional as F
from .. import operations as P

# count_nonzero
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
    r"""
    Count number of nonzero elements across axis of input tensor

    Args:
        x (Tensor): Input data is used to count non-zero numbers.
        axis (Union[int, tuple(int), list(int)]): The dimensions to reduce. Only constant value is allowed.
                                                  Default: (), reduce all dimensions.
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions. Default: False.
        dtype (Union[Number, mstype.bool\_]): The data type of the output tensor. Only constant value is allowed.
                                             Default: mstype.int32

    Returns:
          Tensor, number of nonzero element. The data type is dtype.

    Supported Platforms:
        ``Ascend`` ``GPU``

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

# tensor dot
@constexpr
def _int_to_tuple_conv(axes):
    """
    Converts ints to tuples in input axes, expected by most validation checks.
    """
    for x in [0, 1]:
        if isinstance(axes[x], int):
            axes[x] = (axes[x],)
    return axes


@constexpr
def _check_axes(axes):
    """
    Check for validity and type of axes passed to function.
    """
    validator.check_value_type('axes', axes, [int, tuple, list], "tensor dot")
    if not isinstance(axes, int):
        axes = list(axes) # to avoid immutability issues
        if len(axes) != 2:
            raise ValueError("Require two axes inputs, given less")
        axes = _int_to_tuple_conv(axes) # convert before length checks
        if len(axes[0]) != len(axes[1]):
            raise ValueError("Axes have to be the same size/length")
        if len(axes[0]) != len(set(axes[0])) or len(axes[1]) != len(set(axes[1])):
            raise ValueError("Axes cannot have duplicating values")
    return axes


@constexpr
def _typecheck_input(x1_type, x2_type):
    """
    Check input tensor types to be valid and confirm they are the same type.
    """
    const_utils.check_valid_type(x1_type, [mstype.float32, mstype.float16], 'x1')
    const_utils.check_valid_type(x2_type, [mstype.float32, mstype.float16], 'x2')
    if x1_type != x2_type:
        raise TypeError(f'Both Inputs must be the same Type. x1 is \'{x1_type}\' and x2 is \'{x2_type}\' ')


@constexpr
def _axes_int_check(x1_shape, x2_shape, axes):
    """
    Convert from single int axes to 2d tuple if required
    """
    if isinstance(axes, int):
        if axes < 0:
            raise ValueError(f"axes must be at least 0 for tensor dot, got {axes}")
        if axes == 0:
            # outer product, no input validation required
            return ([], [])
        if axes > len(x1_shape) or axes > len(x2_shape):
            raise ValueError(
                "Axes value too high for given input arrays dimensions.")
        x1_ind = tuple(range(len(x1_shape))[-1 * axes:])
        x2_ind = tuple(range(len(x2_shape))[:axes])
        axes = tuple((x1_ind, x2_ind))
        axes = _int_to_tuple_conv(axes)
    return axes


@constexpr
def _validate_axes(x1_shape, x2_shape, axes):
    """
    Checks for axes having the correct length according to input, for any value in axis
    being out of range with given shape and also checking for compatible axes values
    with given inputs.
    """
    shapes = [x1_shape, x2_shape]

    # axis length check
    for ix_input, x_axes in enumerate(axes):
        axes_len = len(x_axes)
        shape_dim_len = len(shapes[ix_input])
        if axes_len > shape_dim_len:
            raise ValueError(f"axes for input: {ix_input + 1} are of length: {axes_len} "
                             f"can only be max: {shape_dim_len} due to input shape.")

    # axis values range check
    for ix_input, x_axes in enumerate(axes):
        comp_shape = shapes[ix_input]
        max_val = len(comp_shape) - 1
        min_val = -1 * len(comp_shape)
        for _, x_value in enumerate(x_axes):
            if not min_val <= x_value <= max_val:
                raise ValueError(f"axes for input: {ix_input + 1} contains index: "
                                 f"{x_value}, but range is: [{min_val}, {max_val}]")

    # check axis value with input shape - both ways for axis valid
    invalid_a = False
    invalid_b = False
    for i in range(len(axes[0])):  # sizes already validated
        if x1_shape[axes[0][i]] != x2_shape[axes[1][i]]:
            invalid_a = True
        if x1_shape[axes[0][i]] != x2_shape[axes[1][len(axes[0])-1-i]]:
            invalid_b = True
    if invalid_a and invalid_b:
        raise ValueError("Given Axes are incompatible with given input arrays")


@constexpr
def _calc_new_shape(shape, axes, position=0):
    """
    Calculate transpose and reshape parameters for input transformations,
    'position' refers to whether tensor is first or second in the op.
    """
    contraction_axes = tuple(i if i >= 0 else i + len(shape) for i in axes[position])
    prod_contraction = int(np.prod([shape[i] for i in contraction_axes]))
    free_axes = tuple(i for i in range(len(shape)) if i not in contraction_axes)
    free_dims = tuple(shape[i] for i in free_axes)
    prod_free = int(np.prod(free_dims))

    transpose_perm = contraction_axes + free_axes if position else free_axes + contraction_axes
    new_shape = (prod_contraction, prod_free) if position else (prod_free, prod_contraction)
    return new_shape, transpose_perm, free_dims


def tensor_dot(x1, x2, axes):
    """
    Computation of Tensor contraction on arbitrary axes between tensors `a` and `b`.

    Contraction allows for the summation of products of elements of `a` and `b` on specified axes.
    The same number of axes must be specified for both x1 and x2, and values must be within range
    of number of dims of both `a` and `b`.

    Selected dims in both inputs must also match.

    axes = 0 leads to outer product
    axes = 1 leads to normal matrix multiplication when inputs both 2D.
    axes = 1 is the same as axes = ((1,),(0,) where both `a` and `b` are 2D.
    axes = 2 is the same as axes = ((1,2),(0,1)) where both `a` and `b` are 3D.

    Inputs:
        - **x1** (Tensor) - First tensor in tensor_dot with datatype float16 or float32
        - **x2** (Tensor) - Second tensor in tensor_dot with datatype float16 or float32
        - **axes** (Union[int, tuple(int), tuple(tuple(int)), list(list(int))]) - Single value or
          tuple/list of length 2 with dimensions specified for `a` and `b` each. If single value `N` passed,
          automatically picks up last N dims from `a` input shape and first N dims from `b` input shape in order
          as axes for each respectively.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N + M)`. Where :math:`N` and :math:`M` are the free axes not
        contracted in both inputs

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x1 = Tensor(np.ones(shape=[1, 2, 3]), mindspore.float32)
        >>> input_x2 = Tensor(np.ones(shape=[3, 1, 2]), mindspore.float32)
        >>> output = C.tensor_dot(input_x1, input_x2, ((0,1),(1,2)))
        >>> print(output)
        [[2. 2. 2]
         [2. 2. 2]
         [2. 2. 2]]
    """
    shape_op = P.Shape()
    reshape_op = P.Reshape()
    transpose_op = P.Transpose()
    matmul_op = P.MatMul(False, False)
    # input validity checks
    x1_shape = shape_op(x1)
    x2_shape = shape_op(x2)
    x1_type = F.dtype(x1)
    x2_type = F.dtype(x2)
    axes = _check_axes(axes)
    _typecheck_input(x1_type, x2_type)
    # input compatibility check & axes format update
    axes = _axes_int_check(x1_shape, x2_shape, axes)
    _validate_axes(x1_shape, x2_shape, axes)
    x1_reshape_fwd, x1_transpose_fwd, x1_ret = _calc_new_shape(x1_shape, axes, 0)
    x2_reshape_fwd, x2_transpose_fwd, x2_ret = _calc_new_shape(x2_shape, axes, 1)
    output_shape = x1_ret + x2_ret  # combine free axes from both inputs
    # run tensor_dot op
    x1_transposed = transpose_op(x1, x1_transpose_fwd)
    x2_transposed = transpose_op(x2, x2_transpose_fwd)
    x1_reshaped = reshape_op(x1_transposed, x1_reshape_fwd)
    x2_reshaped = reshape_op(x2_transposed, x2_reshape_fwd)
    mul_result = matmul_op(x1_reshaped, x2_reshaped)
    final_result = reshape_op(mul_result, output_shape)
    return final_result
