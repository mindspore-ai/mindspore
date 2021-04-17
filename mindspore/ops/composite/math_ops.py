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
from itertools import zip_longest
from collections import deque
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
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[0, 1, 0], [1, 1, 0]]).astype(np.float32))
        >>> nonzero_num = count_nonzero(x=input_x, axis=[0, 1], keep_dims=True, dtype=mstype.int32)
        >>> print(nonzero_num)
        [[3]]
    """

    const_utils.check_type_valid(F.dtype(x), mstype.number_type, 'input x')
    axis = _check_validate_axis(axis, "count_nonzero")
    keep_dims = _check_validate_keepdims(keep_dims, "count_nonzero")
    const_utils.check_type_valid(dtype, mstype.number_type + (mstype.bool_,), 'dtype')

    not_equal = P.NotEqual()
    cast = P.Cast()
    reduce_sum = P.ReduceSum(keep_dims)
    nonzero_bool = not_equal(x, 0)
    # ReduceSum only support float16 or float32 tensor.
    nonzero_val = cast(nonzero_bool, mstype.float32)
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
        axes = list(axes)  # to avoid immutability issues
        if len(axes) != 2:
            raise ValueError("Require two axes inputs, given less")
        axes = _int_to_tuple_conv(axes)  # convert before length checks
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
    const_utils.check_type_valid(x1_type, [mstype.float32, mstype.float16], 'x1')
    const_utils.check_type_valid(x2_type, [mstype.float32, mstype.float16], 'x2')
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

    Raises:
        TypeError: If `x1` or `x2` is not a Tensor.
        TypeError: If `axes` is not one of the following: int, tuple, list.

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


@constexpr
def _check_invalid_input(x1_shape, x2_shape):
    if len(x1_shape) < 2 or len(x2_shape) < 2:
        raise ValueError('C.dot inputs x1, x2 should has dimension >= 2,'
                         + f'while x1 is ({len(x1_shape)}) and x2 is ({len(x2_shape)}).')


@constexpr
def _typecheck_input_dot(x1_type, x2_type):
    """
    Check input tensor types to be valid and confirm they are the same type for dot and batch dot ops.
    """
    const_utils.check_type_valid(x1_type, [mstype.float16, mstype.float32], 'x1')
    const_utils.check_type_valid(x2_type, [mstype.float16, mstype.float32], 'x2')
    if x1_type != x2_type:
        raise TypeError(f'Both Inputs must be the same Type. x1 is \'{x1_type}\' and x2 is \'{x2_type}\' ')


@constexpr
def _get_transpose_shape(x2_shape):
    x2_shape_range = tuple(range(len(x2_shape)))
    x2_shape_transpose = x2_shape_range[-2:-1] + x2_shape_range[:-2] + x2_shape_range[-1:]
    return x2_shape_transpose


def dot(x1, x2):
    """
    Computation a dot product between samples in two tensors.

    Inputs:
        - **x1** (Tensor) - First tensor in Dot op with datatype float16 or float32
        - **x2** (Tensor) - Second tensor in Dot op with datatype float16 or float32

    Outputs:
        Tensor, dot product of x1 and x2.

    Raises:
        TypeError: If type of x1 and x2 are not the same.
        TypeError: If dtype of x1 or x2 is not float16 or float32.
        ValueError: If rank of x1 or x2 less than 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x1 = Tensor(np.ones(shape=[2, 3]), mindspore.float32)
        >>> input_x2 = Tensor(np.ones(shape=[1, 3, 2]), mindspore.float32)
        >>> output = C.dot(input_x1, input_x2)
        >>> print(output)
        [[[3. 3.]]
         [[3. 3.]]]
    """
    shape_op = P.Shape()
    reshape_op = P.Reshape()
    transpose_op = P.Transpose()
    matmul_op = P.MatMul(False, False)
    x1_shape = shape_op(x1)
    x2_shape = shape_op(x2)
    x1_type = F.dtype(x1)
    x2_type = F.dtype(x2)
    _typecheck_input_dot(x1_type, x2_type)
    _check_invalid_input(x1_shape, x2_shape)

    if len(x1_shape) > 2 or len(x2_shape) > 2:
        x2_shape_transpose = _get_transpose_shape(x2_shape)
        x2_transpose = transpose_op(x2, x2_shape_transpose)
        x1_reshape = reshape_op(x1, (-1, x1_shape[-1]))
        x2_reshape = reshape_op(x2_transpose, (x2_shape[-2], -1))
        mul_result = matmul_op(x1_reshape, x2_reshape)
        return reshape_op(mul_result, x1_shape[:-1] + x2_shape[:-2] + x2_shape[-1:])
    return matmul_op(x1, x2)


@constexpr
def _get_batch_size(x1_shape, x2_shape):
    """
    Get batch sizes from two inputs
    """
    if len(x1_shape) < 2 or len(x2_shape) < 2:
        raise ValueError("Require both inputs with rank >= 2.")
    return x1_shape[0], x2_shape[0]


@constexpr
def _typecheck_input_batch_dot(x1_type, x2_type):
    """
    Check input tensor types to be valid and confirm they are the same type for batch dot ops.
    """
    const_utils.check_type_valid(x1_type, [mstype.float32], 'x1')
    const_utils.check_type_valid(x2_type, [mstype.float32], 'x2')
    if x1_type != x2_type:
        raise TypeError(f'Both Inputs must be the same Type. x1 is \'{x1_type}\' and x2 is \'{x2_type}\' ')


@constexpr
def _check_axes_for_batch_dot(x1_shape, x2_shape, axes):
    """
    Check whether axes are valid and cast axes from tuple to list
    """
    if axes is None:
        if len(x2_shape) == 2:
            axes = [len(x1_shape) - 1, len(x2_shape) - 1]
        else:
            axes = [len(x1_shape) - 1, len(x2_shape) - 2]

    if isinstance(axes, (list, tuple)):
        if 0 in axes:
            raise ValueError("Batch dim cannot be used as in axes.")
        if len(axes) != 2:
            raise ValueError("Require two axes inputs, given less")
        if isinstance(axes, tuple):
            axes = list(axes)
        validator.check_value_type('axes[0]', axes[0], [int], 'batch_dot')
        validator.check_value_type('axes[1]', axes[1], [int], 'batch_dot')
        # Reverse if axis < 0
        if axes[0] < 0:
            axes[0] += len(x1_shape)
        if axes[1] < 0:
            axes[1] += len(x2_shape)
        validator.check_non_negative_int(axes[0], 'reversed axes[0]', 'batch_dot')
        validator.check_non_negative_int(axes[1], 'reversed axes[1]', 'batch_dot')
        if axes[0] > len(x1_shape) or axes[1] > len(x2_shape):
            raise ValueError(
                "Axes value too high for given input arrays dimensions.")
    elif isinstance(axes, int):
        if axes == 0:
            raise ValueError("Batch dim cannot be used as in axes.")
        if axes < 0:
            axes = [axes + len(x1_shape), axes + len(x2_shape)]
            validator.check_non_negative_int(axes[0], 'reversed axes', 'batch_dot')
        elif axes > len(x1_shape) or axes > len(x2_shape):
            raise ValueError(
                "Axes value too high for given input arrays dimensions.")
        else:
            axes = [axes, axes]
    else:
        raise ValueError(
            "Axes type must be one of those: int, tuple(int), list(int).")
    return axes


@constexpr
def _calc_new_shape_batchdot(shape, axes, position=0):
    """
    Calculate transpose and reshape parameters for input transformations,
    'position' refers to whether tensor is first or second in the op.
    """
    axis = axes[position]
    contraction_axes = tuple([axis])
    prod_contraction = int(np.prod([shape[i] for i in contraction_axes]))
    free_axes = tuple(i for i in range(1, len(shape)) if i not in contraction_axes)
    free_dims = tuple(shape[i] for i in free_axes)
    prod_free = int(np.prod(free_dims))

    transpose_perm = contraction_axes + free_axes if position else free_axes + contraction_axes
    transpose_perm = tuple([0]) + transpose_perm
    new_shape = (prod_contraction, prod_free) if position else (prod_free, prod_contraction)
    new_shape = tuple([shape[0]]) + new_shape
    return new_shape, transpose_perm, free_dims


@constexpr
def _check_batch_size(x1_batch_size, x2_batch_size):
    """
    Check whether batch size of two inputs are the same
    """
    if x1_batch_size != x2_batch_size:
        raise ValueError("Require both inputs with the same batch sizes.")

@constexpr
def _get_output_shape(batch_size, x1_ret, x2_ret):
    """
    Compute output shape for batch dot
    """
    output_shape = tuple([batch_size]) + x1_ret + x2_ret
    return output_shape

def batch_dot(x1, x2, axes=None):
    """
    Computation of batch dot product between samples in two tensors containing batch dims.

    .. math::
        output = x1[batch, :] * x2[batch, :]

    Inputs:
        - **x1** (Tensor) - First tensor in Batch Dot op with datatype float32
        - **x2** (Tensor) - Second tensor in Batch Dot op with datatype float32. x2's datatype should
          be same as x1's.
        - **axes** (Union[int, tuple(int), list(int)]) - Single value or tuple/list of length 2 with dimensions
          specified for `a` and `b` each. If single value `N` passed, automatically picks up last N dims from
          `a` input shape and last N dims from `b` input shape in order as axes for each respectively.

    Outputs:
        Tensor, batch dot product of x1 and x2. The Shape of output for input shapes (batch, d1, axes, d2) and
        (batch, d3, axes, d4) is (batch, d1, d2, d3, d4)

    Raises:
        TypeError: If type of x1 and x2 are not the same.
        TypeError: If dtype of x1 or x2 is not float32.
        ValueError: If rank of x1 or x2 less than 2.
        ValueError: If batch dim used in axes.
        ValueError: If len(axes) less than 2.
        ValueError: If axes is not one of those: None, int, (int, int).
        ValueError: If axes reversed from negative int is too low for dimensions of input arrays.
        ValueError: If axes value is too high for dimensions of input arrays.
        ValueError: If batch size of x1 and x2 are not the same.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x1 = Tensor(np.ones(shape=[2, 2, 3]), mindspore.float32)
        >>> input_x2 = Tensor(np.ones(shape=[2, 3, 2]), mindspore.float32)
        >>> axes = (-1, -2)
        >>> output = C.batch_dot(input_x1, input_x2, axes)
        >>> print(output)
        [[[3. 3.]
          [3. 3.]]
         [[3. 3.]
          [3. 3.]]]
    """

    transpose_op = P.Transpose()
    batch_matmul_op = P.BatchMatMul()
    squeeze_one_op = P.Squeeze(1)
    squeeze_minus_one_op = P.Squeeze(-1)
    # input validity checks
    x1_shape = F.shape(x1)
    x2_shape = F.shape(x2)
    x1_dim_num = len(x1_shape)
    x2_dim_num = len(x2_shape)
    x1_type = F.dtype(x1)
    x2_type = F.dtype(x2)

    x1_batch_size, x2_batch_size = _get_batch_size(x1_shape, x2_shape)

    _typecheck_input_batch_dot(x1_type, x2_type)
    _check_batch_size(x1_batch_size, x2_batch_size)
    axes = _check_axes_for_batch_dot(x1_shape, x2_shape, axes)

    if x1_dim_num == 2:
        x1 = F.expand_dims(x1, 1)
        axes[0] += 1
    if x2_dim_num == 2:
        x2 = F.expand_dims(x2, 2)

    x1_shape = F.shape(x1)
    x2_shape = F.shape(x2)

    x1_reshape_fwd, x1_transpose_fwd, x1_ret = _calc_new_shape_batchdot(x1_shape, axes, 0)
    x2_reshape_fwd, x2_transpose_fwd, x2_ret = _calc_new_shape_batchdot(x2_shape, axes, 1)
    output_shape = _get_output_shape(x1_batch_size, x1_ret, x2_ret)

    x1_transposed = transpose_op(x1, x1_transpose_fwd)
    x2_transposed = transpose_op(x2, x2_transpose_fwd)
    x1_reshaped = F.reshape(x1_transposed, x1_reshape_fwd)
    x2_reshaped = F.reshape(x2_transposed, x2_reshape_fwd)

    # Batch matmal op part
    mul_result = batch_matmul_op(x1_reshaped, x2_reshaped)

    final_result = F.reshape(mul_result, output_shape)

    # if the original dims are expanded, restore them from 3 to 2
    if x1_dim_num == 2:
        final_result = squeeze_one_op(final_result)
    elif x2_dim_num == 2:
        final_result = squeeze_minus_one_op(final_result)

    return final_result

@constexpr
def _check_same_type(dtype1, dtype2):
    return dtype1 == dtype2

@constexpr
def _max(*args):
    """Returns the maximum value."""
    return max(*args)

@constexpr
def _min(*args):
    """Returns the minimum value."""
    return min(*args)

@constexpr
def _infer_shape_rem(shape1, shape2, ndim1, ndim2, transpose_b):
    """Infers the shape of the last two dimensions after performing matmul."""
    shape_rem = []
    if ndim1 >= 2:
        shape_rem.append(shape1[-2])
    if transpose_b:
        if ndim2 >= 2:
            shape_rem.append(shape2[-2])
    else:
        if ndim1 >= 1:
            shape_rem.append(shape2[-1])
    return tuple(shape_rem)

@constexpr
def _check_matmul_shapes(shape1, shape2):
    """Checks shape1 and shape2 are valid to perform matmul, and returns output shape after broadcasting."""
    ndim1, ndim2 = len(shape1), len(shape2)
    if ndim1 < 1 or ndim2 < 1:
        raise ValueError('input operands must have at least 1 dimension')
    if ndim2 >= 2 and shape1[-1] != shape2[-2]:
        raise ValueError(f'mismatch in core dimension of input operands (size '
                         f'{shape1[-1]} is different from {shape2[-2]})')
    shape_out = deque()
    for items in zip_longest(reversed(shape1[:-2]), reversed(shape2[:-2]), fillvalue=1):
        max_size = max(items)
        if any(item not in (1, max_size) for item in items):
            raise ValueError(f'operands could not be broadcast together with shapes {shape1} {shape2}')
        shape_out.appendleft(max_size)
    return tuple(shape_out)

@constexpr
def _tile_size(shape, out_shape, ndim):
    """Returns tile_size such that shape*tile_size = out_shape"""
    size = [1]*ndim
    for idx, (i, j) in enumerate(zip(shape, out_shape)):
        if i != j:
            size[idx] = j
    return tuple(size)

@constexpr
def _check_need_broadcast(shape1, shape2):
    """Returns True if broadcast is necessary for batchmatmul."""
    return shape1[:-2] != shape2[:-2]

def _expand(x, ndim):
    """Expand x to ndim from axis, which can be 0 or -1."""
    while F.rank(x) < ndim:
        x = F.expand_dims(x, 0)
    return x

def _broadcast_to(x, shape_cur, shape_to, ndim_to):
    """Broadcasts x from shape_cur to shape_to."""
    size = _tile_size(shape_cur, shape_to, ndim_to)
    return F.tile(x, size)

def matmul(x1, x2, dtype=None):
    """
    Returns the matrix product of two arrays.

    Note:
        Numpy arguments `out`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        On GPU, the supported dtypes are np.float16 and np.float32.
        On CPU, the supported dtypes are np.float16 and np.float32.

    Args:
        x1 (Tensor): Input tensor, scalar not allowed.
        x2 (Tensor): Input tensor, scalar not allowed.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the matrix product of the inputs. This is a scalar only
        when both `x1`, `x2` are 1-d vectors.

    Raises:
        ValueError: If the last dimension of `x1` is not the same size as the
            second-to-last dimension of `x2`, or if a scalar value is passed in.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = Tensor(np.arange(2*3*4).reshape(2, 3, 4), mindspore.float32)
        >>> x2 = Tensor(np.arange(4*5).reshape(4, 5), mindspore.float32)
        >>> output = ops.matmul(x1, x2)
        >>> print(output)
        [[[  70.   76.   82.   88.   94.]
        [ 190.  212.  234.  256.  278.]
        [ 310.  348.  386.  424.  462.]]
        [[ 430.  484.  538.  592.  646.]
        [ 550.  620.  690.  760.  830.]
        [ 670.  756.  842.  928. 1014.]]]
    """
    # performs type promotion
    dtype1 = F.dtype(x1)
    dtype2 = F.dtype(x2)
    if not _check_same_type(dtype1, dtype2):
        x1 = x1.astype(mstype.float32)
        x2 = x2.astype(mstype.float32)

    ndim1_orig, ndim2_orig = F.rank(x1), F.rank(x2)
    shape1_orig, shape2_orig = F.shape(x1), F.shape(x2)
    transpose_b = ndim2_orig == 1
    shape_backbone = _check_matmul_shapes(shape1_orig, shape2_orig)
    # infers the shape of the output
    shape_out = shape_backbone + _infer_shape_rem(shape1_orig, shape2_orig,
                                                  ndim1_orig, ndim2_orig, transpose_b)

    x1 = _expand(x1, 2)
    x2 = _expand(x2, 2)
    if F.rank(x2) == 2:
        if F.rank(x1) > 2:
            x1 = F.reshape(x1, (-1, shape1_orig[-1]))
        res = P.MatMul(False, transpose_b)(x1, x2)
    else:
        # broadcasts x1.shape[:-2] with x2.shape[:-2]
        ndim_aligned = _max(ndim1_orig, ndim2_orig)
        x1 = _expand(x1, ndim_aligned)
        x2 = _expand(x2, ndim_aligned)
        shape1_aligned, shape2_aligned = F.shape(x1), F.shape(x2)
        x1 = _broadcast_to(x1, shape1_aligned[:-2], shape_backbone, ndim_aligned)
        x2 = _broadcast_to(x2, shape2_aligned[:-2], shape_backbone, ndim_aligned)
        res = P.BatchMatMul(False, transpose_b)(x1, x2)

    if dtype is not None:
        res = res.astype(dtype)
    return F.reshape(res, shape_out)
