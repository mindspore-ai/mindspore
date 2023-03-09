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
from mindspore.ops.primitive import constexpr, _primexpr
from mindspore.ops import functional as F
from mindspore.ops.function.math_func import cummin as cummin_
from mindspore.ops import operations as P


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


@constexpr
def is_const(x):
    return x is not None


def count_nonzero(x, axis=(), keep_dims=False, dtype=mstype.int32):
    r"""
    Count number of nonzero elements across axis of input tensor

    Args:
        x (Tensor): Input data is used to count non-zero numbers.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
        axis (Union[int, tuple(int), list(int)]): The dimensions to reduce. Only constant value is allowed.
                                                  Default: (), reduce all dimensions.
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions. Default: False.
        dtype (Union[Number, mindspore.bool\_]): The data type of the output tensor. Only constant value is allowed.
                                             Default: mindspore.int32

    Returns:
          Tensor, number of nonzero element. The data type is `dtype`.

    Raises:
        TypeError: If axis is not int or tuple.
        ValueError: If axis is not in range [-x.ndim, x.ndim).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import numpy as np
        >>> # case 1: each value specified.
        >>> x = Tensor(np.array([[0, 1, 0], [1, 1, 0]]).astype(np.float32))
        >>> nonzero_num = ops.count_nonzero(x=x, axis=[0, 1], keep_dims=True, dtype=mindspore.int32)
        >>> print(nonzero_num)
        [[3]]
        >>> # case 2: all value is default.
        >>> nonzero_num = ops.count_nonzero(x=x)
        >>> print(nonzero_num)
        3
        >>> # case 3: axis value was specified 0.
        >>> nonzero_num = ops.count_nonzero(x=x, axis=[0,])
        >>> print(nonzero_num)
        [1 2 0]
        >>> # case 4: axis value was specified 1.
        >>> nonzero_num = ops.count_nonzero(x=x, axis=[1,])
        >>> print(nonzero_num)
        [1 2]
        >>> # case 5: keep_dims value was specified.
        >>> nonzero_num = ops.count_nonzero(x=x,  keep_dims=True)
        >>> print(nonzero_num)
        [[3]]
        >>> # case 6: keep_dims and axis value was specified.
        >>> nonzero_num = ops.count_nonzero(x=x, axis=[0,], keep_dims=True)
        >>> print(nonzero_num)
        [[1 2 0]]
    """

    const_utils.check_type_valid(F.dtype(x), mstype.number_type, 'input x')
    axis = _check_validate_axis(axis, "count_nonzero")
    keep_dims = _check_validate_keepdims(keep_dims, "count_nonzero")
    const_utils.check_type_valid(dtype, mstype.number_type + (mstype.bool_,), 'dtype')

    not_equal = P.NotEqual()
    cast = P.Cast()
    reduce_sum = P.ReduceSum(keep_dims)
    zeros = P.Zeros()
    tensor_0 = zeros(x.shape, x.dtype)
    nonzero_bool = not_equal(x, tensor_0)
    # ReduceSum only support float16 or float32 tensor.
    nonzero_val = cast(nonzero_bool, mstype.float32)
    nonzero_num = cast(reduce_sum(nonzero_val, axis), dtype)

    return nonzero_num


@constexpr
def _int_to_tuple_conv(axes):
    """
    Converts ints to tuples in input axes, expected by most validation checks.
    """
    for x in [0, 1]:
        if isinstance(axes[x], int):
            axes[x] = (axes[x],)
    return axes


@_primexpr
def _check_axes(axes, prim_name=None):
    """
    Check for validity and type of axes passed to function.
    """
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    validator.check_value_type('axes', axes, [int, tuple, list], "tensor dot")
    if not isinstance(axes, int):
        axes = list(axes)  # to avoid immutability issues
        if len(axes) != 2:
            raise ValueError(f"{msg_prefix} dimension of 'axes' must be 2, but got 'axes': {axes}.")
        axes = _int_to_tuple_conv(axes)  # convert before length checks
        if len(axes[0]) != len(axes[1]):
            raise ValueError(f"{msg_prefix} first and second dim of 'axes' have to be the same size/length, "
                             f"but got 'axes': {axes}.")
        if len(axes[0]) != len(set(axes[0])) or len(axes[1]) != len(set(axes[1])):
            raise ValueError(f"{msg_prefix} 'axes' cannot have duplicating values, but got {axes}.")
    return axes


@constexpr
def _typecheck_input(x1_type, x2_type, prim_name=None):
    """
    Check input tensor types to be valid and confirm they are the same type.
    """
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    const_utils.check_type_valid(x1_type, [mstype.float32, mstype.float16], 'x1')
    const_utils.check_type_valid(x2_type, [mstype.float32, mstype.float16], 'x2')
    if x1_type != x2_type:
        raise TypeError(f"{msg_prefix} inputs must be the same type, but got x1_type: {x1_type} "
                        f"and x2_type: {x2_type}.")


@_primexpr
def _axes_int_check(x1_shape, x2_shape, axes, prim_name=None):
    """
    Convert from single int axes to 2d tuple if required
    """
    if isinstance(axes, int):
        if axes == 0:
            # outer product, no input validation required
            return [], []
        x1_ind = tuple(range(len(x1_shape))[-1 * axes:])
        x2_ind = tuple(range(len(x2_shape))[:axes])
        axes = tuple((x1_ind, x2_ind))
        axes = _int_to_tuple_conv(axes)
    return axes


@_primexpr
def _calc_new_shape(shape, axes, position=0):
    """
    Calculate transpose and reshape parameters for input transformations,
    'position' refers to whether tensor is first or second in the op.
    """
    contraction_axes = tuple(i if i >= 0 else i + len(shape) for i in axes[position])
    prod_contraction = 1
    for i in contraction_axes:
        prod_contraction *= shape[i]
    free_axes = tuple(i for i in range(len(shape)) if i not in contraction_axes)
    free_dims = tuple(shape[i] if shape[i] is not None else -1 for i in free_axes)
    prod_free = 1
    for free_dim in free_dims:
        prod_free *= free_dim

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

    axes = 0 leads to outer product.
    axes = 1 leads to normal matrix multiplication when inputs both 2D.
    axes = 1 is the same as axes = ((1,),(0,)) where both `a` and `b` are 2D.
    axes = 2 is the same as axes = ((1,2),(0,1)) where both `a` and `b` are 3D.

    Args:
        x1 (Tensor): First tensor in tensor_dot with datatype float16 or float32
        x2 (Tensor): Second tensor in tensor_dot with datatype float16 or float32
        axes (Union[int, tuple(int), tuple(tuple(int)), list(list(int))]): Single value or
            tuple/list of length 2 with dimensions specified for `a` and `b` each. If single value `N` passed,
            automatically picks up last N dims from `a` input shape and first N dims from `b` input shape in order
            as axes for each respectively.

    Returns:
        Tensor, the shape of the output tensor is :math:`(N + M)`. Where :math:`N` and :math:`M` are the free axes not
        contracted in both inputs

    Raises:
        TypeError: If `x1` or `x2` is not a Tensor.
        TypeError: If `axes` is not one of the following: int, tuple, list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import mindspore
        >>> import numpy as np
        >>> input_x1 = Tensor(np.ones(shape=[1, 2, 3]), mindspore.float32)
        >>> input_x2 = Tensor(np.ones(shape=[3, 1, 2]), mindspore.float32)
        >>> output = ops.tensor_dot(input_x1, input_x2, ((0,1),(1,2)))
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
    axes = _check_axes(axes, 'tensor_dot')
    # input compatibility check & axes format update
    axes = _axes_int_check(x1_shape, x2_shape, axes, 'tensor_dot')
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
def _typecheck_input_dot(x1_type, x2_type, prim_name=None):
    """
    Check input tensor types to be valid and confirm they are the same type for dot and batch dot ops.
    """
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    const_utils.check_type_valid(x1_type, [mstype.float16, mstype.float32], 'x1')
    const_utils.check_type_valid(x2_type, [mstype.float16, mstype.float32], 'x2')
    if x1_type != x2_type:
        raise TypeError(f"{msg_prefix} inputs must be the same type, but got "
                        f"x1_type: {x1_type} and x2_type: {x2_type}.")


@_primexpr
def _get_transpose_shape(x2_shape):
    x2_shape_range = tuple(range(len(x2_shape)))
    x2_shape_transpose = x2_shape_range[-2:-1] + x2_shape_range[:-2] + x2_shape_range[-1:]
    return x2_shape_transpose


def dot(input, other):
    """
    Computation a dot product between samples in two tensors.

    Args:
        input (Tensor): First tensor in Dot op with datatype float16 or float32,
            The rank must be greater than or equal to 2.
        other (Tensor): Second tensor in Dot op with datatype float16 or float32,
            The rank must be greater than or equal to 2.

    Returns:
        Tensor, dot product of input and other.

    Raises:
        TypeError: If type of input and other are not the same.
        TypeError: If dtype of input or other is not float16 or float32.
        ValueError: If rank of input or other less than 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.ones(shape=[2, 3]), mindspore.float32)
        >>> other = Tensor(np.ones(shape=[1, 3, 2]), mindspore.float32)
        >>> output = ops.dot(input, other)
        >>> print(output)
        [[[3. 3.]]
         [[3. 3.]]]
        >>> print(output.shape)
        (2, 1, 2)
        >>> input = Tensor(np.ones(shape=[1, 2, 3]), mindspore.float32)
        >>> other = Tensor(np.ones(shape=[1, 3, 2]), mindspore.float32)
        >>> output = ops.dot(input, other)
        >>> print(output)
        [[[[3. 3.]]
          [[3. 3.]]]]
        >>> print(output.shape)
        (1, 2, 1, 2)
        >>> input = Tensor(np.ones(shape=[1, 2, 3]), mindspore.float32)
        >>> other = Tensor(np.ones(shape=[2, 3, 2]), mindspore.float32)
        >>> output = ops.dot(input, other)
        >>> print(output)
        [[[[3. 3.]
           [3. 3.]]
          [[3. 3.]
           [3. 3.]]]]
        >>> print(output.shape)
        (1, 2, 2, 2)
        >>> input = Tensor(np.ones(shape=[3, 2, 3]), mindspore.float32)
        >>> other = Tensor(np.ones(shape=[2, 1, 3, 2]), mindspore.float32)
        >>> output = ops.dot(input, other)
        >>> print(output)
        [[[[[3. 3.]]
           [[3. 3.]]]
          [[[3. 3.]]
           [[3. 3.]]]]
         [[[[3. 3.]]
           [[3. 3.]]]
          [[[3. 3.]]
           [[3. 3.]]]]
         [[[[3. 3.]]
           [[3. 3.]]]
          [[[3. 3.]]
           [[3. 3.]]]]]
        >>> print(output.shape)
        (3, 2, 2, 1, 2)
    """
    shape_op = P.Shape()
    reshape_op = P.Reshape()
    transpose_op = P.Transpose()
    matmul_op = P.MatMul(False, False)
    input_shape = shape_op(input)
    other_shape = shape_op(other)
    input_type = F.dtype(input)
    other_type = F.dtype(other)
    _typecheck_input_dot(input_type, other_type, 'dot')

    if len(input_shape) > 2 or len(other_shape) > 2:
        other_shape_transpose = _get_transpose_shape(other_shape)
        other_transpose = transpose_op(other, other_shape_transpose)
        input_reshape = reshape_op(input, (-1, input_shape[-1]))
        other_reshape = reshape_op(other_transpose, (other_shape[-2], -1))
        mul_result = matmul_op(input_reshape, other_reshape)
        reshape_shape = input_shape[:-1] + other_shape[:-2] + other_shape[-1:]
        reshape_shape = (-1,) + reshape_shape[1:]
        return reshape_op(mul_result, reshape_shape)
    return matmul_op(input, other)


@constexpr
def _get_batch_size(x1_shape, x2_shape, prim_name=None):
    """
    Get batch sizes from two inputs
    """
    return x1_shape[0], x2_shape[0]


@constexpr
def _typecheck_input_batch_dot(x1_type, x2_type, prim_name=None):
    """
    Check input tensor types to be valid and confirm they are the same type for batch dot ops.
    """
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    const_utils.check_type_valid(x1_type, [mstype.float32], 'x1')
    const_utils.check_type_valid(x2_type, [mstype.float32], 'x2')
    if x1_type != x2_type:
        raise TypeError(f"{msg_prefix} inputs must be the same type, but got x1_type: {x1_type} and "
                        f"x2_type: {x2_type}.")


@_primexpr
def _check_axes_for_batch_dot(x1_shape, x2_shape, axes, prim_name=None):
    """
    Check whether axes are valid and cast axes from tuple to list
    """
    if axes is None:
        if len(x2_shape) == 2:
            axes = [len(x1_shape) - 1, len(x2_shape) - 1]
        else:
            axes = [len(x1_shape) - 1, len(x2_shape) - 2]

    if isinstance(axes, (list, tuple)):
        if isinstance(axes, tuple):
            axes = list(axes)
        # Reverse if axis < 0
        if axes[0] < 0:
            axes[0] += len(x1_shape)
        if axes[1] < 0:
            axes[1] += len(x2_shape)
    elif isinstance(axes, int):
        if axes < 0:
            axes = [axes + len(x1_shape), axes + len(x2_shape)]
        else:
            axes = [axes, axes]
    return axes


@_primexpr
def _calc_new_shape_batchdot(shape, axes, position=0):
    """
    Calculate transpose and reshape parameters for input transformations,
    'position' refers to whether tensor is first or second in the op.
    """
    axis = axes[position]
    contraction_axes = tuple([axis])
    prod_contraction = 1
    for i in contraction_axes:
        prod_contraction *= shape[i]
    free_axes = tuple(i for i in range(1, len(shape)) if i not in contraction_axes)
    free_dims = tuple(shape[i] for i in free_axes)
    prod_free = 1
    for free_dim in free_dims:
        prod_free *= free_dim

    transpose_perm = contraction_axes + free_axes if position else free_axes + contraction_axes
    transpose_perm = tuple([0]) + transpose_perm
    new_shape = (prod_contraction, prod_free) if position else (prod_free, prod_contraction)
    new_shape = tuple([shape[0]]) + new_shape
    return new_shape, transpose_perm, free_dims


@_primexpr
def _check_batch_size(x1_batch_size, x2_batch_size, prim_name=None):
    """
    Check whether batch size of two inputs are the same
    """
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if x1_batch_size != x2_batch_size:
        raise ValueError(f"{msg_prefix} inputs 'x1', 'x2' should have the same batch sizes, but got "
                         f"'x1_batch_size': {x1_batch_size} and 'x2_batch_size': {x2_batch_size}.")


@_primexpr
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

    Args:
        x1 (Tensor): First tensor in Batch Dot op with datatype float32 and the rank of `x1` must be greater
          than or equal to 2.
        x2 (Tensor): Second tensor in Batch Dot op with datatype float32. The datatype of `x2` should
          be same as `x1` and the rank of `x2` must be greater than or equal to 2.
        axes (Union[int, tuple(int), list(int)]): Single value or tuple/list of length 2 with dimensions
          specified for `a` and `b` each. If single value `N` passed, automatically picks up last N dims from
          `a` input shape and last N dimensions from `b` input shape in order as axes for each respectively.
          Default: None.

    Returns:
        Tensor, batch dot product of `x1` and `x2`. For example, the Shape of output
        for input `x1` shapes (batch, d1, axes, d2) and `x2` shapes (batch, d3, axes, d4) is (batch, d1, d2, d3, d4),
        where d1 and d2 means any number.

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
        >>> from mindspore import Tensor, ops
        >>> import numpy as np
        >>> x1 = Tensor(np.ones(shape=[2, 2, 3]), mindspore.float32)
        >>> x2 = Tensor(np.ones(shape=[2, 3, 2]), mindspore.float32)
        >>> axes = (-1, -2)
        >>> output = ops.batch_dot(x1, x2, axes)
        >>> print(output)
        [[[3. 3.]
          [3. 3.]]
         [[3. 3.]
          [3. 3.]]]
        >>> x1 = Tensor(np.ones(shape=[2, 2]), mindspore.float32)
        >>> x2 = Tensor(np.ones(shape=[2, 3, 2]), mindspore.float32)
        >>> axes = (1, 2)
        >>> output = ops.batch_dot(x1, x2, axes)
        >>> print(output)
        [[2. 2. 2.]
         [2. 2. 2.]]
        >>> print(output.shape)
        (2, 3)
        >>> x1 = Tensor(np.ones(shape=[6, 2, 3, 4]), mindspore.float32)
        >>> x2 = Tensor(np.ones(shape=[6, 5, 4, 8]), mindspore.float32)
        >>> output = ops.batch_dot(x1, x2)
        >>> print(output.shape)
        (6, 2, 3, 5, 8)
        >>> x1 = Tensor(np.ones(shape=[2, 2, 4]), mindspore.float32)
        >>> x2 = Tensor(np.ones(shape=[2, 5, 4, 5]), mindspore.float32)
        >>> output = ops.batch_dot(x1, x2)
        >>> print(output.shape)
        (2, 2, 5, 5)

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

    x1_batch_size, x2_batch_size = _get_batch_size(x1_shape, x2_shape, 'batch_dot')

    _typecheck_input_batch_dot(x1_type, x2_type, 'batch_dot')
    _check_batch_size(x1_batch_size, x2_batch_size, 'batch_dot')
    axes = _check_axes_for_batch_dot(x1_shape, x2_shape, axes, 'batch_dot')

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
          The last dimension of `x1` must be the same size as the second last dimension of `x2`.
          And the shape of x1 and x2 could be broadcast.
        x2 (Tensor): Input tensor, scalar not allowed.
          The last dimension of `x1` must be the same size as the second last dimension of `x2`.
          And the shape of x1 and x2 could be broadcast.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the matrix product of the inputs. This is a scalar only
        when both `x1`, `x2` are 1-d vectors.

    Raises:
        ValueError: If the last dimension of `x1` is not the same size as the
            second-to-last dimension of `x2`, or if a scalar value is passed in.
        ValueError: If the shape of `x1` and `x2` could not broadcast together.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import mindspore
        >>> # case 1 : Reasonable application of broadcast mechanism
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
        >>> print(output.shape)
        (2, 3, 5)
        >>> # case 2 : the rank of `x1` is 1
        >>> x1 = Tensor(np.ones([1, 2]), mindspore.float32)
        >>> x2 = Tensor(np.ones([2,]), mindspore.float32)
        >>> output = ops.matmul(x1, x2)
        >>> print(output)
        [2.]
        >>> print(output.shape)
        (1,)
    """
    res = F.matmul(x1, x2)
    if dtype is not None:
        res = res.astype(dtype)
    return res


def mm(mat1, mat2):
    r"""
    Returns the matrix product of two arrays.
    If `mat1` is a :math:`(n \times m)` Tensor, `mat2` is a
    :math:`(m \times p)` Tensor, `out` will be a :math:`(n \times p)` Tensor.

    Note:
        This function cannot support broadcasting.
        Refer to :func:`mindspore.ops.matmul` instead if you need a broadcastable function.

    Args:
        mat1 (Tensor): The first matrix of matrix multiplication.
          The last dimension of `mat1` must be the same size as the first dimension of `mat2`.
        mat2 (Tensor): The second matrix of matrix multiplication.
          The last dimension of `mat1` must be the same size as the first dimension of `mat2`.

    Returns:
        Tensor or scalar, the matrix product of the inputs.

    Raises:
        ValueError: If the last dimension of `mat1` is not the same size as the
            second-to-last dimension of `mat2`.
        ValueError: If `mat1` or `mat2` is not a matrix.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.ops as ops
        >>> import numpy as np
        >>> x1 = ms.Tensor(np.random.rand(2, 3))
        >>> x2 = ms.Tensor(np.random.rand(3, 4))
        >>> out = ops.mm(x1, x2)
        >>> print(out.shape)
        (2, 4)
    """
    if mat1.ndim != 2 or mat2.ndim != 2:
        raise ValueError(f"For mm, the input tensor must be a matrix, "
                         f"but got mat1.ndim:{mat1.ndim}, mat2.ndim:{mat2.ndim}")
    return matmul(mat1, mat2)


def cummin(x, axis):
    r"""
    Returns a tuple (values,indices) where 'values' is the cumulative minimum value of input Tensor `x`
    along the dimension `axis`, and `indices` is the index location of each minimum value.

    .. math::
        \begin{array}{ll} \\
            y{i} = min(x{1}, x{2}, ... , x{i})
        \end{array}

    Args:
        x (Tensor): The input Tensor, rank of `x` > 0.
        axis (int): The dimension to do the operation over. The value of `axis` must be in the range
            `[-x.ndim, x.ndim - 1]`.

    Returns:
        tuple [Tensor], tuple of 2 Tensors, containing the cumulative minimum of elements and the index,
        The shape of each output tensor is the same as input `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `axis` is not an int.
        ValueError: If `axis` is out the range of `[-x.ndim, x.ndim - 1]`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import mindspore
        >>> a = Tensor([-0.2284, -0.6628,  0.0975,  0.2680, -1.3298, -0.4220], mindspore.float32)
        >>> output = ops.cummin(a, axis=0)
        >>> print(output[0])
        [-0.2284 -0.6628 -0.6628 -0.6628 -1.3298 -1.3298]
        >>> print(output[1])
        [0 1 1 1 4 4]
    """
    return cummin_(x, axis)
