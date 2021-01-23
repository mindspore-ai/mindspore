# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""math operations, the function docs are adapted from Numpy API."""
from ..ops import operations as P
from ..ops import functional as F
from ..ops import composite as C
from ..ops.primitive import constexpr
from ..common import dtype as mstype
from .array_ops import ravel
from .array_ops import where as where_
from .array_creations import asarray, full
from .utils import _is_scalar, _expand, _broadcast_to, _is_empty
from .utils_const import _infer_out_shape, _check_axis_valid, _get_device_compile, \
    _check_shape_aligned, _empty, _check_is_tensor, _raise_type_error, _check_same_type, \
    _check_is_float, _check_input_tensor
from .dtypes import nan


_mean_default = P.ReduceMean()
_mean_keepdims = P.ReduceMean(True)
_matmul = P.MatMul(False, False)
_matmul_T = P.MatMul(False, True)


def absolute(x, out=None, where=True, dtype=None):
    """
    Calculates the absolute value element-wise.

    Note:
        Numpy arguments casting, order, dtype, subok, signature, and extobj are
        not supported.
        When argument where is provided, argument out must have a tensor value.
        Argument out is not supported for storing the result, however it can be
        used in combination with argument where to set the value at indices for
        which where is set to False.
        Currently the backend kernel only supports float calculation, if the input
        is not a float, then it will be casted to float32 and casted back.

    Args:
        x (Tensor): Tensor to be used for calculation.
        out (Tensor or None): optional, defaults to None.
        where (Tensor or None): optional. For any non-default value of type other
            than Tensor or None, the output retains its original value.
            This condition is broadcasted over the input. At locations where the
            condition is True, the out array will be set to the ufunc result.
            Elsewhere, the out array will retain its original value. Note that
            if an uninitialized out array is created via the default out=None,
            locations within it where the condition is False will remain
            uninitialized.
        dtype (data type): optional, defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor.

    Raises:
        TypeError: If input arguments have types not specified above.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.asarray([1, 2, 3, -4, -5], np.float32)
        >>> output = np.absolute(x)
        >>> print(output)
        [1. 2. 3. 4. 5.]
    """
    if not _check_is_tensor(F.typeof(x)):
        _raise_type_error("Input is expected to be a tensor, but got ", x)
    original_dtype = x.dtype
    if not _check_is_float(original_dtype) and dtype is None:
        x = x.astype(mstype.float32)
        return _apply_tensor_op(F.absolute, x, out=out, where=where, dtype=dtype).astype(original_dtype)
    return _apply_tensor_op(F.absolute, x, out=out, where=where, dtype=dtype)


def add(x1, x2, out=None, where=True, dtype=None):
    """
    Adds arguments element-wise.

    Note:
        Numpy arguments casting, order, dtype, subok, signature, and extobj are
        not supported.
        When argument where is provided, argument out must have a tensor value.
        Argument out is not supported for storing the result, however it can be
        used in combination with argument where to set the value at indices for
        which where is set to False.

    Args:
        x1 (Tensor): input to be added.
        x2 (Tensor): input to be added.
        out (Tensor or None): optional, defaults to None.
        where (Tensor or None): optional. For any non-default value of type other
            than Tensor or None, the output retains its original value.
            This condition is broadcast over the input. At locations where the
            condition is True, the out array will be set to the ufunc result.
            Elsewhere, the out array will retain its original value. Note that
            if an uninitialized out array is created via the default out=None,
            locations within it where the condition is False will remain
            uninitialized.
        dtype (data type): optional, defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the sum of x1 and x2, element-wise. This is a scalar
        if both x1 and x2 are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = np.full((3, 2), [1, 2])
        >>> x2 = np.full((3, 2), [3, 4])
        >>> output = np.add(x1, x2)
        >>> print(output)
        [[4, 6],
        [4, 6],
        [4, 6]]
    """
    # broadcast is not fully supported in tensor_add on CPU,
    # so we use tensor_sub as a substitute solution
    if _get_device_compile() == 'CPU':
        return subtract(x1, F.neg_tensor(x2), out=out, where=where, dtype=dtype)
    return _apply_tensor_op(F.tensor_add, x1, x2, out=out, where=where, dtype=dtype)


def subtract(x1, x2, out=None, where=True, dtype=None):
    """
    Subtracts arguments, element-wise.

    Note:
        Numpy arguments casting, order, dtype, subok, signature, and extobj are
        not supported.
        When argument where is provided, argument out must have a tensor value.
        Argument out is not supported for storing the result, however it can be
        used in combination with argument where to set the value at indices for
        which where is set to False.

    Args:
        x1 (Tensor): the input to be subtracted from.
        x2 (Tensor): the input to be subtracted by.
        out (Tensor or None): optional, defaults to None.
        where (Tensor or None): optional. For any non-default value of type other
            than Tensor or None, the output retains its original value.
            This condition is broadcast over the input. At locations where the
            condition is True, the out array will be set to the ufunc result.
            Elsewhere, the out array will retain its original value. Note that
            if an uninitialized out array is created via the default out=None,
            locations within it where the condition is False will remain
            uninitialized.
        dtype (data type): optional, defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the difference of x1 and x2, element-wise. This is a
        scalar if both x1 and x2 are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = np.full((3, 2), [1, 2])
        >>> x2 = np.full((3, 2), [3, 4])
        >>> output = np.subtract(x1, x2)
        >>> print(output)
        [[-2, -2],
        [-2, -2],
        [-2, -2]]
    """
    return _apply_tensor_op(F.tensor_sub, x1, x2, out=out, where=where, dtype=dtype)


def multiply(x1, x2, out=None, where=True, dtype=None):
    """
    Multiplies arguments element-wise.

    Note:
        Numpy arguments casting, order, dtype, subok, signature, and extobj are
        not supported.
        When argument where is provided, argument out must have a tensor value.
        Argument out is not supported for storing the result, however it can be
        used in combination with argument where to set the value at indices for
        which where is set to False.

    Args:
        x1 (Tensor): input tensor to be multiplied.
        x2 (Tensor): input tensor to be multiplied.
        out (Tensor or None): optional, defaults to None.
        where (Tensor or None): optional. For any non-default value of type other
            than Tensor or None, the output retains its original value.
            This condition is broadcast over the input. At locations where the
            condition is True, the out array will be set to the ufunc result.
            Elsewhere, the out array will retain its original value. Note that
            if an uninitialized out array is created via the default out=None,
            locations within it where the condition is False will remain
            uninitialized.
        dtype (data type): optional, defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the product of x1 and x2, element-wise. This is a scalar
        if both x1 and x2 are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = np.full((3, 2), [1, 2])
        >>> x2 = np.full((3, 2), [3, 4])
        >>> output = np.multiply(x1, x2)
        >>> print(output)
        [[3, 8],
        [3, 8],
        [3, 8]]
    """
    if _get_device_compile() == 'CPU':
        # broadcast is not fully supported on CPU backend,
        # and explicit broadcasting is performed
        shape_out = _infer_out_shape(F.shape(x1), F.shape(x2))
        ndim_out = F.tuple_len(shape_out)
        x1 = _expand(x1, ndim_out)
        x2 = _expand(x2, ndim_out)
        x1 = _broadcast_to(x1, F.shape(x1), shape_out, ndim_out)
        x2 = _broadcast_to(x2, F.shape(x2), shape_out, ndim_out)
    return _apply_tensor_op(F.tensor_mul, x1, x2, out=out, where=where, dtype=dtype)


def divide(x1, x2, out=None, where=True, dtype=None):
    """
    Returns a true division of the inputs, element-wise.

    Instead of the Python traditional ‘floor division’, this returns a true
    division.

    Note:
        Numpy arguments casting, order, dtype, subok, signature, and extobj are
        not supported.
        When argument where is provided, argument out must have a tensor value.
        Argument out is not supported for storing the result, however it can be
        used in combination with argument where to set the value at indices for
        which where is set to False.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        x1 (Tensor): the divident.
        x2 (Tensor): the divisor.
        out (Tensor or None): optional, defaults to None.
        where (Tensor or None): optional. For any non-default value of type other
            than Tensor or None, the output retains its original value.
            This condition is broadcast over the input. At locations where the
            condition is True, the out array will be set to the ufunc result.
            Elsewhere, the out array will retain its original value. Note that
            if an uninitialized out array is created via the default out=None,
            locations within it where the condition is False will remain
            uninitialized.
        dtype (data type): optional, defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, this is a scalar if both x1 and x2 are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = np.full((3, 2), [1, 2])
        >>> x2 = np.full((3, 2), [3, 4])
        >>> output = np.divide(x1, x2)
        >>> print(output)
        [[0.33333333, 0.5],
        [0.33333333, 0.5],
        [0.33333333, 0.5]]
    """
    if not _check_is_float(F.dtype(x1)) and not _check_is_float(F.dtype(x2)):
        x1 = F.cast(x1, mstype.float32)
        x2 = F.cast(x2, mstype.float32)
    return _apply_tensor_op(F.tensor_div, x1, x2, out=out, where=where, dtype=dtype)


def power(x1, x2, out=None, where=True, dtype=None):
    """
    First array elements raised to powers from second array, element-wise.

    Raises each base in x1 to the positionally-corresponding power in x2.

    Note:
        Numpy arguments casting, order, dtype, subok, signature, and extobj are
        not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        x1 (Tensor): the bases.
        x2 (Tensor): the exponents.
        out (Tensor or None): optional, defaults to None.
        where (Tensor or None): optional. For any non-default value of type other
            than Tensor or None, the output retains its original value.
            This condition is broadcast over the input. At locations where the
            condition is True, the out array will be set to the ufunc result.
            Elsewhere, the out array will retain its original value. Note that
            if an uninitialized out array is created via the default out=None,
            locations within it where the condition is False will remain
            uninitialized.
        dtype (data type): optional, defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the bases in x1 raised to the exponents in x2. This
        is a scalar if both x1 and x2 are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = np.full((3, 2), [1, 2]).astype('float32')
        >>> x2 = np.full((3, 2), [3, 4]).astype('float32')
        >>> output = np.power(x1, x2)
        >>> print(output)
        [[ 1, 16],
        [ 1, 16],
        [ 1, 16]]
    """
    return _apply_tensor_op(F.tensor_pow, x1, x2, out=out, where=where, dtype=dtype)


def mean(a, axis=None, keepdims=False):
    """
    Computes the arithmetic mean along the specified axis.

    Returns the average of the array elements. The average is taken
    over the flattened array by default, otherwise over the specified
    axis.

    Note:
        Numpy arguments dtype and out are not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.
        On CPU, the supported dtypes are np.float16, np.float32, and
        np.float64.

    Args:
        a (Tensor): input tensor containing numbers whose mean is desired.
                    If a is not an array, a conversion is attempted.
        axis (None or int or tuple of ints): optional. Axis or axes along
                    which the means are computed. The default is to compute
                    the mean  of the flattened array. If this is a tuple of
                    ints, a mean is performed over multiple axes.
        keepdims(bool): optional. If this is set to True, the axes which
                    are reduced are left in the result as dimensions with
                    size one. With this option, the result will broadcast
                    correctly against the input tensor.

    Returns:
        Tensor or scalar, an array containing the mean values.

    Raises:
        ValueError: if axes are out of the range of [-a.ndim, a.ndim), or
        if the axes contain duplicates.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.arange(6, dtype='float32')
        >>> output = np.mean(a, 0)
        >>> print(output)
        2.5
    """

    axis = _check_axis_valid(axis, F.rank(a))
    shape_a = F.shape(a)

    if _is_empty(shape_a):
        if keepdims:
            shape_out = _shape_reduced_keepdims(shape_a, axis)
        else:
            shape_out = _shape_reduced(shape_a, axis)
        if _is_empty(shape_out):
            return _empty(F.dtype(a), shape_out)
        return _full_compile(shape_out, nan)

    if _is_scalar(shape_a):
        if keepdims:
            return a
        shape_out = _shape_reduced(shape_a, axis)
        return F.reshape(a, shape_out)

    if keepdims:
        res = _mean_keepdims(a, axis)
    else:
        res = _mean_default(a, axis)
    return res


def inner(a, b):
    """
    Inner product of two tensors.

    Ordinary inner product of vectors for 1-D tensors (without complex
    conjugation), in higher dimensions a sum product over the last
    axes.

    Note:
        Numpy argument out is not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.
        On CPU, the supported dtypes are np.float16, np.float32, and
        np.float64.

    Args:
        a (Tensor): input tensor. If a and b are nonscalar, their last
                    dimensions must match.
        b (Tensor): input tensor. If a and b are nonscalar, their last
                    dimensions must match.

    Returns:
        Tensor or scalar, out.shape = a.shape[:-1] + b.shape[:-1].

    Raises:
        ValueError: if x1.shape[-1] != x2.shape[-1].

    Supported Platforms:
        Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((5, 3))
        >>> b = np.ones((2, 7, 3))
        >>> output = np.inner(a, b)
        >>> print(output)
        [[[3. 3. 3. 3. 3. 3. 3.]
        [3. 3. 3. 3. 3. 3. 3.]]

        [[3. 3. 3. 3. 3. 3. 3.]
        [3. 3. 3. 3. 3. 3. 3.]]

        [[3. 3. 3. 3. 3. 3. 3.]
        [3. 3. 3. 3. 3. 3. 3.]]

        [[3. 3. 3. 3. 3. 3. 3.]
        [3. 3. 3. 3. 3. 3. 3.]]

        [[3. 3. 3. 3. 3. 3. 3.]
        [3. 3. 3. 3. 3. 3. 3.]]]
    """
    if F.rank(a) == 0 or F.rank(b) == 0:
        a = _expand(a, 1)
        b = _expand(b, 1)
        if F.rank(a) < F.rank(b):
            a, b = b, a
        return F.tensor_mul(a, b)

    _ = _check_shape_aligned(F.shape(a), F.shape(b))
    aligned_shape_a = (F.shape_mul(F.shape(a)[:-1]), F.shape(a)[-1])
    aligned_shape_b = (F.shape_mul(F.shape(b)[:-1]), F.shape(a)[-1])
    a_aligned = F.reshape(a, aligned_shape_a)
    b_aligned = F.reshape(b, aligned_shape_b)

    res = _matmul_T(a_aligned, b_aligned)
    res = F.reshape(res, F.shape(a)[:-1] + F.shape(b)[:-1])
    return res


@constexpr
def _nan():
    """Returns a Tensor with nan value"""
    return asarray(float('nan'))


def dot(a, b):
    """
    Dot product of two arrays.

    Specifically,
    If both a and b are 1-D arrays, it is inner product of vectors
    (without complex conjugation).
    If both a and b are 2-D arrays, it is matrix multiplication.
    If either a or b is 0-D (scalar), it is equivalent to multiply.
    If a is an N-D array and b is a 1-D array, it is a sum product
    over the last axis of a and b.
    If a is an N-D array and b is an M-D array (where M>=2), it is a
    sum product over the last axis of a and the second-to-last axis of b:
    dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    Note:
        Numpy argument out is not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.
        On CPU, the supported dtypes are np.float16, np.float32, and
        np.float64.

    Args:
        a (Tensor): input tensor
        b (Tensor): input tensor

    Returns:
        Tensor or scalar, the dot product of a and b. If a and b are
        both scalars or both 1-D arrays then a scalar is returned;
        otherwise an array is returned

    Raises:
        ValueError: If the last dimension of a is not the same size
        as the second-to-last dimension of b.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.full((1, 3), 7).astype('float32')
        >>> b = np.full((2, 3, 4), 5).astype('float32')
        >>> output = np.dot(a, b)
        >>> print(output)
        [[[105, 105, 105, 105],
        [105, 105, 105, 105]]]
    """
    ndim_a, ndim_b = F.rank(a), F.rank(b)
    if ndim_a > 0 and ndim_b >= 2:
        perm = F.make_range(ndim_b)
        perm = perm[:-2] + (perm[-1],) + (perm[-2],)
        b = F.transpose(b, perm)
    return inner(a, b)


def outer(a, b):
    """
    Computes the outer product of two vectors.

    Given two vectors, a = [a0, a1, ..., aM] and b = [b0, b1, ..., bN],
    the outer product [1] is:
    [[a0*b0  a0*b1 ... a0*bN ]
    [a1*b0    .
    [ ...          .
    [aM*b0            aM*bN ]]

    Note:
        Numpy argument out is not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.
        On CPU, the supported dtypes are np.float16, np.float32, and
        np.float64.

    Args:
        a (Tensor): first input vector. Input is flattened if not
                    already 1-dimensional.
        b (Tensor): second input vector. Input is flattened if not
                    already 1-dimensional.

    Returns:
        Tensor or scalar, out[i, j] = a[i] * b[j].

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.full(7, 2).astype('float32')
        >>> b = np.full(4, 3).astype('float32')
        >>> output = np.outer(a, b)
        >>> print(output)
        [[6, 6, 6, 6],
        [6, 6, 6, 6],
        [6, 6, 6, 6],
        [6, 6, 6, 6],
        [6, 6, 6, 6],
        [6, 6, 6, 6],
        [6, 6, 6, 6]]
    """
    _check_input_tensor(F.typeof(a))
    _check_input_tensor(F.typeof(b))

    if F.rank(a) != 1:
        a = ravel(a)
    if F.rank(b) != 1:
        b = ravel(b)
    a = F.reshape(a, (F.shape(a)[0], 1))
    b = _expand(b, 2)
    return _matmul(a, b)


def tensordot(a, b, axes=2):
    """
    Computes tensor dot product along specified axes.

    Given two tensors, a and b, and an array_like object containing two array_like
    objects, (a_axes, b_axes), sum the products of a’s and b’s elements (components)
    over the axes specified by a_axes and b_axes. The third argument can be a single
    non-negative integer_like scalar, N; if it is such, then the last N dimensions of
    a and the first N dimensions of b are summed over.
    Three common use cases are:
        axes = 0 : tensor product
        axes = 1 : tensor dot product
        axes = 2 : (default) tensor double contraction
    When axes is integer_like, the sequence for evaluation will be: first the -Nth
    axis in a and 0th axis in b, and the -1th axis in a and Nth axis in b last.
    When there is more than one axis to sum over - and they are not the last (first)
    axes of a (b) - the argument axes should consist of two sequences of the same
    length, with the first axis to sum over given first in both sequences, the second
    axis second, and so forth.
    The shape of the result consists of the non-contracted axes of the first tensor,
    followed by the non-contracted axes of the second.

    Note:
        On CPU, the supported dypes are np.float16 and np.float32.
        On GPU, the supported dypes are np.float16 and np.float32.

    Args:
        a, b (Tensor): Tensors to “dot”.
        axes (int or (2,) array_like):
            integer_like: If an int N, sum over the last N axes of a and the first N
            axes of b in order. The sizes of the corresponding axes must match.
            (2,) array_like: Or, a list of axes to be summed over, first sequence
            applying to a, second to b. Both elements array_like must be of the same
            length.

    Returns:
        Tensor, or list of tensors, the tensor dot product of the input.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = np.ones((3, 4, 5))
        >>> b = np.ones((4, 3, 2))
        >>> output = np.tensordot(a, b, axes=([1,0],[0,1]))
        >>> print(output.shape)
        (5, 2)
    """
    _check_input_tensor(F.typeof(a))
    _check_input_tensor(F.typeof(b))

    if F.rank(a)*F.rank(b) == 0 and axes == 0:
        return F.tensor_mul(a, b)
    return C.tensor_dot(a, b, axes)


@constexpr
def _full_compile(shape, value):
    return full(shape, value)


@constexpr
def _shape_reduced_keepdims(shape, axes):
    """
    Reduces dimensions corresponding to argument axes while
    keeping the number of dimensions unchanged.
    """
    ndim_out = F.tuple_len(shape)
    shape_out = [1]*ndim_out
    for i in range(ndim_out):
        if not i in axes:
            shape_out[i] = shape[i]
    return tuple(shape_out)


@constexpr
def _shape_reduced(shape, axes):
    """Removes dimensions corresponding to argument axes"""
    ndim_orig = F.tuple_len(shape)
    ndim_out = ndim_orig - F.tuple_len(axes)
    shape_out = [0]*ndim_out
    idx_out = 0
    for i in range(ndim_orig):
        if not i in axes:
            shape_out[idx_out] = shape[i]
            idx_out += 1
    return tuple(shape_out)


def _infer_shape_rem(shape1, shape2, ndim1, ndim2, transpose_b):
    """Infers the shape of the last two dimensions after performing matmul."""
    shape_rem = ()
    if ndim1 >= 2:
        shape_rem += (shape1[-2],)
    if transpose_b:
        if ndim2 >= 2:
            shape_rem += (shape2[-2],)
    else:
        if ndim1 >= 1:
            shape_rem += (shape2[-1],)
    return shape_rem


def _apply_tensor_op(fn, *args, out=None, where=True, dtype=None):
    """applies tensor operations based on fn"""
    for arg in args:
        _check_input_tensor(F.typeof(arg))
    res = fn(*args)

    # if out is set to a non-default value, return tensor will have the same
    # dtype as out, which overrides the dtype passed into the keyword argument
    if _check_is_tensor(F.typeof(out)):
        dtype_out = F.dtype(out)
    elif dtype is not None:
        dtype_out = dtype
    else:
        dtype_out = F.dtype(res)

    if  _check_is_tensor(F.typeof(out)) and _check_is_tensor(F.typeof(where)):
        out = where_(where, res, out)
    elif out is None or where is not None:
        out = res

    if not _check_same_type(F.dtype(out), dtype_out):
        out = F.cast(out, dtype_out)

    return out
