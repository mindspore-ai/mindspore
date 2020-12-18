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
"""math operations, the function docs are adapted from Numpy API."""
from ..ops import operations as P
from ..ops import functional as F
from ..ops.primitive import constexpr
from .array_ops import squeeze, asarray
from .utils import _infer_out_shape, _is_scalar, _check_axis_valid, _get_device_compile, \
    _check_shape_aligned


def mean(a, axis=None, keepdims=False):
    """
    Computes the arithmetic mean along the specified axis.

    Returns the average of the array elements. The average is taken
    over the flattened array by default, otherwise over the specified
    axis.

    Note:
        Numpy arguments dtype and out are not supported.
        On GPU, the supported dtypes are mstype.float16, and mstype.float32.
        On CPU, the supported dtypes are mstype.float16, and mstype.float32.

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
    axis = _check_axis_valid(axis, P.Rank()(a))
    if _is_empty(F.shape(a)):
        return _nan()
    if _is_scalar(a.shape):
        if keepdims:
            return a
        return squeeze(a)
    if keepdims:
        res = P.ReduceMean(True)(a, axis)
    else:
        res = P.ReduceMean(False)(a, axis)
    return res


def inner(a, b):
    """
    Inner product of two tensors.

    Ordinary inner product of vectors for 1-D tensors (without complex
    conjugation), in higher dimensions a sum product over the last
    axes.

    Note:
        Numpy argument out is not supported.
        On GPU, the supported dtypes are mstype.float16, and mstype.float32.
        On CPU, the supported dtype is mstype.float32.

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
    if P.Rank()(a) == 0 or P.Rank()(b) == 0:
        if _is_scalar(a.shape):
            a, b = b, a
        return _apply_bin_op(P.Mul(), a, b)

    _ = _check_shape_aligned(a.shape, b.shape)
    aligned_shape_a = (F.shape_mul(a.shape[:-1]), a.shape[-1])
    aligned_shape_b = (F.shape_mul(b.shape[:-1]), a.shape[-1])
    a_aligned = P.Reshape()(a, aligned_shape_a)
    b_aligned = P.Reshape()(b, aligned_shape_b)

    res = P.MatMul(False, True)(a_aligned, b_aligned)
    res = P.Reshape()(res, a.shape[:-1] + b.shape[:-1])
    return res


@constexpr
def _nan():
    """Returns a Tensor with nan value"""
    return asarray(float('nan'))


def _is_empty(shape):
    """Checks if the shape is empty"""
    return F.shape_mul(shape) == 0


def _expand(x, ndim):
    """Expand x to ndim"""
    while P.Rank()(x) < ndim:
        x = P.ExpandDims()(x, 0)
    return x


def _apply_bin_op(fn, x1, x2):
    """apply binary operations based on fn."""
    device = _get_device_compile()
    out_shape = _infer_out_shape(device, x1.shape, x2.shape)
    if device == 'CPU':
        # built-in operations on CPU does not support operands with
        # dimensions of size 1 or with shape 0, therefore squeeze
        # and scalar promotion is performed
        x1, x2 = squeeze(x1), squeeze(x2)
        x1, x2 = _expand(x1, 1), _expand(x2, 1)
    res = fn(x1, x2)
    res = P.Reshape()(res, out_shape)
    return res
