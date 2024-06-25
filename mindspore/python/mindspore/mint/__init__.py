# Copyright 2024 Huawei Technologies Co., Ltd
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
"""mint module."""
from __future__ import absolute_import
from mindspore.ops.extend import gather, conv2d, max, min
from mindspore.ops.extend import array_func, math_func, nn_func
from mindspore.mint.nn.functional import *
from mindspore.mint.nn import functional
from mindspore.mint import linalg
from mindspore.ops import erf, where, triu
from mindspore.ops.function.math_func import linspace_ext as linspace
from mindspore.ops.function.array_func import full_ext as full
from mindspore.ops.function.array_func import ones_like_ext as ones_like
from mindspore.ops.function.array_func import zeros_like_ext as zeros_like
from mindspore.ops.function.array_func import unique_ext as unique
from mindspore.ops.function.math_func import isclose
from mindspore.ops.auto_generate import abs
# 1
from mindspore.ops.function.math_func import divide, div
from mindspore.ops.auto_generate import topk_ext as topk
# 2
from mindspore.ops.function.math_func import sin
# 3
from mindspore.ops.function.clip_func import clamp
# 4

# 5
from mindspore.ops.auto_generate import cumsum_ext as cumsum
# 6
from mindspore.ops.auto_generate import stack_ext as stack

# 7
from mindspore.ops.auto_generate import ones as ones_ext
from mindspore.ops.auto_generate import zeros as zeros_ext
# 8

# 9

# 10
from mindspore.ops.function.math_func import ne
# 11

# 12
from mindspore.ops.function.array_func import repeat_interleave_ext as repeat_interleave
# 13
from mindspore.ops.functional import flip
# 14

# 15
from mindspore.ops.auto_generate import flatten_ext as flatten
# 16
from mindspore.ops.functional import matmul
from mindspore.ops.auto_generate import bmm_ext as bmm
# 17
from mindspore.ops.functional import mean_ext
# 18
from mindspore.ops.functional import sum
# 19
from mindspore.ops.functional import log
# 20
from mindspore.ops.functional import prod
# 21
from mindspore.ops.functional import mul
# 22

# 23

# 24

# 25
from mindspore.ops.functional import greater, gt
# 26
from mindspore.ops.functional import eq
# 27
from mindspore.ops.functional import reciprocal
# 28
from mindspore.ops.functional import exp
# 29
from mindspore.ops.auto_generate import sqrt as sqrt_ext
# 30
from mindspore.ops.functional import searchsorted
# 31

# 32
from mindspore.ops.auto_generate import sub_ext
# 33
from mindspore.ops.function.array_func import split_ext
# 34

# 35
from mindspore.ops.functional import erfinv
# 36

# 37
from mindspore.ops.function.array_func import nonzero
# 38

# 39

# 40
from mindspore.ops.functional import any as any_ext

# 41
from mindspore.ops.auto_generate import add_ext

# 42
from mindspore.ops.functional import argmax_ext as argmax
# 43
from mindspore.ops.auto_generate import cat as cat_ext

# 44
from mindspore.ops.functional import cos
# 45

# 46

# 47

# 48

# 49

# 50
from mindspore.ops.functional import tile
# 51
from mindspore.ops.functional import permute as permute_ext
# 52

# 53

# 54
from mindspore.ops import normal_ext as normal
# 55

# 56

# 57
from mindspore.ops.functional import broadcast_to
# 58
from mindspore.ops.function.math_func import greater_equal
# 59
from mindspore.ops.functional import square
# 60
from mindspore.ops.function.math_func import all

# 61
from mindspore.ops.functional import rsqrt
# 62
from mindspore.ops.functional import maximum
# 63
from mindspore.ops.functional import minimum
# 64

# 65
from mindspore.ops.functional import logical_and
# 66
from mindspore.ops.functional import logical_not
# 67
from mindspore.ops.functional import logical_or
# 68

# 69
from mindspore.ops.functional import less_equal, le
# 70
from mindspore.ops.functional import negative, neg
# 71
from mindspore.ops.functional import isfinite
# 72

# 73
from mindspore.ops.functional import ceil
# 74
from mindspore.ops.function.array_func import sort_ext as sort
# 75
from mindspore.ops.functional import less, lt
# 76
from mindspore.ops.functional import pow
# 77

# 78
from mindspore.ops.function import arange_ext as arange
# 79

# 80

# 81
from mindspore.ops.function.array_func import index_select_ext as index_select
# 82

# 83
from mindspore.ops.function.array_func import narrow_ext as narrow
# 84

# 85
from mindspore.mint import nn, optim
# 86

# 87

# 88
from mindspore.ops.function.array_func import chunk_ext as chunk
# 89

# 90

# 91

# 92

# 93

# 94
from mindspore.ops.function.math_func import tanh
# 95

# 96

# 97

# 98

# 99

# 100

# 122

# 176
from mindspore.ops.function.math_func import atan2_ext as atan2
from mindspore.ops.function.math_func import arctan2_ext as arctan2


# 208
from mindspore.ops.function.array_func import eye
from mindspore.ops import rand_ext as rand
from mindspore.ops import rand_like_ext as rand_like
# 210
from mindspore.ops.auto_generate import floor
# 231
from mindspore.ops.function.math_func import inverse_ext as inverse

# 285
from mindspore.ops.function.array_func import scatter_add_ext as scatter_add


def add(input, other, *, alpha=1):
    r"""
    Adds scaled other value to input Tensor.

    .. math::

        out_{i} = input_{i} + alpha \times other_{i}

    Note:
        - When the two inputs have different shapes,
          they must be able to broadcast to a common shape.
        - The two inputs and alpha comply with the implicit type conversion rules to make the data types
          consistent.

    Args:
        input (Union[Tensor, number.Number, bool]): The first input is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.
        other (Union[Tensor, number.Number, bool]): The second input, is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.

    Keyword Args:
        alpha (number.Number): A scaling factor applied to `other`, default 1.

    Returns:
        Tensor, the shape is the same as the one of the input `input`, `other` after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs and alpha.

    Raises:
        TypeError: If the type of `input`, `other`, or `alpha` is not one of the following: Tensor, number.Number, bool.
        TypeError: If `alpha` is of type float but `input` and `other` are not of type float.
        TypeError: If `alpha` is of type bool but `input` and `other` are not of type bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore import mint
        >>> x = Tensor(1, mindspore.int32)
        >>> y = Tensor(np.array([4, 5, 6]).astype(np.float32))
        >>> alpha = 0.5
        >>> output = mint.add(x, y, alpha)
        >>> print(output)
        [3. 3.5 4.]
        >>> # the data type of x is int32, the data type of y is float32,
        >>> # alpha is a float, and the output is the data format of higher precision float32.
        >>> print(output.dtype)
        Float32
    """
    return add_ext(input, other, alpha)


def any(input, dim=None, keepdim=False):
    r"""
    Reduces a dimension of `input` by the "logical OR" of all elements in the dimension, by default. And also can
    reduce a dimension of `input` along the `dim`. Determine whether the dimensions of the output and input are the
    same by controlling `keep_dims`.

    Note:
        The `dim` with tensor type is only used for compatibility with older versions and is not recommended.

    Args:
        input (Tensor): Input Tensor, has the shape :math:`(N, *)` where :math:`*` means,
            any number of additional dimensions.
        dim (Union[int, tuple(int), list(int), Tensor], optional): The dimensions to reduce.
            Suppose the rank of `input` is r, `dim` must be in the range [-rank(input), rank(input)).
            Default: ``None`` , all dimensions are reduced.
        keep_dims (bool, optional): If ``True`` , keep these reduced dimensions and the length is 1.
            If ``False`` , don't keep these dimensions. Default : ``False`` .

    Returns:
        Tensor, the dtype is bool.

        - If `dim` is ``None`` , and `keep_dims` is ``False`` ,
          the output is a 0-D Tensor representing the "logical OR" of all elements in the input Tensor.
        - If `dim` is int, such as 2, and `keep_dims` is ``False`` ,
          the shape of output is :math:`(input_1, input_3, ..., input_R)`.
        - If `dim` is tuple(int), such as (2, 3), and `keep_dims` is ``False`` ,
          the shape of output is :math:`(input_1, input_4, ..., input_R)`.
        - If `dim` is 1-D Tensor, such as [2, 3], and `keep_dims` is ``False`` ,
          the shape of output is :math:`(input_1, input_4, ..., input_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `input` is not a Tensor.
        TypeError: If `dim` is not one of the following: int, tuple, list or Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> x = Tensor(np.array([[True, False], [True, True]]))
        >>> # case 1: Reduces a dimension by the "logical OR" of all elements in the dimension.
        >>> output = mint.any(x, keep_dims=True)
        >>> print(output)
        [[ True]]
        >>> print(output.shape)
        (1, 1)
        >>> # case 2: Reduces a dimension along dim 0.
        >>> output = mint.any(x, dim=0)
        >>> print(output)
        [ True True]
        >>> # case 3: Reduces a dimension along dim 1.
        >>> output = mint.any(x, dim=1)
        >>> print(output)
        [ True True]
    """
    return any_ext(input, dim, keepdim)


def cat(tensors, dim=0):
    r"""
    Connect input tensors along with the given dimension.

    The input data is a tuple or a list of tensors. These tensors have the same rank :math:`R`.
    Set the given dimension as :math:`m`, and :math:`0 \le m < R`. Set the number of input tensors as :math:`N`.
    For the :math:`i`-th tensor :math:`t_i`, it has the shape of :math:`(x_1, x_2, ..., x_{mi}, ..., x_R)`.
    :math:`x_{mi}` is the :math:`m`-th dimension of the :math:`t_i`. Then, the shape of the output tensor is

    .. math::

        (x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)

    Args:
        tensors (Union[tuple, list]): A tuple or a list of input tensors.
            Suppose there are two tensors in this tuple or list, namely t1 and t2.
            To perform `concat` in the dimension 0 direction, except for the :math:`0`-th dimension,
            all other dimensions should be equal, that is,
            :math:`t1.shape[1] = t2.shape[1], t1.shape[2] = t2.shape[2], ..., t1.shape[R-1] = t2.shape[R-1]`,
            where :math:`R` represents the rank of tensor.
        dim (int): The specified dimension, whose value is in range :math:`[-R, R)`. Default: ``0`` .

    Returns:
        Tensor, the shape is :math:`(x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)`.
        The data type is the same with `tensors`.

    Raises:
        TypeError: If `dim` is not an int.
        ValueError: If `tensors` have different dimension of tensor.
        ValueError: If `dim` not in range :math:`[-R, R)`.
        ValueError: If tensor's shape in `tensors` except for `dim` are different.
        ValueError: If `tensors` is an empty tuple or list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import mint
        >>> input_x1 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
        >>> input_x2 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
        >>> output = mint.cat((input_x1, input_x2))
        >>> print(output)
        [[0. 1.]
         [2. 1.]
         [0. 1.]
         [2. 1.]]
        >>> output = mint.cat((input_x1, input_x2), 1)
        >>> print(output)
        [[0. 1. 0. 1.]
         [2. 1. 2. 1.]]
    """
    return cat_ext(tensors, dim)


def mean(input, dim=None, keepdim=False, *, dtype=None):
    r"""
    Reduces all dimension of a tensor by averaging all elements in the dimension, by default.
    And reduce a dimension of `input` along the specified `dim`. `keepdim`
    determines whether the dimensions of the output and input are the same.

    Note:
        The `dim` with tensor type is only used for compatibility with older versions and is not recommended.

    Args:
        input (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
            :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        dim (Union[int, tuple(int), list(int), Tensor]): The dimensions to reduce. Default: ``None`` ,
            reduce all dimensions. Only constant value is allowed. Assume the rank of `input` is r,
            and the value range is [-r,r).
        keepdim (bool): If ``True`` , keep these reduced dimensions and the length is 1.
            If ``False`` , don't keep these dimensions. Default: ``False`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The desired data type of returned Tensor. Default: ``None`` .

    Returns:
        Tensor, has the same data type as the `input`.

        - If `dim` is ``None`` , and `keepdim` is ``False`` ,
          the output is a 0-D tensor representing the product of all elements in the input tensor.
        - If `dim` is int, set as 1, and `keepdim` is ``False`` ,
          the shape of output is :math:`(input_0, input_2, ..., input_R)`.
        - If `dim` is tuple(int), set as (1, 2), and `keepdim` is ``False`` ,
          the shape of output is :math:`(input_0, input_3, ..., input_R)`.
        - If `dim` is 1-D Tensor, set as [1, 2], and `keepdim` is ``False`` ,
          the shape of output is :math:`(input_0, input_3, ..., input_R)`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `dim` is not one of the following: int, tuple, list or Tensor.
        TypeError: If `keepdim` is not a bool.
        ValueError: If `dim` is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> output = mint.mean(x, 1, keep_dims=True)
        >>> result = output.shape
        >>> print(result)
        (3, 1, 5, 6)
        >>> # case 1: Reduces a dimension by averaging all elements in the dimension.
        >>> x = Tensor(np.array([[[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]],
        ... [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ... [[6, 6, 6, 6, 6, 6], [8, 8, 8, 8, 8, 8], [10, 10, 10, 10, 10, 10]]]),
        ... mindspore.float32)
        >>> output = mint.mean(x)
        >>> print(output)
        5.0
        >>> print(output.shape)
        ()
        >>> # case 2: Reduces a dimension along the axis 0
        >>> output = mint.mean(x, 0, True)
        >>> print(output)
        [[[4. 4. 4. 4. 4. 4.]
        [5. 5. 5. 5. 5. 5.]
        [6. 6. 6. 6. 6. 6.]]]
        >>> # case 3: Reduces a dimension along the axis 1
        >>> output = mint.mean(x, 1, True)
        >>> print(output)
        [[[2. 2. 2. 2. 2. 2.]]
        [[5. 5. 5. 5. 5. 5.]]
        [[8. 8. 8. 8. 8. 8.]]]
        >>> # case 4: Reduces a dimension along the axis 2
        >>> output = mint.mean(x, 2, True)
        >>> print(output)
        [[[ 2.]
        [ 2.]
        [ 2.]]
        [[ 4.]
        [ 5.]
        [ 6.]]
        [[ 6.]
        [ 8.]
        [10.]]]
        """
    return mean_ext(input, axis=dim, keep_dims=keepdim, dtype=dtype)


def ones(size, *, dtype=None):
    r"""
    Creates a tensor filled with value ones.

    Creates a tensor with shape described by the first argument and fills it with value ones in type of the second
    argument.

    Args:
        size (Union[tuple[int], list[int], int, Tensor]): The specified shape of output tensor. Only positive integer or
            tuple or Tensor containing positive integers are allowed. If it is a Tensor,
            it must be a 0-D or 1-D Tensor with int32 or int64 dtypes.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The specified type of output tensor. If `dtype` is ``None`` ,
            `mindspore.float32` will be used. Default: ``None`` .

    Returns:
        Tensor, whose dtype and size are defined by input.

    Raises:
        TypeError: If `size` is neither an int nor an tuple/list/Tensor of int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import mint
        >>> output = mint.ones((2, 2), mindspore.float32)
        >>> print(output)
        [[1. 1.]
         [1. 1.]]
    """
    return ones_ext(size, dtype)


def permute(input, dims):
    """
    Permutes the dimensions of the input tensor according to input `dims` .

    Args:
        input (Tensor): Input Tensor.
        dims (tuple(int)): The order of the dimensions. Permute rearranges the `input` according
            to the order of the `dims`.

    Returns:
        Tensor, has the same dimension as input tensor, with `axis` suitably permuted.

    Raises:
        ValueError: If `dims` is None.
        ValueError: If the number of elements of `dims` is not equal to `input` ndim.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> input_x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
        >>> input_perm = (0, 2, 1)
        >>> print(mint.permute(input_x, input_perm))
        [[[ 1.  4.]
          [ 2.  5.]
          [ 3.  6.]]
         [[ 7. 10.]
          [ 8. 11.]
          [ 9. 12.]]]
    """
    return permute_ext(input, dims)


def split(tensor, split_size_or_sections, dim=0):
    """
    Splits the Tensor into chunks along the given dim.

    Args:
        tensor (Tensor): A Tensor to be divided.
        split_size_or_sections (Union[int, tuple(int), list(int)]):
            If `split_size_or_sections` is an int type, `tensor` will be split into equally sized chunks,
            each chunk with size `split_size_or_sections`. Last chunk will be smaller than `split_size_or_sections`
            if `tensor.shape[dim]` is not divisible by `split_size_or_sections`.
            If `split_size_or_sections` is a list type, then `tensor` will be split into len(split_size_or_sections)
            chunks with sizes `split_size_or_sections` along the given `dim`.
        dim (int): The dim along which to split. Default: ``0`` .

    Returns:
        A tuple of sub-tensors.

    Raises:
        TypeError: If argument `tensor` is not Tensor.
        TypeError: If argument `dim` is not int.
        ValueError: If argument `dim` is out of range of :[-tensor.ndim, tensor.ndim).
        TypeError: If each element in `split_size_or_sections` is not integer.
        TypeError: If argument `split_size_or_sections` is not int, tuple(int) or list(int).
        ValueError: The sum of `split_size_or_sections` is not equal to x.shape[dim].

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import ops, Tensor
        >>> input_x = np.arange(9).astype("float32")
        >>> output = ops.split(Tensor(input_x), 3)
        >>> print(output)
        (Tensor(shape=[3], dtype=Float32, value= [ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value= [ 3.00000000e+00,  4.00000000e+00,  5.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value= [ 6.00000000e+00,  7.00000000e+00,  8.00000000e+00]))
    """
    return split_ext(tensor, split_size_or_sections, dim)


def sqrt(input):
    r"""
    Returns sqrt of a tensor element-wise.

    .. math::

        out_{i} = \sqrt{input_{i}}

    Args:
        input (Tensor): The input tensor with a dtype of number.Number.

    Returns:
        Tensor, has the same shape as the `input`.

    Raises:
        TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> input = Tensor(np.array([1.0, 4.0, 9.0]), mindspore.float32)
        >>> output = mint.sqrt(input)
        >>> print(output)
        [1. 2. 3.]
    """
    return sqrt_ext(input)


def sub(input, other, *, alpha=1):
    r"""
    Subtracts scaled other value from input Tensor.

    .. math::

        out_{i} = input_{i} - alpha \times other_{i}

    Note:
        - When the two inputs have different shapes,
          they must be able to broadcast to a common shape.
        - The two inputs and alpha comply with the implicit type conversion rules to make the data types
          consistent.

    Args:
        input (Union[Tensor, number.Number, bool]): The first input is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.
        other (Union[Tensor, number.Number, bool]): The second input, is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.

    Keyword Args:
        alpha (number.Number): A scaling factor applied to `other`, default 1.

    Returns:
        Tensor, the shape is the same as the one of the input `input`, `other` after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs and alpha.

    Raises:
        TypeError: If the type of `input`, `other`, or `alpha` is not one of the following: Tensor, number.Number, bool.
        TypeError: If `alpha` is of type float but `input` and `other` are not of type float.
        TypeError: If `alpha` is of type bool but `input` and `other` are not of type bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore import mint
        >>> x = Tensor(np.array([4, 5, 6]).astype(np.float32))
        >>> y = Tensor(1, mindspore.int32)
        >>> alpha = 0.5
        >>> output = mint.sub(x, y, alpha)
        >>> print(output)
        [3.5 4.5 5.5]
        >>> # the data type of x is float32, the data type of y is int32,
        >>> # alpha is a float, and the output is the data format of higher precision float32.
        >>> print(output.dtype)
        Float32
    """
    return sub_ext(input, other, alpha)


def zeros(size, *, dtype=None):
    """
    Creates a tensor filled with 0 with shape described by `size` and fills it with value 0 in type of `dtype`.

    Args:
        size (Union[tuple[int], list[int], int, Tensor]): The specified shape of output tensor. Only positive integer or
            tuple or Tensor containing positive integers are allowed. If it is a Tensor,
            it must be a 0-D or 1-D Tensor with int32 or int64 dtypes.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The specified type of output tensor. If `dtype` is ``None`` ,
            mindspore.float32 will be used. Default: ``None`` .

    Returns:
        Tensor, whose dtype and size are defined by input.

    Raises:
        TypeError: If `size` is neither an int nor an tuple/list/Tensor of int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import mint
        >>> output = mint.zeros((2, 2), mindspore.float32)
        >>> print(output)
        [[0. 0.]
         [0. 0.]]
    """
    return zeros_ext(size, dtype)


__all__ = [
    'full',
    'ones_like',
    'zeros_like',
    'abs',
    'erf',
    'where',
    'linspace',
    'isclose',
    # 1
    'div',
    'divide',
    'topk',
    # 2
    'sin',
    # 3
    'clamp',
    # 4

    # 5
    'cumsum',
    # 6
    'stack',
    # 7
    'zeros',
    # 8

    # 9

    # 10
    'ne',
    # 11

    # 12
    "repeat_interleave",
    # 13
    "flip",
    # 14

    # 15
    'flatten',
    # 16
    'matmul',
    'bmm',
    # 17
    'mean',
    # 18
    'sum',
    # 19
    'log',
    # 20
    'prod',
    # 21
    'mul',
    # 22

    # 23

    # 24

    # 25
    'greater',
    'gt',
    # 26
    'eq',
    # 27
    'reciprocal',
    # 28
    'exp',
    # 29
    'sqrt',
    # 30
    'searchsorted',
    # 31

    # 32
    'sub',
    # 33
    'split',
    # 34

    # 35
    'erfinv',
    # 36

    # 37
    'nonzero',
    # 38

    # 39

    # 40
    'any',
    # 41
    'add',
    # 42
    'argmax',
    # 43
    'cat',
    # 44
    'cos',
    # 45

    # 46

    # 47

    # 48

    # 49

    # 50
    'tile',
    # 51
    'permute',
    # 52

    # 53

    # 54
    'normal',
    # 55

    # 56

    # 57
    'broadcast_to',
    # 58
    'greater_equal',
    # 59
    'square',
    # 60
    'all',
    # 61
    'rsqrt',
    # 62
    'maximum',
    # 63
    'minimum',
    # 64

    # 65
    'logical_and',
    # 66
    'logical_not',
    # 67
    'logical_or',
    # 68

    # 69
    'less_equal',
    'le',
    # 70
    'negative',
    'neg',
    # 71
    'isfinite',
    # 72

    # 73
    'ceil',
    # 74
    'sort',
    # 75
    'less',
    'lt',
    # 76
    'pow',
    # 77

    # 78
    'arange',

    # 79

    # 80

    # 81
    'index_select',
    # 82

    # 83
    'narrow',
    # 84

    # 85

    # 86

    # 87

    # 88
    'chunk',
    # 89

    # 90

    # 91

    # 92

    # 93

    # 94
    'tanh',
    # 95

    # 96

    # 97

    # 98

    # 99

    # 100

    # 176
    'atan2',
    'arctan2',

    # 208
    'eye',
    'rand',
    'rand_like',
    # 210
    'floor',
    # 231
    'inverse',
    # 285
    'scatter_add',
    # 304

    # 305
    'triu',
]
__all__.extend(array_func.__all__)
__all__.extend(math_func.__all__)
__all__.extend(nn_func.__all__)
__all__.extend(functional.__all__)
__all__.extend(nn.__all__)
__all__.extend(optim.__all__)
__all__.extend(linalg.__all__)
