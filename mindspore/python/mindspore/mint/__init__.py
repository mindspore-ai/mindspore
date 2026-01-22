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
import mindspore
from mindspore import ops
from mindspore.ops.primitive import constexpr
from mindspore.common.tensor import Tensor
from mindspore.ops.function.array_func import gather_ext as gather
from mindspore.ops.function.nn_func import conv2d_ext as conv2d
from mindspore.mint.nn.functional import sigmoid
from mindspore.mint.nn import functional
from mindspore.mint import linalg
from mindspore.mint import special
from mindspore.mint import distributed
from mindspore.ops import erf
from mindspore.ops.function.math_func import linspace_ext as linspace
from mindspore.ops.function.math_func import median_ext as median
from mindspore.ops.function.array_func import ones_like_ext as ones_like
from mindspore.ops.function.array_func import full_ext as full
from mindspore.ops.function.array_func import zeros_like_ext as zeros_like
from mindspore.ops.function.array_func import unique_ext as unique
from mindspore.ops.function.array_func import chunk_view as chunk
from mindspore.ops.functional_overload import empty
from mindspore.ops.functional_overload import empty_like
from mindspore.ops.function.math_func import isclose
from mindspore.ops.auto_generate import abs
from mindspore.ops.auto_generate import clone
from mindspore.ops.function.array_func import full_like_ext as full_like
from mindspore.graph._utils.check_jit_forbidden_api import jit_forbidden_register
from mindspore.graph._check_supported import jit_view_unsupported
# 1
from mindspore.ops.function.math_func import divide
from mindspore.ops.auto_generate import topk_ext as topk
from mindspore.ops.function.math_func import roll
# 2
from mindspore.ops.function.math_func import sin
# 3
from mindspore.ops.functional_overload import clamp, where
from mindspore.ops.functional_overload import clip
from mindspore.ops.functional_overload import fmod
from mindspore.ops.functional_overload import max
from mindspore.ops.functional_overload import min
# 4
from mindspore.ops.auto_generate import sinc
from mindspore.ops.auto_generate import sinh
from mindspore.ops.auto_generate import cosh
from mindspore.ops.functional_overload import xlogy
# 5
from mindspore.ops.auto_generate import cumsum_ext as cumsum
# 6
from mindspore.ops.auto_generate import stack_ext as stack

# 7
from mindspore.ops.function.array_func import unsqueeze_view as unsqueeze
# 8
from mindspore.ops.auto_generate import transpose_ext_view as transpose
from mindspore.ops.auto_generate import batch_norm_elemt
from mindspore.ops.auto_generate import batch_norm_gather_stats_with_counts
from mindspore.ops.auto_generate import batch_norm_stats
# 9
from mindspore.ops.auto_generate import masked_select
from mindspore.ops.function.math_func import cross
# 10
from mindspore.ops.function.math_func import ne
# 11
from mindspore.ops.function.math_func import cdist as cdist_
# 12
from mindspore.ops.functional_overload import repeat_interleave
# 13
from mindspore.ops.functional import flip
# 14
from mindspore.ops.auto_generate import mv
# 15
from mindspore.ops.auto_generate import flatten_ext
# 16
from mindspore.ops.functional import matmul
from mindspore.ops.auto_generate import bmm_ext as bmm
# 17

# 18

# 19
from mindspore.ops.functional import log
# 20

# 21
from mindspore.ops.functional_overload import mul
# 22
from mindspore.ops.functional import cumprod
# 23
from mindspore.ops.auto_generate import exp2
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

# 30
from mindspore.ops.functional import searchsorted
# 31

# 32
from mindspore.ops.function.math_func import einsum_ext as einsum
# 33

# 34

# 35
from mindspore.ops.functional import erfinv
# 36

# 37
from mindspore.ops.function.array_func import nonzero
# 38

# 39

# 40

# 41

# 42
from mindspore.ops.function.math_func import argmax_ext as argmax
# 43

# 44
from mindspore.ops.functional import cos
# 45

# 46
from mindspore.ops.function.math_func import bitwise_and_ext as bitwise_and
# 47
from mindspore.ops.function.math_func import bitwise_or_ext as bitwise_or
# 48
from mindspore.ops.function.math_func import bitwise_xor_ext as bitwise_xor
# 49
from mindspore.ops.function.math_func import baddbmm_ext as baddbmm
# 50
from mindspore.ops.functional import tile
# 51

# 52
from mindspore.ops.functional_overload import addcdiv
# 53

# 54
from mindspore.ops.function.random_func import normal_ext as normal
# 55

# 56
from mindspore.ops.function.math_func import norm_ext as norm
# 57
from mindspore.ops.auto_generate import broadcast_to_view as broadcast_to
# 58
from mindspore.ops.functional_overload import greater_equal, ge

# 59
from mindspore.ops.functional import square
# 60

# 61
from mindspore.ops.functional import rsqrt
# 62
from mindspore.ops.functional import maximum
# 63
from mindspore.ops.functional import minimum
# 64
from mindspore.ops.functional import ravel
# 65
from mindspore.ops.functional import logical_and
# 66
from mindspore.ops.functional import logical_not
# 67
from mindspore.ops.functional import logical_or
# 68
from mindspore.ops.functional import logical_xor
# 69
from mindspore.ops.functional import less_equal, le
# 70
from mindspore.ops.functional import negative, neg
# 71

# 72

# 73
from mindspore.ops.functional import ceil
# 74
from mindspore.ops.function.array_func import sort_ext as sort
# 75
from mindspore.ops.functional import less, lt
# 76
from mindspore.ops.function.math_func import pow_ext as pow
# 77

# 78
from mindspore.ops.function import arange_ext as arange
# 79

# 80
from mindspore.ops.functional_overload import div
# 81
from mindspore.ops.auto_generate import index_select_ext as index_select
# 82
from mindspore.ops.auto_generate import cummin_ext as cummin
# 83
from mindspore.ops.auto_generate import narrow_view as narrow
# 84

# 85
from mindspore.mint import nn, optim
# 86

# 87
from mindspore.ops.auto_generate import trunc
# 88

# 89
from mindspore.ops.auto_generate import argsort_ext as argsort
# 90
from mindspore.ops.auto_generate import isinf
# 91

# 92
from mindspore.ops.function.math_func import polar
# 93

# 94
from mindspore.ops.function.math_func import tanh
# 95
from mindspore.ops.function.math_func import diff_ext as diff
# 96

# 97

# 98

# 99

# 100

# 101

# 102

# 103

# 104

# 105

# 106

# 107

# 108

# 109
from mindspore.ops.auto_generate import argmin_ext as argmin
# 110
from mindspore.ops.function.nn_func import softmax_ext
# 111

# 112

# 113

# 114

# 115

# 116

# 117

# 118
from mindspore.ops.function.array_func import split_view as split
# 119
from mindspore.ops.functional_overload import any
# 120
from mindspore.ops.auto_generate import isneginf_ext as isneginf
# 121

# 122

# 123
from mindspore.ops.function.math_func import var_ext as var

# 151
from mindspore.ops.function.math_func import acos_ext as acos
from mindspore.ops.function.math_func import arccos_ext as arccos
# 152
from mindspore.ops.function.math_func import acosh_ext as acosh
from mindspore.ops.function.math_func import arccosh_ext as arccosh
# 172
from mindspore.ops.function.math_func import addcmul_ext as addcmul

from mindspore.ops.function.math_func import asin_ext as asin
from mindspore.ops.function.math_func import arcsin_ext as arcsin
# 173
from mindspore.ops.function.math_func import asinh_ext as asinh
from mindspore.ops.function.math_func import arcsinh_ext as arcsinh
# 174
from mindspore.ops.function.math_func import atan_ext as atan
from mindspore.ops.function.math_func import arctan_ext as arctan
# 175
from mindspore.ops.function.math_func import atanh
from mindspore.ops.function.math_func import arctanh
# 176
from mindspore.ops.function.math_func import atan2_ext as atan2
from mindspore.ops.function.math_func import arctan2_ext as arctan2

# 177
from mindspore.ops.function.math_func import round

# 182
from mindspore.ops.function.math_func import bernoulli_ext as bernoulli

# 201
from mindspore.ops.auto_generate import diag_ext as diag

# 204
from mindspore.ops.auto_generate import erfc
# 207
from mindspore.ops.auto_generate import expm1
# 208
from mindspore.ops.function.array_func import eye
from mindspore.ops.function.random_func import randperm_ext as randperm
from mindspore.ops.function.random_func import rand_ext as rand
from mindspore.ops.function.random_func import rand_like_ext as rand_like
from mindspore.ops.function.random_func import randn_ext as randn
from mindspore.ops.function.random_func import randn_like_ext as randn_like
from mindspore.ops.function.random_func import randint_ext as randint
from mindspore.ops.function.random_func import randint_like_ext as randint_like
# 210
from mindspore.ops.auto_generate import floor
# 231
from mindspore.ops.function.math_func import inverse_ext as inverse
# 239
from mindspore.ops.functional_overload import lerp
# 244
from mindspore.ops.auto_generate import log1p
# 261
from mindspore.ops.function.random_func import multinomial_ext as multinomial
# 275
from mindspore.ops.functional_overload import remainder
# 285
from mindspore.ops.function.array_func import scatter_add_ext as scatter_add
# 289
from mindspore.ops.auto_generate import sign

from mindspore.ops.auto_generate import select_ext_view as select

# 301
from mindspore.ops.function.math_func import tan

# 303
from mindspore.ops.auto_generate import trace_ext as trace
from mindspore.ops.auto_generate import gcd

from mindspore.ops.auto_generate import outer_ext as outer

# 304
from mindspore.ops.function.array_func import tril_ext as tril
# 520
from mindspore.ops.function.math_func import bincount_ext as bincount

# 305
from mindspore.ops import triu

# 308
from mindspore.ops.auto_generate import mm_ext as mm

# 382
from mindspore.ops.function.math_func import dstack

# 501
from mindspore.ops.function.math_func import addbmm_ext as addbmm

# 502
from mindspore.ops.function.math_func import addmm_ext as addmm

# 505
from mindspore.ops.function.math_func import addmv_ext as addmv

# 510
from mindspore.ops.function.math_func import amax_ext as amax

# 511
from mindspore.ops.function.math_func import amin_ext as amin

# 521
from mindspore.ops.functional_overload import bitwise_not

# 526
from mindspore.ops.auto_generate import dot

# 533
from mindspore.ops.function.math_func import frac_ext as frac

# 538
from mindspore.ops.function.math_func import histc_ext as histc
# 549
from mindspore.ops.functional_overload import kthvalue
# 552
from mindspore.ops.auto_generate import log10_ext as log10

# 553
from mindspore.ops.auto_generate import logaddexp_ext as logaddexp
from mindspore.ops.auto_generate import logaddexp2

# 557
from mindspore.ops.auto_generate import logsumexp_ext as logsumexp

# 582
from mindspore.ops.function.math_func import std_mean_ext as std_mean

# 584
from mindspore.ops.function.array_func import take

# 588
from mindspore.ops.function.math_func import var_mean_ext as var_mean

# 610
from mindspore.ops.function.math_func import nan_to_num

# 613
from mindspore.ops.functional_overload import nansum

# 615
from mindspore.ops.auto_generate import triangular_solve

# 664
from mindspore.ops.function.array_func import meshgrid_ext as meshgrid

# 695
from mindspore.ops.auto_generate import count_nonzero

# 697
from mindspore.ops.function.math_func import float_power_ext as float_power

# 708
from mindspore.ops.function.math_func import std_ext as std

# 719
from mindspore.ops.functional_overload import add

# 720
from mindspore.ops.functional_overload import sub

# 739
from mindspore.ops.function.array_func import hstack

# 826
from mindspore.ops.functional_overload import floor_divide

# 887
from mindspore.ops.auto_generate import log2_ext as log2

# 889
from mindspore.ops.function.math_func import isnan_ext as isnan

# 916
from mindspore.ops.functional_overload import index_add

# 1007
from mindspore.ops.auto_generate.pyboost_inner_prim import squeeze_impl
from mindspore.ops.auto_generate.gen_ops_prim import equal_ext_op

# 1101
from mindspore.ops.functional_overload import real
# 1102
from mindspore.ops.functional_overload import imag

# 1023
from mindspore.ops.function.array_func import unbind_ext as unbind


def all(input, dim=None, keepdim=False):
    r"""
    all(input) -> Tensor

    Tests if all element in `input` evaluates to `True`.

    Args:
        input (Tensor): The input Tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[True, False], [True, True]])
        >>> output = mindspore.mint.all(input)
        >>> print(output)
        False

    .. function:: all(input, dim, keepdim=False) -> Tensor
        :noindex:

    Tests if all element in `input` evaluates to `True` along the given axes.

    Args:
        input (Tensor): The input tensor.
        dim (Union[int, tuple(int), list(int), Tensor]): The dimensions to reduce. If ``None`` ,
                all dimensions are reduced. Default ``None`` .
        keepdim (bool, optional): Whether the output tensor has dim retained or not. Default ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[True, False], [True, True]])
        >>>
        >>> # case 1: Reduces a dimension along dim 1, with keepdim False.
        >>> mindspore.mint.all(input, dim=1)
        Tensor(shape=[2], dtype=Bool, value= [False,  True])
        >>>
        >>> # case 2: Reduces a dimension along dim (0, 1), with keepdim False.
        >>> mindspore.mint.all(input, dim=(0,1))
        Tensor(shape=[], dtype=Bool, value= False)
        >>>
        >>> # case 3: Reduces a dimension along dim [0, 1], with keepdim True.
        >>> mindspore.mint.all(input, dim=[0,1], keepdim=True)
        Tensor(shape=[1, 1], dtype=Bool, value=
        [[False]])
    """
    return ops.function.math_func.all(input, dim, keepdim)


def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Return whether each element of `input` is “close” to the corresponding element of `other`. Closeness is defined as:

    .. math::
        |input-other| ≤ atol + rtol × |other|

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The first input tensor.
        other (Tensor): The second input tensor.
        rtol (Union[float, int, bool], optional): Relative tolerance. Default ``1e-05`` .
        atol (Union[float, int, bool], optional): Absolute tolerance. Default ``1e-08`` .
        equal_nan (bool, optional): Whether two NaNs are considered equal. Default ``False``.

    Returns:
        A bool scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> input = mindspore.tensor(np.array([1.3, 2.1, 3.2, 4.1, 5.1]), mindspore.float16)
        >>> other = mindspore.tensor(np.array([1.3, 3.3, 2.3, 3.1, 5.1]), mindspore.float16)
        >>> output = mindspore.mint.allclose(input, other)
        >>> print(output)
        False
    """
    return isclose(input, other, rtol, atol, equal_nan).all().item()


def cat(tensors, dim=0):
    r"""
    Connect input tensors along with the given dimension.

    The input data is a tuple or a list of tensors. These tensors have the same rank :math:`R`.
    Set the given dimension as :math:`m`, and :math:`0 \le m < R`. Set the number of input tensors as :math:`N`.
    For the :math:`i`-th tensor :math:`t_i`, it has the shape of :math:`(x_1, x_2, ..., x_{mi}, ..., x_R)`.
    :math:`x_{mi}` is the :math:`m`-th dimension of the :math:`t_i`. Then, the shape of the output tensor is

    .. math::

        (x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)

    .. warning::
        Input tensor of inconsistent types are not supported in Graph Mode under Dynamic Shape.

    Args:
        tensors (Union[tuple, list]): A tuple or a list of input tensors.
            Suppose there are two tensors in this tuple or list, namely t1 and t2.
            To perform `concat` in the dimension 0 direction, except for the :math:`0`-th dimension,
            all other dimensions should be equal, that is,
            :math:`t1.shape[1] = t2.shape[1], t1.shape[2] = t2.shape[2], ..., t1.shape[R-1] = t2.shape[R-1]`,
            where :math:`R` represents the rank of tensor.
        dim (int, optional): The specified dimension, whose value is in range :math:`[-R, R)`. Default: ``0`` .

    Returns:
        Tensor, the shape is :math:`(x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)`.

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
    return ops.auto_generate.cat(tensors, dim)


def concat(tensors, dim=0):
    r"""
    Alias for :func:`mindspore.mint.cat`.

    .. warning::
        Input tensor of inconsistent types are not supported in Graph Mode under Dynamic Shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return cat(tensors, dim)


def cummax(input, dim):
    r"""
    Return the cumulative maximum values and their indices along the given dimension of the tensor.

    .. math::
        \begin{array}{ll} \\
            y_{i} = \max(x_{1}, x_{2}, ... , x_{i})
        \end{array}

    .. note::
        GE backend is not supported in Ascend.

    Args:
        input (Tensor): The input tensor.
        dim (int): The dimension to compute the cumulative maximum along.

    Returns:
        Tuple(max, max_indices) of 2 tensors.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]])
        >>> mindspore.mint.cummax(x, dim=0)
        (Tensor(shape=[4, 4], dtype=Int64, value=
        [[ 3,  4,  6, 10]
         [ 3,  6,  7, 10]
         [ 4,  6,  8, 10]
         [ 4,  6,  8, 10]]),
        Tensor(shape=[4, 4], dtype=Int64, value=
        [[0, 0, 0, 0]
         [0, 1, 1, 0]
         [2, 1, 2, 0]
         [2, 1, 2, 0]]))
        >>> mindspore.mint.cummax(x, dim=1)
        (Tensor(shape=[4, 4], dtype=Int64, value=
        [[ 3,  4,  6, 10]
         [ 1,  6,  7,  9]
         [ 4,  4,  8,  8]
         [ 1,  3,  7,  9]]),
        Tensor(shape=[4, 4], dtype=Int64, value=
        [[0, 1, 2, 3]
         [0, 1, 2, 3]
         [0, 0, 2, 2]
         [0, 1, 2, 3]]))
    """
    return ops.auto_generate.cummax(input, dim)


def not_equal(input, other):
    r"""
    Alias for :func:`mindspore.mint.ne` .

    Supported Platforms:
        ``Ascend``
    """
    return ne(input, other)


def softmax(input, dim, *, dtype=None):
    r"""
    Alias for :func:`mindspore.mint.nn.functional.softmax`.

    Supported Platforms:
        ``Ascend``
    """
    return softmax_ext(input, dim, dtype)


def equal(input, other):
    r"""
    Compute the equivalence of the two inputs element-wise.

    Note:
        `input` and `other` comply with the implicit type conversion rules to make the data types consistent.

    Args:
        input (Tensor): The first input tensor.
        other (Tensor): The second input tensor.

    Returns:
        Boolean tensor

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([1, 2, 3], mindspore.int32)
        >>> y = mindspore.tensor([1, 2, 4], mindspore.int32)
        >>> output = mindspore.mint.equal(x, y)
        >>> print(output)
        False
    """
    result = equal_ext_op(input, other)
    return result.item()


def isfinite(input):
    r"""
    Determine which elements are finite for each position. If elements are not ``NaN`` , ``-INF`` , ``INF``,
    they are finite.

    .. math::
        out_i = \begin{cases}
          & \text{ if } input_{i} = \text{Finite},\ \ True \\
          & \text{ if } input_{i} \ne \text{Finite},\ \ False
        \end{cases}

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor, has the same shape of input, and the dtype is bool.

    Raises:
        TypeError: If input is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> output = mint.isfinite(x)
        >>> print(output)
        [False True False]
        >>> x = Tensor(2.1, mindspore.float64)
        >>> output = mint.isfinite(x)
        >>> print(output)
        True
    """
    return ops.auto_generate.isfinite(input)


def item(input):
    r"""
    Returns the value of this tensor as a standard Python number.

    Note:
        This only works for tensors with one element.

    Args:
        input (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
            :math:`(N, *)` where :math:`*` means, any number of additional dimensions.

    Returns:
        number.

    Raises:
        TypeError: If `input` is not a Tensor.
        RuntimeError: If the number of `input` elements is not 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> x = Tensor(np.array([1]).astype(np.float32))
        >>> result = mint.item(x)
        >>> print(result)
        1.0
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"the input must be a Tensor, but got {type(input)}")
    if input.size != 1:
        raise RuntimeError(
            "a Tensor with {} elements cannot be converted to Scalar".format(input.size))
    return input.asnumpy().item()


def mean(input, dim=None, keepdim=False, *, dtype=None):
    r"""
    mean(input, *, dtype=None) -> Tensor

    Compute the mean of the tensor.

    Args:
        input (Tensor[Number]): The input tensor. 

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The desired data type of returned tensor. Default ``None`` .

    Returns:
        Tensor.


    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> x = mindspore.tensor(np.array([[[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]],
        ... [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ... [[6, 6, 6, 6, 6, 6], [8, 8, 8, 8, 8, 8], [10, 10, 10, 10, 10, 10]]]),
        ... mindspore.float32)
        >>> output = mindspore.mint.mean(x)
        >>> print(output)
        5.0
        >>> print(output.shape)
        ()

    .. function:: mean(input, dim, keepdim=False, *, dtype=None) -> Tensor
        :noindex:

    Compute the mean(s) of the tensor along the specified dimension(s).

    Note:
        The `dim` with tensor type is only used for compatibility with older versions and is not recommended.

    Args:
        input (Tensor[Number]): The input tensor. 
        dim (Union[int, tuple(int), list(int), Tensor]): Specify the dimension(s) for computation. 
        keepdim (bool): Whether the output tensor has `dim` retained. Default ``False`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The desired data type of returned tensor. Default ``None`` .

    Returns:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> x = mindspore.tensor(np.array([[[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]],
        ... [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ... [[6, 6, 6, 6, 6, 6], [8, 8, 8, 8, 8, 8], [10, 10, 10, 10, 10, 10]]]),
        ... mindspore.float32)
        >>> output = mindspore.mint.mean(x, 0, True)
        >>> print(output)
        [[[4. 4. 4. 4. 4. 4.]
          [5. 5. 5. 5. 5. 5.]
          [6. 6. 6. 6. 6. 6.]]]
    """
    return ops.auto_generate.mean_ext(input, dim, keepdim, dtype)


def prod(input, dim=None, keepdim=False, *, dtype=None):
    r"""
    prod(input, *, dtype=None) -> Tensor

    Multiplying all elements of input.

    Args:
        input (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
            :math:`(N, *)` where :math:`*` means, any number of additional dimensions.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The desired data type of returned Tensor. Default: ``None`` .

    Returns:
        Tensor.

    Raises:
        TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ...                      [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ...                      [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mindspore.float32)
        >>> output = mint.prod(x)
        >>> print(output)
        2.2833798e+33
        >>> print(output.shape)
        ()

    .. function:: prod(input, dim, keepdim=False, *, dtype=None) -> Tensor
        :noindex:

    Reduces a dimension of a tensor by multiplying all elements in the dimension, by default. And also can
    reduce a dimension of `input` along the `dim`. Determine whether the dimensions of the output and input are the
    same by controlling `keepdim`.

    Args:
        input (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
            :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        dim (int): The dimensions to reduce. Only constant value is allowed.
            Assume the rank of `x` is r, and the value range is [-r,r).
        keepdim (bool): If ``True`` , keep these reduced dimensions and the length is 1.
            If ``False`` , don't keep these dimensions. Default: ``False`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The desired data type of returned Tensor. Default: ``None`` .

    Returns:
        Tensor.

        - If `dim` is int, set as 1, and `keepdim` is ``False`` ,
          the shape of output is :math:`(input_0, input_2, ..., input_R)`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `dim` is not int.
        TypeError: If `keepdim` is not a bool.
        ValueError: If `dim` is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ...                      [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ...                      [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mindspore.float32)
        >>> output = mint.prod(x, 0, True)
        >>> print(output)
        [[[ 28.  28.  28.  28.  28.  28.]
          [ 80.  80.  80.  80.  80.  80.]
          [162. 162. 162. 162. 162. 162.]]]
    """
    return ops.auto_generate.prod_ext(input, dim, keepdim, dtype)


def sum(input, dim=None, keepdim=False, *, dtype=None):
    r'''
    sum(input, *, dtype=None) -> Tensor

    Calculate sum of all elements in Tensor.

    Args:
        input (Tensor): The input tensor.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The desired data type of returned Tensor. Default: ``None`` .

    Returns:
        A Tensor, sum of all elements in `input`.

    Raises:
        TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> from mindspore import dtype as mstype
        >>> x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ...                      [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ...                      [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mstype.float32)
        >>> out = mint.sum(x)
        >>> print(out)
        270.0

    .. function:: sum(input, dim, keepdim=False, *, dtype=None) -> Tensor
        :noindex:

    Calculate sum of Tensor elements over a given dim.

    Note:
        The `dim` with tensor type is only used for compatibility with older versions and is not recommended.

    Args:
        input (Tensor): The input tensor.
        dim (Union[int, tuple(int), list(int), Tensor]): Dimensions along which a sum is performed.
            If the `dim` is a tuple or list of ints, a sum is performed on all the dimensions specified in the tuple.
            Must be in the range :math:`[-input.ndim, input.ndim)` .
        keepdim (bool): Whether the output tensor has `dim` retained or not.
            If ``True`` , keep these reduced dimensions and the length is 1.
            If ``False`` , don't keep these dimensions. Default: ``False`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The desired data type of returned Tensor. Default: ``None`` .

    Returns:
        A Tensor, sum of elements over a given `dim` in `input`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `dim` is not an int, tulpe(int), list(int) or Tensor.
        ValueError: If `dim` is not in the range :math:`[-input.ndim, input.ndim)` .
        TypeError: If `keepdim` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> from mindspore import dtype as mstype
        >>> x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ...                      [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ...                      [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mstype.float32)
        >>> out = mint.sum(x)
        >>> print(out)
        270.0
        >>> out = mint.sum(x, dim=2)
        >>> print(out)
        [[ 6. 12. 18.]
         [24. 30. 36.]
         [42. 48. 54.]]
        >>> out = mint.sum(x, dim=2, keepdim=True)
        >>> print(out)
        [[[ 6.]
          [12.]
          [18.]]
         [[24.]
          [30.]
          [36.]]
         [[42.]
          [48.]
          [54.]]]
    '''
    return ops.auto_generate.sum_ext(input, dim, keepdim, dtype)


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
        TypeError: If `size` is neither an int nor a tuple/list/Tensor of int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import mint
        >>> output = mint.ones((2, 2), dtype=mindspore.float32)
        >>> print(output)
        [[1. 1.]
         [1. 1.]]
    """
    return ops.auto_generate.ones(size, dtype)


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
    return ops.auto_generate.transpose_view(input, dims)


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
    return ops.auto_generate.sqrt(input)


@jit_view_unsupported
def squeeze(input, dim):
    r"""
    Return the Tensor after deleting the dimension of size 1 in the specified `dim`.

    If :math:`dim=()`, it will remove all the dimensions of size 1.
    If `dim` is specified, it will remove the dimensions of size 1 in the given `dim`.
    For example, if the dimension is not specified :math:`dim=()`, input shape is (A, 1, B, C, 1, D),
    then the shape of the output Tensor is (A, B, C, D). If the dimension is specified, the squeeze operation
    is only performed in the specified dimension. If input shape is (A, 1, B), when :math:`dim=0` or :math:`dim=2`,
    the input tensor is not changed, while when :math:`dim=1`, the input tensor shape is changed to (A, B).

    Note:
        - Please note that in dynamic graph mode, the output Tensor will share data with the input Tensor,
          and there is no Tensor data copy process.
        - The dimension index starts at 0 and must be in the range `[-input.ndim, input.ndim]`.
        - In GE mode, only support remove dimensions of size 1 from the shape of input tensor.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): Used to calculate Squeeze. The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        dim (Union[int, tuple(int)]): Specifies the dimension indexes of shape to be removed, which will
            remove all the dimensions of size 1 in the given dim parameter. If specified, it must be int32 or int64.

    Returns:
        Tensor, the shape of tensor is :math:`(x_1, x_2, ..., x_S)`.

    Raises:
        TypeError: If `input` is not a tensor.
        TypeError: If `dim` is not an int, tuple.
        TypeError: If `dim` is a tuple whose elements are not all int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> input = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
        >>> output = mint.squeeze(input, 2)
        >>> print(output)
        [[1. 1.]
         [1. 1.]
         [1. 1.]]
    """
    return squeeze_impl(input, dim)


def swapaxes(input, axis0, axis1):
    '''
    Alias for :func:`mindspore.mint.transpose` . The `input` corresponds to the `input` in the reference interface,
    and the parameters `axis0` and `axis1` correspond to `dim0` and `dim1` in the reference interface respectively.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import mint
        >>> from mindspore import Tensor
        >>> input = Tensor(np.ones((2,3,4), dtype=np.float32))
        >>> output = mint.swapaxes(input, 0, 2)
        >>> print(output.shape)
        (4, 3, 2)
    '''
    return transpose(input, axis0, axis1)


def unique_consecutive(input, return_inverse=False, return_counts=False, dim=None):
    r"""
    Returns the elements that are unique in each consecutive group of equivalent elements in the input tensor.

    When `return_inverse=True` , it returns a tensor containing the indices of the elements in the input tensor
    within the output tensor.

    When `return_counts=True` , it returns a tensor representing the number of occurrences of each output element
    in the input.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The input tensor.
        return_inverse (bool, optional): Whether to return the index of where the element in the original input
            maps to the position in the output. Default: ``False`` .
        return_counts (bool, optional): Whether to return the counts of each unique element. Default: ``False`` .
        dim (int, optional): The dimension to apply unique. If ``None`` , the unique of the flattened input is
            returned. If the dimension is specified, it must be int32 or int64. Default: ``None`` .

    Returns:
        A tensor or a tuple of tensors containing tensor objects (`output`, `inverse_indices`, `counts`).

        - **output** (Tensor): the output tensor has the same type as `input` and represents the output list of
          unique scalar elements.
        - **inverse_indices** (Tensor, optional): if `return_inverse` is True, there will be an additional returned
          tensor `inverse_indices`. `inverse_indices` has the same shape as `input` and represents the index of where
          the element in the original input maps to the position in the output.
        - **counts** (Tensor, optional):  if `return_counts` is True, there will be an additional returned tensor
          `counts`. `counts` has the same shape as `output` or `output.shape[dim]` if dim was specified and represents
          the number of occurrences for each unique value or tensor.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If dtype of `input` is not supported.
        TypeError: If `return_inverse` is not a bool.
        TypeError: If `return_counts` is not a bool.
        TypeError: If `dim` is not an int.
        ValueError: If `dim` is not in the range of :math:`[-ndim, ndim-1]`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> from mindspore import dtype as mstype
        >>> x = Tensor(np.array([1, 1, 2, 2, 3, 1, 1, 2]), mstype.int64)
        >>> output, inverse_indices, counts = mint.unique_consecutive(x, True, True, None)
        >>> print(output)
        [1 2 3 1 2]
        >>> print(inverse_indices)
        [0 0 1 1 2 3 3 4]
        >>> print(counts)
        [2 2 1 2 1]
    """

    return ops.function.array_func.unique_consecutive(input, return_inverse, return_counts, dim)


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
        TypeError: If `size` is neither an int nor a tuple/list/Tensor of int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import mint
        >>> output = mint.zeros((2, 2), dtype=mindspore.float32)
        >>> print(output)
        [[0. 0.]
         [0. 0.]]
    """
    return ops.auto_generate.zeros(size, dtype)


def fix(input):
    """
    Alias for :func:`mindspore.mint.trunc` .

    Supported Platforms:
        ``Ascend``
    """
    return trunc(input)


def scatter(input, dim, index, src):
    """
    Update the value in `src` to `input` according to the specified index.
    For a 3-D tensor, the output will be:

    .. code-block::

        output[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0

        output[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1

        output[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

    .. note::
        The backward is supported only for the case `src.shape == index.shape` when `src` is a tensor.

    Args:
        input (Tensor): The target tensor. The rank of `input` must be at least 1.
        dim (int): Which axis to scatter. Accepted range is [-r, r) where r = rank(input).
        index (Tensor): The index to do update operation whose data must be positive number with type of mindspore.int32
            or mindspore.int64. Same rank as `input` . And accepted range is [-s, s) where s is the size along axis.
        src (Tensor, float): The data doing the update operation with `input`. Can be a tensor with the same data type
            as `input` or a float number to scatter.

    Returns:
        Tensor, has the same shape and type as `input` .

    Raises:
        TypeError: If `index` is neither int32 nor int64.
        ValueError: If rank of any of `input` , `index` and `src` is less than 1.
        ValueError: If the rank of `src` is not equal to the rank of `input` .
        TypeError: If the data types of `input` and `src` have different dtypes.
        RuntimeError: If `index` has negative elements.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor, mint
        >>> input = Tensor(np.array([[1, 2, 3, 4, 5]]), dtype=ms.float32)
        >>> src = Tensor(np.array([[8, 8]]), dtype=ms.float32)
        >>> index = Tensor(np.array([[2, 4]]), dtype=ms.int64)
        >>> out = mint.scatter(input=input, dim=1, index=index, src=src)
        >>> print(out)
        [[1. 2. 8. 4. 8.]]
        >>> input = Tensor(np.zeros((5, 5)), dtype=ms.float32)
        >>> src = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=ms.float32)
        >>> index = Tensor(np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]]), dtype=ms.int64)
        >>> out = mint.scatter(input=input, dim=0, index=index, src=src)
        >>> print(out)
        [[1. 2. 3. 0. 0.]
        [0. 0. 0. 0. 0.]
        [4. 5. 6. 0. 0.]
        [0. 0. 0. 0. 0.]
        [7. 8. 9. 0. 0.]]
        >>> input = Tensor(np.zeros((5, 5)), dtype=ms.float32)
        >>> src = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=ms.float32)
        >>> index = Tensor(np.array([[0, 2, 4], [0, 2, 4], [0, 2, 4]]), dtype=ms.int64)
        >>> out = mint.scatter(input=input, dim=1, index=index, src=src)
        >>> print(out)
        [[1. 0. 2. 0. 3.]
        [4. 0. 5. 0. 6.]
        [7. 0. 8. 0. 9.]
        [0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0.]]
    """
    return ops.function.array_func.scatter(input, dim, index, src)


def cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary'):
    """
    Computes p-norm distance between each pair of row vectors of two input Tensors.

    .. warning::
        This is an experimental optimizer API that is subject to change.

    Note:
        On Ascend, the supported dtypes are float16 and float32.

    Args:
        x1 (Tensor): Input tensor of shape :math:`(B, P, M)`.
            Letter :math:`B` represents 0 or positive int number.
            When :math:`B` is equal to 0, it means this dimension can be ignored,
            i.e. shape of the tensor is :math:`(P, M)`.
        x2 (Tensor): Input tensor of shape :math:`(B, R, M)`, has the same dtype as `x1`.
        p (float, optional): P value for the p-norm distance to calculate between each
            vector pair, P >= 0. Default: ``2.0`` .
        compute_mode (str, optional): Specify the cumpute mode. Setting this parameter currently has no effect.
            Default: ``'use_mm_for_euclid_dist_if_necessary'`` .

    Returns:
        Tensor, p-norm distance, has the same dtype as `x1`, its shape is :math:`(B, P, R)`.

    Raises:
        TypeError: If `x1` or `x2` is not Tensor.
        TypeError: If dtype of `x1` or `x2` is not listed in the "Note" above.
        TypeError: If `p` is not float32.
        ValueError: If `p` is negative.
        ValueError: If dimension of `x1` is not the same as `x2`.
        ValueError: If dimension of `x1` or `x2` is neither 2 nor 3.
        ValueError: If the batch dim of `x1` and `x2` can not broadcast.
        ValueError: If the number of columns of `x1` is not the same as that of `x2`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
        >>> y = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
        >>> output = ops.cdist(x, y, 2.0)
        >>> print(output)
        [[[2.8284273 2.8284273]
          [1.4142137 1.4142137]]]
    """
    return cdist_(x1, x2, p)


@jit_view_unsupported
def flatten(input, start_dim=0, end_dim=-1):
    """
    Flatten a tensor along dimensions from `start_dim` to `end_dim`.

    Args:
        input (Tensor): The input Tensor.
        start_dim (int, optional): The first dimension to flatten. Default: ``0`` .
        end_dim (int, optional): The last dimension to flatten. Default: ``-1`` .

    Returns:
        Tensor. If no dimensions are flattened, returns the original `input`, otherwise return the flattened Tensor.
        If `input` is a 0-dimensional Tensor, a 1-dimensional Tensor will be returned.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `start_dim` or `end_dim` is not int.
        ValueError: If `start_dim` is greater than `end_dim` after canonicalized.
        ValueError: If `start_dim` or `end_dim` is not in range of [-input.dim, input.dim-1].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> input_x = Tensor(np.ones(shape=[1, 2, 3, 4]), mindspore.float32)
        >>> output = mindspore.mint.flatten(input_x)
        >>> print(output.shape)
        (24,)
    """

    return flatten_ext(input, start_dim, end_dim)


@jit_view_unsupported
def reshape(input, shape):
    """
    Reshape the input tensor based on the given shape.

    .. note::
        The -1 in the parameter `shape` indicates that the size of that dimension is inferred from the other
        dimensions and the total number of elements in input tensor.

    Args:
        input (Tensor): The input tensor.
        shape (Union[tuple[int], list[int], Tensor[int]]): New shape.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]], mindspore.float32)
        >>> # case1: Parameter `shape` does not contain -1.
        >>> output = mindspore.mint.reshape(input, (3, 2))
        >>> print(output)
        [[-0.1  0.3]
         [ 3.6  0.4]
         [ 0.5 -3.2]]
        >>> # case2: Parameter `shape` contains -1.
        >>> output = mindspore.mint.reshape(input, (-1, 6))
        >>> print(output)
        [[-0.1  0.3  3.6  0.4  0.5 -3.2]]
    """

    return mindspore.ops.function.array_func.reshape(input, shape)


@jit_view_unsupported
def t(input):
    """
    Transpose the input tensor.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor, transpose 2D tensor, return 1D tensor as it is.

    Raises:
        ValueError: If the dimension of `input` is greater than 2.
        ValueError: If `input` is empty.
        TypeError: If `input` is not a tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> input = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), mindspore.float32)
        >>> output = mindspore.mint.t(input)
        >>> print(output)
        [[ 1. 4.]
         [ 2. 5.]
         [ 3. 6.]]
    """
    return mindspore.ops.auto_generate.t_ext(input)


__all__ = [
    'conv2d',
    'full',
    'ones_like',
    'zeros_like',
    'abs',
    'clone',
    'erf',
    'where',
    'isclose',
    'empty',
    'empty_like',
    'full_like',
    # 1
    'div',
    'divide',
    'topk',
    'roll',
    # 2
    'sin',
    # 3
    'clamp',
    'xlogy',
    'fmod',
    # 4
    'sinc',
    'sinh',
    'cosh',
    # 5
    'cumsum',
    # 6
    'stack',
    # 7
    'zeros',
    # 8
    'transpose',
    'swapaxes',
    "batch_norm_elemt",
    "batch_norm_gather_stats_with_counts",
    "batch_norm_stats",
    # 9
    'squeeze',
    # 10
    'ne',
    'not_equal',
    # 11
    'unsqueeze',
    # 12
    "repeat_interleave",
    # 13
    "flip",
    # 14
    'mv',
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
    'cumprod',
    # 23
    'exp2',
    # 24
    'cdist',
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
    'cummax',
    'cummin',
    'einsum',
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
    'concat',
    # 46
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    # 47
    'max',
    # 48
    'min',
    # 49
    'baddbmm',
    # 50
    'tile',
    # 51
    'permute',
    # 52
    'addcdiv',
    # 53

    # 54
    'normal',
    # 55
    'cross',
    # 56
    'norm',
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
    'ravel',
    # 65
    'logical_and',
    # 66
    'logical_not',
    # 67
    'logical_or',
    # 68
    'logical_xor',
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
    'masked_select',

    # 86
    'select',

    # 87

    # 88
    'chunk',
    # 89
    'argsort',
    # 90
    'isinf',
    # 91

    # 92
    'polar',
    # 93

    # 94
    'tanh',
    # 95

    # 96

    # 97

    # 98

    # 99

    # 100

    # 101

    # 102

    # 103

    # 104

    # 105

    # 106

    # 107

    # 108

    # 109
    'argmin',
    # 110
    'softmax',
    # 111

    # 112

    # 113

    # 114

    # 115

    # 116

    # 117

    # 118

    # 119

    # 120
    'isneginf',
    # 121

    # 122

    # 123
    'var',

    # 151
    'acos',
    'arccos',
    # 152
    'acosh',
    'arccosh',
    # 153

    # 154

    # 155

    # 156

    # 157
    'scatter',
    # 172
    'asin',
    'arcsin',
    # 173
    'asinh',
    'arcsinh',
    # 174
    'atan',
    'arctan',
    # 175
    'atanh',
    'arctanh',
    # 176
    'atan2',
    'arctan2',

    # 177
    'round',

    # 182
    'bernoulli',
    # 201
    'diag',
    # 207
    'expm1',
    # 204
    'erfc',
    # 208
    'eye',
    # 239
    'lerp',
    # 256
    'median',
    'randperm',
    'rand',
    'rand_like',
    'randn',
    'randn_like',
    'randint',
    'randint_like',
    # 210
    'floor',
    # 231
    'inverse',
    # 244
    'log1p',
    # 261
    'multinomial',
    # 275
    'remainder',
    # 285
    'scatter_add',
    # 289
    'sign',
    # 301
    'tan',
    # 303
    'trace',
    'gcd',
    'reshape',
    'outer',
    # 304
    'tril',

    # 305
    'triu',

    # 308
    'mm',

    # 382
    'dstack',

    # 406
    'allclose',

    # 501
    'addbmm',

    # 502
    'addmm',

    # 505
    'addmv',

    # 510
    'amax',

    # 511
    'amin',

    # 520
    'bincount',

    # 521
    'bitwise_not',

    # 526
    'dot',

    # 533
    'frac',

    # 538
    'histc',

    # 549
    'kthvalue',

    # 552
    'log10',

    # 553
    'logaddexp',
    'logaddexp2',

    # 557
    'logsumexp',

    # 582
    'std_mean',

    # 584
    'take',

    # 588
    'var_mean',

    # 586
    'unique_consecutive',

    # 610
    'nan_to_num',

    # 613
    'nansum',

    # 615
    'triangular_solve',

    # 664
    'meshgrid',

    # 695
    'count_nonzero',

    # 697
    'float_power',

    # 708
    'std',

    # 739
    'hstack',

    # 826
    'floor_divide',

    # 887
    'log2',

    # 889
    'isnan',

    # 890

    # 891

    # 892

    # 893

    # 894

    # 895

    # 896

    # 897

    # 898

    # 899

    # 900

    # 916
    'index_add',

    # 1007
    't',

    # 1023
    'unbind',

    # 1100
    'diff',

    # 1101
    'real',
    # 1102
    'imag',
]

__all__.extend(functional.__all__)
__all__.extend(nn.__all__)
__all__.extend(optim.__all__)
__all__.extend(linalg.__all__)
__all__.extend(special.__all__)
__all__.extend(distributed.__all__)
