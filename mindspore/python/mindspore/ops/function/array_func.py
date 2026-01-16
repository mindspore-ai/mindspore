# Copyright 2022-2023 Huawei Technologies Co., Ltd
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

"""Operators for function."""
from __future__ import absolute_import

import builtins
import operator
import numbers
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.primitive import constexpr
from mindspore.ops.primitive import _primexpr
from mindspore import ops
from mindspore.ops.operations._inner_ops import DynamicBroadcastTo
from mindspore.ops.operations._sequence_ops import TupleToTensor
from mindspore.ops.composite.multitype_ops import _constexpr_utils as const_utils
from mindspore.ops.operations._sequence_ops import TensorToList
# 1
from mindspore.ops.auto_generate import OnesLikeExt, ZerosLikeExt, FillScalar, FillTensor, Arange, UniqueDim, \
    Unique2, SortExt, NonZero, NonZeroExt, Scatter, ScatterValue, NewOnes, NewZeros
# 2
from mindspore.ops.auto_generate.pyboost_inner_prim import squeeze_impl
# 3

# 4

# 5

# 6

# 7

# 8

# 9

# 10

# 11

# 12

# 13

# 14

# 15

# 16

# 17

# 18

# 19

# 20

from mindspore.ops.auto_generate.gen_ops_prim import SplitTensor, Meshgrid
from mindspore.ops.auto_generate.gen_ops_prim import SplitWithSize, RepeatInterleaveInt, RepeatInterleaveTensor
from mindspore.ops.auto_generate.pyboost_inner_prim import _PyboostSearchSortedPrim, meshgrid_impl, concat_impl, \
    unique_consecutive_impl
from mindspore.ops.operations.array_ops import (
    MatrixDiagV3,
    MatrixDiagPartV3,
    MatrixSetDiagV3,
    Fills,
    Col2Im,
    ScatterNdMax,
    ScatterNdMul,
    IndexFill,
    AffineGrid,
    Im2Col,
    Expand,
    Lstsq,
    Mvlgamma,
    Tril,
    Argmax,
    ArgMaxWithValue,
    ArgMinWithValue
)
from mindspore.common import Tensor
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore import _checkparam as validator
from mindspore.ops._utils.utils import ms_arrange

from mindspore.ops.auto_generate import cat, range, scatter_nd, deepcopy, masked_fill, diagonal, expand_dims, \
    flip, transpose, triu, unsorted_segment_sum, diag, gather, gather_d, gather_nd, reshape, masked_select, \
    broadcast_to, strided_slice, ones, zeros, max_, min_, select, zero_, view_as, \
    expand_as, unstack_ext_view_op, full_like_op, tensor_scatter_add, \
    index_fill_scalar, index_fill_tensor
from mindspore.ops.auto_generate import take, tensor_scatter_elements as tensor_scatter_elements_ext
from mindspore.ops.auto_generate.gen_ops_prim import scatter_add_ext_op, gather_d_op, slice_op, tril_ext_op, \
    split_tensor_op, split_with_size_op, chunk_op
from mindspore.ops.operations.manually_defined import tile, rank, scalar_cast
from mindspore.ops.auto_generate.pyboost_inner_prim import _PyboostOneHotExtPrim
from mindspore.common._decorator import deprecated

arg_max_with_value_ = ArgMaxWithValue()
arg_min_with_value_ = ArgMinWithValue()
batch_to_space_nd_v2_ = P.BatchToSpaceNDV2()
cast_ = P.Cast()
diag_ = P.Diag()
dynamic_broadcast_to_ = DynamicBroadcastTo()
eye_ = P.Eye()
fills_ = Fills()
fillv2_ = P.FillV2()
flatten_ = P.Flatten()
gather_ = P.Gather()
gather_d_ = P.GatherD()
gather_nd_ = P.GatherNd()
ger_ = P.Ger()
index_fill_ = IndexFill()
lstsq_ = Lstsq()
matrix_band_part_ = P.array_ops.MatrixBandPart()
ones_ = P.Ones()
population_count_ = P.PopulationCount()
range_ = P.Range()
rank_ = P.Rank()
reduce_max_ = P.ReduceMax()
reduce_min_ = P.ReduceMin()
reshape_ = P.Reshape()
scalar_to_tensor_ = P.ScalarToTensor()
scatter_add_ = P.ScatterAdd()
scatter_div_ = P.ScatterDiv()
scatter_max_ = P.ScatterMax()
scatter_min_ = P.ScatterMin()
scatter_mul_ = P.ScatterMul()
scatter_nd_ = P.ScatterNd()
scatter_update_ = P.ScatterUpdate()
search_sorted_ = _PyboostSearchSortedPrim()
shape_ = P.Shape()
split_tensor = SplitTensor()
split_with_size = SplitWithSize()
size_ = P.Size()
tensor_scatter_div_ = P.TensorScatterDiv()
tensor_scatter_max_ = P.TensorScatterMax()
tensor_scatter_min_ = P.TensorScatterMin()
tensor_scatter_mul_ = P.TensorScatterMul()
tensor_scatter_sub_ = P.TensorScatterSub()
tensor_select_ = P.Select()
tensor_shape_ = P.TensorShape()
tensor_slice = slice_op
tile_ = P.Tile()
transpose_ = P.Transpose()
type_as_ = P.TypeAs()
tuple_to_array_ = P.TupleToArray()
tuple_to_tensor_ = TupleToTensor()
unique_ = P.Unique()
unsorted_segment_max_ = P.UnsortedSegmentMax()
unsorted_segment_min_ = P.UnsortedSegmentMin()
unsorted_segment_prod_ = P.UnsortedSegmentProd()
unsorted_segment_sum_ = P.UnsortedSegmentSum()
ones_like_ = P.OnesLike()
one_hot_ext_impl = _PyboostOneHotExtPrim()
zeros_like_ = P.ZerosLike()
ones_like_ext_ = OnesLikeExt()
zeros_like_ext_ = ZerosLikeExt()
new_ones_ = NewOnes()
new_zeros_ = NewZeros()
fill_scalar_ = FillScalar()
fill_tensor_ = FillTensor()
sort_ext_ = SortExt()
scatter_prim = Scatter()
scatter_value_ = ScatterValue()
arange_ = Arange()
repeat_interleave_int_ = RepeatInterleaveInt()
repeat_interleave_tensor_ = RepeatInterleaveTensor()
unique_dim_ = UniqueDim()
unique2_ = Unique2()
non_zero_ = NonZero()
non_zero_ext_ = NonZeroExt()


@_primexpr
def get_x_shape(x_shape):
    if ops.is_sequence_shape_unknown(x_shape):
        return (-2,)
    if ops.is_sequence_value_unknown(x_shape):
        return (-1,)
    s = 1
    for i in x_shape:
        s = s * i
    return (s,)


@constexpr
def _check_attr_dtype(param_name, input_dtype, allow_dtypes, cls_name):
    validator.check_value_type(param_name, input_dtype, allow_dtypes, cls_name)


check_flatten_order_const = constexpr(validator.check_flatten_order)


##############################
# Tensor Creation Functions.
##############################


def _cast_type(x, to_type):
    """cast input to the specified type or cast input to tensor"""
    if isinstance(x, Tensor):
        x = cast_(x, to_type)
    else:
        x = scalar_to_tensor_(x, to_type)
    return x


def _get_type(x):
    """get the dtype of input"""
    if isinstance(x, Tensor):
        return x.dtype
    return ops.typeof(x)


def _get_max_type(start, end, step):
    """get max input type with `level`"""
    valid_dtypes = [mstype.int32, mstype.float32, mstype.int64, mstype.float64]
    arg_map = [start, end, step]
    arg_type_map = [str(_get_type(i)) for i in arg_map]
    for arg_value in arg_map:
        if not (isinstance(arg_value, (float, int))
                or (isinstance(arg_value, Tensor) and arg_value.dtype in valid_dtypes)):
            raise TypeError(
                f"For arange, the input type must be int or float or a TensorScalar in {valid_dtypes},"
                f" but got {_get_type(arg_value)}")

    type_map = {'Float64': '3', 'Float32': '2', "<class 'float'>": '2', 'Int64': '1', "<class 'int'>": '1',
                'Int32': '0'}
    type_map_reverse = {'3': mstype.float64,
                        '2': mstype.float32, '1': mstype.int64, '0': mstype.int32}
    type_level = [type_map.get(i) for i in arg_type_map]
    max_level = builtins.max(type_level)
    return type_map_reverse.get(max_level)


def arange(start=0, end=None, step=1, *, dtype=None):
    r"""
    Returns a tensor with a step length of `step` in the interval [ `start` , `end` ).

    Args:
        start (Union[float, int, Tensor], optional): The start value of the interval.
            Default ``0`` .
        end (Union[float, int, Tensor], optional): The end value of the interval.
            Default ``None`` represents to the value of `start` , and `0` is used as the start value.
        step (Union[float, int, Tensor], optional): The interval between each value. Default ``1`` .

    Keyword Args:
        dtype (mindspore.dtype, optional): The data type specified. Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> mindspore.ops.arange(4)
        Tensor(shape=[4], dtype=Int64, value= [0, 1, 2, 3])
        >>> mindspore.ops.arange(1, 6)
        Tensor(shape=[5], dtype=Int64, value= [1, 2, 3, 4, 5])
        >>> mindspore.ops.arange(0, 2, 0.5)
        Tensor(shape=[4], dtype=Float32, value= [ 0.00000000e+00,  5.00000000e-01,  1.00000000e+00,  1.50000000e+00])
    """
    if end is None:
        start, end = 0, start
    max_type = _get_max_type(start, end, step)
    start = _cast_type(start, max_type)
    end = _cast_type(end, max_type)
    step = _cast_type(step, max_type)

    if start.shape != () or end.shape != () or step.shape != ():
        raise ValueError(f"For arange, the input args must be a TensorScalar,"
                         f" but got start shape:{start.shape}, end shape:{end.shape}, step shape:{step.shape}")
    data = range_(start, end, step)
    if dtype is not None:
        data = cast_(data, dtype)
    return data


def arange_ext(start=0, end=None, step=1, *, dtype=None):
    r"""
    Creates a sequence of numbers that begins at `start` and extends by increments of
    `step` up to but not including `end`.

    Args:
        start (Union[float, int], optional): The start of the interval. Default: ``0`` .
        end (Union[float, int], optional): The end of the interval, exclusive.
            Default: ``None`` . If ``None`` , it defaults to the value of `start`, and 0 is used as the starting value.
        step (Union[float, int], optional): The step size with which the array element increments. Default: ``1`` .

    Keyword Args:
        dtype (mindspore.dtype, optional): The required data type of returned Tensor. Default: ``None`` .
            When `dtype` is not specified or ``None``:

            If `start`, `end`, and `step` are all integers, the dtype of output is int64,

            If `start`, `end`, and `step` contain at least one floating-point number, the dtype of output is float32.

    Returns:
        A 1-D Tensor, cast to `dtype` if provided, may potentially lose precision due to casting.

    Raises:
        TypeError: If `start`, `end` or `step` are not of type int or float.
        ValueError: If `step` = 0.
        ValueError: If `start` >= `end` when `step` > 0.
        ValueError: If `start` <= `end` when `step` < 0.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor, ops
        >>> output = ops.arange_ext(1, 6)
        >>> print(output)
        [1 2 3 4 5]
        >>> print(output.dtype)
        Int64
        >>> output = ops.arange_ext(0, 3, 1.2)
        >>> print(output)
        [0.  1.2 2.4]
        >>> print(output.dtype)
        Float32
        >>> output = ops.arange_ext(7, 1, -2)
        >>> print(output)
        [7 5 3]
        >>> print(output.dtype)
        Int64
        >>> output = ops.arange_ext(12, 2, -1, dtype=ms.bfloat16)
        >>> print(output)
        [12. 11. 10.  9.  8.  7.  6.  5.  4.  3.]
        >>> print(output.dtype)
        BFloat16
    """
    if end is None:
        start, end = 0, start
    return arange_(start, end, step, dtype)


def concat(tensors, axis=0):
    """
    Alias for :func:`mindspore.ops.cat()`.

    Tutorial Examples:
        - `Tensor - Tensor Operation <https://mindspore.cn/tutorials/en/master/beginner/tensor.html#tensor-operation>`_
        - `Vision Transformer Image Classification - Building ViT as a whole
          <https://mindspore.cn/tutorials/en/master/cv/vit.html#building-vit-as-a-whole>`_
        - `Sentiment Classification Implemented by RNN - Dense
          <https://mindspore.cn/tutorials/en/master/nlp/sentiment_analysis.html#dense>`_
    """
    return cat(tensors, axis)


def eye(n, m=None, dtype=None):
    """
    Returns a tensor with ones on the diagonal and zeros in the rest.

    Args:
        n (int): The number of rows returned.
        m (int, optional): The number of columns returned. If ``None`` , the number of columns is as the same as n.
        dtype (mindspore.dtype, optional): The data type returned.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> output = mindspore.ops.eye(3)
        >>> print(output)
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
        >>>
        >>> output = mindspore.ops.eye(3, 4)
        >>> print(output)
        [[1. 0. 0. 0.]
         [0. 1. 0. 0.]
         [0. 0. 1. 0.]]
    """
    if m is None:
        m = n
    if dtype is None:
        dtype = ms.float32
    return eye_(n, m, dtype)


def hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, *, dtype=None):
    r"""
    Hamming window function.

    .. math::

        w[n]=\alpha − \beta \cos \left( \frac{2 \pi n}{N - 1} \right),

    where :math:`N` is the full window size, and n is natural number less than :math:`N` :[0, 1, ..., N-1].

    Args:
        window_length (int): The size of window.
        periodic (bool, optional): If ``True`` , return a periodic window. If ``False``, return a symmetric window.
            Default ``True`` .
        alpha (float, optional): The coefficient α. Default ``0.54`` .
        beta (float, optional): The coefficient β. Default ``0.46`` .

    Keyword Args:
        dtype (mindspore.dtype, optional): The data type specified. Default ``None`` .

    Returns:
        A 1-D tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> output = mindspore.ops.hamming_window(5)
        >>> print(output)
        [0.08000001 0.3978522  0.9121478  0.9121478  0.3978522 ]
        >>> output = mindspore.ops.hamming_window(5, periodic=False)
        >>> print(output)
        [0.08000001 0.54       1.         0.54       0.08000001]
    """
    if not isinstance(window_length, int):
        raise TypeError(f"For array function 'hamming_window', 'window_length' must be int, but got"
                        f" {type(window_length)}.")
    if window_length < 0:
        raise ValueError("For array function 'hamming_window', 'window_length' must be non negative number.")
    if not isinstance(periodic, bool):
        raise TypeError(
            f"For array function 'hamming_window', 'periodic' must be bool, but got {type(periodic)}.")
    if not isinstance(alpha, float):
        raise TypeError(
            f"For array function 'hamming_window', 'alpha' must be float, but got {type(alpha)}.")
    if not isinstance(beta, float):
        raise TypeError(
            f"For array function 'hamming_window', 'beta' must be float, but got {type(beta)}.")
    if window_length <= 1:
        return Tensor(np.ones(window_length))
    if dtype is not None and dtype not in mstype.float_type:
        raise TypeError(
            f"For array function 'hamming_window', 'dtype' must be floating point dtypes, but got {dtype}.")

    dtype = mstype.float32 if dtype is None else dtype
    op = _get_cache_prim(P.HammingWindow)(periodic, alpha, beta, dtype)
    length = Tensor(np.array([window_length]).astype(np.int32))
    out = op(length)
    return out


def where(condition, input, other):
    r"""
    Return a tensor in which the elements are selected from `input` or `other` based on the `condition`.

    Support broadcasting.

    .. math::
        output_i = \begin{cases} input_i,\quad &if\ condition_i \\ other_i,\quad &otherwise \end{cases}

    Args:
        condition (Tensor[bool]): If True, yield `input`, otherwise yield `other`.
        input (Union[Tensor, Scalar]): When `condition` is True, values to select from.
        other (Union[Tensor, Scalar]): When `condition` is False, values to select from.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[0, 1],
        ...                           [2, 3]])
        >>> other = mindspore.tensor([[1, 1],
        ...                           [1, 1]])
        >>> condition = input < 3
        >>> mindspore.ops.where(condition, input, other)
        Tensor(shape=[2, 2], dtype=Int64, value=
        [[0, 1],
         [2, 1]])
    """
    return tensor_select_(condition, input, other)


def reverse(x, axis):
    """
    This interface will be deprecated in the future, and use :func:`mindspore.ops.flip` instead.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return flip(x, axis)


def ravel(input):
    """
    Expand the multidimensional Tensor into 1D along the 0 axis direction.

    Args:
        input (Tensor): A tensor to be flattened.

    Returns:
        Tensor, a 1-D tensor, containing the same elements of the input.

    Raises:
        TypeError: If argument `input` is not Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
        >>> output = ops.ravel(x)
        >>> print(output)
        [0. 1. 2. 1.]
        >>> print(output.shape)
        (4,)
    """
    return ops.reshape(input, (-1,))


def matrix_band_part(x, lower, upper):
    r"""
    Copy a tensor setting everything outside a central band in each innermost matrix to zero.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        x (Tensor): The input tensor.
        lower (Union[int, Tensor]): Number of subdiagonals to keep. If negative, keep entire lower triangle.
        upper (Union[int, Tensor]): Number of superdiagonals to keep. If negative, keep entire upper triangle.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.ops.ones([2, 4, 4])
        >>> output = mindspore.ops.matrix_band_part(x, 2, 1)
        >>> print(output)
        [[[1. 1. 0. 0.]
          [1. 1. 1. 0.]
          [1. 1. 1. 1.]
          [0. 1. 1. 1.]]
         [[1. 1. 0. 0.]
          [1. 1. 1. 0.]
          [1. 1. 1. 1.]
          [0. 1. 1. 1.]]]
    """
    return matrix_band_part_(x, lower, upper)


@deprecated("2.8.0", "ops.pad", False, "ops.")
def padding(x, pad_dim_size=8):
    r"""
    `ops.padding` is deprecated from version 2.8.0 and will be removed in a
    future version, please use :func:`mindspore.ops.pad` instead.

    Extends the last dimension of the input tensor from 1 to pad_dim_size, by filling with 0.

    Args:
        x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. The rank of `x` must be at least 2.
            The last dimension of `x` must be 1. The data type is Number.
        pad_dim_size (int): The value of the last dimension of `x` to be extended, which must be positive.
            Default: ``8`` .

    Returns:
        Tensor, has the same type and shape as input shape value.

    Raises:
        TypeError: If `pad_dim_size` is not an int.
        ValueError: If `pad_dim_size` is less than 1.
        ValueError: If last dim of `x` is not equal to 1.

    Supported Platforms:
        Deprecated

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([[8], [10]]), mindspore.float32)
        >>> pad_dim_size = 4
        >>> output = ops.padding(x, pad_dim_size)
        >>> print(output)
        [[ 8.  0.  0.  0.]
         [10.  0.  0.  0.]]
    """
    padding_ = _get_cache_prim(P.array_ops.Padding)(pad_dim_size)
    return padding_(x)


@constexpr
def _check_axis_type(axis, type_int=True, type_tuple=True, type_list=True, ops_name="ops"):
    """Check axis argument type."""
    if type_int and isinstance(axis, int):
        return True
    if (type_tuple and isinstance(axis, tuple)) or (type_list and isinstance(axis, list)):
        for ax in axis:
            if not isinstance(ax, int):
                raise TypeError(
                    f"For {ops_name}, each axis must be integer, but got {type(ax)} in {axis}.")
        return True

    type_str = ""
    if type_int:
        type_str += "int, "
    if type_tuple:
        type_str += "tuple, "
    if type_list:
        type_str += "list, "
    raise TypeError(
        f"For {ops_name}, the axis should be {type_str}, but got {type(axis)}.")


def one_hot(indices, depth, on_value=1, off_value=0, axis=-1):
    r"""
    Generate a new tensor, where the positions specified by `indices` are assigned `on_value`, and all
    other positions are assigned `off_value`.

    Note:
        If the input `indices` has rank `N`, the output will have rank `N+1`.
        The new axis is created at dimension `axis`. On Ascend, if `on_value` is int64 dtype, `indices` must be
        int64 dtype, and the value for `on_value` and `off_value` can only be 1 and 0.

    Args:
        indices(Tensor): The input tensor of indices.
        depth(int): The depth of the one-hot.
        on_value(Union[Tensor, int, float], optional): The value used to fill indexed positions. Default ``1`` .
        off_value(Union[Tensor, int, float], optional): The value used to fill non-indexed positions. Default ``0`` .
        axis(int, optional): Specify the axis for computation. Default ``-1`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> indices = mindspore.tensor([0, 1, 2, 4])
        >>> mindspore.ops.one_hot(indices, depth=5)
        Tensor(shape=[4, 5], dtype=Int64, value=
        [[1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1]])
        >>>
        >>> mindspore.ops.one_hot(indices, depth=3)
        Tensor(shape=[4, 3], dtype=Int64, value=
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1],
         [0, 0, 0]])
        >>> # If shape of indices is (N, C), and axis=-1, the returned shape will be (N, C, depth).
        >>> indices = mindspore.tensor([[0, 2], [1, -1]])
        >>> mindspore.ops.one_hot(indices, depth=3, on_value=10, off_value=4, axis=-1)
        Tensor(shape=[2, 2, 3], dtype=Int64, value=
        [[[10,  4,  4],
          [ 4,  4, 10]],
         [[ 4, 10,  4],
          [ 4,  4,  4]]])
        >>> # If axis=0, the returned shape will be (depth, N, C).
        >>> mindspore.ops.one_hot(indices, depth=3, on_value=10, off_value=4, axis=0)
        Tensor(shape=[3, 2, 2], dtype=Int64, value=
        [[[10,  4],
          [ 4,  4]],
         [[ 4,  4],
          [10,  4]],
         [[ 4, 10],
          [ 4,  4]]])
    """
    if not isinstance(on_value, Tensor):
        on_value = Tensor(on_value)
    if not isinstance(off_value, Tensor):
        off_value = Tensor(off_value)
    onehot = _get_cache_prim(P.OneHot)(axis)
    return onehot(indices, depth, on_value, off_value)


def fill(type, shape, value):  # pylint: disable=redefined-outer-name
    """
    Create a tensor filled with a specified value.

    Args:
        type (mindspore.dtype): The data type specified.
        shape (Union(Tensor, tuple[int])): The shape specified.
        value (Union(Tensor, number.Number, bool)): The value specified.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> mindspore.ops.fill(mindspore.float32, (2, 3),  1.2)
        Tensor(shape=[2, 3], dtype=Float32, value=
        [[ 1.20000005e+00,  1.20000005e+00,  1.20000005e+00],
         [ 1.20000005e+00,  1.20000005e+00,  1.20000005e+00]])
    """
    value = cast_(value, type)
    return fillv2_(shape, value)


def full(size, fill_value, *, dtype=None):  # pylint: disable=redefined-outer-name
    """
    Create a tensor filled with a specified value.

    Note:
        `fill_value` 's data type not support complex numbers.

    Args:
        size (Union(tuple[int], list[int])): The shape specified.
        fill_value (number.Number): The value specified.

    Keyword Args:
        dtype (mindspore.dtype): The data type specified. Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> mindspore.ops.full((2, 3), 1.2)
        Tensor(shape=[2, 3], dtype=Int64, value=
        [[1, 1, 1],
         [1, 1, 1]])
        >>> mindspore.ops.full((2, 3), 1.2, dtype=mindspore.float32)
        Tensor(shape=[2, 3], dtype=Float32, value=
        [[ 1.20000005e+00,  1.20000005e+00,  1.20000005e+00],
         [ 1.20000005e+00,  1.20000005e+00,  1.20000005e+00]])
    """
    if not isinstance(size, (list, tuple)):
        raise TypeError(
            f"For 'ops.full', 'size' must be a tuple or list of ints, but got {type(size)}.")
    if dtype is None:
        dtype = mstype.int64
    if dtype not in mstype.all_types:
        raise TypeError(
            f"For 'ops.full', 'dtype' must be mindspore.type, but got {dtype}.")
    if isinstance(size, list):
        size = tuple(size)
    return ops.fill(dtype, size, fill_value)


def full_ext(size, fill_value, *, dtype=None):  # pylint: disable=redefined-outer-name
    """
    Create a Tensor of the specified shape and fill it with the specified value.

    Args:
        size (Union(tuple[int], list[int])): The specified shape of output tensor.
        fill_value (Union(number.Number, Tensor)): Value to fill the returned tensor. It can be a Scalar number, a 0-D
            Tensor, or a 1-D Tensor with only one element.

    Keyword Args:
        dtype (mindspore.dtype, optional): The specified type of output tensor.
            `bool` and `number` are supported, for details,
            please refer to :class:`mindspore.dtype` . Default: ``None`` .

    Returns:
        Tensor.

    Raises:
        TypeError: If `size` is not a tuple or list.
        ValueError: The element in `size` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> output = ops.full_ext((2, 2), 1)
        >>> print(output)
        [[1. 1.]
         [1. 1.]]
        >>> output = ops.full_ext((3, 3), 0)
        >>> print(output)
        [[0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]]
    """
    return fill_scalar_(size, fill_value, dtype)


def full_like(input, fill_value, *, dtype=None):
    """
    Return a tensor of the same shape as `input` and filled with a specified value.

    Note:
        `fill_value` 's data type not support complex numbers.

    Args:
        input (Tensor): The input tensor.
        fill_value (Number): The specified value.

    Keyword Args:
        dtype (mindspore.dtype, optional): The data type specified. Default: ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.ops.arange(4)
        >>> mindspore.ops.full_like(input, 1.2)
        Tensor(shape=[4], dtype=Int64, value= [1, 1, 1, 1])
        >>> mindspore.ops.full_like(input, 1.2, dtype=mindspore.float32)
        Tensor(shape=[4], dtype=Float32, value= [ 1.20000005e+00,  1.20000005e+00,  1.20000005e+00,  1.20000005e+00])
    """
    if not isinstance(input, Tensor):
        raise TypeError(
            f"For ops.full_like, the argument 'x' must be tensor, but got {type(input)}")
    if dtype is None:
        dtype = input.dtype
    return full(input.shape, fill_value, dtype=dtype)


def full_like_ext(input, fill_value, *, dtype=None):
    """
    Return a Tensor of the same shape as `input` and filled with `fill_value`.

    Args:
        input (Tensor): input Tensor and the output Tensor have the same shape as `input`.
        fill_value (Number): Value to fill the returned tensor. Complex numbers are not supported for now.

    Keyword Args:
        dtype (mindspore.dtype, optional): The specified type of output tensor. `bool` and `number` are supported,
            for details, please refer to :class:`mindspore.dtype` . Default: ``None`` .

    Returns:
        Tensor.

    Raises:
        TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> input = mindspore.Tensor([[0, 1], [2, 1]], dtype=mindspore.int32)
        >>> output = mindspore.ops.function.array_func.full_like_ext(input, 1)
        >>> print(output)
        [[1 1]
         [1 1]]
        >>> input = mindspore.Tensor([[0, 1, 1], [2, 1, 2], [1, 3, 4]], dtype=mindspore.int32)
        >>> output = mindspore.ops.function.array_func.full_like_ext(input, 0, dtype=mindspore.float32)
        >>> print(output)
        [[0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]]
    """
    if dtype is None:
        dtype = input.dtype
    return full_like_op(input, fill_value, dtype)


def chunk(input, chunks, axis=0):
    """
    Split the input tensor into multiple sub-tensors along the specified axis.

    Note:
        This function may return less than the specified number of chunks!

    Args:
        input (Tensor): A tensor to be split.
        chunks (int): The number of splits.
        axis (int, optional): The axis along which to split. Default ``0`` .

    Returns:
        Tuple of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.ops.arange(0, 9, dtype=mindspore.float32)
        >>> output = mindspore.ops.chunk(input, 3)
        >>> print(output)
        (Tensor(shape=[3], dtype=Float32, value= [ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value= [ 3.00000000e+00,  4.00000000e+00,  5.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value= [ 6.00000000e+00,  7.00000000e+00,  8.00000000e+00]))
    """
    if not isinstance(input, Tensor):
        raise TypeError(
            f'For ops.chunk parameter `input` must be Tensor, but got {type(input)}')
    _check_axis_type(axis, True, False, False, "ops.chunk")
    arr_axis = _canonicalize_axis(axis, input.ndim)

    if not isinstance(chunks, int):
        raise TypeError(
            f"For ops.chunk type of argument `chunks` should be integer, but got {type(chunks)}")
    if chunks <= 0:
        raise ValueError(
            f"For ops.chunk parameter 'chunks' must be greater than 0, but got {chunks}")

    arr_shape = input.shape
    length_along_dim = arr_shape[arr_axis]

    if length_along_dim == 0:
        res = _get_cache_prim(P.Split)(arr_axis)(input)
    elif chunks > length_along_dim:
        res = _get_cache_prim(P.Split)(arr_axis, length_along_dim)(input)
    elif length_along_dim % chunks == 0:
        res = _get_cache_prim(P.Split)(arr_axis, chunks)(input)
    else:
        block_size = int(np.ceil(length_along_dim / chunks))
        true_chunks = int(length_along_dim // block_size)
        length1 = true_chunks * block_size
        length2 = length_along_dim - length1
        start1 = _list_comprehensions(rank_(input), 0, True)
        size1 = _tuple_setitem(arr_shape, arr_axis, length1)
        start2 = _tuple_setitem(start1, arr_axis, length1)
        size2 = _tuple_setitem(arr_shape, arr_axis, length2)
        res = _get_cache_prim(P.Split)(arr_axis, true_chunks)(
            tensor_slice(input, start1, size1))
        if length2:
            res += _get_cache_prim(P.Split)(arr_axis,
                                            1)(tensor_slice(input, start2, size2))
    return res


def chunk_ext(input, chunks, dim=0):
    """
    Cut the input Tensor into `chunks` sub-tensors along the specified axis.

    Note:
        This function may return less than the specified number of chunks!

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): A Tensor to be cut.
        chunks (int): Number of sub-tensors to cut.
        dim (int, optional): Specify the dimensions that you want to split. Default: ``0`` .

    Returns:
        A tuple of sub-tensors.

    Raises:
        TypeError: If argument `input` is not Tensor.
        TypeError: If argument `chunks` is not int.
        TypeError: If argument `dim` is not int.
        ValueError: If argument `dim` is out of range of :math:`[-input.ndim, input.ndim)` .
        ValueError: If argument `chunks` is not positive number.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> input_x = np.arange(9).astype("float32")
        >>> output = mindspore.ops.function.array_func.chunk_ext(mindspore.Tensor(input_x), 3)
        >>> print(output)
        (Tensor(shape=[3], dtype=Float32, value= [ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value= [ 3.00000000e+00,  4.00000000e+00,  5.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value= [ 6.00000000e+00,  7.00000000e+00,  8.00000000e+00]))
    """
    return chunk_op(input, chunks, dim)


def chunk_view(input, chunks, dim=0): # pylint: disable=redefined-builtin
    """
    Cut the input Tensor into `chunks` sub-tensors along the specified axis.

    Note:
        This function may return less than the specified number of chunks!

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): A Tensor to be cut.
        chunks (int): Number of sub-tensors to cut.
        dim (int, optional): Specify the dimensions that you want to split. Default: ``0`` .

    Returns:
        A tuple of sub-tensors.

    Raises:
        TypeError: If argument `input` is not Tensor.
        TypeError: If argument `chunks` is not int.
        TypeError: If argument `dim` is not int.
        ValueError: If argument `dim` is out of range of :math:`[-input.ndim, input.ndim)` .
        ValueError: If argument `chunks` is not positive number.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> input_x = np.arange(9).astype("float32")
        >>> output = mindspore.ops.function.array_func.chunk_view(mindspore.Tensor(input_x), 3)
        >>> print(output)
        (Tensor(shape=[3], dtype=Float32, value= [ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value= [ 3.00000000e+00,  4.00000000e+00,  5.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value= [ 6.00000000e+00,  7.00000000e+00,  8.00000000e+00]))
    """
    return ops.auto_generate.chunk_view_op(input, chunks, dim)


def fills(x, value):
    """
    `fills` is deprecated, please use `ops.fill` instead.
    """
    if isinstance(value, float):
        value_ = value
    elif isinstance(value, int):
        value_ = float(value)
    elif isinstance(value, Tensor):
        if value.ndim != 0:
            raise ValueError(f"For 'ops.fills', if the argument 'value' is a tensor, the number of its dimension"
                             f" should be 0, but got {value.ndim}")
        value_ = value.astype(mstype.float32)
    else:
        raise TypeError(f"For 'ops.fills', the type of argument 'value' should be int, float or Tensor,"
                        f" but got {type(value)}")
    return fills_(x, value_)


def ones_like(input, *, dtype=None):
    """
    Return a tensor filled with 1, with the same size as `input`.

    Args:
        input (Tensor): The input tensor.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The data type specified. Default ``None`` represents the same data
            type as the input.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.ops.arange(4)
        >>> mindspore.ops.ones_like(input)
        Tensor(shape=[4], dtype=Int64, value= [1, 1, 1, 1])
    """
    output = ones_like_(input)
    _dtype = input.dtype if dtype is None else dtype
    output = cast_(output, _dtype)
    return output


def zeros_like(input, *, dtype=None):
    r"""
    Return a tensor filled with 0, with the same size as `input` .

    Args:
        input (Tensor): The input tensor.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The data type specified. Default ``None`` represents the same data
            type as the input.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.ops.arange(4)
        >>> mindspore.ops.zeros_like(input)
        Tensor(shape=[4], dtype=Int64, value= [0, 0, 0, 0])
    """
    _dtype = input.dtype if dtype is None else dtype
    output = zeros_like_(input)
    output = cast_(output, _dtype)
    return output


def ones_like_ext(input, *, dtype=None):
    """
    Creates a tensor filled with 1, with the same shape as input, and its data type is determined by the given dtype.

    If `dtype = None`, the tensor will have the same dtype as input `input`.

    Args:
        input (Tensor): Tensor of any dimension.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The specified dtype of the output tensor. If `dtype` is ``None`` ,
            the dtype of the input tensor will be used. Default: ``None`` .

    Returns:
        Tensor, has the same shape as `input` but filled with ones.

    Raises:
        TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))
        >>> output = ops.function.array_func.ones_like_ext(x)
        >>> print(output)
        [[1 1]
         [1 1]]
    """
    return ones_like_ext_(input, dtype)


def zeros_like_ext(input, *, dtype=None):
    r"""
    Creates a tensor filled with 0, with the same size as input. Its data type is determined by the given dtype.

    If `dtype = None`, the tensor will have the same dtype as input `input`.

    Args:
        input (Tensor): Tensor of any dimension.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The specified dtype of the output tensor. If `dtype` is ``None`` ,
            the dtype of the input tensor will be used. Default: ``None`` .

    Returns:
        Tensor, filled with 0.

    Raises:
        TypeError: If dtype is not a MindSpore dtype.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.arange(4).reshape(2, 2))
        >>> output = ops.function.array_func.zeros_like_ext(x, dtype=mindspore.float32)
        >>> print(output)
        [[0. 0.]
         [0. 0.]]
    """
    return zeros_like_ext_(input, dtype)


def new_ones(input, size, *, dtype=None):
    """
    Return a tensor of `size` filled with ones.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): Tensor of any dimension.
        size (Union[int, tuple(int), list(int)]): An int, list or tuple of integers defining the output shape.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The desired dtype of the output tensor. If None, the returned
            tensor has the same dtype as `self`. Default: ``None``.

    Returns:
        Tensor, the shape and dtype is defined above and filled with ones.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `size` is neither an int nor a tuple/list of int.
        TypeError: If `dtype` is not a MindSpore dtype.
        ValueError: If `size` contains negative values.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> input = Tensor([1, 2, 3, 4], mindspore.int32)
        >>> output = ops.function.array_func.new_ones(input, (2, 3))
        >>> print(output)
        [[1 1 1]
         [1 1 1]]
    """
    return new_ones_(input, size, dtype)


def new_zeros(input, size, *, dtype=None):
    """
    Return a tensor of `size` filled with zeros.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): Tensor of any dimension.
        size (Union[int, tuple(int), list(int)]): An int, list or tuple of integers defining the output shape.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The desired dtype of the output tensor. If None, the returned
            tensor has the same dtype as `self`. Default: ``None``.

    Returns:
        Tensor, the shape and dtype is defined above and filled with zeros.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `size` is neither an int nor a tuple/list of int.
        TypeError: If `dtype` is not a MindSpore dtype.
        ValueError: If `size` contains negative values.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> input = Tensor([1, 2, 3, 4], mindspore.int32)
        >>> output = ops.function.array_func.new_zeros(input, (2, 3))
        >>> print(output)
        [[0 0 0]
         [0 0 0]]
    """
    return new_zeros_(input, size, dtype)


##############################
# Tensor Operation Functions.
##############################


def unique(input):
    """
    Remove duplicate elements from the input tensor.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tuple(output, indices) of 2 tensors.

        - **output** (Tensor) - The deduplicated output tensor.
        - **indices** (Tensor) - The indices of the elements of the input tensor in the `output` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([1, 2, 5, 2], mindspore.int32)
        >>> output, indices = mindspore.ops.unique(x)
        >>> print(output)
        [1 2 5]
        >>> print(indices)
        [0 1 2 1]
    """
    shape_x = input.shape
    length_x = get_x_shape(shape_x)
    input = reshape_(input, length_x)
    y, idx = unique_(input)
    idx = reshape_(idx, shape_x)
    return y, idx


def unique_ext(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    """
    Return the unique elements of input tensor.

    Args:
        input (Tensor): The input tensor.
        sorted (bool, optional): Whether to sort the unique elements in ascending order before returning as output.
            Default ``True`` .
        return_inverse (bool, optional): Whether to additionally return a tensor
            indicating the indices of `input` corresponding to `output`.
            Default ``False`` .
        return_counts (bool, optional): Whether to additionally return a tensor
            indicating the count of each element in `output` within the `input`. Default ``False`` .
        dim (int, optional): Specify the dimension for computation. 


    Returns:
        A tensor or a tuple of tensors.

        - output (Tensor) - The unique elements of input tensor.
        - inverse_indices (Tensor) - Return when ``return_inverse`` is True. It represents the indices for where
          elements in `input` map to in `output`. When ``dim`` is ``None``, it has same shape as `input`,
          otherwise, the shape is input.shape[dim].
        - counts (Tensor) - Return when ``return_counts`` is True. It represents the number of occurrences for each
          unique element from `input` within `output`.  When ``dim`` is ``None``,
          it has same shape as `output`, otherwise, the shape is output.shape[dim].


    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> x = mindspore.tensor(np.array([1, 2, 5, 2]), mindspore.int32)
        >>> output = mindspore.ops.unique_ext(x, return_inverse=True, return_counts=True)
        >>> print(output)
        (Tensor(shape=[3], dtype=Int32, value= [1, 2, 5]), Tensor(shape=[4], dtype=Int64,
        value= [0, 1, 2, 1]), Tensor(shape=[3], dtype=Int64, value= [1, 2, 1]))
        >>> y = output[0]
        >>> print(y)
        [1 2 5]
        >>> inverse_indices = output[1]
        >>> print(inverse_indices)
        [0 1 2 1]
        >>> counts = output[2]
        >>> print(counts)
        [1 2 1]
    """
    if not F.isconstant(return_inverse) or not F.isconstant(return_counts):
        raise ValueError("For 'unique_ext', 'return_inverse' and 'return_counts' cannot be mutable")
    if dim is None:
        y, inverse, counts = unique2_(
            input, sorted, return_inverse, return_counts)
    else:
        validator.check_value_type(
            "return_counts", return_counts, [bool], "unique_ext")
        y, inverse, counts = unique_dim_(input, sorted, return_inverse, dim)
    if return_inverse and return_counts:
        return y, inverse, counts
    if return_inverse:
        return y, inverse
    if return_counts:
        return y, counts
    return y


@deprecated("2.4.0", "ops.unique combined with ops.pad", False, "ops.")
def unique_with_pad(x, pad_num):
    """
    `ops.unique_with_pad` is deprecated from version 2.4.0 and will be removed
    in a future version, please use :func:`mindspore.ops.unique` combined with :func:`mindspore.ops.pad` to realize
    the same function.

    Returns unique elements and relative indexes in 1-D tensor, filled with padding num.

    The basic function is the same as the Unique operator, but the UniqueWithPad operator adds a Pad function.
    The returned tuple(`y`, `idx`) after the input Tensor `x` is processed by the unique operator,
    in which the shapes of `y` and `idx` are mostly not equal. Therefore, in order to solve the above situation,
    the UniqueWithPad operator will fill the `y` Tensor with the `pad_num` specified by the user
    to make it have the same shape as the Tensor `idx`.

    Args:
        x (Tensor): The tensor need to be unique. Must be 1-D vector with types: int32, int64.
        pad_num (int): Pad num. The data type is an int.

    Returns:
        tuple(Tensor), tuple of 2 tensors, `y` and `idx`.

        - y (Tensor) - The unique elements filled with pad_num, the shape and data type same as `x`.
        - idx (Tensor) - The index of each value of `x` in the unique output `y`, the shape and data type same as `x`.

    Raises:
        TypeError: If dtype of `x` is neither int32 nor int64.
        ValueError: If `x` is not a 1-D Tensor.

    Supported Platforms:
        Deprecated

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> from mindspore import ops
        >>> x = Tensor(np.array([1, 2, 2, 3, 5, 5]), mindspore.int32)
        >>> output = ops.unique_with_pad(x, 0)
        >>> print(output)
        (Tensor(shape=[6], dtype=Int32, value= [1, 2, 3, 5, 0, 0]),
         Tensor(shape=[6], dtype=Int32, value= [0, 1, 1, 2, 3, 3]))
        >>> y = output[0]
        >>> print(y)
        [1 2 3 5 0 0]
        >>> idx = output[1]
        >>> print(idx)
        [0 1 1 2 3 3]
    """
    return _get_cache_prim(P.UniqueWithPad)()(x, pad_num)


def unique_consecutive(input, return_inverse=False, return_counts=False, dim=None):
    """
    Remove consecutive duplicate elements in the input tensor,
    retaining only the first occurrence from each repeated group.

    Args:
        input (Tensor): The input tensor.
        return_inverse (bool, optional): Whether to also return the indices for where elements
            in the original input ended up in the returned unique list. Default ``False`` .
        return_counts (bool, optional): Whether to also return the counts for each unique element. Default ``False`` .
        dim (int, optional): Specify the dimension for unique. Default ``None`` , the input tensor will be flattened.

    Returns:
        Tensor or tuple(output, inverse_indices, counts) of tensors.

        - **output** (Tensor) - The deduplicated output tensor.
        - **inverse_indices** (Tensor, optional) - The indices of the elements of the input tensor in the `output` .
        - **counts** (Tensor, optional) - The counts for each unique element.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([1, 1, 2, 2, 3, 1, 1, 2], mindspore.int32)
        >>> output, inverse_indices, counts = mindspore.ops.unique_consecutive(x, True, True, None)
        >>> print(output)
        [1 2 3 1 2]
        >>> print(inverse_indices)
        [0 0 1 1 2 3 3 4]
        >>> print(counts)
        [2 2 1 2 1]
    """

    if not F.isconstant(return_inverse) or not F.isconstant(return_counts):
        raise ValueError("For 'unique_consecutive', 'return_inverse' and 'return_counts' cannot be mutable")
    output, idx, counts = unique_consecutive_impl(input, return_inverse, return_counts, dim)
    if return_inverse and return_counts:
        return output, idx, counts
    if return_inverse:
        return output, idx
    if return_counts:
        return output, counts
    return output


def searchsorted(sorted_sequence, values, *, out_int32=False, right=False, side=None, sorter=None):
    """
    Return the position indices where the elements can be inserted into the input tensor to maintain the increasing
    order of the input tensor.

    Args:
        sorted_sequence (Tensor): The input tensor. If `sorter` not provided, it must contain a increasing sequence
            on the innermost dimension.
        values (Tensor): The value that need to be inserted.

    Keyword Args:
        out_int32 (bool, optional): Whether the output datatype will be mindspore.int32.
            if ``False`` , the output datatype will be mindspore.int64. Default ``False`` .
        right (bool, optional): Search Strategy. If ``True`` , return the last suitable index found;
            if ``False`` , return the first such index. Default ``False`` .
        side (str, optional): the same as right but preferred. ``left`` corresponds to ``False`` for `right`
            and ``right`` corresponds to ``True`` for `right`. An error will be reported if this parameter is
            set to ``left`` while `right` is ``True``. Default ``None`` .
        sorter(Tensor, optional): An index sequence sorted in ascending order along the innermost dimension of
            `sorted_sequence` , which is used together with the unsorted `sorted_sequence` . CPU and GPU only support
            ``None``. Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> sorted_sequence = mindspore.tensor([1, 2, 2, 3, 4, 5, 5])
        >>> values = mindspore.tensor([2])
        >>> mindspore.ops.searchsorted(sorted_sequence, values)
        Tensor(shape=[1], dtype=Int64, value= [1])
    """

    validator.check_value_type("out_int32", out_int32, [bool], "search_sorted")
    validator.check_value_type("right", right, [bool], "search_sorted")
    dtype = mstype.int32 if bool(out_int32) else mstype.int64
    if (side == "left" and right is True):
        raise ValueError("For 'searchsorted', side and right can't be set to opposites, "
                         "got side of left while right was True.")
    if side == "right":
        right = True
    return search_sorted_(sorted_sequence, values, sorter, dtype, right)


def ger(input, vec2):
    r"""
    Calculate the outer product of two arrays `input` and `vec2`.

    Note:
        Currently Ascend does not support float64 data input.

    Args:
        input (Tensor): The 1-D input tensor.
        vec2 (Tensor): The 1-D input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([1., 2., 3., 4.])
        >>> vec2 = mindspore.tensor([1., 2., 3.])
        >>> output = mindspore.ops.ger(input, vec2)
        >>> print(output)
        [[ 1.  2.  3.]
         [ 2.  4.  6.]
         [ 3.  6.  9.]
         [ 4.  8. 12.]]
    """
    return ger_(input, vec2)


def size(input_x):
    r"""
    Count the total number of elements in `input_x` .

    Args:
        input_x (Tensor): The input tensor.

    Returns:
        int

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.tensor([[2, 2], [2, 2]], mindspore.float32)
        >>> output = mindspore.ops.size(input_x)
        >>> print(output)
        4
    """
    return size_(input_x)


def shape(input_x):
    """
    Return the shape of the input tensor.

    Args:
        input_x (Tensor): The input tensor.

    Returns:
        tuple[int]

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.ops.ones(shape=[3, 2, 1])
        >>> output = mindspore.ops.shape(input_x)
        >>> print(output)
        (3, 2, 1)
    """
    return shape_(input_x)


def dyn_shape(input_x):
    """
    Returns the shape of the input tensor.

    Args:
        input_x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.tensor(mindspore.ops.ones(shape=[3, 2, 1]), mindspore.float32)
        >>> output = mindspore.ops.dyn_shape(input_x)
        >>> print(output)
        [3 2 1]
    """
    return tensor_shape_(input_x)


def reverse_sequence(x, seq_lengths, seq_dim, batch_dim=0):
    r"""
    Partially reverse the input sequence.

    Args:
        x (Tensor): The input tensor.
        seq_lengths (Tensor): The specified reversing length.
        seq_dim (int): The specified dimension for reversal.
        batch_dim (int): The specified slice dimension. Default ``0`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mindspore.float32)
        >>> seq_lengths = mindspore.tensor([1, 2, 3])
        >>> output = mindspore.ops.reverse_sequence(x, seq_lengths, seq_dim=1)
        >>> print(output)
        [[1. 2. 3.]
         [5. 4. 6.]
         [9. 8. 7.]]
        >>> x = mindspore.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mindspore.float32)
        >>> seq_lengths = mindspore.tensor([1, 2, 3])
        >>> output = mindspore.ops.reverse_sequence(x, seq_lengths, seq_dim=0, batch_dim=1)
        >>> print(output)
        [[1. 5. 9.]
         [4. 2. 6.]
         [7. 8. 3.]]
        >>> x = mindspore.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mindspore.float32)
        >>> seq_lengths = mindspore.tensor([2, 2, 3])
        >>> output = mindspore.ops.reverse_sequence(x, seq_lengths, seq_dim=1)
        >>> print(output)
        [[2. 1. 3.]
         [5. 4. 6.]
         [9. 8. 7.]]
        >>> x = mindspore.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mindspore.float32)
        >>> seq_lengths = mindspore.tensor([3, 2, 3])
        >>> output = mindspore.ops.reverse_sequence(x, seq_lengths, seq_dim=1)
        >>> print(output)
        [[3. 2. 1.]
         [5. 4. 6.]
         [9. 8. 7.]]
        >>> x = mindspore.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], mindspore.float32)
        >>> seq_lengths = mindspore.tensor([4, 4])
        >>> output = mindspore.ops.reverse_sequence(x, seq_lengths, seq_dim=1)
        >>> print(output)
        [[4. 3. 2. 1.]
         [8. 7. 6. 5.]]
    """
    return _get_cache_prim(P.ReverseSequence)(seq_dim=seq_dim, batch_dim=batch_dim)(x, seq_lengths)


def flatten(input, order='C', *, start_dim=1, end_dim=-1):
    r"""
    Flatten a tensor along dimensions from `start_dim` to `start_dim`.

    Args:
        input (Tensor): The input Tensor.
        order (str, optional): Only ``'C'`` and ``'F'`` are supported.
            ``'C'`` means to flatten in row-major (C-style) order.
            ``'F'`` means to flatten in column-major (Fortran-style) order. Default: ``'C'`` .

    Keyword Args:
        start_dim (int, optional): The first dimension to flatten. Default: ``1`` .
        end_dim (int, optional): The last dimension to flatten. Default: ``-1`` .

    Returns:
        Tensor. If no dimensions are flattened, returns the original `input`, otherwise return the flattened Tensor.
        If `input` is a 0-dimensional Tensor, a 1-dimensional Tensor will be returned.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `order` is not string type.
        ValueError: If `order` is string type, but not ``'C'`` or ``'F'``.
        TypeError: If `start_dim` or `end_dim` is not int.
        ValueError: If `start_dim` is greater than `end_dim` after canonicalized.
        ValueError: If `start_dim` or `end_dim` is not in range of [-input.dim, input.dim-1].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.ones(shape=[1, 2, 3, 4]), mindspore.float32)
        >>> output = ops.flatten(input_x)
        >>> print(output.shape)
        (1, 24)
    """

    def check_axis_valid(axis, ndim):
        if axis < -ndim or axis >= ndim:
            raise ValueError("'start_dim' or 'end_dim' out of range.")

    def check_dim_valid(start_dim, end_dim):
        if start_dim > end_dim:
            raise ValueError(
                "For 'flatten', 'start_dim' cannot come after 'end_dim'.")

    def canonicalize_axis(axis, x_rank):
        ndim = x_rank if x_rank != 0 else 1
        check_axis_valid(axis, ndim)
        return axis if axis >= 0 else axis + ndim

    # Check the types of arguments.
    if not isinstance(input, Tensor):
        raise TypeError("For 'flatten', argument 'input' must be Tensor.")
    if not isinstance(start_dim, int) or not isinstance(end_dim, int) or \
            isinstance(start_dim, bool) or isinstance(end_dim, bool):
        raise TypeError(
            "For 'flatten', both 'start_dim' and 'end_dim' must be int.")
    check_flatten_order_const(order)
    if order == 'F':
        x_rank = rank_(input)
        # If input is a 0-dimensional Tensor, a 1-dimensional Tensor will be returned.
        if x_rank in (0, 1):
            return reshape_(input, (-1,))
        perm = ops.make_range(0, x_rank)
        new_order = ops.tuple_reversed(perm)
        input = transpose_(input, new_order)

    # Handle the default case.
    x_shape = shape_(input)
    x_rank = rank_(input)
    if start_dim == 1 and end_dim == -1:
        if x_rank in (0, 1):
            return reshape_(input, (-1,))
        return flatten_(input)

    # Check axis.
    start_dim = canonicalize_axis(start_dim, x_rank)
    end_dim = canonicalize_axis(end_dim, x_rank)
    check_dim_valid(start_dim, end_dim)
    # If input is a 0-dimensional Tensor, a 1-dimensional Tensor will be returned.
    if x_rank in (0, 1):
        return reshape_(input, (-1,))
    # If no dimensions to flatten, return the original object.
    if start_dim == end_dim:
        return input
    # Flatten elements along specified dimensions.
    dim_length = 1
    idx = start_dim
    while idx <= end_dim:
        dim_length *= x_shape[idx]
        idx += 1
    new_shape = x_shape[:start_dim] + (dim_length,) + x_shape[end_dim + 1:]
    return reshape_(input, new_shape)


def slice(input_x, begin, size):
    r"""
    Slice a tensor in the specified shape.

    Note:
        `begin` is zero-based and `size` is one-based.

    If `size[i]` is -1, all remaining elements in dimension i are included in the slice.
    This is equivalent to setting :math:`size[i] = input\_x.shape(i) - begin[i]`.

    Args:
        input_x (Tensor): The input tensor.
        begin (Union[tuple, list]): The beginning of the slice which represents the offset in each dimension.
        size (Union[tuple, list]): The size of the slice.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> data = mindspore.tensor([[[1, 1, 1], [2, 2, 2]],
        ...                         [[3, 3, 3], [4, 4, 4]],
        ...                         [[5, 5, 5], [6, 6, 6]]], mindspore.int32)
        >>> output = mindspore.ops.slice(data, (1, 0, 0), (1, 1, 3))
        >>> print(output)
        [[[3 3 3]]]
        >>> output = mindspore.ops.slice(data, (1, 0, 0), (1, 1, 2))
        >>> print(output)
        [[[3 3]]]
        >>> output = mindspore.ops.slice(data, (1, 0, 0), (1, 1, 1))
        >>> print(output)
        [[[3]]]
        >>> output = mindspore.ops.slice(data, (1, 1, 0), (1, 1, 3))
        >>> print(output)
        [[[4 4 4]]]
        >>> output = mindspore.ops.slice(data, (1, 0, 1), (1, 1, 2))
        >>> print(output)
        [[[3 3]]]
    """
    return tensor_slice(input_x, begin, size)


def stack(tensors, axis=0):
    r"""
    Stack input tensors in specified axis.

    Args:
        tensors (Union[tuple, list]): The input tensors.
        axis (int): Axis to stack. Default ``0`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x1 = mindspore.tensor([0, 1], mindspore.float32)
        >>> input_x2 = mindspore.tensor([2, 3], mindspore.float32)
        >>> output = mindspore.ops.stack((input_x1, input_x2), 0)
        >>> print(output)
        [[0. 1.]
         [2. 3.]]
    """
    _stack = _get_cache_prim(P.Stack)(axis)
    return _stack(tensors)


def unstack(input_x, axis=0):
    r"""
    Unstack the input tensor along the specified axis.

    Args:
        input_x (Tensor): The input tensor.
        axis (int): The specified axis. Default ``0`` .

    Returns:
        Tuple of tensors

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[1, 1, 1, 1], [2, 2, 2, 2]])
        >>> mindspore.ops.unstack(input, 0)
        (Tensor(shape=[4], dtype=Int64, value= [1, 1, 1, 1]),
         Tensor(shape=[4], dtype=Int64, value= [2, 2, 2, 2]))
    """
    _unstack = _get_cache_prim(P.Unstack)(axis)
    return _unstack(input_x)


def unbind(input, dim=0):
    r"""
    Remove a tensor dimension, return a tuple of all slices along a given dimension.

    Args:
        input (Tensor): The input tensor.
        dim (int): Specipy the dimension to remove. Default ``0`` .

    Returns:
        tuple of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> output = mindspore.ops.unbind(x, dim=0)
        >>> print(output)
        (Tensor(shape=[3], dtype=Int64, value=[1, 2, 3]), Tensor(shape=[3], dtype=Int64, value=[4, 5, 6]),
        Tensor(shape=[3], dtype=Int64, value=[7, 8, 9]))
    """
    _unstack = _get_cache_prim(P.Unstack)(dim)
    return _unstack(input)


def unbind_ext(input, dim=0):
    r"""
    Unbind a tensor dimension in specified axis.

    Given a tensor of shape :math:`(n_1, n_2, ..., n_R)` and unbinding it in the specified `dim`,
    multiple tensors with shape :math:`(n_1, n_2, ..., n_{dim}, n_{dim+2}, ..., n_R)` are returned.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The input tensor to unbind, with a shape of :math:`(n_1, n_2, ..., n_R)`.
            The rank of the tensor must be greater than 0.
        dim (int, optional): Dimension along which to unbind. The range is [-R, R). Default: ``0`` .

    Returns:
        A tuple of tensors, the shape of each objects is the same.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `dim` is not an int.
        ValueError: If `dim` is out of the range [-R, R).

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        >>> output = ops.unbind_ext(input, dim=0)
        >>> print(output)
        (Tensor(shape=[3], dtype=Int64, value=[1, 2, 3]), Tensor(shape=[3], dtype=Int64, value=[4, 5, 6]),
        Tensor(shape=[3], dtype=Int64, value=[7, 8, 9]))
    """
    return unstack_ext_view_op(input, dim)


def unsqueeze(input, dim):
    """
    Adds an additional dimension to the input tensor at the given dimension.

    Args:
        input (Tensor): The input tensor.
        dim (int): The dimension specified.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([1, 2, 3, 4])
        >>> mindspore.ops.unsqueeze(input, 0)
        Tensor(shape=[1, 4], dtype=Int64, value=
        [[1, 2, 3, 4]])
        >>> mindspore.ops.unsqueeze(input, 1)
        Tensor(shape=[4, 1], dtype=Int64, value=
        [[1],
         [2],
         [3],
         [4]])
    """
    return expand_dims(input, dim)


def unsqueeze_view(input, dim): # pylint: disable=redefined-builtin
    """
    Adds an additional dimension to the input tensor at the given dimension.

    Args:
        input (Tensor): The input tensor.
        dim (int): The dimension specified.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([1, 2, 3, 4])
        >>> mindspore.ops.function.array_func.unsqueeze_view(input, 0)
        Tensor(shape=[1, 4], dtype=Int64, value=
        [[1, 2, 3, 4]])
        >>> mindspore.ops.function.array_func.unsqueeze_view(input, 1)
        Tensor(shape=[4, 1], dtype=Int64, value=
        [[1],
         [2],
         [3],
         [4]])
    """
    return ops.auto_generate.expand_dims_view(input, dim)


def squeeze(input, axis=None):
    """
    Remove length one axes from input tensor.

    Note:
        - Please note that in dynamic graph mode, the output tensor will share data with the input tensor,
          and there is no Tensor data copy process.
        - The dimension index starts at 0 and must be in the range `[-input.ndim, input.ndim]`.
        - In GE mode, only support remove dimensions of size 1 from the shape of input tensor.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The input tensor.
        axis (Union[int, tuple(int), list(int)]): The axis to be removed. Default: ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.ops.ones(shape=[3, 2, 1])
        >>> output = mindspore.ops.squeeze(input)
        >>> print(output)
        [[1. 1.]
         [1. 1.]
         [1. 1.]]
    """
    if axis is None:
        axis = ()
    return squeeze_impl(input, axis)


def scatter_mul(input_x, indices, updates):
    r"""
    Perform a multiplication update on `input_x` based on the specified indices and update values.

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :] \mathrel{*}= \text{updates}[i, ..., j, :]

    .. note::
        - Support implicit type conversion and type promotion.
        - Since Parameter objects do not support type conversion, an exception will be thrown when `input_x` is
          of a low-precision data type.
        - The shape of `updates` is `indices.shape + input_x.shape[1:]`.

    Args:
        input_x (Union[Parameter, Tensor]): The input parameter or tensor.
        indices (Tensor): The specified indices.
        updates (Tensor): The update values.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.Parameter(mindspore.tensor([[1.0, 1.0, 1.0],
        ...                               [2.0, 2.0, 2.0]], mindspore.float32), name="x")
        >>> indices = mindspore.tensor([0, 1], mindspore.int32)
        >>> updates = mindspore.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], mindspore.float32)
        >>> output = mindspore.ops.scatter_mul(input_x, indices, updates)
        >>> print(output)
        [[2. 2. 2.]
         [4. 4. 4.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = mindspore.Parameter(mindspore.tensor([[1.0, 1.0, 1.0],
        ...                                                [2.0, 2.0, 2.0]], mindspore.float32), name="x")
        >>> # for indices = [[0, 1], [1, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [1.0, 1.0, 1.0] * [1.0, 1.0, 1.0] = [1.0, 1.0, 1.0]
        >>> # input_x[1] = [2.0, 2.0, 2.0] * [3.0, 3.0, 3.0] = [6.0, 6.0, 6.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [6.0, 6.0, 6.0] * [7.0, 7.0, 7.0] = [42.0, 42.0, 42.0]
        >>> # input_x[1] = [42.0, 42.0, 42.0] * [9.0, 9.0, 9.0] = [378.0, 378.0, 378.0]
        >>> indices = mindspore.tensor([[0, 1], [1, 1]], mindspore.int32)
        >>> updates = mindspore.tensor([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]], mindspore.float32)
        >>> output = mindspore.ops.scatter_mul(input_x, indices, updates)
        >>> print(output)
        [[  1.   1.   1.]
         [378. 378. 378.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = mindspore.Parameter(mindspore.tensor([[1.0, 1.0, 1.0],
        ...                                                [2.0, 2.0, 2.0]], mindspore.float32), name="x")
        >>> # for indices = [[1, 0], [1, 1]]
        >>> # step 1: [1, 0]
        >>> # input_x[0] = [1.0, 1.0, 1.0] * [3.0, 3.0, 3.0] = [3.0, 3.0, 3.0]
        >>> # input_x[1] = [2.0, 2.0, 2.0] * [1.0, 1.0, 1.0] = [2.0, 2.0, 2.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [2.0, 2.0, 2.0] * [7.0, 7.0, 7.0] = [14.0, 14.0, 14.0]
        >>> # input_x[1] = [14.0, 14.0, 14.0] * [9.0, 9.0, 9.0] = [126.0, 126.0, 126.0]
        >>> indices = mindspore.tensor([[1, 0], [1, 1]], mindspore.int32)
        >>> updates = mindspore.tensor([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]], mindspore.float32)
        >>> output = mindspore.ops.scatter_mul(input_x, indices, updates)
        >>> print(output)
        [[  3.   3.   3.]
         [126. 126. 126.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = mindspore.Parameter(mindspore.tensor([[1.0, 1.0, 1.0],
        ...                                                [2.0, 2.0, 2.0]], mindspore.float32), name="x")
        >>> # for indices = [[0, 1], [0, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [1.0, 1.0, 1.0] * [1.0, 1.0, 1.0] = [1.0, 1.0, 1.0]
        >>> # input_x[1] = [2.0, 2.0, 2.0] * [3.0, 3.0, 3.0] = [6.0, 6.0, 6.0]
        >>> # step 2: [0, 1]
        >>> # input_x[0] = [1.0, 1.0, 1.0] * [7.0, 7.0, 7.0] = [7.0, 7.0, 7.0]
        >>> # input_x[1] = [6.0, 6.0, 6.0] * [9.0, 9.0, 9.0] = [54.0, 54.0, 54.0]
        >>> indices = mindspore.tensor([[0, 1], [0, 1]], mindspore.int32)
        >>> updates = mindspore.tensor([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]], mindspore.float32)
        >>> output = mindspore.ops.scatter_mul(input_x, indices, updates)
        >>> print(output)
        [[ 7.  7.  7.]
         [54. 54. 54.]]
    """
    return scatter_mul_(input_x, indices, updates)


def scatter_max(input_x, indices, updates):
    r"""
    Perform a maximum update on `input_x` based on the specified indices and update values.

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :]
        = \max(\text{input_x}[\text{indices}[i, ..., j], :], \text{updates}[i, ..., j, :])

    .. note::
        - Support implicit type conversion and type promotion.
        - Since Parameter objects do not support type conversion, an exception will be thrown when `input_x` is
          of a low-precision data type.
        - The shape of `updates` is `indices.shape + input_x.shape[1:]`.

    Args:
        input_x (Union[Parameter, Tensor]): The input parameter or tensor.
        indices (Tensor): The specified indices.
        updates (Tensor): The update values.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.Parameter(mindspore.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ...                               mindspore.float32), name="input_x")
        >>> indices = mindspore.tensor([[0, 0], [1, 1]], mindspore.int32)
        >>> updates = mindspore.tensor(mindspore.ops.ones([2, 2, 3]) * 88, mindspore.float32)
        >>> output = mindspore.ops.scatter_max(input_x, indices, updates)
        >>> print(output)
        [[88. 88. 88.]
         [88. 88. 88.]]
    """
    return scatter_max_(input_x, indices, updates)


def scatter_add(input_x, indices, updates):
    r"""
    Perform an addition update on `input_x` based on the specified indices and update values.

    .. math::
        \text{input_x}[\text{indices}[i, ..., j], :] \mathrel{+}= \text{updates}[i, ..., j, :]

    .. note::
        - Support implicit type conversion and type promotion.
        - Since Parameter objects do not support type conversion, an exception will be thrown when `input_x` is
          of a low-precision data type.
        - The shape of `updates` is `indices.shape + input_x.shape[1:]`.

    Args:
        input_x (Union[Parameter, Tensor]): The input parameter or tensor.
        indices (Tensor): The specified indices.
        updates (Tensor): The update values.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.Parameter(mindspore.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ...                               mindspore.float32), name="x")
        >>> indices = mindspore.tensor([[0, 1], [1, 1]], mindspore.int32)
        >>> updates = mindspore.tensor([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]], mindspore.float32)
        >>> output = mindspore.ops.scatter_add(input_x, indices, updates)
        >>> print(output)
        [[ 1.  1.  1.]
         [19. 19. 19.]]
    """
    return scatter_add_(input_x, indices, updates)


def scatter_min(input_x, indices, updates):
    r"""
    Perform a minimum update on `input_x` based on the specified indices and update values.

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :]
        = \min(\text{input_x}[\text{indices}[i, ..., j], :], \text{updates}[i, ..., j, :])

    .. note::
        - Support implicit type conversion and type promotion.
        - Since Parameter objects do not support type conversion, an exception will be thrown when `input_x` is
          of a low-precision data type.
        - The shape of `updates` is `indices.shape + input_x.shape[1:]`.

    Args:
        input_x (Union[Parameter, Tensor]): The input parameter or tensor.
        indices (Tensor): The specified indices.
        updates (Tensor): The update values.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.Parameter(mindspore.tensor(mindspore.ops.zeros((2, 3)),
        ...                               mindspore.float32), name="input_x")
        >>> indices = mindspore.tensor([1, 0], mindspore.int32)
        >>> update = mindspore.tensor(mindspore.ops.arange(0, 6).reshape((2, 3)), mindspore.float32)
        >>> output = mindspore.ops.scatter_min(input_x, indices, update)
        >>> print(output)
        [[0. 0. 0.]
         [0. 0. 0.]]
    """
    return scatter_min_(input_x, indices, updates)


def scatter_div(input_x, indices, updates):
    r"""
    Perform a division update on `input_x` based on the specified indices and update values.

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :] \mathrel{/}= \text{updates}[i, ..., j, :]

    .. note::
        - Support implicit type conversion and type promotion.
        - Since Parameter objects do not support type conversion, an exception will be thrown when `input_x` is
          of a low-precision data type.
        - The shape of `updates` is `indices.shape + input_x.shape[1:]`.

    Args:
        input_x (Union[Parameter, Tensor]): The input parameter or tensor.
        indices (Tensor): The specified indices.
        updates (Tensor): The update values.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.Parameter(mindspore.tensor([[6.0, 6.0, 6.0], [2.0, 2.0, 2.0]],
        ...                               mindspore.float32), name="x")
        >>> indices = mindspore.tensor([0, 1], mindspore.int32)
        >>> updates = mindspore.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], mindspore.float32)
        >>> output = mindspore.ops.scatter_div(input_x, indices, updates)
        >>> print(output)
        [[3. 3. 3.]
         [1. 1. 1.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = mindspore.Parameter(mindspore.tensor([[105.0, 105.0, 105.0],
        ...                                                [315.0, 315.0, 315.0]], mindspore.float32), name="x")
        >>> # for indices = [[0, 1], [1, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [105.0, 105.0, 105.0] / [1.0, 1.0, 1.0] = [105.0, 105.0, 105.0]
        >>> # input_x[1] = [315.0, 315.0, 315.0] / [3.0, 3.0, 3.0] = [105.0, 105.0, 105.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [105.0, 105.0, 105.0] / [5.0, 5.0, 5.0] = [21.0, 21.0, 21.0]
        >>> # input_x[1] = [21.0, 21.0, 21.0] / [7.0, 7.0, 7.0] = [3.0, 3.0, 3.0]
        >>> indices = mindspore.tensor([[0, 1], [1, 1]], mindspore.int32)
        >>> updates = mindspore.tensor([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[5.0, 5.0, 5.0], [7.0, 7.0, 7.0]]], mindspore.float32)
        >>> output = mindspore.ops.scatter_div(input_x, indices, updates)
        >>> print(output)
        [[105. 105. 105.]
         [  3.   3.   3.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = mindspore.Parameter(mindspore.tensor([[105.0, 105.0, 105.0],
        ...                                                [315.0, 315.0, 315.0]], mindspore.float32), name="x")
        >>> # for indices = [[1, 0], [1, 1]]
        >>> # step 1: [1, 0]
        >>> # input_x[0] = [105.0, 105.0, 105.0] / [3.0, 3.0, 3.0] = [35.0, 35.0, 35.0]
        >>> # input_x[1] = [315.0, 315.0, 315.0] / [1.0, 1.0, 1.0] = [315.0, 315.0, 315.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [315.0, 315.0, 315.0] / [5.0, 5.0, 5.0] = [63.0 63.0 63.0]
        >>> # input_x[1] = [63.0 63.0 63.0] / [7.0, 7.0, 7.0] = [9.0, 9.0, 9.0]
        >>> indices = mindspore.tensor([[1, 0], [1, 1]], mindspore.int32)
        >>> updates = mindspore.tensor([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[5.0, 5.0, 5.0], [7.0, 7.0, 7.0]]], mindspore.float32)
        >>> output = mindspore.ops.scatter_div(input_x, indices, updates)
        >>> print(output)
        [[35. 35. 35.]
         [ 9.  9.  9.]]
    """
    return scatter_div_(input_x, indices, updates)


def scatter_update(input_x, indices, updates):
    r"""
    Updates the input tensor values using the given input indices and update values.

    .. note::
        - Support implicit type conversion and type promotion.
        - Since Parameter objects do not support type conversion,
          an exception will be thrown when input_x is of a low-precision data type.
        - The `updates` with a shape of `indices.shape + input_x.shape[1:]` .

    for each `i, ..., j` in `indices.shape`:

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :] = \text{updates}[i, ..., j, :]

    Args:
        input_x (Union[Parameter, Tensor]): The input parameter or tensor.
        indices (Tensor): Specify the indices for update operation.
            If there are duplicates in indices, the order for updating is undefined.
        updates (Tensor): The values to update.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> np_x = mindspore.tensor([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]], mindspore.float32)
        >>> input_x = mindspore.Parameter(np_x, name="x")
        >>> indices = mindspore.tensor([0, 1], mindspore.int32)
        >>> np_updates =  mindspore.tensor([[2.0, 1.2, 1.0], [3.0, 1.2, 1.0]])
        >>> updates =  mindspore.tensor(np_updates, mindspore.float32)
        >>> output = mindspore.ops.scatter_update(input_x, indices, updates)
        >>> print(output)
        [[2. 1.2  1.]
         [3. 1.2  1.]]
    """
    return scatter_update_(input_x, indices, updates)


def scatter_nd_add(input_x, indices, updates, use_locking=False):
    r"""
    Perform a sparse addition update on `input_x` based on the specified indices and update values.

    .. math::
        \text{input_x}[\text{indices}[i, ..., j]] \mathrel{+}= \text{updates}[i, ..., j]

    .. note::
        - Support implicit type conversion and type promotion.
        - The dimension of `indices` is at least 2, and its shape must be `indices.shape[-1] <= len(indices.shape)`.
        - The shape of `updates` is `indices.shape[:-1] + input_x.shape[indices.shape[-1]:]`.

    Args:
        input_x (Union[Parameter, Tensor]): The input parameter or tensor.
        indices (Tensor): The specified indices.
        updates (Tensor): The update values.
        use_locking (bool, optional): Whether to protect the assignment by a lock. Default: ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.Parameter(mindspore.tensor([1, 2, 3, 4, 5, 6, 7, 8],
        ...                               mindspore.float32), name="x")
        >>> indices = mindspore.tensor([[2], [4], [1], [7]], mindspore.int32)
        >>> updates = mindspore.tensor([6, 7, 8, 9], mindspore.float32)
        >>> output = mindspore.ops.scatter_nd_add(input_x, indices, updates, False)
        >>> print(output)
        [ 1. 10.  9.  4. 12.  6.  7. 17.]
        >>> input_x = mindspore.Parameter(mindspore.tensor(mindspore.ops.zeros((4, 4, 4)), mindspore.int32))
        >>> indices = mindspore.tensor([[0], [2]], mindspore.int32)
        >>> updates = mindspore.tensor([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]], mindspore.int32)
        >>> output = mindspore.ops.scatter_nd_add(input_x, indices, updates, False)
        >>> print(output)
        [[[1 1 1 1]
          [2 2 2 2]
          [3 3 3 3]
          [4 4 4 4]]
         [[0 0 0 0]
          [0 0 0 0]
          [0 0 0 0]
          [0 0 0 0]]
         [[5 5 5 5]
          [6 6 6 6]
          [7 7 7 7]
          [8 8 8 8]]
         [[0 0 0 0]
          [0 0 0 0]
          [0 0 0 0]
          [0 0 0 0]]]
    """
    scatter_nd_add_inner = _get_cache_prim(P.ScatterNdAdd)(use_locking)
    return scatter_nd_add_inner(input_x, indices, updates)


def scatter_nd_sub(input_x, indices, updates, use_locking=False):
    r"""
    Perform a sparse subtraction update on `input_x` based on the specified indices and update values.

    .. math::
        \text{input_x}[\text{indices}[i, ..., j]] \mathrel{-}= \text{updates}[i, ..., j]

    .. note::
        - Support implicit type conversion and type promotion.
        - The dimension of `indices` is at least 2, and its shape must be `indices.shape[-1] <= len(indices.shape)`.
        - The shape of `updates` is `indices.shape[:-1] + input_x.shape[indices.shape[-1]:]`.

    Args:
        input_x (Union[Parameter, Tensor]): The input parameter or tensor.
        indices (Tensor): The specified indices.
        updates (Tensor): The update values.
        use_locking (bool): Whether to protect the assignment by a lock. Default: ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.Parameter(mindspore.tensor([1, 2, 3, 4, 5, 6, 7, 8],
        ...                               mindspore.float32), name="x")
        >>> indices = mindspore.tensor([[2], [4], [1], [7]], mindspore.int32)
        >>> updates = mindspore.tensor([6, 7, 8, 9], mindspore.float32)
        >>> output = mindspore.ops.scatter_nd_sub(input_x, indices, updates, False)
        >>> print(output)
        [ 1. -6. -3.  4. -2.  6.  7. -1.]
        >>> input_x = mindspore.Parameter(mindspore.tensor(mindspore.ops.zeros((4, 4, 4)), mindspore.int32))
        >>> indices = mindspore.tensor([[0], [2]], mindspore.int32)
        >>> updates = mindspore.tensor([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]], mindspore.int32)
        >>> output = mindspore.ops.scatter_nd_sub(input_x, indices, updates, False)
        >>> print(output)
        [[[-1 -1 -1 -1]
          [-2 -2 -2 -2]
          [-3 -3 -3 -3]
          [-4 -4 -4 -4]]
         [[ 0  0  0  0]
          [ 0  0  0  0]
          [ 0  0  0  0]
          [ 0  0  0  0]]
         [[-5 -5 -5 -5]
          [-6 -6 -6 -6]
          [-7 -7 -7 -7]
          [-8 -8 -8 -8]]
         [[ 0  0  0  0]
          [ 0  0  0  0]
          [ 0  0  0  0]
          [ 0  0  0  0]]]
    """
    scatter_nd_sub_inner = _get_cache_prim(P.ScatterNdSub)(use_locking)
    return scatter_nd_sub_inner(input_x, indices, updates)


def scatter_nd_mul(input_x, indices, updates, use_locking=False):
    r"""
    Perform a sparse multiplication update on `input_x` based on the specified indices and update values.

    .. math::
        \text{input_x}[\text{indices}[i, ..., j]] \mathrel{*}= \text{updates}[i, ..., j]

    .. note::
        - Support implicit type conversion and type promotion.
        - The dimension of `indices` is at least 2, and its shape must be `indices.shape[-1] <= len(indices.shape)`.
        - The shape of `updates` is `indices.shape[:-1] + input_x.shape[indices.shape[-1]:]`.

    Args:
        input_x (Union[Parameter, Tensor]): The input parameter or tensor.
        indices (Tensor): The specified indices.
        updates (Tensor): The update values.
        use_locking (bool): Whether to protect the assignment by a lock. Default: ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.Parameter(mindspore.tensor([1, 2, 3, 4, 5, 6, 7, 8],
        ...                               mindspore.float32), name="x")
        >>> indices = mindspore.tensor([[2], [4], [1], [7]], mindspore.int32)
        >>> updates = mindspore.tensor([6, 7, 8, 9], mindspore.float32)
        >>> output = mindspore.ops.scatter_nd_mul(input_x, indices, updates)
        >>> print(output)
        [ 1. 16. 18.  4. 35.  6.  7. 72.]
        >>> input_x = mindspore.Parameter(mindspore.tensor(mindspore.ops.ones((4, 4, 4)), mindspore.int32))
        >>> indices = mindspore.tensor([[0], [2]], mindspore.int32)
        >>> updates = mindspore.tensor([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]], mindspore.int32)
        >>> output = mindspore.ops.scatter_nd_mul(input_x, indices, updates)
        >>> print(output)
        [[[1 1 1 1]
          [2 2 2 2]
          [3 3 3 3]
          [4 4 4 4]]
         [[1 1 1 1]
          [1 1 1 1]
          [1 1 1 1]
          [1 1 1 1]]
         [[5 5 5 5]
          [6 6 6 6]
          [7 7 7 7]
          [8 8 8 8]]
         [[1 1 1 1]
          [1 1 1 1]
          [1 1 1 1]
          [1 1 1 1]]]
    """
    scatter_nd_mul_inner = _get_cache_prim(ScatterNdMul)(use_locking)
    return scatter_nd_mul_inner(input_x, indices, updates)


def scatter_nd_div(input_x, indices, updates, use_locking=False):
    r"""
    Perform a sparse division update on `input_x` based on the specified indices and update values.

    .. math::
        \text{input_x}[\text{indices}[i, ..., j]] \mathrel{/}= \text{updates}[i, ..., j]

    .. note::
        - Support implicit type conversion and type promotion.
        - The dimension of `indices` is at least 2, and its shape must be `indices.shape[-1] <= len(indices.shape)`.
        - The shape of `updates` is `indices.shape[:-1] + input_x.shape[indices.shape[-1]:]`.

    Args:
        input_x (Union[Parameter, Tensor]): The input parameter or tensor.
        indices (Tensor): The specified indices.
        updates (Tensor): The update values.
        use_locking (bool): Whether to protect the assignment by a lock. Default: ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.Parameter(mindspore.tensor([1, 2, 3, 4, 5, 6, 7, 8],
        ...                               mindspore.float32), name="x")
        >>> indices = mindspore.tensor([[2], [4], [1], [7]], mindspore.int32)
        >>> updates = mindspore.tensor([6, 7, 8, 9], mindspore.float32)
        >>> output = mindspore.ops.scatter_nd_div(input_x, indices, updates, False)
        >>> print(output)
        [1.         0.25       0.5        4.         0.71428573 6.
         7.         0.8888889 ]
        >>> input_x = mindspore.Parameter(mindspore.tensor(mindspore.ops.ones((4, 4, 4)), mindspore.float32))
        >>> indices = mindspore.tensor([[0], [2]], mindspore.int32)
        >>> updates = mindspore.tensor([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]], mindspore.float32)
        >>> output = mindspore.ops.scatter_nd_div(input_x, indices, updates, False)
        >>> print(output)
        [[[1.         1.         1.         1.        ]
          [0.5        0.5        0.5        0.5       ]
          [0.33333334 0.33333334 0.33333334 0.33333334]
          [0.25       0.25       0.25       0.25      ]]
         [[1.         1.         1.         1.        ]
          [1.         1.         1.         1.        ]
          [1.         1.         1.         1.        ]
          [1.         1.         1.         1.        ]]
         [[0.2        0.2        0.2        0.2       ]
          [0.16666667 0.16666667 0.16666667 0.16666667]
          [0.14285715 0.14285715 0.14285715 0.14285715]
          [0.125      0.125      0.125      0.125     ]]
         [[1.         1.         1.         1.        ]
          [1.         1.         1.         1.        ]
          [1.         1.         1.         1.        ]
          [1.         1.         1.         1.        ]]]
    """
    scatter_nd_div_inner = _get_cache_prim(P.ScatterNdDiv)(use_locking)
    return scatter_nd_div_inner(input_x, indices, updates)


def scatter_nd_max(input_x, indices, updates, use_locking=False):
    r"""
    Perform a sparse maximum update on `input_x` based on the specified indices and update values.

    .. math::
        \text{input_x}[\text{indices}[i, ..., j]]
        = \max(\text{input_x}[\text{indices}[i, ..., j]], \text{updates}[i, ..., j])

    .. note::
        - Support implicit type conversion and type promotion.
        - The dimension of `indices` is at least 2, and its shape must be `indices.shape[-1] <= len(indices.shape)`.
        - The shape of `updates` is `indices.shape[:-1] + input_x.shape[indices.shape[-1]:]`.

    Args:
        input_x (Union[Parameter, Tensor]): The input parameter or tensor.
        indices (Tensor): The specified indices.
        updates (Tensor): The update values.
        use_locking (bool): Whether to protect the assignment by a lock. Default: ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.Parameter(mindspore.tensor([1, 2, 3, 4, 5, 6, 7, 8],
        ...                               mindspore.float32), name="x")
        >>> indices = mindspore.tensor([[2], [4], [1], [7]], mindspore.int32)
        >>> updates = mindspore.tensor([6, 7, 8, 9], mindspore.float32)
        >>> output = mindspore.ops.scatter_nd_max(input_x, indices, updates, False)
        >>> print(output)
        [1. 8. 6. 4. 7. 6. 7. 9.]
        >>> input_x = mindspore.Parameter(mindspore.tensor(mindspore.ops.ones((4, 4, 4)), mindspore.int32))
        >>> indices = mindspore.tensor([[0], [2]], mindspore.int32)
        >>> updates = mindspore.tensor([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]], mindspore.int32)
        >>> output = mindspore.ops.scatter_nd_max(input_x, indices, updates, False)
        >>> print(output)
        [[[1 1 1 1]
          [2 2 2 2]
          [3 3 3 3]
          [4 4 4 4]]
         [[1 1 1 1]
          [1 1 1 1]
          [1 1 1 1]
          [1 1 1 1]]
         [[5 5 5 5]
          [6 6 6 6]
          [7 7 7 7]
          [8 8 8 8]]
         [[1 1 1 1]
          [1 1 1 1]
          [1 1 1 1]
          [1 1 1 1]]]
    """
    scatter_nd_max_inner = _get_cache_prim(ScatterNdMax)(use_locking)
    return scatter_nd_max_inner(input_x, indices, updates)


def scatter_nd_min(input_x, indices, updates, use_locking=False):
    r"""
    Perform a sparse minimum update on `input_x` based on the specified indices and update values.

    .. math::
        \text{input_x}[\text{indices}[i, ..., j]]
        = \min(\text{input_x}[\text{indices}[i, ..., j]], \text{updates}[i, ..., j])

    .. note::
        - Support implicit type conversion and type promotion.
        - The dimension of `indices` is at least 2, and its shape must be `indices.shape[-1] <= len(indices.shape)`.
        - The shape of `updates` is `indices.shape[:-1] + input_x.shape[indices.shape[-1]:]`.

    Args:
        input_x (Union[Parameter, Tensor]): The input parameter or tensor.
        indices (Tensor): The specified indices.
        updates (Tensor): The update values.
        use_locking (bool): Whether to protect the assignment by a lock. Default: ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.Parameter(mindspore.tensor(mindspore.ops.ones(8) * 10, mindspore.float32), name="x")
        >>> indices = mindspore.tensor([[2], [4], [1], [7]], mindspore.int32)
        >>> updates = mindspore.tensor([6, 7, 8, 9], mindspore.float32)
        >>> output = mindspore.ops.scatter_nd_min(input_x, indices, updates, False)
        >>> print(output)
        [10.  8.  6. 10.  7. 10. 10.  9.]
        >>> input_x = mindspore.Parameter(mindspore.tensor(mindspore.ops.ones((4, 4, 4)) * 10, mindspore.int32))
        >>> indices = mindspore.tensor([[0], [2]], mindspore.int32)
        >>> updates = mindspore.tensor([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]], mindspore.int32)
        >>> output = mindspore.ops.scatter_nd_min(input_x, indices, updates, False)
        >>> print(output)
        [[[ 1  1  1  1]
          [ 2  2  2  2]
          [ 3  3  3  3]
          [ 4  4  4  4]]
         [[10 10 10 10]
          [10 10 10 10]
          [10 10 10 10]
          [10 10 10 10]]
         [[ 5  5  5  5]
          [ 6  6  6  6]
          [ 7  7  7  7]
          [ 8  8  8  8]]
         [[10 10 10 10]
          [10 10 10 10]
          [10 10 10 10]
          [10 10 10 10]]]
    """
    scatter_nd_min_inner = _get_cache_prim(P.ScatterNdMin)(use_locking)
    return scatter_nd_min_inner(input_x, indices, updates)


def sort(input_x, axis=-1, descending=False):
    r"""
    Sort the elements of the input tensor along the given axis.

    .. note::
        The Ascend backend only supports sorting the last dimension.

    Args:
        input_x(Tensor): The input tensor.
        axis (int, optional): The axis to sort along. Default ``-1`` , means the last dimension.
        descending (bool, optional): Sorting method. ``True`` means the elements
            are sorted in descending order, or else sorted in ascending order. Default ``False`` .

    .. warning::
        Currently, the data types of float16, uint8, int8, int16, int32, int64 are well supported.
        If use float32, it may cause loss of accuracy.

    Returns:
        Tuple(sorted_tensor, indices) of 2 tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[8, 2, 1], [5, 9, 3], [4, 6, 7]], mindspore.float16)
        >>> output = mindspore.ops.sort(x)
        >>> # The output below is based on the Ascend platform.
        >>> print(output)
        (Tensor(shape=[3, 3], dtype=Float16, value=
        [[ 1.0000e+00,  2.0000e+00,  8.0000e+00],
        [ 3.0000e+00,  5.0000e+00,  9.0000e+00],
        [ 4.0000e+00,  6.0000e+00,  7.0000e+00]]), Tensor(shape=[3, 3], dtype=Int32, value=
        [[2, 1, 0],
        [2, 0, 1],
        [0, 1, 2]]))
    """
    _sort = _get_cache_prim(P.Sort)(axis, descending)
    return _sort(input_x)


def sort_ext(input, *, dim=-1, descending=False, stable=False):
    r"""
    Sorts the elements of the input tensor along the given dimension in the specified order.

    .. warning::
        Currently, the data types of float16, uint8, int8, int16, int32, int64 are well supported.
        If use float32, it may cause loss of accuracy.

    Args:
        input(Tensor): The input tensor to sort.
            The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Keyword Args:
        dim (int, optional): The dimension to sort along. Default: ``-1``, means the last dimension.
        descending (bool, optional): Controls the sort order. If `descending` is True, the elements
            are sorted in descending order, or else sorted in ascending order. Default: ``False`` .
        stable (bool, optional): Controls the sort order. If stable is True then the sorting routine
            becomes stable, preserving the order of equivalent elements. Default: ``False`` .

    Returns:
        - y1, a tensor whose values are the sorted values, with the same shape and data type as input.
        - y2, a tensor that consists of the indices of the elements in the original input tensor.
          Data type is int64.

    Raises:
        TypeError: If `dim` is not an int.
        TypeError: If `descending` is not a bool.
        TypeError: If `input` not in float16, float32, uint8, int8, int16, int32, int64, bfloat16
        TypeError: If `stable` is not a bool.
        ValueError: If `dim` is not in range of [-len(input.shape), len(input.shape)).

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]), mindspore.float16)
        >>> output = ops.function.array_func.sort_ext(x)
        >>> # The output below is based on the Ascend platform.
        >>> print(output)
        (Tensor(shape=[3, 3], dtype=Float16, value=
        [[ 1.0000e+00,  2.0000e+00,  8.0000e+00],
        [ 3.0000e+00,  5.0000e+00,  9.0000e+00],
        [ 4.0000e+00,  6.0000e+00,  7.0000e+00]]), Tensor(shape=[3, 3], dtype=Int64, value=
        [[2, 1, 0],
        [2, 0, 1],
        [0, 1, 2]]))
    """
    return sort_ext_(input, dim, descending, stable)


def argsort(input, axis=-1, descending=False):
    r"""
    Return the indices that sort the tensor along the specified axis.

    .. note::
        The Ascend backend only supports sorting the last dimension.

    Args:
        input(Tensor): The input tensor.
        axis (int): Specify the axis to sort along. Default ``-1`` .
        descending (bool): Specify the sorting order (ascending or descending).

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: 1-dimensional sort
        >>> input = mindspore.tensor([1, 3, 5, 4, 2, 1])
        >>> mindspore.ops.argsort(input)
        Tensor(shape=[6], dtype=Int32, value= [0, 5, 4, 1, 3, 2])
        >>>
        >>> # case 2: multi-dimensional sort
        >>> input = mindspore.tensor([[2, 1, 3],
        ...                           [6, 4, 3]])
        >>> mindspore.ops.argsort(input, axis=1)
        Tensor(shape=[2, 3], dtype=Int32, value=
        [[1, 0, 2],
         [2, 1, 0]])
        >>> mindspore.ops.argsort(input, axis=1, descending=True)
        Tensor(shape=[2, 3], dtype=Int32, value=
        [[2, 0, 1],
         [0, 1, 2]])
    """
    _sort = _get_cache_prim(P.Sort)(axis, descending)
    _, arg_sort = _sort(input)
    return arg_sort


def gather_elements(input, dim, index):
    """
    Gathers elements along the specified dim and indices.

    .. note::
        `input` and `index` have the same length of dimensions, and `index.shape[axis] <= input.shape[axis]`
        where axis goes through all dimensions of `input` except `dim`.

    .. warning::
        On Ascend, the behavior is unpredictable in the following cases:

        - the value of `index` is not in the range `[-input.shape[dim], input.shape[dim])` in forward;
        - the value of `index` is not in the range `[0, input.shape[dim])` in backward.

    Args:
        input (Tensor): The input tensor.
        dim (int): The specified dim.
        index (Tensor): The specified indices.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[1, 2], [3, 4]], mindspore.int32)
        >>> index = mindspore.tensor([[0, 0], [1, 0]], mindspore.int32)
        >>> dim = 1
        >>> output = mindspore.ops.gather_elements(x, dim, index)
        >>> print(output)
        [[1 1]
         [4 3]]
    """
    return gather_d_(input, dim, index)


def tensor_scatter_sub(input_x, indices, updates):
    r"""
    Return a new tensor by performing a subtraction update on `input_x` at the specified indices with the given update
    values.

    .. math::
        output[indices] = input\_x - updates

    Note:
        On GPU, if some values of the `indices` are out of bound, instead of raising an index error,
        the corresponding `updates` will not be updated to self tensor. On CPU, if some values of
        the `indices` are out of bound, raising an index error. On Ascend, out of bound checking is
        not supported, if some values of the `indices` are out of bound, unknown errors may be caused.

    Args:
        input_x (Tensor): The input tensor.
        indices (Tensor): The specified indices. The rank must be at least 2.
        updates (Tensor): The update values.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[1, 2, 3], [4, 5, 6]], mindspore.float32)
        >>> indices = mindspore.tensor([[0, 0], [1, 1]])
        >>> updates = mindspore.tensor([5, 5], mindspore.float32)
        >>> mindspore.ops.tensor_scatter_sub(input, indices, updates)
        Tensor(shape=[2, 3], dtype=Float32, value=
        [[-4.00000000e+00,  2.00000000e+00,  3.00000000e+00],
         [ 4.00000000e+00,  0.00000000e+00,  6.00000000e+00]])
    """

    return tensor_scatter_sub_(input_x, indices, updates)


def tensor_scatter_max(input_x, indices, updates):
    r"""
    Return a new tensor by performing a maximum update on `input_x` at the specified indices with the given update
    values.

    .. math::
        output\left [indices  \right ] = \max(input\_x, updates)

    Note:
        - On GPU, if some values of the `indices` are out of bound, instead of raising an index error,
          the corresponding `updates` will not be updated to self tensor.
        - On CPU, if some values of the `indices` are out of bound, raising an index error.
        - On Ascend, out of bound checking is not supported, if some values of the `indices` are out of bound,
          unknown errors may be caused.

    Args:
        input_x (Tensor): The input tensor.
        indices (Tensor): The specified indices. The rank must be at least 2.
        updates (Tensor): The update values.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.tensor([[1, 2, 3], [4, 5, 6]])
        >>> indices = mindspore.tensor([[0, 0], [1, 1]])
        >>> updates = mindspore.tensor([5, 5])
        >>> mindspore.ops.tensor_scatter_max(input_x, indices, updates)
        Tensor(shape=[2, 3], dtype=Int64, value=
        [[5, 2, 3],
         [4, 5, 6]])
    """
    return tensor_scatter_max_(input_x, indices, updates)


def tensor_scatter_min(input_x, indices, updates):
    r"""
    Return a new tensor by performing a minimum update on `input_x` at the specified indices with the given update
    values.

    .. math::
        output\left [indices  \right ] = \min(input\_x, updates)

    Note:
        - On GPU, if some values of the `indices` are out of bound, instead of raising an index error,
          the corresponding `updates` will not be updated to self tensor.
        - On CPU, if some values of the `indices` are out of bound, raising an index error.
        - On Ascend, out of bound checking is not supported, if some values of the `indices` are out of bound,
          unknown errors may be caused.

    Args:
        input_x (Tensor): The input tensor.
        indices (Tensor): The specified indices. The rank must be at least 2.
        updates (Tensor): The update values.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.tensor([[1, 2, 3], [4, 5, 6]], mindspore.float32)
        >>> indices = mindspore.tensor([[0, 0], [1, 1]])
        >>> updates = mindspore.tensor([5, 1], mindspore.float32)
        >>> mindspore.ops.tensor_scatter_min(input_x, indices, updates)
        Tensor(shape=[2, 3], dtype=Float32, value=
        [[ 1.00000000e+00,  2.00000000e+00,  3.00000000e+00],
         [ 4.00000000e+00,  1.00000000e+00,  6.00000000e+00]])
    """
    return tensor_scatter_min_(input_x, indices, updates)


def tensor_scatter_elements(input_x, indices, updates, axis=0, reduction="none"):
    """
    Return a new tensor by performing a specified operation update on `input_x` at the specified indices with the given
    update values.

    Not support implicit type conversion.

    For example:  the output of a 3-D tensor is

    .. code-block::

        output[indices[i][j][k]][j][k] = updates[i][j][k]  # if axis == 0, reduction == "none"

        output[i][indices[i][j][k]][k] += updates[i][j][k]  # if axis == 1, reduction == "add"

        output[i][j][indices[i][j][k]] = updates[i][j][k]  # if axis == 2, reduction == "none"

    .. warning::
        - The order in which updates are applied is nondeterministic, meaning that if there are multiple index vectors
          in `indices` that correspond to the same position, the value of that position in the output will be
          nondeterministic.
        - On Ascend, the reduction only support set to "none" for now.
        - On Ascend, the data type of `input_x` must be float16 or float32.
        - This is an experimental API that is subject to change or deletion.

    Note:
        If some values of the `indices` exceed the upper or lower bounds of the index of `input_x`, instead of raising
        an index error, the corresponding `updates` will not be updated to `input_x`.
        The backward is supported only for the case `updates.shape == indices.shape`.

    Args:
        input_x (Tensor): The input tensor. The rank must be at least 1.
        indices (Tensor): The specified indices.
        updates (Tensor): The update values.
        axis (int): The axis along which to index. Default ``0``.
        reduction (str): The specified operation, supports ``none`` , ``add`` .

          - If ``none``, `updates` will be assigned to `input_x` according to  `indices`.
          - If ``add``, `updates` will be added to `input_x` according to  `indices`. Default ``none``.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.tensor([[1, 2, 3, 4, 5]])
        >>> indices = mindspore.tensor([[2, 4]])
        >>> updates = mindspore.tensor([[8, 8]])
        >>> output = mindspore.ops.tensor_scatter_elements(input_x, indices, updates, axis=1, reduction="none")
        >>> print(output)
        [[1 2 8 4 8]]
        >>> output = mindspore.ops.tensor_scatter_elements(input_x, indices, updates, axis=1, reduction="add")
        >>> print(output)
        [[ 1  2 11  4 13]]
    """
    return tensor_scatter_elements_ext(input_x, indices, updates, axis, reduction)


def scatter(input, axis, index, src):
    """
    Update the value in `src` to `input` according to the specified index.
    Refer to :func:`mindspore.ops.tensor_scatter_elements` for more details.

    .. note::
        If `src` is a tensor, the backward is supported only for the case `src.shape == index.shape`.

    Args:
        input (Tensor): The input tensor.
        axis (int): The axis to do update operation.
        index (Tensor): The index to do update operation.
        src (Tensor, float): The data to do the update operation with `input` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[1, 2, 3, 4, 5]], dtype=mindspore.float32)
        >>> src = mindspore.tensor([[8, 8]], dtype=mindspore.float32)
        >>> index = mindspore.tensor([[2, 4]], dtype=mindspore.int64)
        >>> out = mindspore.ops.scatter(input=input, axis=1, index=index, src=src)
        >>> print(out)
        [[1. 2. 8. 4. 8.]]
        >>> input = mindspore.tensor(mindspore.ops.zeros((5, 5)), dtype=mindspore.float32)
        >>> src = mindspore.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=mindspore.float32)
        >>> index = mindspore.tensor([[0, 0, 0], [2, 2, 2], [4, 4, 4]], dtype=mindspore.int64)
        >>> out = mindspore.ops.scatter(input=input, axis=0, index=index, src=src)
        >>> print(out)
        [[1. 2. 3. 0. 0.]
        [0. 0. 0. 0. 0.]
        [4. 5. 6. 0. 0.]
        [0. 0. 0. 0. 0.]
        [7. 8. 9. 0. 0.]]
        >>> input = mindspore.tensor(mindspore.ops.zeros((5, 5)), dtype=mindspore.float32)
        >>> src = mindspore.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=mindspore.float32)
        >>> index = mindspore.tensor([[0, 2, 4], [0, 2, 4], [0, 2, 4]], dtype=mindspore.int64)
        >>> out = mindspore.ops.scatter(input=input, axis=1, index=index, src=src)
        >>> print(out)
        [[1. 0. 2. 0. 3.]
        [4. 0. 5. 0. 6.]
        [7. 0. 8. 0. 9.]
        [0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0.]]
    """
    if isinstance(src, Tensor):
        return scatter_prim(input, axis, index, src)
    return scatter_value_(input, axis, index, src)


def scatter_add_ext(input, dim, index, src):
    """
    Add all elements in `src` to the index specified by `index` to `input` along dimension specified by `dim`.
    It takes three inputs `input`, `src` and `index` of the same rank r >= 1.

    For a 3-D tensor, the operation updates input as follows:

    .. code-block::

        input[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0

        input[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1

        input[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2

    Args:
        input (Tensor): The target tensor. The rank must be at least 1.
        dim (int): Which dim to scatter. Accepted range is [-r, r) where r = rank(`input`).
        index (Tensor): The index of `input` to do scatter operation whose data type must be mindspore.int32 or
            mindspore.int64. Same rank as `input`. Except for the dimension specified by `dim`,
            the size of each dimension of `index` must be less than or equal to the size of
            the corresponding dimension of `input`.
        src (Tensor): The tensor doing the scatter operation with `input`, has the same type as `input` and
            the size of each dimension must be greater than or equal to that of `index`.

    Returns:
        Tensor, has the same shape and type as `input`.

    Raises:
        TypeError: If `index` is neither int32 nor int64.
        ValueError: If anyone of the rank among `input`, `index` and `src` is less than 1.
        ValueError: If the rank of `input`, `index` and `src` is not the same.
        ValueError: The size of any dimension of `index` except the dimension specified by `dim` is
            greater than the size of the corresponding dimension of `input`.
        ValueError: If the size of any dimension of `src` is less than that of `index`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([[1, 2, 3, 4, 5]]), dtype=ms.float32)
        >>> src = Tensor(np.array([[8, 8]]), dtype=ms.float32)
        >>> index = Tensor(np.array([[2, 4]]), dtype=ms.int64)
        >>> out = ops.function.array_func.scatter_add_ext(input=input, dim=1, index=index, src=src)
        >>> print(out)
        [[1. 2. 11. 4. 13.]]
        >>> input = Tensor(np.zeros((5, 5)), dtype=ms.float32)
        >>> src = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=ms.float32)
        >>> index = Tensor(np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]]), dtype=ms.int64)
        >>> out = ops.function.array_func.scatter_add_ext(input=input, dim=0, index=index, src=src)
        >>> print(out)
        [[1. 2. 3. 0. 0.]
         [0. 0. 0. 0. 0.]
         [4. 5. 6. 0. 0.]
         [0. 0. 0. 0. 0.]
         [7. 8. 9. 0. 0.]]
        >>> input = Tensor(np.zeros((5, 5)), dtype=ms.float32)
        >>> src = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=ms.float32)
        >>> index = Tensor(np.array([[0, 2, 4], [0, 2, 4], [0, 2, 4]]), dtype=ms.int64)
        >>> out = ops.function.array_func.scatter_add_ext(input=input, dim=1, index=index, src=src)
        >>> print(out)
        [[1. 0. 2. 0. 3.]
         [4. 0. 5. 0. 6.]
         [7. 0. 8. 0. 9.]
         [0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0.]]
    """
    return scatter_add_ext_op(input, dim, index, src)


def _get_slice_scatter_const(x_shape, axis, start, end, step):
    r"""
    Calculate the rank of input, embedded dimensions and index.
    """
    x_rank = len(x_shape)
    axis = axis if axis >= 0 else axis + x_rank
    start = start if start is not None else 0
    start = start if start >= 0 else start + x_shape[axis]
    end = end if end is not None else x_shape[axis]
    end = end if end >= 0 else end + x_shape[axis]
    end = end if end < x_shape[axis] else x_shape[axis]
    index = list(builtins.range(start, end, step))
    return x_rank, index, axis


def slice_scatter(input, src, axis=0, start=None, end=None, step=1):
    r"""
    Embed `src` into the sliced `input` along the specified `axis` .

    Args:
        input (Tensor): The input tensor.
        src (Tensor): The source tensor to be embedded into `input` .
        axis (int, optional): The axis of `input` to be sliced. Default ``0`` .
        start (int, optional): The start index for embedding in the specified axis.
            Default ``None`` , which means `start` is ``0`` .
        end (int, optional): The end index for embedding in the specified axis.
            Default ``None`` , which means `end` is the length of `input` in the specified axis.
        step (int, optional): The step size to skip during embedding. Default ``1`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> a = mindspore.ops.zeros((4, 6))
        >>> b = mindspore.ops.ones((4, 3))
        >>> output = mindspore.ops.slice_scatter(input=a, src=b, axis=1, start=0, end=5, step=2)
        >>> print(output)
        [[1. 0. 1. 0. 1. 0.]
         [1. 0. 1. 0. 1. 0.]
         [1. 0. 1. 0. 1. 0.]
         [1. 0. 1. 0. 1. 0.]]
    """
    _check_is_tensor("input", input, "slice_scatter")
    _check_is_tensor("src", src, "slice_scatter")
    input_shape = input.shape
    input_rank, index, axis = _get_slice_scatter_const(
        input_shape, axis, start, end, step)

    src_shape = src.shape
    index_shape = input_shape[:axis] + (len(index),) + input_shape[axis + 1:]
    index_tensor = ms.Tensor(index)
    for _ in builtins.range(axis):
        index_tensor = index_tensor.expand_dims(0)

    if index_shape != src_shape:
        raise ValueError(f"For slice_scatter, src shape should be equal to the slice size,"
                         f"but got src shape {src_shape} and slice shape {index_shape}")
    for _ in builtins.range(input_rank - axis - 1):
        index_tensor = index_tensor.expand_dims(-1)
    index_tensor = index_tensor.broadcast_to(src.shape)
    if index_tensor.dtype not in mstype.int_type:
        index_tensor = index_tensor.astype(mstype.int64)
    return tensor_scatter_elements(input, axis=axis, indices=index_tensor, updates=src)


def select_scatter(input, src, axis, index):
    r"""
    On the specified dimension `axis` of `input` , `src` is scattered into `input` on the specified `index` of `input` .

    Args:
        input (Tensor): The input tensor.
        src (Tensor): The source tensor.
        axis (int): The dimension of `input` to be embedded.
        index (int): The location of scattering on the specified dimension.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.ops.zeros((2, 3, 3))
        >>> src = mindspore.ops.ones((2, 3))
        >>> output = mindspore.ops.select_scatter(input, src, axis=1, index=1)
        >>> print(output)
        [[[0. 0. 0.]
          [1. 1. 1.]
          [0. 0. 0.]]
         [[0. 0. 0.]
          [1. 1. 1.]
          [0. 0. 0.]]]
    """
    _check_is_tensor("input", input, "select_scatter")
    _check_is_tensor("src", src, "select_scatter")
    src = src.expand_dims(axis=axis)
    x_rank = input.ndim
    axis = axis if axis >= 0 else axis + x_rank
    index = index if index >= 0 else index + input.shape[axis]
    return slice_scatter(input, src, axis, start=index, end=index + 1)


def space_to_batch_nd(input_x, block_size, paddings):
    r"""
    Divides a tensor's spatial dimensions into blocks and combines the block sizes with the original batch.

    .. math::
        \begin{array}{ll} \\
            n' = n*(block\_size[0] * ... * block\_size[M]) \\
            w'_i = (w_i + paddings[i][0] + paddings[i][1])//block\_size[i]
        \end{array}

    .. note::
        - This operation divides the spatial dimensions [1, ..., M] of the input into blocks of size `block_size` and
          interleaves them into the batch dimension (default: dimension 0). Before splitting, the spatial dimensions are
          padded with zeros according to `paddings`.
        - If the input shape is :math:`(n, c_1, ... c_k, w_1, ..., w_M)`, then the output shape will be
          :math:`(n', c_1, ... c_k, w'_1, ..., w'_M)`.
        - If `block_size` is a tuple or list, the length of `block_size` is M corresponding to the number of spatial
          dimensions. If `block_size` is an int, the block size of M dimensions are the same, equal to `block_size`.
          M must be 2 on Ascend.

    Args:
        input_x (Tensor): The input tensor, must be a 4-D tensor on Ascend.
        block_size (Union[list(int), tuple(int), int]): Specifies the block size for spatial dimension division.
        paddings (Union[tuple, list]): The padding size for each spatial dimension.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> block_size = [2, 2]
        >>> paddings = [[0, 0], [0, 0]]
        >>> input_x = mindspore.tensor([[[[1, 2], [3, 4]]]], mindspore.float32)
        >>> output = mindspore.ops.space_to_batch_nd(input_x, block_size, paddings)
        >>> print(output)
        [[[[1.]]]
         [[[2.]]]
         [[[3.]]]
         [[[4.]]]]
    """
    _space_to_batch_nd = _get_cache_prim(
        P.SpaceToBatchND)(block_size, paddings)
    return _space_to_batch_nd(input_x)


def batch_to_space_nd(input_x, block_shape, crops):
    r"""
    Divides batch dimension with blocks and interleaves these blocks back into spatial dimensions.

    This operation will divide batch dimension N into blocks with block_shape, the output tensor's N dimension
    is the corresponding number of blocks after division. The output tensor's :math:`w_1, ..., w_M` dimension is
    the product of original :math:`w_1, ..., w_M` dimension and block_shape with given amount to crop from dimension,
    respectively.

    If the input shape is :math:`(n, c_1, ... c_k, w_1, ..., w_M)`, the output shape is
    :math:`(n', c_1, ... c_k, w'_1, ..., w'_M)`, where

    .. math::
        \begin{array}{ll} \\
            n' = n//(block\_shape[0]*...*block\_shape[M-1]) \\
            w'_i = w_i*block\_shape[i-1]-crops[i-1][0]-crops[i-1][1]
        \end{array}

    Args:
        input_x (Tensor): The input tensor. It must be greater or equal to 2-D tensor(equal to 4-D tensor on Ascend),
            batch dimension must be divisible by product of `block_shape`.
        block_shape (Union[list(int), tuple(int), int]): The block shape of dividing block with all value greater
            than or equal to 1. If `block_shape` is a tuple or list, the length of `block_shape` is M corresponding
            to the number of spatial dimensions. If `block_shape` is an int, the block size of M dimensions are the
            same, equal to `block_shape`. In this case of Ascend, M must be 2.
        crops (Union[list(int), tuple(int)]): The crops values for spatial dimensions, containing M subtraction list.
            Each contains 2 integer values. All values must be >= 0. crops[i] specifies the crops values for spatial
            dimension i, which corresponds to input dimension i + offset,where offset = N-M, and N is the number of
            input dimensions. It is required that
            :math:`input\_shape[i+offset]*block\_shape[i] > crops[i][0]+crops[i][1]`

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> block_shape = [2, 2]
        >>> crops = [[0, 0], [0, 0]]
        >>> input_x = mindspore.tensor([[[[1]]], [[[2]]], [[[3]]], [[[4]]]], mindspore.float32)
        >>> output = mindspore.ops.batch_to_space_nd(input_x, block_shape, crops)
        >>> print(output)
        [[[[1.  2.]
           [3.  4.]]]]
    """
    if isinstance(block_shape, Tensor):
        return batch_to_space_nd_v2_(input_x, block_shape, crops)
    _batch_to_space_nd = _get_cache_prim(P.BatchToSpaceND)(block_shape, crops)
    return _batch_to_space_nd(input_x)


def matrix_diag(x, k=0, num_rows=-1, num_cols=-1, padding_value=0, align="RIGHT_LEFT"):
    r"""
    Return a tensor with the contents in `x` as k[0]-th to k[1]-th diagonals of a matrix, with everything else padded
    with `padding_value` .

    `num_rows` and `num_cols` are tensors type of int32 with only one value, which is -1, indicating that the innermost
    matrix of the output tensor is a square.

    Args:
        x (Tensor): The input tensor.
        k (Union[int, Tensor], optional): Diagonal offsets. Positive value means superdiagonal, and negative value
            means subdiagonals. When `k` is a pair of integers specifying the low and high ends of a matrix band.
            Default ``0`` .
        num_rows (Union[int, Tensor], optional): The number of rows of the output tensor. Default ``-1`` .
        num_cols (Union[int, Tensor], optional): The number of columns of the output tensor. Default ``-1`` .
        padding_value (Union[int, float, Tensor], optional): The number to fill the area outside the specified
            diagonal band. Default ``0`` .
        align (str, optional): specifies how superdiagonals and subdiagonals should be aligned.
            Supported values ``"RIGHT_LEFT"`` , ``"LEFT_RIGHT"`` , ``"LEFT_LEFT"`` , ``"RIGHT_RIGHT"`` .
            Default ``"RIGHT_LEFT"`` .

            - When set to "RIGHT_LEFT", the alignment of superdiagonals will be towards the right side
              (padding the row on the left), while subdiagonals will be towards the left side
              (padding the row on the right)
            - When set to "LEFT_RIGHT", the alignment of superdiagonals will be towards the left side
              (padding the row on the right), while subdiagonals will be towards the right side
              (padding the row on the left)
            - When set to "LEFT_LEFT", the alignment of  both superdiagonals and subdiagonals will be towards
              the left side(padding the row on the right).
            - When set to "RIGHT_RIGHT", the alignment of both superdiagonals and subdiagonals will be towards
              the right side(padding the row on the left).

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[8., 9., 0.],
        ...                      [1., 2., 3.],
        ...                      [0., 4., 5.]])
        >>> k = mindspore.tensor([-1, 1], mindspore.int32)
        >>> padding_value = mindspore.tensor(11.)
        >>> num_rows = mindspore.tensor(3, mindspore.int32)
        >>> num_cols = mindspore.tensor(3, mindspore.int32)
        >>> output = mindspore.ops.matrix_diag(x, k, num_rows, num_cols, padding_value, align='LEFT_RIGHT')
        >>> print(output)
        [[ 1.  8. 11.]
         [ 4.  2.  9.]
         [11.  5.  3.]]
        >>> print(output.shape)
        (3, 3)
    """
    if isinstance(k, int) and not isinstance(k, bool):
        k = cast_(k, mstype.int32)
    if isinstance(num_rows, int) and not isinstance(num_rows, bool):
        num_rows = cast_(num_rows, mstype.int32)
    if isinstance(num_cols, int) and not isinstance(num_cols, bool):
        num_cols = cast_(num_cols, mstype.int32)
    if isinstance(padding_value, (float, int)) and not isinstance(padding_value, bool):
        padding_value = cast_(padding_value, x.dtype)
    matrix_diag_v3 = _get_cache_prim(MatrixDiagV3)(align)
    return matrix_diag_v3(x, k, num_rows, num_cols, padding_value)


def matrix_diag_part(x, k, padding_value, align="RIGHT_LEFT"):
    r"""
    Return a tensor that retains the values of the specified diagonal while setting all other elements to zero.

    Input `k` and `padding_value` must be const tensor when taking graph mode.

    Args:
        x (Tensor): The input tensor with rank r, where r >= 2.
        k (Union[int, Tensor], optional): Diagonal offsets. Positive value means superdiagonal, and negative value
            means subdiagonals. When `k` is a pair of integers specifying the low and high ends of a matrix band.
        padding_value (Tensor): The number to fill the area outside the specified
            diagonal band.
        align (str, optional): specifies how superdiagonals and subdiagonals should be aligned.
            Supported values ``"RIGHT_LEFT"`` , ``"LEFT_RIGHT"`` , ``"LEFT_LEFT"`` , ``"RIGHT_RIGHT"`` .
            Default ``"RIGHT_LEFT"`` .

            - When set to "RIGHT_LEFT", the alignment of superdiagonals will be towards the right side
              (padding the row on the left), while subdiagonals will be towards the left side
              (padding the row on the right)
            - When set to "LEFT_RIGHT", the alignment of superdiagonals will be towards the left side
              (padding the row on the right), while subdiagonals will be towards the right side
              (padding the row on the left)
            - When set to "LEFT_LEFT", the alignment of  both superdiagonals and subdiagonals will be towards
              the left side(padding the row on the right).
            - When set to "RIGHT_RIGHT", the alignment of both superdiagonals and subdiagonals will be towards
              the right side(padding the row on the left).

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[1., 2., 3., 4.],
        ...                       [5., 6., 7., 8.],
        ...                       [9., 8., 7., 6.]])
        >>> k = mindspore.tensor([1, 3], mindspore.int32)
        >>> output = mindspore.ops.matrix_diag_part(x, k, mindspore.tensor(9.), align='RIGHT_LEFT')
        >>> print(output)
        [[9. 9. 4.]
         [9. 3. 8.]
         [2. 7. 6.]]
        >>> print(output.shape)
        (3, 3)
    """
    matrix_diag_part_v3 = _get_cache_prim(MatrixDiagPartV3)(align)
    return matrix_diag_part_v3(x, k, padding_value)


def matrix_set_diag(x, diagonal, k=0, align="RIGHT_LEFT"):  # pylint: disable=redefined-outer-name
    r"""
    Return a tensor by replacing the elements on the k[0]-th to k[1]-th diagonals of the matrix `x` with the values
    from the input `diagonal` .

    Args:
        x (Tensor): The input tensor with rank r, where r >= 2.
        diagonal (Tensor): A diagonal tensor.
        k (Union[int, Tensor], optional): Diagonal offsets. Positive value means superdiagonal, and negative value
            means subdiagonals. When `k` is a pair of integers specifying the low and high ends of a matrix band.
            Default ``0`` .
        align (str, optional): specifies how superdiagonals and subdiagonals should be aligned.
            Supported values ``"RIGHT_LEFT"`` , ``"LEFT_RIGHT"`` , ``"LEFT_LEFT"`` , ``"RIGHT_RIGHT"`` .
            Default ``"RIGHT_LEFT"`` .

            - When set to "RIGHT_LEFT", the alignment of superdiagonals will be towards the right side
              (padding the row on the left), while subdiagonals will be towards the left side
              (padding the row on the right)
            - When set to "LEFT_RIGHT", the alignment of superdiagonals will be towards the left side
              (padding the row on the right), while subdiagonals will be towards the right side
              (padding the row on the left)
            - When set to "LEFT_LEFT", the alignment of  both superdiagonals and subdiagonals will be towards
              the left side(padding the row on the right).
            - When set to "RIGHT_RIGHT", the alignment of both superdiagonals and subdiagonals will be towards
              the right side(padding the row on the left).

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[7., 7., 7., 7.],
        ...                       [7., 7., 7., 7.],
        ...                       [7., 7., 7., 7.]])
        >>> diagonal = mindspore.tensor([[0., 9., 1.],
        ...                    [6., 5., 8.],
        ...                    [1., 2., 3.],
        ...                    [4., 5., 0.]])
        >>> k = mindspore.tensor(([-1, 2]), mindspore.int32)
        >>> align = 'RIGHT_LEFT'
        >>> output = ops.matrix_set_diag(x, diagonal, k, align)
        >>> print(output)
        [[1. 6. 9. 7.]
         [4. 2. 5. 1.]
         [7. 5. 3. 8.]]
        >>> print(output.shape)
        (3, 4)
    """
    matrix_set_diag_v3_op = _get_cache_prim(MatrixSetDiagV3)(align)
    if isinstance(k, int) and not isinstance(k, bool):
        k = cast_(k, mstype.int32)
    return matrix_set_diag_v3_op(x, diagonal, k)


def meshgrid_ext(*tensors, indexing=None):
    """
    Generates coordinate matrices from given coordinate tensors.

    Given N one-dimensional coordinate tensors, returns a tuple outputs of N N-D
    coordinate tensors for evaluating expressions on an N-D grid.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        tensors (Union(tuple[Tensor], list[Tensor])): In GRAPH_MODE, a tuple of N 1-D Tensor objects and
            the length of input should be greater than 1. In PYNATIVE_MODE, a tuple of N 0-D or 1-D Tensor objects
            and the length of input should be greater than 0. The data type is Number.

    Keyword Args:
        indexing (str, optional): Cartesian ('xy', default) or
            matrix ('ij') indexing of output. Valid options: xy' or ``'ij'``. In the 2-D case with
            inputs of length `M` and `N`, for ``'xy'`` indexing, the shape of outputs is :math:`(N, M)`
            for ``'ij'`` indexing, the shape of outputs is :math:`(M, N)`. In the 3-D
            case with inputs of length `M`, `N` and `P`, for ``'xy'`` indexing, the shape of outputs is
            :math:`(N, M, P)` and for ``'ij'`` indexing, the shape of outputs is :math:`(M, N, P)`.
            Default: ``None`` , which is equivalent to the value ``'ij'`` .

    Returns:
        Tensors, a Tuple of N N-D Tensor objects. The data type is the same with the Inputs.

    Raises:
        TypeError: If `indexing` is not a str or `tensors` is not a tuple.
        ValueError: If `indexing` is neither ``'xy'`` nor ``'ij'``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> x = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
        >>> y = Tensor(np.array([5, 6, 7]).astype(np.int32))
        >>> z = Tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
        >>> output = ops.meshgrid(x, y, z, indexing='xy')
        >>> print(output)
        (Tensor(shape=[3, 4, 5], dtype=Int32, value=
         [[[1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]],
          [[1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]],
          [[1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]]]),
         Tensor(shape=[3, 4, 5], dtype=Int32, value=
         [[[5, 5, 5, 5, 5],
           [5, 5, 5, 5, 5],
           [5, 5, 5, 5, 5],
           [5, 5, 5, 5, 5]],
          [[6, 6, 6, 6, 6],
           [6, 6, 6, 6, 6],
           [6, 6, 6, 6, 6],
           [6, 6, 6, 6, 6]],
          [[7, 7, 7, 7, 7],
           [7, 7, 7, 7, 7],
           [7, 7, 7, 7, 7],
           [7, 7, 7, 7, 7]]]),
         Tensor(shape=[3, 4, 5], dtype=Int32, value=
         [[[8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2]],
          [[8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2]],
          [[8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2]]]))
    """
    if indexing is None:
        indexing = 'ij'
    if len(tensors) == 1 and isinstance(tensors[0], (tuple, list)):
        tensors = tensors[0]
    return meshgrid_impl(tensors, indexing)


def meshgrid(*inputs, indexing='xy'):
    """
    Creates grids of coordinates specified by the 1D inputs。

    .. note::
        - In graph mode, a tuple of N 1-D tensors and N should be greater than 1.
        - In pynative mode, a tuple of N 0-D or 1-D tensors and N should be greater than 0. The data type is Number.
        - In the 2-D case with inputs of length `M` and `N`, the outputs are of shape :math:`(N, M)`
          for ``'xy'`` indexing and :math:`(M, N)` for ``'ij'`` indexing.
        - In the 3-D case with inputs of length `M`, `N` and `P`, outputs are of shape :math:`(N, M, P)`
          for ``'xy'`` indexing and :math:`(M, N, P)` for ``'ij'`` indexing.

    Args:
        inputs (Union[tuple[Tensor], list[Tensor]]): Tuple of tensors or list of tensors.

    Keyword Args:
        indexing (str, optional): Cartesian ('xy', default) or matrix ('ij') indexing of output.
            Valid options ``'xy'`` or ``'ij'``. Default ``'xy'`` .

    Returns:
        Tuple of N N-D tensors

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([1, 2, 3, 4], mindspore.int32)
        >>> y = mindspore.tensor([5, 6, 7], mindspore.int32)
        >>> z = mindspore.tensor([8, 9, 0, 1, 2], mindspore.int32)
        >>> output = mindspore.ops.meshgrid(x, y, z, indexing='xy')
        >>> print(output)
        (Tensor(shape=[3, 4, 5], dtype=Int32, value=
         [[[1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]],
          [[1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]],
          [[1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]]]),
         Tensor(shape=[3, 4, 5], dtype=Int32, value=
         [[[5, 5, 5, 5, 5],
           [5, 5, 5, 5, 5],
           [5, 5, 5, 5, 5],
           [5, 5, 5, 5, 5]],
          [[6, 6, 6, 6, 6],
           [6, 6, 6, 6, 6],
           [6, 6, 6, 6, 6],
           [6, 6, 6, 6, 6]],
          [[7, 7, 7, 7, 7],
           [7, 7, 7, 7, 7],
           [7, 7, 7, 7, 7],
           [7, 7, 7, 7, 7]]]),
         Tensor(shape=[3, 4, 5], dtype=Int32, value=
         [[[8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2]],
          [[8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2]],
          [[8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2],
           [8, 9, 0, 1, 2]]]))
    """
    meshgrid_op = _get_cache_prim(Meshgrid)(indexing)
    return meshgrid_op(inputs)


def affine_grid(theta, size, align_corners=False):
    r"""
    Returns a 2D or 3D flow field (sampling grid) based on `theta`, a batch of affine matrices.

    Args:
        theta (Tensor): The input tensor of flow field whose dtype is float16, float32.
            Input batch of affine matrices with shape :math:`(N, 2, 3)` for 2D grid or :math:`(N, 3, 4)` for 3D grid.
        size (tuple[int]): The target output image size.
            The value of target output with format :math:`(N, C, H, W)` for 2D grid or :math:`(N, C, D, H, W)` for 3D
            grid.
        align_corners (bool, optional): Geometrically, each pixel of input is viewed as a squqre instead of dot.
            If ``True`` , consider extremum -1 and 1 referring to the centers of the pixels rather than pixel corners.
            The default value is ``False`` , extremum -1 and 1 refer to the corners of the pixels, so that sampling is
            irrelevant to resolution of the image. Default: ``False`` .

    Returns:
        Tensor, a tensor whose data type is same as 'theta', and the shape is :math:`(N, H, W, 2)` for 2D grid
        or :math:`(N, D, H, W, 3)` for 3D grid.

    Raises:
        TypeError: If `theta` is not a Tensor or `size` is not a tuple.
        ValueError: If the shape of `theta` is not :math:`(N, 2, 3)` or :math:`(N, 3, 4)`.
        ValueError: If the size of `size` is not 4 or 5.
        ValueError: If the shape of `theta` is :math:`(N, 2, 3)`, the size of `size` is not 4;
                    If the shape of `theta` is :math:`(N, 3, 4)`, the size of `size` is not 5.
        ValueError: If the size[0] is not equal to the shape[0] of theta.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> theta = Tensor([[[0.8, 0.5, 0],[-0.5, 0.8, 0]]], mindspore.float32)
        >>> out_size = (1, 3, 2, 3)
        >>> output = ops.affine_grid(theta, out_size, False)
        >>> print(output)
        [[[[-0.78333336 -0.06666666]
        [-0.25       -0.4       ]
        [ 0.28333336 -0.73333335]]
        [[-0.28333336  0.73333335]
        [ 0.25        0.4       ]
        [ 0.78333336  0.06666666]]]]
    """
    affine_grid_op = AffineGrid(align_corners)
    return affine_grid_op(theta, size)


def unsorted_segment_min(x, segment_ids, num_segments):
    r"""
    Compute the minimum of the input tensor along segments.

    The following figure shows the calculation process of unsorted_segment_min:

    .. image:: UnsortedSegmentMin.png

    .. math::

        \text { output }_i=\text{min}_{j \ldots} \text { data }[j \ldots]

    where :math:`min` over tuples :math:`j...` such that :math:`segment\_ids[j...] == i`.

    Note:
        - If the segment_id i is absent in the segment_ids, then output[i] will be filled with
          the maximum value of the x's type.
        - The `segment_ids` must be non-negative tensor.

    Args:
        x (Tensor): The input tensor.
        segment_ids (Tensor): Indicate the segment to which each element belongs.
        num_segments (Union[int, Tensor], optional): Number of segments, it can be an int or 0-D tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[1, 2, 3], [4, 5, 6], [4, 2, 1]])
        >>> segment_ids = mindspore.tensor([0, 1, 1])
        >>> num_segments = 2
        >>> mindspore.ops.unsorted_segment_min(x, segment_ids, num_segments)
        Tensor(shape=[2, 3], dtype=Int64, value=
        [[1, 2, 3],
         [4, 2, 1]])
    """
    return unsorted_segment_min_(x, segment_ids, num_segments)


def unsorted_segment_max(x, segment_ids, num_segments):
    r"""
    Compute the maximum of the input tensor along segments.

    The following figure shows the calculation process of unsorted_segment_max:

    .. image:: UnsortedSegmentMax.png

    .. math::

        \text { output }_i=\text{max}_{j \ldots} \text { data }[j \ldots]

    where :math:`max` over tuples :math:`j...` such that :math:`segment\_ids[j...] == i`.

    Note:
        - If the segment_id i is absent in the segment_ids, then output[i] will be filled with
          the minimum value of the x's type.
        - The `segment_ids` must be non-negative tensor.

    Args:
        x (Tensor): The input tensor.
        segment_ids (Tensor): Indicate the segment to which each element belongs.
            Set the shape as :math:`(x_1, x_2, ..., x_N)`, where 0 < N <= R.
        num_segments (Union[int, Tensor]): Number of segments, it can be an int or 0-D tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[1, 2, 3], [4, 5, 6], [4, 2, 1]])
        >>> segment_ids = mindspore.tensor([0, 1, 1])
        >>> num_segments = 2
        >>> mindspore.ops.unsorted_segment_max(x, segment_ids, num_segments)
        Tensor(shape=[2, 3], dtype=Int64, value=
        [[1, 2, 3],
         [4, 5, 6]])
    """
    return unsorted_segment_max_(x, segment_ids, num_segments)


def unsorted_segment_prod(x, segment_ids, num_segments):
    r"""
    Compute the product of the input tensor along segments.

    The following figure shows the calculation process of unsorted_segment_prod:

    .. image:: UnsortedSegmentProd.png

    Note:
        - If the segment_id i is absent in the segment_ids, then output[i] will be filled with 1.
        - The `segment_ids` must be non-negative tensor.

    Args:
        x (Tensor): The input tensor.
        segment_ids (Tensor): Indicate the segment to which each element belongs.
        num_segments (Union[int, Tensor], optional): Number of segments, it can be an int or 0-D tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[1, 2, 3], [4, 5, 6], [4, 2, 1]])
        >>> segment_ids = mindspore.tensor([0, 1, 0])
        >>> num_segments = 2
        >>> mindspore.ops.unsorted_segment_prod(x, segment_ids, num_segments)
        Tensor(shape=[2, 3], dtype=Int64, value=
        [[4, 4, 3],
         [4, 5, 6]])
    """
    return unsorted_segment_prod_(x, segment_ids, num_segments)


def index_fill(x, axis, index, value):
    """
    Fills the elements of the input `x` with `value` along the given axis and indices.

    Args:
        x (Tensor): The input tensor.
        axis (Union[int, Tensor]): The specified axis.
        index (Tensor): The specified indices.
        value (Union[bool, int, float, Tensor]): Value to fill the returned tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mindspore.float32)
        >>> index = mindspore.tensor([0, 2], mindspore.int32)
        >>> value = mindspore.tensor(-2.0, mindspore.float32)
        >>> y = mindspore.ops.index_fill(x, 1, index, value)
        >>> print(y)
        [[-2. 2. -2.]
         [-2. 5. -2.]
         [-2. 8. -2.]]
    """
    if isinstance(axis, int) and not isinstance(axis, bool):
        axis = cast_(axis, mstype.int32)
    if isinstance(value, (bool, float, int)):
        value = cast_(value, x.dtype)
    return index_fill_(x, axis, index, value)


def index_fill_ext(input, dim, index, value):
    """
    Fills the elements under the `dim` dimension of the input Tensor `input` with the input `value`
    by selecting the indices in the order given in `index`.

    Args:
        input (Tensor): Input Tensor.  The supported data type is Number or Bool.
        dim (int): Dimension along which to fill the input Tensor. Only supports
            an int number, which data type is int32 or int64.
        index (Tensor): Indices of the input Tensor to fill in. The dtype must be int32 or int64.
        value (Union[bool, int, float, Tensor]): Value to fill the returned Tensor. If `value` is
            a Tensor, it must be a 0-dimensional Tensor and has the same dtype as `input`. Otherwise,
            the `value` will be a value with the same data type as `input`.

    Returns:
        Tensor, has the same dtype and shape as input Tensor.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `dim` is neither int number nor Tensor.
        TypeError: When `dim` is a Tensor, its dtype is not int32 or int64.
        TypeError: If `index` is not a Tensor.
        TypeError: If dtype of `index` is not int32.
        TypeError: If `value` is not a bool, int, float, or Tensor.
        TypeError: When `value` is a Tensor, the dtype of `input` and `value` are not the same.
        ValueError: If `dim` is a Tensor and its rank is not equal to 0.
        ValueError: If the rank of `index` is greater than 1D.
        ValueError: When `value` is a Tensor and its rank is not equal to 0.
        RuntimeError: If the value of `dim` is out the range of `[-x.ndim, x.ndim - 1]`.
        RuntimeError: If the values of `index` are out the range of `[-x.shape[dim], x.shape[dim]-1]`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> input = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32))
        >>> index = Tensor([0, 2], mindspore.int32)
        >>> value = Tensor(-2.0, mindspore.float32)
        >>> y = ops.index_fill_ext(x, 1, index, value)
        >>> print(y)
        [[-2. 2. -2.]
         [-2. 5. -2.]
         [-2. 8. -2.]]
    """
    if isinstance(value, Tensor):
        return index_fill_tensor(input, dim, index, value)
    return index_fill_scalar(input, dim, index, value)


@constexpr
def _check_check_axis_in_range(axis, ndim):
    """Checks axes are with the bounds of ndim"""
    axis = validator.check_axis_in_range(axis, ndim)
    return axis


def index_select(input, axis, index):
    """
    Select the input tensor according to the specified axis and index and return a new tensor.

    .. note::
        - The value of `index` must be in the range of `[0, input.shape[axis])`, the result is undefined out of range.
        - The returned tensor has the same number of dimensions as the input tensor.The `axis` dimension has the same
          size as the length of `index` , other dimensions have the same size as the input tensor.

    Args:
        input (Tensor): The input tensor.
        axis (int): The specified axis.
        index (Tensor): The specified indices, a 1-D tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor(mindspore.ops.arange(0, 16).reshape(2, 2, 4), mindspore.float32)
        >>> print(input)
        [[[ 0.  1.  2.  3.]
          [ 4.  5.  6.  7.]]
         [[ 8.  9. 10. 11.]
          [12. 13. 14. 15.]]]
        >>> index = mindspore.tensor([0,], mindspore.int32)
        >>> y = mindspore.ops.index_select(input, 1, index)
        >>> print(y)
        [[[ 0.  1.  2.  3.]]
         [[ 8.  9. 10. 11.]]]
    """
    if not (isinstance(input, Tensor) and isinstance(index, Tensor)):
        raise TypeError(
            "For 'index_select', `input` and `index` must be all tensors.")
    if index.ndim != 1:
        raise ValueError(
            f"For 'index_select', the dimension of `index` must be 1, but got {index.ndim}")
    axis = _check_check_axis_in_range(axis, input.ndim)
    return gather_(input, index, axis)


def population_count(input_x):
    r"""
    Calculate the number of 1 bits in the binary representation of each element in the input tensor.

    Args:
        input_x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.tensor([0, 1, 3], mindspore.int16)
        >>> output = mindspore.ops.population_count(input_x)
        >>> print(output)
        [0 1 2]
    """
    return population_count_(input_x)


##############################
# Type Conversion Functions.
##############################


def is_tensor(obj):
    r"""
    Check whether the input object is :class:`mindspore.Tensor` .

    Args:
        obj (Object): input object.

    Returns:
        Bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> a = mindspore.tensor([1.9, 2.2, 3.1])
        >>> mindspore.ops.is_tensor(a)
        True
    """
    return isinstance(obj, Tensor)


def is_nonzero(input):
    """
    Determine whether the input Tensor contains ``0`` or ``False``. The input can only be a single element.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Bool

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x1 = mindspore.tensor([[[False]]])
        >>> x2 = mindspore.tensor([[3.5]])
        >>> out1 = mindspore.ops.is_nonzero(x1)
        >>> print(out1)
        False
        >>> out2 = mindspore.ops.is_nonzero(x2)
        >>> print(out2)
        True
    """
    if not isinstance(input, Tensor):
        raise TypeError(
            f'For is_nonzero, the input must be a Tensor, but got {type(input)}.')
    if input.numel() != 1:
        raise ValueError(
            f"For is_nonzero, the numel of input must be 1, but got {input.numel()}.")
    out = ops.squeeze(input)
    return bool(out)


def tensor_scatter_mul(input_x, indices, updates):
    r"""
    Return a new tensor by performing a multiplication update on `input_x` at the specified indices with the given
    update values.

    .. math::
        output\left [indices  \right ] = input\_x\times  updates

    Note:
        - If some values of the `indices` are out of bound, instead of raising an index error,
          the corresponding `updates` will not be updated to `input_x`.

    Args:
        input_x (Tensor): The input tensor.
        indices (Tensor): The specified indices. The rank must be at least 2.
        updates (Tensor): The update values.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.tensor([[1, 2, 3], [4, 5, 6]])
        >>> indices = mindspore.tensor([[0, 0], [1, 1]])
        >>> updates = mindspore.tensor([5, 5])
        >>> mindspore.ops.tensor_scatter_mul(input_x, indices, updates)
        Tensor(shape=[2, 3], dtype=Int64, value=
        [[ 5,  2,  3],
         [ 4, 25,  6]])
    """
    return tensor_scatter_mul_(input_x, indices, updates)


def tensor_scatter_div(input_x, indices, updates):
    r"""
    Return a new tensor which `input_x` is divided by the values from `updates` indicated by `indices` .

    .. math::
        output\left [indices  \right ] = input\_x \div updates

    Note:
        - On GPU, if some values of the `indices` are out of bound, instead of raising an index error,
          the corresponding `updates` will not be updated to self tensor.
        - On CPU, if some values of the `indices` are out of bound, raising an index error.
        - On Ascend, out of bound checking is not supported, if some values of the `indices` are out of bound,
          unknown errors may be caused.
        - The operator can't handle division by 0 exceptions, so the user needs to make sure
          there is no 0 value in `updates`.

    Args:
        input_x (Tensor): The input tensor.
        indices (Tensor): The specified indices.
        updates (Tensor): The update values.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.tensor([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]], mindspore.float32)
        >>> indices = mindspore.tensor([[0, 0], [0, 0]], mindspore.int32)
        >>> updates = mindspore.tensor([1.0, 2.0], mindspore.float32)
        >>> output = mindspore.ops.tensor_scatter_div(input_x, indices, updates)
        >>> print(output)
        [[-0.05  0.3  3.6  ]
         [ 0.4   0.5  -3.2 ]]
    """
    return tensor_scatter_div_(input_x, indices, updates)


def scalar_to_array(input_x):
    """
    The  interface is deprecated. Please use the :func:`mindspore.ops.scalar_to_tensor` instead.
    """
    return P.ScalarToArray()(input_x)


def scalar_to_tensor(input_x, dtype=mstype.float32):
    """
    Converts a scalar to a tensor with the specified dtype.

    Args:
        input_x (Union[bool, int, float]): The input scalar. Only constant value is allowed.
        dtype (mindspore.dtype): The dtype of returned tensor. Only constant value is allowed.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> data = 1
        >>> output = mindspore.ops.scalar_to_tensor(data, mindspore.float32)
        >>> print(output)
        1.0
    """
    return scalar_to_tensor_(input_x, dtype)


def tuple_to_array(input_x):
    """
    Converts a tuple to a tensor.

    .. note::
        If the type of the first number in the tuple is integer, the data type of the output tensor is int.
        Otherwise, the data type of the output tensor is float.

    Args:
        input_x (tuple): A tuple of numbers. Only constant value is allowed.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = (1, 2, 3)
        >>> output = mindspore.ops.tuple_to_array(input_x)
        >>> print(type(output))
        <class 'mindspore.common.tensor.Tensor'>
        >>> print(output)
        [1 2 3]
    """
    if isinstance(input_x[0], int):
        dtype = mstype.int32
    else:
        dtype = mstype.float32
    return tuple_to_tensor_(input_x, dtype)


def diagflat(input, offset=0):
    r"""
    If `input` is a vector (1-D tensor), then returns a 2-D square tensor with the elements of `input` as the diagonal,
    If `input` is a tensor with more than one dimension, then returns a 2-D tensor with diagonal elements equal to a
    flattened `input`.

    Args:
        input (Tensor): Input Tensor.
        offset (int, optional): Diagonal offset. Default ``0`` .

            - When `offset` is a positive integer, shift the diagonal upward.
            - When `offset` is a negative integer, shift the diagonal downward.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> mindspore.ops.diagflat(mindspore.tensor([1, 2, 3]))
        Tensor(shape=[3, 3], dtype=Int64, value=
        [[1, 0, 0],
         [0, 2, 0],
         [0, 0, 3]])
        >>> mindspore.ops.diagflat(mindspore.tensor([1, 2, 3]), 1)
        Tensor(shape=[4, 4], dtype=Int64, value=
        [[0, 1, 0, 0],
         [0, 0, 2, 0],
         [0, 0, 0, 3],
         [0, 0, 0, 0]])
        >>> mindspore.ops.diagflat(mindspore.tensor([[1, 2], [3, 4]]))
        Tensor(shape=[4, 4], dtype=Int64, value=
        [[1, 0, 0, 0],
         [0, 2, 0, 0],
         [0, 0, 3, 0],
         [0, 0, 0, 4]])
    """
    if not isinstance(input, Tensor):
        raise TypeError(
            f"For diagflat, the input x must be tensor, but got {type(input)}")
    if not isinstance(offset, int):
        raise TypeError(
            f"For diagflat, the offset must be int, but got {type(offset)}")
    offset_abs = abs(offset)
    if input.size == 0:
        return zeros((offset_abs, offset_abs), input.dtype)
    input = input.ravel()
    res = diag(input)
    if offset != 0:
        pad_y = zeros((input.size + offset_abs, offset_abs), input.dtype)
        pad_x = zeros((offset_abs, input.size), input.dtype)
        if offset < 0:
            res = cat((pad_x, res), axis=0)
            res = cat((res, pad_y), axis=1)
        else:
            res = cat((res, pad_x), axis=0)
            res = cat((pad_y, res), axis=1)
    return res


def col2im(input_x, output_size, kernel_size, dilation, padding_value, stride):
    """
    Combines an array of sliding local blocks into a large containing tensor.

    Args:
        input_x (Tensor): 4D tensor with data type float16 or float32.
        output_size (Tensor): 1D tensor with 2 elements of data type int.
        kernel_size (Union[int, tuple[int], list[int]]): The size of the kernel, should be two int
            for height and width. If type is int, it means that height equal with width. Must be specified.
        dilation (Union[int, tuple[int], list[int]]): The size of the dilation, should be two int
            for height and width. If type is int, it means that height equal with width.
        padding_value (Union[int, tuple[int], list[int]]): The size of the padding, should be two int
            for height and width. If type is int, it means that height equal with width.
        stride (Union[int, tuple[int], list[int]]): The size of the stride, should be two int
            for height and width. If type is int, it means that height equal with width.

    Returns:
        A 4D Tensor, with same type as `input_x`.

    Raises:
        TypeError: If :attr:`kernel_size`, `dilation`, `padding_value`, `stride` data type is not in
            Union[int, tuple[int], list[int]].
        ValueError: If :attr:`kernel_size`, `dilation`, `padding_value`, `stride` value is not
            greater than zero or elements number more than 2.
        ValueError: If :attr:`padding_value` value is less than zero or elements number more than 2.
        ValueError: If input_x.shape[2] != kernel_size[0] * kernel_size[1].
        ValueError: If input_x.shape[3] does not match the calculated number of sliding blocks.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> from mindspore import dtype as mstype
        >>> x = Tensor(input_data=np.random.rand(16, 16, 4, 25), dtype=mstype.float32)
        >>> output_size = Tensor(input_data=[8, 8], dtype=mstype.int32)
        >>> output = ops.col2im(x, output_size, [2, 2], [2, 2], [2, 2], [2, 2])
        >>> print(output.shape)
        (16, 16, 8, 8)
    """
    c2i = _get_cache_prim(Col2Im)(kernel_size, dilation, padding_value, stride)
    return c2i(input_x, output_size)


def _split_int(x, split_size_or_sections, axis):
    """
    Splits the input tensor `x` into multiple sub-tensors along the axis according to the given `split_size_or_sections`
    with int type.
    """
    arr_shape = x.shape
    length_along_dim = arr_shape[axis]
    if length_along_dim == 0:
        res = _get_cache_prim(P.Split)(axis)(x)
    elif split_size_or_sections > length_along_dim:
        res = _get_cache_prim(P.Split)(axis, 1)(x)
    elif length_along_dim % split_size_or_sections == 0:
        sections = length_along_dim // split_size_or_sections
        res = _get_cache_prim(P.Split)(axis, sections)(x)
    else:
        num_sections = length_along_dim // split_size_or_sections
        length1 = num_sections * split_size_or_sections
        length2 = length_along_dim - length1
        start1 = _list_comprehensions(rank_(x), 0, True)
        size1 = _tuple_setitem(arr_shape, axis, length1)
        start2 = _tuple_setitem(start1, axis, length1)
        size2 = _tuple_setitem(arr_shape, axis, length2)
        res = _get_cache_prim(P.Split)(axis, num_sections)(tensor_slice(x, start1, size1)) + \
            _get_cache_prim(P.Split)(axis, 1)(tensor_slice(x, start2, size2))
    return res


def _split_sub_tensors(x, split_size_or_sections, axis):
    """
    Splits the input tensor `x` into multiple sub-tensors along the axis according to the given `split_size_or_sections`
    with type of tuple or list.
    """
    new_indices = [0]
    for i, split_size in enumerate(split_size_or_sections):
        new_indices.append(new_indices[i] + split_size)
    new_indices = new_indices[1:]
    sub_tensors = []
    strides = _list_comprehensions(x.ndim, 1, True)
    begin = _list_comprehensions(x.ndim, 0)
    end = _list_comprehensions(x.shape)
    for i in ms_arrange(len(new_indices)):
        idx = new_indices[i]
        begin[axis] = 0 if i == 0 else new_indices[i - 1]
        end[axis] = idx
        sliced_tensor = strided_slice(x, tuple(begin), tuple(end), strides)
        sub_tensors.append(sliced_tensor)
    return sub_tensors


def split(tensor, split_size_or_sections, axis=0):
    """
    Split the tensor into chunks along the given axis.

    Args:
        tensor (Tensor): The input tensor.
        split_size_or_sections (Union[int, tuple(int), list(int)]): The size of chunks after splitting.
        axis (int, optional): The axis along which to split. Default ``0`` .

    .. note::
        - If `split_size_or_sections` is an int type, the input tensor will be evenly divided into chunks of
          size `split_size_or_sections` . The last chunk will have a size equal to the remainder if tensor.shape[axis]
          is not divisible by `split_size_or_sections` .
        - If `split_size_or_sections` is a tuple or list, `tensor` will be split along `axis` into
          len( `split_size_or_sections` ) chunks with sizes specified by `split_size_or_sections` .

    Returns:
        Tuple of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case1: `split_size_or_sections` is an int type
        >>> input_x = mindspore.ops.arange(10).astype("float32")
        >>> output = mindspore.ops.split(tensor=input_x, split_size_or_sections=3)
        >>> print(output)
        (Tensor(shape=[3], dtype=Float32, value=[0.00000000e+00, 1.00000000e+00, 2.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value=[3.00000000e+00, 4.00000000e+00, 5.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value=[6.00000000e+00, 7.00000000e+00, 8.00000000e+00]),
         Tensor(shape=[1], dtype=Float32, value=[9.00000000e+00]))
        >>> # case2: `split_size_or_sections` is a list type
        >>> output = mindspore.ops.split(tensor=input_x, split_size_or_sections=[3, 3, 4])
        >>> print(output)
        (Tensor(shape=[3], dtype=Float32, value=[0.00000000e+00, 1.00000000e+00, 2.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value=[3.00000000e+00, 4.00000000e+00, 5.00000000e+00]),
         Tensor(shape=[4], dtype=Float32, value=[6.00000000e+00, 7.00000000e+00, 8.00000000e+00, 9.00000000e+00]))
    """
    if not isinstance(tensor, Tensor):
        raise TypeError(f'expect `tensor` is a Tensor, but got {type(tensor)}')
    if type(axis) is not int:
        raise TypeError(
            f"Type of Argument `axis` should be integer but got {type(axis)}")
    arr_axis = _canonicalize_axis(axis, tensor.ndim)

    if type(split_size_or_sections) is int:
        if split_size_or_sections > 0:
            res = _split_int(tensor, split_size_or_sections, arr_axis)
        else:
            raise ValueError(f"For split, the value of 'split_size_or_sections' must be more than zero, "
                             f"but got {split_size_or_sections}.")
    elif isinstance(split_size_or_sections, (list, tuple)):
        for item in split_size_or_sections:
            if type(item) is not int:
                raise TypeError(
                    f"Each element in 'split_size_or_sections' should be integer, but got {type(item)}.")
            if item < 0:
                raise TypeError(f"Each element in 'split_size_or_sections' should be non-negative, "
                                f"but got {split_size_or_sections}.")

        if sum(split_size_or_sections) != tensor.shape[arr_axis]:
            raise ValueError(f"The sum of 'split_size_or_sections' should be equal to {tensor.shape[arr_axis]}, "
                             f"but got {sum(split_size_or_sections)}.")
        res = _split_sub_tensors(tensor, split_size_or_sections, arr_axis)
    else:
        raise TypeError(f"Type of Argument `split_size_or_sections` should be integer, tuple(int) or list(int), "
                        f"but got {type(split_size_or_sections)}")
    return tuple(res)


def split_ext(tensor, split_size, dim=0):
    """
    Splits the Tensor into chunks along the given dim.

    Args:
        tensor (Tensor): A Tensor to be divided.
        split_size (Union[int, tuple(int), list(int)]):
            If `split_size` is an int type, `tensor` will be split into equally sized chunks,
            each chunk with size `split_size`. Last chunk will be smaller than `split_size`
            if `tensor.shape[dim]` is not divisible by `split_size`.
            If `split_size` is a list type, then `tensor` will be split into len(split_size)
            chunks with sizes `split_size` along the given `dim`.
        dim (int): The dim along which to split. Default: ``0`` .

    Returns:
        A tuple of sub-tensors.

    Raises:
        TypeError: If argument `tensor` is not Tensor.
        TypeError: If argument `dim` is not int.
        ValueError: If argument `dim` is out of range of :[-tensor.ndim, tensor.ndim).
        TypeError: If each element in `split_size` is not integer.
        TypeError: If argument `split_size` is not int, tuple(int) or list(int).
        ValueError: The sum of `split_size` is not equal to x.shape[dim].

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import ops, Tensor
        >>> input_x = np.arange(9).astype("float32")
        >>> output = ops.split_ext(Tensor(input_x), 3)
        >>> print(output)
        (Tensor(shape=[3], dtype=Float32, value= [ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value= [ 3.00000000e+00,  4.00000000e+00,  5.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value= [ 6.00000000e+00,  7.00000000e+00,  8.00000000e+00]))
    """
    if isinstance(split_size, int):
        res = split_tensor_op(tensor, split_size, dim)
    elif isinstance(split_size, (list, tuple)):
        res = split_with_size_op(tensor, split_size, dim)
    else:
        raise TypeError(f"Type of Argument `split_size` should be integer, tuple(int) or list(int), "
                        f"but got {type(split_size)}")
    return res


def split_view(tensor, split_size_or_sections, dim=0):
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
        dim (int, optional): The dim along which to split. Default: ``0`` .

    Returns:
        A tuple of sub-tensors.

    Raises:
        TypeError: If argument `tensor` is not Tensor.
        TypeError: If argument `dim` is not int.
        ValueError: If argument `dim` is out of range of [-tensor.ndim, tensor.ndim).
        TypeError: If each element in `split_size_or_sections` is not integer.
        TypeError: If argument `split_size_or_sections` is not int, tuple(int) or list(int).
        ValueError: The sum of `split_size_or_sections` is not equal to tensor.shape[dim].

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> input_x = np.arange(9).astype("float32")
        >>> output = mindspore.ops.function.array_func.split_view(Tensor(input_x), 3)
        >>> print(output)
        (Tensor(shape=[3], dtype=Float32, value= [ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value= [ 3.00000000e+00,  4.00000000e+00,  5.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value= [ 6.00000000e+00,  7.00000000e+00,  8.00000000e+00]))
    """
    if isinstance(split_size_or_sections, int):
        res = ops.auto_generate.split_tensor_view_op(tensor, split_size_or_sections, dim)
    elif isinstance(split_size_or_sections, (list, tuple)):
        res = ops.auto_generate.split_with_size_view_op(tensor, split_size_or_sections, dim)
    else:
        raise TypeError(f"Type of Argument `split_size_or_sections` should be integer, tuple(int) or list(int), "
                        f"but got {type(split_size_or_sections)}")
    return res


def tril(input, diagonal=0):  # pylint: disable=redefined-outer-name
    """
    Zero the input tensor above the diagonal specified.

    Args:
        input (Tensor): The input tensor. The rank must be at least 2.
        diagonal (int, optional): The diagonal specified of 2-D tensor. Default ``0`` represents the main diagonal.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[ 1,  2,  3,  4],
        ...                           [ 5,  6,  7,  8],
        ...                           [10, 11, 12, 13],
        ...                           [14, 15, 16, 17]])
        >>> mindspore.ops.tril(input)
        Tensor(shape=[4, 4], dtype=Int64, value=
        [[ 1,  0,  0,  0],
         [ 5,  6,  0,  0],
         [10, 11, 12,  0],
         [14, 15, 16, 17]])
        >>> mindspore.ops.tril(input, 1)
        Tensor(shape=[4, 4], dtype=Int64, value=
        [[ 1,  2,  0,  0],
         [ 5,  6,  7,  0],
         [10, 11, 12, 13],
         [14, 15, 16, 17]])
        >>> mindspore.ops.tril(input, -1)
        Tensor(shape=[4, 4], dtype=Int64, value=
        [[ 0,  0,  0,  0],
         [ 5,  0,  0,  0],
         [10, 11,  0,  0],
         [14, 15, 16,  0]])
        >>> input = mindspore.tensor([[[ 1,  2,  3],
        ...                            [ 5,  6,  7],
        ...                            [10, 11, 12]],
        ...                           [[ 1,  2,  3],
        ...                            [ 5,  6,  7],
        ...                            [10, 11, 12]]])
        >>> mindspore.ops.tril(input)
        Tensor(shape=[2, 3, 3], dtype=Int64, value=
        [[[ 1,  0,  0],
          [ 5,  6,  0],
          [10, 11, 12]],
         [[ 1,  0,  0],
          [ 5,  6,  0],
          [10, 11, 12]]])
    """
    tril_ = _get_cache_prim(Tril)(diagonal)
    return tril_(input)


def tril_ext(input, diagonal=0):
    """
    Zero the input tensor above the diagonal specified.

    Args:
        input (Tensor): The input tensor. The rank must be at least 2.
        diagonal (int, optional): The diagonal specified of 2-D tensor. Default ``0`` represents the main diagonal.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[ 1,  2,  3,  4],
        ...                           [ 5,  6,  7,  8],
        ...                           [10, 11, 12, 13],
        ...                           [14, 15, 16, 17]])
        >>> mindspore.ops.function.array_func.tril_ext(input)
        Tensor(shape=[4, 4], dtype=Int64, value=
        [[ 1,  0,  0,  0],
         [ 5,  6,  0,  0],
         [10, 11, 12,  0],
         [14, 15, 16, 17]])
        >>> mindspore.ops.function.array_func.tril_ext(input, 1)
        Tensor(shape=[4, 4], dtype=Int64, value=
        [[ 1,  2,  0,  0],
         [ 5,  6,  7,  0],
         [10, 11, 12, 13],
         [14, 15, 16, 17]])
        >>> mindspore.ops.function.array_func.tril_ext(input, -1)
        Tensor(shape=[4, 4], dtype=Int64, value=
        [[ 0,  0,  0,  0],
         [ 5,  0,  0,  0],
         [10, 11,  0,  0],
         [14, 15, 16,  0]])
        >>> input = mindspore.tensor([[[ 1,  2,  3],
        ...                            [ 5,  6,  7],
        ...                            [10, 11, 12]],
        ...                           [[ 1,  2,  3],
        ...                            [ 5,  6,  7],
        ...                            [10, 11, 12]]])
        >>> mindspore.ops.function.array_func.tril_ext(input)
        Tensor(shape=[2, 3, 3], dtype=Int64, value=
        [[[ 1,  0,  0],
          [ 5,  6,  0],
          [10, 11, 12]],
         [[ 1,  0,  0],
          [ 5,  6,  0],
          [10, 11, 12]]])
    """
    return tril_ext_op(input, diagonal)


@_primexpr
def _canonicalize_axis(axis, ndim):
    """
    Check axes are within the number of dimensions of tensor x and normalize the negative axes.

    Args:
        axis (Union[int, tuple(int), list(int)]): Axes of the tensor.
        ndim (int): The number of dimensions of the tensor.

    Return:
        Axis (Union[int, tuple(int)]). If input is integer, return integer, else tuple.
    """
    if isinstance(axis, int):
        axis = [axis]
    for ax in axis:
        if not isinstance(ax, int):
            raise TypeError(f'axis should be integers, not {type(ax)}')
        if not -ndim <= ax < ndim:
            raise ValueError(
                f'axis {ax} is out of bounds for array of dimension {ndim}')

    def canonicalizer(ax):
        return ax + ndim if ax < 0 else ax

    axis = tuple(canonicalizer(ax) for ax in axis)
    if all(axis.count(el) <= 1 for el in axis):
        return tuple(sorted(axis)) if len(axis) > 1 else axis[0]
    raise ValueError(f"duplicate axis in {axis}.")


@_primexpr
def _list_comprehensions(obj, item=None, return_tuple=False):
    """
    Generates a new list or tuple by list comprehension.

    Args:
        obj (Union[int, list, tuple]):
            If integer, it will be the length of the returned tuple/list.
        item: The value to be filled. Default: ``None`` .
            If ``None`` , the values in the new list/tuple are the same as obj
            or range(obj) when obj is integer.
        return_tuple(bool): If ``true`` , returns tuple, else returns list.

    Returns:
        List or tuple.
    """
    lst = obj
    if isinstance(obj, int):
        lst = []
        for i in ms_arrange(obj):
            lst.append(i)
    if item is None:
        res = list(lst)
    else:
        res = [item for _ in lst]
    if return_tuple:
        return tuple(res)
    return res


@_primexpr
def _tuple_setitem(tup, idx, value):
    """
    Returns a tuple with specified `idx` set to `value`.
    """
    tup = list(tup)
    tup[idx] = value
    return tuple(tup)


def _tensor_split_sub_tensors(x, indices_or_sections, axis):
    """
    Splits the input tensor `x` into multiple sub-tensors along the axis according to the given `indices_or_sections`
    with type of tuple or list.
    """
    length_along_dim = x.shape[axis]
    indices_or_sections = tuple(indices_or_sections)
    indices_or_sections += (length_along_dim,)

    sub_tensors = []
    strides = _list_comprehensions(x.ndim, 1, True)
    begin = _list_comprehensions(x.ndim, 0)
    end = _list_comprehensions(x.shape)
    for i in ms_arrange(len(indices_or_sections)):
        idx = indices_or_sections[i]
        begin[axis] = 0 if i == 0 else indices_or_sections[i - 1]
        end[axis] = idx
        if begin[axis] == end[axis]:
            empty_shape = x.shape[0:axis] + (0,) + x.shape[axis + 1:]
            sliced_tensor = ms.Tensor(shape=empty_shape, dtype=x.dtype)
        else:
            sliced_tensor = strided_slice(x, tuple(begin), tuple(end), strides)
        sub_tensors.append(sliced_tensor)
    return tuple(sub_tensors)


def _tensor_split_sub_int(x, indices_or_sections, axis):
    """
    Splits the input tensor `x` into multiple sub-tensors along the axis according to the given `indices_or_sections`
    with type if int.
    """
    arr_shape = x.shape
    length_along_dim = arr_shape[axis]
    if length_along_dim == 0:
        res = _get_cache_prim(P.Split)(axis)(x)
    elif indices_or_sections > length_along_dim:
        res = _get_cache_prim(P.Split)(axis, length_along_dim)(x)
        indices_or_sections_n = [length_along_dim]
        res2 = _tensor_split_sub_tensors(x, indices_or_sections_n, axis)
        for _ in np.arange(length_along_dim, indices_or_sections):
            res += tuple(res2)[1:]
    elif length_along_dim % indices_or_sections == 0:
        res = _get_cache_prim(P.Split)(axis, indices_or_sections)(x)
    else:
        num_long_tensor = length_along_dim % indices_or_sections
        num_short_tensor = indices_or_sections - num_long_tensor
        length1 = num_long_tensor * \
            (length_along_dim // indices_or_sections + 1)
        length2 = length_along_dim - length1
        start1 = _list_comprehensions(rank_(x), 0, True)
        size1 = _tuple_setitem(arr_shape, axis, length1)
        start2 = _tuple_setitem(start1, axis, length1)
        size2 = _tuple_setitem(arr_shape, axis, length2)
        res = _get_cache_prim(P.Split)(axis, num_long_tensor)(tensor_slice(x, start1, size1)) + \
            _get_cache_prim(P.Split)(axis, num_short_tensor)(
                tensor_slice(x, start2, size2))
    return res


def tensor_split(input, indices_or_sections, axis=0):
    r"""
    Split the input tensor into multiple subtensors according to the specified indices or chunks.

    Args:
        input (Tensor): The input tensor.
        indices_or_sections (Union[int, tuple(int), list(int)]): The specified indices or chunks.

            - If it is an integer, input tensor will be split into `indices_or_sections` sections.

              - If :math:`input.shape[axis]` can be divisible by `indices_or_sections`, sub-sections will have equal
                size :math:`input.shape[axis] / n` .
              - If :math:`input.shape[axis]` can not be divisible by `indices_or_sections`, the first
                :math:`input.shape[axis] \bmod n` sections will have size :math:`input.shape[axis] // n + 1` , and the
                rest will have size :math:`input.shape[axis] // n` .
            - If it is a tuple(int) or list(int) type, it represts indices and the input tensor will be split at the
              indices.

        axis (int, optional): The axis along which to split. Default ``0`` .

    Returns:
        Tuple of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        >>> mindspore.ops.tensor_split(input, 3)
        (Tensor(shape=[3], dtype=Int64, value= [0, 1, 2]),
         Tensor(shape=[3], dtype=Int64, value= [3, 4, 5]),
         Tensor(shape=[2], dtype=Int64, value= [6, 7]))
        >>> input = mindspore.tensor([0, 1, 2, 3, 4, 5, 6])
        >>> mindspore.ops.tensor_split(input, 3)
        (Tensor(shape=[3], dtype=Int64, value= [0, 1, 2]),
         Tensor(shape=[2], dtype=Int64, value= [3, 4]),
         Tensor(shape=[2], dtype=Int64, value= [5, 6]))
        >>> mindspore.ops.tensor_split(input, (1, 6))
        (Tensor(shape=[1], dtype=Int64, value= [0]),
         Tensor(shape=[5], dtype=Int64, value= [1, 2, 3, 4, 5]),
         Tensor(shape=[1], dtype=Int64, value= [6]))
        >>> input = mindspore.tensor([[ 0,  1,  2,  3,  4,  5,  6],
        ...                           [ 7,  8,  9, 10, 11, 12, 13]])
        >>> mindspore.ops.tensor_split(input, 3, axis=1)
        (Tensor(shape=[2, 3], dtype=Int64, value=
         [[0, 1, 2],
          [7, 8, 9]]),
         Tensor(shape=[2, 2], dtype=Int64, value=
         [[ 3,  4],
          [10, 11]]),
         Tensor(shape=[2, 2], dtype=Int64, value=
         [[ 5,  6],
          [12, 13]]))
        >>> mindspore.ops.tensor_split(input, (1, 6), axis=1)
        (Tensor(shape=[2, 1], dtype=Int64, value=
         [[0],
          [7]]),
         Tensor(shape=[2, 5], dtype=Int64, value=
         [[ 1,  2,  3,  4,  5],
          [ 8,  9, 10, 11, 12]]),
         Tensor(shape=[2, 1], dtype=Int64, value=
         [[ 6],
          [13]]))
    """
    if not isinstance(input, Tensor):
        raise TypeError(f'expect `x` is a Tensor, but got {type(input)}')

    if type(axis) is not int:
        raise TypeError(
            f"Type of Argument `axis` should be integer but got {type(axis)}")
    handle_axis = _canonicalize_axis(axis, input.ndim)
    if type(indices_or_sections) is int:
        if indices_or_sections > 0:
            res = _tensor_split_sub_int(
                input, indices_or_sections, handle_axis)
        else:
            raise ValueError(f"For tensor_split, the value of 'indices_or_sections' must be more than zero "
                             f"but got {indices_or_sections}")
    elif isinstance(indices_or_sections, (list, tuple)):
        for item in indices_or_sections:
            if type(item) is not int:
                raise TypeError(
                    f"Each element in 'indices_or_sections' should be integer, but got {type(item)}.")
        res = _tensor_split_sub_tensors(
            input, indices_or_sections, handle_axis)
    else:
        raise TypeError(f"Type of Argument `indices_or_sections` should be integer, tuple(int) or list(int), "
                        f"but got {type(indices_or_sections)}")

    return res


def vsplit(input, indices_or_sections):
    """
    Split the input tensor with two or more dimensions, into multiple sub-tensors vertically
    according to `indices_or_sections`.

    It is equivalent to `ops.tensor_split` with :math:`axis=0` .

    Args:
        input (Tensor): The input tensor.
        indices_or_sections (Union[int, tuple(int), list(int)]): See `indices_or_sections` argument in
            :func:`mindspore.ops.tensor_split`.

    Returns:
        Tuple of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[ 0,  1,  2,  3],
        ...                           [ 4,  5,  6,  7],
        ...                           [ 8,  9, 10, 11],
        ...                           [12, 13, 14, 15]])
        >>> mindspore.ops.vsplit(input, 2)
        (Tensor(shape=[2, 4], dtype=Int64, value=
         [[0, 1, 2, 3],
          [4, 5, 6, 7]]),
         Tensor(shape=[2, 4], dtype=Int64, value=
         [[ 8,  9, 10, 11],
          [12, 13, 14, 15]]))
        >>> mindspore.ops.vsplit(input, [3, 6])
        (Tensor(shape=[3, 4], dtype=Int64, value=
         [[ 0,  1,  2,  3],
          [ 4,  5,  6,  7],
          [ 8,  9, 10, 11]]),
         Tensor(shape=[1, 4], dtype=Int64, value=
         [[12, 13, 14, 15]]),
         Tensor(shape=[0, 4], dtype=Int64, value=
         ))
    """
    if not isinstance(input, Tensor):
        raise TypeError(f'expect `x` is a Tensor, but got {type(input)}')
    if input.ndim < 1:
        raise ValueError(
            f'vsplit expect `x` is a Tensor with at least 1 dimension, but got {input.ndim}')
    return tensor_split(input, indices_or_sections, 0)


def hsplit(input, indices_or_sections):
    """
    Splits a tensor into multiple sub-tensors horizontally.
    It is equivalent to `ops.tensor_split` with :math:`axis=1` .

    Args:
        input (Tensor): The input tensor.
        indices_or_sections (Union[int, tuple(int), list(int)]): See argument in :func:`mindspore.ops.tensor_split`.

    Returns:
        Tuple of tensors

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.ops.arange(0, 6).reshape((2, 3))
        >>> output = mindspore.ops.hsplit(mindspore.tensor(input_x, mindspore.float32), 3)
        >>> print(output)
        (Tensor(shape=[2, 1], dtype=Float32, value=[[ 0.00000000e+00], [ 3.00000000e+00]]),
         Tensor(shape=[2, 1], dtype=Float32, value=[[ 1.00000000e+00], [ 4.00000000e+00]]),
         Tensor(shape=[2, 1], dtype=Float32, value=[[ 2.00000000e+00], [ 5.00000000e+00]]))
    """
    if not isinstance(input, Tensor):
        raise TypeError(f'expect `x` is a Tensor, but got {type(input)}')
    if input.ndim < 2:
        raise ValueError(
            f'hsplit expect `x` is a Tensor with at least 2 dimension, but got {input.ndim}')

    return tensor_split(input, indices_or_sections, 1)


def dsplit(input, indices_or_sections):
    """
    Splits a tensor along the 3rd axis.
    It is equivalent to `ops.tensor_split` with :math:`axis=2` .

    Args:
        input (Tensor): A tensor to be divided.
        indices_or_sections (Union[int, tuple(int), list(int)]): See `indices_or_sections` argument in
            :func:`mindspore.ops.tensor_split`.

    Returns:
        Tuple of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.ops.arange(16.0).reshape(2, 2, 4)
        >>> print(input)
        [[[ 0.  1.  2.  3.]
          [ 4.  5.  6.  7.]]
         [[ 8.  9. 10. 11.]
          [12. 13. 14. 15.]]]
        >>> output = mindspore.ops.dsplit(input, 2)
        >>> print(output)
        (Tensor(shape=[2, 2, 2], dtype=Float32, value=
        [[[ 0.00000000e+00,  1.00000000e+00],
          [ 4.00000000e+00,  5.00000000e+00]],
         [[ 8.00000000e+00,  9.00000000e+00],
          [ 1.20000000e+01,  1.30000000e+01]]]), Tensor(shape=[2, 2, 2], dtype=Float32, value=
        [[[ 2.00000000e+00,  3.00000000e+00],
          [ 6.00000000e+00,  7.00000000e+00]],
         [[ 1.00000000e+01,  1.10000000e+01],
          [ 1.40000000e+01,  1.50000000e+01]]]))
        >>> output = mindspore.ops.dsplit(input, [3, 6])
        >>> print(output)
        (Tensor(shape=[2, 2, 3], dtype=Float32, value=
        [[[ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00],
          [ 4.00000000e+00,  5.00000000e+00,  6.00000000e+00]],
         [[ 8.00000000e+00,  9.00000000e+00,  1.00000000e+01],
          [ 1.20000000e+01,  1.30000000e+01,  1.40000000e+01]]]), Tensor(shape=[2, 2, 1], dtype=Float32, value=
        [[[ 3.00000000e+00],
          [ 7.00000000e+00]],
         [[ 1.10000000e+01],
          [ 1.50000000e+01]]]), Tensor(shape=[2, 2, 0], dtype=Float32, value=
        ))
    """
    if not isinstance(input, Tensor):
        raise TypeError(f'expect `x` is a Tensor, but got {type(input)}')
    if input.ndim < 3:
        raise ValueError(
            f'dsplit expect `x` is a Tensor with at least 3 dimension, but got {input.ndim}')

    return tensor_split(input, indices_or_sections, 2)


def _init_and_select_elem(input, initial, where, cmp_fn):  # pylint: disable=redefined-outer-name
    """Initialize the input according to Initial, and select the element according to where."""
    if initial is not None:
        initial = ops.fill(input.dtype, input.shape, initial)
        input = cmp_fn(input, initial)

    if where is not None and not isinstance(where, Tensor):
        where = Tensor(where, dtype=mstype.bool_)

    if where is not None and (where.shape or not where):
        if initial is None:
            raise ValueError('initial value must be provided for where masks')
        where = where.broadcast_to(input.shape)
        initial = initial.broadcast_to(input.shape)
        input = ops.select(where, input, initial)
    return input


def max(input, axis=None, keepdims=False, *, initial=None, where=None):  # pylint: disable=redefined-outer-name
    """
    Return the maximum values and their indices along the given axis of the tensor.

    Args:
        input (Tensor): The input tensor.
        axis (int): Specify the axis for computation. If ``None`` , compute all elements in the `input` . Default:
            ``False`` .
        keepdims (bool): Whether the output tensor has dim retained. Default ``False``.

    Keyword Args:
        initial (scalar, optional): Initial value for the maximum. Default ``None``.
        where (Tensor[bool], optional): Specifies the range over which to compute the maximum values. The shape of this
            tensor must be broadcastable to the shape of `input` . An `initial` value must be specified. Default
            ``None`` , indicating that all elements are to be computed.

    Returns:
        Tuple(max, max_indices) of 2 tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[9, 3, 4, 5],
        ...                           [5, 2, 7, 4],
        ...                           [8, 1, 3, 6]])
        >>> # case 1: By default, compute the maximum of all elements.
        >>> mindspore.ops.max(input)
        (Tensor(shape=[], dtype=Int64, value= 9),
         Tensor(shape=[], dtype=Int64, value= 0))
        >>>
        >>> # case 2: Compute maximum along axis 1.
        >>> mindspore.ops.max(input, axis=1)
        (Tensor(shape=[3], dtype=Int64, value= [9, 7, 8]),
         Tensor(shape=[3], dtype=Int64, value= [0, 2, 0]))
        >>>
        >>> # case 3: If keepdims=True, the output shape will be same of that of the input.
        >>> mindspore.ops.max(input, axis=1, keepdims=True)
        (Tensor(shape=[3, 1], dtype=Int64, value=
         [[9],
          [7],
          [8]]),
         Tensor(shape=[3, 1], dtype=Int64, value=
         [[0],
          [2],
          [0]]))
        >>>
        >>> # case 4: Use "where" to include only specific elements in computing the maximum.
        >>> where = mindspore.tensor([[0, 0, 1, 0],
        ...                           [0, 0, 1, 1],
        ...                           [1, 1, 1, 0]], dtype=mindspore.bool)
        >>> mindspore.ops.max(input, axis=1, keepdims=True, initial=0, where=where)
        (Tensor(shape=[3, 1], dtype=Int64, value=
         [[4],
          [7],
          [8]]),
         Tensor(shape=[3, 1], dtype=Int64, value=
         [[2],
          [2],
          [0]]))
        >>>
        >>> # case 5: The shape of "where" must be broadcast compatible with input.
        >>> where = mindspore.tensor([[False],
        ...                           [False],
        ...                           [False]])
        >>> mindspore.ops.max(input, axis=0, keepdims=True, initial=0, where=where)
        (Tensor(shape=[1, 4], dtype=Int64, value=
         [[0, 0, 0, 0]]),
         Tensor(shape=[1, 4], dtype=Int64, value=
         [[0, 0, 0, 0]]))
    """
    if not input.shape:
        return (input, Tensor(0, dtype=mstype.int64))
    if axis is None:
        return (max_(input), Tensor(0, dtype=mstype.int64))
    if initial is not None and not isinstance(initial, numbers.Number):
        raise TypeError(
            f"For 'max', 'initial' must be a scalar, but got {type(initial)}")
    if axis is not None and not isinstance(axis, int):
        raise TypeError(f"For 'max', 'axis' must be int, but got {type(axis)}")
    input = _init_and_select_elem(input, initial, where, ops.maximum)
    argmax_with_value_op = _get_cache_prim(ArgMaxWithValue)(axis, keepdims)
    indices, values = argmax_with_value_op(input)
    return values, indices


def argmax(input, dim=None, keepdim=False):
    """
    Return the indices of the maximum values along a specified dimension of the tensor.

    Args:
        input (Tensor): The input tensor.
        dim (Union[int, None], optional): Specify the dimension for computation. If ``None`` ,
            compute maximum value indexes of all elements in
            the `input` . Default ``None`` .
        keepdim (bool, optional): Whether the output tensor has dim retained. Default ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[9, 3, 4, 5],
        ...                           [5, 2, 7, 4],
        ...                           [8, 1, 3, 6]])
        >>> # case 1: By default, compute the maximum indice of all elements.
        >>> mindspore.ops.argmax(input)
        Tensor(shape=[], dtype=Int64, value= 0)
        >>>
        >>> # case 2: Compute maximum indice along dim 1.
        >>> mindspore.ops.argmax(input, dim=1)
        Tensor(shape=[3], dtype=Int64, value= [0, 2, 0])
        >>>
        >>> # case 3: If keepdim=True, the output shape will be same of that of the input.
        >>> mindspore.ops.argmax(input, dim=1, keepdim=True)
        Tensor(shape=[3, 1], dtype=Int64, value=
        [[0],
         [2],
         [0]])
    """
    _check_attr_dtype("keepdim", keepdim, [bool], "argmax")
    if not input.shape:
        return Tensor(0)
    if input.dtype == mstype.bool_:
        input = input.astype(mstype.int32)
    is_dim_none = False
    if dim is None:
        input = reshape_(input, (-1,))
        dim = 0
        is_dim_none = True
    out = _get_cache_prim(Argmax)(dim, mstype.int64)(input)
    if keepdim and not is_dim_none:
        out = expand_dims(out, dim)
    return out


def min(input, axis=None, keepdims=False, *, initial=None, where=None):  # pylint: disable=redefined-outer-name
    """
    Return the minimum values and their indices along the given axis of the tensor.

    Args:
        input (Tensor): The input tensor.
        axis (int): Specify the axis for computation. If ``None`` , compute all elements in the `input` . Default
            ``None`` .
        keepdims (bool): Whether the output tensor has dim retained. Default ``False`` .

    Keyword Args:
        initial (scalar, optional): Initial value for the minimum. Default ``None`` .
        where (Tensor[bool], optional): Specifies the range over which to compute the maximum values. The shape of this
            tensor must be broadcastable to the shape of `input` . An `initial` value must be specified. Default
            ``None`` , indicating that all elements are to be computed.

    Returns:
        Tuple(min, min_indices) of 2 tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[2, 5, 1, 6],
        ...                           [3, -7, -2, 4],
        ...                           [8, -4, 1, -3]])
        >>> # case 1: By default, compute the minimum of all elements.
        >>> mindspore.ops.min(input)
        (Tensor(shape=[], dtype=Int64, value= -7),
         Tensor(shape=[], dtype=Int64, value= 0))
        >>>
        >>> # case 2: Compute minimum along axis 1.
        >>> mindspore.ops.min(input, axis=1)
        (Tensor(shape=[3], dtype=Int64, value= [ 1, -7, -4]),
         Tensor(shape=[3], dtype=Int64, value= [2, 1, 1]))
        >>>
        >>> # case 3: If keepdims=True, the output shape will be same of that of the input.
        >>> mindspore.ops.min(input, axis=1, keepdims=True)
        (Tensor(shape=[3, 1], dtype=Int64, value=
         [[ 1],
          [-7],
          [-4]]),
         Tensor(shape=[3, 1], dtype=Int64, value=
         [[2],
          [1],
          [1]]))
        >>>
        >>> # case 4: Use "where" to include only specific elements in computing the minimum.
        >>> where = mindspore.tensor([[1, 0, 1, 0],
        ...                           [0, 0, 1, 1],
        ...                           [1, 1, 1, 0]], dtype=mindspore.bool)
        >>> mindspore.ops.min(input, axis=1, keepdims=True, initial=0, where=where)
        (Tensor(shape=[3, 1], dtype=Int64, value=
         [[ 0],
          [-2],
          [-4]]),
         Tensor(shape=[3, 1], dtype=Int64, value=
         [[0],
          [2],
          [1]]))
        >>>
        >>> # case 5: The shape of "where" must be broadcast compatible with input.
        >>> where = mindspore.tensor([[False],
        ...                           [False],
        ...                           [False]])
        >>> mindspore.ops.min(input, axis=0, keepdims=True, initial=0, where=where)
        (Tensor(shape=[1, 4], dtype=Int64, value=
         [[0, 0, 0, 0]]),
         Tensor(shape=[1, 4], dtype=Int64, value=
         [[0, 0, 0, 0]]))
    """
    if not input.shape:
        return (input, Tensor(0, dtype=mstype.int64))
    if axis is None:
        return (min_(input), Tensor(0, dtype=mstype.int64))
    if initial is not None and not isinstance(initial, numbers.Number):
        raise TypeError(
            f"For 'min', 'initial' must be a scalar, but got {type(initial)}")
    if axis is not None and not isinstance(axis, int):
        raise TypeError(f"For 'min', 'axis' must be int, but got {type(axis)}")
    input = _init_and_select_elem(input, initial, where, ops.minimum)
    argmin_with_value_op = _get_cache_prim(ArgMinWithValue)(axis, keepdims)
    indices, values = argmin_with_value_op(input)
    return values, indices


def aminmax(input, *, axis=0, keepdims=False):
    """
    Return the minimum values and maximum values along the given axes of the tensor.

    Args:
        input (Tensor): The input tensor.

    Keyword Args:
        axis (int, optional): Specify the axis for computation. If ``None`` , compute all elements in the `input` .
            Default ``0`` .
        keepdims (bool, optional): Whether the output tensor has dim retained. Default ``False`` .

    Returns:
        Tuple(min, max) of 2 tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[9, 3, 4, 5],
        ...                           [5, 2, 7, 4],
        ...                           [8, 1, 3, 6]])
        >>>
        >>> # case 1: By default, compute along axis 0.
        >>> mindspore.ops.aminmax(input)
        (Tensor(shape=[4], dtype=Int64, value= [5, 1, 3, 4]),
         Tensor(shape=[4], dtype=Int64, value= [9, 3, 7, 6]))
        >>>
        >>> # case 2: Disregard NaN (Not a Number) values present in the input during computation.
        >>> input = mindspore.tensor([[9, 3, 4, 5],
        ...                           [5, 2, 7, 4],
        ...                           [8, 1, 3, float('nan')]])
        >>> mindspore.ops.aminmax(input, axis=None)
        (Tensor(shape=[], dtype=Float32, value= 1),
         Tensor(shape=[], dtype=Float32, value= 9))
        >>>
        >>> # case 3: If keepdims=True, the output shape will be same of that of the input.
        >>> mindspore.ops.aminmax(input, axis=None, keepdims=True)
        (Tensor(shape=[1, 1], dtype=Float32, value=
         [[ 1.00000000e+00]]),
         Tensor(shape=[1, 1], dtype=Float32, value=
         [[ 9.00000000e+00]]))
    """
    if axis is None:
        output0, _ = ops.min(input, axis, keepdims)
        output1, _ = ops.max(input, axis, keepdims)
        if keepdims is True:
            output0 = ops.reshape(output0, [1] * input.ndim)
            output1 = ops.reshape(output1, [1] * input.ndim)
        return output0, output1
    argmin_with_value_op = _get_cache_prim(ArgMinWithValue)(axis, keepdims)
    argmax_with_value_op = _get_cache_prim(ArgMaxWithValue)(axis, keepdims)
    _, output0 = argmin_with_value_op(input)
    _, output1 = argmax_with_value_op(input)
    return output0, output1


def narrow(input, axis, start, length):
    """
    Slice the tensor from the `start` position with a length of `length` along `axis` .

    Args:
        input (Tensor): The input tensor.
        axis (int): the specified axis.
        start (int): the specified starting position.
        length (int): the specified length.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mindspore.int32)
        >>> output = mindspore.ops.narrow(x, 0, 0, 2)
        >>> print(output)
        [[ 1 2 3]
         [ 4 5 6]]
        >>> output = mindspore.ops.narrow(x, 1, 1, 2)
        >>> print(output)
        [[ 2 3]
         [ 5 6]
         [ 8 9]]
    """
    validator.check_value_type("input", input, Tensor, "narrow")
    validator.check_axis_in_range(axis, input.ndim)
    validator.check_int_range(start, 0, input.shape[axis], validator.INC_LEFT)
    validator.check_int_range(
        length, 1, input.shape[axis] - start, validator.INC_BOTH)

    begins = [0] * input.ndim
    begins[axis] = start
    sizes = list(input.shape)
    sizes[axis] = length
    return tensor_slice(input, begins, sizes)


def topk(input, k, dim=None, largest=True, sorted=True):
    r"""
    Return the top `k` largest or smallest elements of the input tensor along a specified dimension.

    .. warning::
        - If sorted is set to False, it will use the aicpu operator, the performance may be reduced. In addition, due to
          different memory layout and traversal methods on different platforms, the display order of calculation results
          may be inconsistent when `sorted` is False.

    Args:
        input (Tensor): The input tensor.
        k (int): The number elements to be returned.
        dim (int, optional): Specify the dimension for sorting. Default ``None`` .
        largest (bool, optional): If ``True`` , return largest elements. If ``False`` , then return smallest elements.
            Default ``True`` .
        sorted (bool, optional): If ``True`` , the elements are returned in descending order. If ``False`` , the
            obtained elements will not be sorted. Default ``True`` .

    Returns:
        Tuple(values, indices) of 2 tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[8, 2, 1],
        ...                           [5, 9, 3],
        ...                           [4, 6, 7]])
        >>> # case 1: If dim is not given, the last dimension of the input is chosen.
        >>> mindspore.ops.topk(input, 2)
        (Tensor(shape=[3, 2], dtype=Int64, value=
         [[8, 2],
          [9, 5],
          [7, 6]]),
         Tensor(shape=[3, 2], dtype=Int32, value=
         [[0, 1],
          [1, 0],
          [2, 1]]))
        >>> # case 2: when dim is 0:
        >>> mindspore.ops.topk(input, 2, dim=0)
        (Tensor(shape=[2, 3], dtype=Int64, value=
        [[8, 9, 7],
         [5, 6, 3]]),
        Tensor(shape=[2, 3], dtype=Int32, value=
        [[0, 1, 2],
         [1, 2, 1]]))
        >>> # case 3: when largest is False, return smallest values.
        >>> mindspore.ops.topk(input, 2, dim=0, largest=False)
        (Tensor(shape=[2, 3], dtype=Int64, value=
         [[4, 2, 1],
          [5, 6, 3]]),
         Tensor(shape=[2, 3], dtype=Int32, value=
         [[2, 0, 0],
          [1, 2, 1]]))
    """
    validator.check_value_type("largest", largest, [bool], "topk")
    top_k_ = _get_cache_prim(P.TopK)(sorted)
    if not largest:
        input = -input
    if dim is None or dim == input.ndim - 1:
        if not largest:
            res = top_k_(input, k)
            values, indices = -res[0], res[1]
            return values, indices
        return top_k_(input, k)
    input = input.swapaxes(dim, input.ndim - 1)
    output = top_k_(input, k)
    values = output[0].swapaxes(dim, input.ndim - 1)
    indices = output[1].swapaxes(dim, input.ndim - 1)
    if not largest:
        res = (-values, indices)
    else:
        res = (values, indices)
    return res


def expand(input_x, size):
    r"""
    This interface will be deprecated in the future, and use :func:`mindspore.ops.broadcast_to` instead.
    """
    expand_op = _get_cache_prim(Expand)()
    return expand_op(input_x, size)


@_primexpr
def _check_fold_param(param, param_name):
    """Check the parameters of fold op."""
    validator.check_value_type(param_name, param, [int, list, tuple], 'fold')
    param = (param, param) if isinstance(param, int) else param
    validator.check_int(len(param), 2, validator.EQ, param_name, 'fold')
    if param_name == "padding":
        validator.check_non_negative_int_sequence(param, param_name, 'fold')
    else:
        validator.check_positive_int_sequence(param, param_name, 'fold')
    return param


def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    r"""
    Combines an array of sliding local blocks into a large containing tensor.

    Consider a batched input tensor of shape :math:`(N, C \times \prod(\text{kernel_size}), L)` ,
    where :math:`N` is the batch dimension, :math:`C \times \prod(\text{kernel_size})` is the
    total number of values within each block (a block has :math:`\prod(\text{kernel_size})` spatial
    locations each containing a `C`-channeled vector), and :math:`L` is the total number of such blocks:

    .. math::
        L = \prod_d \left\lfloor\frac{\text{output_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilations}[d] \times (\text{kernel_size}[d] - 1) - 1}{\text{strides}[d]} + 1\right\rfloor,

    where :math:`d` is over all spatial dimensions.

    Therefore, `output_size` is the spatial shape of the large containing tensor of the sliding local blocks.

    The `dilation`, `padding` and `stride` arguments specify how the sliding blocks are retrieved.

    .. warning::
        - The input must be a 3-dimensional Tensor with shape
          :math:`(N, C \times \prod(\text{kernel_size}), L)` .
        - The output must be a 4-dimensional Tensor with shape
          :math:`(N, C, output\_size[0], output\_size[1], ...)` .

    Args:
        input (Tensor): 3-D Tensor, supported dtypes: float16, float32, float64, complex64 and complex128.
        output_size (Tensor): 1D tensor with `2` elements of data type int.
        kernel_size (Union[int, tuple[int], list[int]]): The size of the kernel, should be two int
            for height and width. If type is int, it means that height equal with width. Must be specified.
        dilation (Union[int, tuple[int], list[int]], optional): The size of the dilation, should be two int
            for height and width. If type is int, it means that height equal with width. Default: ``1`` .
        padding (Union[int, tuple[int], list[int]], optional): The size of the padding, should be two int
            for height and width. If type is int, it means that height equal with width. Default: ``0`` .
        stride (Union[int, tuple[int], list[int]], optional): The size of the stride, should be two int
            for height and width. If type is int, it means that height equal with width. Default: ``1`` .

    Returns:
        A Tensor, with same type as `input` . And its shape is as described above.

    Raises:
        TypeError: If `output_size`, `kernel_size`, `stride`, `dilation`, `padding` data type is not int, tuple or list.
        ValueError: If `output_size`, `kernel_size`, `dilation`, `stride` value is not
            greater than zero or elements number more than `2`.
        ValueError: If `padding` value is less than zero or elements number more than `2`.
        ValueError: If `input.shape[1] != kernel_size[0] * kernel_size[1]`
        ValueError: If `input.shape[2]` does not match the calculated number of sliding blocks.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> from mindspore import dtype as mstype
        >>> x = Tensor(input_data=np.random.rand(16, 64, 25), dtype=mstype.float32)
        >>> output_size = Tensor(input_data=[8, 8], dtype=mstype.int32)
        >>> output = ops.fold(x, output_size, [2, 2], [2, 2], [2, 2], [2, 2])
        >>> print(output.shape)
        (16, 16, 8, 8)
    """
    kernel_size = _check_fold_param(kernel_size, "kernel_size")
    dilation = _check_fold_param(dilation, "dilation")
    padding = _check_fold_param(padding, "padding")
    stride = _check_fold_param(stride, "stride")
    fold_op = _get_cache_prim(Col2Im)(kernel_size, dilation, padding, stride)
    input_shape = ops.shape(input)
    k = kernel_size[0] * kernel_size[-1]
    r_shape = input_shape[:1] + (-1, k) + input_shape[-1:]
    input = ops.reshape(input, r_shape)
    return fold_op(input, output_size)


@_primexpr
def _check_unfold_params(param, param_name, param_size):
    """Check the parameters of unfold op."""
    validator.check_value_type(param_name, param, [int, tuple, list], 'unfold')
    param = (param, param) if isinstance(param, int) else param
    validator.check(param_name + " size", len(param), "",
                    param_size, validator.IN, 'unfold')
    if param_name == "padding":
        validator.check_non_negative_int_sequence(param, param_name, 'unfold')
    else:
        validator.check_positive_int_sequence(param, param_name, 'unfold')
    return param


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    r"""
    Extracts sliding local blocks from a batched input tensor.

    Consider a batched input tensor of shape :math:`(N, C, *)`,
    where :math:`N` is the batch dimension, :math:`C` is the channel dimension,
    and :math:`*` represent arbitrary spatial dimensions. This operation flattens
    each sliding `Kernel_size`- sized block within the spatial dimensions
    of input `input` into a column (i.e., last dimension) of a 3-D output
    tensor of shape :math:`(N, C \times \prod(\text{kernel_size}), L)`, where
    :math:`C \times \prod(\text{kernel_size})` is the total number of values
    within each block (a block has :math:`\prod(\text{kernel_size})` spatial
    locations, each containing a `C`-channeled vector), and :math:`L` is
    the total number of such blocks:

    .. math::
        L = \prod_d \left\lfloor\frac{\text{spatial_size}[d] + 2 \times \text{pads}[d] %
            - \text{dilations}[d] \times (\text{kernel_size}[d] - 1) - 1}{\text{strides}[d]} + 1\right\rfloor,

    where :math:`\text{spatial_size}` is formed by the spatial dimensions
    of input `input` (:math:`*` above), and :math:`d` is over all spatial
    dimensions.

    Therefore, indexing `output` at the last dimension (column dimension)
    gives all values within a certain block.

    The `dilation`, `padding` and `stride` arguments specify
    how the sliding blocks are retrieved.

    .. warning::
        - The output is a 3-dimensional Tensor whose shape is
          :math:`(N, C \times \prod(\text{kernel_size}), L)` .
        - This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): 4-D Tensor, supported dtypes: float16, float32, float64, complex64 and complex128.
        kernel_size (Union[int, tuple[int], list[int]]): The size of the kernel, should be two int
            for height and width. If type is int, it means that height equal with width.
        dilation (Union[int, tuple[int], list[int]], optional): The dilation of the window, should be two int
            for height and width. If type is int, it means that height equal with width. Default: ``1`` .
        padding (Union[int, tuple[int], list[int]], optional): The pad of the window, that must be
            a tuple/list of one or two `int` for height and width. Default: ``0`` .

            - If one int, pad_height = pad_width.
            - If two int, pad_height = padding[0], pad_width = padding[1].

        stride (Union[int, tuple[int], list[int]], optional): The stride of the window, should be two int
            for height and width. If type is int, it means that height equal with width. Default: ``1`` .

    Returns:
        A Tensor, with same type as `input` . And its shape is as described above.

    Raises:
        TypeError: If any data type of `kernel_size`, `stride`, `dilation`, `padding` is not int, tuple or list.
        ValueError: If `kernel_size`, `dilation`, `stride` value is not
            greater than zero or elements number more than `2`.
        ValueError: If `padding` value is less than zero.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.random.rand(4, 4, 32, 32), mindspore.float64)
        >>> output = ops.unfold(x, kernel_size=3, dilation=1, stride=1)
        >>> print(output.shape)
        (4, 36, 900)
    """
    kernel_size = _check_unfold_params(kernel_size, "kernel_size", [1, 2])
    dilation = _check_unfold_params(dilation, "dilation", [1, 2])
    padding = _check_unfold_params(padding, "padding", [1, 2])
    stride = _check_unfold_params(stride, "stride", [1, 2])
    unfold_op = _get_cache_prim(Im2Col)(ksizes=kernel_size,
                                        strides=stride,
                                        dilations=dilation,
                                        pads=padding)
    tmp = unfold_op(input)
    tmp_shape = ops.shape(tmp)
    out_shape = tmp_shape[:1] + (-1,) + tmp_shape[-1:]
    out = ops.reshape(tmp, out_shape)
    return out


@_primexpr
def _check_diagonal_axes(dim1, dim2, x_ndim):
    """Check the parameters of unfold op."""
    axes = validator.check_axis_valid((dim1, dim2), x_ndim)
    return axes


def _check_is_tensor(param_name, input, cls_name):
    """Returns True if input is Tensor."""
    if not isinstance(input, Tensor):
        raise TypeError(
            f"For {cls_name}, {param_name} must be a Tensor, but got {type(input)}.")


@_primexpr
def _check_diagonal_scatter_shape(diag_shape, src_shape):
    if diag_shape != src_shape:
        raise ValueError(f"For diagonal_scatter, the shape of src should equal to the shape of input diagonal,"
                         f"but got src.shape {src_shape} and diagonal shape {diag_shape}.")


def diagonal_scatter(input, src, offset=0, dim1=0, dim2=1):
    """
    Embeds the values of the `src` tensor into `input` along the diagonal elements of input, with respect to `dim1`
    and `dim2` .

    Note:
        Currently, ``inf`` value of elements in `input` or `src` is not supported.

    Args:
        input (Tensor): The input tensor, whose dimension is larger than 1.
        src (Tensor): The source tensor to embed.
        offset (int, optional): Diagonal offset. Default ``0`` .

              - When `offset` is a positive integer, shift the diagonal upward.
              - When `offset` is a negative integer, shift the diagonal downward.
        dim1 (int, optional): First dimension with respect to which to take diagonal. Default ``0`` .
        dim2 (int, optional): Second dimension with respect to which to take diagonal. Default ``1`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.ops.zeros((3, 3))
        >>> output = mindspore.ops.diagonal_scatter(input, mindspore.ops.ones(3), 0)
        >>> print(output)
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
        >>> output = mindspore.ops.diagonal_scatter(input, mindspore.ops.ones(2), 1)
        >>> print(output)
        [[0. 1. 0.]
         [0. 0. 1.]
         [0. 0. 0.]]
    """
    _check_is_tensor("input", input, "diagonal_scatter")
    _check_is_tensor("src", src, "diagonal_scatter")
    input_diag = input.diagonal(offset, dim1, dim2)
    _check_diagonal_scatter_shape(input_diag.shape, src.shape)
    input_shape = input.shape
    zeros_shape = list(input_shape)
    m, n = input_shape[dim1], input_shape[dim2]
    if m == n:
        src = ops.diag_embed(src, offset, dim1, dim2)
        input = input - ops.diag_embed(input_diag, offset, dim1, dim2)
        return input + src
    if m > n:
        axis = dim2
        zeros_shape[axis] = m - n
    else:
        axis = dim1
        zeros_shape[axis] = n - m
    zeros_tensor = zeros(zeros_shape, dtype=input.dtype)
    input = concat((input, zeros_tensor), axis)
    input_diag = input.diagonal(offset, dim1, dim2)
    if src.shape != input_diag.shape:
        zeros_shape = []
        for i, ax in enumerate(src.shape):
            if ax == input_diag.shape[i]:
                zeros_shape.append(ax)
            else:
                axis = i
                zeros_shape.append(input_diag.shape[i] - ax)
        zeros_tensor = zeros(zeros_shape, dtype=src.dtype)
        src = concat((src, zeros_tensor), axis)
    src = ops.diag_embed(src, offset, dim1, dim2)
    input = input - ops.diag_embed(input_diag, offset, dim1, dim2)
    input = input + src
    begin = (0,) * input.ndim
    return slice(input, begin, input_shape)


def lstsq(input, A):
    r"""
    Computes the solutions of the least squares and minimum norm problems of full-rank
    matrix `x` of size :math:`(m \times n)` and matrix `a` of size :math:`(m \times k)`.

    If :math:`m \geq n`, `lstsq` solves the least-squares problem:

    .. math::

       \begin{array}{ll}
       \min_y & \|xy-a\|_2.
       \end{array}

    If :math:`m < n`, `lstsq` solves the least-norm problem:

    .. math::

       \begin{array}{llll}
       \min_y & \|y\|_2 & \text{subject to} & xy = a.
       \end{array}

    where `y` is the returned tensor.

    Args:
        input (Tensor): The :math:`(m \times n)` matrix equivalent to :math:`x` in above.
            The input tensor whose data type is float16, float32 or float64.
        A (Tensor): The :math:`(m \times k)` matrix equivalent to :math:`a` in above.
            The input tensor whose data type is float16, float32 or float64.

    Returns:
        Tensor, the least squares or minimum norm problems solution, which has shape :math:`(n \times k)`.
        The data type is the same with `input`.

    Raises:
        TypeError: If `input` or `A` is not a Tensor.
        TypeError: If dtype of `input` or `A` is not one of: float16, float32, float64.
        TypeError: If the dtypes of `input` and `A` are not the same.
        ValueError: If the dimension of `input` is not equal to 2.
        ValueError: If the dimension of `A` is not equal to 2 or 1.
        ValueError: If the length of input_dims[0] is not equal to the length of A_dims[0].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([[2,1,5],[3,5,1],[1,1,1]]),mindspore.float32)
        >>> a = Tensor(np.array([[10,5],[15,8],[7,4]]),mindspore.float32)
        >>> output = ops.lstsq(x, a)
        >>> print(output)
        [[17.000002  11.000002 ]
         [-6.5000005 -4.500001 ]
         [-3.500002  -2.5000017]]
    """
    return lstsq_(input, A)


def mvlgamma(input, p):
    r"""
    Compute the multivariate log-gamma function with dimension `p` element-wise.
    The mathematical calculation process of Mvlgamma is shown as follows:

    .. math::

        \log (\Gamma_{p}(input))=C+\sum_{i=1}^{p} \log (\Gamma(input-\frac{i-1}{2}))

    where :math:`C = \log(\pi) \times \frac{p(p-1)}{4}` and :math:`\Gamma(\cdot)` is the Gamma function.

    Args:
        input (Tensor): The input tensor of the multivariate log-gamma function,
          And the value of any element in `input` must be greater than :math:`(p - 1) / 2`.
        p (int): The number of dimensions. And the value of `p` must be greater than or equal to 1.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[3, 4, 5], [4, 2, 6]], mindspore.float32)
        >>> y = mindspore.ops.mvlgamma(x, p=3)
        >>> print(y)
        [[2.694925 5.402975 9.140645]
         [5.402975 1.596312 13.64045]]
    """
    mvlgamma_op = _get_cache_prim(Mvlgamma)(p)
    return mvlgamma_op(input)


def nonzero(input, *, as_tuple=False):
    r"""
    Return the positions of all non-zero values.

    Args:
        input (Tensor): The input tensor.

    .. note::
        - Ascend: Rank of Input tensor can be equal to 0 except GE backend.
        - CPU/GPU: Rank of Input tensor should be greater than or eaqual to 1.
        - Currently, only the Ascend backend is supported when `as_tuple` is ``True``.

    Keyword Args:
        as_tuple (bool, optional): Whether the output is tuple. Default ``False`` .

    Returns:
        Tensor or tuple of tensors

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[[1,  0], [-5, 0]]], mindspore.int32)
        >>> output = mindspore.ops.nonzero(x)
        >>> print(output)
        [[0 0 0]
         [0 1 0]]
        >>> x = mindspore.tensor([1, 0, 2, 0, 3], mindspore.int32)
        >>> output = mindspore.ops.nonzero(x, as_tuple=False)
        >>> print(output)
        [[0]
         [2]
         [4]]
        >>> x = mindspore.tensor([[[1,  0], [-5, 0]]], mindspore.int32)
        >>> output = mindspore.ops.nonzero(x, as_tuple=True)
        >>> print(output)
        (Tensor(shape=[2], dtype=Int64, value=[0, 0]),
         Tensor(shape=[2], dtype=Int64, value=[0, 1]),
         Tensor(shape=[2], dtype=Int64, value=[0, 0]))
        >>> x = mindspore.tensor([1, 0, 2, 0, 3], mindspore.int32)
        >>> output = mindspore.ops.nonzero(x, as_tuple=True)
        >>> print(output)
        (Tensor(shape=[3], dtype=Int64, value=[0, 2, 4]), )
    """
    if not isinstance(as_tuple, bool):
        raise TypeError(
            f"For array function 'nonzero', 'as_tuple' must be bool, but got {type(as_tuple)}.")
    if as_tuple:
        return non_zero_ext_(input)
    return non_zero_(input)


def argwhere(input):
    """
    Return a tensor of containing the positions of all non-zero elements in the input tensor.

    Args:
        input (Tensor): The input tensor.

    Returns:
        2-D Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[[1,  0], [-5, 0]]], mindspore.int32)
        >>> output = mindspore.ops.argwhere(x)
        >>> print(output)
        [[0 0 0]
         [0 1 0]]
    """
    return nonzero(input)


def column_stack(tensors):
    """
    Creates a new tensor by horizontally stacking the tensors in `tensors`.
    Similar to :func:`mindspore.ops.hstack`.

    Args:
        tensors (Union[tuple[Tensor], list[Tensor]]): The tensors to be concatenated.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x1 = mindspore.tensor([1, 1, 1])
        >>> x2 = mindspore.tensor([2, 2, 2])
        >>> output = mindspore.ops.column_stack((x1, x2))
        >>> print(output)
        [[1 2]
         [1 2]
         [1 2]]
    """
    if not isinstance(tensors, (list, tuple)):
        raise TypeError(
            f"For column_stack, the input must be list or tuple of tensors, but got {type(tensors)}.")

    trans_x = ()
    for tensor in tensors:
        if not isinstance(tensor, Tensor):
            raise TypeError(
                f"For column_stack, the input element must be tensor, but got {type(tensor)}.")
        if tensor.ndim < 1:
            tensor = expand_dims(tensor, 0)
        if tensor.ndim == 1:
            tensor = expand_dims(tensor, 1)
        trans_x += (tensor,)
    if not trans_x:
        raise ValueError(
            "For column_stack, the input must have at least 1 tensor, but got 0.")
    _concat = _get_cache_prim(P.Concat)(1)
    return _concat(trans_x)


def hstack(tensors):
    """
    Stacks tensors in sequence horizontally.

    .. note::
        - Dynamic rank input of 8-D tensors with type float64 is not supported in `graph mode
          (mode=mindspore.GRAPH_MODE) <https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html>`_.
        - This is equivalent to concatenation along the second axis, except for 1-D tensors
          where it concatenates along the first axis.
        - The tensors must have the same shape along all but the second axis, except 1-D tensors
          which can be any length.

    Args:
        tensors (Union[tuple[Tensor], list[Tensor]]): Tuple of tensors or list of tensors.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x1 = mindspore.tensor([1, 1, 1])
        >>> x2 = mindspore.tensor([2, 2, 2])
        >>> output = mindspore.ops.hstack((x1, x2))
        >>> print(output)
        [1. 1. 1. 2. 2. 2.]
    """
    if not isinstance(tensors, (list, tuple)):
        raise TypeError(
            f"For hstack, the input must be list or tuple, but got {type(tensors)}.")

    tuple_of_tensor = ()
    for tensor in tensors:
        if not isinstance(tensor, Tensor):
            raise TypeError(
                f"For hstack, the input element must be tensor, but got {type(tensor)}.")
        if tensor.ndim < 1:
            tensor = expand_dims(tensor, 0)
        tuple_of_tensor += (tensor,)
    if not tuple_of_tensor:
        raise ValueError(
            "For hstack, the input must have at least 1 tensor, but got 0.")
    axis = 0 if tuple_of_tensor[0].ndim == 1 else 1
    return concat_impl(tuple_of_tensor, axis)


@constexpr
def _check_axis_valid(axis, ndim):
    """
    Checks axis are valid given ndim, and returns axis that can be passed
    to the built-in operator (non-negative, int or tuple).
    """
    if axis is None:
        axis = ops.make_range(ndim)
        return axis
    if isinstance(axis, (tuple, list)):
        axis = tuple(map(lambda x: _check_check_axis_in_range(x, ndim), axis))
        return axis
    return (_check_check_axis_in_range(axis, ndim),)


@constexpr
def _get_moved_perm(ndim, source, destination):
    """
    Helper function for movedim, returns permutation after moving axis
    from source to destination.
    """
    dest_sorted_idx = [i for i, _ in sorted(enumerate(destination), key=operator.itemgetter(1))]
    axis_orig = [i for i in builtins.range(0, ndim) if i not in source]

    k = 0
    m = 0
    perm = []
    for i in dest_sorted_idx:
        # inserts an axis that has been moved, denoted by n, and axis that remain
        # in their original position, indexed from k to k + n - m, into index m in
        # the list of permuted axis
        n = destination[i]
        j = k + n - m
        perm += axis_orig[k:j]
        perm.append(source[i])
        k += n - m
        m = n + 1
    perm += axis_orig[k:]
    return tuple(perm)


def movedim(x, source, destination):
    """
    Swap two dimensions of the input tensor.

    Args:
        x (Tensor): The input tensor.
        source (Union[int, sequence[int]]): Original dimensions.
        destination (Union[int, sequence[int]]): Destination positions for each of the original dimensions.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case1 : moving single axis
        >>> import mindspore
        >>> x = mindspore.tensor(mindspore.ops.zeros((3, 4, 5)))
        >>> output = mindspore.ops.movedim(x, 0, -1)
        >>> print(output.shape)
        (4, 5, 3)
        >>> # case 2 : moving multiple axes
        >>> x = mindspore.tensor(mindspore.ops.zeros((3, 4, 5)))
        >>> output = mindspore.ops.movedim(x, (0, 2), (1, 2))
        >>> print(output.shape)
        (4, 3, 5)
    """
    ndim = ops.rank(x)
    source = _check_axis_valid(source, ndim)
    destination = _check_axis_valid(destination, ndim)
    if len(source) != len(destination):
        raise ValueError(
            f"For `source` and `destination` arguments, the number of elements must be the same, but got 'source':"
            f" {len(source)} and 'destination': {len(destination)}.")
    perm = _get_moved_perm(ndim, source, destination)
    return transpose_(x, perm)


def moveaxis(x, source, destination):
    """
    Move axis of an array from source to destination.

    Refer to :func:`mindspore.ops.movedim` for more detail.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.ops.zeros((3, 4, 5))
        >>> output = mindspore.ops.moveaxis(x, 0, -1)
        >>> print(output.shape)
        (4, 5, 3)
    """

    return movedim(x, source, destination)


@_primexpr
def _check_swapaxes_axis(axes, ndim):
    return validator.check_swapaxes_axis(axes, ndim)


def swapaxes(input, axis0, axis1):
    '''
    Interchange two axes of a tensor.

    Args:
        input(Tensor): The input tensor.
        axis0 (int): First axis.
        axis1 (int): Second axis.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor(mindspore.ops.ones(([2, 3, 4])))
        >>> output = mindspore.ops.swapaxes(input, 0, 2)
        >>> print(output.shape)
        (4, 3, 2)
    '''
    if not isinstance(input, Tensor):
        raise TypeError(
            f'For ops.swapaxes, parameter `input` must be Tensor, but got {type(input)}')

    axis0, axis1 = _check_swapaxes_axis((axis0, axis1), input.ndim)
    if axis0 == axis1:
        return input
    if axis0 > axis1:
        axis0, axis1 = axis1, axis0

    perm = ops.make_range(0, input.ndim)
    if axis1 + 1 < input.ndim:
        new_perm = perm[0:axis0] + perm[axis1:axis1 + 1] + \
            perm[axis0 + 1:axis1] + perm[axis0:axis0 + 1] + perm[axis1 + 1:]
    else:
        new_perm = perm[0:axis0] + perm[axis1:axis1 + 1] + \
            perm[axis0 + 1:axis1] + perm[axis0:axis0 + 1]

    return transpose_(input, new_perm)


def swapdims(input, dim0, dim1):
    '''
    Interchange two dims of a tensor.
    This function is equivalent to :func:`mindspore.ops.swapaxes` function.

    Args:
        input(Tensor): The input tensor.
        dim0 (int): First dim.
        dim1 (int): Second dim.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor(mindspore.ops.ones([2, 3, 4]))
        >>> output = mindspore.ops.swapdims(input, 0, 2)
        >>> print(output.shape)
        (4, 3, 2)
    '''
    return ops.swapaxes(input, dim0, dim1)


@constexpr
def _check_is_int(arg_value, arg_name, op_name):
    arg_value = validator.check_is_int(arg_value, arg_name, op_name)
    return arg_value


@_primexpr
def _check_positive_int(arg_value, arg_name, op_name):
    arg_value = validator.check_int_range(
        arg_value, 0, 2147483647, validator.INC_RIGHT, arg_name, op_name)
    return arg_value


@constexpr
def _check_axis_range(arg_value, limit, arg_name, op_name):
    arg_value = validator.check_int_range(
        arg_value, -limit, limit, validator.INC_LEFT, arg_name, op_name)
    return arg_value


@_primexpr
def _cal_repeat_dims(x_rank, rep, expand_axis):
    rep_dims = [1] * (x_rank + 1)
    rep_dims[expand_axis] = rep
    return tuple(rep_dims)


@_primexpr
def _cal_reshape(x_shape, rep, axis):
    x_reshape = list(x_shape)
    x_reshape[axis] *= rep
    return tuple(x_reshape)


@_primexpr
def _check_rank_range(x_rank, limit, arg_name, op_name):
    if x_rank > limit:
        raise ValueError(
            f"For {op_name}, the rank of {arg_name} should be less than or equal to {limit}, but got {x_rank}.")
    return x_rank


def repeat_interleave(input, repeats, axis=None):
    """
    Repeat elements of a tensor along an axis, like :func:`mindspore.numpy.repeat`.

    Args:
        input (Tensor): The input tensor.
        repeats (Union[int, tuple, list, Tensor]): The number of times to repeat, must be positive.
        axis (int, optional): The axis along which to repeat, Default ``None``. if dims is None,
            both the input and output tensors will be flattened into 1-D.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>>  # case 1 : repeat on axis 0
        >>> input = mindspore.tensor([[0, 1, 2], [3, 4, 5]], mindspore.int32)
        >>> output = mindspore.ops.repeat_interleave(input, repeats=2, axis=0)
        >>> print(output)
        [[0 1 2]
         [0 1 2]
         [3 4 5]
         [3 4 5]]
        >>>  # case 2 : repeat on axis 1
        >>> input = mindspore.tensor([[0, 1, 2], [3, 4, 5]], mindspore.int32)
        >>> output = mindspore.ops.repeat_interleave(input, repeats=2, axis=1)
        >>> print(output)
        [[0 0 1 1 2 2]
        [3 3 4 4 5 5]]
    """
    if axis is None:
        input = input.reshape(-1)
        axis = 0
    elif not isinstance(axis, int):
        raise TypeError(f"For 'repeat_interleave', the argument 'axis' should be int, but got {type(axis)}.")
    if isinstance(repeats, Tensor):
        repeats = TensorToList()(repeats)
    if not isinstance(repeats, (tuple, list)):
        repeats = (repeats,)
    for index, element in enumerate(repeats):
        if not isinstance(element, int):
            raise TypeError(f"For 'repeat_interleave', each element in {repeats} should be int, but got "
                            f"{type(element)} at index {index}.")

    validator.check_axis_in_range(axis, input.ndim)
    axis = axis + input.ndim if axis < 0 else axis

    if len(repeats) == 1:
        repeats = repeats[0]
        if repeats == 0:
            return Tensor(input.dtype, (0,))
        return repeat_elements(input, repeats, axis=axis)
    size = input.shape[axis]
    if len(repeats) != size:
        raise ValueError(f"For 'repeat_interleave', the length of 'repeats' must be the same as the shape "
                         f"of the original tensor in the 'axis' dimension, but got the length of 'repeats' "
                         f"{len(repeats)}, the shape of the original tensor in the 'axis' dimension {size}.")
    subs = tensor_split(input, size, axis=axis)
    repeated_subs = []
    for sub, rep in zip(subs, repeats):
        if rep != 0:
            repeated_subs.append(repeat_elements(sub, rep, axis=axis))
    return concat(repeated_subs, axis=axis)


def repeat_interleave_ext(input, repeats, dim=None, output_size=None):
    r"""
    Repeat elements of a tensor along an axis, like :func:`mindspore.numpy.repeat`.

    .. warning::
        Only support on Atlas A2 training series.

    Args:
        input (Tensor): The tensor to repeat values for. Must be of type: float16,
            float32, int8, uint8, int16, int32, or int64.
        repeats (Union[int, tuple, list, Tensor]): The number of times to repeat, must be positive.
        dim (int, optional): The dim along which to repeat, Default: ``None``. if dims is None,
            the input Tensor will be flattened and the output will alse be flattened.
        output_size (int, optional): Total output size for the given axis (e.g. sum of repeats),
            Default: ``None``.

    Returns:
        One tensor with values repeated along the specified dim. If input has shape
        :math:`(s1, s2, ..., sn)` and dim is i, the output will have shape :math:`(s1, s2, ...,
        si * repeats, ..., sn)`. The output type will be the same as the type of `input`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([[0, 1, 2], [3, 4, 5]]), mindspore.int32)
        >>> output = ops.function.array_func.repeat_interleave_ext(input, repeats=2, dim=0)
        >>> print(output)
        [[0 1 2]
         [0 1 2]
         [3 4 5]
         [3 4 5]]
    """
    if isinstance(repeats, int):
        return repeat_interleave_int_(input, repeats, dim, output_size)
    return repeat_interleave_tensor_(input, repeats, dim, output_size)


def repeat_elements(x, rep, axis=0):
    """
    Repeat elements of a tensor along an axis, like :func:`mindspore.numpy.repeat` .

    Note:
        It is recommended to use :func:`mindspore.mint.repeat_interleave`, the dimension of input 'x' can support
        a maximum of 8, and get better performance.

    Args:
        x (Tensor): The input tensor.
        rep (int): The number of times to repeat, must be positive.
        axis (int): The axis along which to repeat. Default 0.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1 : repeat on axis 0
        >>> x = mindspore.tensor([[0, 1, 2], [3, 4, 5]], mindspore.int32)
        >>> output = mindspore.ops.repeat_elements(x, rep = 2, axis = 0)
        >>> print(output)
        [[0 1 2]
         [0 1 2]
         [3 4 5]
         [3 4 5]]
        >>> # case 2 : repeat on axis 1
        >>> x = mindspore.tensor([[0, 1, 2], [3, 4, 5]], mindspore.int32)
        >>> output = mindspore.ops.repeat_elements(x, rep = 2, axis = 1)
        >>> print(output)
        [[0 0 1 1 2 2]
         [3 3 4 4 5 5]]
    """
    const_utils.check_type_valid(ops.dtype(x), mstype.number_type, 'input x')
    rep = _check_positive_int(rep, "rep", "repeat_elements")
    axis = _check_is_int(axis, "axis", "repeat_elements")
    x_rank = rank_(x)
    x_rank = _check_rank_range(x_rank, 7, "x", "repeat_elements")
    axis = _check_axis_range(axis, x_rank, "axis", "repeat_elements")
    axis = axis + x.ndim if axis < 0 else axis
    expand_axis = axis + 1
    x_expand = expand_dims(x, expand_axis)
    rep_dims = _cal_repeat_dims(x_rank, rep, expand_axis)
    x_expand = tile_(x_expand, rep_dims)
    x_shape = shape_(x)
    x_reshape = _cal_reshape(x_shape, rep, axis)
    x_rep = reshape_(x_expand, x_reshape)
    return x_rep


def sequence_mask(lengths, maxlen=None):
    """
    Returns a mask tensor representing the first N positions of each cell.

    If `lengths` has shape :math:`(d_1, d_2, ..., d_n)`, then the resulting tensor mask has type and shape
    :math:`(d_1, d_2, ..., d_n, maxlen)`, with mask :math:`[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])`.

    Args:
        lengths (Tensor): Tensor to calculate the mask for. All values in this tensor should be
            less than or equal to `maxlen`. Values greater than `maxlen` will be treated as `maxlen`.
        maxlen (int): size of the last dimension of returned tensor. Must be positive and same
            type as elements in `lengths`. Default ``None`` .

    Returns:
        Tensor, shape is `lengths.shape + (maxlen,)` .

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: When maxlen is assigned
        >>> x = mindspore.tensor([1, 2, 3, 4])
        >>> output = mindspore.ops.sequence_mask(x, 5)
        >>> print(output)
        [[ True False False False False]
         [ True  True False False False]
         [ True  True  True False False]
         [ True  True  True  True False]]
        >>> # case 2: When there is 0 in x
        >>> x = mindspore.tensor([[1, 3], [2, 0]])
        >>> output = mindspore.ops.sequence_mask(x, 5)
        >>> print(output)
        [[[ True False False False False]
          [ True  True  True False False]]
         [[ True  True False False False]
          [False False False False False]]]
        >>> # case 3: when the maxlen is not assigned
        >>> x = mindspore.tensor([[1, 3], [2, 4]])
        >>> output = mindspore.ops.sequence_mask(x)
        >>> print(output)
        [[[ True False False False ]
          [ True  True  True False ]]
         [[ True  True False False ]
          [ True  True  True  True ]]]
    """
    const_utils.check_type_valid(
        ops.dtype(lengths), [mstype.int64, mstype.int32], 'lengths')

    if maxlen is None:
        flatten_data = reshape_(lengths, (-1,))
        flatten_data = cast_(flatten_data, mstype.float32)
        _, value = arg_max_with_value_(flatten_data)
        maxlen = cast_(value, mstype.int32)
    else:
        maxlen = _check_positive_int(maxlen, "maxlen", "sequence_mask")
        maxlen = scalar_to_tensor_(maxlen, mstype.int32)

    range_vector = range_(scalar_to_tensor_(0, mstype.int32),
                          maxlen, scalar_to_tensor_(1, mstype.int32))
    mask = expand_dims(lengths, -1)
    result = range_vector < mask
    return result


def top_k(input_x, k, sorted=True):
    r"""
    `top_k` is deprecated, please use `ops.topk` instead.
    """
    top_k_ = _get_cache_prim(P.TopK)(sorted)
    return top_k_(input_x, k)


def gather_ext(input, dim, index):
    r"""
    Gather data from a tensor by indices.

    .. math::
        output[(i_0, i_1, ..., i_{dim}, i_{dim+1}, ..., i_n)] =
        input[(i_0, i_1, ..., index[(i_0, i_1, ..., i_{dim}, i_{dim+1}, ..., i_n)], i_{dim+1}, ..., i_n)]

    .. warning::
        On Ascend, the behavior is unpredictable in the following cases:

        - the value of `index` is not in the range `[-input.shape[dim], input.shape[dim])` in forward;
        - the value of `index` is not in the range `[0, input.shape[dim])` in backward.

    Args:
        input (Tensor): The target tensor to gather values.
        dim (int): the axis to index along, must be in range `[-input.rank, input.rank)`.
        index (Tensor): The index tensor, with int32 or int64 data type. A valid `index` should be:

            - `index.rank == input.rank`;
            - for `axis != dim`, `index.shape[axis] <= input.shape[axis]`;
            - the value of `index` is in range `[-input.shape[dim], input.shape[dim])`.

    Returns:
        Tensor, has the same type as `input` and the same shape as `index`.

    Raises:
        ValueError: If the shape of `index` is illegal.
        ValueError: If `dim` is not in `[-input.rank, input.rank)`.
        ValueError: If the value of `index` is out of the valid range.
        TypeError: If the type of `index` is illegal.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> index = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> output = ops.function.array_func.gather_ext(input, 1, index)
        >>> print(output)
        [[-0.1 -0.1]
         [0.5   0.5]]
    """
    return gather_d_op(input, dim, index)


def max_ext(input, dim=None, keepdim=False):
    """
    Calculates the maximum value along with the given dimension for the input tensor.

    Args:
        input (Tensor): The input tensor, can be any dimension. Complex tensor is not supported for now.
        dim (int, optional): The dimension to reduce. Default: ``None`` .
        keepdim (bool, optional): Whether to reduce dimension, if true, the output will keep same dimension
            with the input, the output will reduce dimension if false. Default: ``False`` .

    Returns:
        Tensor if `dim` is the default value ``None`` , the maximum value of input tensor, with the shape :math:`()` ,
        and same dtype as `input`.

        tuple (Tensor) if `dim` is not the default value ``None`` , tuple of 2 tensors, containing the maximum
        value of the input tensor along the given dimension `dim` and the corresponding index.

        - **values (Tensor)** - The maximum value of input tensor along the given dimension `dim`, with same dtype as
          `input`. If `keepdim` is ``True`` , the shape of output tensors is :math:`(input_1, input_2, ...,
          input_{axis-1}, 1, input_{axis+1}, ..., input_N)` . Otherwise, the shape is :math:`(input_1, input_2, ...,
          input_{axis-1}, input_{axis+1}, ..., input_N)` .
        - **index (Tensor)** - The index for the maximum value of the input tensor along the given dimension `dim`, with
          the same shape as `values`.

    Raises:
        ValueError: If `dim` is the default value ``None`` and `keepdim` is not ``False`` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> y = Tensor(np.array([[0.0, 0.3, 0.4, 0.5, 0.1],
        ...                      [3.2, 0.4, 0.1, 2.9, 4.0]]), mindspore.float32)
        >>> output, index = ops.function.array_func.max_ext(y, 0, True)
        >>> print(output, index)
        [[3.2 0.4 0.4 2.9 4. ]] [[1 1 0 1 1]]
    """
    if dim is None:
        if keepdim is not False:
            raise ValueError(
                f"For 'max', the `keepdim` must be False when the `dim` is None, but got {keepdim}")
        return max_(input)
    argmax_with_value_op = _get_cache_prim(ArgMaxWithValue)(dim, keepdim)
    indices, values = argmax_with_value_op(input)
    return values, indices


def min_ext(input, dim=None, keepdim=False):
    """
    Calculates the minimum value along with the given dimension for the input tensor.

    Args:
        input (Tensor): The input tensor, can be any dimension. Complex tensor is not supported for now.
        dim (int, optional): The dimension to reduce. Default: ``None`` .
        keepdim (bool, optional): Whether to reduce dimension, if true, the output will keep same dimension
            with the input, the output will reduce dimension if false. Default: ``False`` .

    Returns:
        Tensor if `dim` is the default value ``None`` , the minimum value of input tensor, with the shape :math:`()` ,
        and same dtype as `input`.

        tuple (Tensor) if `dim` is not the default value ``None`` , tuple of 2 tensors, containing the minimum value
        of the input tensor along the given dimension `dim` and the corresponding index.

        - **values (Tensor)** - The minimum value of input tensor along the given dimension `dim`, with same dtype as
          `input`. If `keepdim` is ``True`` , the shape of output tensors is :math:`(input_1, input_2, ...,
          input_{axis-1}, 1, input_{axis+1}, ..., input_N)` . Otherwise, the shape is :math:`(input_1, input_2, ...,
          input_{axis-1}, input_{axis+1}, ..., input_N)` .
        - **index (Tensor)** - The index for the minimum value of the input tensor along the given dimension `dim`,
          with the same shape as `values`.

    Raises:
        ValueError: If `dim` is the default value ``None`` and `keepdim` is not ``False`` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), mindspore.float32)
        >>> output, index = ops.function.array_func.min_ext(x, 0, keepdim=True)
        >>> print(output, index)
        [0.0] [0]
    """
    if dim is None:
        if keepdim is not False:
            raise ValueError(
                f"For 'min', the `keepdim` must be False when the `dim` is None, but got {keepdim}")
        return min_(input)
    argmin_with_value_op = _get_cache_prim(ArgMinWithValue)(dim, keepdim)
    indices, values = argmin_with_value_op(input)
    return values, indices


def one_hot_ext(tensor, num_classes):
    r"""
    Computes a one-hot tensor.

    The locations represented by tensor in `tensor` take value `1`, while all
    other locations take value `0`.

    Args:
        tensor (Tensor): A tensor of indices. Tensor of shape :math:`(X_0, \ldots, X_n)`.
            Data type must be int32 or int64.
        num_classes (int): A scalar defining the depth of the one-hot dimension.

    Returns:
        Tensor, one-hot tensor.

    Raises:
        TypeError: If `num_classes` is not an int.
        TypeError: If dtype of `tensor` is not int32 or int64.
        ValueError: If `num_classes` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> from mindspore.ops.function.array_func import one_hot_ext
        >>> tensor = Tensor(np.array([0, 1, 2]), mindspore.int32)
        >>> num_classes = 3
        >>> output = one_hot_ext(tensor, num_classes)
        >>> print(output)
        [[1. 0. 0.]
        [0. 1. 0.]
        [0. 0. 1.]]
    """
    on_value = Tensor(1, dtype=tensor.dtype)
    off_value = Tensor(0, dtype=tensor.dtype)
    return one_hot_ext_impl(tensor, num_classes, on_value, off_value, -1)


def from_numpy(array):
    r"""
    Convert numpy array to Tensor.
    If the data is not C contiguous, the data will be copied to C contiguous, then construct the tensor.
    Otherwise, the tensor will be constructed using this numpy array without copy.

    Args:
        array (numpy.array): The input array.

    Returns:
        Tensor, has the same data type as input array.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> x = np.array([1, 2])
        >>> output = ms.from_numpy(x)
        >>> print(output)
        [1 2]
    """
    return Tensor.from_numpy(array)



def type_as(input, other):
    r"""
    Returns input cast to the type of the with the other.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Note:
        When converting complex numbers to boolean type, the imaginary part of the complex number is not
        taken into account. As long as the real part is non-zero, it returns True; otherwise, it returns False.

    Args:
        input (Tensor): The shape of tensor is :math:`(x_0, x_1, ..., x_R)`.
            The tensor whose data type is to be converted.
        other (Tensor): The shape of tensor is :math:`(x_0, x_1, ..., x_R)`.
            The tensor whose data type is specified.

    Returns:
        Tensor, the shape of tensor is the same as `input`, :math:`(x_0, x_1, ..., x_R)`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `other` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
        >>> input = Tensor(input_np)
        >>> other_np = np.random.randn(2, 3, 4).astype(np.int32)
        >>> other = Tensor(other_np)
        >>> output = ops.type_as(input, other)
        >>> print(output.dtype)
        Int32
        >>> print(output.shape)
        (2, 3, 4, 5)
    """
    return type_as_(input, other)


__all__ = [
    'unique',
    'unique_with_pad',
    'unique_consecutive',
    'eye',
    'matrix_band_part',
    'padding',
    'fill',
    'fills',
    'tile',
    'take',
    'size',
    'ger',
    'ones',
    'ones_like',
    'zeros',
    'zeros_like',
    'zero_',
    'shape',
    'shape_',
    'reverse',
    'reverse_sequence',
    'hamming_window',
    'chunk',
    'full',
    'full_like',
    'dyn_shape',
    'rank',
    'arange',
    'range',
    'reshape',
    'reshape_',
    'flatten',
    'tensor_slice',
    'strided_slice',
    'slice',
    'slice_scatter',
    'select_scatter',
    'cat',
    'concat',
    'stack',
    'unbind',
    'unstack',
    'is_tensor',
    'scalar_cast',
    'scalar_to_array',
    'scalar_to_tensor',
    'space_to_batch_nd',
    'batch_to_space_nd',
    'tuple_to_array',
    'expand_dims',
    'squeeze',
    'unsqueeze',
    'transpose',
    'scatter_nd',
    'scatter_nd_add',
    'scatter_nd_sub',
    'scatter_nd_mul',
    'scatter_nd_div',
    'scatter_nd_max',
    'scatter_nd_min',
    'tensor_scatter_add',
    'tensor_scatter_sub',
    'tensor_scatter_mul',
    'tensor_scatter_div',
    'tensor_scatter_max',
    'tensor_scatter_min',
    'tensor_scatter_elements',
    'scatter',
    'unsorted_segment_min',
    'unsorted_segment_max',
    'unsorted_segment_prod',
    'gather',
    'gather_d',
    'gather_elements',
    'gather_nd',
    'one_hot',
    'masked_fill',
    'masked_select',
    'where',
    'narrow',
    'ravel',
    'scatter_add',
    'scatter_mul',
    'scatter_max',
    'scatter_min',
    'scatter_div',
    'scatter_update',
    'select',
    'tril',
    'triu',
    'nonzero',
    'is_nonzero',
    'matrix_diag',
    'matrix_diag_part',
    'matrix_set_diag',
    'diag',
    'diagflat',
    'meshgrid',
    'affine_grid',
    'meshgrid',
    'broadcast_to',
    'col2im',
    'split',
    'tensor_split',
    'vsplit',
    'hsplit',
    'dsplit',
    'index_fill',
    'index_select',
    'max',
    'argmax',
    'min',
    'unsorted_segment_sum',
    'population_count',
    'topk',
    'expand',
    'fold',
    'unfold',
    'diagonal',
    'diagonal_scatter',
    'lstsq',
    'mvlgamma',
    'swapaxes',
    'swapdims',
    'searchsorted',
    'argsort',
    'sequence_mask',
    'repeat_elements',
    'repeat_interleave',
    'argwhere',
    'column_stack',
    'hstack',
    'movedim',
    'moveaxis',
    'aminmax',
    'sort',
    'top_k',
    'deepcopy',
    'flip',
    'view_as',
    'type_as',
    'expand_as',
]
__all__.sort()
