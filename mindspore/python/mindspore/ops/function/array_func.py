# Copyright 2022 Huawei Technologies Co., Ltd
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
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops.primitive import constexpr
from mindspore.ops.primitive import _primexpr
import mindspore.ops.function as ops
from mindspore.ops import functional as F
from mindspore.ops.operations._inner_ops import DynamicBroadcastTo
from mindspore.ops.composite.multitype_ops import _constexpr_utils as const_utils

from mindspore.ops.operations.array_ops import (
    UniqueConsecutive,
    SearchSorted,
    NonZero,
    MatrixDiagV3,
    MatrixDiagPartV3,
    MatrixSetDiagV3,
    Col2Im,
    ArgMaxWithValue,
    ArgMinWithValue,
    ScatterNdMax,
    ScatterNdMul,
    IndexFill,
    AffineGrid,
    Im2Col,
    Expand,
    Lstsq,
    Mvlgamma,
    CountNonZero,
    Tril,
    Argmax
)
from mindspore.ops.operations.array_ops import TensorScatterElements
from mindspore.common import Tensor
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore._c_expression import Tensor as Tensor_

eye_ = P.Eye()
fill_ = P.Fill()
ones_ = P.Ones()
ones_like_ = P.OnesLike()
tile_ = P.Tile()
unique_with_pad_ = P.UniqueWithPad()
size_ = P.Size()
shape_ = P.Shape()
rank_ = P.Rank()
tensor_shape_ = P.TensorShape()
reshape_ = P.Reshape()
tensor_slice = P.Slice()
expand_dims_ = P.ExpandDims()
transpose_ = P.Transpose()
scatter_add_ = P.ScatterAdd()
scatter_max_ = P.ScatterMax()
scatter_min_ = P.ScatterMin()
scatter_mul_ = P.ScatterMul()
scatter_div_ = P.ScatterDiv()
scatter_nd_ = P.ScatterNd()
gather_ = P.Gather()
gather_d_ = P.GatherD()
gather_nd_ = P.GatherNd()
nonzero_ = NonZero()
scalar_cast_ = P.ScalarCast()
tensor_scatter_add_ = P.TensorScatterAdd()
tensor_scatter_sub_ = P.TensorScatterSub()
tensor_scatter_mul_ = P.TensorScatterMul()
tensor_scatter_div_ = P.TensorScatterDiv()
tensor_scatter_min_ = P.TensorScatterMin()
tensor_scatter_max_ = P.TensorScatterMax()
scalar_to_tensor_ = P.ScalarToTensor()
tuple_to_array_ = P.TupleToArray()
masked_select_ = P.MaskedSelect()
matrix_band_part_ = P.array_ops.MatrixBandPart()
ger_ = P.Ger()
diag_ = P.Diag()
range_ = P.Range()
zeros_like_ = P.ZerosLike()
cast_ = P.Cast()
tensor_select_ = P.Select()
index_fill_ = IndexFill()
unsorted_segment_sum_ = P.UnsortedSegmentSum()
population_count_ = P.PopulationCount()


@_primexpr
def get_x_shape(x_shape):
    if F.is_sequence_shape_unknown(x_shape):
        return (-2,)
    if F.is_sequence_value_unknown(x_shape):
        return (-1,)
    s = 1
    for i in x_shape:
        s = s * i
    return (s,)


@constexpr
def _check_attr_dtype(param_name, input_dtype, allow_dtypes, cls_name):
    validator.check_value_type(param_name, input_dtype, allow_dtypes, cls_name)


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
    return type(x)


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
    type_map_reverse = {'3': mstype.float64, '2': mstype.float32, '1': mstype.int64, '0': mstype.int32}
    type_level = [type_map.get(i) for i in arg_type_map]
    max_level = builtins.max(type_level)
    return type_map_reverse.get(max_level)


def arange(start=0, end=None, step=1, *, dtype=None):
    r"""
    Creates a sequence of numbers that begins at `start` and extends by increments of
    `step` up to but not including `end`.

    Args:
        start (Union[float, int, Tensor], optional): The first number in the sequence.
            If Tensor, the shape must be (). Default: 0.
        end (Union[float, int, Tensor], optional): Upper or lower limit of the sequence, exclusive.
            If Tensor, the shape must be ().
            Default: None. If None, it defaults to the value of `start`, and 0 is used as the starting value.
        step (Union[float, int, Tensor], optional): Number that increments `start`.
            If Tensor, the shape must be (). Default: 1.

    Keyword Args:
        dtype (mindspore.dtype, optional): The required data type of returned Tensor. Default: None.
            If the value is not specified or is None, the type with the highest precision in the
            `start`, `end`, and `step` parameters is inferred.

    Returns:
        A 1-D Tensor, with the same type as the inputs.

    Raises:
        TypeError: If `start`, `end` or `step` is not an int or a float or a TensorScalar(Special Tensor with shape ())
                   in valid dtypes.
        ValueError: If `step` = 0.
        ValueError: If `start` >= `end` when `step` > 0.
        ValueError: If `start` <= `end` when `step` < 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor, ops
        >>> output = ops.arange(1, 6)
        >>> print(output)
        [1 2 3 4 5]
        >>> print(output.dtype)
        Int64
        >>> output = ops.arange(0, 3, 1.2)
        >>> print(output)
        [0.  1.2 2.4]
        >>> print(output.dtype)
        Float32
        >>> output = ops.arange(7, 1, -2)
        >>> print(output)
        [7 5 3]
        >>> print(output.dtype)
        Int64
        >>> output = ops.arange(ms.Tensor(12.0, dtype=ms.float64), 2, ms.Tensor(-1.0, dtype=ms.float32))
        >>> print(output)
        [12. 11. 10.  9.  8.  7.  6.  5.  4.  3.]
        >>> print(output.dtype)
        Float64
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
    data = P.Range()(start, end, step)
    if dtype is not None:
        data = cast_(data, dtype)
    return data


def cat(tensors, axis=0):
    r"""
    Connect input tensors along with the given axis.

    The input data is a tuple of tensors. These tensors have the same rank :math:`R`. Set the given axis as :math:`m`,
    and :math:`0 \le m < R`. Set the number of input tensors as :math:`N`. For the :math:`i`-th tensor :math:`t_i`,
    it has the shape of :math:`(x_1, x_2, ..., x_{mi}, ..., x_R)`. :math:`x_{mi}` is the :math:`m`-th dimension of the
    :math:`t_i`. Then, the shape of the output tensor is

    .. math::

        (x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)

    Args:
        tensors (Union[tuple, list]): A tuple or a list of input tensors.
            Suppose there are two tensors in this tuple or list, namely t1 and t2.
            To perform `concat` in the axis 0 direction, except for the :math:`0`-th axis,
            all other dimensions should be equal, that is,
            :math:`t1.shape[1] = t2.shape[1], t1.shape[2] = t2.shape[2], ..., t1.shape[R-1] = t2.shape[R-1]`,
        axis (int): The specified axis, whose value is in range :math:`[-R, R)`. Default: 0.

    Returns:
        Tensor, the shape is :math:`(x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)`.
            The data type is the same with `tensors`.

    Raises:
        TypeError: If `axis` is not an int.
        ValueError: If `tensors` have different dimension of tensor.
        ValueError: If `axis` not in range :math:`[-R, R)`.
        RuntimeError: If tensor's shape in `tensors` except for `axis` are different.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x1 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
        >>> input_x2 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
        >>> output = ops.cat((input_x1, input_x2))
        >>> print(output)
        [[0. 1.]
         [2. 1.]
         [0. 1.]
         [2. 1.]]
        >>> output = ops.cat((input_x1, input_x2), 1)
        >>> print(output)
        [[0. 1. 0. 1.]
         [2. 1. 2. 1.]]
    """
    _concat = _get_cache_prim(P.Concat)(axis)
    return _concat(tensors)


def eye(n, m=None, dtype=None):
    """
    Creates a tensor with ones on the diagonal and zeros in the rest.

    Note:
        Combines ReverseV2 operator to get an anti-diagonal Tensor,
        but ReverseV2 only supports Ascend and GPU platforms currently.

    Args:
        n (int): The number of rows of returned tensor. Constant value only.
        m (int): The number of columns of returned tensor. Constant value only.
            Default: if None, the number of columns is as the same as n.
        dtype (mindspore.dtype): MindSpore's dtype, the data type of the returned tensor.
            The data type can be Number. Default: if None, the data type of the returned tensor is mindspore.float32.

    Returns:
        Tensor, a tensor with ones on the diagonal and the rest of elements are zero. The shape of `output` depends on
        the user's Inputs `n` and `m`. And the data type depends on Inputs `dtype`.

    Raises:
        TypeError: If `m` or `n` is not an int.
        ValueError: If `m` or `n` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = ops.eye(2, 2, mindspore.int32)
        >>> print(output)
        [[1 0]
         [0 1]]
        >>> print(output.dtype)
        Int32
        >>> output = ops.eye(1, 2, mindspore.float64)
        >>> print(output)
        [[1. 0.]]
        >>> print(output.dtype)
        Float64
        >>> output = ops.eye(2, dtype=mindspore.int32)
        >>> print(output)
        [[1 0]
         [0 1]]
        >>> print(output.dtype)
        Int32
        >>> output = ops.eye(2)
        >>> print(output)
        [[1. 0.]
         [0. 1.]]
        >>> print(output.dtype)
        Float32
    """
    if m is None:
        m = n
    if dtype is None:
        dtype = ms.float32
    return eye_(n, m, dtype)


def hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, *, dtype=None):
    r"""
    Returns the Hamming window.

    .. math::

        w[n]=\alpha − \beta \cos \left( \frac{2 \pi n}{N - 1} \right),

    where N is the full window size.

    Args:
        window_length (int): The size of returned window. Must be a non negative integer.
        periodic (bool, optional): If True, return a periodic window. If False, return a symmetric window.
        alpha (float, optional): The coefficient α.
        beta (float, optional): The coefficient β.

    Keyword Args:
        dtype (mindspore.dtype, optional): The output window data type. Default: None.

    Returns:
        Tensor, a 1-D tensor of size (window_length) containing the window.

    Raises:
        TypeError: If `window_length` is a negative integer.
        TypeError: If `periodic` is not bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> print(ops.hamming_window(6, False))
        [0.08 0.39785218 0.91214782  0.91214782  0.39785218 0.08]
    """
    if not isinstance(window_length, int):
        raise TypeError(f"For array function 'hamming_window', 'window_length' must be int, but got" \
            f" {type(window_length)}.")
    if window_length < 0:
        raise ValueError(f"For array function 'hamming_window', 'window_length' must be non negative number.")
    if not isinstance(periodic, bool):
        raise TypeError(f"For array function 'hamming_window', 'periodic' must be bool, but got {type(periodic)}.")
    if not isinstance(alpha, float):
        raise TypeError(f"For array function 'hamming_window', 'alpha' must be float, but got {type(alpha)}.")
    if not isinstance(beta, float):
        raise TypeError(f"For array function 'hamming_window', 'beta' must be float, but got {type(beta)}.")
    if window_length <= 1:
        return Tensor(np.ones(window_length))
    if dtype is not None and dtype not in mstype.float_type:
        raise TypeError(f"For array function 'hamming_window', 'dtype' must be floating point dtypes, but got {dtype}.")

    if periodic:
        window_length += 1
    n = arange(0, window_length)
    w = alpha - beta * ops.cos((2 * np.pi / (window_length - 1)) * n)

    if dtype:
        w = P.Cast()(w, dtype)
    return w[:-1] if periodic else w


def where(condition, x, y):
    r"""
    Selects elements from `x` or `y` based on `condition` and returns a tensor.

    .. math::
        output_i = \begin{cases} x_i,\quad &if\ condition_i \\ y_i,\quad &otherwise \end{cases}

    Args:
        condition (Union[Bool Tensor, bool, scalar]): If True, yield `x`, otherwise yield `y`.
        x (Union[Tensor, Scalar]): When `condition` is True, values to select from.
        y (Union[Tensor, Scalar]): When `condition` is Fasle, values to select from.

    Returns:
        Tensor, elements are selected from `x` and `y`.

    Raises:
        ValueError: If `condition` can not be broadcast to the shape of `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = Tensor(np.arange(4).reshape((2, 2)), mstype.float32)
        >>> b = Tensor(np.ones((2, 2)), mstype.float16)
        >>> condition = a < 3
        >>> output = ops.where(condition, a, b)
        >>> print(output)
        [[0. 1.]
         [2. 1.]]
    """
    not_op = P.LogicalNot()
    x_mask = cast_(condition, mstype.bool_)
    y_mask = not_op(x_mask)
    if x_mask.all():
        output = x
    elif not x_mask.any():
        output = y
    else:
        output = x * x_mask + y * y_mask
    return output


def reverse(x, axis):
    """
    Reverses specific dimensions of a tensor.

    .. warning::
        The value range of "axis" is [-dims, dims - 1]. "dims" is the dimension length of "input_x".

    Args:
        x (Tensor): The target tensor. The data type is Number except float64.
            The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
        axis (Union[tuple(int), list(int)]): The indices of the dimensions to reverse.

    Outputs:
        Tensor, has the same shape and type as `x`.

    Raises:
        TypeError: If `axis` is neither list nor tuple.
        TypeError: If element of `axis` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), mindspore.int32)
        >>> output = ops.reverse(input_x, axis=[1])
        >>> print(output)
        [[4 3 2 1]
         [8 7 6 5]]
        >>> input_x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), mindspore.int32)
        >>> output = ops.reverse(input_x, axis=[1, 0])
        >>> print(output)
        [[8 7 6 5]
         [4 3 2 1]]
    """
    return P.ReverseV2(axis)(x)


def ravel(x):
    """
    Return a contiguous flattened tensor.

    Args:
        x (Tensor): A tensor to be flattened.

    Returns:
        Tensor, a 1-D tensor, containing the same elements of the input.

    Raises:
        TypeError: If argument `x` is not Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
        >>> output = ops.ravel(x)
        >>> print(output)
        [0. 1. 2. 1.]
        >>> print(output.shape)
        (4,)
    """
    return ops.reshape(x, (-1,))


def matrix_band_part(x, lower, upper):
    r"""
    Copy a tensor setting everything outside a central band in each innermost matrix to zero.

    Args:
        x (Tensor): Input tensor. :math:`(*, m, n)` where :math:`*` means, any number of additional dimensions.
            The data type must be float16, float32, float64, int32 or int64.
        lower (Union[int, Tensor]): Number of subdiagonals to keep. The data type must be int32 or int64.
            If negative, keep entire lower triangle.
        upper (Union[int, Tensor]): Number of superdiagonals to keep. The data type must be int32 or int64.
            If negative, keep entire upper triangle.

    Returns:
        Tensor, has the same type and shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not one of float16, float32, float64, int32 or int64.
        TypeError: If `lower` is neither a number nor a Tensor.
        TypeError: If `upper` is neither a number nor a Tensor.
        TypeError: If dtype of `lower` is neither int32 nor int64.
        TypeError: If dtype of `upper` is neither int32 nor int64.
        ValueError: If the shape of `x` is not greater than or equal to 2D.
        ValueError: If the shape of `lower` is not equal to 0D.
        ValueError: If the shape of `upper` is not equal to 0D.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.ones([2, 4, 4]).astype(np.float32))
        >>> output = ops.matrix_band_part(x, 2, 1)
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


def padding(x, pad_dim_size=8):
    r"""
    Extends the last dimension of the input tensor from 1 to pad_dim_size, by filling with 0.

    Args:
        x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. The rank of `x` must be at least 2.
            The last dimension of `x` must be 1. The data type is Number.
        pad_dim_size (int): The value of the last dimension of `x` to be extended, which must be positive. Default: 8.

    Returns:
        Tensor, has the same type and shape as input shape value.

    Raises:
        TypeError: If `pad_dim_size` is not an int.
        ValueError: If `pad_dim_size` is less than 1.
        ValueError: If last dim of `x` is not equal to 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
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
                raise TypeError(f"For {ops_name}, each axis must be integer, but got {type(ax)} in {axis}.")
        return True

    type_str = ""
    if type_int:
        type_str += "int, "
    if type_tuple:
        type_str += "tuple, "
    if type_list:
        type_str += "list, "
    raise TypeError(f"For {ops_name}, the axis should be {type_str}, but got {type(axis)}.")


def one_hot(indices, depth, on_value, off_value, axis=-1):
    r"""
    Computes a one-hot tensor.

    The locations represented by indices in `indices` take value `on_value`, while all
    other locations take value `off_value`.

    Note:
        If the input indices is rank `N`, the output will have rank `N+1`. The new axis is created at dimension `axis`.

    Args:
        indices(Tensor): A tensor of indices. Tensor of shape :math:`(X_0, \ldots, X_n)`.
            Data type must be uint8, int32 or int64.
        depth(int): A scalar defining the depth of the one-hot dimension.
        on_value(Union[Tensor, int, float]): A value to fill in output when `indices[j] = i`.
            Support uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float32, float64,
            bool, complex64, complex128.
        off_value(Union[Tensor, int, float]): A value to fill in output when `indices[j] != i`.
            Has the same data type as `on_value`.
        axis(int): Position to insert the value. e.g. If shape of `self` is :math:`(N, C)`, and `axis` is -1,
            the output shape will be :math:`(N, C, D)`, If `axis` is 0, the output shape will be :math:`(D, N, C)`.
            Default: -1.

    Returns:
        Tensor, one-hot tensor. Tensor of shape :math:`(X_0, \ldots, X_{axis}, \text{depth} ,X_{axis+1}, \ldots, X_n)`.

    Raises:
        TypeError: If `axis` or `depth` is not an int.
        TypeError: If dtype of `indices` is not uint8, int32 or int64.
        TypeError: If `indices`, `on_value` or `off_value` is not a Tensor.
        ValueError: If `axis` is not in range [-1, ndim].
        ValueError: If `depth` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor(np.array([0, 1, 2]), mindspore.int32)
        >>> depth, on_value, off_value = 3, Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32)
        >>> output = ops.one_hot(indices, depth, on_value, off_value, axis=-1)
        >>> print(output)
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
    """
    if not isinstance(on_value, Tensor):
        on_value = Tensor(on_value)
    if not isinstance(off_value, Tensor):
        off_value = Tensor(off_value)
    onehot = _get_cache_prim(P.OneHot)(axis)
    return onehot(indices, depth, on_value, off_value)


def fill(type, shape, value):  # pylint: disable=redefined-outer-name
    """
    Create a Tensor of the specified shape and fill it with the specified value.

    Args:
        type (mindspore.dtype): The specified type of output tensor. The data type only supports
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ and
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ .
        shape (Union(Tensor, tuple[int])): The specified shape of output tensor.
        value (Union(Tensor, number.Number, bool)): Value to fill the returned tensor.

    Returns:
        Tensor.

    Raises:
        TypeError: If `shape` is not a tuple or a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = ops.fill(mindspore.float32, (2, 2), 1)
        >>> print(output)
        [[1. 1.]
         [1. 1.]]
        >>> output = ops.fill(mindspore.float32, (3, 3), 0)
        >>> print(output)
        [[0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]]
    """
    return fill_(type, shape, value)


def full(size, fill_value, *, dtype=None): # pylint: disable=redefined-outer-name
    """
    Create a Tensor of the specified shape and fill it with the specified value.

    Args:
        size (Union(tuple[int], list[int])): The specified shape of output tensor.
        fill_value (number.Number): Value to fill the returned tensor. Complex numbers are not supported for now.

    Keyword Args:
        dtype (mindspore.dtype): The specified type of output tensor. `bool_` and `number` are supported, for details,
            please refer to :class:`mindspore.dtype` . Default: None.

    Returns:
        Tensor.

    Raises:
        TypeError: If `size` is not a tuple or list.
        ValueError: The element in `size` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = ops.full((2, 2), 1)
        >>> print(output)
        [[1. 1.]
         [1. 1.]]
        >>> output = ops.full((3, 3), 0)
        >>> print(output)
        [[0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]]
    """
    if not isinstance(size, (list, tuple)):
        raise TypeError(f"For 'ops.full', 'size' must be a tuple or list of ints, but got {type(size)}.")
    if dtype is None:
        dtype = mstype.int64
    if dtype not in mstype.all_types:
        raise TypeError(f"For 'ops.full', 'dtype' must be mindspore.type, but got {dtype}.")
    if isinstance(size, list):
        size = tuple(size)
    return fill_(dtype, size, fill_value)


def full_like(x, fill_value, *, dtype=None):
    """
    Return a Tensor of the same shape as `x` and filled with `fill_value`.

    Args:
        x (Tensor): input Tensor and the output Tensor have the same shape as `x`.
        fill_value (number.Number): Value to fill the returned Tensor. Complex numbers are not supported for now.

    Keyword Args:
        dtype (mindspore.dtype, optional): The specified type of output tensor. `bool_` and `number` are supported,
            for details, please refer to :class:`mindspore.dtype` . Default: None.

    Returns:
        Tensor.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input = Tensor([[0, 1], [2, 1]], dtype=mindspore.int32)
        >>> output = ops.full_like(input, 1)
        >>> print(output)
        [[1. 1.]
         [1. 1.]]
        >>> input = Tensor([[0, 1, 1], [2, 1, 2], [1, 3, 4]], dtype=mindspore.int32)
        >>> output = ops.full_like(input, 0)
        >>> print(output)
        [[0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]]
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"For ops.full_like, the argument 'x' must be tensor, but got {type(x)}")
    if dtype is None:
        dtype = x.dtype
    return full(x.shape, fill_value, dtype=dtype)


def chunk(x, chunks, axis=0):
    """
    Cut the input Tensor into `chunks` sub-tensors along the specified axis.

    Note:
        This function may return less then the specified number of chunks!

    Args:
        x (Tensor): A Tensor to be cut.
        chunks (int): Number of sub-tensors to cut.
        axis (int): Specify the dimensions that you want to split. Default: 0.

    Returns:
        A tuple of sub-tensors.

    Raises:
        TypeError: If argument `x` is not Tensor.
        TypeError: The sum of `chunks` is not int.
        TypeError: If argument `axis` is not int.
        ValueError: If argument `axis` is out of range of :math:`[-x.ndim, x.ndim)` .
        ValueError: If argument `chunks` is not positive number.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = np.arange(9).astype("float32")
        >>> output = ops.chunk(Tensor(input_x), 3)
        >>> print(output)
        (Tensor(shape=[3], dtype=Float32, value= [ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value= [ 3.00000000e+00,  4.00000000e+00,  5.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value= [ 6.00000000e+00,  7.00000000e+00,  8.00000000e+00]))
    """
    if not isinstance(x, Tensor):
        raise TypeError(f'For ops.chunk parameter `x` must be Tensor, but got {type(x)}')
    _check_axis_type(axis, True, False, False, "ops.chunk")
    axis = _canonicalize_axis(axis, x.ndim)

    if not isinstance(chunks, int):
        raise TypeError(f"For ops.chunk type of argument `chunks` should be integer, but got {type(chunks)}")
    if chunks <= 0:
        raise ValueError(f"For ops.chunk parameter 'chunks' must be greater than 0, but got {chunks}")

    arr_shape = x.shape
    length_along_dim = arr_shape[axis]

    if chunks > length_along_dim:
        res = P.Split(axis, length_along_dim)(x)
    elif length_along_dim % chunks == 0:
        res = P.Split(axis, chunks)(x)
    else:
        block_size = int(np.ceil(length_along_dim / chunks))
        true_chunks = int(length_along_dim // block_size)
        length1 = true_chunks * block_size
        length2 = length_along_dim - length1
        start1 = _list_comprehensions(rank(x), 0, True)
        size1 = _tuple_setitem(arr_shape, axis, length1)
        start2 = _tuple_setitem(start1, axis, length1)
        size2 = _tuple_setitem(arr_shape, axis, length2)
        res = P.Split(axis, true_chunks)(tensor_slice(x, start1, size1))
        if length2:
            res += P.Split(axis, 1)(tensor_slice(x, start2, size2))
    return res


def ones(shape, dtype=None):  # pylint: disable=redefined-outer-name
    r"""
    Creates a tensor filled with value ones.

    Creates a tensor with shape described by the first argument and fills it with value ones in type of the second
    argument.

    Args:
        shape (Union[tuple[int], int]): The specified shape of output tensor. Only constant positive int is allowed.
        dtype (:class:`mindspore.dtype`): The specified type of output tensor. If `dtype` is None,
            `mindspore.float32` will be used. Default: None.

    Returns:
        Tensor, has the same type and shape as input shape value.

    Raises:
        TypeError: If `shape` is neither tuple nor int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = ops.ones((2, 2), mindspore.float32)
        >>> print(output)
        [[1. 1.]
         [1. 1.]]
    """
    _dtype = mstype.float32 if dtype is None else dtype
    ones_op = P.FillV2()
    value = Tensor(1, _dtype)
    if isinstance(shape, int):
        shape = tuple([shape])
    shape_tensor = shape
    if isinstance(shape, (list, tuple)) and not shape:
        shape_tensor = Tensor(shape, dtype=mstype.int64)
    elif not isinstance(shape, Tensor):
        shape_tensor = Tensor(shape)
    if shape_tensor.ndim == 0 and shape_tensor.size == 1:
        shape_tensor = shape_tensor.reshape(1)
    output = ones_op(shape_tensor, value)
    return output


def ones_like(input_x):
    """
    Returns a Tensor with a value of 1 and its shape and data type is the same as the input.

    Args:
        input_x (Tensor): Tensor of any dimension.

    Returns:
        Tensor, has the same shape and type as `input_x` but filled with ones.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))
        >>> output = ops.ones_like(input_x)
        >>> print(output)
        [[1 1]
         [1 1]]
    """
    ones_like_op = P.OnesLike()
    output = ones_like_op(input_x)
    return output


def zeros(shape, dtype=None):  # pylint: disable=redefined-outer-name
    r"""
    Creates a tensor filled with 0  with shape described by `shape` and fills it with value 0 in type of `dtype`.

    Args:
        shape (Union[tuple[int], int]): The specified shape of output tensor. Only constant positive int is allowed.
        dtype (:class:`mindspore.dtype`, optional): The specified type of output tensor. If `dtype` is None,
            mindspore.float32 will be used. Default: None.

    Returns:
        Tensor, has the same dtype and shape as input.

    Raises:
        TypeError: If `shape` is neither a tuple of int nor an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = ops.zeros((2, 2), mindspore.float32)
        >>> print(output)
        [[0. 0.]
         [0. 0.]]
    """
    zero_op = P.FillV2()
    _dtype = mstype.float32 if dtype is None else dtype
    value = Tensor(0, _dtype)
    if isinstance(shape, int):
        shape = tuple([shape])
    shape_tensor = shape
    if isinstance(shape, (list, tuple)) and not shape:
        shape_tensor = Tensor(shape, dtype=mstype.int64)
    elif not isinstance(shape, Tensor):
        shape_tensor = Tensor(shape)
    if shape_tensor.ndim == 0 and shape_tensor.size == 1:
        shape_tensor = shape_tensor.reshape(1)
    output = zero_op(shape_tensor, value)
    return output


def zeros_like(x, *, dtype=None):
    r"""
    Creates a tensor filled with 0, with the same size as x, and the given dtype.

    If `dtype = None`, the tensor will have the same dtype as input `x`.

    Args:
        x (Tensor): Tensor of any dimension.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The specified dtype of the output tensor. If `dtype` is None,
            the dtype of the input tensor will be used. Default: None.

    Returns:
        Tensor, filled with 0.

    Raises:
        TypeError: If dtype is not a MindSpore dtype.

    Examples:
        >>> x = Tensor(np.arange(4).reshape(2, 2))
        >>> output = ops.zeros_like(x, dtype=mindspore.float32)
        >>> print(output)
        [[0. 0.]
         [0. 0.]]
    """
    _dtype = x.dtype if dtype is None else dtype
    zeros_like_op = P.ZerosLike()
    output = zeros_like_op(x)
    output = cast_(output, _dtype)
    return output


def tile(input_x, multiples):
    r"""
    Replicates an input tensor with given multiples times.

    Creates a new tensor by replicating `input_x` `multiples` times. The i'th dimension of
    output tensor has `input_x.shape[i] * multiples[i]` elements, and the values of `input_x`
    are replicated `multiples[i]` times along the i'th dimension.

    Note:
        The length of `multiples` must be greater or equal to the length of dimension in `input_x`.

    Args:
        input_x (Tensor): 1-D or higher dimensional Tensor. Set the shape of input tensor as
            :math:`(x_1, x_2, ..., x_S)` .

        multiples (tuple[int]): The parameter that specifies the number of replications,
            the parameter type is tuple, and the data type is int, i.e., :math:`(y_1, y_2, ..., y_S)`.
            The length of `multiples` cannot be smaller than the length of the shape of `input_x`.
            Only constant value is allowed.

    Returns:
        Tensor, has the same data type as the `input_x`. Suppose the length of `multiples` is `d`,
        the dimension of `input_x` is `input_x.dim`, and the shape of `input_x` is :math:`(x_1, x_2, ..., x_S)`.

        - If `input_x.dim = d`, then the shape of their corresponding positions can be multiplied, and
          the shape of Outputs is :math:`(x_1*y_1, x_2*y_2, ..., x_S*y_R)`.
        - If `input_x.dim < d`, fill in multiple 1 in the length of the shape of `input_x` until their
          lengths are consistent. Such as set the shape of `input_x` as :math:`(1, ..., x_1, x_2, ..., x_S)`,
          then the shape of their corresponding positions can be multiplied, and the shape of Outputs is
          :math:`(1*y_1, ..., x_S*y_R)`.

    Raises:
        TypeError: If `multiples` is not a tuple or its elements are not all int.
        ValueError: If the elements of `multiples` are not all greater than 0.
        ValueError: If the length of `multiples` are smaller than the length of dimension in `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[1, 2], [3, 4]]), mindspore.float32)
        >>> multiples = (2, 3)
        >>> output = ops.tile(input_x, multiples)
        >>> print(output)
        [[1.  2.  1.  2.  1.  2.]
         [3.  4.  3.  4.  3.  4.]
         [1.  2.  1.  2.  1.  2.]
         [3.  4.  3.  4.  3.  4.]]
        >>> multiples = (2, 3, 2)
        >>> output = ops.tile(input_x, multiples)
        >>> print(output)
        [[[1. 2. 1. 2.]
          [3. 4. 3. 4.]
          [1. 2. 1. 2.]
          [3. 4. 3. 4.]
          [1. 2. 1. 2.]
          [3. 4. 3. 4.]]
         [[1. 2. 1. 2.]
          [3. 4. 3. 4.]
          [1. 2. 1. 2.]
          [3. 4. 3. 4.]
          [1. 2. 1. 2.]
          [3. 4. 3. 4.]]]
    """
    return tile_(input_x, multiples)


def range(start, end, step):
    r"""
    Creates a sequence of numbers that begins at `start` and extends by increments of
    `limit` up to but not including `end`.

    The types of all 3 inputs must be the same. The type of the resulting tensor is
    the same as the type of the inputs.

    Args:
        start (Tensor): A scalar Tensor. The first number in the sequence. Must have
          type: int32 ,int64, float32 or float64.
        end (Tensor): A scalar Tensor. Upper limit of the sequence, exclusive. Must
          have type: int32 ,int64, float32 or float64.
        step (Tensor): A scalar Tensor. Number that increments `start`. Must have
          type: int32 ,int64, float32 or float64.

    Returns:
        A 1-D Tensor, with the same type as the inputs.

    Raises:
        TypeError: If `start`, `end` or `step` is not scalar Tensor.
        TypeError: If datatype of `start`, `end` or `step` is not same.
        TypeError: If datatype of `start`, `end` or `step` is not supported.
        ValueError: If `step` = 0.
        ValueError: If `start` >= `end` when `step` > 0.
        ValueError: If `start` <= `end` when `step` < 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> start = Tensor(0, mstype.int32)
        >>> end = Tensor(10, mstype.int32)
        >>> step = Tensor(4, mstype.int32)
        >>> output = ops.range(start, end, step)
        >>> print(output)
        [0 4 8]
    """
    return range_(start, end, step)


##############################
# Tensor Operation Functions.
##############################


def unique(x):
    """
    Returns the unique elements of input tensor and also return a tensor containing the index of each value of input
    tensor corresponding to the output unique tensor.

    The output contains Tensor `y` and Tensor `idx`, the format is probably similar to (`y`, `idx`).
    The shape of Tensor `y` and Tensor `idx` is different in most cases, because Tensor `y` will be deduplicated,
    and the shape of Tensor `idx` is consistent with the input.

    To get the same shape between `idx` and `y`, please ref to :class:'mindspore.ops.UniqueWithPad' operator.

    Args:
        x (Tensor): The input tensor.
            The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Returns:
        Tuple, containing Tensor objects (`y`, `idx`), `y` is a tensor with the
        same type as `x`, and contains the unique elements in `x`.
        `idx` is a tensor containing indices of elements in
        the input corresponding to the output tensor, have the same shape with `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> from mindspore import ops
        >>> x = Tensor(np.array([1, 2, 5, 2]), mindspore.int32)
        >>> output = ops.unique(x)
        >>> print(output)
        (Tensor(shape=[3], dtype=Int32, value= [1, 2, 5]), Tensor(shape=[4], dtype=Int32, value= [0, 1, 2, 1]))
        >>> y = output[0]
        >>> print(y)
        [1 2 5]
        >>> idx = output[1]
        >>> print(idx)
        [0 1 2 1]
    """

    unique_op = _get_cache_prim(P.Unique)()
    reshape_op = _get_cache_prim(P.Reshape)()

    shape_x = x.shape
    length_x = get_x_shape(shape_x)
    x = reshape_op(x, length_x)
    y, idx = unique_op(x)
    idx = reshape_op(idx, shape_x)
    return y, idx


def unique_with_pad(x, pad_num):
    """
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
        ValueError: If length of shape of `x` is not equal to 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
    return unique_with_pad_(x, pad_num)


def unique_consecutive(x, return_idx=False, return_counts=False, axis=None):
    """
    Returns the elements that are unique in each consecutive group of equivalent elements in the input tensor.

    Args:
        x (Tensor): The input tensor.
        return_idx (bool, optional): Whether to return the index of where the element in the original input
            maps to the position in the output. Default: False.
        return_counts (bool, optional): Whether to return the counts of each unique element. Default: False.
        axis (int, optional): The dimension to apply unique. If None, the unique of the flattened input is
            returned. If specified, it must be int32 or int64. Default: None.

    Returns:
        A tensor or a tuple of tensors containing tensor objects (`output`, `idx`, `counts`). `output` has the
        same type as `x` and is used to represent the output list of unique scalar elements. If `return_idx` is
        True, there will be an additional returned tensor, `idx`, which has the same shape as `x` and represents
        the index of where the element in the original input maps to the position in the output. If `return_counts`
        is True, there will be an additional returned tensor, `counts`, which represents the number of occurrences
        for each unique value or tensor.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not supported.
        TypeError: If `return_idx` is not a bool.
        TypeError: If `return_counts` is not a bool.
        TypeError: If `axis` is not an int.
        ValueError: If `axis` is not in the range of :math:`[-ndim, ndim-1]`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 1, 2, 2, 3, 1, 1, 2]), mstype.int32)
        >>> output, idx, counts = ops.unique_consecutive(x, True, True, None)
        >>> print(output)
        [1 2 3 1 2]
        >>> print(idx)
        [0 0 1 1 2 3 3 4]
        >>> print(counts)
        [2 2 1 2 1]
    """

    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError("For 'unique_consecutive', 'x' must be Tensor.")
    unique_consecutive_op = _get_cache_prim(UniqueConsecutive)(return_idx, return_counts, axis)
    output, idx, counts = unique_consecutive_op(x)
    if return_idx and return_counts:
        return output, idx, counts
    if return_idx:
        return output, idx
    if return_counts:
        return output, counts
    return output


def searchsorted(sorted_sequence, values, *, out_int32=False, right=False):
    """
    Return the position indices such that after inserting the values into the `sorted_sequence`, the order of innermost
    dimension of the `sorted_sequence` remains unchanged.

    Args:
        sorted_sequence (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R-1, x_R)` or `(x_1)`.
            It must contain a monotonically increasing sequence on the innermost dimension.
        values (Tensor): The value that should be inserted.
            The shape of tensor is :math:`(x_1, x_2, ..., x_R-1, x_S)`.

    Keyword Args:
        out_int32 (bool, optional): Output datatype. If True, the output datatype will be int32;
            if False, the output datatype will be int64. Default: False.
        right (bool, optional): Search Strategy. If True, return the last suitable index found;
            if False, return the first such index. Default: False.

    Returns:
        Tensor containing the indices from the innermost dimension of `sorted_sequence` such that,
        if insert the corresponding value in the `values` tensor, the order of `sorted_sequence` would be preserved,
        whose datatype is int32 if out_int32 is True, otherwise int64, and shape is the same as the shape of `values`.

    Raises:
        ValueError: If the dimension of `sorted_sequence` isn't 1 and all dimensions except the last dimension of
            `sorted_sequence` and `values` are different.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> sorted_sequence = Tensor(np.array([[0, 1, 3, 5, 7], [2, 4, 6, 8, 10]]), mindspore.float32)
        >>> values = Tensor(np.array([[3, 6, 9], [3, 6, 9]]), mindspore.float32)
        >>> output = ops.searchsorted(sorted_sequence, values)
        >>> print(output)
        [[2 4 5]
         [1 2 4]]
    """

    _check_attr_dtype("out_int32", out_int32, [bool], "search_sorted")
    dtype = mstype.int64 if not out_int32 else mstype.int32
    search_sorted_ = SearchSorted(dtype, right)
    return search_sorted_(sorted_sequence, values)


def ger(x1, x2):
    r"""
    Ger product of `x1` and `x2`. Calculate the outer product of two arrays. If `x1` is a 1D Tensor of
    shape :math:`(m,)` and `x2` is a 1D Tensor of shape :math:`(n,)`, then `output` must be a 2D Tensor of shape
    :math:`(m, n)`.

    Note:
        Currently Ascend does not support float64 data input.

    Args:
        x1 (Tensor): input Tensor, with dtype of float16, float32 or float64.
        x2 (Tensor): input Tensor, with dtype of float16, float32 or float64, must have the same dtype as `x1`.

    Returns:
        Tensor, output matrix with the same dtype as inputs. With `x1` shape :math:`(m,)` and
        `x2` shape of :math:`(n,)`, the `output` has shape :math:`(m, n)`.

    Raises:
        TypeError: If `x1` or `x2` is not a 1-D Tensor.
        TypeError: If the dtype of `x1` and `x2` is not float16, float32 or float64.
        TypeError: If the dtype of `x1` and `x2` are not the same.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = Tensor([1., 2., 3., 4.], mindspore.float32)
        >>> x2 = Tensor([1., 2., 3.], mindspore.float32)
        >>> output = ops.ger(x1, x2)
        >>> print(output)
        [[ 1.  2.  3.]
         [ 2.  4.  6.]
         [ 3.  6.  9.]
         [ 4.  8. 12.]]
    """
    return ger_(x1, x2)


def size(input_x):  # pylint: disable=redefined-outer-name
    r"""
    Returns a Scalar of type int that represents the size of the input Tensor and the total number of elements in the
    Tensor.

    Args:
        input_x (Tensor): Input parameters, the shape of tensor is :math:`(x_1, x_2, ..., x_R)`. The data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.

    Returns:
        int. A scalar representing the elements' size of `input_x`, tensor is the number of elements
        in a tensor, :math:`size=x_1*x_2*...x_R`. The data type is an int.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> output = ops.size(input_x)
        >>> print(output)
        4
    """
    return size_(input_x)


def shape(input_x):
    """
    Returns the shape of the input tensor. This operation is used in static shape cases.

    static shape: A shape that can be obtained without running the graph. It is an inherent property of tensor and
    may be unknown. The static shape information can be completed by artificial setting.
    No matter what the input of the graph is, the static shape is not affected.

    Args:
        input_x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Returns:
        tuple[int], the output tuple is constructed by multiple integers,
        :math:`(x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
        >>> output = ops.shape(input_x)
        >>> print(output)
        (3, 2, 1)
    """
    return shape_(input_x)


def dyn_shape(input_x):
    """
    Returns the shape of the input tensor.

    Args:
        input_x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Returns:
        Tensor[int], 1-dim Tensor of type int32

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
        >>> output = ops.dyn_shape(input_x)
        >>> print(output)
        [3 2 1]
    """
    return tensor_shape_(input_x)


def rank(input_x):
    """
    Returns the rank of a tensor.

    Returns a 0-D int32 Tensor representing the rank of input; the rank of a tensor
    is the number of indices required to uniquely select each element of the tensor.

    Args:
        input_x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. The data type is Number.

    Returns:
        Tensor. 0-D int32 Tensor representing the rank of input, i.e., :math:`R`. The data type is an int.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> output = ops.rank(input_tensor)
        >>> print(output)
        2
        >>> print(type(output))
        <class 'int'>
    """
    return rank_(input_x)


def reshape(input, shape):
    """
    Rearranges the input Tensor based on the given shape.

    The 'shape' can only have one -1 at most, in which case it’s inferred from the remaining dimensions and
    the number of elements in the input.

    Args:
        input (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        shape (Union[tuple[int], Tensor[int]]): Constructed by multiple
            integers, i.e., :math:`(y_1, y_2, ..., y_S)`. Only constant value is allowed.

    Returns:
        Tensor, the shape of tensor is :math:`(y_1, y_2, ..., y_S)`.

    Raises:
        ValueError: Given a shape tuple, if it has several -1; or if the product
            of its elements is less than or equal to 0 or cannot be divided by the product
            of the input tensor shape; or if it does not match the input's array size.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> output = ops.reshape(input, (3, 2))
        >>> print(output)
        [[-0.1  0.3]
         [ 3.6  0.4]
         [ 0.5 -3.2]]
    """
    return reshape_(input, shape)


def reverse_sequence(x, seq_lengths, seq_dim, batch_dim=0):
    """
    Reverses variable length slices.

    Args:
        x (Tensor): The input to reverse, supporting all number types including bool.
        seq_lengths (Tensor): Specified reversing length, must be a 1-D vector with int32 or int64 types.
        seq_dim (int): The dimension where reversal is performed. Required.
        batch_dim (int): The input is sliced in this dimension. Default: 0.

    Returns:
        Tensor, with the same shape and data type as `x`.

    Raises:
        TypeError: If `seq_dim` or `batch_dim` is not an int.
        ValueError: If value of `batch_dim` is equal to or greater than length of shape of input.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([1, 2, 3]))
        >>> output = ops.reverse_sequence(x, seq_lengths, seq_dim=1)
        >>> print(output)
        [[1. 2. 3.]
         [5. 4. 6.]
         [9. 8. 7.]]
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([1, 2, 3]))
        >>> output = ops.reverse_sequence(x, seq_lengths, seq_dim=0, batch_dim=1)
        >>> print(output)
        [[1. 5. 9.]
         [4. 2. 6.]
         [7. 8. 3.]]
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([2, 2, 3]))
        >>> output = ops.reverse_sequence(x, seq_lengths, seq_dim=1)
        >>> print(output)
        [[2. 1. 3.]
         [5. 4. 6.]
         [9. 8. 7.]]
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([3, 2, 3]))
        >>> output = ops.reverse_sequence(x, seq_lengths, seq_dim=1)
        >>> print(output)
        [[3. 2. 1.]
         [5. 4. 6.]
         [9. 8. 7.]]
        >>> x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([4, 4]))
        >>> output = ops.reverse_sequence(x, seq_lengths, seq_dim=1)
        >>> print(output)
        [[4. 3. 2. 1.]
         [8. 7. 6. 5.]]
    """
    return P.ReverseSequence(seq_dim=seq_dim, batch_dim=batch_dim)(x, seq_lengths)


def flatten(input_x):
    r"""
    Flattens a tensor without changing its batch size on the 0-th axis.

    Args:
        input_x (Tensor): Tensor of shape :math:`(N, \ldots)` to be flattened, where :math:`N` is batch size.

    Returns:
        Tensor, the shape of the output tensor is :math:`(N, X)`, where :math:`X` is
        the product of the remaining dimension.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        ValueError: If length of shape of `input_x` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.ones(shape=[1, 2, 3, 4]), mindspore.float32)
        >>> output = ops.flatten(input_x)
        >>> print(output.shape)
        (1, 24)
    """
    _flatten = _get_cache_prim(P.Flatten)()
    return _flatten(input_x)


@constexpr
def _check_select_type_match(scalar, tensor_type, scalar_name, tensor_name):
    if isinstance(scalar, int) and tensor_type != mstype.int32:
        raise TypeError(f"For functional operator[select], the input[{scalar_name}] is int, "
                        f"then the input[{tensor_name}] must be a Tensor of int32.")
    if isinstance(scalar, float) and tensor_type != mstype.float32:
        raise TypeError(f"For functional operator[select], the input[{scalar_name}] is float, "
                        f"then the input[{tensor_name}] must be a Tensor of float32.")


@constexpr
def _check_select_type(is_cond_tensor, is_x_scalar, is_y_scalar, is_x_tensor, is_y_tensor):
    if not is_cond_tensor:
        raise TypeError(f"For functional operator[select], the input[cond] must be a Tensor.")
    if is_x_scalar and not is_y_tensor:
        raise TypeError(f"For functional operator[select], the input[x] is int or float, "
                        f"then the input[y] must be a Tensor.")
    if is_y_scalar and not is_x_tensor:
        raise TypeError(f"For functional operator[select], the input[y] is int or float, "
                        f"then the input[x] must be a Tensor.")


def select(cond, x, y):
    r"""
    The conditional tensor determines whether the corresponding element in the output must be
    selected from :math:`x` (if true) or :math:`y` (if false) based on the value of each element.

    It can be defined as:

    .. math::
        out_i = \begin{cases}
        x_i, & \text{if } cond_i \\
        y_i, & \text{otherwise}
        \end{cases}

    Args:
        cond (Tensor[bool]): The condition tensor, decides which element is chosen.
          The shape is :math:`(x_1, x_2, ..., x_N, ..., x_R)`.
        x (Union[Tensor, int, float]): The first Tensor or number to be selected.
          If x is a Tensor, the shape is :math:`(x_1, x_2, ..., x_N, ..., x_R)`. If x is an int or a float,
          it will be cast to the type of int32 or float32, and broadcast to the same shape as y.
          One of x and y must be a Tensor.
        y (Union[Tensor, int, float]): The second Tensor or number to be selected.
          If y is a Tensor, The shape is :math:`(x_1, x_2, ..., x_N, ..., x_R)`. If y is an int or a float,
          it will be cast to the type of int32 or float32, and broadcast to the same shape as x.
          One of x and y must be a Tensor.

    Returns:
        Tensor, has the same shape as `cond`.

    Raises:
        TypeError: If `x` or `y` is not a Tensor, int or float.
        ValueError: The shapes of inputs are different.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # 1) Both inputs are Tensor
        >>>
        >>> cond = Tensor([True, False])
        >>> x = Tensor([2,3], mindspore.float32)
        >>> y = Tensor([1,2], mindspore.float32)
        >>> output = ops.select(cond, x, y)
        >>> print(output)
        [2. 2.]
        >>> # 2) y is a float
        >>> cond = Tensor([True, False])
        >>> x = Tensor([2,3], mindspore.float32)
        >>> y = 2.0
        >>> output = ops.select(cond, x, y)
        >>> print(output)
        [2. 2.]
    """
    is_x_scalar = isinstance(x, (int, float))
    is_y_scalar = isinstance(y, (int, float))
    is_x_tensor = isinstance(x, Tensor)
    is_y_tensor = isinstance(y, Tensor)
    is_cond_tensor = isinstance(cond, Tensor)
    _check_select_type(is_cond_tensor, is_x_scalar, is_y_scalar, is_x_tensor, is_y_tensor)
    input_x = x
    input_y = y
    if is_x_scalar:
        _check_select_type_match(x, y.dtype, "x", "y")
        input_x = zeros_like_(y) + x
        if isinstance(x, int):
            input_x = cast_(input_x, mstype.int32)
        else:
            input_x = cast_(input_x, mstype.float32)

    if is_y_scalar:
        _check_select_type_match(y, x.dtype, "y", "x")
        input_y = zeros_like_(x) + y
        if isinstance(y, int):
            input_y = cast_(input_y, mstype.int32)
        else:
            input_y = cast_(input_y, mstype.float32)
    return tensor_select_(cond, input_x, input_y)


def strided_slice(input_x,
                  begin,
                  end,
                  strides,
                  begin_mask=0,
                  end_mask=0,
                  ellipsis_mask=0,
                  new_axis_mask=0,
                  shrink_axis_mask=0):
    r"""
    Extracts a strided slice of a Tensor based on `begin/end` index and `strides`.

    This operation extracts a fragment of size (end-begin)/strides from the given 'input_tensor'.
    Starting from the beginning position, the fragment continues adding strides to the index until
    all dimensions are not less than the ending position.

    Note:
        - `begin` 、 `end` and `strides` must have the same shape.
        - `begin` 、 `end` and `strides` are all 1-D Tensor,  and their shape size
          must not greater than the dim of `input_x`.

    During the slicing process, the fragment (end-begin)/strides are extracted from each dimension.

    Example: For Tensor `input_x` with shape :math:`(5, 6, 7)`,
    set `begin`, `end` and `strides` to (1, 3, 2), (3, 5, 6),
    (1, 1, 2) respectively, then elements from index 1 to 3 are extrected for dim 0, index 3 to 5
    are extrected for dim 1 and index 2 to 6 with a `stirded` of 2 are extrected for dim 2, this
    process is equivalent to a pythonic slice `input_x[1:3, 3:5, 2:6:2]`.

    If the length of `begin` 、 `end` and `strides` is smaller than the dim of `input_x`,
    then all elements are extracted from the missing dims, it behaves like all the
    missing dims are filled with zeros, size of that missing dim and ones.

    Example: For Tensor `input_x` with shape :math:`(5, 6, 7)`,
    set `begin`, `end` and `strides` to (1, 3),
    (3, 5), (1, 1) respectively, then elements from index 1 to 3 are extrected
    for dim 0, index 3 to 5 are extrected for dim 1 and index 3 to 5 are extrected
    for dim 2, this process is equivalent to a pythonic slice `input_x[1:3, 3:5, 0:7]`.

    Here's how a mask works:
    For each specific mask, it will be converted to a binary representation internally, and then
    reverse the result to start the calculation. For Tensor `input_x` with
    shape :math:`(5, 6, 7)`. Given mask value of 3 which
    can be represented as 0b011. Reverse that we get 0b110, which implies the first and second dim of the
    original Tensor will be effected by this mask. See examples below, for simplicity all mask mentioned
    below are all in their reverted binary form:

    - `begin_mask` and `end_mask`

      If the ith bit of `begin_mask` is 1, `begin[i]` is ignored and the fullest
      possible range in that dimension is used instead. `end_mask` is analogous,
      except with the end range. For Tensor `input_x` with shape :math:`(5, 6, 7, 8)`,  if `begin_mask`
      is 0b110, `end_mask` is 0b011, the slice `input_x[0:3, 0:6, 2:7:2]` is produced.

    - `ellipsis_mask`

      If the ith bit of `ellipsis_mask` is 1, as many unspecified dimensions as needed
      will be inserted between other dimensions. Only one non-zero bit is allowed
      in `ellipsis_mask`. For a 5*6*7*8 Tensor `input_x`,  `input_x[2:,...,:6]`
      is equivalent to `input_x[2:5,:,:,0:6]` ,  `input_x[2:,...]` is equivalent
      to `input_x[2:5,:,:,:]`.

    - `new_axis_mask`

      If the ith bit of `new_axis_mask` is 1, `begin`, `end` and `strides` are
      ignored and a new length 1 dimension is added at the specified position
      in the output Tensor. For Tensor `input_x` with shape :math:`(5, 6, 7)`, if `new_axis_mask`
      is 0b110,  a new dim is added to the second dim, which will produce
      a Tensor with shape :math:`(5, 1, 6, 7)`.

    - `shrink_axis_mask`

      If the ith bit of `shrink_axis_mask` is 1, `begin`, `end` and `strides`
      are ignored and dimension i will be shrunk to 0.
      For Tensor `input_x` with shape :math:`(5, 6, 7)`,
      if `shrink_axis_mask` is 0b010, it is equivalent to slice `x[:, 5, :]`
      and results in an output shape of :math:`(5, 7)`.

    Note:
        `new_axis_mask` and  `shrink_axis_mask` are not recommended to
        use at the same time, it might incur unexpected result.

    Args:
        input_x (Tensor): The input Tensor to be extracted from.
        begin (tuple[int]): A tuple which represents the location where to start.
            Only non-negative int is allowed.
        end (tuple[int]): A tuple or which represents the maximum location where to end.
            Only non-negative int is allowed.
        strides (tuple[int]): A tuple which represents the strides is continuously added
            before reaching the maximum location. Only int is allowed, it can be negative
            which results in reversed slicing.
        begin_mask (int, optional): Starting index of the slice. Default: 0.
        end_mask (int, optional): Ending index of the slice. Default: 0.
        ellipsis_mask (int, optional): An int mask, ignore slicing operation when set to 1. Default: 0.
        new_axis_mask (int, optional): An int mask for adding new dims. Default: 0.
        shrink_axis_mask (int, optional): An int mask for shrinking dims. Default: 0.

    Returns:
        Tensor, return the extracts a strided slice of a Tensor based on `begin/end` index and `strides`.

    Raises:
        TypeError: If `begin_mask`, `end_mask`, `ellipsis_mask`, `new_axis_mask` or
            `shrink_axis_mask` is not an int.
        TypeError: If `begin`, `end` or `strides` is not tuple[int].
        ValueError: If `begin_mask`, `end_mask`, `ellipsis_mask`, `new_axis_mask` or
            `shrink_axis_mask` is less than 0.
        ValueError: If `begin`, `end` and `strides` have different shapes.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]],
        ...                   [[5, 5, 5], [6, 6, 6]]], mindspore.float32)
        >>> output = ops.strided_slice(input_x, (1, 0, 2), (3, 1, 3), (1, 1, 1))
        >>> # Take this " output = strided_slice(input_x, (1, 0, 2), (3, 1, 3), (1, 1, 1)) " as an example,
        >>> # start = [1, 0, 2] , end = [3, 1, 3], strides = [1, 1, 1], Find a segment of (start, end),
        >>> # note that end is an open interval
        >>> # To facilitate understanding, this operator can be divided into three steps:
        >>> # Step 1: Calculation of the first dimension:
        >>> # start = 1, end = 3, strides = 1, So can take 1st, 2nd rows, and then gets the final output at this time.
        >>> # output_1th =
        >>> # [
        >>> #     [
        >>> #         [3,3,3]
        >>> #         [4,4,4]
        >>> #     ]
        >>> #     [
        >>> #         [5,5,5]
        >>> #         [6,6,6]
        >>> #     ]
        >>> # ]
        >>> # Step 2: Calculation of the second dimension
        >>> # 2nd dimension, start = 0, end = 1, strides = 1. So only 0th rows
        >>> # can be taken, and the output at this time.
        >>> # output_2nd =
        >>> # [
        >>> #     [
        >>> #         [3,3,3]
        >>> #     ]
        >>> #     [
        >>> #         [5,5,5]
        >>> #     ]
        >>> # ]
        >>> # Step 3: Calculation of the third dimension
        >>> # 3nd dimension,start = 2, end = 3, strides = 1, So can take 2th cols,
        >>> # and you get the final output at this time.
        >>> # output_3ed =
        >>> # [
        >>> #     [
        >>> #         [3]
        >>> #     ]
        >>> #     [
        >>> #         [5]
        >>> #     ]
        >>> # ]
        >>> # The final output after finishing is:
        >>> print(output)
        [[[3.]]
         [[5.]]]
        >>> # another example like :
        >>> output = strided_slice(input_x, (1, 0, 0), (2, 1, 3), (1, 1, 1))
        >>> print(output)
        [[[3. 3. 3.]]]
    """
    strided_slice_ = _get_cache_prim(P.StridedSlice)(
        begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
    return strided_slice_(input_x, begin, end, strides)


def slice(input_x, begin, size):
    r"""
    Slices a tensor in the specified shape.

    Slice the tensor `input_x` in shape of `size` and starting at the location specified by `begin`.
    The slice `begin` represents the offset in each dimension of `input_x`.
    The slice `size` represents the size of the output tensor.

    Note:
        `begin` is zero-based and `size` is one-based.

    If `size[i]` is -1, all remaining elements in dimension i are included in the slice.
    This is equivalent to setting :math:`size[i] = input\_x.shape(i) - begin[i]`.

    Args:
        input_x (Tensor): The target tensor.
            The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
        begin (Union[tuple, list]): The beginning of the slice. Only constant value(>=0) is allowed.
        size (Union[tuple, list]): The size of the slice. Only constant value is allowed.

    Returns:
        Tensor, the shape is input `size`, the data type is the same as `input_x`.

    Raises:
        TypeError: If `begin` or `size` is neither tuple nor list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> import numpy as np
        >>> data = Tensor(np.array([[[1, 1, 1], [2, 2, 2]],
        ...                         [[3, 3, 3], [4, 4, 4]],
        ...                         [[5, 5, 5], [6, 6, 6]]]).astype(np.int32))
        >>> output = ops.slice(data, (1, 0, 0), (1, 1, 3))
        >>> print(output)
        [[[3 3 3]]]
        >>> output = ops.slice(data, (1, 0, 0), (1, 1, 2))
        >>> print(output)
        [[[3 3]]]
        >>> output = ops.slice(data, (1, 0, 0), (1, 1, 1))
        >>> print(output)
        [[[3]]]
        >>> output = ops.slice(data, (1, 1, 0), (1, 1, 3))
        >>> print(output)
        [[[4 4 4]]]
        >>> output = ops.slice(data, (1, 0, 1), (1, 1, 2))
        >>> print(output)
        [[[3 3]]]
    """
    return tensor_slice(input_x, begin, size)


def concat(input_x, axis=0):
    """Alias for :func:`mindspore.ops.cat()`"""
    return cat(input_x, axis)


def stack(input_x, axis=0):
    r"""
    Stacks a list of tensors in specified axis.

    Stacks the list of input tensors with the same rank `R`, output is a tensor of rank `(R+1)`.

    Given input tensors of shape :math:`(x_1, x_2, ..., x_R)`. Set the number of input tensors as `N`.
    If :math:`0 \le axis`, the shape of the output tensor is
    :math:`(x_1, x_2, ..., x_{axis}, N, x_{axis+1}, ..., x_R)`.

    Args:
        input_x (Union[tuple, list]): A Tuple or list of Tensor objects with the same shape and type.
        axis (int): Dimension to stack. Default: 0.
            Negative values wrap around. The range is [-(R+1), R+1).

    Returns:
        Tensor. A stacked Tensor with the same type as `input_x`.

    Raises:
        TypeError: If the data types of elements in `input_x` are not the same.
        ValueError: If the length of `input_x` is not greater than 1;
                    or if axis is out of the range [-(R+1), R+1);
                    or if the shapes of elements in input_x are not the same.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x1 = Tensor(np.array([0, 1]).astype(np.float32))
        >>> input_x2 = Tensor(np.array([2, 3]).astype(np.float32))
        >>> output = ops.stack((input_x1, input_x2), 0)
        >>> print(output)
        [[0. 1.]
         [2. 3.]]
    """
    _stack = _get_cache_prim(P.Stack)(axis)
    return _stack(input_x)


def unstack(input_x, axis=0):
    r"""
    Unstacks tensor in specified axis.

    Unstacks a tensor of rank `R` along axis dimension, output tensors will have rank `(R-1)`.

    Given a tensor of shape :math:`(x_1, x_2, ..., x_R)`. If :math:`0 \le axis`,
    the shape of tensor in output is :math:`(x_1, x_2, ..., x_{axis}, x_{axis+2}, ..., x_R)`.

    This is the opposite of pack.

    Args:
        input_x (Tensor): The shape is :math:`(x_1, x_2, ..., x_R)`.
            A tensor to be unstacked and the rank of the tensor must be greater than 0.
        axis (int): Dimension along which to unpack. Default: 0.
            Negative values wrap around. The range is [-R, R).

    Returns:
        A tuple of tensors, the shape of each objects is the same.

    Raises:
        ValueError: If axis is out of the range [-len(input_x.shape), len(input_x.shape)).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]))
        >>> output = ops.unstack(input_x, 0)
        >>> print(output)
        (Tensor(shape=[4], dtype=Int64, value= [1, 1, 1, 1]), Tensor(shape=[4], dtype=Int64, value= [2, 2, 2, 2]))
    """
    _unstack = _get_cache_prim(P.Unstack)(axis)
    return _unstack(input_x)


def unbind(x, dim=0):
    r"""
    Removes a tensor dimension in specified axis.

    Unstacks a tensor of rank `R` along axis dimension, and output tensors will have rank `(R-1)`.

    Given a tensor of shape :math:`(x_1, x_2, ..., x_R)`. If :math:`0 \le axis`,
    the shape of tensor in output is :math:`(x_1, x_2, ..., x_{axis}, x_{axis+2}, ..., x_R)`.

    Args:
        x (Tensor): The shape is :math:`(x_1, x_2, ..., x_R)`.
            A tensor to be unstacked and the rank of the tensor must be greater than 0.
        dim (int): Dimension along which to unpack. Negative values wrap around. The range is [-R, R). Default: 0.

    Returns:
        A tuple of tensors, the shape of each objects is the same.

    Raises:
        ValueError: If axis is out of the range [-R, R).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        >>> output = ops.unbind(x, dim=0)
        >>> print(output)
        (Tensor(shape=[3], dtype=Int64, value=[1, 2, 3]), Tensor(shape=[3], dtype=Int64, value=[4, 5, 6]),
        Tensor(shape=[3], dtype=Int64, value=[7, 8, 9]))
    """
    _unstack = _get_cache_prim(P.Unstack)(dim)
    return _unstack(x)


def expand_dims(input_x, axis):
    """
    Adds an additional dimension to `input_x` at the given axis.

    Note:
        If the specified axis is a negative number, the index is counted
        backward from the end and starts at 1.

    Args:
        input_x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        axis (int): Specifies the dimension index at which to expand
            the shape of `input_x`. The value of axis must be in the range
            `[-input_x.ndim-1, input_x.ndim]`. Only constant value is allowed.

    Returns:
        Tensor, the shape of tensor is :math:`(1, x_1, x_2, ..., x_R)` if the
        value of `axis` is 0. It has the same data type as `input_x`.

    Raises:
        TypeError: If `axis` is not an int.
        ValueError: If `axis` is not in the valid range :math:`[-a.ndim-1, a.ndim]`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> output = ops.expand_dims(input_tensor, 0)
        >>> print(output)
        [[[2. 2.]
          [2. 2.]]]
    """
    return expand_dims_(input_x, axis)


def unsqueeze(input_x, dim):
    """
    Adds an additional dimension to `input_x` at the given dim.

    Note:
        If the specified dim is a negative number, the index is counted
        backward from the end and starts at 1.

    Args:
        input_x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        dim (int): Specifies the dimension index at which to expand
            the shape of `input_x`. The value of `dim` must be in the range
            `[-input_x.ndim-1, input_x.ndim]`. Only constant value is allowed.

    Returns:
        Tensor, the shape of tensor is :math:`(1, x_1, x_2, ..., x_R)` if the
        value of `dim` is 0. It has the same data type as `input_x`.

    Raises:
        TypeError: If `dim` is not an int.
        ValueError: If `dim` is not in the valid range :math:`[-input_x.ndim-1, input_x.ndim]`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> output = ops.unsqueeze(input_tensor, dim=0)
        >>> print(output)
        [[[2. 2.]
          [2. 2.]]]
    """
    return expand_dims_(input_x, dim)


def squeeze(input_x, axis=()):
    """
    Return the Tensor after deleting the dimension of size 1 in the specified `axis`.

    If :math:`axis=()`, it will remove all the dimensions of size 1.
    If `axis` is specified, it will remove the dimensions of size 1 in the given `axis`.
    For example, if the dimension is not specified :math:`axis=()`, input shape is (A, 1, B, C, 1, D),
    then the shape of the output Tensor is (A, B, C, D). If the dimension is specified, the squeeze operation
    is only performed in the specified dimension. If input shape is (A, 1, B), input Tensor will not be
    changed when :math:`axis=0` , but when :math:`axis=1` , the shape of the input Tensor will be changed to (A, B).

    Note:
        - Please note that in dynamic graph mode, the output Tensor will share data with the input Tensor,
          and there is no Tensor data copy process.
        - The dimension index starts at 0 and must be in the range `[-input.ndim, input.ndim]`.

    Args:
        input_x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        axis (Union[int, tuple(int)]): Specifies the dimension indexes of shape to be removed, which will remove
            all the dimensions of size 1 in the given axis parameter. If specified, it must be int32 or int64.
            Default: (), an empty tuple.

    Returns:
        Tensor, the shape of tensor is :math:`(x_1, x_2, ..., x_S)`.

    Raises:
        TypeError: If `input_x` is not a tensor.
        TypeError: If `axis` is neither an int nor tuple.
        TypeError: If `axis` is a tuple whose elements are not all int.
        ValueError: If the corresponding dimension of the specified axis isn't equal to 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
        >>> output = ops.squeeze(input_x)
        >>> print(output)
        [[1. 1.]
         [1. 1.]
         [1. 1.]]
    """
    squeeze_ = _get_cache_prim(P.Squeeze)(axis)
    return squeeze_(input_x)


def transpose(input_x, input_perm):
    """
    Permutes the dimensions of the input tensor according to input permutation.

    For a 1-D array this has no effect, as a transposed vector is simply the same vector.
    To convert a 1-D array into a 2D column vector please refer the class: mindspore.ops.ExpandDims.
    For a 2-D array, this is a standard matrix transpose. For an n-D array, if axes are given,
    their order indicates how the axes are permuted (see Examples).
    If axes are not provided and a.shape = (i[0], i[1], ... i[n-2], i[n-1]),
    then a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0]).

    Note:
        On GPU and CPU, if the value of `input_perm` is negative, its actual value is `input_perm[i] + rank(input_x)`.
        Negative value of `input_perm` is not supported on Ascend.

    Args:
        input_x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        input_perm (tuple[int]): The permutation to be converted. The elements in `input_perm` are composed of
            the indexes of each dimension of `input_x`. The length of `input_perm` and the shape of `input_x` must be
            the same. Only constant value is allowed. Must be in the range [-rank(input_x), rank(input_x)).

    Returns:
        Tensor, the type of output tensor is the same as `input_x` and the shape of output tensor is decided by the
        shape of `input_x` and the value of `input_perm`.

    Raises:
        TypeError: If `input_perm` is not a tuple.
        ValueError: If length of shape of `input_x` is not equal to length of shape of `input_perm`.
        ValueError: If the same element exists in `input_perm`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
        >>> input_perm = (0, 2, 1)
        >>> output = ops.transpose(input_x, input_perm)
        >>> print(output)
        [[[ 1.  4.]
          [ 2.  5.]
          [ 3.  6.]]
         [[ 7. 10.]
          [ 8. 11.]
          [ 9. 12.]]]
    """
    return transpose_(input_x, input_perm)


def scatter_mul(input_x, indices, updates):
    r"""
    Using given values to update tensor value through the mul operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    for each `i, ..., j` in `indices.shape`:

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :] \mathrel{*}= \text{updates}[i, ..., j, :]

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type. A RuntimeError will be reported
    when the data types of parameters need to be converted.

    Args:
        input_x (Parameter): The target tensor to be updated, with data type of Parameter.
            The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        indices (Tensor): The index to do mul operation whose data type must be int32 or int64.
        updates (Tensor): The tensor doing the mul operation with `input_x`,
            the data type is same as `input_x`, the shape is `indices.shape + input_x.shape[1:]`.

    Returns:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `indices` is not an int32 or int64.
        ValueError: If the shape of `updates` is not equal to `indices.shape + input_x.shape[1:]`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> input_x = Parameter(Tensor(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([0, 1]), mindspore.int32)
        >>> updates = Tensor(np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]), mindspore.float32)
        >>> output = ops.scatter_mul(input_x, indices, updates)
        >>> print(output)
        [[2. 2. 2.]
         [4. 4. 4.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), mindspore.float32), name="x")
        >>> # for indices = [[0, 1], [1, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [1.0, 1.0, 1.0] * [1.0, 1.0, 1.0] = [1.0, 1.0, 1.0]
        >>> # input_x[1] = [2.0, 2.0, 2.0] * [3.0, 3.0, 3.0] = [6.0, 6.0, 6.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [6.0, 6.0, 6.0] * [7.0, 7.0, 7.0] = [42.0, 42.0, 42.0]
        >>> # input_x[1] = [42.0, 42.0, 42.0] * [9.0, 9.0, 9.0] = [378.0, 378.0, 378.0]
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> output = ops.scatter_mul(input_x, indices, updates)
        >>> print(output)
        [[  1.   1.   1.]
         [378. 378. 378.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), mindspore.float32), name="x")
        >>> # for indices = [[1, 0], [1, 1]]
        >>> # step 1: [1, 0]
        >>> # input_x[0] = [1.0, 1.0, 1.0] * [3.0, 3.0, 3.0] = [3.0, 3.0, 3.0]
        >>> # input_x[1] = [2.0, 2.0, 2.0] * [1.0, 1.0, 1.0] = [2.0, 2.0, 2.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [2.0, 2.0, 2.0] * [7.0, 7.0, 7.0] = [14.0, 14.0, 14.0]
        >>> # input_x[1] = [14.0, 14.0, 14.0] * [9.0, 9.0, 9.0] = [126.0, 126.0, 126.0]
        >>> indices = Tensor(np.array([[1, 0], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> output = ops.scatter_mul(input_x, indices, updates)
        >>> print(output)
        [[  3.   3.   3.]
         [126. 126. 126.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), mindspore.float32), name="x")
        >>> # for indices = [[0, 1], [0, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [1.0, 1.0, 1.0] * [1.0, 1.0, 1.0] = [1.0, 1.0, 1.0]
        >>> # input_x[1] = [2.0, 2.0, 2.0] * [3.0, 3.0, 3.0] = [6.0, 6.0, 6.0]
        >>> # step 2: [0, 1]
        >>> # input_x[0] = [1.0, 1.0, 1.0] * [7.0, 7.0, 7.0] = [7.0, 7.0, 7.0]
        >>> # input_x[1] = [6.0, 6.0, 6.0] * [9.0, 9.0, 9.0] = [54.0, 54.0, 54.0]
        >>> indices = Tensor(np.array([[0, 1], [0, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> output = ops.scatter_mul(input_x, indices, updates)
        >>> print(output)
        [[ 7.  7.  7.]
         [54. 54. 54.]]
    """
    return scatter_mul_(input_x, indices, updates)


def scatter_max(input_x, indices, updates):
    r"""
    Using given values to update tensor value through the max operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    Args:
        input_x (Parameter): The target tensor, with data type of Parameter.
            The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        indices (Tensor): The index to do max operation whose data type must be mindspore.int32.
        updates (Tensor): The tensor doing the max operation with `input_x`,
            the data type is same as `input_x`, the shape is `indices.shape + x.shape[1:]`.

    Returns:
        Tensor, the updated `input_x`, the type and shape same as `input_x`.

    Raises:
        TypeError: If `indices` is not an int32 or int64.
        ValueError: If the shape of `updates` is not equal to `indices.shape + input_x.shape[1:]`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.
        RuntimeError: On the Ascend platform, the input data dimension of `input_x` , `indices`
                      and `updates` is greater than 8 dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Parameter(Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32), name="input_x")
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.ones([2, 2, 3]) * 88, mindspore.float32)
        >>> output = ops.scatter_max(input_x, indices, updates)
        >>> print(output)
        [[88. 88. 88.]
         [88. 88. 88.]]
    """
    return scatter_max_(input_x, indices, updates)


def scatter_add(input_x, indices, updates):
    r"""
    Using given values to update tensor value through the add operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    Args:
        input_x (Parameter): The target tensor, with data type of Parameter.
            The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        indices (Tensor): The index to do add operation whose data type must be int32 or int64.
        updates (Tensor): The tensor doing the add operation with `input_x`,
            the data type is same as `input_x`, the shape is `indices.shape + x.shape[1:]`.

    Returns:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `indices` is not an int32 or int64.
        ValueError: If the shape of `updates` is not equal to `indices.shape + input_x.shape[1:]`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
            is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor, Parameter
        >>> from mindspore import ops
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> output = ops.scatter_add(input_x, indices, updates)
        >>> print(output)
        [[ 1.  1.  1.]
         [19. 19. 19.]]
    """
    return scatter_add_(input_x, indices, updates)


def scatter_min(input_x, indices, updates):
    r"""
    Using given values to update tensor value through the min operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    for each :math:`i, ..., j` in `indices.shape`:

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :]
        = min(\text{input_x}[\text{indices}[i, ..., j], :], \text{updates}[i, ..., j, :])

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type. A RuntimeError will be reported
    when `updates` does not support conversion to the data type required by `input_x`.

    Args:
        input_x (Parameter): The target tensor, with data type of Parameter.
            The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        indices (Tensor): The index to do min operation whose data type must be mindspore.int32 or mindspore.int64.
        updates (Tensor): The tensor doing the min operation with `input_x`,
            the data type is same as `input_x`, the shape is `indices.shape + input_x.shape[1:]`.

    Returns:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `indices` is not an int32 or an int64.
        ValueError: If the shape of `updates` is not equal to `indices.shape + input_x.shape[1:]`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.
        RuntimeError: On the Ascend platform, the input data dimension of `input_x` , `indices`
                      and `updates` is greater than 8 dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor, Parameter
        >>> from mindspore import ops
        >>> input_x = Parameter(Tensor(np.zeros((2, 3)), mindspore.float32), name="input_x")
        >>> indices = Tensor(np.array([1, 0]), mindspore.int32)
        >>> update = Tensor(np.arange(6).reshape((2, 3)), mindspore.float32)
        >>> output = ops.scatter_min(input_x, indices, update)
        >>> print(output)
        [[0. 0. 0.]
         [0. 0. 0.]]
    """
    return scatter_min_(input_x, indices, updates)


def scatter_div(input_x, indices, updates):
    r"""
    Using given values to update tensor value through the div operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    for each :math:`i, ..., j` in `indices.shape`:

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :] \mathrel{/}= \text{updates}[i, ..., j, :]

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type. A RuntimeError will be reported
    when `updates` does not support conversion to the data type required by `input_x`.

    Args:
        input_x (Parameter): The target tensor, with data type of Parameter.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        indices (Tensor): The index to do divide operation whose data type must be mindspore.int32 or
          mindspore.int64.
        updates (Tensor): The tensor doing the divide operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices.shape + input_x.shape[1:]`.

    Returns:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `indices` is not an int32 or an int64.
        ValueError: If the shape of `updates` is not equal to `indices.shape + input_x.shape[1:]`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.
        RuntimeError: On the Ascend platform, the input data dimension of `input_x` , `indices`
                      and `updates` is greater than 8 dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Parameter(Tensor(np.array([[6.0, 6.0, 6.0], [2.0, 2.0, 2.0]]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([0, 1]), mindspore.int32)
        >>> updates = Tensor(np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]), mindspore.float32)
        >>> output = ops.scatter_div(input_x, indices, updates)
        >>> print(output)
        [[3. 3. 3.]
         [1. 1. 1.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[105.0, 105.0, 105.0],
        ...                                      [315.0, 315.0, 315.0]]), mindspore.float32), name="x")
        >>> # for indices = [[0, 1], [1, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [105.0, 105.0, 105.0] / [1.0, 1.0, 1.0] = [105.0, 105.0, 105.0]
        >>> # input_x[1] = [315.0, 315.0, 315.0] / [3.0, 3.0, 3.0] = [105.0, 105.0, 105.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [105.0, 105.0, 105.0] / [5.0, 5.0, 5.0] = [21.0, 21.0, 21.0]
        >>> # input_x[1] = [21.0, 21.0, 21.0] / [7.0, 7.0, 7.0] = [3.0, 3.0, 3.0]
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[5.0, 5.0, 5.0], [7.0, 7.0, 7.0]]]), mindspore.float32)
        >>> output = ops.scatter_div(input_x, indices, updates)
        >>> print(output)
        [[105. 105. 105.]
         [  3.   3.   3.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[105.0, 105.0, 105.0],
        ...                                      [315.0, 315.0, 315.0]]), mindspore.float32), name="x")
        >>> # for indices = [[1, 0], [1, 1]]
        >>> # step 1: [1, 0]
        >>> # input_x[0] = [105.0, 105.0, 105.0] / [3.0, 3.0, 3.0] = [35.0, 35.0, 35.0]
        >>> # input_x[1] = [315.0, 315.0, 315.0] / [1.0, 1.0, 1.0] = [315.0, 315.0, 315.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [315.0, 315.0, 315.0] / [5.0, 5.0, 5.0] = [63.0 63.0 63.0]
        >>> # input_x[1] = [63.0 63.0 63.0] / [7.0, 7.0, 7.0] = [9.0, 9.0, 9.0]
        >>> indices = Tensor(np.array([[1, 0], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[5.0, 5.0, 5.0], [7.0, 7.0, 7.0]]]), mindspore.float32)
        >>> output = ops.scatter_div(input_x, indices, updates)
        >>> print(output)
        [[35. 35. 35.]
         [ 9.  9.  9.]]
    """
    return scatter_div_(input_x, indices, updates)


def scatter_nd(indices, updates, shape):
    r"""
    Scatters a tensor into a new tensor depending on the specified indices.

    Creates an empty tensor with the given `shape`, and set values by scattering the update tensor
    depending on indices. The empty tensor has rank :math:`P` and `indices` has rank :math:`Q`.

    The `shape` is :math:`(s_0, s_1, ..., s_{P-1})`, where :math:`P \ge 1`.

    `indices` has shape :math:`(i_0, i_1, ..., i_{Q-2}, N)`, where :math:`Q \ge 2` and :math:`N \le P`.

    The last dimension of `indices` (with length :math:`N` ) indicates slices along the :math:`N` th dimension of the
    empty tensor.

    `updates` is a tensor of rank :math:`Q-1+P-N`, and
    its shape is :math:`(i_0, i_1, ..., i_{Q-2}, s_N, s_{N+1}, ..., s_{P-1})`.

    If `indices` contains duplicates, the duplicate `updates` are summed.

    The following figure shows the calculation process of inserting two new value matrices into the first dimension
    with rank-3:

    .. image:: ScatterNd.png

    Args:
        indices (Tensor): Define the index of scattering in the new tensor with int32 or int64 data type.
            The rank of `indices` must be at least 2 and `indices.shape[-1] <= len(shape)`.
        updates (Tensor): Define the source Tensor to be updated.
            It has shape `indices.shape[:-1] + shape[indices.shape[-1]:]`.
        shape (tuple[int]): Define the shape of the output tensor, has the same data type as indices.
            `shape` can not be empty, and the elements in `shape` must be greater than or equal to 1.

    Returns:
        Tensor, the new tensor, has the same type as `update` and the same shape as `shape`.

    Raises:
        TypeError: If `shape` is not a tuple.
        ValueError: If any element of `shape` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor(np.array([[0], [2]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2],
        ...                             [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[1, 1, 1, 1], [2, 2, 2, 2],
        ...                             [3, 3, 3, 3], [4, 4, 4, 4]]]), mindspore.float32)
        >>> shape = (4, 4, 4)
        >>> output = ops.scatter_nd(indices, updates, shape)
        >>> print(output)
        [[[1. 1. 1. 1.]
          [2. 2. 2. 2.]
          [3. 3. 3. 3.]
          [4. 4. 4. 4.]]
         [[0. 0. 0. 0.]
          [0. 0. 0. 0.]
          [0. 0. 0. 0.]
          [0. 0. 0. 0.]]
         [[1. 1. 1. 1.]
          [2. 2. 2. 2.]
          [3. 3. 3. 3.]
          [4. 4. 4. 4.]]
         [[0. 0. 0. 0.]
          [0. 0. 0. 0.]
          [0. 0. 0. 0.]
          [0. 0. 0. 0.]]]
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([3.2, 1.1]), mindspore.float32)
        >>> shape = (3, 3)
        >>> output = ops.scatter_nd(indices, updates, shape)
        >>> # In order to facilitate understanding, explain the operator pseudo-operation process step by step:
        >>> # Step 1: Generate an empty Tensor of the specified shape according to the shape
        >>> # [
        >>> #     [0. 0. 0.]
        >>> #     [0. 0. 0.]
        >>> #     [0. 0. 0.]
        >>> # ]
        >>> # Step 2: Modify the data at the specified location according to the indicators
        >>> # 0th row of indices is [0, 1], 0th row of updates is 3.2.
        >>> # means that the empty tensor in the 0th row and 1st col set to 3.2
        >>> # [
        >>> #     [0. 3.2. 0.]
        >>> #     [0. 0.   0.]
        >>> #     [0. 0.   0.]
        >>> # ]
        >>> # 1th row of indices is [1, 1], 1th row of updates is 1.1.
        >>> # means that the empty tensor in the 1th row and 1st col set to 1.1
        >>> # [
        >>> #     [0. 3.2. 0.]
        >>> #     [0. 1.1  0.]
        >>> #     [0. 0.   0.]
        >>> # ]
        >>> # The final result is as follows:
        >>> print(output)
        [[0. 3.2 0.]
         [0. 1.1 0.]
         [0. 0.  0.]]
    """
    return scatter_nd_(indices, updates, shape)


def scatter_update(input_x, indices, updates):
    r"""
    Updates tensor values by using input indices and value.

    Using given values to update tensor value, along with the input indices.

    for each `i, ..., j` in `indices.shape`:

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :] = \text{updates}[i, ..., j, :]

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.

    Args:
        input_x (Parameter): The target tensor, with data type of Parameter.
            The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        indices (Tensor): The index of input tensor. With int32 or int64 data type.
            If there are duplicates in indices, the order for updating is undefined.
        updates (Tensor): The tensor to update the input tensor, has the same type as input,
            and updates.shape = indices.shape + input_x.shape[1:].

    Returns:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `indices` is not an int32 or an int64.
        ValueError: If the shape of `updates` is not equal to `indices.shape + input_x.shape[1:]`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> np_x = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]])
        >>> input_x = mindspore.Parameter(Tensor(np_x, mindspore.float32), name="x")
        >>> indices = Tensor(np.array([0, 1]), mindspore.int32)
        >>> np_updates = np.array([[2.0, 1.2, 1.0], [3.0, 1.2, 1.0]])
        >>> updates = Tensor(np_updates, mindspore.float32)
        >>> output = ops.scatter_update(input_x, indices, updates)
        >>> print(output)
        [[2. 1.2  1.]
         [3. 1.2  1.]]
    """
    scatter_update_inner = _get_cache_prim(P.ScatterUpdate)()
    return scatter_update_inner(input_x, indices, updates)


def scatter_nd_add(input_x, indices, updates, use_locking=False):
    r"""
    Applies sparse addition to individual values or slices in a tensor.

    Using given values to update tensor value through the add operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    `input_x` has rank P and `indices` has rank Q where `Q >= 2`.

    `indices` has shape :math:`(i_0, i_1, ..., i_{Q-2}, N)` where `N <= P`.

    The last dimension of `indices` (with length `N` ) indicates slices along the `N` th dimension of `input_x`.

    `updates` is a tensor of rank `Q-1+P-N`. Its shape is:
    :math:`(i_0, i_1, ..., i_{Q-2}, x\_shape_N, ..., x\_shape_{P-1})`.

    Args:
        input_x (Parameter): The target tensor, with data type of Parameter.
            The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        indices (Tensor): The index to do min operation whose data type must be mindspore.int32 or mindspore.int64.
            The rank of indices must be at least 2 and `indices.shape[-1] <= len(shape)`.
        updates (Tensor): The tensor doing the addition operation with `input_x`,
            the data type is same as `input_x`, the shape is `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.
        use_locking (bool): Whether to protect the assignment by a lock. Default: False.

    Returns:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If the dtype of `use_locking` is not bool.
        TypeError: If the dtype of `indices` is not int32 or int64.
        TypeError: If dtype of `input_x` and `updates` are not the same.
        ValueError: If the shape of `updates` is not equal to `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
        >>> updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
        >>> output = ops.scatter_nd_add(input_x, indices, updates, False)
        >>> print(output)
        [ 1. 10.  9.  4. 12.  6.  7. 17.]
        >>> input_x = Parameter(Tensor(np.zeros((4, 4, 4)), mindspore.int32))
        >>> indices = Tensor(np.array([[0], [2]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]), mindspore.int32)
        >>> output = ops.scatter_nd_add(input_x, indices, updates, False)
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
    Applies sparse subtraction to individual values or slices in a tensor.

    Using given values to update tensor value through the subtraction operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    `input_x` has rank P and `indices` has rank Q where `Q >= 2`.

    `indices` has shape :math:`(i_0, i_1, ..., i_{Q-2}, N)` where `N <= P`.

    The last dimension of `indices` (with length `N` ) indicates slices along the `N` th dimension of `input_x`.

    `updates` is a tensor of rank `Q-1+P-N`. Its shape is:
    :math:`(i_0, i_1, ..., i_{Q-2}, x\_shape_N, ..., x\_shape_{P-1})`.

    Args:
        input_x (Parameter): The target tensor, with data type of Parameter.
            The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        indices (Tensor): The index of input tensor, with int32 or int64 data type.
            The rank of indices must be at least 2 and `indices.shape[-1] <= len(shape)`.
        updates (Tensor): The tensor doing the subtraction operation with `input_x`, has the same type as input.
            The shape is `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.
        use_locking (bool): Whether to protect the assignment by a lock. Default: False.

    Returns:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If the dtype of `use_locking` is not bool.
        TypeError: If the dtype of `indices` is not int32 or int64.
        TypeError: If dtype of `input_x` and `updates` are not the same.
        ValueError: If the shape of `updates` is not equal to `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
        >>> updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
        >>> output = ops.scatter_nd_sub(input_x, indices, updates, False)
        >>> print(output)
        [ 1. -6. -3.  4. -2.  6.  7. -1.]
        >>> input_x = Parameter(Tensor(np.zeros((4, 4, 4)), mindspore.int32))
        >>> indices = Tensor(np.array([[0], [2]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]), mindspore.int32)
        >>> output = ops.scatter_nd_sub(input_x, indices, updates, False)
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
    Applies sparse multiplication to individual values or slices in a tensor.

    Using given values to update parameter value through the multiplication operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    `input_x` has rank P and `indices` has rank Q, where `Q >= 2`.

    `indices` has shape :math:`(i_0, i_1, ..., i_{Q-2}, N)` where `N <= P`.

    The last dimension of `indices` (with length `N` ) indicates slices along the `N` th dimension of `input_x`.

    `updates` is a tensor of rank `Q-1+P-N`. Its shape is:
    :math:`(i_0, i_1, ..., i_{Q-2}, x\_shape_N, ..., x\_shape_{P-1})`.

    Args:
        input_x (Parameter): The target tensor, with data type of Parameter.
            The shape is :math:`(N,*)`, where :math:`*` means any number of additional dimensions.
        indices (Tensor): The index to do multiplication operation whose data type must be mindspore.int32 or
            mindspore.int64. The rank of indices must be at least 2 and `indices.shape[-1] <= len(shape)`.
        updates (Tensor): The tensor to do the multiplication operation with `input_x`.
            The data type is same as `input_x`, and the shape is `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.
        use_locking (bool): Whether to protect the assignment by a lock. Default: False.

    Returns:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If the dtype of `use_locking` is not bool.
        TypeError: If the dtype of `indices` is not int32 or int64.
        TypeError: If dtype of `input_x` and `updates` are not the same.
        ValueError: If the shape of `updates` is not equal to `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> input_x = Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
        >>> updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
        >>> output = ops.scatter_nd_mul(input_x, indices, updates)
        >>> print(output)
        [ 1. 16. 18.  4. 35.  6.  7. 72.]
        >>> input_x = Parameter(Tensor(np.ones((4, 4, 4)), mindspore.int32))
        >>> indices = Tensor(np.array([[0], [2]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]), mindspore.int32)
        >>> output = ops.scatter_nd_mul(input_x, indices, updates)
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
    Applying sparse division to individual values or slices in a tensor.

    Using given values to update tensor value through the div operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    `input_x` has rank P and `indices` has rank Q, where `Q >= 2`.

    `indices` has shape :math:`(i_0, i_1, ..., i_{Q-2}, N)` where `N <= P`.

    The last dimension of `indices` (with length `N` ) indicates slices along the `N` th dimension of `input_x`.

    `updates` is a tensor of rank `Q-1+P-N`. Its shape is:
    :math:`(i_0, i_1, ..., i_{Q-2}, x\_shape_N, ..., x\_shape_{P-1})`.

    Args:
        input_x (Parameter): The target tensor, with data type of Parameter.
            The shape is :math:`(N,*)`, where :math:`*` means any number of additional dimensions.
        indices (Tensor): The index to do div operation whose data type must be mindspore.int32 or mindspore.int64.
            The rank of indices must be at least 2 and `indices.shape[-1] <= len(shape)`.
        updates (Tensor): The tensor to do the div operation with `input_x`.
            The data type is same as `input_x`, and the shape is `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.
        use_locking (bool): Whether to protect the assignment by a lock. Default: False.

    Returns:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If the dtype of `use_locking` is not bool.
        TypeError: If the dtype of `indices` is not int32 or int64.
        TypeError: If dtype of `input_x` and `updates` are not the same.
        ValueError: If the shape of `updates` is not equal to `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> input_x = Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
        >>> updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
        >>> output = ops.scatter_nd_div(input_x, indices, updates, False)
        >>> print(output)
        [1.         0.25       0.5        4.         0.71428573 6.
         7.         0.8888889 ]
        >>> input_x = Parameter(Tensor(np.ones((4, 4, 4)), mindspore.float32))
        >>> indices = Tensor(np.array([[0], [2]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]), mindspore.float32)
        >>> output = ops.scatter_nd_div(input_x, indices, updates, False)
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
    Applying sparse maximum to individual values or slices in a tensor.

    Using given values to update parameter value through the max operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    `input_x` has rank P and `indices` has rank Q where `Q >= 2`.

    `indices` has shape :math:`(i_0, i_1, ..., i_{Q-2}, N)` where `N <= P`.

    The last dimension of `indices` (with length `N` ) indicates slices along the `N` th dimension of `input_x`.

    `updates` is a tensor of rank `Q-1+P-N`. Its shape is:
    :math:`(i_0, i_1, ..., i_{Q-2}, x\_shape_N, ..., x\_shape_{P-1})`.

    Args:
        input_x (Parameter): The target tensor, with data type of Parameter.
            The shape is :math:`(N,*)`, where :math:`*` means any number of additional dimensions.
        indices (Tensor): The index to do maximum operation whose data type must be mindspore.int32 or mindspore.int64.
            The rank of indices must be at least 2 and `indices.shape[-1] <= len(shape)`.
        updates (Tensor): The tensor to do the max operation with `input_x`.
            The data type is same as `input_x`, and the shape is `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.
        use_locking (bool): Whether to protect the assignment by a lock. Default: False.

    Returns:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If the dtype of `use_locking` is not bool.
        TypeError: If the dtype of `indices` is not int32 or int64.
        TypeError: If dtype of `input_x` and `updates` are not the same.
        ValueError: If the shape of `updates` is not equal to `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
        >>> updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
        >>> output = ops.scatter_nd_max(input_x, indices, updates, False)
        >>> print(output)
        [1. 8. 6. 4. 7. 6. 7. 9.]
        >>> input_x = Parameter(Tensor(np.ones((4, 4, 4)), mindspore.int32))
        >>> indices = Tensor(np.array([[0], [2]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]), mindspore.int32)
        >>> output = ops.scatter_nd_max(input_x, indices, updates, False)
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
    Applying sparse minimum to individual values or slices in a tensor.

    Using given values to update tensor value through the min operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    `input_x` has rank P and `indices` has rank Q where `Q >= 2`.

    `indices` has shape :math:`(i_0, i_1, ..., i_{Q-2}, N)` where `N <= P`.

    The last dimension of `indices` (with length `N` ) indicates slices along the `N` th dimension of `input_x`.

    `updates` is a tensor of rank `Q-1+P-N`. Its shape is:
    :math:`(i_0, i_1, ..., i_{Q-2}, x\_shape_N, ..., x\_shape_{P-1})`.

    Args:
        input_x (Parameter): The target tensor, with data type of Parameter.
            The shape is :math:`(N,*)`, where :math:`*` means any number of additional dimensions.
        indices (Tensor): The index to do min operation whose data type must be mindspore.int32 or mindspore.int64.
            The rank of indices must be at least 2 and `indices.shape[-1] <= len(shape)`.
        updates (Tensor): The tensor to do the min operation with `input_x`.
            The data type is same as `input_x`, and the shape is `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.
        use_locking (bool): Whether to protect the assignment by a lock. Default: False.

    Returns:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If the dtype of `use_locking` is not bool.
        TypeError: If the dtype of `indices` is not int32 or int64.
        TypeError: If dtype of `input_x` and `updates` are not the same.
        ValueError: If the shape of `updates` is not equal to `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Parameter(Tensor(np.ones(8) * 10, mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
        >>> updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
        >>> output = ops.scatter_nd_min(input_x, indices, updates, False)
        >>> print(output)
        [10.  8.  6. 10.  7. 10. 10.  9.]
        >>> input_x = Parameter(Tensor(np.ones((4, 4, 4)) * 10, mindspore.int32))
        >>> indices = Tensor(np.array([[0], [2]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]), mindspore.int32)
        >>> output = ops.scatter_nd_min(input_x, indices, updates, False)
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
    Sorts the elements of the input tensor along the given dimension in the specified order.

    Args:
        input_x(Tensor): The input tensor to sort.
            The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
        axis (int, optional): The dimension to sort along. Default: -1.
        descending (bool, optional): Controls the sort order. If `descending` is True, the elements
            are sorted in descending order, or else sorted in ascending order. Default: False.

    .. warning::
        Currently, the data types of Float16, UInt8, Int8, Int16, Int32, Int64 are well supported.
        If use Float32, it may cause loss of accuracy.

    Returns:

        - y1, a tensor whose values are the sorted values, with the same shape and data type as input.
        - y2, a tensor that consists of the indices of the elements in the original input tensor.
          Data type is int32.

    Raises:
        TypeError: If `axis` is not an int.
        TypeError: If `descending` is not a bool.
        TypeError: If dtype of `input_x` is neither float16, float32, uint8, int8, int16, int32, int64.
        ValueError: If `axis` is not in range of [-len(input_x.shape), len(input_x.shape)).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]), mindspore.float16)
        >>> output = ops.sort(x)
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


def argsort(input_x, axis=-1, descending=False):
    r"""
    Sorts the input tensor along the given dimension in specified order and return the sorted indices.

    Args:
        input_x(Tensor): The input tensor to sort.
        axis (int): The axis to sort along. Default: -1, means the last axis
        descending (bool): The sort order. If `descending` is True then the elements
            are sorted in descending order by value. Otherwise sort in descending order. Default: False.

    Returns:
        Tensor, the indices of sorted input tensor. Data type is int32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]), mindspore.float16)
        >>> sort = ops.argsort(x)
        >>> print(sort)
    """
    _sort = _get_cache_prim(P.Sort)(axis, descending)
    _, arg_sort = _sort(input_x)
    return arg_sort


def gather(input_params, input_indices, axis, batch_dims=0):
    r"""
    Returns the slice of the input tensor corresponding to the elements of `input_indices` on the specified `axis`.

    The following figure shows the calculation process of Gather commonly:

    .. image:: Gather.png

    where params represents the input `input_params`, and indices represents the index to be sliced `input_indices`.

    .. note::
        1. The value of input_indices must be in the range of `[0, input_param.shape[axis])`, the result is undefined
           out of range.

        2. The data type of input_params cannot be
           `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ on Ascend
           platform currently.

    Args:
        input_params (Tensor): The original Tensor. The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        input_indices (Tensor): Index tensor to be sliced, the shape of tensor is :math:`(y_1, y_2, ..., y_S)`.
            Specifies the indices of elements of the original Tensor. The data type can be int32 or int64.
        axis (int): Specifies the dimension index to gather indices.

    Returns:
        Tensor, the shape of tensor is
        :math:`input\_params.shape[:axis] + input\_indices.shape + input\_params.shape[axis + 1:]`.

    Raises:
        TypeError: If `axis` is not an int.
        TypeError: If `input_params` is not a tensor.
        TypeError: If `input_indices` is not a tensor of type int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case1: input_indices is a Tensor with shape (5, ).
        >>> input_params = Tensor(np.array([1, 2, 3, 4, 5, 6, 7]), mindspore.float32)
        >>> input_indices = Tensor(np.array([0, 2, 4, 2, 6]), mindspore.int32)
        >>> axis = 0
        >>> output = ops.gather(input_params, input_indices, axis)
        >>> print(output)
        [1. 3. 5. 3. 7.]
        >>> # case2: input_indices is a Tensor with shape (2, 2). When the input_params has one dimension,
        >>> # the output shape is equal to the input_indices shape.
        >>> input_indices = Tensor(np.array([[0, 2], [2, 6]]), mindspore.int32)
        >>> axis = 0
        >>> output = ops.gather(input_params, input_indices, axis)
        >>> print(output)
        [[ 1. 3.]
         [ 3. 7.]]
        >>> # case3: input_indices is a Tensor with shape (2, ) and
        >>> # input_params is a Tensor with shape (3, 4) and axis is 0.
        >>> input_params = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), mindspore.float32)
        >>> input_indices = Tensor(np.array([0, 2]), mindspore.int32)
        >>> axis = 0
        >>> output = ops.gather(input_params, input_indices, axis)
        >>> print(output)
        [[1.  2.  3.  4.]
         [9. 10. 11. 12.]]
        >>> # case4: input_indices is a Tensor with shape (2, ) and
        >>> # input_params is a Tensor with shape (3, 4) and axis is 1.
        >>> input_params = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), mindspore.float32)
        >>> input_indices = Tensor(np.array([0, 2]), mindspore.int32)
        >>> axis = 1
        >>> output = ops.gather(input_params, input_indices, axis)
        >>> print(output)
        [[1.  3.]
         [5.  7.]
         [9. 11.]]
    """
    _gather = _get_cache_prim(P.Gather)(batch_dims)
    return _gather(input_params, input_indices, axis)


def gather_d(x, dim, index):
    """
    Gathers elements along an axis specified by dim.

    Refer to :func:`mindspore.ops.gather_elements` for more detail.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1, 2], [3, 4]]), mindspore.int32)
        >>> index = Tensor(np.array([[0, 0], [1, 0]]), mindspore.int32)
        >>> dim = 1
        >>> output = ops.gather_d(x, dim, index)
        >>> print(output)
        [[1 1]
         [4 3]]
    """
    return gather_d_(x, dim, index)


def gather_elements(x, dim, index):
    """
    Gathers elements along an axis specified by dim.

    For a 3-D tensor, the output is:

    .. code-block::

        output[i][j][k] = x[index[i][j][k]][j][k]  # if dim == 0

        output[i][j][k] = x[i][index[i][j][k]][k]  # if dim == 1

        output[i][j][k] = x[i][j][index[i][j][k]]  # if dim == 2

    `x` and `index` have the same length of dimensions, and all dimensions except `dim` have the same size.
    If `dim` = i, `x` is an n-D tensor with shape :math:`(z_0, z_1, ..., z_i, ..., z_{n-1})`,
    the `index` must be an n-D tensor with shape :math:`(z_0, z_1, ..., y, ..., z_{n-1})`
    where `y`>=1 and the output will have the same shape with `index`.

    Args:
        x (Tensor): The input tensor.
        dim (int): The axis along which to index. It must be int32 or int64. The value range is [-x.ndim, x.ndim).
        index (Tensor): The indices of elements to gather. It can be one of the following data types:
            int32, int64. The value range of each index element is [-x.shape(dim), x.shape(dim)).

    Returns:
        Tensor, has the same shape as index tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_{n-1})`,
        and has the same data type with `x`.

    Raises:
        TypeError: If dtype of `dim` or `index` is neither int32 nor int64.
        ValueError: If length of shape of `x` is not equal to length of shape of `index`.
        ValueError: If the size of the dimension except `dim` is not equal between `x` and `index`.
        ValueError: If the value of `dim` is not in the expected range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> x = Tensor(np.array([[1, 2], [3, 4]]), mindspore.int32)
        >>> index = Tensor(np.array([[0, 0], [1, 0]]), mindspore.int32)
        >>> dim = 1
        >>> output = mindspore.ops.gather_elements(x, dim, index)
        >>> print(output)
        [[1 1]
         [4 3]]
    """
    return gather_d_(x, dim, index)


def gather_nd(input_x, indices):
    r"""
    Gathers slices from a tensor by indices.

    Using given indices to gather slices from a tensor with a specified shape.

    `indices` is an K-dimensional integer tensor. Supposes it as a (K-1)-dimensional tensor and each element of it
    defines a slice of `input_x`:

    .. math::
        output[(i_0, ..., i_{K-2})] = input\_x[indices[(i_0, ..., i_{K-2})]]

    The last dimension of `indices` can not more than the rank of `input_x`:
    :math:`indices.shape[-1] <= input\_x.rank`.

    Args:
        input_x (Tensor): The target tensor to gather values.
            The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        indices (Tensor): The index tensor, with int32 or int64 data type.

    Returns:
        Tensor, has the same type as `input_x` and the shape is
        :math:`indices\_shape[:-1] + input\_x\_shape[indices\_shape[-1]:]`.

    Raises:
        ValueError: If length of shape of `input_x` is less than the last dimension of `indices`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> output = ops.gather_nd(input_x, indices)
        >>> print(output)
        [-0.1  0.5]
    """
    return gather_nd_(input_x, indices)


def tensor_scatter_add(input_x, indices, updates):
    """
    Creates a new tensor by adding the values from the positions in `input_x` indicated by
    `indices`, with values from `updates`. When multiple values are given for the same
    index, the updated result will be the sum of all values. This operation is almost
    equivalent to using ScatterNdAdd, except that the updates are applied on output `Tensor`
    instead of input `Parameter`.

    The last axis of `indices` is the depth of each index vectors. For each index vector,
    there must be a corresponding value in `updates`. The shape of `updates` should be
    equal to the shape of `input_x[indices]`. For more details, see use cases.

    Note:
        On GPU, if some values of the `indices` are out of bound, instead of raising an index error,
        the corresponding `updates` will not be updated to self tensor. On CPU, if some values of
        the `indices` are out of bound, raising an index error. On Ascend, out of bound checking is
        not supported, if some values of the `indices` are out of bound, unknown errors may be caused.

    Args:
        input_x (Tensor): The target tensor. The dimension of input_x must be no less than indices.shape[-1].
        indices (Tensor): The index of input tensor whose data type is int32 or int64.
            The rank must be at least 2.
        updates (Tensor): The tensor to update the input tensor, has the same type as input,
            and updates. Shape should be equal to indices.shape[:-1] + input_x.shape[indices.shape[-1]:].

    Returns:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If dtype of `indices` is neither int32 nor int64.
        ValueError: If length of shape of `input_x` is less than the last dimension of shape of `indices`.
        RuntimeError: If a value of `indices` is not in `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> from mindspore import ops
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [0, 0]]), mindspore.int32)
        >>> updates = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> output = ops.tensor_scatter_add(input_x, indices, updates)
        >>> print(output)
        [[ 3.1  0.3  3.6]
         [ 0.4  0.5 -3.2]]
    """

    return tensor_scatter_add_(input_x, indices, updates)


def tensor_scatter_sub(input_x, indices, updates):
    """
    Creates a new tensor by subtracting the values from the positions in `input_x` indicated by
    `indices`, with values from `updates`. When multiple values are provided for the same
    index, the result of the update will be to subtract these values respectively. This operation is almost
    equivalent to using :class:`mindspore.ops.ScatterNdSub` , except that the updates are applied on output `Tensor`
    instead of input `Parameter`.

    The last axis of `indices` is the depth of each index vectors. For each index vector,
    there must be a corresponding value in `updates`. The shape of `updates` should be
    equal to the shape of `input_x[indices]`. For more details, see use cases.

    Note:
        On GPU, if some values of the `indices` are out of bound, instead of raising an index error,
        the corresponding `updates` will not be updated to self tensor. On CPU, if some values of
        the `indices` are out of bound, raising an index error. On Ascend, out of bound checking is
        not supported, if some values of the `indices` are out of bound, unknown errors may be caused.

    Args:
        input_x (Tensor): The target tensor. The dimension of input_x must be no less than indices.shape[-1].
        indices (Tensor): The index of input tensor whose data type is int32 or int64.
            The rank must be at least 2.
        updates (Tensor): The tensor to update the input tensor, has the same type as input,
            and updates.shape should be equal to indices.shape[:-1] + input_x.shape[indices.shape[-1]:].

    Returns:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If dtype of `indices` is neither int32 nor int64.
        ValueError: If length of shape of `input_x` is less than the last dimension of shape of `indices`.
        RuntimeError: If a value of `indices` is not in `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [0, 0]]), mindspore.int32)
        >>> updates = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> output = ops.tensor_scatter_sub(input_x, indices, updates)
        >>> print(output)
        [[-3.3000002  0.3        3.6      ]
         [ 0.4        0.5       -3.2      ]]
    """

    return tensor_scatter_sub_(input_x, indices, updates)


def tensor_scatter_max(input_x, indices, updates):
    """
    By comparing the value at the position indicated by `indices` in `input_x` with the value in the `updates`,
    the value at the index will eventually be equal to the largest one to create a new tensor.

    The last axis of the index is the depth of each index vector. For each index vector,
    there must be a corresponding value in `updates`. The shape of `updates` should be
    equal to the shape of input_x[indices].

    Note:
        If some values of the `indices` are out of bound, instead of raising an index error,
        the corresponding `updates` will not be updated to `input_x`.

    Args:
        input_x (Tensor): The target tensor. The dimension of input_x must be no less than indices.shape[-1].
        indices (Tensor): The index of input tensor whose data type is int32 or int64.
            The rank must be at least 2.
        updates (Tensor): The tensor to update the input tensor, has the same type as input,
            and updates.shape should be equal to indices.shape[:-1] + input_x.shape[indices.shape[-1]:].

    Returns:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If dtype of `indices` is neither int32 nor int64.
        ValueError: If length of shape of `input_x` is less than the last dimension of shape of `indices`.
        RuntimeError: If a value of `indices` is not in `input_x`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [0, 0]]), mindspore.int32)
        >>> updates = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> # Next, demonstrate the approximate operation process of this operator:
        >>> # 1, indices[0] = [0, 0], indices[1] = [0, 0]
        >>> # 2, And input_x[0, 0] = -0.1
        >>> # 3, So input_x[indices] = [-0.1, -0.1]
        >>> # 4, Satisfy the above formula: input_x[indices].shape=(2) == updates.shape=(2)
        >>> output = ops.tensor_scatter_max(input_x, indices, updates)
        >>> # 5, Perform the max operation for the first time:
        >>> #      first_input_x = Max(input_x[0][0], updates[0]) = [[1.0, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> # 6, Perform the max operation for the second time:
        >>> #      second_input_x = Max(input_x[0][0], updates[1]) = [[2.2, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> print(output)
        [[ 2.2  0.3  3.6]
         [ 0.4  0.5 -3.2]]
    """
    return tensor_scatter_max_(input_x, indices, updates)


def tensor_scatter_min(input_x, indices, updates):
    """
    By comparing the value at the position indicated by `indices` in `input_x` with the value in the `updates`,
    the value at the index will eventually be equal to the smallest one to create a new tensor.

    The last axis of the index is the depth of each index vector. For each index vector,
    there must be a corresponding value in `updates`. The shape of `updates` should be
    equal to the shape of `input_x[indices]`. For more details, see case below.

    Note:
        If some values of the `indices` are out of range, instead of raising an index error,
        the corresponding `updates` will not be hw to `input_x`.

    Args:
        input_x (Tensor): The input tensor. The dimension of input_x must be no less than indices.shape[-1].
        indices (Tensor): The index of input tensor whose data type is int32 or int64.
            The rank must be at least 2.
        updates (Tensor): The tensor to update the input tensor, has the same type as input,
            and updates.shape should be equal to indices.shape[:-1] + input_x.shape[indices.shape[-1]:].

    Returns:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If dtype of `indices` is neither int32 nor int64.
        ValueError: If length of shape of `input_x` is less than the last dimension of shape of `indices`.
        RuntimeError: If a value of `indices` is not in `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [0, 0]]), mindspore.int32)
        >>> updates = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> output = ops.tensor_scatter_min(input_x, indices, updates)
        >>> print(output)
        [[ -0.1  0.3  3.6]
        [ 0.4  0.5 -3.2]]
    """
    return tensor_scatter_min_(input_x, indices, updates)


def tensor_scatter_elements(input_x, indices, updates, axis=0, reduction="none"):
    """
    Updates the value of the input tensor through the reduction operation.

    tensor_scatter_elements takes three inputs data, updates, and indices of the same rank r >= 1,
    an optional attribute axis that identifies an axis of data (default is 0),  and another optional attribute reduction
    that identifies reduction operation. When reduction is set to "none", the update value will be assigned to the
    output value according to the indices. When reduction is set to "add", the update value will be added to the output
    value according to the indices.

    For a 3-D tensor, the output is:

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

    .. note::
        If some values of the `indices` are out of bound, instead of raising an index error,
        the corresponding `updates` will not be updated to `input_x`.

    Args:
        input_x (Tensor): The target tensor.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        indices (Tensor): The index to do add operation whose data type must be mindspore.int32 or
          mindspore.int64. Same rank as input_x. And accepted range is [-s, s) where s is the size along axis.
        updates (Tensor): The tensor doing the add operation with `input_x`, has the same type as input_x,
          and update.shape should be equal to indices.shape.
        axis (int): Which axis to scatter, default is 0. Accepted range is [-r, r) where r = rank(input_x).
        reduction (str): Which reduction operation to scatter, default is "none". Other option: "add".

    Returns:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `indices` is neither int32 nor int64.
        ValueError: If anyone of the rank among `input_x`, `indices` and `updates` less than 1.
        ValueError: If the shape of `updates` is not equal to the shape of `indices`.
        ValueError: If the rank of `updates` is not equal to the rank of `input_x`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                        is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Parameter(Tensor(np.array([[1, 2, 3, 4, 5]]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2, 4]]), mindspore.int32)
        >>> updates = Tensor(np.array([[8, 8]]), mindspore.float32)
        >>> axis = 1
        >>> reduction = "none"
        >>> output = ops.tensor_scatter_elements(input_x, indices, updates, axis, reduction)
        >>> print(output)
        [[ 1  2  8  4  8]]
    """
    _tensor_scatter_elements = _get_cache_prim(TensorScatterElements)(axis, reduction)
    return _tensor_scatter_elements(input_x, indices, updates)


def scatter(x, axis, index, src):
    """
    Update the value in `src` to `x` according to the specified index.
    Refer to :func:`mindspore.ops.tensor_scatter_elements` for more details.

    Args:
        x (Tensor): The target tensor.
            The shape is :math:`(N,*)` where :math:`*` means any number of additional dimensions.
        axis (int): Which axis to scatter. Accepted range is [-r, r) where r = rank(x).
        index (Tensor): The index to do update operation whose data type must be mindspore.int32 or
            mindspore.int64. Same rank as `x` . And accepted range is [-s, s) where s is the size along axis.
        src (Tensor): The tensor doing the update operation with `x` , has the same type as `x` ,
            and the shape of `src` should be equal to the shape of `index` .

    Returns:
        Tensor, has the same shape and type as `x` .

    Raises:
        TypeError: If `index` is neither int32 nor int64.
        ValueError: If anyone of the rank among `x` , `index` and `src` less than 1.
        ValueError: If the shape of `src` is not equal to the shape of `index` .
        ValueError: If the rank of `src` is not equal to the rank of `x` .
        RuntimeError: If the data type of `x` and `src` conversion of Parameter
            is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1, 2, 3, 4, 5]]), dtype=ms.float32)
        >>> src = Tensor(np.array([[8, 8]]), dtype=ms.float32)
        >>> index = Tensor(np.array([[2, 4]]), dtype=ms.int64)
        >>> out = ops.scatter(x=x, axis=1, index=index, src=src)
        >>> print(out)
        [[1. 2. 8. 4. 8.]]
        >>> x = Tensor(np.zeros((5, 5)), dtype=ms.float32)
        >>> src = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=ms.float32)
        >>> index = Tensor(np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]]), dtype=ms.int64)
        >>> out = ops.scatter(x=x, axis=0, index=index, src=src)
        >>> print(out)
        [[1. 2. 3. 0. 0.]
        [0. 0. 0. 0. 0.]
        [4. 5. 6. 0. 0.]
        [0. 0. 0. 0. 0.]
        [7. 8. 9. 0. 0.]]
        >>> x = Tensor(np.zeros((5, 5)), dtype=ms.float32)
        >>> src = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=ms.float32)
        >>> index = Tensor(np.array([[0, 2, 4], [0, 2, 4], [0, 2, 4]]), dtype=ms.int64)
        >>> out = ops.scatter(x=x, axis=1, index=index, src=src)
        >>> print(out)
        [[1. 0. 2. 0. 3.]
        [4. 0. 5. 0. 6.]
        [7. 0. 8. 0. 9.]
        [0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0.]]
    """
    return F.tensor_scatter_elements(input_x=x, indices=index, updates=src, axis=axis)


def space_to_batch_nd(input_x, block_size, paddings):
    r"""
    Divides a tensor's spatial dimensions into blocks and combines the block sizes with the original batch.

    This operation will divide spatial dimensions into blocks with `block_size`,
    and after division, the output tensor's spatial dimension is the corresponding number of blocks.
    The output tensor's batch dimension is the product of the original batch and the product of `block_size`.
    Before division, the spatial dimensions of the input are zero padded according to paddings if necessary.
    Assume input shape is :math:`(n, c_1, ... c_k, w_1, ..., w_M)` with
    :math:`block\_size` and :math:`paddings`, then the shape of the output tensor will be
    :math:`(n', c_1, ... c_k, w'_1, ..., w'_M)`, where

    .. math::
        \begin{array}{ll} \\
            n' = n*(block\_size[0] * ... * block\_size[M]) \\
            w'_i = (w_i + paddings[i][0] + paddings[i][1])//block\_size[i]
        \end{array}

    Args:
        input_x (Tensor): The input tensor. It must be a 4-D tensor on Ascend.
        block_size (Union[list(int), tuple(int), int]): The block size of dividing block with all value greater
            than 1. If `block_size` is a tuple or list, the length of `block_size` is M corresponding to the
            number of spatial dimensions. If `block_size` is an int, the block size of M dimensions are the same,
            equal to `block_size`. M must be 2 on Ascend.
        paddings (Union[tuple, list]): The padding values for spatial dimensions, containing M subtraction list.
            Each contains 2 integer values. All values must be greater than 0.
            `paddings[i]` specifies the paddings for the spatial dimension i,
            which corresponds to the input dimension i + offset.
            It is required that input_shape[i+offset]+paddings[i][0]+paddings[i][1] is divisible by block_size[i].
            M must be 2 on Ascend.

    Returns:
        Tensor, the output tensor with the same data type as input.

    Raises:
        ValueError: If `block_size` is not one dimensional when `block_size` is a list or tuple.
        ValueError: If the length of `block_size` is not 2 on Ascend.
        ValueError: If the element of `block_size` is not an integer larger than 1.
        ValueError: If shape of `paddings` is not (M, 2), where M is the length of `block_size`.
        ValueError: If the element of `paddings` is not an integer larger than 0.
        TypeError: If `block_size` is not one of list, tuple, int.
        TypeError: If `paddings` is neither list nor tuple.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> block_size = [2, 2]
        >>> paddings = [[0, 0], [0, 0]]
        >>> input_x = Tensor(np.array([[[[1, 2], [3, 4]]]]), mindspore.float32)
        >>> output = ops.space_to_batch_nd(input_x, block_size, paddings)
        >>> print(output)
        [[[[1.]]]
         [[[2.]]]
         [[[3.]]]
         [[[4.]]]]
    """
    _space_to_batch_nd = _get_cache_prim(P.SpaceToBatchND)(block_size, paddings)
    return _space_to_batch_nd(input_x)


def batch_to_space_nd(input_x, block_shape, crops):
    r"""
    Divides batch dimension with blocks and interleaves these blocks back into spatial dimensions.

    This operation will divide batch dimension N into blocks with block_shape, the output tensor's N dimension
    is the corresponding number of blocks after division. The output tensor's :math:`w_1, ..., w_M` dimension is
    the product of original :math:`w_1, ..., w_M` dimension and block_shape with given amount to crop from dimension,
    respectively.

    If the input shape is :math:`(n, c_1, ... c_k, w_1, ..., w_M)`, the output shape is
    :math:`(n, c_1, ... c_k, w_1, ..., w_M)`.

    :math:`n' = n//(block\_shape[0]*...*block\_shape[M-1])`

    :math:`w'_i = w_i*block\_shape[i-1]-crops[i-1][0]-crops[i-1][1]`

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
        Tensor, the output tensor with the same type as input. Assume input shape is
        :math:`(n, c_1, ... c_k, w_1, ..., w_M)` with block_shape and crops. The output shape will be
        :math:`(n', c_1, ... c_k, w'_1, ..., w'_M)`, where

        :math:`n' = n//(block\_shape[0]*...*block\_shape[M-1])`

        :math:`w'_i = w_i*block\_shape[i-1]-crops[i-1][0]-crops[i-1][1]`

    Raises:
        TypeError: If `block_shape` is not one of list, tuple, int.
        TypeError: If `crops` is neither list nor tuple.
        ValueError: If `block_shape` is not one dimensional when `block_shape` is a list or tuple.
        ValueError: If the length of `block_shape` is not 2 on Ascend.
        ValueError: If the element of `block_shape` is not an integer larger than or euqal to 1.
        ValueError: If shape of `crops` is not (M, 2), where M is the length of `block_shape`.
        ValueError: If the element of `crops` is not an integer larger than or euqal to 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> block_shape = [2, 2]
        >>> crops = [[0, 0], [0, 0]]
        >>> input_x = Tensor(np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]]), mindspore.float32)
        >>> output = ops.batch_to_space_nd(input_x, block_shape, crops)
        >>> print(output)
        [[[[1.  2.]
           [3.  4.]]]]
    """
    if isinstance(block_shape, Tensor):
        _batch_to_space_ndv2 = _get_cache_prim(P.BatchToSpaceNDV2)()
        return _batch_to_space_ndv2(input_x, block_shape, crops)
    _batch_to_space_nd = _get_cache_prim(P.BatchToSpaceND)(block_shape, crops)
    return _batch_to_space_nd(input_x)


def nonzero(x):
    """
    Return a Tensor of the positions of all non-zero values.

    Args:
        x (Tensor): The shape of Tensor is :math:`(x_1, x_2, ..., x_R)`. The data type is int, float or bool.

    Returns:
        Tensor, a 2-D Tensor whose data type is int64, containing the positions of all non-zero values of the input.

    Raises:
        TypeError: If `x` is not Tensor.
        ValueError: If dim of `x` equals to 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.ops as ops
        >>> x = Tensor(np.array([[[1,  0], [-5, 0]]]), mindspore.int32)
        >>> output = ops.nonzero(x)
        >>> print(output)
        [[0 0 0]
         [0 1 0]]
    """
    return nonzero_(x)


def matrix_diag(x, k=0, num_rows=-1, num_cols=-1, padding_value=0, align="RIGHT_LEFT"):
    r"""
    Returns a Tensor with the contents in `x` as k[0]-th to k[1]-th diagonals of a matrix, with everything else padded
    with `padding_value`. `num_rows` and `num_cols` specify the dimension of the innermost matrix of the output. If both
    are not specified, the op assumes the innermost matrix of output Tensor is square and infers its size from `k` and
    the innermost dimension of `x`. If the `num_rows` and `num_cols` specify only one of them, the operator will derive
    the smallest legal value as the dimension of output. Moreover, when only one diagonal is given
    (k is an integer or k[0] == k[1]), the first to the second innermost dimension of `x` is the batch size. Otherwise,
    the second innermost dimension is not a part of batch size.

    Args:
        x (Tensor): The diagonal Tensor.
        k (Union[int, Tensor], optional): A Tensor of type int32. Diagonal offsets. Positive value means superdiagonal,
            0 refers to the main diagonal, and negative value means subdiagonals. `k` can be a single integer
            (for a single diagonal) or a pair of integers specifying the low and high ends of a matrix band.
            k[0] must not be larger than k[1]. The value must be in the range of given or derivated `num_rows`
            and `num_cols`, meaning value of k must be in (-num_rows, num_cols). Default: 0.
        num_rows (Union[int, Tensor], optional): A Tensor of type int32 with only one value. The number of rows of the
            output Tensor. If `num_rows` is -1, indicating that the innermost matrix of the output Tensor is a square
            matrix, and the real number of rows will be derivated by other inputs. That is
            :math:`num\_rows = x.shape[-1] - min(k[1], 0)`. Otherwise, the value must be equal or greater than
            :math:`x.shape[-1] - min(k[1], 0)`. Default: -1.
        num_cols (Union[int, Tensor], optional): A Tensor of type int32 with only one value.
            The number of columns of
            the output Tensor. If `num_cols` is -1, indicating that the innermost matrix of the output
            Tensor is a square matrix, and the real number of columns will be derivated by other inputs.
            That is :math:`num\_cols = x.shape[-1] + max(k[0], 0)`. Otherwise, the value must be equal or
            greater than :math:`x.shape[-1] - min(k[1], 0)`.  Default: -1.
        padding_value (Union[int, float, Tensor], optional): A Tensor with only one value. Have the same dtype as x.
            The number to fill the area outside the specified diagonal band.  Default: 0.
        align (str, optional): An optional string from: "RIGHT_LEFT"(default), "LEFT_RIGHT", "LEFT_LEFT",
            "RIGHT_RIGHT". Align is a string specifying how superdiagonals and subdiagonals should be aligned,
            respectively. "RIGHT_LEFT" aligns superdiagonals to the right (left-pads the row) and subdiagonals
            to the left (right-pads the row).

    Returns:
        A Tensor. Has the same type as `x`.
        Suppose `x` has r dimensions with shape `(I, J, ..., M, N)`. The output Tensor has rank r + 1 with shape
        `(I, J, ..., M, num_rows, num_cols)` when only one diagonal is given (k is an integer or k[0] == k[1]).
        Otherwise, it has rank r with shape `(I, J, ..., num_rows, num_cols)`.

    Raises:
        TypeError: If `x` is not Tensor.
        TypeError: If input `x` and `padding_value` are not the same dtype.
        TypeError: If `k`, `num_rows` or `num_cols` is not int32 dtype.
        ValueError: If rank of `k` is not equal to 0 or 1.
        ValueError: If rank of `num_rows`, `num_cols` or `padding_value` is not equal to 0.
        ValueError: If size of `k` is not equal to 1 or 2.
        ValueError: If the value of `k` is not in (-num_rows, num_cols).
        ValueError: If k[1] is not greater equal to k[0] when k[0] != k[1].
        ValueError: If rank of `x` is not greater than or is equal to 1 when k is an integer or k[0] == k[1].
        ValueError: If rank of `x` is not greater than or is equal to 2 when k[0] != k[1].
        ValueError: If x.shape[-2] is not equal to k[1] - k[0] + 1 when k[0] != k[1].
        ValueError: If `num_rows` and `num_cols` do not match the dimensions of `x` and the values of `k`.
        ValueError: If `align` is not a string or not in the valid set of values.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> x = Tensor(np.array([[8, 9, 0],
        ...                      [1, 2, 3],
        ...                      [0, 4, 5]]), mindspore.float32)
        >>> k =Tensor(np.array([-1, 1]), mindspore.int32)
        >>> num_rows = Tensor(np.array(3), mindspore.int32)
        >>> num_cols = Tensor(np.array(3), mindspore.int32)
        >>> padding_value = Tensor(np.array(11), mindspore.float32)
        >>> output = ops.matrix_diag(x, k, num_rows, num_cols, padding_value, align='LEFT_RIGHT')
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


def matrix_diag_part(x, k=0, padding_value=0, align="RIGHT_LEFT"):
    r"""
    Returns the diagonal part of input tensor.
    Returns a tensor with the k[0]-th to k[1]-th diagonals of `x`. Some diagonals are shorter than
    max_diag_len and need to be padded. Input k and padding_value must be const Tensor when taking Graph mode.

    Args:
        x (Tensor): The input Tensor with rank r, where r >= 2.
        k (Union[int, Tensor], optional): A Tensor of type int32. Diagonal offset(s). Positive value means
            superdiagonal, 0 refers to the main diagonal, and negative value means subdiagonals. k can be
            a single integer (for a single diagonal) or a pair of integers specifying the low and high ends
            of a matrix band. k[0] must not be larger than k[1]. The value of k has restructions, meaning
            value of k must be in (-x.shape[-2], x.shape[-1]).
        padding_value (Union[int, float, Tensor], optional): A Tensor with only one value. Have the same dtype as x.
            The number to fill the area outside the specified diagonal band. Default: 0.
        align (str, optional): An optional string from: "RIGHT_LEFT"(default), "LEFT_RIGHT", "LEFT_LEFT",
            "RIGHT_RIGHT". Align is a string specifying how superdiagonals and subdiagonals should be aligned,
            respectively. "RIGHT_LEFT" aligns superdiagonals to the right (left-pads the row) and subdiagonals
            to the left (right-pads the row).

    Returns:
        A Tensor. Has the same type as `x`.
        Assume `x` has r dimensions :math:`[I, J, ..., L, M, N]`. Let `max_diag_len` be the maximum length among all
        diagonals to be extracted, :math:`max\_diag\_len = min(M + min(k[1], 0), N + min(-k[0], 0))`
        Let `num_diags` be the number of diagonals to extract, :math:`num\_diags = k[1] - k[0] + 1`.
        If :math:`num\_diags == 1`, the output tensor is of rank r - 1 with shape :math:`[I, J, ..., L, max\_diag\_len]`
        Otherwise, the output tensor has rank r with dimensions :math:`[I, J, ..., L, num\_diags, max\_diag\_len]`

    Raises:
        TypeError: If `x` is not Tensor.
        TypeError: If input `x` and `padding_value` are not the same dtype.
        TypeError: If `k` is not int32 dtype.
        ValueError: If `align` is not a string or not in the valid range.
        ValueError: If rank of `k` is not equal to 0 or 1.
        ValueError: If rank of `padding_value` is not equal to 0.
        ValueError: If rank of `x` is not greater equal to 2.
        ValueError: If size of `k` is not equal to 1 or 2.
        ValueError: If k[1] is not greater equal to k[0] in case the size of `k` is 2.
        ValueError: If the value of `k` is not in (-x.shape[-2], x.shape[-1]).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1, 2, 3, 4],
        ...                      [5, 6, 7, 8],
        ...                      [9, 8, 7, 6]]), mindspore.float32)
        >>> k =Tensor(np.array([1, 3]), mindspore.int32)
        >>> padding_value = Tensor(np.array(9), mindspore.float32)
        >>> output = ops.matrix_diag_part(x, k, padding_value, align='RIGHT_LEFT')
        >>> print(output)
        [[9. 9. 4.]
         [9. 3. 8.]
         [2. 7. 6.]]
        >>> print(output.shape)
        (3, 3)
    """
    matrix_diag_part_v3 = _get_cache_prim(MatrixDiagPartV3)(align)
    return matrix_diag_part_v3(x, k, padding_value)


def matrix_set_diag(x, diagonal, k=0, align="RIGHT_LEFT"): # pylint: disable=redefined-outer-name
    r"""
    Returns a batched matrix tensor with new batched diagonal values.
    Given x and diagonal, this operation returns a tensor with the same shape and values as x, except for the specified
    diagonals of the innermost matrices. These will be overwritten by the values in diagonal. Some diagonals are shorter
    than max_diag_len and need to be padded.
    The diagonal :math:`shape[-2]` must be equal to num_diags calculated by :math:`k[1] - k[0] + 1`.
    The diagonal :math:`shape[-1]` must be
    equal to the longest diagonal value max_diag_len calculated
    by :math:`min(x.shape[-2] + min(k[1], 0), x.shape[-1] + min(-k[0], 0))`.
    Let x have r + 1 dimensions :math:`[I, J, ..., L, M, N]`.
    The diagonal tensor has rank r with shape :math:`[I, J, ..., L, max\_diag\_len]`
    when k is an integer or :math:`k[0] == k[1]`. Otherwise, it has rank r + 1
    with shape :math:`[I, J, ... L, num\_diags, max\_diag\_len]`.

    Args:
        x (Tensor): Rank r + 1, where r >= 1.
        diagonal (Tensor): A Tensor. Have the same dtype as x. Rank r when k is an integer or k[0] == k[1].
            Otherwise, it has rank r + 1.
        k (Union[int, Tensor], optional): A int32 Scalar or int32 Tensor. Diagonal offset(s). Positive value means
            superdiagonal, 0 refers to the main diagonal, and negative value means subdiagonals. k can be a
            single integer (for a single diagonal) or a pair of integers specifying the low and high ends of
            a matrix band. k[0] must not be larger than k[1].
            The alue of k has restructions, meaning value of k must be in (-x.shape[-2], x.shape[-1]).
            Input k must be const Tensor when taking Graph mode.
        align (str, optional): An optional string from: "RIGHT_LEFT"(default), "LEFT_RIGHT", "LEFT_LEFT",
            "RIGHT_RIGHT". Align is a string specifying how superdiagonals and subdiagonals should be aligned,
            respectively. "RIGHT_LEFT" aligns superdiagonals to the right (left-pads the row) and subdiagonals
            to the left (right-pads the row).

    Returns:
        Tensor, The same type as x. Let x has r+1 dimensions [I, J, ..., L, M, N].
        The output is a tensor of rank r+1 with dimensions [I, J, ..., L, M, N], the same as input x.

    Raises:
        TypeError: If input `x` or `diagonal` is not Tensor.
        TypeError: If input `x` and `diagonal` are not the same dtype.
        TypeError: If `k` is not int32 dtype.
        ValueError: If `align` is not a string or not in the valid range.
        ValueError: If rank of `k` is not equal to 0 or 1.
        ValueError: If rank of `x` is not greater equal to 2.
        ValueError: If size of `k` is not equal to 1 or 2.
        ValueError: If k[1] is not greater equal to k[0] in case the size of `k` is 2.
        ValueError: If the `diagonal` rank size don't match with input `x` rank size.
        ValueError: If the `diagonal` shape value don't match with input `x` shape value.
        ValueError: If the diagonal.shape[-2] is not equal to num_diags calculated by k[1] - k[0] + 1.
        ValueError: If the value of `k` is not in (-x.shape[-2], x.shape[-1]).
        ValueError: If the diagonal.shape[-1] is not equal to the max_diag_len calculated by min(x.shape[-2] + min(k[1],
            0), x.shape[-1] + min(-k[0], 0)).

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[7, 7, 7, 7],
        ...                      [7, 7, 7, 7],
        ...                      [7, 7, 7, 7]]), mindspore.float32)
        >>> diagonal = Tensor(np.array([[0, 9, 1],
        ...                             [6, 5, 8],
        ...                             [1, 2, 3],
        ...                             [4, 5, 0]]), mindspore.float32)
        >>> k = Tensor(np.array([-1, 2]), mindspore.int32)
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


def meshgrid(*inputs, indexing='xy'):
    """
    Generates coordinate matrices from given coordinate tensors.

    Given N one-dimensional coordinate tensors, returns a tuple outputs of N N-D
    coordinate tensors for evaluating expressions on an N-D grid.

    Args:
        inputs (List[Tensor]): List of 1-D tensors.
            The length of inputs should be greater than 1. The data type is Number.

    Keyword Args:
        indexing ('xy', 'ij', optional): Cartesian ('xy', default) or
            matrix ('ij') indexing of output. In the 2-D case with
            inputs of length `M` and `N`, the outputs are of shape `(N, M)`
            for 'xy' indexing and `(M, N)` for 'ij' indexing. In the 3-D
            case with inputs of length `M`, `N` and `P`, outputs are of shape
            `(N, M, P)` for 'xy' indexing and `(M, N, P)` for 'ij' indexing.  Default: 'xy'.

    Returns:
        Tensors, a Tuple of N N-D Tensor objects. The data type is the same with the Inputs.

    Raises:
        TypeError: If `indexing` is not a str or `inputs` is not a tuple.
        ValueError: If `indexing` is neither 'xy' nor 'ij'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.ops as ops
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
    meshgrid_op = _get_cache_prim(P.Meshgrid)(indexing)
    return meshgrid_op(inputs)


def affine_grid(theta, output_size, align_corners=False):
    r"""
    Returns a 2D or 3D flow field (sampling grid) based on `theta`, a batch of affine matrices.

    Args:
        theta (Tensor): The input tensor of flow field whose dtype is float16, float32.
            Input batch of affine matrices with shape [N, 2, 3] for 2D grid or [N, 3, 4] for 3D grid.
        output_size (tuple[int]): The target output image size.
            The value of target output with format [N, C, H, W] for 2D grid or [N, C, D, H, W] for 3D grid.
        align_corners (bool, optional): Geometrically, each pixel of input is viewed as a squqre instead of dot.
            If ``True``, consider extremum -1 and 1 referring to the centers of the pixels rather than pixel corners.
            The default value is ``False``, extremum -1 and 1 refer to the corners of the pixels, so that sampling is
            irrelevant to resolution of the image.
    Returns:
        Tensor, a tensor whose data type is same as 'theta', and the shape is [N, H, W, 2] for 2D grid
        or [N, D, H, W, 3] for 3D grid.

    Raises:
        TypeError: If `theta` is not a Tensor or `output_size` is not a tuple.
        ValueError: If the shape of `theta` is not [N, 2, 3] or [N, 3, 4].
        ValueError: If the size of `output_size` is not 4 or 5.
        ValueError: If the shape of `theta` is [N, 2, 3], the size of `output_size` is not 4;
                    If the shape of `theta` is [N, 3, 4], the size of `output_size` is not 5.
        ValueError: If the output_size[0] is not equal to the shape[0] of theta.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> import mindspore.ops as ops
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
    return affine_grid_op(theta, output_size)


def broadcast_to(x, shape):
    """
    Broadcasts input tensor to a given shape. The dim of input shape must be smaller
    than or equal to that of target shape. Suppose input shape is :math:`(x_1, x_2, ..., x_m)`,
    target shape is :math:`(*, y_1, y_2, ..., y_m)`, where :math:`*` means any additional dimension.
    The broadcast rules are as follows:

    Compare the value of :math:`x_m` and :math:`y_m`, :math:`x_{m-1}` and :math:`y_{m-1}`, ...,
    :math:`x_1` and :math:`y_1` consecutively and
    decide whether these shapes are broadcastable and what the broadcast result is.

    If the value pairs at a specific dim are equal, then that value goes right into that dim of output shape.
    With an input shape :math:`(2, 3)`, target shape :math:`(2, 3)` , the inferred output shape is :math:`(2, 3)`.

    If the value pairs are unequal, there are three cases:

    Case 1: If the value of the target shape in the dimension is -1, the value of the
    output shape in the dimension is the value of the corresponding input shape in the dimension.

    Case 2: If the value of target shape in the dimension is not -1, but the corresponding
    value in the input shape is 1, then the corresponding value of the output shape
    is that of the target shape. With an input shape :math:`(1, 3)`, target
    shape :math:`(8, 3)`, the output shape is :math:`(8, 3)`.

    Case 3: If the corresponding values of the two shapes do not satisfy the above cases,
    it means that broadcasting from the input shape to the target shape is not supported.

    So far we got the last m dims of the outshape, now focus on the first :math:`*` dims, there are
    two cases:

    If the first :math:`*` dims of output shape does not have -1 in it, then fill the input
    shape with ones until their length are the same, and then refer to
    Case 2 mentioned above to calculate the output shape. With target shape :math:`(3, 1, 4, 1, 5, 9)`,
    input shape :math:`(1, 5, 9)`, the filled input shape will be :math:`(1, 1, 1, 1, 5, 9)` and thus the
    output shape is :math:`(3, 1, 4, 1, 5, 9)`.

    If the first :math:`*` dims of output shape have -1 in it, it implies this -1 is conrresponding to
    a non-existing dim so they're not broadcastable. With target shape :math:`(3, -1, 4, 1, 5, 9)`,
    input shape :math:`(1, 5, 9)`, instead of operating the dim-filling process first, it raises errors directly.

    Args:
        x (Tensor): The input tensor.
                    The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.
        shape (tuple): The target shape to broadcast. Can be fully specified, or have -1 in one position
                       where it will be substituted by the input tensor's shape in that position, see example.

    Returns:
        Tensor, with the given `shape` and the same data type as `x`.

    Raises:
        TypeError: If `shape` is not a tuple.
        ValueError: If the target and input shapes are incompatible, or if a - 1 in the target shape is in an invalid
                    location.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> shape = (2, 3)
        >>> x = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> output = ops.broadcast_to(x, shape)
        >>> print(output)
        [[1. 2. 3.]
         [1. 2. 3.]]
        >>> shape = (-1, 2)
        >>> x = Tensor(np.array([[1], [2]]).astype(np.float32))
        >>> output = ops.broadcast_to(x, shape)
        >>> print(output)
        [[1. 1.]
         [2. 2.]]
    """
    if  isinstance(shape, Tensor) or F.is_sequence_value_unknown(shape):
        _dyn_broadcast_to = _get_cache_prim(DynamicBroadcastTo)()
        return _dyn_broadcast_to(x, shape)
    _broadcast_to = _get_cache_prim(P.BroadcastTo)(shape)
    return _broadcast_to(x)


def unsorted_segment_min(x, segment_ids, num_segments):
    r"""
    Computes the minimum of a tensor along segments.

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
        x (Tensor): The shape is :math:`(x_1, x_2, ..., x_R)`. With float16, float32 or int32 data type.
        segment_ids (Tensor): A `1-D` tensor whose shape is :math:`(x_1)`,
                              the value must be non-negative tensor. The data type must be int32.
        num_segments (int): The value specifies the number of distinct `segment_ids`.

    Returns:
        Tensor, set the number of `num_segments` as `N`, the shape is :math:`(N, x_2, ..., x_R)`.

    Raises:
        TypeError: If `num_segments` is not an int.
        ValueError: If length of shape of `segment_ids` is not equal to 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> import numpy as np
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
        >>> segment_ids = Tensor(np.array([0, 1, 1]).astype(np.int32))
        >>> num_segments = 2
        >>> output = ops.unsorted_segment_min(x, segment_ids, num_segments)
        >>> print(output)
        [[1. 2. 3.]
         [4. 2. 1.]]
    """
    unsorted_segment_min_ = P.UnsortedSegmentMin()
    return unsorted_segment_min_(x, segment_ids, num_segments)


def unsorted_segment_max(x, segment_ids, num_segments):
    r"""
    Computes the maximum along segments of a tensor.

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
        x (Tensor): The shape is :math:`(x_1, x_2, ..., x_R)`. With float16, float32 or int32 data type.
        segment_ids (Tensor): A `1-D` tensor whose shape is :math:`(x_1)`,
                              the value must be non-negative tensor. The data type must be int32.
        num_segments (int): The value specifies the number of distinct `segment_ids`.

    Returns:
        Tensor, set the number of `num_segments` as `N`, the shape is :math:`(N, x_2, ..., x_R)`.

    Raises:
        TypeError: If `num_segments` is not an int.
        ValueError: If length of shape of `segment_ids` is not equal to 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> import numpy as np
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
        >>> segment_ids = Tensor(np.array([0, 1, 1]).astype(np.int32))
        >>> num_segments = 2
        >>> output = ops.unsorted_segment_max(x, segment_ids, num_segments)
        >>> print(output)
        [[1. 2. 3.]
         [4. 5. 6.]]
    """
    unsorted_segment_max_ = P.UnsortedSegmentMax()
    return unsorted_segment_max_(x, segment_ids, num_segments)


def unsorted_segment_prod(x, segment_ids, num_segments):
    r"""
    Computes the product of a tensor along segments.

    The following figure shows the calculation process of unsorted_segment_prod:

    .. image:: UnsortedSegmentProd.png

    Note:
        - If the segment_id i is absent in the segment_ids, then output[i] will be filled with 1.
        - The `segment_ids` must be non-negative tensor.

    Args:
        x (Tensor): The shape is :math:`(x_1, x_2, ..., x_R)`. With float16, float32 or int32 data type.
        segment_ids (Tensor): A `1-D` tensor whose shape is :math:`(x_1)`,
                              the value must be non-negative tensor. The data type must be int32.
        num_segments (int): The value specifies the number of distinct `segment_ids`.

    Returns:
        Tensor, set the number of `num_segments` as `N`, the shape is :math:`(N, x_2, ..., x_R)`.

    Raises:
        TypeError: If `num_segments` is not an int.
        ValueError: If length of shape of `segment_ids` is not equal to 1.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> import numpy as np
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
        >>> segment_ids = Tensor(np.array([0, 1, 0]).astype(np.int32))
        >>> num_segments = 2
        >>> output = ops.unsorted_segment_prod(x, segment_ids, num_segments)
        >>> print(output)
        [[4. 4. 3.]
         [4. 5. 6.]]
    """
    unsorted_segment_prod_ = P.UnsortedSegmentProd()
    return unsorted_segment_prod_(x, segment_ids, num_segments)


def index_fill(x, axis, index, value):
    """
    Fills the elements under the `axis` dimension of the input Tensor `x` with the input `value`
    by selecting the indices in the order given in `index`.

    Args:
        x (Tensor): Input Tensor.  The supported data type is Number or Bool.
        axis (Union[int, Tensor]): Dimension along which to fill the input Tensor. Only supports
            an int number or a 0-dimensional Tensor, whose data type is int32 or int64.
        index (Tensor): Indices of the input Tensor to fill in. The dtype must be int32.
        value (Union[bool, int, float, Tensor]): Value to fill the returned Tensor. If `value` is
            a Tensor, it must be a 0-dimensional Tensor and has the same dtype as `x`. Otherwise,
            the `value` will be cast to a 0-dimensional Tensor with the same data type as `x`.

    Returns:
        Tensor, has the same dtype and shape as input Tensor.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `axis` is neither int number nor Tensor.
        TypeError: When `axis` is a Tensor, its dtype is not int32 or int64.
        TypeError: If `index` is not a Tensor.
        TypeError: If dtype of `index` is not int32.
        TypeError: If `value` is not a bool, int, float, or Tensor.
        TypeError: When `value` is a Tensor, the dtype of `x` and `value` are not the same.
        ValueError: If `axis` is a Tensor and its rank is not equal to 0.
        ValueError: If the rank of `index` is greater than 1D.
        ValueError: When `value` is a Tensor and its rank is not equal to 0.
        RuntimeError: If the value of `axis` is out the range of `[-x.ndim, x.ndim - 1]`.
        RuntimeError: If the values of `index` are out the range of `[-x.shape[axis], x.shape[axis]-1]`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32))
        >>> index = Tensor([0, 2], mindspore.int32)
        >>> value = Tensor(-2.0, mindspore.float32)
        >>> y = ops.index_fill(x, 1, index, value)
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


@constexpr
def _check_check_axis_in_range(axis, ndim):
    """Checks axes are with the bounds of ndim"""
    axis = validator.check_axis_in_range(axis, ndim)
    return axis


def index_select(x, axis, index):
    """
    Generates a new Tensor that accesses the values of `x` along the specified `axis` dimension
    using the indices specified in `index`. The new Tensor has the same number of dimensions as `x`,
    with the size of the `axis` dimension being equal to the length of `index`, and the size of all other
    dimensions will be unchanged from the original `x` Tensor.

    .. note::
        The value of index must be in the range of `[0, x.shape[axis])`, the result is undefined out of range.

    Args:
        x (Tensor): The input Tensor.
        axis (int): The dimension to be indexed.
        index (Tensor): A 1-D Tensor with the indices to access in `x` along the specified axis.

    Returns:
        Tensor, has the same dtype as input Tensor.

    Raises:
        TypeError: If `x` or `index` is not a Tensor.
        TypeError: If `axis` is not int number.
        ValueError: If the value of `axis` is out the range of `[-x.ndim, x.ndim - 1]`.
        ValueError: If the dimension of `index` is not equal to 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, ops
        >>> import numpy as np
        >>> x = Tensor(np.arange(16).astype(np.float32).reshape(2, 2, 4))
        >>> print(x)
        [[[ 0.  1.  2.  3.]
          [ 4.  5.  6.  7.]]
         [[ 8.  9. 10. 11.]
          [12. 13. 14. 15.]]]
        >>> index = Tensor([0,], mindspore.int32)
        >>> y = ops.index_select(x, 1, index)
        >>> print(y)
        [[[ 0.  1.  2.  3.]]
         [[ 8.  9. 10. 11.]]]
    """
    if not (isinstance(x, Tensor) and isinstance(index, Tensor)):
        raise TypeError(f"For 'index_select', inputs `x` and `index` must be all tensors.")
    if index.ndim != 1:
        raise ValueError(f"For 'index_select', the dimension of `index` must be 1, but got {index.ndim}")
    axis = _check_check_axis_in_range(axis, x.ndim)
    return gather_(x, index, axis)


def population_count(input_x):
    r"""
    Computes element-wise population count(a.k.a bitsum, bitcount).
    For each entry in `input_x`, calculates the number of 1 bits in the binary representation of that entry.

    Args:
        input_x (Tensor): Tensor of any dimension. The data type must be int16 or uint16 (Ascend).
            The data type must be int8, int16, int32, int64, uint8, uint16, uint32, uint64 (CPU and GPU).

    Returns:
        Tensor, with the same shape as the input, and the data type is uint8.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        TypeError: If dtype of `input_x` is not int16, uint16 (Ascend).
        TypeError: If dtype of `input_x` is not int8, int16, int32, int64, uint8, uint16, uint32, uint64 (CPU and GPU).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor([0, 1, 3], mindspore.int16)
        >>> output = ops.population_count(input_x)
        >>> print(output)
        [0 1 2]
    """
    return population_count_(input_x)


##############################
# Type Conversion Functions.
##############################


def is_tensor(obj):
    r"""
    Check whether the input object is a :class:`mindspore.Tensor` .

    Args:
        obj (Object): input object.

    Returns:
        Bool. Return True if `obj` is a Tensor, otherwise, return False.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> a = Tensor([1.9, 2.2, 3.1])
        >>> ops.is_tensor(a)
        True
    """
    return isinstance(obj, Tensor)


def scalar_cast(input_x, input_y):
    """
    Casts the input scalar to another type.

    Args:
        input_x (scalar): The input scalar. Only constant value is allowed.
        input_y (mindspore.dtype): The type to be cast. Only constant value is allowed.

    Returns:
        Scalar. The type is the same as the python type corresponding to `input_y`.

    Raises:
        TypeError: If neither `input_x` nor `input_y` is a constant value.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = ops.scalar_cast(255.0, mindspore.int32)
        >>> print(output)
        255
    """
    return scalar_cast_(input_x, input_y)


def tensor_scatter_mul(input_x, indices, updates):
    """
    Creates a new tensor by multiplying the values from the positions in `input_x` indicated by
    `indices`, with values from `updates`. When divided values are provided for the same
    index, the result of the update will multiply these values respectively. Except that
    the updates are applied on output `Tensor` instead of input `Parameter`.

    The last axis of `indices` is the depth of each index vectors. For each index vector,
    there must be a corresponding value in `updates`. The shape of `updates` should be
    equal to the shape of `input_x[indices]`. For more details, see use cases.

    Note:
        - If some values of the `indices` are out of bound, instead of raising an index error,
          the corresponding `updates` will not be updated to `input_x`.

    Args:
        input_x (Tensor): The target tensor. The dimension of input_x must be no less than indices.shape[-1].
        indices (Tensor): The index of input tensor whose data type is int32 or int64. The rank must be at least 2.
        updates (Tensor): The tensor to update the input tensor, has the same type as input,
            and updates shape should be equal to indices.shape[:-1] + input_x.shape[indices.shape[-1]:].

    Returns:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If dtype of `indices` is neither int32 nor int64.
        ValueError: If length of shape of `input_x` is less than the last dimension of shape of `indices`.
        RuntimeError: If a value of `indices` is not in `input_x`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [0, 0]]), mindspore.int32)
        >>> updates = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> # Next, demonstrate the approximate operation process of this operator:
        >>> # 1, indices[0] = [0, 0], indices[1] = [0, 0]
        >>> # 2, And input_x[0, 0] = -0.1
        >>> # 3, So input_x[indices] = [-0.1, -0.1]
        >>> # 4, Satisfy the above formula: input_x[indices].shape=(2) == updates.shape=(2)
        >>> # 5, Perform the multiply operation for the first time:
        >>> #      first_input_x = input_x[0][0] * updates[0] = [[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> # 6, Perform the multiply operation for the second time:
        >>> #      second_input_x = input_x[0][0] * updates[1] = [[-0.22, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> output = ops.tensor_scatter_mul(input_x, indices, updates)
        >>> print(output)
        [[-0.22  0.3   3.6  ]
         [ 0.4   0.5   -3.2 ]]
    """
    return tensor_scatter_mul_(input_x, indices, updates)


def tensor_scatter_div(input_x, indices, updates):
    """
    Creates a new tensor by dividing the values from the positions in `input_x` indicated by
    `indices`, with values from `updates`. When divided values are provided for the same
    index, the result of the update will be to divided these values respectively. Except that
    the updates are applied on output `Tensor` instead of input `Parameter`.

    The last axis of `indices` is the depth of each index vectors. For each index vector,
    there must be a corresponding value in `updates`. The shape of `updates` should be
    equal to the shape of `input_x[indices]`. For more details, see use cases.

    Note:
        - If some values of the `indices` are out of bound, instead of raising an index error,
          the corresponding `updates` will not be updated to `input_x`.
        - The operator can't handle division by 0 exceptions, so the user needs to make sure
          there is no 0 value in `updates`.

    Args:
        input_x (Tensor): The target tensor. The dimension of input_x must be no less than indices.shape[-1].
        indices (Tensor): The index of input tensor whose data type is int32 or int64.
            The rank must be at least 2.
        updates (Tensor): The tensor to update the input tensor, has the same type as input,
            and updates.shape should be equal to indices.shape[:-1] + input_x.shape[indices.shape[-1]:].

    Returns:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If dtype of `indices` is neither int32 nor int64.
        ValueError: If length of shape of `input_x` is less than the last dimension of shape of `indices`.
        RuntimeError: If a value of `indices` is not in `input_x`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor, nn, ops
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [0, 0]]), mindspore.int32)
        >>> updates = Tensor(np.array([1.0, 2.0]), mindspore.float32)
        >>> output = ops.tensor_scatter_div(input_x, indices, updates)
        >>> print(output)
        [[-0.05  0.3  3.6  ]
         [ 0.4   0.5  -3.2 ]]
    """
    return tensor_scatter_div_(input_x, indices, updates)


def scalar_to_tensor(input_x, dtype=mstype.float32):
    """
    Converts a scalar to a `Tensor`, and converts the data type to the specified type.

    Args:
        input_x (Union[bool, int, float]): The input is a scalar. Only constant value is allowed.
        dtype (mindspore.dtype): The target data type. Default: mindspore.float32. Only
            constant value is allowed.

    Returns:
        Tensor. 0-D Tensor and the content is the input.

    Raises:
        TypeError: If `input_x` is neither bool nor int nor float.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> data = 1
        >>> output = ops.scalar_to_tensor(data, mindspore.float32)
        >>> print(output)
        1.0
    """
    return scalar_to_tensor_(input_x, dtype)


def tuple_to_array(input_x):
    """
    Converts a tuple to a tensor.

    If the type of the first number in the tuple is integer, the data type of the output tensor is int.
    Otherwise, the data type of the output tensor is float.

    Args:
        input_x (tuple): A tuple of numbers. These numbers have the same type. Only constant value is allowed.
            The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.

    Returns:
        Tensor, if the input tuple contains `N` numbers, then the shape of the output tensor is (N,).

    Raises:
        TypeError: If `input_x` is not a tuple.
        ValueError: If length of `input_x` is less than or equal to 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = (1,2,3)
        >>> print(type(input_x))
        <class 'tuple'>
        >>> output = ops.tuple_to_array(input_x)
        >>> print(type(output))
        <class 'mindspore.common.tensor.Tensor'>
        >>> print(output)
        [1 2 3]
    """
    return tuple_to_array_(input_x)


def masked_select(x, mask):
    """
    Returns a new 1-D Tensor which indexes the `x` tensor according to the boolean `mask`.
    The shapes of the `mask` tensor and the `x` tensor don't need to match, but they must be broadcastable.

    Args:
        x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        mask (Tensor[bool]): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Returns:
        A 1-D Tensor, with the same type as `x`.

    Raises:
        TypeError: If `x` or `mask` is not a Tensor.
        TypeError: If dtype of `mask` is not bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> x = Tensor(np.array([1, 2, 3, 4]), mindspore.int64)
        >>> mask = Tensor(np.array([1, 0, 1, 0]), mindspore.bool_)
        >>> output = ops.masked_select(x, mask)
        >>> print(output)
        [1 3]
    """
    return masked_select_(x, mask)


def masked_fill(input_x, mask, value):
    """
    Fills elements of Tensor with value where mask is True.
    The shapes of `input_x` and `mask` need to be the same or broadcastable.

    Args:
        input_x (Tensor): The source Tensor whose data type is one of bool, uint8, int8, int16, int32,
                    int64, float16, float32, float64, complex64, complex128.
        mask (Tensor[bool]): The boolean mask.
        value (Union[float, Tensor]): The value to fill in with, which dtype is the same as `input_x`.

    Returns:
        Tensor, has the same type and shape as `input_x`.

    Raises:
        TypeError: If dtype of `mask` is not bool.
        TypeError: If `input_x` or `mask` is not a Tensor.
        ValueError: If the shapes of `input_x` and `mask` could not be broadcast.
        TypeError: If dtype of `input_x` or `value` is not one of bool, uint8, int8, int16, int32,
                   int64, float16, float32, float64, complex64, complex128.
        TypeError: If dtype of `value` is different from that of `input_x`.
        TypeError: If `value` is neither float number nor Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1., 2., 3., 4.]), mindspore.float32)
        >>> mask = Tensor(np.array([True, True, False, True]), mindspore.bool_)
        >>> output = ops.masked_fill(input_x, mask, 0.5)
        >>> print(output)
        [0.5 0.5 3.  0.5]
    """
    _fill = _get_cache_prim(P.FillV2)()
    if isinstance(value, (float, int)) and isinstance(input_x, Tensor):
        value = scalar_to_tensor_(value, input_x.dtype)
    masked_value = _fill(input_x.shape, value)
    return select(mask, masked_value, input_x)


def diag(input_x):
    r"""
    Constructs a diagonal tensor with a given diagonal values.

    Assume `input_x` has dimensions :math:`[D_1,... D_k]`, the output is a tensor of
    rank 2k with dimensions :math:`[D_1,..., D_k, D_1,..., D_k]` where:
    :math:`output[i_1,..., i_k, i_1,..., i_k] = input_x[i_1,..., i_k]` and 0 everywhere else.

    Args:
        input_x (Tensor): The input tensor.

    Returns:
        Tensor, has the same dtype as the `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        ValueError: If rank of `input_x` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> import mindspore.ops as ops
        >>> input_x = Tensor([1, 2, 3, 4]).astype('int32')
        >>> output = ops.diag(input_x)
        >>> print(output)
        [[1 0 0 0]
         [0 2 0 0]
         [0 0 3 0]
         [0 0 0 4]]
    """
    return diag_(input_x)


def diagflat(x, offset=0):
    r"""
    Create a 2-D Tensor which diagonal is the flattened `x` .

    Args:
        x (Tensor): Input Tensor, which is flattened and set as the diagonal of the output.
        offset (int, optional): `offset` controls which diagonal to choose. Default: 0.

            - When `offset` is zero, the diagonal chosen is the main diagonal.
            - When `offset` is a positive integer, the diagonal chosen is up the main diagonal.
            - When `offset` is a negative integer, the diagonal chosen is down the main diagonal.

    Returns:
        The 2-D Tensor, whose diagonal is the flattened `x`.

    Raises:
        TypeError: If `x` is not a tensor.
        TypeError: If `offset` is not an integer.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([1, 2], mindspore.float32)
        >>> output = ops.diagflat(x, 1)
        >>> print(output)
        [[0. 1. 0.]
         [0. 0. 2.]
         [0. 0. 0.]]
    """
    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError(f"For diagflat, the input x must be tensor, but got {type(x)}")
    if not isinstance(offset, int):
        raise TypeError(f"For diagflat, the offset must be int, but got {type(offset)}")
    offset_abs = abs(offset)
    if x.size == 0:
        return zeros((offset_abs, offset_abs), x.dtype)
    x = x.ravel()
    res = diag(x)
    if offset != 0:
        pad_y = zeros((x.size + offset_abs, offset_abs), x.dtype)
        pad_x = zeros((offset_abs, x.size), x.dtype)
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
        input_x (Tensor): 4D tensor with data type float16 or float.
        output_size (Tensor): 1D tensor with 2 elements of data type int.
        kernel_size (Union[int, tuple[int], list[int]]): The size of the kernel, should be two int
            for height and width. If type is int, it means that height equal with width. Must be specified.
        dilation (Union[int, tuple[int], list[int]]): The size of the dilation, should be two int
            for height and width. If type is int, it means that height equal with width. Default: 1.
        padding_value (Union[int, tuple[int], list[int]]): The size of the padding, should be two int
            for height and width. If type is int, it means that height equal with width. Default: 1.
        stride (Union[int, tuple[int], list[int]]): The size of the stride, should be two int
            for height and width. If type is int, it means that height equal with width. Default: 0.

    Returns:
        A 4D Tensor, with same type as 'input_x'.

    Raises:
        TypeError: If :attr:`kernel_size`, `dilation`, `padding_value`, `stride` data type is not in
            Union[int, tuple[int], list[int]].
        ValueError: If :attr:`kernel_size`, `dilation`, `padding_value`, `stride` value is not
            greater than zero or elements number more than 2.
        ValueError: If :attr:`padding_value` value is less than zero or elements number more than 2.
        ValueError: If input_x.shape[2] != kernel_size[0] * kernel_size[1].
        ValueError: If input_x.shape[3] does not match the calculated number of sliding blocks.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
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
    if split_size_or_sections > length_along_dim:
        res = P.Split(axis, 1)(x)
    elif length_along_dim % split_size_or_sections == 0:
        sections = length_along_dim // split_size_or_sections
        res = P.Split(axis, sections)(x)
    else:
        num_sections = length_along_dim // split_size_or_sections
        length1 = num_sections * split_size_or_sections
        length2 = length_along_dim - length1
        start1 = _list_comprehensions(rank(x), 0, True)
        size1 = _tuple_setitem(arr_shape, axis, length1)
        start2 = _tuple_setitem(start1, axis, length1)
        size2 = _tuple_setitem(arr_shape, axis, length2)
        res = P.Split(axis, num_sections)(tensor_slice(x, start1, size1)) + \
              P.Split(axis, 1)(tensor_slice(x, start2, size2))
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
    for i, idx in enumerate(new_indices):
        begin[axis] = 0 if i == 0 else new_indices[i - 1]
        end[axis] = idx
        sliced_tensor = strided_slice(x, tuple(begin), tuple(end), strides)
        sub_tensors.append(sliced_tensor)
    return sub_tensors


def split(x, split_size_or_sections, axis=0):
    """
    Splits the Tensor into chunks along the given axis.

    Args:
        x (Tensor): A Tensor to be divided.
        split_size_or_sections (Union[int, tuple(int), list(int)]):
            If `split_size_or_sections` is an int type, `x` will be split into equally sized chunks, each chunk with
            size `split_size_or_sections`. Last chunk will be smaller than `split_size_or_sections` if `x.shape[axis]`
            is not divisible by `split_size_or_sections`.
            If `split_size_or_sections` is a list type, then `x` will be split into len(split_size_or_sections)
            chunks with sizes `split_size_or_sections` along the given `axis`.
        axis (int): The axis along which to split. Default: 0.

    Returns:
        A tuple of sub-tensors.

    Raises:
        TypeError: If argument `x` is not Tensor.
        TypeError: If argument `axis` is not Tensor.
        ValueError: If argument `axis` is out of range of :math:`[-x.ndim, x.ndim)` .
        TypeError: If each element in 'split_size_or_sections' is not integer.
        TypeError: If argument `indices_or_sections` is not int, tuple(int) or list(int).
        ValueError: The sum of 'split_size_or_sections' is not equal to x.shape[axis].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = np.arange(9).astype("float32")
        >>> output = ops.split(Tensor(input_x), 3)
        >>> print(output)
        (Tensor(shape=[3], dtype=Float32, value= [ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value= [ 3.00000000e+00,  4.00000000e+00,  5.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value= [ 6.00000000e+00,  7.00000000e+00,  8.00000000e+00]))
    """
    if not isinstance(x, Tensor):
        raise TypeError(f'expect `x` is a Tensor, but got {type(x)}')
    if type(axis) is not int:
        raise TypeError(f"Type of Argument `axis` should be integer but got {type(axis)}")
    axis = _canonicalize_axis(axis, x.ndim)

    if type(split_size_or_sections) is int:
        if split_size_or_sections > 0:
            res = _split_int(x, split_size_or_sections, axis)
        else:
            raise ValueError(f"For split, the value of 'split_size_or_sections' must be more than zero, "
                             f"but got {split_size_or_sections}.")
    elif isinstance(split_size_or_sections, (list, tuple)):
        for item in split_size_or_sections:
            if type(item) is not int:
                raise TypeError(f"Each element in 'split_size_or_sections' should be integer, but got {type(item)}.")
            if item < 0:
                raise TypeError(f"Each element in 'split_size_or_sections' should be non-negative, "
                                f"but got {split_size_or_sections}.")

        if sum(split_size_or_sections) != x.shape[axis]:
            raise ValueError(f"The sum of 'split_size_or_sections' should be equal to {x.shape[axis]}, "
                             f"but got {sum(split_size_or_sections)}.")
        res = _split_sub_tensors(x, split_size_or_sections, axis)
    else:
        raise TypeError(f"Type of Argument `split_size_or_sections` should be integer, tuple(int) or list(int), " \
                        f"but got {type(split_size_or_sections)}")
    return tuple(res)


def tril(input_x, diagonal=0): # pylint: disable=redefined-outer-name
    """
    Returns the lower triangle part of 'input_x' (elements that contain the diagonal and below),
    and set the other elements to zeros.

    Args:
        input_x (Tensor): A Tensor with shape :math:`(x_1, x_2, ..., x_R)`. The rank must be at least 2.
          Supporting all number types including bool.
        diagonal (int, optional): An optional attribute indicates the diagonal to consider, default: 0,
            indicating the main diagonal.

    Returns:
        Tensor, the same shape and data type as the input `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `diagonal` is not an int.
        TypeError: If the type of `x` is neither number nor bool.
        ValueError: If the rank of `x` is less than 2.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[ 1,  2,  3,  4],
        ...                      [ 5,  6,  7,  8],
        ...                      [10, 11, 12, 13],
        ...                      [14, 15, 16, 17]]))
        >>> result = ops.tril(x)
        >>> print(result)
        [[ 1  0  0  0]
         [ 5  6  0  0]
         [10 11 12  0]
         [14 15 16 17]]
        >>> x = Tensor(np.array([[ 1,  2,  3,  4],
        ...                      [ 5,  6,  7,  8],
        ...                      [10, 11, 12, 13],
        ...                      [14, 15, 16, 17]]))
        >>> result = ops.tril(x, diagonal=1)
        >>> print(result)
        [[ 1  2  0  0]
         [ 5  6  7  0]
         [10 11 12 13]
         [14 15 16 17]]
        >>> x = Tensor(np.array([[ 1,  2,  3,  4],
        ...                      [ 5,  6,  7,  8],
        ...                      [10, 11, 12, 13],
        ...                      [14, 15, 16, 17]]))
        >>> result = ops.tril(x, diagonal=-1)
        >>> print(result)
        [[ 0  0  0  0]
         [ 5  0  0  0]
         [10 11  0  0]
         [14 15 16  0]]
    """
    tril_ = Tril(diagonal)
    return tril_(input_x)


@constexpr
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
            raise ValueError(f'axis {ax} is out of bounds for array of dimension {ndim}')

    def canonicalizer(ax):
        return ax + ndim if ax < 0 else ax

    axis = tuple([canonicalizer(ax) for ax in axis])
    if all(axis.count(el) <= 1 for el in axis):
        return tuple(sorted(axis)) if len(axis) > 1 else axis[0]
    raise ValueError(f"duplicate axis in {axis}.")


@constexpr
def _list_comprehensions(obj, item=None, return_tuple=False):
    """
    Generates a new list or tuple by list comprehension.

    Args:
        obj (Union[int, list, tuple]):
            If integer, it will be the length of the returned tuple/list.
        item: The value to be filled. Default: None.
            If None, the values in the new list/tuple are the same as obj
            or range(obj) when obj is integer.
        return_tuple(bool): If true, returns tuple, else returns list.

    Returns:
        List or tuple.
    """
    lst = obj
    if isinstance(obj, int):
        lst = np.arange(obj)
    if item is None:
        res = list(lst)
    else:
        res = [item for _ in lst]
    if return_tuple:
        return tuple(res)
    return res


@constexpr
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
    for i, idx in enumerate(indices_or_sections):
        begin[axis] = 0 if i == 0 else indices_or_sections[i - 1]
        end[axis] = idx
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
    if indices_or_sections > length_along_dim:
        res = P.Split(axis, length_along_dim)(x)
        indices_or_sections_n = [length_along_dim, length_along_dim + 1]
        res2 = _tensor_split_sub_tensors(x, indices_or_sections_n, axis)
        for _ in np.arange(length_along_dim, indices_or_sections):
            res += tuple(res2)[1:]
    elif length_along_dim % indices_or_sections == 0:
        res = P.Split(axis, indices_or_sections)(x)
    else:
        num_long_tensor = length_along_dim % indices_or_sections
        num_short_tensor = indices_or_sections - num_long_tensor
        length1 = num_long_tensor * (length_along_dim // indices_or_sections + 1)
        length2 = length_along_dim - length1
        start1 = _list_comprehensions(rank(x), 0, True)
        size1 = _tuple_setitem(arr_shape, axis, length1)
        start2 = _tuple_setitem(start1, axis, length1)
        size2 = _tuple_setitem(arr_shape, axis, length2)
        res = P.Split(axis, num_long_tensor)(tensor_slice(x, start1, size1)) + \
              P.Split(axis, num_short_tensor)(tensor_slice(x, start2, size2))
    return res


def tensor_split(x, indices_or_sections, axis=0):
    r"""
    Splits a tensor into multiple sub-tensors along the given axis.

    Args:
        x (Tensor): A Tensor to be divided.
        indices_or_sections (Union[int, tuple(int), list(int)]):

            - If `indices_or_sections` is an integer n, input tensor will be split into n sections.

              - If :math:`x.size(axis)` can be divisible by n, sub-sections will have equal size
                :math:`x.size(axis) / n` .
              - If :math:`x.size(axis)` is not divisible by n, the first :math:`x.size(axis) % n` sections
                will have size :math:`x.size(axis) // n + 1` , and the rest will have size :math:`x.size(axis) // n` .

            - If `indices_or_sections` is of type tuple(int) or list(int), the input tensor will be split at the
              indices in the list or tuple. For example, given parameters :math:`indices\_or\_sections=[1, 4]`
              and :math:`axis=0` , the input tensor will be split into sections :math:`x[:1]` , :math:`x[1:4]` ,
              and :math:`x[4:]` .

        axis (int): The axis along which to split. Default: 0.

    Returns:
        A tuple of sub-tensors.

    Raises:
        TypeError: If argument `x` is not Tensor.
        TypeError: If argument `axis` is not int.
        ValueError: If argument `axis` is out of range of :math:`[-x.ndim, x.ndim)` .
        TypeError: If each element in 'indices_or_sections' is not integer.
        TypeError: If argument `indices_or_sections` is not int, tuple(int) or list(int).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = np.arange(9).astype("float32")
        >>> output = ops.tensor_split(Tensor(input_x), 3)
        >>> print(output)
        (Tensor(shape=[3], dtype=Float32, value= [ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00]),
        Tensor(shape=[3], dtype=Float32, value= [ 3.00000000e+00,  4.00000000e+00,  5.00000000e+00]),
        Tensor(shape=[3], dtype=Float32, value= [ 6.00000000e+00,  7.00000000e+00,  8.00000000e+00]))
    """
    if not isinstance(x, Tensor):
        raise TypeError(f'expect `x` is a Tensor, but got {type(x)}')

    if type(axis) is not int:
        raise TypeError(f"Type of Argument `axis` should be integer but got {type(axis)}")
    axis = _canonicalize_axis(axis, x.ndim)
    if type(indices_or_sections) is int:
        if indices_or_sections > 0:
            res = _tensor_split_sub_int(x, indices_or_sections, axis)
        else:
            raise ValueError(f"For tensor_split, the value of 'indices_or_sections' must be more than zero "
                             f"but got {indices_or_sections}")
    elif isinstance(indices_or_sections, (list, tuple)):
        for item in indices_or_sections:
            if type(item) is not int:
                raise TypeError(f"Each element in 'indices_or_sections' should be integer, but got {type(item)}.")
        res = _tensor_split_sub_tensors(x, indices_or_sections, axis)
    else:
        raise TypeError(f"Type of Argument `indices_or_sections` should be integer, tuple(int) or list(int), " \
                        f"but got {type(indices_or_sections)}")

    return res


def vsplit(x, indices_or_sections):
    """
    Splits a tensor into multiple sub-tensors vertically.
    It is equivalent to `ops.tensor_split` with :math:`axis=0` .

    Args:
        x (Tensor): A Tensor to be divided.
        indices_or_sections (Union[int, tuple(int), list(int)]): See argument in :func:`mindspore.ops.tensor_split`.

    Returns:
        A list of sub-tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = np.arange(9).reshape((3, 3)).astype('float32')
        >>> output = ops.vsplit(Tensor(input_x), 3)
        >>> print(output)
        (Tensor(shape=[1, 3], dtype=Float32, value=[[ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00]]),
         Tensor(shape=[1, 3], dtype=Float32, value=[[ 3.00000000e+00,  4.00000000e+00,  5.00000000e+00]]),
         Tensor(shape=[1, 3], dtype=Float32, value=[[ 6.00000000e+00,  7.00000000e+00,  8.00000000e+00]]))
    """
    if not isinstance(x, Tensor):
        raise TypeError(f'expect `x` is a Tensor, but got {type(x)}')
    if x.ndim < 1:
        raise ValueError(f'vsplit expect `x` is a Tensor with at least 1 dimension, but got {x.ndim}')
    return tensor_split(x, indices_or_sections, 0)


def hsplit(x, indices_or_sections):
    """
    Splits a tensor into multiple sub-tensors horizontally.
    It is equivalent to `ops.tensor_split` with :math:`axis=1` .

    Args:
        x (Tensor): A Tensor to be divided.
        indices_or_sections (Union[int, tuple(int), list(int)]): See argument in :func:`mindspore.ops.tensor_split`.

    Returns:
        A list of sub-tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = np.arange(6).reshape((2, 3)).astype('float32')
        >>> output = ops.hsplit(Tensor(input_x), 3)
        >>> print(output)
        (Tensor(shape=[2, 1], dtype=Float32, value=[[ 0.00000000e+00], [ 3.00000000e+00]]),
         Tensor(shape=[2, 1], dtype=Float32, value=[[ 1.00000000e+00], [ 4.00000000e+00]]),
         Tensor(shape=[2, 1], dtype=Float32, value=[[ 2.00000000e+00], [ 5.00000000e+00]]))
    """
    if not isinstance(x, Tensor):
        raise TypeError(f'expect `x` is a Tensor, but got {type(x)}')
    if x.ndim < 2:
        raise ValueError(f'hsplit expect `x` is a Tensor with at least 2 dimension, but got {x.ndim}')

    return tensor_split(x, indices_or_sections, 1)


def dsplit(x, indices_or_sections):
    """
    Splits a tensor into multiple sub-tensors along the 3rd axis.
    It is equivalent to `ops.tensor_split` with :math:`axis=2` .

    Args:
        x (Tensor): A Tensor to be divided.
        indices_or_sections (Union[int, tuple(int), list(int)]): See argument in :func:`mindspore.ops.tensor_split`.

    Returns:
        A list of sub-tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = np.arange(6).reshape((1, 2, 3)).astype('float32')
        >>> output = ops.dsplit(Tensor(input_x), 3)
        >>> print(output)
        (Tensor(shape=[1, 2, 1], dtype=Float32, value=[[[ 0.00000000e+00], [ 3.00000000e+00]]]),
         Tensor(shape=[1, 2, 1], dtype=Float32, value=[[[ 1.00000000e+00], [ 4.00000000e+00]]]),
         Tensor(shape=[1, 2, 1], dtype=Float32, value=[[[ 2.00000000e+00], [ 5.00000000e+00]]]))
    """
    if not isinstance(x, Tensor):
        raise TypeError(f'expect `x` is a Tensor, but got {type(x)}')
    if x.ndim < 3:
        raise ValueError(f'dsplit expect `x` is a Tensor with at least 3 dimension, but got {x.ndim}')

    return tensor_split(x, indices_or_sections, 2)


def max(x, axis=0, keep_dims=False):
    """
    Calculates the maximum value along with the given axis for the input tensor. It returns the maximum values and
    indices.

    Note:
        In auto_parallel and semi_auto_parallel mode, the first output index can not be used.

    .. warning::
        - If there are multiple maximum values, the index of the first maximum value is used.
        - The value range of "axis" is [-dims, dims - 1]. "dims" is the dimension length of "x".

    Also see: :class:`mindspore.ops.ArgMaxWithValue`.

    Args:
        x (Tensor): The input tensor, can be any dimension. Set the shape of input tensor as
          :math:`(x_1, x_2, ..., x_N)`.
        axis (int): The dimension to reduce. Default: 0.
        keep_dims (bool): Whether to reduce dimension, if true, the output will keep same dimension with the input,
                          the output will reduce dimension if false. Default: False.

    Returns:
        tuple (Tensor), tuple of 2 tensors, containing the corresponding index and the maximum value of the input
        tensor.

        - index (Tensor) - The index for the maximum value of the input tensor, with dtype int32. If `keep_dims`
          is true, the shape of output tensors is :math:`(x_1, x_2, ..., x_{axis-1}, 1, x_{axis+1}, ..., x_N)`.
          Otherwise, the shape is :math:`(x_1, x_2, ..., x_{axis-1}, x_{axis+1}, ..., x_N)` .
        - values (Tensor) - The maximum value of input tensor, with the same shape as index, and same dtype as x.

    Raises:
        TypeError: If `x` is not Tensor.
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `axis` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), mindspore.float32)
        >>> index, output = ops.max(x)
        >>> print(index, output)
        3 0.7
        >>> index, output = ops.max(x, keep_dims=True)
        >>> print(index, output)
        [3] [0.7]
    """
    if x.shape == ():
        return (Tensor(0), x)
    argmax_with_value_op = ArgMaxWithValue(axis, keep_dims)
    return argmax_with_value_op(x)


def argmax(input, dim=None, keepdim=False):
    """
    Return the indices of the maximum values of a tensor across a dimension.

    Args:
        input (Tensor): Input tensor.
        dim (Union[int, None]): The dimension to reduce. If `dim` is None, the indices of the maximum
            value within the flattened input will be returned. Default: None.
        keepdim (bool): Whether the output tensor retains the specified
            dimension. Ignored if `dim` is None. Default: False.

    Returns:
        Tensor, indices of the maximum values across a dimension.

    Raises:
        ValueError: If `dim` is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1, 20, 5], [67, 8, 9], [130, 24, 15]]).astype(np.float32))
        >>> output = ops.argmax(x, dim=-1)
        >>> print(output)
        [1 0 0]
    """
    if input.shape == ():
        return Tensor(0)
    is_dim_none = False
    if dim is None:
        input = reshape_(input, (-1,))
        dim = 0
        is_dim_none = True
    out = _get_cache_prim(Argmax)(dim, mstype.int64)(input)
    if keepdim and not is_dim_none:
        out = expand_dims_(out, dim)
    return out


def min(x, axis=0, keep_dims=False):
    """
    Calculates the minimum value along with the given axis for the input tensor. It returns the minimum values and
    indices.

    Note:
        In auto_parallel and semi_auto_parallel mode, the first output index can not be used.

    .. warning::
        - If there are multiple minimum values, the index of the first minimum value is used.
        - The value range of "axis" is [-dims, dims - 1]. "dims" is the dimension length of "x".

    Args:
        x (Tensor) - The input tensor, can be any dimension. Set the shape of input tensor as
          :math:`(x_1, x_2, ..., x_N)` .Complex tensor is not supported.
        axis (int): The dimension to reduce. Default: 0.
        keep_dims (bool): Whether to reduce dimension, if true the output will keep the same dimension as the input,
                          the output will reduce dimension if false. Default: False.

    Returns:
        tuple (Tensor), tuple of 2 tensors, containing the corresponding index and the minimum value of the input
        tensor.

        - **index** (Tensor) - The index for the minimum value of the input tensor, with dtype int32. If `keep_dims`
          is true, the shape of output tensors is :math:`(x_1, x_2, ..., x_{axis-1}, 1, x_{axis+1}, ..., x_N)`.
          Otherwise, the shape is :math:`(x_1, x_2, ..., x_{axis-1}, x_{axis+1}, ..., x_N)` .
        - **values** (Tensor) - The minimum value of input tensor, with the same
          shape as `index`, and same dtype as `x`.

    Raises:
        TypeError: If `x` is not Tensor.
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `axis` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), mindspore.float32)
        >>> index, output = ops.min(x)
        >>> print(index, output)
        0 0.0
        >>> index, output = ops.min(x, keep_dims=True)
        >>> print(index, output)
        [0] [0.0]
    """
    if x.shape == ():
        return (Tensor(0), x)
    argmin_with_value_ = ArgMinWithValue(axis=axis, keep_dims=keep_dims)
    return argmin_with_value_(x)


def aminmax(x, *, axis=0, keepdims=False):
    """
    It returns the minimum and maximum value along the given axis of input tensor.

    Args:
        x (Tensor): The input tensor, can be any dimension. Set the shape of input tensor as
          :math:`(x_1, x_2, ..., x_N)` .

    Keyword Args:
        axis (int, optional): The dimension to reduce. The value range of `axis` is [-rank, rank),
            where "rank" is the dimension of `x`. Default: 0.
        keepdims (bool, optional): Whether to maintain dimension. When set to True, the output will keep the same
            dimension as the input, or the dimension specified by `axis` is reduced. Default: False.

    Returns:
        tuple (Tensor), containing the minimum value and maximum value of the input tensor.

        - If `keepdims` is True, the shape of output tensors is
          :math:`(x_1, x_2, ..., x_{axis-1}, 1, x_{axis+1}, ..., x_N)`.
        - If `keepdims` is False, the shape of output tensors is
          :math:`(x_1, x_2, ..., x_{axis-1}, x_{axis+1}, ..., x_N)`.

    Raises:
        TypeError: If `keepdims` is not a bool.
        TypeError: If `axis` is not an int.
        ValueError: If `axis` is not in range [-rank, rank).

    Supported Platforms:
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), mindspore.float32)
        >>> output0, output1 = ops.aminmax(x)
        >>> print(output0, output1)
        0.0 0.7
    """
    argmin_with_value_op = P.ArgMinWithValue(axis, keepdims)
    argmax_with_value_op = P.ArgMaxWithValue(axis, keepdims)
    _, output0 = argmin_with_value_op(x)
    _, output1 = argmax_with_value_op(x)
    return output0, output1


def narrow(inputs, axis, start, length):
    """
    Returns a narrowed tensor from input tensor, and
    the dimension axis is input from start to start + length.

    Args:
        inputs (Tensor): the tensor to narrow.
        axis (int): the axis along which to narrow.
        start (int): the starting dimension.
        length (int): the distance to the ending dimension.

    Returns:
        Tensor.

        - output (Tensors) - The narrowed tensor.

    Raises:
        TypeError: If the input is not a tensor or tuple or list of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mindspore.int32)
        >>> output = ops.narrow(x, 0, 0, 2)
        >>> print(output)
        [[ 1 2 3]
         [ 4 5 6]]
        >>> output = ops.narrow(x, 1, 1, 2)
        >>> print(output)
        [[ 2 3]
         [ 5 6]
         [ 8 9]]
    """
    validator.check_axis_in_range(axis, inputs.ndim)
    validator.check_int_range(start, 0, inputs.shape[axis], Rel.INC_LEFT)
    validator.check_int_range(length, 1, inputs.shape[axis] - start, Rel.INC_BOTH)

    begins = [0] * inputs.ndim
    begins[axis] = start
    sizes = [i for i in inputs.shape]
    sizes[axis] = length
    return P.Slice()(inputs, begins, sizes)


def unsorted_segment_sum(input_x, segment_ids, num_segments):
    r"""
    Computes the sum of a tensor along segments.

    Calculates a tensor such that :math:`\text{output}[i] = \sum_{segment\_ids[j] == i} \text{data}[j, \ldots]`, where
    :math:`j` is a tuple describing the index of element in data.  `segment_ids` selects which elements in data to sum
    up. Segment_ids does not need to be sorted, and it does not need to cover all values in the entire valid value
    range.

    The following figure shows the calculation process of unsorted_segment_sum:

    .. image:: UnsortedSegmentSum.png

    Note:
        - If the segment_id i is absent in the segment_ids, then output[i] will be filled with 0.
        - On Ascend, if the value of segment_id is less than 0 or greater than the length of the input data shape, an
          execution error will occur.

    If the sum of the given segment_ids :math:`i` is empty, then :math:`\text{output}[i] = 0`. If the given segment_ids
    is negative, the value will be ignored. 'num_segments' must be equal to the number of different segment_ids.

    Args:
        input_x (Tensor): The shape is :math:`(x_1, x_2, ..., x_R)`.
        segment_ids (Tensor): Set the shape as :math:`(x_1, x_2, ..., x_N)`, where 0 < N <= R.
        num_segments (Union[int, Tensor], optional): Set :math:`z` as num_segments.

    Returns:
        Tensor, the shape is :math:`(z, x_{N+1}, ..., x_R)`.

    Raises:
        TypeError: If `num_segments` is not an int or 0-D Tensor.
        ValueError: If length of shape of `segment_ids` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> import mindspore
        >>> input_x = Tensor([1, 2, 3, 4], mindspore.float32)
        >>> segment_ids = Tensor([0, 0, 1, 2], mindspore.int32)
        >>> num_segments = 4
        >>> output = ops.unsorted_segment_sum(input_x, segment_ids, num_segments)
        >>> print(output)
        [3. 3. 4. 0.]
        >>> input_x = Tensor([1, 2, 3, 4, 2, 5], mindspore.float32)
        >>> segment_ids = Tensor([0, 0, 1, 2, 3, 4], mindspore.int32)
        >>> num_segments = 6
        >>> output = ops.unsorted_segment_sum(input_x, segment_ids, num_segments)
        >>> print(output)
        [3. 3. 4. 2. 5. 0.]
    """
    return unsorted_segment_sum_(input_x, segment_ids, num_segments)


def topk(input_x, k, dim=None, largest=True, sorted=True):
    r"""
    Finds values and indices of the `k` largest or smallest entries along a given dimension.

    .. warning::
        - If sorted is set to False, it will use the aicpu operator, the performance may be reduced. In addition, due to
          different memory layout and traversal methods on different platforms, the display order of calculation results
          may be inconsistent when `sorted` is False.

    If the `input_x` is a one-dimensional Tensor, finds the `k` largest  or smallest entries in the Tensor,
    and outputs its value and index as a Tensor. values[`k`] is the `k` largest item in `input_x`,
    and its index is indices [`k`].

    For a multi-dimensional matrix,
    calculates the first or last `k` entries in a given dimension, therefore:

    .. math::

        values.shape = indices.shape

    If the two compared elements are the same, the one with the smaller index value is returned first.

    Args:
        input_x (Tensor): Input to be computed, data type must be float16, float32 or int32.
        k (int): The number of top or bottom elements to be computed along the last dimension, constant input is needed.
        dim (int, optional): The dimension to sort along. Default: None.
        largest (bool, optional): If largest is False then the k smallest elements are returned. Default: True.
        sorted (bool, optional): If True, the obtained elements will be sorted by the values in descending order.
            If False, the obtained elements will not be sorted. Default: True.

    Returns:
        A tuple consisting of `values` and `indexes`.

        - values (Tensor): The `k` largest or smallest elements in each slice of the given dimension.
        - indices (Tensor): The indices of values within the last dimension of input.

    Raises:
        TypeError: If `sorted` is not a bool.
        TypeError: If `input_x` is not a Tensor.
        TypeError: If `k` is not an int.
        TypeError: If dtype of `input_x` is not one of the following: float16, float32 or int32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import ops
        >>> x = ms.Tensor([[0.5368, 0.2447, 0.4302, 0.9673],
        ...                [0.4388, 0.6525, 0.4685, 0.1868],
        ...                [0.3563, 0.5152, 0.9675, 0.8230]], dtype=ms.float32)
        >>> output = ops.topk(x, 2, dim=1)
        >>> print(output)
        (Tensor(shape=[3, 2], dtype=Float32, value=
        [[ 9.67299998e-01,  5.36800027e-01],
         [ 6.52499974e-01,  4.68499988e-01],
         [ 9.67499971e-01,  8.23000014e-01]]), Tensor(shape=[3, 2], dtype=Int32, value=
        [[3, 0],
         [1, 2],
         [2, 3]]))
        >>> output2 = ops.topk(x, 2, dim=1, largest=False)
        >>> print(output2)
        (Tensor(shape=[3, 2], dtype=Float32, value=
        [[ 2.44700000e-01,  4.30200011e-01],
         [ 1.86800003e-01,  4.38800007e-01],
         [ 3.56299996e-01,  5.15200019e-01]]), Tensor(shape=[3, 2], dtype=Int32, value=
        [[1, 2],
         [3, 0],
         [0, 1]]))
    """
    top_k_ = _get_cache_prim(P.TopK)(sorted)
    if not largest:
        input_x = -input_x
    if dim is None or dim == input_x.ndim - 1:
        if not largest:
            res = top_k_(input_x, k)
            values, indices = -res[0], res[1]
            return values, indices
        return top_k_(input_x, k)
    input_x = input_x.swapaxes(dim, input_x.ndim - 1)
    output = top_k_(input_x, k)
    values = output[0].swapaxes(dim, input_x.ndim - 1)
    indices = output[1].swapaxes(dim, input_x.ndim - 1)
    if not largest:
        res = (-values, indices)
    else:
        res = (values, indices)
    return res


def expand(input_x, size):
    r"""
    Returns a new view of the self tensor with singleton dimensions expanded to a larger size.

    Note:
        Passing -1 as the `size` for a dimension means not changing the size of that dimension.
        Tensor can be also expanded to a larger number of dimensions, and the new ones will be appended at the front.
        For the new dimensions, the `size` cannot be set to -1.

    Args:
        input_x (Tensor): The shape of tensor is (x_1, x_2, ..., x_R).
        size (Tensor): The expanded shape of `input_x`.

    Returns:
        y (Tensor) - Tensor after expansion whose shape is `size`.

    Raises:
        TypeError: If `input_x` or `size` is not Tensor.
        TypeError: If the type of `size` is not one of the following dtype: int16, int32, int64.
        ValueError: If the size of `size` is less than the size of `input_x.shape`.
        ValueError: If `size` is not a 1-D tensor.
        ValueError: If the expanded `size` is not equal to the existing shape of `input_x` at a dimension
            that is not 1.
        ValueError: If the expanded `size` < 0 and it is in a leading position, corresponding to
            a non-existing dimension in `input_x`.
        ValueError: If the number of elements of output is more than 1000000.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[1], [2], [3]]), mindspore.float32)
        >>> size = Tensor(np.array([3,4]), mindspore.int32)
        >>> y = ops.expand(input_x, size)
        >>> print(y)
        [[1. 1. 1. 1.]
         [2. 2. 2. 2.]
         [3. 3. 3. 3.]]
    """
    expand_op = _get_cache_prim(Expand)()
    return expand_op(input_x, size)


@constexpr
def _check_fold_param(param, param_name):
    """Check the parameters of fold op."""
    validator.check_value_type(param_name, param, [int, list, tuple], 'fold')
    param = (param, param) if isinstance(param, int) else param
    validator.check(param_name + " size", len(param), "", 2, Rel.EQ, 'fold')
    if param_name == "padding":
        validator.check_non_negative_int_sequence(param, param_name, 'fold')
    else:
        validator.check_positive_int_sequence(param, param_name, 'fold')
    return param


def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """
    Combines an array of sliding local blocks into a large containing tensor.

    .. warning::
        - Currently, only 4-D output tensors (batched image-like tensors) are supported.

    Args:
        input (Tensor): 4-D Tensor with data type float16 or float32.
        output_size (Tensor): 1D tensor with `2` elements of data type int.
        kernel_size (Union[int, tuple[int], list[int]]): The size of the kernel, should be two int
            for height and width. If type is int, it means that height equal with width. Must be specified.
        dilation (Union[int, tuple[int], list[int]], optional): The size of the dilation, should be two int
            for height and width. If type is int, it means that height equal with width. Default: 1.
        padding (Union[int, tuple[int], list[int]], optional): The size of the padding, should be two int
            for height and width. If type is int, it means that height equal with width. Default: 0.
        stride (Union[int, tuple[int], list[int]], optional): The size of the stride, should be two int
            for height and width. If type is int, it means that height equal with width. Default: 1.

    Returns:
        A Tensor, with same type as `input` , format of the Tensor is (N, C, H, W).

    Raises:
        TypeError: If `kernel_size`, `dilation`, `padding`, `stride` data type is not int, tuple or list.
        ValueError: If `kernel_size`, `dilation`, `stride` value is not
            greater than zero or elements number more than `2`.
        ValueError: If `padding` value is less than zero or elements number more than `2`.
        ValueError: If `input.shape[2] != kernel_size[0] * kernel_size[1]`.
        ValueError: If `input.shape[3]` does not match the calculated number of sliding blocks.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(input_data=np.random.rand(16, 16, 4, 25), dtype=mstype.float32)
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
    return fold_op(input, output_size)


@constexpr
def _check_unfold_params(param, param_name, param_size):
    """Check the parameters of unfold op."""
    validator.check_value_type(param_name, param, [int, tuple, list], 'unfold')
    param = (param, param) if isinstance(param, int) else param
    validator.check(param_name + " size", len(param), "", param_size, Rel.IN, 'unfold')
    if param_name == "padding":
        validator.check_non_negative_int_sequence(param, param_name, 'unfold')
    else:
        validator.check_positive_int_sequence(param, param_name, 'unfold')
    return param


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    """
    Reshapes a tensor of format (N, C, H, W) by extracting sliding local blocks from the input Tensor
    and concatenating them along a new dimension.

    .. warning::
        - Currently, only 4-D input tensors (batched image-like tensors) are supported.

    Args:
        input (Tensor): 4-D Tensor. Support all real number data type.
        kernel_size (Union[int, tuple[int], list[int]]): The size of the kernel, should be two int
            for height and width. If type is int, it means that height equal with width. Must be specified.
        dilation (Union[int, tuple[int], list[int]], optional): The dilation of the window, should be two int
            for height and width. If type is int, it means that height equal with width. Default: 1.
        padding (Union[int, tuple[int], list[int]], optional): The pad of the window, that must be
            a tuple/list of one or two or four `int` for height and width.
            If one int, pad_height = pad_width.
            If two int, pad_height = padding[0], pad_width = padding[1].
            If four int, padding = [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]
            Default: 0.
        stride (Union[int, tuple[int], list[int]], optional): The stride of the window, should be two int
            for height and width. If type is int, it means that height equal with width. Default: 1.

    Returns:
        A Tensor, with same type as `input`.

    Raises:
        TypeError: If any data type of `kernel_size`, `stride`, `dilation`, `kernel_size` is not int, tuple or list.
        ValueError: If `kernel_size`, `dilation`, `stride` value is not
            greater than zero or elements number more than `2`.
        ValueError: If `padding` value is less than zero.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x = Tensor(np.random.rand(4, 4, 32, 32), mindspore.float64)
        >>> output = ops.unfold(x, kernel_size=3, dilation=1, stride=1)
        >>> print(output.shape)
        (4, 36, 30, 30)
    """
    kernel_size = _check_unfold_params(kernel_size, "kernel_size", [1, 2])
    dilation = _check_unfold_params(dilation, "dilation", [1, 2])
    padding = _check_unfold_params(padding, "padding", [1, 2, 4])
    stride = _check_unfold_params(stride, "stride", [1, 2])
    unfold_op = _get_cache_prim(Im2Col)(ksizes=kernel_size,
                                        strides=stride,
                                        dilations=dilation,
                                        padding_mode="CALCULATED",
                                        pads=padding)
    return unfold_op(input)


@constexpr
def _check_diagonal_axes(dim1, dim2, x_ndim):
    """Check the parameters of unfold op."""
    axes = validator.check_axis_valid((dim1, dim2), x_ndim)
    return axes


def diagonal(input, offset=0, dim1=0, dim2=1):
    """
    Returns specified diagonals of `input`.

    If `input` is 2-D, returns the diagonal of `input` with the given offset.
    If `input` has more than two
    dimensions, then the axes specified by `dim1` and `dim2` are used to determine
    the 2-D sub-array whose diagonal is returned. The shape of the resulting
    array can be determined by removing `dim1` and `dim2` and appending an index
    to the right equal to the size of the resulting diagonals.

    Args:
        input (Tensor): Array from which the diagonals are taken.
        offset (int, optional): Offset of the diagonal from the main diagonal.
            Can be positive or negative. Defaults: 0.
        dim1 (int, optional): Axis to be used as the first axis of the 2-D
            sub-arrays from which the diagonals should be taken. Defaults to
            first axis (0).
        dim2 (int, optional): Axis to be used as the second axis of the 2-D
            sub-arrays from which the diagonals should be taken. Defaults to
            second axis (1).

    Returns:
        Tensor, if `input` is 2-D, then `input` 1-D array containing the diagonal. If
        ``input.ndim > 2``, then the dimensions specified by `dim1` and `dim2` are removed,
        and a new axis inserted at the end corresponding to the diagonal.

    Raises:
        ValueError: if the input tensor has less than two dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([[0, 1], [2, 3]], mstype.float32)
        >>> output = ops.diagonal(x)
        >>> print(output)
        [0 3]
    """
    x_ndim = input.ndim
    if x_ndim < 2:
        raise ValueError(f"ops.diagonal requires an array of at least two dimensions")
    dtype = input.dtype

    axes = _check_diagonal_axes(dim1, dim2, x_ndim)
    perm = ()
    for i in np.arange(x_ndim):
        if i not in axes:
            perm += (i,)
    perm += axes
    input = input.transpose(perm)

    x_shape = input.shape
    n, m = x_shape[-2:]

    fill_op = _get_cache_prim(P.Fill)()
    e = _get_cache_prim(P.Eye)()(n, m, dtype)
    if offset >= m or offset <= -n:
        e = fill_op(dtype, (n, m), 0)
    elif offset != 0:
        e = e.astype(mstype.float32)
        if offset > 0:
            e_left = fill_op(mstype.float32, (n, offset), 0)
            e_right = e[..., 0:m - offset:1]
            e = _get_cache_prim(P.Concat)(1)((e_left, e_right)).astype(dtype)
        elif offset < 0:
            e_upper = fill_op(mstype.float32, (-offset, m), 0)
            e_lower = e[0:n + offset:1, ...]
            e = _get_cache_prim(P.Concat)(0)((e_upper, e_lower)).astype(dtype)
    e = _get_cache_prim(P.BroadcastTo)(x_shape)(e)

    prod_val = _get_cache_prim(P.Mul)()(input, e)
    res = _get_cache_prim(P.ReduceSum)()(prod_val.astype(mstype.float32), -1)

    begin = ()
    for _ in np.arange((x_ndim - 2)):
        begin += (0,)
    last_dim_begin = np.max((0, -offset)).astype(np.int64)
    begin += (last_dim_begin,)
    res_size = res.shape[:-1]
    last_dim_end = np.min((x_shape[-2], np.max((0, (x_shape[-1] - offset))))) - last_dim_begin
    if last_dim_end <= 0:
        return Tensor([])
    res_size += (last_dim_end,)
    res = _get_cache_prim(P.Slice)()(res, begin, res_size)
    return res.astype(dtype)


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
        input (Tensor): The m by n matrix equivalent to `x` in above.
            The input tensor whose data type is float16, float32 or float64.
        A (Tensor): The m by k matrix equivalent to `a` in above.
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
        >>> x = Tensor(np.array([[2,1,5],[3,5,1],[1,1,1]]),mindspore.float32)
        >>> a = Tensor(np.array([[10,5],[15,8],[7,4]]),mindspore.float32)
        >>> output = ops.lstsq(x, a)
        >>> print(output)
        [[17.000002  11.000002 ]
         [-6.5000005 -4.500001 ]
         [-3.500002  -2.5000017]]
    """
    lstsq_op = _get_cache_prim(Lstsq)()
    return lstsq_op(input, A)


def mvlgamma(input, p):
    r"""
    Computes the multivariate log-gamma function with dimension `p` element-wise.

    The mathematical calculation process of Mvlgamma is shown as follows:

    .. math::

        \log (\Gamma_{p}(a))=C+\sum_{i=1}^{p} \log (\Gamma(a-\frac{i-1}{2}))

    where :math:`C = \log(\pi) \times \frac{p(p-1)}{4}` and :math:`\Gamma(\cdot)` is the Gamma function.

    Args:
        input (Tensor): The tensor to compute the multivariate log-gamma function,
          which must be one of the following types: float32, float64.
          The shape is :math:`(N,*)`, where :math:`*` means any number of additional dimensions.
          And the value of any element in `input` must be greater than :math:`(p - 1) / 2`.
        p (int): The number of dimensions. And the value of `p` must be greater than or equal to 1.

    Returns:
        Tensor, has the same shape and type as `input`.

    Raises:
        TypeError: If dtype of `input` is neither float32 nor float64.
        TypeError: If `p` is not an int.
        ValueError: If `p` is less than 1.
        ValueError: If not all elements of `input` are greater than :math:`(p - 1) / 2`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[3, 4, 5], [4, 2, 6]]), mindspore.float32)
        >>> y = ops.mvlgamma(x, p=3)
        >>> print(y)
        [[2.694925 5.402975 9.140645]
         [5.402975 1.596312 13.64045]]
    """
    mvlgamma_op = _get_cache_prim(Mvlgamma)(p)
    return mvlgamma_op(input)


def argwhere(x):
    """
    Return a Tensor of the positions of all non-zero values.

    Args:
        x (Tensor): The shape of Tensor is :math:`(x_1, x_2, ..., x_R)`. The data type is Number or Bool.

    Returns:
        Tensor, a 2-D Tensor whose data type is int64, containing the positions of all non-zero values of the input.

    Raises:
        TypeError: If `x` is not Tensor.
        ValueError: If dim of `x` equals to 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, ops
        >>> import numpy as np
        >>> x = Tensor(np.array([[[1,  0], [-5, 0]]]), mindspore.int32)
        >>> output = ops.argwhere(x)
        >>> print(output)
        [[0 0 0]
         [0 1 0]]
    """
    return nonzero_(x)


def column_stack(x):
    """
    Stacks 1-D tensors as columns into a 2-D tensor. 2-D tensors are stacked as-is,
    like ops.hstack.

    Args:
        x (Union[Tensor, tuple, list]): A sequence of 1-D or 2-D tensors. All
            of them must have the same shape except the axis to be concatenated.

    Returns:
        2-D Tensor, formed by stacking the given tensors.

    Raises:
        TypeError: If `x` is not Tensor, list or tuple.
        ValueError: If `x` is empty.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> x1 = Tensor([1, 1, 1])
        >>> x2 = Tensor([2, 2, 2])
        >>> output = ops.column_stack((x1, x2))
        >>> print(output)
        [[1 2]
         [1 2]
         [1 2]]
    """
    if not isinstance(x, (list, tuple)):
        raise TypeError(f"For column_stack, the input must be list or tuple or tensor, but got {type(x)}.")

    trans_x = ()
    _expand_dims = _get_cache_prim(P.ExpandDims)()
    for tensor in x:
        if tensor.ndim < 1:
            tensor = _expand_dims(tensor, 0)
        if tensor.ndim == 1:
            tensor = _expand_dims(tensor, 1)
        trans_x += (tensor,)
    if not trans_x:
        raise ValueError(f"For column_stack, the input must have at least 1 tensor, but got 0.")
    _concat = _get_cache_prim(P.Concat)(-1)
    return _concat(trans_x)


def hstack(x):
    """
    Stacks tensors in sequence horizontally.
    This is equivalent to concatenation along the second axis, except for 1-D tensors
    where it concatenates along the first axis.

    Args:
        x (Union[Tensor, tuple, list]): A sequence of 1-D or 2-D tensors. The
            tensors must have the same shape along all but the second axis, except
            1-D tensors which can be any length.

    Returns:
        Stacked Tensor, formed by stacking the given tensors.

    Raises:
        TypeError: If `x` is not Tensor, list or tuple.
        ValueError: If `x` is empty.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> x1 = Tensor([1, 1, 1])
        >>> x2 = Tensor([2, 2, 2])
        >>> output = ops.hstack((x1, x2))
        >>> print(output)
        [1. 1. 1. 2. 2. 2.]
    """
    if not isinstance(x, (list, tuple)):
        raise TypeError(f"For hstack, the input must be list or tuple, but got {type(x)}.")

    tuple_of_tensor = ()
    for tensor in x:
        if tensor.ndim < 1:
            tensor = expand_dims_(tensor, 0)
        tuple_of_tensor += (tensor,)
    if not tuple_of_tensor:
        raise ValueError("For hstack, the input must have at least 1 tensor, but got 0.")
    if tuple_of_tensor[0].ndim <= 1:
        _concat = _get_cache_prim(P.Concat)(0)
        return _concat(tuple_of_tensor)
    _concat = _get_cache_prim(P.Concat)(1)
    return _concat(tuple_of_tensor)


@constexpr
def _check_axis_valid(axis, ndim):
    """
    Checks axis are valid given ndim, and returns axis that can be passed
    to the built-in operator (non-negative, int or tuple).
    """
    if axis is None:
        axis = F.make_range(ndim)
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
    Moves axis of an array from source to destination.

    Other axis remain in their original order.

    Args:
        x (Tensor): The tensor array whose axis should be reordered.
        source (Union[int, sequence[int]]): Original positions of the
            axis to move. These must be unique.
        destination (Union[int, sequence[int]]): Destination positions
            for each of the original axis. These must also be unique.

    Returns:
        Tensor, array with moved axis.

    Raises:
        ValueError: If axis are out of the range of `[-a.ndim, a.ndim)`, or
            if the axis contain duplicates.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import ops, Tensor
        >>> import numpy as np
        >>> x = Tensor(np.zeros((3, 4, 5)))
        >>> output = ops.movedim(x, 0, -1)
        >>> print(output.shape)
        (4, 5, 3)
    """
    ndim = F.rank(x)
    source = _check_axis_valid(source, ndim)
    destination = _check_axis_valid(destination, ndim)
    if len(source) != len(destination):
        raise ValueError(
            f"For `source` and `destination` arguments, the number of elements must be the same, but got 'source':"
            f" {len(source)} and 'destination': {len(destination)}.")
    perm = _get_moved_perm(ndim, source, destination)
    return _get_cache_prim(P.Transpose)()(x, perm)


def moveaxis(x, source, destination):
    """
    Alias for `ops.movedim`. Moves axis of an array from source to destination.

    Refer to :func:`mindspore.ops.movedim` for more detail.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import ops, Tensor
        >>> import numpy as np
        >>> x = Tensor(np.zeros((3, 4, 5)))
        >>> output = ops.moveaxis(x, 0, -1)
        >>> print(output.shape)
        (4, 5, 3)
    """

    return movedim(x, source, destination)


def count_nonzero(x, dims=None):
    """
    Calculates the total number of non-zero entries in the input tensor along the
    specified dimensions. If no dimensions are given, then the function will
    count all non-zero entries in the tensor.

    Note:
        The value range of `dims` is [-x_dims, x_dims), `x_dims` is the dimension length of input "x".

    Args:
        x (Tensor): Input to be computed, Tensor of any dimension. Set the shape of input tensor as
          :math:`(x_1, x_2, ..., x_N)` .
        dims (Union[int, list(int), tuple(int)], optional): The dimension to count the number of non-zero values along.
            Default: None.

    Returns:
        A N-D Tensor, represents the number of the nonzero elements of the input tensor along the `dims`.
        Reduces x_shape along the dimensions given in `dims`. For example, if the size of `x` is :math:`(2, 3, 4)`,
        `dims` is :math:`[0, 1]`, y_shape will be :math:`(4,)`.

    Raises:
        TypeError: If the data type of `x` is not supported.
        TypeError: If the data type of `dims` is not int.
        ValueError: If any of the values of `dims` is not in range [-x_dims, x_dims).

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x = Tensor([[0, 0, 1], [1, 1, 2], [0, 0, 1]], mindspore.int64)
        >>> y = ops.count_nonzero(x, dims=[1])
        >>> print(y)
        [1 3 1]
    """
    dims = [] if dims is None else dims
    count_nonzero_ = CountNonZero(dims)
    return count_nonzero_(x)


@constexpr
def _check_swapaxes_axis(axes, ndim):
    return validator.check_swapaxes_axis(axes, ndim)


def swapaxes(x, axis0, axis1):
    '''
    Interchange two axes of a tensor.

    Args:
        x(Tensor): Input tensor.
        axis0 (int): First axis.
        axis1 (int): Second axis.

    Returns:
        Transposed tensor, has the same data type as `x`.

    Raises:
        TypeError: If argument `x` is not Tensor.
        TypeError: If `axis0` or `axis1` is not integer.
        ValueError: If `axis0` or `axis1` is not in the range of :math:`[-ndim, ndim-1]`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> x = Tensor(np.ones((2,3,4), dtype=np.float32))
        >>> output = ops.swapaxes(x, 0, 2)
        >>> print(output.shape)
        (4,3,2)
    '''
    if not isinstance(x, Tensor):
        raise TypeError(f'For ops.swapaxes, parameter `x` must be Tensor, but got {type(x)}')

    axis0, axis1 = _check_swapaxes_axis((axis0, axis1), x.ndim)
    if axis0 == axis1:
        return x
    if axis0 > axis1:
        axis0, axis1 = axis1, axis0

    perm = F.make_range(0, x.ndim)
    if axis1 + 1 < x.ndim:
        new_perm = perm[0:axis0] + perm[axis1:axis1 + 1] + \
                   perm[axis0 + 1:axis1] + perm[axis0:axis0 + 1] + perm[axis1 + 1:]
    else:
        new_perm = perm[0:axis0] + perm[axis1:axis1 + 1] + \
                   perm[axis0 + 1:axis1] + perm[axis0:axis0 + 1]

    return _get_cache_prim(P.Transpose)()(x, new_perm)


def swapdims(x, dim0, dim1):
    '''
    Interchange two dims of a tensor.
    This function is equivalent to :func:`mindspore.ops.swapaxes` function.

    Args:
        x(Tensor): Input tensor.
        dim0 (int): First dim.
        dim1 (int): Second dim.

    Returns:
        Transposed tensor, has the same data type as `x`.

    Raises:
        TypeError: If argument `x` is not Tensor.
        TypeError: If `dim0` or `dim1` is not integer.
        ValueError: If `dim0` or `dim1` is not in the range of :math:`[-ndim, ndim-1]`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> x = Tensor(np.ones((2,3,4), dtype=np.float32))
        >>> output = ops.swapdims(x, 0, 2)
        >>> print(output.shape)
        (4,3,2)
    '''
    return F.swapaxes(x, dim0, dim1)


@constexpr
def _check_is_int(arg_value, arg_name, op_name):
    arg_value = validator.check_is_int(arg_value, arg_name, op_name)
    return arg_value


@constexpr
def _check_positive_int(arg_value, arg_name, op_name):
    arg_value = validator.check_positive_int(arg_value, arg_name, op_name)
    return arg_value


@constexpr
def _check_axis_range(arg_value, limit, arg_name, op_name):
    arg_value = validator.check_int_range(arg_value, -limit, limit, Rel.INC_LEFT, arg_name, op_name)
    return arg_value


@_primexpr
def _cal_repeat_dims(x_rank, rep, expand_axis):
    rep_dims = [1] * (x_rank + 1)
    rep_dims[expand_axis] = rep
    return tuple(rep_dims)


@constexpr
def _cal_reshape(x_shape, rep, axis):
    x_reshape = list(x_shape)
    x_reshape[axis] *= rep
    return tuple(x_reshape)


def repeat_interleave(x, repeats, dim=None):
    """
    Repeat elements of a tensor along an axis, like `numpy.repeat`.

    Args:
        x (Tensor): The tensor to repeat values for. Must be of type: float16,
            float32, int8, uint8, int16, int32, or int64.
        repeats (int): The number of times to repeat, must be positive.
        dim (int, optional): The axis along which to repeat, default: None. if dims is None, the input Tensor will be
            flattened and the output will alse be flattened.

    Returns:
        One tensor with values repeated along the specified axis. If x has shape
        (s1, s2, ..., sn) and axis is i, the output will have shape (s1, s2, ...,
        si * repeats, ..., sn). The output type will be the same as the type of `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[0, 1, 2], [3, 4, 5]]), mindspore.int32)
        >>> output = ops.repeat_interleave(x, repeats=2, dim=0)
        >>> print(output)
        [[0 1 2]
         [0 1 2]
         [3 4 5]
         [3 4 5]]
    """
    if dim is None:
        x = x.reshape(-1)
        dim = 0
    return repeat_elements(x, repeats, dim)


def repeat_elements(x, rep, axis=0):
    """
    Repeat elements of a tensor along an axis, like `np.repeat` .

    Args:
        x (Tensor): The tensor to repeat values for. Must be of type: float16,
            float32, int8, uint8, int16, int32, or int64.
        rep (int): The number of times to repeat, must be positive.
        axis (int): The axis along which to repeat, default 0.

    Returns:
        One tensor with values repeated along the specified axis. If x has shape
        (s1, s2, ..., sn) and axis is i, the output will have shape (s1, s2, ...,
        si * rep, ..., sn). The output type will be the same as the type of `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case 1 : repeat on axis 0
        >>> x = Tensor(np.array([[0, 1, 2], [3, 4, 5]]), mindspore.int32)
        >>> output = ops.repeat_elements(x, rep = 2, axis = 0)
        >>> print(output)
        [[0 1 2]
         [0 1 2]
         [3 4 5]
         [3 4 5]]
        >>> # case 2 : repeat on axis 1
        >>> x = Tensor(np.array([[0, 1, 2], [3, 4, 5]]), mindspore.int32)
        >>> output = ops.repeat_elements(x, rep = 2, axis = 1)
        >>> print(output)
        [[0 0 1 1 2 2]
         [3 3 4 4 5 5]]
    """
    const_utils.check_type_valid(F.dtype(x), mstype.number_type, 'input x')
    rep = _check_positive_int(rep, "rep", "repeat_elements")
    axis = _check_is_int(axis, "axis", "repeat_elements")
    shape_op = P.Shape()
    rank_op = P.Rank()
    tile_op = P.Tile()
    expand_dims_op = P.ExpandDims()
    reshape_op = P.Reshape()
    x_rank = rank_op(x)
    axis = _check_axis_range(axis, x_rank, "axis", "repeat_elements")
    expand_axis = axis + 1
    x_expand = expand_dims_op(x, expand_axis)
    rep_dims = _cal_repeat_dims(x_rank, rep, expand_axis)
    x_expand = tile_op(x_expand, rep_dims)
    x_shape = shape_op(x)
    x_reshape = _cal_reshape(x_shape, rep, axis)
    x_rep = reshape_op(x_expand, x_reshape)
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
            type as elements in `lengths`. Default is None.

    Returns:
        One mask tensor of shape `lengths.shape + (maxlen,)` .

    Raises:
        TypeError: If `lengths` is not a Tensor.
        TypeError: If `maxlen` is not an int.
        TypeError: If dtype of `lengths` is neither int32 nor int64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> # case 1: When maxlen is assigned
        >>> x = Tensor(np.array([1, 2, 3, 4]))
        >>> output = ops.sequence_mask(x, 5)
        >>> print(output)
        [[ True False False False False]
         [ True  True False False False]
         [ True  True  True False False]
         [ True  True  True  True False]]
        >>> # case 2: When there is 0 in x
        >>> x = Tensor(np.array([[1, 3], [2, 0]]))
        >>> output = ops.sequence_mask(x, 5)
        >>> print(output)
        [[[ True False False False False]
          [ True  True  True False False]]
         [[ True  True False False False]
          [False False False False False]]]
        >>> # case 3: when the maxlen is not assigned
        >>> x = Tensor(np.array([[1, 3], [2, 4]]))
        >>> output = ops.sequence_mask(x)
        >>> print(output)
        [[[ True False False False ]
          [ True  True  True False ]]
         [[ True  True False False ]
          [ True  True  True  True ]]]
    """

    argmax_op = P.ArgMaxWithValue()
    reshape_op = P.Reshape()
    range_op = P.Range()
    expand_op = P.ExpandDims()
    cast_op = P.Cast()
    to_tensor_op = P.ScalarToTensor()

    const_utils.check_type_valid(F.dtype(lengths), [mstype.int64, mstype.int32], 'lengths')

    if maxlen is None:
        flatten_data = reshape_op(lengths, (-1,))
        flatten_data = cast_op(flatten_data, mstype.float32)
        _, value = argmax_op(flatten_data)
        maxlen = cast_op(value, mstype.int32)
    else:
        maxlen = _check_positive_int(maxlen, "maxlen", "sequence_mask")
        maxlen = to_tensor_op(maxlen, mstype.int32)

    range_vector = range_op(to_tensor_op(0, mstype.int32), maxlen
                            , to_tensor_op(1, mstype.int32))
    mask = expand_op(lengths, -1)
    result = range_vector < mask
    return result


__all__ = [
    'unique',
    'unique_with_pad',
    'unique_consecutive',
    'eye',
    'matrix_band_part',
    'padding',
    'fill',
    'fill_',
    'tile',
    'size',
    'ger',
    'ones',
    'ones_like',
    'zeros',
    'zeros_like',
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
    'range',
    'arange',
    'reshape',
    'reshape_',
    'flatten',
    'tensor_slice',
    'strided_slice',
    'slice',
    'cat',
    'concat',
    'stack',
    'unbind',
    'unstack',
    'is_tensor',
    'scalar_cast',
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
    'nonzero',
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
    'count_nonzero',
    'sort'
]
__all__.sort()
