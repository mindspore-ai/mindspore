# Copyright 2020-2023 Huawei Technologies Co., Ltd
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

"""Operators for array."""
import copy
import itertools
import numbers

import numpy as np

from mindspore import log as logger
from mindspore import context
from mindspore.common.initializer import Zero
from mindspore.ops import signature as sig
from mindspore.ops._utils import get_broadcast_shape
from mindspore.common._utils import is_shape_unknown, is_dim_unknown
from mindspore.ops.primitive import Primitive, PrimitiveWithInfer, PrimitiveWithCheck, prim_attr_register, _run_op
from mindspore import _checkparam as validator
from mindspore._checkparam import _check_3d_int_or_tuple
from mindspore.common import dtype as mstype
from mindspore.common._decorator import deprecated
from mindspore.common import Tensor, CSRTensor, COOTensor
from mindspore._c_expression import Tensor as Tensor_
from mindspore._c_expression import CSRTensor as CSRTensor_
from mindspore._c_expression import COOTensor as COOTensor_
from ..auto_generate import (ExpandDims, Reshape, TensorShape, Transpose, Gather,
                             OnesLike, ZerosLike, Argmax, ArgMaxExt,
                             ReverseV2, Diag, Eye, ScatterNd, ResizeNearestNeighborV2,
                             GatherNd, GatherD, Range, MaskedFill, RightShift, NonZero,
                             ResizeNearestNeighbor, Identity, Split, CumSum, CumProd,
                             Cummax, Cummin, Argmin, Concat, UnsortedSegmentSum, ScalarToTensor,
                             Triu, BroadcastTo, StridedSlice, Select, TopkExt, SearchSorted)
from .manually_defined import Rank, Shape, Tile, Cast, Ones, Zeros
from ..auto_generate import ArgMaxWithValue, ArgMinWithValue

class _ScatterOp(PrimitiveWithInfer):
    """
    Defines Scatter operators
    """
    __mindspore_signature__ = (
        sig.make_sig('x', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('updates', dtype=sig.sig_dtype.T)
    )

    def _check_scatter_shape(self, x_shape, indices_shape, updates_shape, prim_name):
        if indices_shape != [-1] and updates_shape and updates_shape != indices_shape + x_shape[1:]:
            raise ValueError(f"For '{prim_name}', "
                             f"updates_shape = indices_shape + input_x_shape[1:], but got input_x_shape: {x_shape}, "
                             f"indices_shape: {indices_shape}, updates_shape: {updates_shape}.")

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize _ScatterOp"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, x_shape, indices_shape, updates_shape):
        self._check_scatter_shape(x_shape, indices_shape, updates_shape, self.name)
        return x_shape

    def infer_dtype(self, x_dtype, indices_dtype, updates_dtype):
        validator.check_tensor_dtype_valid('indices', indices_dtype, [mstype.int32], self.name)
        args = {"x": x_dtype, "updates": updates_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        return x_dtype


class UnravelIndex(Primitive):
    """
    Transforms an array consisting of flattened indices into a tuple that contains coordinate arrays.

    Inputs:
        - **indices** (Tensor) - The input Tensor, containing indices that will be transformed
          into the flattened form of an array with dimensions specified by `dims`.
          The dimension of `indices` must be 0-D or 1-D.
          Must be one of the following types: int32, int64.
        - **dims** (Tensor) - The shape of the array to use for unraveling indices.
          The dimension of `dims` must be 1-D. Must have the same type as `indices`.

    Outputs:
        - **y** (Tensor) - Tensor, it should be 2-D or 1-D(if `indices` is 0D)
          and has the same type as `indices`.

    Raises:
        TypeError: If the data type of `indices` and `dims` are different.
        TypeError: If the data type of `indices` and `dims` is not int32 or int64.
        ValueError: If the dimension of `dims` is not 1 or dimension of `indices` is not 1 or 0.
        ValueError: If `indices` contains negative elements.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor(np.array([2, 5]), mindspore.int32)
        >>> dims = Tensor(np.array([3, 3]), mindspore.int32)
        >>> output = ops.UnravelIndex()(indices, dims)
        >>> print(output)
        [[0 2]
         [1 2]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Shape"""


class _ScatterOpDynamic(PrimitiveWithCheck):
    """
    Defines Scatter operators with dynamic shape
    """
    __mindspore_signature__ = (
        sig.make_sig('x', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('updates', dtype=sig.sig_dtype.T)
    )

    def _check_scatter_shape(self, x_shape, indices_shape, updates_shape, prim_name):
        # x_shape cannot be dynamic
        if np.any(np.array(x_shape) == -1):
            raise ValueError(f"For '{prim_name}', the 'input_x' does not support dynamic shape, "
                             f"but got the shape of 'input_x' is {x_shape}.")
        # support indices and updates dynamic
        if is_shape_unknown(indices_shape) or is_shape_unknown(updates_shape):
            pass
        elif indices_shape != [-1] and updates_shape and updates_shape != indices_shape + x_shape[1:]:
            raise ValueError(f"For '{prim_name}', "
                             f"updates_shape = indices_shape + input_x_shape[1:], but got input_x_shape: {x_shape}, "
                             f"indices_shape: {indices_shape}, updates_shape: {updates_shape}.")

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize _ScatterOpDynamic"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])
        self.add_prim_attr('side_effect_mem', True)

    def check_shape(self, x_shape, indices_shape, updates_shape):
        self._check_scatter_shape(x_shape, indices_shape, updates_shape, self.name)

    def check_dtype(self, x_dtype, indices_dtype, updates_dtype):
        validator.check_tensor_dtype_valid('indices', indices_dtype, [mstype.int32, mstype.int64], self.name)
        args = {"x": x_dtype, "updates": updates_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)


class _ScatterNdOp(_ScatterOp):
    """
    Defines _ScatterNd operators
    """

    def _check_scatter_shape(self, x_shape, indices_shape, updates_shape, prim_name):
        validator.check('the dimension of x', len(x_shape),
                        'the dimension of indices', indices_shape[-1], validator.GE)
        if indices_shape[:-1] + x_shape[indices_shape[-1]:] != updates_shape:
            raise ValueError(f"For '{prim_name}', updates_shape = "
                             f"indices_shape[:-1] + x_shape[indices_shape[-1]:], but got x_shape: {x_shape}, "
                             f"indices_shape: {indices_shape}, updates_shape: {updates_shape}.")


def _check_infer_attr_reduce(axis, keep_dims, prim_name):
    validator.check_value_type('keep_dims', keep_dims, [bool], prim_name)
    validator.check_value_type('axis', axis, [int, tuple], prim_name)
    if isinstance(axis, tuple):
        for index, value in enumerate(axis):
            validator.check_value_type('axis[%d]' % index, value, [int], prim_name)


class Expand(Primitive):
    """
    :class:`mindspore.ops.Expand` will be deprecated in the future.
    Please use :class:`mindspore.ops.BroadcastTo` instead.
    """

    @deprecated("2.1", "BroadcastTo", False)
    @prim_attr_register
    def __init__(self):
        """Initialize Expand."""
        self.add_prim_attr("max_length", 1000000)
        self.init_prim_io_names(inputs=['x', 'shape'], outputs=['y'])


class DType(Primitive):
    """
    Returns the data type of the input tensor as mindspore.dtype.

    Inputs:
        - **input_x** (Tensor) - Input Tensor.

    Outputs:
        mindspore.dtype, the data type of a tensor.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> output = ops.DType()(input_tensor)
        >>> print(output)
        Float32
    """

    @prim_attr_register
    def __init__(self):
        """Initialize DType"""

    def __call__(self, x):
        if not isinstance(x, (Tensor, CSRTensor, COOTensor, Tensor_, CSRTensor_, COOTensor_)):
            raise TypeError("For Primitive[Dtype], the input argument[input_x] "
                            "must be a Tensor, CSRTensor or COOTensor, but got " + str(type(x)) + ".")
        return x.dtype


class CheckNumerics(Primitive):
    """
    Checks a tensor for NaN and Inf values. A runtime error is raised if input has NaN or Inf values.

    Inputs:
        - **x** (Tensor) - Input Tensor of any dimension. The data type is float16, float32 or float64.

    Outputs:
        Tensor, has the same shape and data type as `x` if `x` has no NaN or Inf values.

    Raises:
        TypeError: If `x` data type is not float16, float32, float64.
        RuntimeError: If `x` has NaN or Inf values.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1, 3], [2, 4]], dtype=np.float32))
        >>> checknumerics = ops.CheckNumerics()
        >>> output = checknumerics(x)
        >>> print(output)
        [[1. 3.]
         [2. 4.]]
    """

    @prim_attr_register
    def __init__(self):
        """init CheckNumerics"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class Im2Col(Primitive):
    r"""
    Extracts sliding local blocks from a batched input tensor.

    Consider a batched input tensor of shape :math:`(N, C, *)`,
    where :math:`N` is the batch dimension, :math:`C` is the channel dimension,
    and :math:`*` represent arbitrary spatial dimensions. This operation flattens
    each sliding `ksizes`- sized block within the spatial dimensions
    of input `x` into a column (i.e., last dimension) of a 4-D output
    tensor of shape :math:`(N, C, \prod(\text{kernel_size}), L)`, where
    :math:`C \times \prod(\text{kernel_size})` is the total number of values
    within each block (a block has :math:`\prod(\text{kernel_size})` spatial
    locations each containing a `C`-channeled vector), and :math:`L` is
    the total number of such blocks:

    .. math::
        L = \prod_d \left\lfloor\frac{\text{spatial_size}[d] + 2 \times \text{pads}[d] %
            - \text{dilations}[d] \times (\text{kernel_size}[d] - 1) - 1}{\text{strides}[d]} + 1\right\rfloor,

    where :math:`\text{spatial_size}` is formed by the spatial dimensions
    of input `x` (:math:`*` above), and :math:`d` is over all spatial
    dimensions.

    Therefore, indexing `output` at the last dimension (column dimension)
    gives all values within a certain block.

    The `pads`, `strides` and `dilations` arguments specify
    how the sliding blocks are retrieved.

    Note:
        Currently, only 4-D input tensors (batched image-like tensors) are supported.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        ksizes (Union[int, tuple[int], list[int]]): The size of the kernel, should be two int
            for height and width. If type is int, it means that height equal with width. Must be specified.
        strides (Union[int, tuple[int], list[int]], optional): The stride of the window, should be two int
            for height and width. If type is int, it means that height equal with width. Default: ``1`` .
        dilations (Union[int, tuple[int], list[int]], optional): The dilation of the window, should be two int
            for height and width. If type is int, it means that height equal with width. Default: ``1`` .

        pads (Union[int, tuple[int], list[int]], optional): The pad of the window, that must be a tuple of
            one or two `int` for height and width. Default: ``0`` .

            - If one int, :math:`pad\_height = pad\_width`.
            - If two int, :math:`pad\_height = pads[0]`, :math:`pad\_width = pads[1]`.

    Inputs:
        - **x** (Tensor) - input tensor, only 4-D input tensors (batched image-like tensors) are supported.

    Outputs:
        Tensor, a 4-D Tensor with same type of input `x`.

    Raises:
        TypeError: If `ksizes` data type is not in Union[int, tuple[int], list[int]].
        TypeError: If `strides` data type is not in Union[int, tuple[int], list[int]].
        TypeError: If `dilations` data type is not in Union[int, tuple[int], list[int]].
        TypeError: If `pads` data type isnot in Union[int, tuple[int], list[int]].
        ValueError: If `ksizes` value is not greater than zero or elements number more than 2.
        ValueError: If `strides` value is not greater than zero or elements number more than 2.
        ValueError: If `dilations` value is not greater than zero or elements number more than 2.
        ValueError: If `pads` value is not greater than zero.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> from mindspore import dtype as mstype
        >>> x = Tensor(input_data=np.random.rand(4, 4, 32, 32), dtype=mstype.float64)
        >>> im2col = ops.Im2Col(ksizes=3, strides=1, dilations=1)
        >>> y = im2col(x)
        >>> print(y.shape)
        (4, 4, 9, 900)
    """

    @prim_attr_register
    def __init__(self, ksizes, strides=1, dilations=1, pads=0):
        """Initialize Im2Col."""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

        validator.check_value_type('ksizes', ksizes, [int, tuple, list], self.name)
        validator.check_value_type('strides', strides, [int, tuple, list], self.name)
        validator.check_value_type('dilations', dilations, [int, tuple, list], self.name)
        validator.check_value_type('pads', pads, [int, tuple, list], self.name)

        self.ksizes = (ksizes, ksizes) if isinstance(ksizes, int) else ksizes
        self.strides = (strides, strides) if isinstance(strides, int) else strides
        self.dilations = (dilations, dilations) if isinstance(dilations, int) else dilations
        self.pads = (pads, pads) if isinstance(pads, int) else pads

        validator.check("ksizes size", len(self.ksizes), "", [1, 2], validator.IN, self.name)
        validator.check_positive_int_sequence(self.ksizes, "ksizes", self.name)
        validator.check("strides size", len(self.strides), "", [1, 2], validator.IN, self.name)
        validator.check_positive_int_sequence(self.strides, "strides", self.name)
        validator.check("dilations size", len(self.dilations), "", [1, 2], validator.IN, self.name)
        validator.check_positive_int_sequence(self.dilations, "dilations", self.name)
        validator.check("pads size", len(self.pads), "", [1, 2], validator.IN, self.name)
        validator.check_non_negative_int_sequence(self.pads, "pads", self.name)

        self.add_prim_attr('ksizes', self.ksizes)
        self.add_prim_attr('strides', self.strides)
        self.add_prim_attr('dilations', self.dilations)
        self.add_prim_attr('pads', self.pads)
        self.add_prim_attr('padding_mode', "CALCULATED")


class Col2Im(Primitive):
    r"""
    Rearranges a row vector to an image. It is
    usually used to reconstruct an image from a set of image patches(or sliding local blocks).

    Consider an input Tensor of shape :math:`(N, C, \prod(\text{kernel_size}), L)`,
    where :math:`N` is batch dimension, :math:`C` is channel dimension,
    :math:`\prod(\text{kernel_size})` is the block size, and
    :math:`L` is the total number of blocks. This operation combines these
    local blocks into the large :attr:`output` tensor of
    shape :math:`(N, C, \text{output_size}[0], \text{output_size}[1], \dots)`
    by summing the overlapping values.

    .. math::
        L = \prod_d \left\lfloor\frac{\text{output_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilation}[d] \times (\text{kernel_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor

    where :math:`d` is over all spatial dimensions. The `padding`, `stride`
    and `dilation` arguments specify how the sliding blocks are retrieved.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        kernel_size (Union[int, tuple[int], list[int]]): The size of the kernel, should be two positive int
            for height and width. If type is int, it means that height equal with width. Must be specified.
        dilation (Union[int, tuple[int], list[int]], optional): The size of the dilation, should be two positive int
            for height and width. If type is int, it means that height equal with width. Default: ``1`` .
        padding (Union[int, tuple[int], list[int]], optional): The size of the padding, should be two int
            for height and width. If type is int, it means that height equal with width. Default: ``0`` .
        stride (Union[int, tuple[int], list[int]], optional): The size of the stride, should be two positive int
            for height and width. If type is int, it means that height equal with width. Default: ``1`` .

    Inputs:
        - **x** (Tensor) - 4D input Tensor.
        - **output_size** (Tensor) - 1D tensor with 2 elements of data type int32 or int64.

    Outputs:
        Tensor, a 4-D Tensor with same type of input `x`.

    Raises:
        TypeError: If dtype of `kernel_size` , `dilation` , `padding` or `stride` is not in
                   Union[int, tuple[int], list[int]].
        ValueError: If values in `kernel_size` , `dilation` , `padding` or `stride` are not greater than zero or any
                    one of them has more than 2 elements.
        ValueError: If x.shape[2] != kernel_size[0] * kernel_size[1].
        ValueError: If x.shape[3] does not match the calculated number of sliding blocks.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> from mindspore import dtype as mstype
        >>> x = Tensor(input_data=np.random.rand(16, 16, 4, 25), dtype=mstype.float32)
        >>> output_size = Tensor(input_data=[8, 8], dtype=mstype.int32)
        >>> col2im = ops.Col2Im(kernel_size=[2, 2], dilation=[2, 2], padding=[2, 2], stride=[2, 2])
        >>> y = col2im(x, output_size)
        >>> print(y.shape)
        (16, 16, 8, 8)
    """

    @prim_attr_register
    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        """Initialize Col2Im."""
        self.init_prim_io_names(inputs=['x', 'output_size'], outputs=['y'])
        validator.check_value_type('kernel_size', kernel_size, [int, list, tuple], self.name)
        validator.check_value_type('dilation', dilation, [int, list, tuple], self.name)
        validator.check_value_type('padding', padding, [int, list, tuple], self.name)
        validator.check_value_type('stride', stride, [int, list, tuple], self.name)

        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.stride = (stride, stride) if isinstance(stride, int) else stride

        validator.check("kernel_size size", len(self.kernel_size), "", 2, validator.EQ, self.name)
        validator.check_positive_int_sequence(self.kernel_size, "kernel_size", self.name)
        validator.check("dilation size", len(self.dilation), "", 2, validator.EQ, self.name)
        validator.check_positive_int_sequence(self.dilation, "dilation", self.name)
        validator.check("padding size", len(self.padding), "", 2, validator.EQ, self.name)
        validator.check_non_negative_int_sequence(self.padding, "padding", self.name)
        validator.check("stride size", len(self.stride), "", 2, validator.EQ, self.name)
        validator.check_positive_int_sequence(self.stride, "stride", self.name)

        self.add_prim_attr('kernel_size', self.kernel_size)
        self.add_prim_attr('dilation', self.dilation)
        self.add_prim_attr('padding', self.padding)
        self.add_prim_attr('stride', self.stride)


class Unsqueeze(PrimitiveWithCheck):
    """Unsqueeze"""

    @prim_attr_register
    def __init__(self, axis):
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        self.axis = axis


class Squeeze(Primitive):
    """
    Return the Tensor after deleting the dimension of size 1 in the specified `axis`.

    Refer to :func:`mindspore.ops.squeeze` for more details.

    Args:
        axis (Union[int, tuple(int)]): Specifies the dimension indexes of shape to be removed, which will remove
            all the dimensions of size 1 in the given axis parameter. If specified, it must be int32 or int64.
            Default: ``()`` .

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, the shape of tensor is :math:`(x_1, x_2, ..., x_S)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
        >>> squeeze = ops.Squeeze(2)
        >>> output = squeeze(input_x)
        >>> print(output)
        [[1. 1.]
         [1. 1.]
         [1. 1.]]
    """

    @prim_attr_register
    def __init__(self, axis=()):
        """Initialize Squeeze"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])
        validator.check_value_type('axis', axis, [int, tuple], self.name)
        if isinstance(axis, tuple):
            for idx, item in enumerate(axis):
                validator.check_value_type("axis[%d]" % idx, item, [int], self.name)
        else:
            self.axis = (axis,)
            self.add_prim_attr("axis", (axis,))


class ConjugateTranspose(Primitive):
    """
    Calculate the conjugate matrix of input x which has been transposed according to input perm.

    .. math::
        y[i,j,k,...,s,t,u] == conj(x[perm[i], perm[j], perm[k],...,perm[s], perm[t], perm[u]])

    Inputs:
        - **x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        - **perm** (tuple[int]) - The permutation to be converted. The elements in `perm` are composed of
          the indexes of each dimension of `x`. The length of `perm` and the shape of `x` must be
          the same. Only constant value is allowed. Must be in the range [0, rank(x)).

    Outputs:
        Tensor, the type of output tensor is the same as `x` and the shape of output tensor is decided by the
        shape of `x` and the value of `Conj(perm)`:

        .. math::
            y.shape[i] = x.shape[perm[i]]

        where i is in range [0, rank(x) - 1].

    Raises:
        TypeError: If `perm` is not a tuple.
        ValueError: If length of shape of `x` is not equal to length of shape of `perm`.
        ValueError: If the same element exists in `perm`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1 + 1j,2 + 2j], [3 + 3j, 4 + 4j]]), mindspore.complex64)
        >>> perm = (1, 0)
        >>> conjugate_transpose = ops.ConjugateTranspose()
        >>> output = conjugate_transpose(x, perm)
        >>> print(output)
            [[1.-1.j 3.-3.j]
            [2.-2.j 4.-4.j]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ConjugateTranspose"""
        self.init_prim_io_names(inputs=['x', 'perm'], outputs=['output'])


class Unique(Primitive):
    """
    Returns the unique elements of input tensor and also return a tensor containing the index of each value of input
    tensor corresponding to the output unique tensor.

    The output contains Tensor `y` and Tensor `idx`, the format is probably similar to (`y`, `idx`).
    The shape of Tensor `y` and Tensor `idx` is different in most cases, because Tensor `y` will be duplicated,
    and the shape of Tensor `idx` is consistent with the input.

    To get the same shape between `idx` and `y`, please refer to :class:`mindspore.ops.UniqueWithPad`.

    Inputs:
        - **input_x** (Tensor) - The input tensor.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tuple, containing Tensor objects (`y`, `idx`), `y` is a tensor with the
        same type as `input_x`, and contains the unique elements in `x`.
        `idx` is a tensor containing indices of elements in
        the input corresponding to the output tensor.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops, nn
        >>> input_x = Tensor(np.array([1, 2, 5, 2]), mindspore.int32)
        >>> output = ops.Unique()(input_x)
        >>> print(output)
        (Tensor(shape=[3], dtype=Int32, value= [1, 2, 5]), Tensor(shape=[4], dtype=Int32, value= [0, 1, 2, 1]))
        >>> y = output[0]
        >>> print(y)
        [1 2 5]
        >>> idx = output[1]
        >>> print(idx)
        [0 1 2 1]
        >>> # As can be seen from the above, y and idx shape
        >>> # note that for GPU, this operator must be wrapped inside a model, and executed in graph mode.
        >>> class UniqueNet(nn.Cell):
        ...     def __init__(self):
        ...         super(UniqueNet, self).__init__()
        ...         self.unique_op = ops.Unique()
        ...
        ...     def construct(self, x):
        ...         output, indices = self.unique_op(x)
        ...         return output, indices
        ...
        >>> input_x = Tensor(np.array([1, 2, 5, 2]), mindspore.int32)
        >>> net = UniqueNet()
        >>> output = net(input_x)
        >>> print(output)
        (Tensor(shape=[3], dtype=Int32, value= [1, 2, 5]), Tensor(shape=[4], dtype=Int32, value= [0, 1, 2, 1]))
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['output'])


class UniqueConsecutive(Primitive):
    """
    Returns the elements that are unique in each consecutive group of equivalent elements in the input tensor.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Refer to :func:`mindspore.ops.unique_consecutive` for more details.

    Args:
        return_idx (bool, optional): Whether to return the index of where the element in the original input
            maps to the position in the output. Default: ``False`` .
        return_counts (bool, optional): Whether to return the counts of each unique element. Default: ``False`` .
        axis (int, optional): The dimension to apply unique. If ``None`` , the unique of the flattened input is
            returned. If specified, it must be int32 or int64. Default: ``None`` .

    Inputs:
        - **x** (Tensor) - The input tensor.

    Outputs:
        A tensor or a tuple of tensors containing tensor objects (`output`, `idx`, `counts`).

        - `output` has the same type as `x` and is used to represent the output list of unique scalar elements.
        - If `return_idx` is True, there will be an additional returned tensor, `idx`,
          which has the same shape as `x` and represents
          the index of where the element in the original input maps to the position in the output.
        - If `return_counts` is True, there will be an additional returned tensor, `counts`,
          which represents the number of occurrences for each unique value or tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> from mindspore import dtype as mstype
        >>> x = Tensor(np.array([1, 1, 2, 2, 3, 1, 1, 2]), mstype.int32)
        >>> unique_consecutive = ops.UniqueConsecutive(True, True, None)
        >>> output, idx, counts = unique_consecutive(x)
        >>> print(output)
        [1 2 3 1 2]
        >>> print(idx)
        [0 0 1 1 2 3 3 4]
        >>> print(counts)
        [2 2 1 2 1]
    """

    @prim_attr_register
    def __init__(self, return_idx=False, return_counts=False, axis=None):
        """Initialize UniqueConsecutive"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])
        validator.check_value_type("return_idx", return_idx, [bool], self.name)
        validator.check_value_type("return_counts", return_counts, [bool], self.name)
        validator.check_value_type("axis", axis, [int, type(None)], self.name)
        self.add_prim_attr("return_idx", return_idx)
        self.add_prim_attr("return_counts", return_counts)
        self.add_prim_attr("axis", axis)


class SparseGatherV2(Primitive):
    """
    Returns a slice of input tensor based on the specified indices and axis.

    Inputs:
        - **input_params** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        - **input_indices** (Tensor) - The shape of tensor is :math:`(y_1, y_2, ..., y_S)`.
          Specifies the indices of elements of the original Tensor, must be in the range
          `[0, input_params.shape[axis])`.
        - **axis** (Union(int, Tensor[int])) - Specifies the dimension index to gather indices.
          When axis is Tensor, the size must be 1.

    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_params = Tensor(np.array([[1, 2, 7, 42], [3, 4, 54, 22], [2, 2, 55, 3]]), mindspore.float32)
        >>> input_indices = Tensor(np.array([1, 2]), mindspore.int32)
        >>> axis = 1
        >>> out = ops.SparseGatherV2()(input_params, input_indices, axis)
        >>> print(out)
        [[2. 7.]
         [4. 54.]
         [2. 55.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseGatherV2"""
        self.init_prim_io_names(inputs=['params', 'indices', 'axis'], outputs=['output'])
        self.add_prim_attr('bprop_return_sparse', True)


class Padding(Primitive):
    """
    Extends the last dimension of the input tensor from 1 to pad_dim_size, by filling with 0.

    Refer to :func:`mindspore.ops.padding` for more details.

    Args:
        pad_dim_size (int, optional): The value of the last dimension of `x` to be
            extended, which must be positive. Default: ``8`` .

    Inputs:
        - **x** (Tensor) - Input Tensor of 2D or higher-dimensional.
          The last dimension of `x` must be 1. The data type is Number.

    Outputs:
        Tensor, the padded Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([[8], [10]]), mindspore.float32)
        >>> pad_dim_size = 4
        >>> output = ops.Padding(pad_dim_size)(x)
        >>> print(output)
        [[ 8.  0.  0.  0.]
         [10.  0.  0.  0.]]
    """

    @prim_attr_register
    def __init__(self, pad_dim_size=8):
        """Initialize padding"""
        validator.check_value_type("pad_dim_size", pad_dim_size, [int], self.name)
        validator.check_positive_int(pad_dim_size, "pad_dim_size", self.name)
        self.pad_dim_size = pad_dim_size


class UniqueWithPad(Primitive):
    """
    Returns unique elements and relative indexes in 1-D tensor, filled with padding num.

    The basic function is the same as the Unique operator, but the UniqueWithPad operator adds a Pad function.
    The returned tuple(`y`, `idx`) after the input Tensor `x` is processed by the unique operator,
    in which the shapes of `y` and `idx` are mostly not equal. Therefore, in order to solve the above situation,
    the UniqueWithPad operator will fill the `y` Tensor with the `pad_num` specified by the user
    to make it have the same shape as the Tensor `idx`.

    Refer to :func:`mindspore.ops.unique_with_pad` for more details.

    Inputs:
        - **x** (Tensor) - The tensor need to be unique. Must be 1-D vector with types: int32, int64.
        - **pad_num** (int) - Pad num. The data type is an int.

    Outputs:
        tuple(Tensor), tuple of 2 tensors, `y` and `idx`.

        - y (Tensor) - The unique elements filled with pad_num, the shape and data type same as `x`.
        - idx (Tensor) - The index of each value of `x` in the unique output `y`, the shape and data type same as `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([1, 1, 2, 2, 3, 3, 4, 5]), mindspore.int32)
        >>> pad_num = 8
        >>> output = ops.UniqueWithPad()(x, pad_num)
        >>> print(output)
        (Tensor(shape=[8], dtype=Int32, value= [1, 2, 3, 4, 5, 8, 8, 8]),
         Tensor(shape=[8], dtype=Int32, value= [0, 0, 1, 1, 2, 2, 3, 4]))
    """

    @prim_attr_register
    def __init__(self):
        """init UniqueWithPad"""
        self.init_prim_io_names(inputs=['x', 'pad_num'], outputs=['y', 'idx'])


class Size(Primitive):
    r"""
    Returns a Scalar of type int that represents the size of the input Tensor and the total number of elements in the
    Tensor.

    Refer to :func:`mindspore.ops.size` for more details.

    Inputs:
        - **input_x** (Tensor) - Input parameters, the shape of tensor is :math:`(x_1, x_2, ..., x_R)`. The data type is
          `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.

    Outputs:
        int. A scalar representing the elements' size of `input_x`, tensor is the number of elements
        in a tensor, :math:`size=x_1*x_2*...x_R`. The data type is an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> size = ops.Size()
        >>> output = size(input_x)
        >>> print(output)
        4
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Size"""


class MatrixDiagV3(Primitive):
    r"""
    Constructs a diagonal matrix or a batch of diagonal matrices from a given input Tensor.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Refer to :func:`mindspore.ops.matrix_diag` for more details.

    Args:
        align (str, optional): specifies how superdiagonals and subdiagonals should be aligned.
            Supported values: ``"RIGHT_LEFT"`` , ``"LEFT_RIGHT"`` , ``"LEFT_LEFT"`` , ``"RIGHT_RIGHT"`` .
            Default: ``"RIGHT_LEFT"`` .

            - When set to ``"RIGHT_LEFT"`` , the alignment of superdiagonals will be towards the right side
              (padding the row on the left), while subdiagonals will be towards the left side
              (padding the row on the right)
            - When set to ``"LEFT_RIGHT"`` , the alignment of superdiagonals will be towards the left side
              (padding the row on the right), while subdiagonals will be towards the right side
              (padding the row on the left)
            - When set to ``"LEFT_LEFT"`` , the alignment of  both superdiagonals and subdiagonals will be towards
              the left side(padding the row on the right).
            - When set to ``"RIGHT_RIGHT"`` , the alignment of both superdiagonals and subdiagonals will be towards
              the right side(padding the row on the left).

    Inputs:
        - **x** (Tensor) - The diagonal Tensor.
        - **k** (Union[int, Tensor], optional) - Diagonal offsets.
          A Tensor of type int32. Positive value means superdiagonal,
          0 refers to the main diagonal, and negative value means subdiagonals. `k` can be a single integer
          (for a single diagonal) or a pair of integers specifying the low and high ends of a matrix band.
          k[0] must not be larger than k[1]. The value must be in the range of given or derivated `num_rows`
          and `num_cols`, meaning value of k must be in (-num_rows, num_cols). Default: ``0`` .
        - **num_rows** (Union[int, Tensor], optional) - The number of rows of the output Tensor.
          A Tensor of type int32 with only one value. If `num_rows` is -1, indicating that the innermost
          matrix of the output Tensor is a square
          matrix, and the real number of rows will be derivated by other inputs. That is
          :math:`num\_rows = x.shape[-1] - min(k[1], 0)`. Otherwise, the value must be equal or greater than
          :math:`x.shape[-1] - min(k[1], 0)`. Default: -1.
        - **num_cols** (Union[int, Tensor], optional) - The number of columns of
          the output Tensor. A Tensor of type int32 with only one value.
          If `num_cols` is -1, indicating that the innermost matrix of the output
          Tensor is a square matrix, and the real number of columns will be derivated by other inputs.
          That is :math:`num\_cols = x.shape[-1] + max(k[0], 0)`. Otherwise, the value must be equal or
          greater than :math:`x.shape[-1] - min(k[1], 0)`.  Default: -1.
        - **padding_value** (Union[int, float, Tensor], optional) - The number to fill the area outside the specified
          diagonal band. A Tensor with only one value. Have the same dtype as x. Default: ``0`` .

    Outputs:
        A Tensor. Has the same type as `x`.
        Suppose `x` has r dimensions with shape :math:`(I, J, ..., M, N)` . The output Tensor has rank r + 1 with shape
        :math:`(I, J, ..., M, num\_rows, num\_cols)` when only one diagonal is given (k is an integer or k[0] == k[1]).
        Otherwise, it has rank r with shape :math:`(I, J, ..., num\_rows, num\_cols)` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([[8, 9, 0],
        ...                      [1, 2, 3],
        ...                      [0, 4, 5]]), mindspore.float32)
        >>> k =Tensor(np.array([-1, 1]), mindspore.int32)
        >>> num_rows = Tensor(np.array(3), mindspore.int32)
        >>> num_cols = Tensor(np.array(3), mindspore.int32)
        >>> padding_value = Tensor(np.array(11), mindspore.float32)
        >>> matrix_diag_v3 = ops.MatrixDiagV3(align='LEFT_RIGHT')
        >>> output = matrix_diag_v3(x, k, num_rows, num_cols, padding_value)
        >>> print(output)
        [[ 1.  8. 11.]
         [ 4.  2.  9.]
         [11.  5.  3.]]
        >>> print(output.shape)
        (3, 3)
    """

    @prim_attr_register
    def __init__(self, align="RIGHT_LEFT"):
        """"Initialize MatrixDiagV3"""
        validator.check_value_type("align", align, [str], self.name)
        validator.check_string(align, ['LEFT_RIGHT', 'RIGHT_LEFT', 'LEFT_LEFT', 'RIGHT_RIGHT'], 'align', self.name)
        self.init_prim_io_names(inputs=['x', 'k', 'num_rows', 'num_cols', 'padding_value'], outputs=['y'])


class MatrixDiagPartV3(Primitive):
    r"""
    Returns the diagonal part of a tensor.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Refer to :func:`mindspore.ops.matrix_diag_part` for more details.

    Args:
        align (str, optional): specifies how superdiagonals and subdiagonals should be aligned.
            Supported values: ``"RIGHT_LEFT"`` , ``"LEFT_RIGHT"`` , ``"LEFT_LEFT"`` , ``"RIGHT_RIGHT"`` .
            Default: ``"RIGHT_LEFT"`` .

            - When set to ``"RIGHT_LEFT"`` , the alignment of superdiagonals will be towards the right side
              (padding the row on the left), while subdiagonals will be towards the left side
              (padding the row on the right)
            - When set to ``"LEFT_RIGHT"`` , the alignment of superdiagonals will be towards the left side
              (padding the row on the right), while subdiagonals will be towards the right side
              (padding the row on the left)
            - When set to ``"LEFT_LEFT"`` , the alignment of  both superdiagonals and subdiagonals will be towards
              the left side(padding the row on the right).
            - When set to ``"RIGHT_RIGHT"`` , the alignment of both superdiagonals and subdiagonals will be towards
              the right side(padding the row on the left).

    Inputs:
        - **x** (Tensor) - Rank r, where r >= 2.
        - **k** (Tensor) - A Tensor of type int32. Diagonal offset(s). Positive value means superdiagonal, 0 refers to
          the main diagonal, and negative value means subdiagonals. k can be a single integer (for a single diagonal) or
          a pair of integers specifying the low and high ends of a matrix band. k[0] must not be larger than k[1]. The
          value of k has restructions, meaning value of k must be in (-x.shape[-2], x.shape[-1]).
        - **padding_value** (Tensor) - A Tensor. Have the same dtype as x. The number to fill the area outside the
          specified diagonal band with. There must be only one value.

    Outputs:
        A Tensor. Has the same type as `x`.
        Assume `x` has r dimensions :math:`(I, J, ..., M, N)` . Let `max_diag_len` be the maximum length among all
        diagonals to be extracted, :math:`max\_diag\_len = min(M + min(k[1], 0), N + min(-k[0], 0))`
        Let `num_diags` be the number of diagonals to extract, :math:`num\_diags = k[1] - k[0] + 1`.
        If :math:`num\_diags == 1`, the output tensor is of rank r - 1 with shape :math:`(I, J, ..., L, max\_diag\_len)`
        Otherwise, the output tensor has rank r with dimensions :math:`(I, J, ..., L, num\_diags, max\_diag\_len)` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([[1, 2, 3, 4],
        ...                      [5, 6, 7, 8],
        ...                      [9, 8, 7, 6]]), mindspore.float32)
        >>> k =Tensor(np.array([1, 3]), mindspore.int32)
        >>> padding_value = Tensor(np.array(9), mindspore.float32)
        >>> matrix_diag_part_v3 = ops.MatrixDiagPartV3(align='RIGHT_LEFT')
        >>> output = matrix_diag_part_v3(x, k, padding_value)
        >>> print(output)
        [[9. 9. 4.]
         [9. 3. 8.]
         [2. 7. 6.]]
        >>> print(output.shape)
        (3, 3)
    """

    @prim_attr_register
    def __init__(self, align="RIGHT_LEFT"):
        """"Initialize MatrixDiagPartV3"""
        self.add_prim_attr("max_length", 200000000)
        validator.check_value_type("align", align, [str], self.name)
        validator.check_string(align, ['LEFT_RIGHT', 'RIGHT_LEFT', 'LEFT_LEFT', 'RIGHT_RIGHT'], 'align', self.name)
        self.init_prim_io_names(inputs=['x', 'k', 'padding_value'], outputs=['y'])


class MatrixSetDiagV3(Primitive):
    r"""
    Updates the diagonal part of a batched tensor.
    It takes a Tensor `x` and `diagonal` as input and returns a Tensor in which
    the specified diagonal values in the innermost matrices will be replaced
    by the values in the `diagonal`.

    Diagonals shorter than `max_diag_len` need to be padded, where `max_diag_len` is the
    longest diagonal value.
    The dimension of `diagonal` is :math:`shape[-2]` must be equal to num_diags calculated by
    :math:`num\_diags = k[1] - k[0] + 1`.
    The dimension of `diagonal` is :math:`shape[-1]` must be equal to the longest diagonal value `max_diag_len`
    calculated by :math:`max\_diag\_len = min(x.shape[-2] + min(k[1], 0), x.shape[-1] + min(-k[0], 0))`.

    Assume `x` is an n-D Tensor with shape :math:`(d_1, d_2, ..., d_{n-2}, d_{n-1}, d_n)`.
    If `k` is an integer or :math:`k[0] == k[1]`, `diagonal` is an (n-1)-D Tensor with
    shape :math:`(d_1, d_2, ..., d_{n-2}, max\_diag\_len)`
    Otherwise, it has the same rank as `x`
    with shape :math:`(d_1, d_2, ..., d_{n-2}, num\_diags, max\_diag\_len)`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        align (str, optional): specifies how superdiagonals and subdiagonals should be aligned.
            Supported values: ``"RIGHT_LEFT"`` , ``"LEFT_RIGHT"``, ``"LEFT_LEFT"`` , ``"RIGHT_RIGHT"`` .
            Default: ``"RIGHT_LEFT"`` .

            - When set to ``"RIGHT_LEFT"`` , the alignment of superdiagonals will be towards the right side
              (padding the row on the left), while subdiagonals will be towards the left side
              (padding the row on the right)
            - When set to ``"LEFT_RIGHT"`` , the alignment of superdiagonals will be towards the left side
              (padding the row on the right), while subdiagonals will be towards the right side
              (padding the row on the left)
            - When set to ``"LEFT_LEFT"`` , the alignment of  both superdiagonals and subdiagonals will be towards
              the left side(padding the row on the right).
            - When set to ``"RIGHT_RIGHT"`` , the alignment of both superdiagonals and subdiagonals will be towards
              the right side(padding the row on the left).

    Inputs:
        - **x** (Tensor) - A n-D Tensor, where :math:`n >= 2`.
        - **diagonal** (Tensor) - A Tensor with the same dtype as `x`. Its rank depends on `k`.
          If `k` is an integer or :math:`k[0] == k[1]`, its dimension is :math:`n-1`.
          Otherwise, it has dimension :math:`n`.
        - **k** (Tensor) - Diagonal offset(s), Tensor of type int32.
          `k` can either be a single integer, which represents a single diagonal,
          or a pair of integers that specify the low and high ends of a matrix band.
          In this case, `k[0]` should not be greater than `k[1]`.
          The value of `k` has restructions, which means that value of `k` must be in range
          :math:`(-x.shape[-2], x.shape[-1])`.
          Input `k` must be const Tensor when taking Graph mode.

          - `k > 0` refers to a superdiagonal.
          - `k = 0` refers to the main diagonal.
          - `k < 0` refers to subdiagonals.

    Outputs:
        Tensor. The same type and shape as `x`.

    Raises:
        TypeError: If any input is not Tensor.
        TypeError: If input `x` and `diagonal` are not the same dtype.
        TypeError: If `k` is not int32 dtype.
        ValueError: If `align` is not a string or not in the valid range.
        ValueError: If rank of `k` is not equal to 0 or 1.
        ValueError: If rank of `x` is not greater equal to 2.
        ValueError: If size of `k` is not equal to 1 or 2.
        ValueError: If `k[1]` is not greater equal to `k[0]` in case the size of `k` is 2.
        ValueError: If the `diagonal` rank size don't match with input `x` rank size.
        ValueError: If the `diagonal` shape value don't match with input `x` shape value.
        ValueError: If the diagonal :math:`shape[-2]` is not equal to num_diags calculated by
            :math:`num\_diags = k[1] - k[0] + 1` .
        ValueError: If the value of `k` is not in :math:`(-x.shape[-2], x.shape[-1])`.
        ValueError: If the diagonal :math:`shape[-1]` is not equal to the max_diag_len calculated by
            :math:`max\_diag\_len = min(x.shape[-2] + min(k[1], 0), x.shape[-1] + min(-k[0], 0))` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([[7, 7, 7, 7],
        ...                      [7, 7, 7, 7],
        ...                      [7, 7, 7, 7]]), mindspore.float32)
        >>> diagonal = Tensor(np.array([[0, 9, 1],
        ...                             [6, 5, 8],
        ...                             [1, 2, 3],
        ...                             [4, 5, 0]]), mindspore.float32)
        >>> k =Tensor(np.array([-1, 2]), mindspore.int32)
        >>> matrix_set_diag_v3 = ops.MatrixSetDiagV3(align='RIGHT_LEFT')
        >>> output = matrix_set_diag_v3(x, diagonal, k)
        >>> print(output)
        [[1. 6. 9. 7.]
         [4. 2. 5. 1.]
         [7. 5. 3. 8.]]
        >>> print(output.shape)
        (3, 4)
    """
    __mindspore_signature__ = (
        sig.make_sig('x', dtype=sig.sig_dtype.T1),
        sig.make_sig('diagonal', dtype=sig.sig_dtype.T1),
        sig.make_sig('k', dtype=sig.sig_dtype.T2)
    )

    @prim_attr_register
    def __init__(self, align="RIGHT_LEFT"):
        """"Initialize MatrixSetDiagV3"""
        self.add_prim_attr("max_length", 200000000)
        validator.check_value_type("align", align, [str], self.name)
        validator.check_string(align, ['LEFT_RIGHT', 'RIGHT_LEFT', 'LEFT_LEFT', 'RIGHT_RIGHT'], 'align', self.name)
        self.init_prim_io_names(inputs=['x', 'diagonal', 'k'], outputs=['y'])


class MatrixBandPart(Primitive):
    r"""
    Extracts the central diagonal band of each matrix in a tensor, with all values outside
    the central band set to zero.

    Refer to :func:`mindspore.ops.matrix_band_part` for more details.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **x** (Tensor) - Input tensor. :math:`(*, m, n)` where :math:`*` means, any number of additional dimensions.
        - **lower** (Union[int, Tensor]) - Number of subdiagonals to keep. The data type must be int32 or int64.
          If negative, keep entire lower triangle.
        - **upper** (Union[int, Tensor]) - Number of superdiagonals to keep. The data type must be int32 or int64.
          If negative, keep entire upper triangle.

    Outputs:
        Tensor, has the same type and shape as `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> matrix_band_part = ops.MatrixBandPart()
        >>> x = np.ones([2, 4, 4]).astype(np.float32)
        >>> output = matrix_band_part(Tensor(x), 2, 1)
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

    @prim_attr_register
    def __init__(self):
        super().__init__(name="MatrixBandPart")
        self.init_prim_io_names(inputs=['x', 'lower', 'upper'], outputs=['y'])


class Fill(PrimitiveWithCheck):
    """
    The Fill interface is deprecated, please use the :class:`mindspore.ops.FillV2` instead.

    Supported Platforms:
        Deprecated
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Fill"""
        self.init_prim_io_names(inputs=['type', 'shape', 'value'], outputs=['y'])

    def __call__(self, dtype, dims, x):
        if dtype not in mstype.all_types and dtype not in [mstype.uint16, mstype.uint32, mstype.uint64]:
            raise TypeError(
                f"For \'{self.name}\', the supported data type is ['bool', 'int8', 'int16', 'int32', 'int64', 'uint8', "
                "'uint16', 'uint32', 'uint64','float16', 'float32', 'float64'], but got an invalid dtype!.")
        x_nptype = mstype.dtype_to_nptype(dtype)
        if not isinstance(dims, Tensor) and not isinstance(dims, tuple):
            raise TypeError(f"For \'{self.name}\', input[1] must be tensor.")
        if not isinstance(x, Tensor) and not isinstance(x, float) and not isinstance(x, int):
            raise TypeError(f"For \'{self.name}\', the value input only takes scalar or scalar within a tensor!.")
        if isinstance(dims, Tensor):
            dims = dims.asnumpy()
        if isinstance(x, Tensor):
            x = x.asnumpy()
        ret = np.full(dims, x, x_nptype)
        return Tensor(ret, dtype=dtype)

    def infer_value(self, dtype, dims, x):
        x_nptype = mstype.dtype_to_nptype(dtype)
        if dims is not None and None not in dims and x is not None:
            if isinstance(dims, Tensor):
                dims = dims.asnumpy()
            if isinstance(x, Tensor):
                x = x.asnumpy()
            ret = np.full(dims, x, x_nptype)
            return Tensor(ret, dtype=dtype)
        return None


class Fills(Primitive):
    """
    The `Fills` primitive  is deprecated.
    Please use :func:`mindspore.ops.fill` instead.

    Supported Platforms:
        Deprecated

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> a = Tensor(np.arange(4).reshape((2,2)).astype('float32'))
        >>> fills = ops.Fills()
        >>> output = fills(a, float(1))
        >>> print(output)
        [[1. 1.]
         [1. 1.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Fills."""
        self.init_prim_io_names(inputs=['x', 'value'], outputs=['y'])


class FillV2(PrimitiveWithCheck):
    """
    Creates a tensor with shape described by `shape` and fills it with values in `value` .

    Inputs:
        - **shape** (Union[Tuple[int], Tensor[int]]) - 1-D Tensor or Tuple, specify the shape
          of output tensor. Its dtype must be int32 or int64.
        - **value** (Tensor) - A 0-D Tensor, the value to fill the output tensor `y` .

    Outputs:
        - **y** (Tensor) - A tensor, its shape and value are described above.

    Raises:
        TypeError: If `shape` is not a 1-D tensor or tuple.
        TypeError: If the data type of `shape` is not int32 or int64.
        ValueError: If `value` is not a 0-D Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, ops
        >>> fillV2 = ops.FillV2()
        >>> output = fillV2(Tensor([2, 3], mindspore.int32), Tensor(1, mindspore.float32))
        >>> print(output)
        [[1. 1. 1.]
         [1. 1. 1.]]
        >>> output = fillV2(Tensor([3, 3], mindspore.int64), Tensor(0, mindspore.int32))
        >>> print(output)
        [[0 0 0]
         [0 0 0]
         [0 0 0]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize FillV2"""
        self.init_prim_io_names(inputs=['shape', 'value'], outputs=['y'])

    def check_elim(self, dims, x):
        if x is None or (not isinstance(x, (Tensor, Tensor_))) or (x.shape != ()) or \
                dims is None or (isinstance(dims, (tuple, list)) and dims) or \
                isinstance(dims, (Tensor, Tensor_)):
            return (False, None)
        return (True, x)

    def infer_value(self, dims, x):
        if x is None or dims is None or isinstance(dims, (Tensor, Tensor_)):
            return None
        if isinstance(dims, (tuple, list)) and None in dims:
            return None
        if 0 in dims:
            init_func = Zero()
            init_func.__enable_zero_dim__ = True
            out = Tensor(shape=dims, dtype=x.dtype, init=init_func)
            return out
        return Tensor(np.full(dims, x.asnumpy()))


class TupleToArray(PrimitiveWithInfer):
    """
    Converts a tuple to a tensor.

    Refer to :func:`mindspore.ops.tuple_to_array` for more details.

    Inputs:
        - **input_x** (tuple) - A tuple of numbers. These numbers have the same type.
          The shape is :math:`(N,*)` where :math:`*` means any number of additional dimensions.

    Outputs:
        Tensor, if the input tuple contains `N` numbers, then the shape of the output tensor is :math:`(N,)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> input_x = (1,2,3)
        >>> print(type(input_x))
        <class 'tuple'>
        >>> output = ops.TupleToArray()(input_x)
        >>> print(type(output))
        <class 'mindspore.common.tensor.Tensor'>
        >>> print(output)
        [1 2 3]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize TupleToArray"""

    def infer_value(self, x):
        validator.check_value_type("x", x, [tuple], self.name)
        validator.check("size of x", len(x), '', 0, validator.GT, self.name)
        dtype = type(x[0])
        for i, item in enumerate(x):
            validator.check_value_type(f"x[{i}]", item, [numbers.Number], self.name)
        if not all(isinstance(item, dtype) for item in x):
            raise TypeError(f"For \'{self.name}\', all elements of 'input_x' must be have same type.")
        if isinstance(x[0], int):
            ret = np.array(x, np.int32)
        else:
            ret = np.array(x, np.float32)
        return Tensor(ret)

    def __call__(self, *args):
        x, = args
        args = list()
        if isinstance(x, range):
            args.append(tuple(x))
        else:
            args.append(x)
        return _run_op(self, self.name, args)




class InvertPermutation(PrimitiveWithInfer):
    r"""
    Computes the inverse of an index permutation.

    This operator is mainly used to calculate the inverse of index permutation.
    It requires a 1-dimensional integer tensor x, which represents the index of a zero-based array,
    and exchanges each value with its index position. In other words, For output tensor y and input tensor x,
    this operation calculates the following values:

    :math:`y[x[i]] = i, \quad i \in [0, 1, \ldots, \text{len}(x)-1]`.

    Note:
        These values must include 0. There must be no duplicate values and the
        values can not be negative.

    Inputs:
        - **input_x** (Union(tuple[int], list[int])) - The input is constructed by multiple
          integers, i.e., :math:`(y_1, y_2, ..., y_S)` representing the indices.
          The values must include 0. There can be no duplicate values or negative values.
          Only constant value is allowed. The maximum value must be equal to length of input_x.

    Outputs:
        tuple[int]. It has the same length as the input.

    Raises:
        TypeError: If `input_x` is neither tuple nor list.
        TypeError: If element of `input_x` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> invert = ops.InvertPermutation()
        >>> input_data = (3, 4, 0, 2, 1)
        >>> output = invert(input_data)
        >>> print(output)
        (2, 4, 3, 0, 1)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize InvertPermutation"""

    def __infer__(self, x):
        x_shp = x['shape']
        x_value = x['value']
        if mstype._issubclass_(x['dtype'], mstype.tensor_type):  # pylint: disable=W0212
            raise ValueError(f"For \'{self.name}\', the value of 'input_x' must be non-Tensor, but got {x['dtype']}")
        if x_value is None:
            raise ValueError(f"For '{self.name}', the value of 'input_x' can not be None, but got {x_value}.")
        validator.check_value_type("shape", x_shp, [tuple, list], self.name)
        for shp in x_shp:
            if shp:
                x_rank = len(np.array(x_value, np.int64).shape)
                raise ValueError(f"For \'{self.name}\', the dimension of 'input_x' must be 1, but got {x_rank}.")
        for i, value in enumerate(x_value):
            validator.check_value_type("input[%d]" % i, value, [int], self.name)
        z = [x_value[i] for i in range(len(x_value))]
        z.sort()

        for i in range(1, len(z)):
            if z[i - 1] == z[i]:
                raise ValueError(f"For '{self.name}', the 'input_x' can not contain duplicate values, "
                                 f"but got duplicated {z[i]} in the 'input_x'.")
        validator.check(f'value min', min(x_value), '', 0, validator.EQ, self.name)
        validator.check(f'value max', max(x_value), '', len(x_value) - 1, validator.EQ, self.name)

        y = [None] * len(x_value)
        for i, value in enumerate(x_value):
            validator.check_value_type("input[%d]" % i, value, [int], self.name)
            validator.check(f'value', z[i], f'index', i, validator.EQ, self.name)
            y[value] = i
            z.append(value)
        return {'shape': x_shp,
                'dtype': x['dtype'],
                'value': tuple(y)}


class ArgminV2(Primitive):
    """
    Returns the indices of the minimum value of a tensor across the axis.

    If the shape of input tensor is :math:`(x_1, ..., x_N)`, the shape of the output tensor is
    :math:`(x_1, ..., x_{axis-1}, x_{axis+1}, ..., x_N)`.

    Note:
        This operator only supports dynamic shape. As for static shape, please use operator `Argmin` instead.

    Inputs:
        - **x** (Tensor) - Input tensor.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
        - **axis** (int) - Axis where the Argmin operator applies to. Default: ``-1`` .

    Outputs:
        Tensor, indices of the min value of input tensor across the axis.

    Raises:
        TypeError: If `axis` is not an int.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> class ArgMinV2DynatimicShape(nn.Cell):
        ...     def __init__(self, gather_axis=1, argmin_axis=1):
        ...         super(ArgMinV2DynatimicShape, self).__init__()
        ...         self.unique = P.Unique()
        ...         self.gather = P.Gather()
        ...         self.argmin = ArgminV2()
        ...         self.gather_axis = gather_axis
        ...         self.argmin_axis = argmin_axis
        ...     def construct(self, x, indices):
        ...         unique_index, _ = self.unique(indices)
        ...         y = self.gather(x, unique_index, self.gather_axis)
        ...         z = self.argmin(y, self.argmin_axis)
        ...         return z
        >>>
        >>> x = Tensor(np.array([[4, 8, 1, 6], [4, 3, 6, 2], [4, 4, 1, 1]]).astype(np.float32))
        >>> index = Tensor([1, 2], dtype=mindspore.int32)
        >>> net = ArgMinV2DynatimicShape()
        >>> res = net(x, index)
        >>> print(res)
        [1 0 1]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ArgminV2"""
        self.init_prim_io_names(inputs=['x', 'axis'], outputs=['y'])

    def __call__(self, x, axis=-1):
        args = [x, axis]
        output = _run_op(self, self.name, args)
        return output


class UnsortedSegmentMin(PrimitiveWithCheck):
    r"""
    Computes the minimum of a tensor along segments.

    Refer to :func:`mindspore.ops.unsorted_segment_min` for more details.

    Inputs:
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_R)`.
          The data type must be float16, float32 or int32.
        - **segment_ids** (Tensor) - The label indicates the segment to which each element belongs.
          Set the shape as :math:`(x_1, x_2, ..., x_N)`, where 0 < N <= R.
        - **num_segments** (Union[int, Tensor]) - Set :math:`z` as num_segments, it can be an int or 0-D Tensor.

    Outputs:
        Tensor, the shape is :math:`(z, x_{N+1}, ..., x_R)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> import numpy as np
        >>> input_x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
        >>> segment_ids = Tensor(np.array([0, 1, 1]).astype(np.int32))
        >>> num_segments = 2
        >>> unsorted_segment_min = ops.UnsortedSegmentMin()
        >>> output = unsorted_segment_min(input_x, segment_ids, num_segments)
        >>> print(output)
        [[1. 2. 3.]
         [4. 2. 1.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize UnsortedSegmentMin"""
        self.init_prim_io_names(inputs=['x', 'segment_ids', 'num_segments'], outputs=['y'])

    def __check__(self, x, segment_ids, num_segments):
        x_shape = x['shape']
        segment_ids_shape = segment_ids['shape']
        valid_type = [mstype.float16, mstype.float32, mstype.int32, mstype.int8, mstype.uint8,
                      mstype.int16, mstype.uint16, mstype.uint32, mstype.int64, mstype.uint64, mstype.float64]
        validator.check_tensor_dtype_valid("x", x['dtype'], valid_type, self.name)
        validator.check_tensor_dtype_valid("segment_ids", segment_ids['dtype'], [mstype.int32, mstype.int64], self.name)

        # support vmap : segment_ids_shape support batch rank
        if not hasattr(self, 'batch_rank'):
            if not is_dim_unknown(x_shape) and not is_dim_unknown(segment_ids_shape):
                validator.check_int(len(segment_ids_shape), 1, validator.GE, "rank of segment_ids_shape", self.name)

        num_segments_type = num_segments['dtype']
        validator.check_subclass("num_segments", num_segments_type, [mstype.number], self.name)
        if not is_shape_unknown(x_shape) and not is_shape_unknown(segment_ids_shape):
            # only validate when both shapes fully known
            validator.check(f'first shape of input_x', x_shape[0],
                            'length of segments_id', segment_ids_shape[0], validator.EQ, self.name)
        num_segments_v = num_segments['value']
        validator.check_value_type('num_segments', num_segments_v, [int], self.name)
        validator.check_positive_int(num_segments_v, "num_segments", self.name)


class UnsortedSegmentMax(PrimitiveWithCheck):
    r"""
    Computes the maximum along segments of a tensor.

    Refer to :func:`mindspore.ops.unsorted_segment_max` for more details.

    Inputs:
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_R)`.
          The data type must be float16, float32 or int32.
        - **segment_ids** (Tensor) - The label indicates the segment to which each element belongs.
          Set the shape as :math:`(x_1, x_2, ..., x_N)`, where 0 < N <= R.
        - **num_segments** (Union[int, Tensor]) - Set :math:`z` as num_segments, it can be an int or 0-D Tensor.

    Outputs:
        Tensor, the shape is :math:`(z, x_{N+1}, ..., x_R)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case 1: Only have two num_segments, where is 0 and 1, and segment_ids=[0, 1, 1]
        >>> # num_segments = 2 indicates that there are two types of segment_id,
        >>> # the first number '0' in [0, 1, 1] indicates input_x[0],
        >>> # the second number '1' in [0, 1, 1] indicates input_x[1],
        >>> # the third number '1' in [0, 1, 1] indicates input_x[2],
        >>> # input_x[0], which is [1, 2, 3] will not be compared to other segment_id.
        >>> # Only the same segment_id will be compared.
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> import numpy as np
        >>> input_x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
        >>> segment_ids = Tensor(np.array([0, 1, 1]).astype(np.int32))
        >>> num_segments = 2
        >>> unsorted_segment_max = ops.UnsortedSegmentMax()
        >>> output = unsorted_segment_max(input_x, segment_ids, num_segments)
        >>> print(output)
        [[1. 2. 3.]
         [4. 5. 6.]]
        >>>
        >>> # case 2: The segment_ids=[0, 0, 1, 1].
        >>> # [1, 2, 3] will compare with [4, 2, 0],
        >>> # and [4, 5, 6] will compare with [4, 2, 1].
        >>> input_x = Tensor(np.array([[1, 2, 3], [4, 2, 0], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
        >>> segment_ids = Tensor(np.array([0, 0, 1, 1]).astype(np.int32))
        >>> num_segments = 2
        >>> unsorted_segment_max = ops.UnsortedSegmentMax()
        >>> output = unsorted_segment_max(input_x, segment_ids, num_segments)
        >>> print(input_x.shape)
            (4, 3)
        >>> print(output)
            [[4. 2. 3.]
             [4. 5. 6.]]
        >>> # case 3: If the input_x have three dimensions even more, what will happen?
        >>> # The shape of input_x is (2, 4, 3),
        >>> # and the length of segment_ids should be the same as the first dimension of input_x.
        >>> # Because the segment_ids are different, input_x[0] will not be compared to input_x[1].
        >>> input_x = Tensor(np.array([[[1, 2, 3], [4, 2, 0], [4, 5, 6], [4, 2, 1]],
        ...                            [[1, 2, 3], [4, 2, 0], [4, 5, 6], [4, 2, 1]]]).astype(np.float32))
        >>> segment_ids = Tensor(np.array([0, 1]).astype(np.int32))
        >>> num_segments = 2
        >>> unsorted_segment_max = ops.UnsortedSegmentMax()
        >>> output = unsorted_segment_max(input_x, segment_ids, num_segments)
        >>> print(input_x.shape)
            (2, 4, 3)
        >>> print(output)
            [[[1. 2. 3.]
              [4. 2. 0.]
              [4. 5. 6.]
              [4. 2. 1.]]
             [[1. 2. 3.]
              [4. 2. 0.]
              [4. 5. 6.]
              [4. 2. 1.]]]
        >>> # case 4: It has the same input with the 3rd case.
        >>> # Because num_segments is equal to 2, there are two segment_ids, but currently only one 0 is used.
        >>> # the segment_id i is absent in the segment_ids, then output[i] will be filled with
        >>> # the smallest possible value of the input_x's type.
        >>> segment_ids = Tensor(np.array([0, 0]).astype(np.int32))
        >>> output = unsorted_segment_max(input_x, segment_ids, num_segments)
        >>> print(output)
            [[[ 1.0000000e+00  2.0000000e+00  3.0000000e+00]
              [ 4.0000000e+00  2.0000000e+00  0.0000000e+00]
              [ 4.0000000e+00  5.0000000e+00  6.0000000e+00]
              [ 4.0000000e+00  2.0000000e+00  1.0000000e+00]]
             [[-3.4028235e+38 -3.4028235e+38 -3.4028235e+38]
              [-3.4028235e+38 -3.4028235e+38 -3.4028235e+38]
              [-3.4028235e+38 -3.4028235e+38 -3.4028235e+38]
              [-3.4028235e+38 -3.4028235e+38 -3.4028235e+38]]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize UnsortedSegmentMax"""
        self.init_prim_io_names(inputs=['x', 'segment_ids', 'num_segments'], outputs=['y'])

    def __check__(self, x, segment_ids, num_segments):
        x_shape = x['shape']
        segment_ids_shape = segment_ids['shape']
        valid_type = [mstype.float16, mstype.float32, mstype.int32, mstype.int8, mstype.uint8,
                      mstype.int16, mstype.uint16, mstype.uint32, mstype.int64, mstype.uint64, mstype.float64]
        validator.check_tensor_dtype_valid("x", x['dtype'], valid_type, self.name)
        validator.check_tensors_dtypes_same_and_valid({"segment_ids": segment_ids['dtype']},
                                                      [mstype.int32, mstype.int64], self.name)

        # support vmap : segment_ids_shape support batch rank
        if not hasattr(self, 'batch_rank'):
            if not is_dim_unknown(x_shape) and not is_dim_unknown(segment_ids_shape):
                validator.check_int(len(segment_ids_shape), 1, validator.GE, "rank of segment_ids_shape", self.name)

        num_segments_type = num_segments['dtype']
        validator.check_subclass("num_segments", num_segments_type, [mstype.number], self.name)
        if not is_shape_unknown(x_shape) and not is_shape_unknown(segment_ids_shape):
            # only validate when both shapes fully known
            validator.check(f'first shape of input_x', x_shape[0],
                            'length of segments_id', segment_ids_shape[0], validator.EQ, self.name)
        num_segments_v = num_segments['value']
        if num_segments_v is not None:
            validator.check_value_type('num_segments', num_segments_v, [int], self.name)
            validator.check_positive_int(num_segments_v, "num_segments", self.name)


class UnsortedSegmentProd(Primitive):
    """
    Computes the product of a tensor along segments.

    Refer to :func:`mindspore.ops.unsorted_segment_prod` for more details.

    Inputs:
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_R)`.
          With float16, float32 or int32 data type.
        - **segment_ids** (Tensor) - The label indicates the segment to which each element belongs.
          Set the shape as :math:`(x_1, x_2, ..., x_N)`, where 0 < N <= R. Data type must be int32.
        - **num_segments** (Union[int, Tensor]) - Set :math:`z` as num_segments, it can be an int or 0-D Tensor.

    Outputs:
        Tensor, the shape is :math:`(z, x_{N+1}, ..., x_R)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> import numpy as np
        >>> input_x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
        >>> segment_ids = Tensor(np.array([0, 1, 0]).astype(np.int32))
        >>> num_segments = 2
        >>> unsorted_segment_prod = ops.UnsortedSegmentProd()
        >>> output = unsorted_segment_prod(input_x, segment_ids, num_segments)
        >>> print(output)
        [[4. 4. 3.]
         [4. 5. 6.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize UnsortedSegmentProd"""
        self.init_prim_io_names(inputs=['x', 'segment_ids', 'num_segments'], outputs=['y'])


class ConcatOffsetV1(Primitive):
    r"""
    primitive for computing Concat’s gradient.

    Computes offsets of concat inputs within its output. Accumulate offsets from zero along `axis`.
    If tensor element in `x` isn't along `axis`, they should be the same along their axis.

    Inputs:
        - **axis** (Tensor): The specified axis, required to be 0-D Tensor object with dtype int32.
          Input `axis` should fall in :math:`[-numelement, numelement - 1]`,
          say numelement is the element number of first tensor in `x`.
        - **x** (tuple[Tensor], list[Tensor]) - A tuple or a list of input tensors.
          The tensors in `x` are all required to be a vector, in other word, 1-D Tensor object with dtype int32.
          Suppose there are two tensors in this tuple or list, namely x1 and x2.
          To perform `ConcatOffsetV1` in the axis 0 direction,
          except for the 0th axis, all elements in other axes should be equal,
          that is, :math:`x1[1] == x2[1], x1[2] == x2[2], ..., x1[R] == x2[R]`,
          where the :math:`R` indicates the last axis.

    Outputs:
        Tensors. A tuple of N 1-D Tensor objects.
        The data type is the same with the Inputs `x`, dtype int32.
        The shape is the same with the Inputs `x`.

    Raises:
        TypeError: If `axis` is not a tensor.
        TypeError: If dtype of tensor in `axis` is not int32.
        TypeError: If `x` have different type of tensor.
        TypeError: If dtype of tensor in `x` is not int32.
        ValueError: If the shape rank of `axis` does not equal to 0.
        ValueError: If the number of tensors in `x` is less than 2.
        ValueError: If the shape rank of tensor in `x` does not equal to 1.
        ValueError: If the element number of tensor in `x` is less than 1.
        ValueError: If `x` have different shape of tensors.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> axis = Tensor(1, dtype=mstype.int32)
        >>> x1 = Tensor(np.array([1, 2, 3]).astype(np.int32))
        >>> x2 = Tensor(np.array([1, 5, 3]).astype(np.int32))
        >>> x3 = Tensor(np.array([1, 4, 3]).astype(np.int32))
        >>> op = ops.ConcatOffsetV1()
        >>> output = op(axis, (x1, x2, x3))
        >>> print(output)
        (Tensor(shape=[3,], dtype=Int32, value=[0, 0, 0]),
         Tensor(shape=[3,], dtype=Int32, value=[0, 2, 0]),
         Tensor(shape=[3,], dtype=Int32, value=[0, 7, 0]))

    """

    @prim_attr_register
    def __init__(self):
        """Initialize ConcatOffsetV1"""


class ParallelConcat(Primitive):
    r"""
    Concats input tensors along the first dimension.

    The difference between Concat and ParallelConcat is that Concat requires all of the inputs be computed
    before the operation will begin but doesn't require that the input shapes be known during graph construction.
    Parallel concat will copy pieces of the input into the output as they become available, in some situations
    this can provide a performance benefit.

    Note:
        The input tensors are all required to have size 1 in the first dimension.

    Inputs:
        - **values** (tuple, list) - A tuple or a list of input tensors. The data type and shape of these
          tensors must be the same and their rank should not be less than 1.
          The supported date type is Number on CPU, the same for Ascend except
          [float64, complex64, complex128].

    Outputs:
        Tensor, data type is the same as `values`.

    Raises:
        TypeError: If any type of the inputs is not a Tensor.
        TypeError: If the data type of these tensors are not the same.
        ValueError: If any tensor.shape[0] is not 1.
        ValueError: If rank of any Tensor in `values` is less than 1.
        ValueError: If the shape of these tensors are not the same.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> data1 = Tensor(np.array([[0, 1]]).astype(np.int32))
        >>> data2 = Tensor(np.array([[2, 1]]).astype(np.int32))
        >>> op = ops.ParallelConcat()
        >>> output = op((data1, data2))
        >>> print(output)
        [[0 1]
         [2 1]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ParallelConcat"""


def _get_stack_shape(value, x_shape, x_type, axis, prim_name):
    """for stack output shape"""
    validator.check_value_type("shape", x_shape, [tuple, list], prim_name)
    validator.check_int(len(x_shape), 1, validator.GE, "len of input_x", prim_name)
    validator.check_subclass("input_x[0]", x_type[0], mstype.tensor_type, prim_name)

    out_n = len(x_shape)
    for i in range(1, out_n):
        if x_type[i] != x_type[i - 1]:
            raise TypeError(f"For {prim_name}, all types should be same, but got {x_type}")

    new_x_shape = []
    for i, shp in enumerate(x_shape):
        if is_dim_unknown(shp):
            continue
        new_x_shape.append({"shape": shp, "id": i})

    if not new_x_shape:
        out = {"shape": x_shape[0]}
        return out

    out_shape = new_x_shape[0]["shape"]
    n = len(new_x_shape)

    rank_base = len(new_x_shape[0]["shape"])
    for i in range(1, n):
        validator.check('len of x_shape[%d]' % new_x_shape[i]["id"], len(new_x_shape[i]["shape"]),
                        'len of x_shape[0]', rank_base, validator.EQ, prim_name, ValueError)
        for j in range(0, rank_base):
            if new_x_shape[i]["shape"][j] != new_x_shape[0]["shape"][j] and \
                    new_x_shape[i]["shape"][j] != -1 and new_x_shape[0]["shape"][j] != -1:
                raise ValueError(f"For {prim_name} element {new_x_shape[i]['id']} shape"
                                 f"in input can not pack with first element")

    validator.check_int_range(axis, -rank_base - 1, rank_base, validator.INC_BOTH, 'axis', prim_name)
    if axis < 0:
        axis = axis + rank_base + 1

    if is_shape_unknown(out_shape):
        out = {}
        out_shape.insert(axis, out_n)
        out['shape'] = out_shape
        return out

    out_shape.insert(axis, out_n)
    return out_shape


class Stack(PrimitiveWithInfer):
    r"""
    Stacks a list of tensors in specified axis.

    Refer to :func:`mindspore.ops.stack` for more details.

    Args:
        axis (int, optional): Dimension to stack. The range is [-(R+1), R+1). Default: ``0`` .

    Inputs:
        - **input_x** (Union[tuple, list]) - A Tuple or list of Tensor objects with the same shape and type.

    Outputs:
        Tensor. A stacked Tensor with the same type as `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, ops
        >>> import numpy as np
        >>> data1 = Tensor(np.array([0, 1]).astype(np.float32))
        >>> data2 = Tensor(np.array([2, 3]).astype(np.float32))
        >>> stack = ops.Stack()
        >>> output = stack([data1, data2])
        >>> print(output)
        [[0. 1.]
         [2. 3.]]
    """

    @prim_attr_register
    def __init__(self, axis=0):
        """Initialize Stack"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        validator.check_value_type("axis", axis, [int], self.name)
        self.axis = axis

    def __infer__(self, value):
        x_shape = value['shape']
        x_type = value['dtype']
        self.add_prim_attr('num', len(x_shape))
        self.add_prim_attr('N', len(x_shape))
        all_shape = _get_stack_shape(value, x_shape, x_type, self.axis, self.name)
        out = {}
        tuple_value = value['value']
        input_array = []
        infered_value = None
        dtype = x_type[0]
        if tuple_value is not None and None not in tuple_value:
            for item in tuple_value:
                npy_item = item.asnumpy()
                input_array.append(npy_item)
            infered_value = Tensor(np.stack(input_array, axis=self.axis))

        shape = all_shape.get('shape') if isinstance(all_shape, dict) else all_shape
        out = {'shape': shape,
               'dtype': dtype,
               'value': infered_value}

        return out


class Unstack(Primitive):
    r"""
    Unstacks tensor in specified axis, this is the opposite of ops.Stack.
    Assuming input is a tensor of rank `R`, output tensors will have rank `(R-1)`.

    Refer to :func:`mindspore.ops.unstack` for more details.

    Args:
        axis (int): Dimension along which to unpack. Default: ``0`` .
            Negative values wrap around. The range is [-R, R).
        num (Union[None, int]): The number of output tensors.
            Automatically inferred by input_x and axis if ``None`` . Default: ``None`` .

    Inputs:
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_R)`.
          A tensor to be unstacked and the rank of the tensor must be greater than 0.

    Outputs:
        A tuple of tensors, the shape of each objects is the same.
        Given a tensor of shape :math:`(x_1, x_2, ..., x_R)`. If :math:`0 \le axis`,
        the shape of tensor in output is :math:`(x_1, x_2, ..., x_{axis}, x_{axis+2}, ..., x_R)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> unstack = ops.Unstack()
        >>> input_x = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]))
        >>> output = unstack(input_x)
        >>> print(output)
        (Tensor(shape=[4], dtype=Int64, value= [1, 1, 1, 1]), Tensor(shape=[4], dtype=Int64, value= [2, 2, 2, 2]))
    """

    @prim_attr_register
    def __init__(self, axis=0, num=None):
        """Initialize Unstack"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        validator.check_value_type("axis", axis, [int], self.name)
        if num is not None:
            validator.check_value_type("num", num, [int], self.name)


class Slice(Primitive):
    """
    Slices a tensor in the specified shape.

    Refer to :func:`mindspore.ops.slice` for more details.

    Inputs:
        - **input_x** (Tensor) - The target tensor.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **begin** (Union[tuple, list]) - The beginning of the slice. Only constant value(>=0) is allowed.
        - **size** (Union[tuple, list]) - The size of the slice. Only constant value is allowed.

    Outputs:
        Tensor, the shape is: input `size`, the data type is the same as `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> import numpy as np
        >>> data = Tensor(np.array([[[1, 1, 1], [2, 2, 2]],
        ...                         [[3, 3, 3], [4, 4, 4]],
        ...                         [[5, 5, 5], [6, 6, 6]]]).astype(np.int32))
        >>> slice_op = ops.Slice()
        >>> output = slice_op(data, (1, 0, 0), (1, 1, 3))
        >>> print(output)
        [[[3 3 3]]]
        >>> output = slice_op(data, (1, 0, 0), (1, 1, 2))
        >>> print(output)
        [[[3 3]]]
        >>> output = slice_op(data, (1, 0, 0), (1, 1, 1))
        >>> print(output)
        [[[3]]]
        >>> output = slice_op(data, (1, 1, 0), (1, 1, 3))
        >>> print(output)
        [[[4 4 4]]]
        >>> output = slice_op(data, (1, 0, 1), (1, 1, 2))
        >>> print(output)
        [[[3 3]]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize slice"""
        self.init_prim_io_names(inputs=['x', 'begin', 'size'], outputs=['output'])


class Coalesce(Primitive):
    """
    Returns the coalesced sparse tensor of the input.

    Inputs:
        - **x_indices** (Tensor) - A 2-D Tensor, represents the indices of the nonzero elements of the sparse tensor.
          Supported data type is int64. Its elements should be non-negative. The shape is :math:`(y, x)`.
        - **x_values** (Tensor) - A 1-D Tensor, represents the values corresponding to the indices in `x_indices`.
          Supported data types are float16 and float32. The shape is :math:`(x,)`.
        - **x_shape** (Tensor) - A 1-D Tensor, specifies the shape of the sparse tensor.
          Supported data type is int64. The shape is :math:`(y,)`.

    Outputs:
        - **y_indices** (Tensor) - A 2-D Tensor, represents the indices of the nonzero elements of the sparse tensor.
          Data type is int64. It's elements are non-negative. The shape is :math:`(y, z)`.
          `z` represents the number of different indices in `x_indices`.
        - **y_values** (Tensor) - A 1-D Tensor, represents the values corresponding to the indices in `y_indices`.
          Data type is the same as `x_values`'s. The shape is :math:`(z,)`.
        - **y_shape** (Tensor) - A 1-D Tensor, specifies the shape of the sparse tensor.
          Data type is int64. The shape is :math:`(y,)`.

    Raises:
        TypeError: If the data type of `x_values` is neither float32 nor float16.
        TypeError: If any of the data types of `x_indices` and `x_shape` is not int64.
        ValueError: If any of `x_values` and `x_shape` is not a 1-D tensor.
        ValueError: If `x_indices` is not a 2-D tensor.
        ValueError: If sizes of second dimension of `x_indices` and first dimension of `x_values` are not the same.
        ValueError: If sizes of first dimension of `x_indices` and first dimension of `x_shape` are not the same.
        ValueError: If any of the values of elements of `x_indices` is negative.
        ValueError: If any of the values of elements of `x_indices` exceed the limit set by `x_shape`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x_indices = Tensor([[0, 0, 1], [1, 1, 2]], dtype=mstype.int64)
        >>> x_values = Tensor([1, 5, 4], dtype=mstype.float32)
        >>> x_shape = Tensor([3, 3], dtype=mstype.int64)
        >>> coalesce = ops.Coalesce()
        >>> y_indices, y_values, y_shape = coalesce(x_indices, x_values, x_shape)
        >>> print(y_indices)
        [[0 1]
         [1 2]]
        >>> print(y_values)
        [6. 4.]
        >>> print(y_shape)
        [3 3]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Coalesce."""
        self.init_prim_io_names(inputs=['x_indices', 'x_values', 'x_shape'],
                                outputs=['y_indices', 'y_values', 'y_shape'])


class Rint(Primitive):
    """
    Returns an integer that is closest to `input_x` element-wise.

    Inputs:
        - **input_x** (Tensor) - Input tensor of any dimension, which must be one of the following types:
          float16, float32, float64.
    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If dtype of `input_x` is not in [float16, float32, float64].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.array([-1.6, -0.1, 1.5, 2.0]), mindspore.float32)
        >>> op = ops.Rint()
        >>> output = op(input_x)
        >>> print(output)
        [-2.  0.  2.  2.]
        >>> input_x = Tensor(np.array([[-2.0, -1.9, -1.8, -1.7, -1.6],
        ...                            [-2.0, -1.9, -1.8, -1.7, -1.6]]), mindspore.float32)
        >>> output = op(input_x)
        >>> print(output)
        [[-2. -2. -2. -2. -2.]
         [-2. -2. -2. -2. -2.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Rint."""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])


class StridedSliceV2(Primitive):
    r"""
    StridedSliceV2 will be deprecated by StridedSlice in the future.
    Extracts a strided slice of a tensor.
    Refer to class StridedSlice for more details.

    Args:
        begin_mask (int): Starting index of the slice. Default: ``0`` .
        end_mask (int): Ending index of the slice. Default: ``0`` .
        ellipsis_mask (int): An int mask. Default: ``0`` .
        new_axis_mask (int): An int mask. Default: ``0`` .
        shrink_axis_mask (int): An int mask. Default: ``0`` .

    Inputs:
        - **input_x** (Tensor) - The input Tensor.
        - **begin** (tuple[int]) - A tuple which represents the location where to start. Only
          constant value is allowed.
        - **end** (tuple[int]) - A tuple or which represents the maximum location where to end.
          Only constant value is allowed.
        - **strides** (tuple[int]) - A tuple which represents the stride is continuously added
          before reaching the maximum location. Only constant value is allowed.

    Outputs:
        Tensor, The output is explained by following example.

    Raises:
        TypeError: If `begin_mask`, `end_mask`, `ellipsis_mask`, `new_axis_mask` or `shrink_axis_mask` is not an int.
        TypeError: If `begin`, `end` or `strides` is not a tuple.
        ValueError: If `begin_mask`, `end_mask`, `ellipsis_mask`, `new_axis_mask` or `shrink_axis_mask` is less than 0.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> input_x = Tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]],
        ...                   [[5, 5, 5], [6, 6, 6]]], mindspore.float32)
        >>> strided_slice_v2 = ops.StridedSliceV2()
        >>> output = strided_slice_v2(input_x, (1, 0, 2), (3, 1, 3), (1, 1, 1))
        >>> print(output)
        [[[3.]]
         [[5.]]]
    """

    @prim_attr_register
    def __init__(self,
                 begin_mask=0,
                 end_mask=0,
                 ellipsis_mask=0,
                 new_axis_mask=0,
                 shrink_axis_mask=0):
        """Initialize StridedSliceV2"""
        self.init_prim_io_names(inputs=['x', 'begin', 'end', 'strides'], outputs=['output'])


class DiagPart(PrimitiveWithCheck):
    r"""

    Extracts the diagonal elements from the given Tensor.

    If the `input_x` is a Tensor of shape :math:`[D_1,..., D_k, D_1,..., D_k]`, then the
    output will be a Tensor of rank k of shape :math:`[D_1,..., D_k]` where:
    :math:`output[i_1,..., i_k] = input\_x[i_1,..., i_k, i_1,..., i_k]`.

    Inputs:
        - **input_x** (Tensor) - The rank of input tensor is 2k(k > 0).

    Outputs:
        Tensor, the extracted diagonal has the same dtype as the `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        ValueError: If rank of `input_x` is not even or zero.
        ValueError: If input_shape[i] is not equal to input_shape[i + len(input_shape)/2].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor([[1, 0, 0, 0],
        ...                   [0, 2, 0, 0],
        ...                   [0, 0, 3, 0],
        ...                   [0, 0, 0, 4]])
        >>> diag_part = ops.DiagPart()
        >>> output = diag_part(input_x)
        >>> print(output)
        [1 2 3 4]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize DiagPart"""

    def infer_value(self, x):
        if x is None:
            return None
        # do constant-folding only when x rank is 2
        if len(x.shape) != 2:
            return None
        ret = np.diag(x.asnumpy())
        return Tensor(ret)


class Mvlgamma(Primitive):
    r"""
    Calculates the multivariate log-gamma function element-wise for a given dimension `p`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Refer to :func:`mindspore.ops.mvlgamma` for more details.

    Args:
        p(int): The number of dimensions. And the value of `p` must be greater than or equal to 1.

    Inputs:
        - **x** (Tensor) - The tensor to compute the multivariate log-gamma function,
          which must be one of the following types: float32, float64.
          The shape is :math:`(N,*)`, where :math:`*` means any number of additional dimensions.
          And the value of any element in `x` must be greater than :math:`(p - 1) / 2`.

    Outputs:
        Tensor, has the same shape and type as `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([[3, 4, 5], [4, 2, 6]]), mindspore.float32)
        >>> op = ops.Mvlgamma(p=3)
        >>> y = op(x)
        >>> print(y)
        [[ 2.694925   5.402975   9.140645 ]
         [ 5.402975   1.5963125 13.640454 ]]
    """

    @prim_attr_register
    def __init__(self, p):
        """Initialize Mvlgamma."""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        validator.check_value_type('p', p, [int], self.name)
        validator.check_positive_int(p, 'p', self.name)


class ScatterUpdate(Primitive):
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
        use_locking (bool): Whether to protect the assignment by a lock. Default: ``True`` .

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is 0-D or :math:`(N, *)` where :math:`*` means any number of additional dimensions.
        - **indices** (Tensor) - The index of input tensor. With int32 data type.
          If there are duplicates in indices, the order for updating is undefined.
        - **updates** (Tensor) - The tensor to update the input tensor, has the same type as input,
          and updates.shape = indices.shape + input_x.shape[1:].

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `indices` is not an int32.
        ValueError: If the shape of `updates` is not equal to `indices.shape + input_x.shape[1:]`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> np_x = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]])
        >>> input_x = mindspore.Parameter(Tensor(np_x, mindspore.float32), name="x")
        >>> indices = Tensor(np.array([0, 1]), mindspore.int32)
        >>> np_updates = np.array([[2.0, 1.2, 1.0], [3.0, 1.2, 1.0]])
        >>> updates = Tensor(np_updates, mindspore.float32)
        >>> op = ops.ScatterUpdate()
        >>> output = op(input_x, indices, updates)
        >>> print(output)
        [[2. 1.2  1.]
         [3. 1.2  1.]]
    """
    __mindspore_signature__ = (
        sig.make_sig('x', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('updates', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self, use_locking=True):
        """Initialize ScatterUpdate"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])
        self.add_prim_attr('side_effect_mem', True)


class ScatterNdUpdate(Primitive):
    r"""
    Updates tensor values by using input indices and value.

    Using given values to update tensor value, along with the input indices.

    `input_x` has rank P and `indices` has rank Q where `Q >= 2`.

    `indices` has shape :math:`(i_0, i_1, ..., i_{Q-2}, N)` where `N <= P`.

    The last dimension of `indices` (with length `N` ) indicates slices along the `N` th dimension of `input_x`.

    `updates` is a tensor of rank `Q-1+P-N`, and its shape is:
    :math:`(i_0, i_1, ..., i_{Q-2}, x\_shape_N, ..., x\_shape_{P-1})`.

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.

    Args:
        use_locking (bool): Whether to protect the assignment by a lock. Default: ``True`` .

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is :math:`(N, *)` where :math:`*` means any number of additional dimensions.
        - **indices** (Tensor) - The index of input tensor, with int32 or int64 data type.
        - **updates** (Tensor) - N-D(2D or 3D) Tensor The tensor to be updated to the input tensor,
          has the same type as input. The shape is `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `indices` is not an int32 or an int64.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> np_x = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]])
        >>> input_x = mindspore.Parameter(Tensor(np_x, mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> op = ops.ScatterNdUpdate()
        >>> output = op(input_x, indices, updates)
        >>> print(output)
        [[1.   0.3   3.6]
         [0.4  2.2  -3.2]]
    """

    __mindspore_signature__ = (
        sig.make_sig('input_x', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('updates', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self, use_locking=True):
        """Initialize ScatterNdUpdate"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['input_x', 'indices', 'value'], outputs=['y'])
        self.add_prim_attr('side_effect_mem', True)


class ScatterMax(_ScatterOpDynamic):
    r"""
    Updates the value of the input tensor through the maximum operation.

    Using given values to update tensor value through the max operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    for each :math:`i, ..., j` in `indices.shape`:

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :]
        = \max(\text{input_x}[\text{indices}[i, ..., j], :], \text{updates}[i, ..., j, :])

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type. A RuntimeError will be reported
    when `updates` does not support conversion to the data type required by `input_x`.

    Args:
        use_locking (bool): Whether to protect the assignment by a lock. Default: ``False`` .

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is :math:`(N, *)` where :math:`*` means any number of additional dimensions.
        - **indices** (Tensor) - The index to do max operation whose data type must be mindspore.int32 or
          mindspore.int64.
        - **updates** (Tensor) - The tensor that performs the maximum operation with `input_x`,
          the data type is the same as `input_x`, the shape is `indices.shape + input_x.shape[1:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `indices` is not an int32 or an int64.
        ValueError: If the shape of `updates` is not equal to `indices.shape + x.shape[1:]`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.
        RuntimeError: On the Ascend platform, the input data dimension of `input_x` , `indices`
                      and `updates` is greater than 8 dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops, Parameter
        >>> input_x = Parameter(Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32),
        ...                     name="input_x")
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.ones([2, 2, 3]) * 88, mindspore.float32)
        >>> scatter_max = ops.ScatterMax()
        >>> output = scatter_max(input_x, indices, updates)
        >>> print(output)
        [[88. 88. 88.]
         [88. 88. 88.]]
    """


class ScatterMin(_ScatterOpDynamic):
    r"""
    Updates the value of the input tensor through the minimum operation.

    Using given values to update tensor value through the min operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    for each :math:`i, ..., j` in `indices.shape`:

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :]
        = \min(\text{input_x}[\text{indices}[i, ..., j], :], \text{updates}[i, ..., j, :])

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type. A RuntimeError will be reported
    when `updates` does not support conversion to the data type required by `input_x`.

    Args:
        use_locking (bool): Whether to protect the assignment by a lock. Default: ``False`` .

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is :math:`(N, *)` where :math:`*` means any number of additional dimensions.
        - **indices** (Tensor) - The index to do min operation whose data type must be mindspore.int32 or
          mindspore.int64.
        - **updates** (Tensor) - The tensor doing the min operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices.shape + input_x.shape[1:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `indices` is not an int32 or an int64.
        ValueError: If the shape of `updates` is not equal to `indices.shape + input_x.shape[1:]`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.
        RuntimeError: On the Ascend platform, the input data dimension of `input_x` , `indices`
                      and `updates` is greater than 8 dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops, Parameter
        >>> input_x = Parameter(Tensor(np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]), mindspore.float32),
        ...                     name="input_x")
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> update = Tensor(np.ones([2, 2, 3]), mindspore.float32)
        >>> scatter_min = ops.ScatterMin()
        >>> output = scatter_min(input_x, indices, update)
        >>> print(output)
        [[0. 1. 1.]
         [0. 0. 0.]]
    """


class ScatterAdd(Primitive):
    r"""
    Updates the value of the input tensor through the addition operation.

    Using given values to update tensor value through the add operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    for each `i, ..., j` in `indices.shape`:

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :] \mathrel{+}= \text{updates}[i, ..., j, :]

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.

    Note:
        This is an in-place update operator. Therefore, the `input_x` will be updated after the operation is completed.

    Args:
        use_locking (bool): Whether to protect the assignment by a lock.
            If ``True`` , `input_x` will be protected by the lock.
            Otherwise, the calculation result is undefined. Default: ``False`` .

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
        - **indices** (Tensor) - The index to do min operation whose data type must be mindspore.int32 or
          mindspore.int64.
        - **updates** (Tensor) - The tensor doing the min operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices.shape + x.shape[1:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `indices` is not an int32 or an int64.
        ValueError: If the shape of `updates` is not equal to `indices.shape + x.shape[1:]`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops, Parameter
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.ones([2, 2, 3]), mindspore.float32)
        >>> scatter_add = ops.ScatterAdd()
        >>> output = scatter_add(input_x, indices, updates)
        >>> print(output)
        [[1. 1. 1.]
         [3. 3. 3.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mindspore.float32), name="x")
        >>> # for indices = [[0, 1], [1, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [0.0, 0.0, 0.0] + [1.0, 1.0, 1.0] = [1.0, 1.0, 1.0]
        >>> # input_x[1] = [0.0, 0.0, 0.0] + [3.0, 3.0, 3.0] = [3.0, 3.0, 3.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [3.0, 3.0, 3.0] + [7.0, 7.0, 7.0] = [10.0, 10.0, 10.0]
        >>> # input_x[1] = [10.0, 10.0, 10.0] + [9.0, 9.0, 9.0] = [19.0, 19.0, 19.0]
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> scatter_add = ops.ScatterAdd()
        >>> output = scatter_add(input_x, indices, updates)
        >>> print(output)
        [[ 1.  1.  1.]
         [19. 19. 19.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mindspore.float32), name="x")
        >>> # for indices = [[1, 0], [1, 1]]
        >>> # step 1: [1, 0]
        >>> # input_x[0] = [0.0, 0.0, 0.0] + [3.0, 3.0, 3.0] = [3.0, 3.0, 3.0]
        >>> # input_x[1] = [0.0, 0.0, 0.0] + [1.0, 1.0, 1.0] = [1.0, 1.0, 1.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [1.0, 1.0, 1.0] + [7.0, 7.0, 7.0] = [8.0, 8.0, 8.0]
        >>> # input_x[1] = [8.0, 8.0, 8.0] + [9.0, 9.0, 9.0] = [17.0, 17.0, 17.0]
        >>> indices = Tensor(np.array([[1, 0], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> scatter_add = ops.ScatterAdd()
        >>> output = scatter_add(input_x, indices, updates)
        >>> print(output)
        [[ 3.  3.  3.]
         [17. 17. 17.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mindspore.float32), name="x")
        >>> # for indices = [[0, 1], [0, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [0.0, 0.0, 0.0] + [1.0, 1.0, 1.0] = [1.0, 1.0, 1.0]
        >>> # input_x[1] = [0.0, 0.0, 0.0] + [3.0, 3.0, 3.0] = [3.0, 3.0, 3.0]
        >>> # step 2: [0, 1]
        >>> # input_x[0] = [1.0, 1.0, 1.0] + [7.0, 7.0, 7.0] = [8.0, 8.0, 8.0]
        >>> # input_x[1] = [3.0, 3.0, 3.0] + [9.0, 9.0, 9.0] = [12.0, 12.0, 12.0]
        >>> indices = Tensor(np.array([[0, 1], [0, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> scatter_add = ops.ScatterAdd()
        >>> output = scatter_add(input_x, indices, updates)
        >>> print(output)
        [[ 8.  8.  8.]
         [12. 12. 12.]]
    """
    __mindspore_signature__ = (
        sig.make_sig('x', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('updates', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ScatterAdd"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])
        self.add_prim_attr('side_effect_mem', True)


class ScatterSub(Primitive):
    r"""
    Updates the value of the input tensor through the subtraction operation.

    Using given values to update tensor value through the subtraction operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    for each `i, ..., j` in `indices.shape`:

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :] \mathrel{-}= \text{updates}[i, ..., j, :]

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.

    Args:
        use_locking (bool): Whether to protect the assignment by a lock. Default: ``False`` .

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is :math:`(N, *)` where :math:`*` means any number of additional dimensions.
        - **indices** (Tensor) - The index to do min operation whose data type must be mindspore.int32 or
          mindspore.int64.
        - **updates** (Tensor) - The tensor doing the min operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices_shape + x_shape[1:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `indices` is not an int32.
        ValueError: If the shape of `updates` is not equal to `indices_shape + x_shape[1:]`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops, Parameter
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[0, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]), mindspore.float32)
        >>> scatter_sub = ops.ScatterSub()
        >>> output = scatter_sub(input_x, indices, updates)
        >>> print(output)
        [[-1. -1. -1.]
         [-1. -1. -1.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mindspore.float32), name="x")
        >>> # for indices = [[0, 1], [1, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [0.0, 0.0, 0.0] - [1.0, 1.0, 1.0] = [-1.0, -1.0, -1.0]
        >>> # input_x[1] = [0.0, 0.0, 0.0] - [3.0, 3.0, 3.0] = [-3.0, -3.0, -3.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [-3.0, -3.0, -3.0] - [7.0, 7.0, 7.0] = [-10.0, -10.0, -10.0]
        >>> # input_x[1] = [-10.0, -10.0, -10.0] - [9.0, 9.0, 9.0] = [-19.0, -19.0, -19.0]
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> scatter_sub = ops.ScatterSub()
        >>> output = scatter_sub(input_x, indices, updates)
        >>> print(output)
        [[ -1.  -1.  -1.]
         [-19. -19. -19.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mindspore.float32), name="x")
        >>> # for indices = [[1, 0], [1, 1]]
        >>> # step 1: [1, 0]
        >>> # input_x[0] = [0.0, 0.0, 0.0] - [3.0, 3.0, 3.0] = [-3.0, -3.0, -3.0]
        >>> # input_x[1] = [0.0, 0.0, 0.0] - [1.0, 1.0, 1.0] = [-1.0, -1.0, -1.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [-1.0, -1.0, -1.0] - [7.0, 7.0, 7.0] = [-8.0, -8.0, -8.0]
        >>> # input_x[1] = [-8.0, -8.0, -8.0] - [9.0, 9.0, 9.0] = [-17.0, -17.0, -17.0]
        >>> indices = Tensor(np.array([[1, 0], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> scatter_sub = ops.ScatterSub()
        >>> output = scatter_sub(input_x, indices, updates)
        >>> print(output)
        [[ -3.  -3.  -3.]
         [-17. -17. -17.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mindspore.float32), name="x")
        >>> # for indices = [[0, 1], [0, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [0.0, 0.0, 0.0] - [1.0, 1.0, 1.0] = [-1.0, -1.0, -1.0]
        >>> # input_x[1] = [0.0, 0.0, 0.0] - [3.0, 3.0, 3.0] = [-3.0, -3.0, -3.0]
        >>> # step 2: [0, 1]
        >>> # input_x[0] = [-1.0, -1.0, -1.0] - [7.0, 7.0, 7.0] = [-8.0, -8.0, -8.0]
        >>> # input_x[1] = [-3.0, -3.0, -3.0] - [9.0, 9.0, 9.0] = [-12.0, -12.0, -12.0]
        >>> indices = Tensor(np.array([[0, 1], [0, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mindspore.float32)
        >>> scatter_sub = ops.ScatterSub()
        >>> output = scatter_sub(input_x, indices, updates)
        >>> print(output)
        [[ -8.  -8.  -8.]
         [-12. -12. -12.]]
    """
    __mindspore_signature__ = (
        sig.make_sig('input_x', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('updates', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ScatterSub"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['input_x', 'indices', 'updates'], outputs=['y'])
        self.add_prim_attr('side_effect_mem', True)


class ScatterMul(_ScatterOpDynamic):
    r"""
    Updates the value of the input tensor through the multiply operation.

    Using given values to update tensor value through the mul operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    for each `i, ..., j` in `indices.shape`:

    .. math::

        \text{input_x}[\text{indices}[i, ..., j], :] \mathrel{*}= \text{updates}[i, ..., j, :]

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.

    Args:
        use_locking (bool): Whether to protect the assignment by a lock. Default: ``False`` .

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is :math:`(N, *)` where :math:`*` means any number of additional dimensions.
        - **indices** (Tensor) - The index to do multiply operation whose data type must be mstype.int32 or
          mstype.int64.
        - **updates** (Tensor) - The tensor doing the multiply operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices.shape + input_x.shape[1:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `indices` is not an int32 or an int64.
        ValueError: If the shape of `updates` is not equal to `indices.shape + input_x.shape[1:]`.
        RuntimeError: If the data type of `input_x` and `updates` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor, ops, Parameter
        >>> input_x = Parameter(Tensor(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), mstype.float32), name="x")
        >>> indices = Tensor(np.array([0, 1]), mstype.int32)
        >>> updates = Tensor(np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]), mstype.float32)
        >>> scatter_mul = ops.ScatterMul()
        >>> output = scatter_mul(input_x, indices, updates)
        >>> print(output)
        [[2. 2. 2.]
         [4. 4. 4.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), mstype.float32), name="x")
        >>> # for indices = [[0, 1], [1, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [1.0, 1.0, 1.0] * [1.0, 1.0, 1.0] = [1.0, 1.0, 1.0]
        >>> # input_x[1] = [2.0, 2.0, 2.0] * [3.0, 3.0, 3.0] = [6.0, 6.0, 6.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [6.0, 6.0, 6.0] * [7.0, 7.0, 7.0] = [42.0, 42.0, 42.0]
        >>> # input_x[1] = [42.0, 42.0, 42.0] * [9.0, 9.0, 9.0] = [378.0, 378.0, 378.0]
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mstype.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mstype.float32)
        >>> scatter_mul = ops.ScatterMul()
        >>> output = scatter_mul(input_x, indices, updates)
        >>> print(output)
        [[  1.   1.   1.]
         [378. 378. 378.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), mstype.float32), name="x")
        >>> # for indices = [[1, 0], [1, 1]]
        >>> # step 1: [1, 0]
        >>> # input_x[0] = [1.0, 1.0, 1.0] * [3.0, 3.0, 3.0] = [3.0, 3.0, 3.0]
        >>> # input_x[1] = [2.0, 2.0, 2.0] * [1.0, 1.0, 1.0] = [2.0, 2.0, 2.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [2.0, 2.0, 2.0] * [7.0, 7.0, 7.0] = [14.0, 14.0, 14.0]
        >>> # input_x[1] = [14.0, 14.0, 14.0] * [9.0, 9.0, 9.0] = [126.0, 126.0, 126.0]
        >>> indices = Tensor(np.array([[1, 0], [1, 1]]), mstype.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mstype.float32)
        >>> scatter_mul = ops.ScatterMul()
        >>> output = scatter_mul(input_x, indices, updates)
        >>> print(output)
        [[  3.   3.   3.]
         [126. 126. 126.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), mstype.float32), name="x")
        >>> # for indices = [[0, 1], [0, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [1.0, 1.0, 1.0] * [1.0, 1.0, 1.0] = [1.0, 1.0, 1.0]
        >>> # input_x[1] = [2.0, 2.0, 2.0] * [3.0, 3.0, 3.0] = [6.0, 6.0, 6.0]
        >>> # step 2: [0, 1]
        >>> # input_x[0] = [1.0, 1.0, 1.0] * [7.0, 7.0, 7.0] = [7.0, 7.0, 7.0]
        >>> # input_x[1] = [6.0, 6.0, 6.0] * [9.0, 9.0, 9.0] = [54.0, 54.0, 54.0]
        >>> indices = Tensor(np.array([[0, 1], [0, 1]]), mstype.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[7.0, 7.0, 7.0], [9.0, 9.0, 9.0]]]), mstype.float32)
        >>> scatter_mul = ops.ScatterMul()
        >>> output = scatter_mul(input_x, indices, updates)
        >>> print(output)
        [[ 7.  7.  7.]
         [54. 54. 54.]]
    """


class ScatterDiv(_ScatterOpDynamic):
    r"""
    Updates the value of the input tensor through the divide operation.

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
        use_locking (bool): Whether to protect the assignment by a lock. Default: ``False`` .

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is :math:`(N, *)` where :math:`*` means any number of additional dimensions.
        - **indices** (Tensor) - The index to do divide operation whose data type must be mstype.int32 or
          mstype.int64.
        - **updates** (Tensor) - The tensor doing the divide operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices.shape + input_x.shape[1:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `use_locking` is not a bool.
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
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor, ops, Parameter
        >>> input_x = Parameter(Tensor(np.array([[6.0, 6.0, 6.0], [2.0, 2.0, 2.0]]), mstype.float32), name="x")
        >>> indices = Tensor(np.array([0, 1]), mstype.int32)
        >>> updates = Tensor(np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]), mstype.float32)
        >>> scatter_div = ops.ScatterDiv()
        >>> output = scatter_div(input_x, indices, updates)
        >>> print(output)
        [[3. 3. 3.]
         [1. 1. 1.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[105.0, 105.0, 105.0],
        ...                                      [315.0, 315.0, 315.0]]), mstype.float32), name="x")
        >>> # for indices = [[0, 1], [1, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [105.0, 105.0, 105.0] / [1.0, 1.0, 1.0] = [105.0, 105.0, 105.0]
        >>> # input_x[1] = [315.0, 315.0, 315.0] / [3.0, 3.0, 3.0] = [105.0, 105.0, 105.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [105.0, 105.0, 105.0] / [5.0, 5.0, 5.0] = [21.0, 21.0, 21.0]
        >>> # input_x[1] = [21.0, 21.0, 21.0] / [7.0, 7.0, 7.0] = [3.0, 3.0, 3.0]
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mstype.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[5.0, 5.0, 5.0], [7.0, 7.0, 7.0]]]), mstype.float32)
        >>> scatter_div = ops.ScatterDiv()
        >>> output = scatter_div(input_x, indices, updates)
        >>> print(output)
        [[105. 105. 105.]
         [  3.   3.   3.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[105.0, 105.0, 105.0],
        ...                                      [315.0, 315.0, 315.0]]), mstype.float32), name="x")
        >>> # for indices = [[1, 0], [1, 1]]
        >>> # step 1: [1, 0]
        >>> # input_x[0] = [105.0, 105.0, 105.0] / [3.0, 3.0, 3.0] = [35.0, 35.0, 35.0]
        >>> # input_x[1] = [315.0, 315.0, 315.0] / [1.0, 1.0, 1.0] = [315.0, 315.0, 315.0]
        >>> # step 2: [1, 1]
        >>> # input_x[1] = [315.0, 315.0, 315.0] / [5.0, 5.0, 5.0] = [63.0 63.0 63.0]
        >>> # input_x[1] = [63.0 63.0 63.0] / [7.0, 7.0, 7.0] = [9.0, 9.0, 9.0]
        >>> indices = Tensor(np.array([[1, 0], [1, 1]]), mstype.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[5.0, 5.0, 5.0], [7.0, 7.0, 7.0]]]), mstype.float32)
        >>> scatter_div = ops.ScatterDiv()
        >>> output = scatter_div(input_x, indices, updates)
        >>> print(output)
        [[35. 35. 35.]
         [ 9.  9.  9.]]
        >>> # for input_x will be updated after the operation is completed. input_x need to be re-initialized.
        >>> input_x = Parameter(Tensor(np.array([[105.0, 105.0, 105.0],
        ...                                      [315.0, 315.0, 315.0]]), mstype.float32), name="x")
        >>> # for indices = [[0, 1], [0, 1]]
        >>> # step 1: [0, 1]
        >>> # input_x[0] = [105.0, 105.0, 105.0] / [1.0, 1.0, 1.0] = [105.0, 105.0, 105.0]
        >>> # input_x[1] = [315.0, 315.0, 315.0] / [3.0, 3.0, 3.0] = [105.0, 105.0, 105.0]
        >>> # step 2: [0, 1]
        >>> # input_x[0] = [105.0, 105.0, 105.0] / [5.0, 5.0, 5.0] = [21.0, 21.0, 21.0]
        >>> # input_x[1] = [105.0, 105.0, 105.0] / [7.0, 7.0, 7.0] = [15.0, 15.0, 15.0]
        >>> indices = Tensor(np.array([[0, 1], [0, 1]]), mstype.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        ...                            [[5.0, 5.0, 5.0], [7.0, 7.0, 7.0]]]), mstype.float32)
        >>> scatter_div = ops.ScatterDiv()
        >>> output = scatter_div(input_x, indices, updates)
        >>> print(output)
        [[21. 21. 21.]
         [15. 15. 15.]]
    """


class ScatterNdAdd(Primitive):
    r"""
    Applies sparse addition to individual values or slices in a tensor.

    Using given values to update tensor value through the add operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    Refer to :func:`mindspore.ops.scatter_nd_add` for more details.

    Args:
        use_locking (bool, optional): Whether to protect the assignment by a lock. Default: ``False`` .

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is :math:`(N, *)` where :math:`*` means any number of additional dimensions.
        - **indices** (Tensor) - The index to do add operation whose data type must be mindspore.int32.
          The rank of indices must be at least 2 and `indices.shape[-1] <= len(shape)`.
        - **updates** (Tensor) - The tensor doing the add operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops, Parameter
        >>> input_x = Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
        >>> updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
        >>> use_locking = False
        >>> scatter_nd_add = ops.ScatterNdAdd(use_locking)
        >>> output = scatter_nd_add(input_x, indices, updates)
        >>> print(output)
        [ 1. 10.  9.  4. 12.  6.  7. 17.]
        >>> input_x = Parameter(Tensor(np.zeros((4, 4, 4)), mindspore.int32))
        >>> indices = Tensor(np.array([[0], [2]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]), mindspore.int32)
        >>> use_locking = False
        >>> scatter_nd_add = ops.ScatterNdAdd(use_locking)
        >>> output = scatter_nd_add(input_x, indices, updates)
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
    __mindspore_signature__ = (
        sig.make_sig('x', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('updates', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize _ScatterOp"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])
        self.add_prim_attr('side_effect_mem', True)


class ScatterNdSub(Primitive):
    r"""
    Applies sparse subtraction to individual values or slices in a tensor.

    Using given values to update tensor value through the subtraction operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    Refer to :func:`mindspore.ops.scatter_nd_sub` for more details.

    Args:
        use_locking (bool, optional): Whether to protect the assignment by a lock. Default: ``False`` .

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
          The shape is :math:`(N, *)` where :math:`*` means any number of additional dimensions.
        - **indices** (Tensor) - The index to do sub operation whose data type must be mindspore.int32.
          The rank of indices must be at least 2 and `indices.shape[-1] <= len(shape)`.
        - **updates** (Tensor) - The tensor doing the sub operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops, Parameter
        >>> input_x = Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
        >>> updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
        >>> use_locking = False
        >>> scatter_nd_sub = ops.ScatterNdSub(use_locking)
        >>> output = scatter_nd_sub(input_x, indices, updates)
        >>> print(output)
        [ 1. -6. -3.  4. -2.  6.  7. -1.]
        >>> input_x = Parameter(Tensor(np.zeros((4, 4, 4)), mindspore.int32))
        >>> indices = Tensor(np.array([[0], [2]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]), mindspore.int32)
        >>> use_locking = False
        >>> scatter_nd_sub = ops.ScatterNdSub(use_locking)
        >>> output = scatter_nd_sub(input_x, indices, updates)
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

    __mindspore_signature__ = (
        sig.make_sig('input_x', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('updates', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ScatterNdSub"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['input_x', 'indices', 'updates'], outputs=['y'])
        self.add_prim_attr('side_effect_mem', True)


class ScatterNdMul(_ScatterNdOp):
    r"""
    Applies sparse multiplication to individual values or slices in a tensor.

    Using given values to update parameter value through the multiplication operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Refer to :func:`mindspore.ops.scatter_nd_mul` for more details.

    Args:
        use_locking (bool, optional): Whether to protect the assignment by a lock. Default: ``False`` .

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
        - **indices** (Tensor) - The index to do mul operation whose data type must be int32 or int64.
          The rank of indices must be at least 2 and `indices.shape[-1] <= len(shape)`.
        - **updates** (Tensor) - The tensor to do the mul operation with `input_x`.
          The data type is same as `input_x`, and the shape is `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops, Parameter
        >>> input_x = Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
        >>> updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
        >>> scatter_nd_mul = ops.ScatterNdMul()
        >>> output = scatter_nd_mul(input_x, indices, updates)
        >>> print(output)
        [ 1. 16. 18.  4. 35.  6.  7. 72.]
        >>> input_x = Parameter(Tensor(np.ones((4, 4, 4)), mindspore.int32))
        >>> indices = Tensor(np.array([[0], [2]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]), mindspore.int32)
        >>> scatter_nd_mul = ops.ScatterNdMul()
        >>> output = scatter_nd_mul(input_x, indices, updates)
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


class ScatterNdDiv(_ScatterNdOp):
    r"""
    Applies sparse division to individual values or slices in a tensor.

    Using given values to update tensor value through the division operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Refer to :func:`mindspore.ops.scatter_nd_div` for more details.

    Args:
        use_locking (bool, optional): Whether to protect the assignment by a lock. Default: ``False`` .

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
        - **indices** (Tensor) - The index to do div operation whose data type must be int32 or int64.
          The rank of indices must be at least 2 and `indices.shape[-1] <= len(shape)`.
        - **updates** (Tensor) - The tensor to do the div operation with `input_x`.
          The data type is same as `input_x`, and the shape is `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops, Parameter
        >>> input_x = Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
        >>> updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
        >>> use_locking = False
        >>> scatter_nd_div = ops.ScatterNdDiv(use_locking)
        >>> output = scatter_nd_div(input_x, indices, updates)
        >>> print(output)
        [1.         0.25       0.5        4.         0.71428573 6.
         7.         0.8888889 ]
        >>> input_x = Parameter(Tensor(np.ones((4, 4, 4)), mindspore.float32))
        >>> indices = Tensor(np.array([[0], [2]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]), mindspore.float32)
        >>> use_locking = False
        >>> scatter_nd_div = ops.ScatterNdDiv(use_locking)
        >>> output = scatter_nd_div(input_x, indices, updates)
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


class ScatterNdMax(_ScatterNdOp):
    r"""
    Applies sparse maximum to individual values or slices in a tensor.

    Using given values to update parameter value through the maximum operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    Refer to :func:`mindspore.ops.scatter_nd_max` for more details.

    Args:
        use_locking (bool, optional): Whether to protect the assignment by a lock. Default: ``False`` .

    Inputs:
        - **input_x** (Parameter) -The target tensor, with data type of Parameter.
        - **indices** (Tensor) - The index to do maximum operation whose data type must be int32 or int64.
          The rank of indices must be at least 2 and `indices.shape[-1] <= len(shape)`.
        - **updates** (Tensor) - The tensor to do the max operation with `input_x`.
          The data type is same as `input_x`, and the shape is `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops, Parameter
        >>> input_x = Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
        >>> updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
        >>> scatter_nd_max = ops.ScatterNdMax()
        >>> output = scatter_nd_max(input_x, indices, updates)
        >>> print(output)
        [ 1. 8. 6.  4. 7.  6.  7. 9.]
        >>> input_x = Parameter(Tensor(np.ones((4, 4, 4)), mindspore.int32))
        >>> indices = Tensor(np.array([[0], [2]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]), mindspore.int32)
        >>> scatter_nd_max = ops.ScatterNdMax()
        >>> output = scatter_nd_max(input_x, indices, updates)
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

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ScatterNdMax"""
        super().__init__(use_locking)


class ScatterNdMin(_ScatterNdOp):
    r"""
    Applies sparse minimum to individual values or slices in a tensor.

    Using given values to update tensor value through the minimum operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    Refer to :func:`mindspore.ops.scatter_nd_min` for more details.

    Args:
        use_locking (bool, optional): Whether to protect the assignment by a lock. Default: ``False`` .

    Inputs:
        - **input_x** (Parameter) -The target tensor, with data type of Parameter.
        - **indices** (Tensor) - The index to do minimum operation whose data type must be int32 or int64.
          The rank of indices must be at least 2 and `indices.shape[-1] <= len(shape)`.
        - **updates** (Tensor) - The tensor to do the max operation with `input_x`.
          The data type is same as `input_x`, and the shape is `indices.shape[:-1] + x.shape[indices.shape[-1]:]`.

    Outputs:
        Tensor, the updated `input_x`, has the same shape and type as `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops, Parameter
        >>> input_x = Parameter(Tensor(np.ones(8) * 10, mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
        >>> updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
        >>> use_locking = False
        >>> scatter_nd_min = ops.ScatterNdMin(use_locking)
        >>> output = scatter_nd_min(input_x, indices, updates)
        >>> print(output)
        [10.  8.  6. 10.  7. 10. 10.  9.]
        >>> input_x = Parameter(Tensor(np.ones((4, 4, 4)) * 10, mindspore.int32))
        >>> indices = Tensor(np.array([[0], [2]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ...                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]), mindspore.int32)
        >>> use_locking = False
        >>> scatter_nd_min = ops.ScatterNdMin(use_locking)
        >>> output = scatter_nd_min(input_x, indices, updates)
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

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ScatterNdMin"""
        super().__init__(use_locking)


class SpaceToDepth(Primitive):
    r"""
    Rearrange blocks of spatial data into depth.

    The output tensor's `height` dimension is :math:`height / block\_size`.

    The output tensor's `weight` dimension is :math:`weight / block\_size`.

    The depth of output tensor is :math:`block\_size * block\_size * input\_depth`.

    The input tensor's height and width must be divisible by `block_size`.
    The data format is "NCHW".

    Args:
        block_size (int): The block size used to divide spatial data. It must be >= 2.

    Inputs:
        - **x** (Tensor) - The target tensor. The data type is Number. It must be a 4-D tensor.

    Outputs:
        Tensor, the same data type as `x`. It must be a 4-D tensor. Tensor of shape
        :math:`(N, (C_{in} * \text{block_size} * 2), H_{in} / \text{block_size}, W_{in} / \text{block_size})`.

    Raises:
        TypeError: If `block_size` is not an int.
        ValueError: If `block_size` is less than 2.
        ValueError: If length of shape of `x` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.random.rand(1,3,2,2), mindspore.float32)
        >>> block_size = 2
        >>> space_to_depth = ops.SpaceToDepth(block_size)
        >>> output = space_to_depth(x)
        >>> print(output.shape)
        (1, 12, 1, 1)
    """

    @prim_attr_register
    def __init__(self, block_size):
        """Initialize SpaceToDepth"""
        validator.check_value_type('block_size', block_size, [int], self.name)
        validator.check('block_size', block_size, self.name, 2, validator.GE)
        self.block_size = block_size
        self.add_prim_attr("data_format", "NCHW")
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class DepthToSpace(Primitive):
    r"""
    Rearrange blocks of depth data into spatial dimensions.

    This is the reverse operation of SpaceToDepth.

    The depth of output tensor is :math:`input\_depth / (block\_size * block\_size)`.

    The output tensor's `height` dimension is :math:`height * block\_size`.

    The output tensor's `weight` dimension is :math:`weight * block\_size`.

    The input tensor's depth must be divisible by `block_size * block_size`.
    The data format is "NCHW".

    Args:
        block_size (int): The block size used to divide depth data. It must be >= 2.

    Inputs:
        - **x** (Tensor) - The target tensor. It must be a 4-D tensor with shape :math:`(N, C_{in}, H_{in}, W_{in})`.
          The data type is Number.

    Outputs:
        Tensor of shape :math:`(N, C_{in} / \text{block_size} ^ 2, H_{in} * \text{block_size},
        W_{in} * \text{block_size})`.

    Raises:
        TypeError: If `block_size` is not an int.
        ValueError: If `block_size` is less than 2.
        ValueError: If length of shape of `x` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.random.rand(1, 12, 1, 1), mindspore.float32)
        >>> block_size = 2
        >>> depth_to_space = ops.DepthToSpace(block_size)
        >>> output = depth_to_space(x)
        >>> print(output.shape)
        (1, 3, 2, 2)
    """

    @prim_attr_register
    def __init__(self, block_size):
        """Initialize DepthToSpace"""
        validator.check_value_type('block_size', block_size, [int], self.name)
        validator.check('block_size', block_size, '', 2, validator.GE, self.name)
        self.block_size = block_size
        self.add_prim_attr("data_format", "NCHW")
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class SpaceToBatch(Primitive):
    r"""
    SpaceToBatch is deprecated. Please use :class:`mindspore.ops.SpaceToBatchND` instead.
    Divides spatial dimensions into blocks and combines the block size with the original batch.

    This operation will divide spatial dimensions (H, W) into blocks with `block_size`, the output tensor's H and W
    dimension is the corresponding number of blocks after division. The output tensor's batch dimension is the
    product of the original batch and the square of block_size. Before division, the spatial dimensions
    of the input are zero padded according to paddings if necessary.

    Args:
        block_size (int): The block size of dividing blocks with value greater than or equal to 2.
        paddings (Union[tuple, list]): The padding values for H and W dimension, containing 2 subtraction lists.
            Each subtraction list contains 2 integer value. All values must be greater than 0.
            paddings[i] specifies the paddings for the spatial dimension i, which corresponds to the
            input dimension i+2. It is required that input_shape[i+2]+paddings[i][0]+paddings[i][1]
            is divisible by block_size.

    Inputs:
        - **input_x** (Tensor) - The input tensor. It must be a 4-D tensor. The data type is Number.

    Outputs:
        Tensor, the output tensor with the same data type as input. Assume input shape is :math:`(n, c, h, w)` with
        :math:`block\_size` and :math:`paddings`. The shape of the output tensor will be :math:`(n', c', h', w')`,
        where

        :math:`n' = n*(block\_size*block\_size)`

        :math:`c' = c`

        :math:`h' = (h+paddings[0][0]+paddings[0][1])//block\_size`

        :math:`w' = (w+paddings[1][0]+paddings[1][1])//block\_size`

    Raises:
        TypeError: If `block_size` is not an int.
        ValueError: If `block_size` is less than 2.

    Supported Platforms:
        Deprecated

    Examples:
        >>> block_size = 2
        >>> paddings = [[0, 0], [0, 0]]
        >>> space_to_batch = ops.SpaceToBatch(block_size, paddings)
        >>> input_x = Tensor(np.array([[[[1, 2], [3, 4]]]]), mindspore.float32)
        >>> output = space_to_batch(input_x)
        >>> print(output)
        [[[[1.]]]
         [[[2.]]]
         [[[3.]]]
         [[[4.]]]]
    """

    @prim_attr_register
    def __init__(self, block_size, paddings):
        """Initialize SpaceToBatch"""
        logger.warning("WARN_DEPRECATED: The usage of SpaceToBatch is deprecated."
                       " Please use SpaceToBatchND.")
        validator.check_value_type('block_size', block_size, [int], self.name)
        validator.check('block_size', block_size, self.name, 2, validator.GE, self.name)
        self.block_size = block_size
        validator.check('paddings shape', np.array(paddings).shape, self.name, (2, 2), validator.EQ, self.name)
        for elem in itertools.chain(*paddings):
            validator.check_non_negative_int(elem, 'paddings element', self.name)
            validator.check_value_type('paddings element', elem, [int], self.name)
        self.paddings = paddings


class BatchToSpace(PrimitiveWithInfer):
    r"""
    Divides batch dimension with blocks and interleaves these blocks back into spatial dimensions.

    This operation will divide batch dimension N into blocks with block_size, the output tensor's N dimension
    is the corresponding number of blocks after division. The output tensor's H, W dimension is product of
    original H, W dimension and block_size with given amount to crop from dimension, respectively.

    Args:
        block_size (int): The block size of division, has the value not less than 2.
        crops (Union[list(int), tuple(int)]): The crop value for H and W dimension, containing 2 subtraction lists.
            Each list contains 2 integers.
            All values must be not less than 0. crops[i] specifies the crop values for the spatial dimension i, which
            corresponds to the input dimension i+2. It is required that
            :math:`input\_shape[i+2]*block\_size > crops[i][0]+crops[i][1]` .

    Inputs:
        - **input_x** (Tensor) - The input tensor. It must be a 4-D tensor, dimension 0 must be divisible by
          product of `block_shape`. The data type is float16 or float32.

    Outputs:
        Tensor, the output tensor with the same type as input. Assume input shape is :math:`(n, c, h, w)` with
        block_size and crops. The output shape will be :math:`(n', c', h', w')`, where

        :math:`n' = n//(block\_size*block\_size)`

        :math:`c' = c`

        :math:`h' = h*block\_size-crops[0][0]-crops[0][1]`

        :math:`w' = w*block\_size-crops[1][0]-crops[1][1]`

    Raises:
        TypeError: If `block_size` or element of `crops` is not an int.
        TypeError: If `crops` is neither list nor tuple.
        ValueError: If `block_size` is less than 2.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> block_size = 2
        >>> crops = [[0, 0], [0, 0]]
        >>> batch_to_space = ops.BatchToSpace(block_size, crops)
        >>> input_x = Tensor(np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]]), mindspore.float32)
        >>> output = batch_to_space(input_x)
        >>> print(output)
        [[[[1.  2.]
           [3.  4.]]]]

    """

    @prim_attr_register
    def __init__(self, block_size, crops):
        """Initialize BatchToSpace"""
        logger.warning("WARN_DEPRECATED: The usage of BatchToSpace is deprecated."
                       " Please use BatchToSpaceND.")
        validator.check_value_type('block_size', block_size, [int], self.name)
        validator.check('block_size', block_size, '', 2, validator.GE, self.name)
        self.block_size = block_size
        validator.check_value_type('crops type', crops, [list, tuple], self.name)
        validator.check('crops shape', np.array(crops).shape, self.name, (2, 2))
        for elem in itertools.chain(*crops):
            validator.check_non_negative_int(elem, 'crops element', self.name)
            validator.check_value_type('crops element', elem, [int], self.name)
        self.crops = crops

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('input_x', x_dtype, mstype.number_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape):
        validator.check('rank of input_x', len(x_shape), self.name, 4)
        out_shape = copy.deepcopy(x_shape)
        for i in range(2):
            x_block_prod = out_shape[i + 2] * self.block_size
            crops_sum = self.crops[i][0] + self.crops[i][1]
            validator.check("x block shape prod", x_block_prod, 'crops sum', crops_sum, validator.GT, self.name)
            out_shape[i + 2] = x_block_prod - crops_sum
        block_size_prod = self.block_size * self.block_size
        if out_shape[0] % block_size_prod != 0:
            raise ValueError(f"For '{self.name}', the shape of output with index 0 must be divided exactly "
                             f"by block_size_prod, but got the shape of output: {out_shape} and "
                             f"block_size_prod: {block_size_prod}.")
        out_shape[0] = out_shape[0] // block_size_prod
        return out_shape


class SpaceToBatchND(Primitive):
    r"""
    Divides spatial dimensions into blocks and combines the block size with the original batch.

    This operation will divide spatial dimensions into blocks with `block_shape`, and then the output tensor's spatial
    dimension is the corresponding number of blocks after division. The output tensor's batch dimension is the
    product of the original batch and all elements in `block_shape`.
    Before division, the spatial dimensions of the input are zero padded according to paddings if necessary.

    Args:
        block_shape (Union[list(int), tuple(int), int]): The block shape of dividing block
            with all elements greater than or euqal to 1. If `block_shape` is a list or tuple,
            the length of `block_shape` is the number of spatial dimensions, called M later.
            If `block_shape` is an int, the block size of M dimensions are the same, equal to `block_shape`.
            In this case of Ascend, M must be 2.
        paddings (Union[tuple, list]): The padding values for spatial dimensions, containing M subtraction list.
            Each contains 2 integer values. All values must be greater than or equal to 0.
            `paddings[i]` specifies the paddings for the spatial dimension i,
            which corresponds to the input dimension i + offset,where offset = N-M,
            and N is the number of input dimensions.
            For each i, input_shape[i + offset]+paddings[i][0]+paddings[i][1]
            should be divisible by block_shape[i].

    Inputs:
        - **input_x** (Tensor) - The input tensor. The input tensor must be a 4-D tensor on Ascend.

    Outputs:
        Tensor, the output tensor with the same data type as the input.
        Assume the input shape is :math:`(n, c_1, ... c_k, w_1, ..., w_M)` with
        :math:`block\_shape` and :math:`paddings`.
        The shape of the output tensor will be :math:`(n', c_1, ... c_k, w'_1, ..., w'_M)`,
        where

        .. math::
            \begin{array}{ll} \\
                n' = n*(block\_shape[0]*...*block\_shape[M-1]) \\
                w'_i = (w_i+paddings[i-1][0]+paddings[i-1][1])//block\_shape[i-1]
            \end{array}

    Raises:
        TypeError: If `block_shape` is not one of list, tuple, int.
        TypeError: If `paddings` is neither list nor tuple.
        ValueError: If `block_shape` is not one dimensional when `block_shape` is a list or tuple.
        ValueError: If the length of `block_shape` is not 2 on Ascend.
        ValueError: If shape of `paddings` is not (M, 2), where M is the length of `block_shape`.
        ValueError: If the element of `block_shape` is not an integer larger than or equal to 1.
        ValueError: If the element of `paddings` is not an integer larger than or euqal to 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, ops
        >>> import numpy as np
        >>> block_shape = [2, 2]
        >>> paddings = [[0, 0], [0, 0]]
        >>> space_to_batch_nd = ops.SpaceToBatchND(block_shape, paddings)
        >>> input_x = Tensor(np.array([[[[1, 2], [3, 4]]]]), mindspore.float32)
        >>> output = space_to_batch_nd(input_x)
        >>> print(output)
        [[[[1.]]]
         [[[2.]]]
         [[[3.]]]
         [[[4.]]]]
    """

    @prim_attr_register
    def __init__(self, block_shape, paddings):
        """Initialize SpaceToBatchND"""
        validator.check_value_type('paddings type', paddings, [list, tuple], self.name)
        validator.check('paddings length', len(paddings), '', 1, validator.GE, self.name)

        if isinstance(block_shape, int):
            block_shape = (block_shape,) * np.array(paddings).shape[0]

        self.add_prim_attr("block_shape", block_shape)
        validator.check_value_type('block_shape type', block_shape, [list, tuple], self.name)
        validator.check('block_shape shape', len(np.array(block_shape).shape),
                        'default value', 1, validator.EQ, self.name)
        block_rank = len(block_shape)
        if context.get_context("device_target") == "Ascend":
            validator.check('block_shape length', block_rank, 'default value', 2, validator.EQ, self.name)
        for elem in block_shape:
            validator.check('block_shape element', elem, 'min value', 1, validator.GE, self.name)
            validator.check_value_type('block_shape element', elem, [int], self.name)
        self.block_shape = block_shape

        validator.check(
            'paddings shape', np.array(paddings).shape, 'default value', (block_rank, 2), validator.EQ, self.name)
        for elem in itertools.chain(*paddings):
            validator.check_non_negative_int(elem, 'paddings element', self.name)
            validator.check_value_type('paddings element', elem, [int], self.name)
        self.paddings = paddings


class BatchToSpaceNDV2(Primitive):
    r"""
    Divides batch dimension with blocks and interleaves these blocks back into spatial dimensions.

    Refer to :func:`mindspore.ops.batch_to_space_nd` for more details.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **input_x** (Tensor) - The input tensor. It must be greater or equal to 2-D
          tensor(equal to 4-D tensor on Ascend), batch dimension must be divisible by product of `block_shape`.
        - **block_shape** (Tensor) - The block shape of dividing block with all value greater
          than or equal to 1. If `block_shape` is a tuple or list, the length of `block_shape` is M corresponding
          to the number of spatial dimensions. If `block_shape` is an int, the block size of M dimensions are the
          same, equal to `block_shape`. In this case of Ascend, M must be 2.
        - **crops** (Union[list(int), tuple(int)]) - The crops values for spatial dimensions, containing
          M subtraction list. Each contains 2 integer values. All values must be >= 0. crops[i] specifies
          the crops values for spatial dimension i, which corresponds to input dimension i + offset,
          where offset = N-M, and N is the number of input dimensions. It is required that
          :math:`input\_shape[i+offset]*block\_shape[i] > crops[i][0]+crops[i][1]`

    Outputs:
        Tensor, contains the result of batch division and rearrangement of the original Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> block_shape = Tensor(np.array([2, 2]), mindspore.int32)
        >>> crops = [[0, 0], [0, 0]]
        >>> input_x = Tensor(np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]]), mindspore.float32)
        >>> output = ops.BatchToSpaceNDV2(input_x, block_shape, crops)
        >>> print(output)
        [[[[1.  2.]
           [3.  4.]]]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize BatchToSpaceNDV2"""
        self.init_prim_io_names(inputs=['input_x', 'block_shape', 'crops'], outputs=['y'])
        self.add_prim_attr('origin_format', 'NHWC')


class Meshgrid(PrimitiveWithInfer):
    """
    Generates coordinate matrices from given coordinate tensors.

    Refer to :func:`mindspore.ops.meshgrid` for more details.

    Args:
        indexing (str, optional): Cartesian ``'xy'`` or
            matrix ``'ij'`` indexing of output. In the 2-D case with
            inputs of length `M` and `N`, the outputs are of shape :math:`(N, M)`
            for ``'xy'`` indexing and :math:`(M, N)` for ``'ij'`` indexing. In the 3-D
            case with inputs of length `M`, `N` and `P`, outputs are of shape
            :math:`(N, M, P)` for ``'xy'`` indexing and :math:`(M, N, P)` for ``'ij'`` indexing.
            Default: ``'xy'``.

    Inputs:
        - **input** (Union[tuple]) - A Tuple of N 1-D Tensor objects.
          The length of input should be greater than 1. The data type is Number.

    Outputs:
        Tensors, A Tuple of N N-D Tensor objects. The data type is the same with the Inputs.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
        >>> y = Tensor(np.array([5, 6, 7]).astype(np.int32))
        >>> z = Tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
        >>> inputs = (x, y, z)
        >>> meshgrid = ops.Meshgrid(indexing='xy')
        >>> output = meshgrid(inputs)
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

    @prim_attr_register
    def __init__(self, indexing="xy"):
        """Initialize Meshgrid."""
        validator.check_value_type("indexing", indexing, (str), self.name)
        validator.check_string(indexing.lower(), ["xy", "ij"], "indexing", self.name)
        self.indexing = indexing

    def infer_shape(self, x_shape):
        validator.check_value_type("shape", x_shape, [tuple], self.name)
        validator.check_int(len(x_shape), 2, validator.GE, "len of input", self.name)
        n = len(x_shape)
        shape_0 = []
        for s in x_shape:
            validator.check_int(len(s), 1, validator.EQ, 'each input rank', self.name)
            shape_0.append(s[0])
        if self.indexing == "xy":
            shape_0[0], shape_0[1] = shape_0[1], shape_0[0]
        out_shape = tuple(tuple(shape_0) for _ in range(n))
        return out_shape

    def infer_dtype(self, x_type):
        validator.check_subclass("input[0]", x_type[0], mstype.tensor_type, self.name)
        n = len(x_type)
        for i in range(1, n):
            validator.check('x_type[%d]' % i, x_type[i], 'base', x_type[0], validator.EQ, self.name, TypeError)
        return x_type


class ReverseSequence(PrimitiveWithInfer):
    r"""
    Reverses variable length slices.

    Args:
        seq_dim (int): The dimension where reversal is performed. Required.
        batch_dim (int): The input is sliced in this dimension. Default: ``0`` .

    Inputs:
        - **x** (Tensor) - The input to reverse, supporting all number types including bool.
        - **seq_lengths** (Tensor) - Must be a 1-D vector with int32 or int64 types.

    Outputs:
        Tensor, with the same shape and data type as `x`.

    Raises:
        TypeError: If `seq_dim` or `batch_dim` is not an int.
        ValueError: If :math:`len(seq\_lengths) != x.shape[batch\_dim]`.
        ValueError: If :math:`batch\_dim == seq\_dim`.
        ValueError: If :math:`seq\_dim < 0` or :math:`seq\_dim >= len(x.shape)`.
        ValueError: If :math:`batch\_dim < 0` or :math:`batch\_dim >= len(x.shape)`.
        RuntimeError: If any value of `seq_lengths` is less than 0.
        RuntimeError: If any value of `seq_lengths` is larger than `x.shape[seq_dim]`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([1, 2, 3]))
        >>> reverse_sequence = ops.ReverseSequence(seq_dim=1)
        >>> output = reverse_sequence(x, seq_lengths)
        >>> print(output)
        [[1. 2. 3.]
         [5. 4. 6.]
         [9. 8. 7.]]
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([1, 2, 3]))
        >>> reverse_sequence = ops.ReverseSequence(seq_dim=0, batch_dim=1)
        >>> output = reverse_sequence(x, seq_lengths)
        >>> print(output)
        [[1. 5. 9.]
         [4. 2. 6.]
         [7. 8. 3.]]
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([2, 2, 3]))
        >>> reverse_sequence = ops.ReverseSequence(seq_dim=1)
        >>> output = reverse_sequence(x, seq_lengths)
        >>> print(output)
        [[2. 1. 3.]
         [5. 4. 6.]
         [9. 8. 7.]]
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([3, 2, 3]))
        >>> reverse_sequence = ops.ReverseSequence(seq_dim=1)
        >>> output = reverse_sequence(x, seq_lengths)
        >>> print(output)
        [[3. 2. 1.]
         [5. 4. 6.]
         [9. 8. 7.]]
        >>> x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([4, 4]))
        >>> reverse_sequence = ops.ReverseSequence(seq_dim=1)
        >>> output = reverse_sequence(x, seq_lengths)
        >>> print(output)
        [[4. 3. 2. 1.]
         [8. 7. 6. 5.]]
    """

    @prim_attr_register
    def __init__(self, seq_dim, batch_dim=0):
        """Initialize ReverseSequence"""
        self.init_prim_io_names(inputs=['x', 'seq_lengths'], outputs=['y'])
        validator.check_value_type("seq_dim", seq_dim, [int], self.name)
        self.seq_dim_ = seq_dim
        validator.check_value_type("batch_dim", batch_dim, [int], self.name)
        self.batch_dim_ = batch_dim


class EditDistance(Primitive):
    r"""
    Computes the Levenshtein Edit Distance. It is used to measure the similarity of two sequences. The inputs are
    variable-length sequences provided by SparseTensors (hypothesis_indices, hypothesis_values, hypothesis_shape)
    and (truth_indices, truth_values, truth_shape).

    .. math::

        \operatorname{lev}_{a, b}(i, j)=\left\{\begin{array}{ll}
        \max (i, j)  \qquad \qquad \qquad \qquad \qquad \quad \  \text { if } \min (i, j)=0 \\
        \min \left\{\begin{array}{ll}
        \operatorname{lev}_{a, b}(i-1, j)+1 & \\
        \operatorname{lev}_{a, b}(i, j-1)+1 & \text { otherwise. } \\
        \operatorname{lev}_{a, b}(i-1, j-1)+1_{\left(a_{i} \neq b_{j}\right)}
        \end{array}\right. &
        \end{array}\right.

    Where the :math:`a` indicates the hypothesis and the :math:`b` indicates the truth. For ease of understanding,
    i and j here in may be considered as lengths of a and b.

    .. warning::
        Unorded `truth_indices` or `hypothesis_indices` might lead to expected result, so it is suggested to
        make sure `truth_indices` and `hypothesis_indices` are both in ascending order before
        calling this API.

    Args:
        normalize (bool): If ``True`` , edit distances are normalized by length of truth. Default: ``True`` .

    Inputs:
        - **hypothesis_indices** (Tensor) - The indices of the hypothesis list SparseTensor. With int64 data type.
          The shape of tensor is :math:`(N, R)`.
        - **hypothesis_values** (Tensor) - The values of the hypothesis list SparseTensor.
          Must be 1-D vector with length of N.
        - **hypothesis_shape** (Tensor) - The shape of the hypothesis list SparseTensor.
          Must be R-length vector with int64 data type. Only constant value is allowed.
        - **truth_indices** (Tensor) - The indices of the truth list SparseTensor. With int64 data type.
          The shape of tensor is :math:`(M, R)`.
        - **truth_values** (Tensor) - The values of the truth list SparseTensor. Must be 1-D vector with length of M.
        - **truth_shape** (Tensor) - The shape of the truth list SparseTensor.
          Must be R-length vector with int64 data type. Only constant value is allowed.

    Outputs:
        Tensor, a dense tensor with rank `R-1` and float32 data type.

    Raises:
        TypeError: If `normalize` is not a bool.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.nn as nn
        >>> from mindspore import ops
        >>> class EditDistance(nn.Cell):
        ...     def __init__(self, hypothesis_shape, truth_shape, normalize=True):
        ...         super(EditDistance, self).__init__()
        ...         self.edit_distance = ops.EditDistance(normalize)
        ...         self.hypothesis_shape = hypothesis_shape
        ...         self.truth_shape = truth_shape
        ...
        ...     def construct(self, hypothesis_indices, hypothesis_values, truth_indices, truth_values):
        ...         return self.edit_distance(hypothesis_indices, hypothesis_values, self.hypothesis_shape,
        ...                                   truth_indices, truth_values, self.truth_shape)
        ...
        >>> hypothesis_indices = Tensor(np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]]).astype(np.int64))
        >>> hypothesis_values = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> hypothesis_shape = Tensor(np.array([1, 1, 2]).astype(np.int64))
        >>> truth_indices = Tensor(np.array([[0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]).astype(np.int64))
        >>> truth_values = Tensor(np.array([1, 3, 2, 1]).astype(np.float32))
        >>> truth_shape = Tensor(np.array([2, 2, 2]).astype(np.int64))
        >>> edit_distance = EditDistance(hypothesis_shape, truth_shape)
        >>> output = edit_distance(hypothesis_indices, hypothesis_values, truth_indices, truth_values)
        >>> print(output)
        [[1. 1.]
         [1. 1.]]
    """

    @prim_attr_register
    def __init__(self, normalize=True):
        """Initialize EditDistance"""
        self.normalize = validator.check_value_type("normalize", normalize, [bool], self.name)


class TransShape(PrimitiveWithInfer):
    """
    Transforms the shape of input tensor to target shape.

    Inputs:
        - **input_x** (Tensor) - A input tensor.
        - **out_shape** (tuple[int]) - The shape of output data.

    Outputs:
        Tensor, a tensor whose data type is same as 'input_x', and the shape is the same as the `out_shape`.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize TransShape."""
        self.__setattr_flag__ = True

    def __infer__(self, x, shape):
        shp = shape['value']
        dtype = x['dtype']
        validator.check_tensor_dtype_valid('x', dtype, mstype.number_type + (mstype.bool_,), self.name)
        self.add_prim_attr('out_shape', tuple(shp))
        return {'shape': shp,
                'dtype': dtype,
                'value': None}


class Sort(Primitive):
    """
    Sorts the elements of the input tensor along the given dimension in the specified order.

    .. warning::
        Currently, the data types of float16, uint8, int8, int16, int32, int64 are well supported.
        If use float32, it may cause loss of accuracy.

    Args:
        axis (int, optional): The dimension to sort along. Default: ``-1``, means the last dimension.
            The Ascend backend only supports sorting the last dimension.
        descending (bool, optional): Controls the sort order. If descending is ``True`` then the elements
            are sorted in descending order by value. Default: ``False`` .

    Inputs:
        - **x** (Tensor) - The input tensor.

    Outputs:
        - **y1** (Tensor) - A tensor whose values are the sorted values, with the same shape and data type as input.
        - **y2** (Tensor) - the indices of the elements in the original input tensor. Data type is int32.

    Raises:
        TypeError: If `axis` is not an int.
        TypeError: If `descending` is not a bool.
        ValueError: If `axis` is not in range of [-len(x.shape), len(x.shape)).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]), mindspore.float16)
        >>> sort = ops.Sort()
        >>> output = sort(x)
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

    @prim_attr_register
    def __init__(self, axis=-1, descending=False):
        """Initialize Sort"""
        self.axis = validator.check_value_type("axis", axis, [int], self.name)
        self.descending = validator.check_value_type("descending", descending, [bool], self.name)
        self.init_prim_io_names(inputs=['x'], outputs=['y1', 'y2'])


class EmbeddingLookup(Primitive):
    """
    Returns a slice of input tensor based on the specified indices.

    This Primitive has the similar functionality as GatherV2 operating on `axis = 0`, but has one more inputs:
    `offset`.

    Inputs:
        - **input_params** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
          This represents a Tensor slice, instead of the entire Tensor. Currently, the dimension is restricted to be 2.
        - **input_indices** (Tensor) - The shape of tensor is :math:`(y_1, y_2, ..., y_S)`.
          Specifies the indices of elements of the original Tensor. Values can be out of range of `input_params`,
          and the exceeding part will be filled with 0 in the output. Values do not support negative and the result
          is undefined if values are negative. The data type should be int32 or int64.
        - **offset** (int) - Specifies the offset value of this `input_params` slice. Thus the real indices
          are equal to `input_indices` minus `offset`.

    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`. The data type is the same with `input_params`.

    Raises:
        TypeError: If dtype of `input_indices` is not int.
        ValueError: If length of shape of `input_params` is greater than 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_params = Tensor(np.array([[8, 9], [10, 11], [12, 13], [14, 15]]), mindspore.float32)
        >>> input_indices = Tensor(np.array([[5, 2], [8, 5]]), mindspore.int32)
        >>> offset = 4
        >>> output = ops.EmbeddingLookup()(input_params, input_indices, offset)
        >>> print(output)
        [[[10. 11.]
          [ 0.  0.]]
         [[ 0.  0.]
          [10. 11.]]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize EmbeddingLookup."""
        self.__setattr_flag__ = True
        self.init_prim_io_names(inputs=['params', 'indices', 'offset'],
                                outputs=['output'])
        self.add_prim_attr('bprop_return_sparse', True)


class IdentityN(Primitive):
    """
    Return a tuple of tensors with the same shapes and contents as the input.

    This op can be used to override the gradient for complicated functions. For
    example, suppose :math:`y = f(x)` and we wish to apply a custom function g for backprop
    such that :math:`dx=g(dy)`.

    Inputs:
        - **x** (Union[tuple[Tensor], list[Tensor]]) - Input, the data type is RealNumber.

    Outputs:
        Tensors - tuple(Tensor), the shape of tensor and the data type are the same as input `x`.

    Raises:
        TypeError: If `x` is not tuple(Tensor) or List(Tensor).
        TypeError: If input `x` type is not RealNumber.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = [Tensor(np.array([1, 2, 3, 4]), mstype.int64), Tensor(np.array([4, 3, 1, 1]), mstype.int64)]
        >>> output = ops.IdentityN()(x)
        >>> print(np.allclose(output[0].asnumpy(), x[0].asnumpy()))
        True
        >>> print(np.allclose(output[1].asnumpy(), x[1].asnumpy()))
        True
        >>> print(output)
        (Tensor(shape=[4], dtype=Int64, value= [1, 2, 3, 4]), Tensor(shape=[4], dtype=Int64, value= [4, 3, 1, 1]))
    """

    @prim_attr_register
    def __init__(self):
        """Initialize IdentityN"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])


class RangeV2(Primitive):
    """
    Creates a sequence of numbers that begins at `start`, ends at `limit` but not including `limit`
    and extends by increments of `delta`.

    The types of all 3 inputs must be the same. The type of the resulting tensor is
    the same as the type of the inputs.

    Args:
        maxlen (int): Memory that can fit `maxlen` many elements
            will be allocated for the output. Optional, must be positive, defaults to 1000000.
            If the output has more than `maxlen` elements, a `ValueError` will occur.

    Inputs:
        - **start** (Tensor) - A scalar Tensor. The first number in the sequence. Must have
          type: int32 or float32 or int64 or float64
        - **limit** (Tensor) - A scalar Tensor. Upper limit of the sequence, exclusive. Must
          have type: int32 or float32 or int64 or float64
        - **delta** (Tensor) - A scalar Tensor. Number that increments `start`. Must have
          type: int32 or float32 or int64 or float64

    Outputs:
       A 1D Tensor, with the same type as the inputs.

    Raises:
        TypeError: If datatype of `start`, `limit` and `delta` not supported.
        TypeError: If datatype of `start`, `limit` and `delta` not same.
        TypeError: If attr `max_len` is not int.
        TypeError: If `start` or `limit` or `delta` is not scalar Tensor.
        ValueError: If value of `max_len` is negative.
        ValueError: If `delta` >= 0 when `start` > `limit`.
        ValueError: If `delta` <= 0 when `start` < `limit`.
        ValueError: If the output has more than `maxlen` elements

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> start = Tensor(0, mstype.int32)
        >>> limit = Tensor(10, mstype.int32)
        >>> delta = Tensor(4, mstype.int32)
        >>> output = ops.RangeV2()(start, limit, delta)
        >>> print(output)
        [0 4 8]
    """

    @prim_attr_register
    def __init__(self, maxlen=1000000):
        """Initialize RangeV2"""
        self.init_prim_io_names(inputs=['start', 'limit', 'delta'], outputs=['output'])
        validator.check_value_type("maxlen", maxlen, [int], self.name)
        validator.check_positive_int(maxlen, "maxlen", self.name)


class MaskedScatter(Primitive):
    """
    Updates the value in the input with value in `updates` according to the `mask`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **x** (Tensor): The input Tensor to be updated.
        - **mask** (Tensor[bool]): The mask Tensor indicating which elements should be modified or replaced.
          The shapes of `mask` and `x` must be the same or broadcastable.
        - **updates** (Tensor): The values to scatter into the target tensor `x`. It has the same data type as `x`. The
          number of elements must be greater than or equal to the number of True's in `mask`.

    Outputs:
        Tensor, with the same type and shape as `x`.

    Raises:
        TypeError: If `x`, `mask` or `updates` is not a Tensor.
        TypeError: If data type of `x` is not be supported.
        TypeError: If dtype of `mask` is not bool.
        TypeError: If the dim of `x` less than the dim of `mask`.
        ValueError: If `mask` can not be broadcastable to `x`.
        ValueError: If the number of elements in `updates` is less than number of True's in `mask`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.array([1., 2., 3., 4.]), mindspore.float32)
        >>> mask = Tensor(np.array([True, True, False, True]), mindspore.bool_)
        >>> updates = Tensor(np.array([5., 6., 7.]), mindspore.float32)
        >>> output = ops.MaskedScatter()(input_x, mask, updates)
        >>> print(output)
        [5. 6. 3. 7.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize MaskedScatter"""
        self.init_prim_io_names(inputs=['x', 'mask', 'updates'], outputs=['y'])


class MaskedSelect(PrimitiveWithCheck):
    """
    Returns a new 1-D Tensor which indexes the `x` tensor according to the boolean `mask`.
    The shapes of the `mask` tensor and the `x` tensor don't need to match, but they must be broadcastable.

    Inputs:
        - **x** (Tensor) - Input Tensor of any dimension.
        - **mask** (Tensor[bool]) - Boolean mask Tensor, has the same shape as `x`.

    Outputs:
        A 1-D Tensor, with the same type as x.

    Raises:
        TypeError: If `x` or `mask` is not a Tensor.
        TypeError: If dtype of `mask` is not bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([1, 2, 3, 4]), mindspore.int32)
        >>> mask = Tensor(np.array([1, 0, 1, 0]), mindspore.bool_)
        >>> output = ops.MaskedSelect()(x, mask)
        >>> print(output)
        [1 3]
        >>> x = Tensor(2.1, mindspore.float32)
        >>> mask = Tensor(True, mindspore.bool_)
        >>> output = ops.MaskedSelect()(x, mask)
        >>> print(output)
        [2.1]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x', 'mask'], outputs=['output'])

    def check_shape(self, x_shape, mask_shape):
        get_broadcast_shape(x_shape, mask_shape, self.name, arg_name1="x", arg_name2="mask")

    def check_dtype(self, x_dtype, mask_dtype):
        validator.check_tensor_dtype_valid('mask', mask_dtype, [mstype.bool_], self.name)
        validator.check_tensor_dtype_valid('x', x_dtype, (mstype.bool_,) + mstype.number_type, self.name)


class _TensorScatterOp(PrimitiveWithInfer):
    """
    Defines TensorScatter Base Operators
    """

    def infer_shape(self, input_x_shape, indices_shape, updates_shape):
        if indices_shape != [-2] and len(indices_shape) < 2:
            raise ValueError(f"For '{self.name}', the dimension of 'indices' cannot be less than 2,"
                             f" but got {len(indices_shape)}.")
        if indices_shape[-1] > 0:
            if indices_shape[-1] > len(input_x_shape):
                raise ValueError(f"For '{self.name}', the last dimension of 'indices' must be less than or equal to "
                                 f"the dimension of 'input_x', but got the "
                                 f"last dimension of 'indices': {indices_shape[-1]} and the dimension of 'input_x': "
                                 f"{len(input_x_shape)}.")
            updates_shape_check = indices_shape[:-1] + input_x_shape[indices_shape[-1]:]
            if self._check_shape(updates_shape_check, updates_shape) is False:
                raise ValueError(f"For '{self.name}', the shape of 'update' must be equal to updates_shape_check, "
                                 f"where updates_shape_check = indices_shape[:-1] + input_x_shape[indices_shape[-1]:] "
                                 f"but got the shape of 'update': {updates_shape}, "
                                 f"updates_shape_check: {updates_shape_check}, indices_shape: {indices_shape} and "
                                 f"input_x_shape: {input_x_shape}. Please check input_x_shape and indices_shape.")

        return input_x_shape

    def infer_dtype(self, input_x_dtype, indices_dtype, updates_dtype):
        validator.check_tensor_dtype_valid('indices', indices_dtype, [mstype.int32, mstype.int64], self.name)
        args = {"input_x": input_x_dtype, "updates": updates_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        return input_x_dtype

    def _check_shape(self, expect, real):
        """check shape"""
        if -2 in expect or -2 in real:
            return True
        if len(expect) != len(real):
            return False
        for a, b in zip(expect, real):
            if a == -1 or b == -1:
                continue
            if a != b:
                return False
        return True


class TensorScatterUpdate(_TensorScatterOp):
    r"""
    Creates a new tensor by updating the positions in `input_x` indicated by
    `indices`, with values from `update`. This operation is almost equivalent to using
    `mindspore.ops.ScatterNdUpdate` , except that the updates are applied on `input_x` instead of a zero tensor.

    `indices` must have rank at least 2, the last axis is the depth of each index
    vectors. For each index vector, there must be a corresponding value in `update`. If
    the depth of each index tensor matches the rank of `input_x`, then each index
    vector corresponds to a scalar in `input_x` and each `update` updates a scalar. If
    the depth of each index tensor is less than the rank of `input_x`, then each index
    vector corresponds to a slice in `input_x`, and each `update` updates a slice.

    The order in which updates are applied is nondeterministic, meaning that if there
    are multiple index vectors in `indices` that correspond to the same position, the
    value of that position in the output will be nondeterministic.

    .. math::
        output[indices] = update

    Inputs:
        - **input_x** (Tensor) - The input tensor. The dimension of input_x must be no less than indices.shape[-1].
          The shape is :math:`(N, *)` where :math:`*` means,any number of additional dimensions.
          The data type is Number.
        - **indices** (Tensor) - The index of input tensor whose data type is int32 or int64.
          The rank must be at least 2.
        - **update** (Tensor) - The tensor to update the input tensor, has the same type as `input_x`, and
          :math:`update.shape = indices.shape[:-1]+input\_x.shape[indices.shape[-1]:]`

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If dtype of `indices` is neither int32 nor int64.
        ValueError: If length of shape of `input_x` is less than the last dimension of shape of `indices`.
        ValueError: If the value of `input_x` are not match with input `indices`.
        RuntimeError: If a value of `indices` is not in `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> update = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> op = ops.TensorScatterUpdate()
        >>> output = op(input_x, indices, update)
        >>> print(output)
        [[ 1.   0.3  3.6]
         [ 0.4  2.2 -3.2]]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['input_x', 'indices', 'updates'], outputs=['y'])

    def infer_dtype(self, input_x_dtype, indices_dtype, updates_dtype):
        validator.check_tensor_dtype_valid('indices', indices_dtype, [mstype.int32, mstype.int64], self.name)
        args = {"input_x": input_x_dtype, "updates": updates_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.bool_,) + mstype.number_type, self.name)
        return input_x_dtype


class TensorScatterMax(Primitive):
    r"""
    By comparing the value at the position indicated by `indices` in `x` with the value in the `updates`,
    the value at the index will eventually be equal to the largest one to create a new tensor.

    Refer to :func:`mindspore.ops.tensor_scatter_max` for more details.

    Inputs:
        - **input_x** (Tensor) - The target tensor. The dimension of input_x must be no less than indices.shape[-1].
        - **indices** (Tensor) - The index of input tensor whose data type is int32 or int64.
          The rank must be at least 2.
        - **updates** (Tensor) - The tensor to update the input tensor, has the same type as input,
          and updates.shape should be equal to :math:`indices.shape[:-1] + input\_x.shape[indices.shape[-1]:]`.

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [0, 0]]), mindspore.int32)
        >>> updates = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> # Next, demonstrate the approximate operation process of this operator:
        >>> # 1, indices[0] = [0, 0], indices[1] = [0, 0]
        >>> # 2, And input_x[0, 0] = -0.1
        >>> # 3, So input_x[indices] = [-0.1, -0.1]
        >>> # 4, Satisfy the above formula: input_x[indices].shape=(2) == updates.shape=(2)
        >>> op = ops.TensorScatterMax()
        >>> # 5, Perform the max operation for the first time:
        >>> #      first_input_x = Max(input_x[0][0], updates[0]) = [[1.0, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> # 6, Perform the max operation for the second time:
        >>> #      second_input_x = Max(input_x[0][0], updates[1]) = [[2.2, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> output = op(input_x, indices, updates)
        >>> print(output)
        [[ 2.2  0.3  3.6]
         [ 0.4  0.5 -3.2]]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['input_x', 'indices', 'updates'], outputs=['y'])


class TensorScatterMin(Primitive):
    r"""
    By comparing the value at the position indicated by `indices` in `input_x` with the value in the `updates`,
    the value at the index will eventually be equal to the smallest one to create a new tensor.

    Refer to :func:`mindspore.ops.tensor_scatter_min` for more details.

    Inputs:
        - **input_x** (Tensor) - The target tensor. The dimension of input_x must be no less than indices.shape[-1].
        - **indices** (Tensor) - The index of input tensor whose data type is int32 or int64.
          The rank must be at least 2.
        - **updates** (Tensor) - The tensor to update the input tensor, has the same type as input,
          and updates.shape should be equal to :math:`indices.shape[:-1] + input\_x.shape[indices.shape[-1]:]`.

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [0, 0]]), mindspore.int32)
        >>> updates = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> # Next, demonstrate the approximate operation process of this operator:
        >>> # 1, indices[0] = [0, 0], indices[1] = [0, 0]
        >>> # 2, And input_x[0, 0] = -0.1
        >>> # 3, So input_x[indices] = [-0.1, -0.1]
        >>> # 4, Satisfy the above formula: input_x[indices].shape=(2) == updates.shape=(2)
        >>> op = ops.TensorScatterMin()
        >>> # 5, Perform the min operation for the first time:
        >>> #      first_input_x = Min(input_x[0][0], updates[0]) = [[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> # 6, Perform the min operation for the second time:
        >>> #      second_input_x = Min(input_x[0][0], updates[1]) = [[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> output = op(input_x, indices, updates)
        >>> print(output)
        [[ -0.1  0.3  3.6]
         [ 0.4  0.5 -3.2]]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['input_x', 'indices', 'updates'], outputs=['y'])


class TensorScatterSub(Primitive):
    r"""
    Creates a new tensor by subtracting the values from the positions in `input_x` indicated by
    `indices`, with values from `updates`. When multiple values are provided for the same
    index, the result of the update will be to subtract these values respectively. This operation is almost
    equivalent to using :class:`mindspore.ops.ScatterNdSub` , except that the updates are applied on output `Tensor`
    instead of input `Parameter`.

    .. math::
        output\left [indices  \right ] = input\_x- update

    Refer to :func:`mindspore.ops.tensor_scatter_sub` for more details.

    Inputs:
        - **input_x** (Tensor) - The target tensor. The dimension of input_x must be no less than indices.shape[-1].
        - **indices** (Tensor) - The index of input tensor whose data type is int32 or int64.
          The rank must be at least 2.
        - **updates** (Tensor) - The tensor to update the input tensor, has the same type as `input_x`,
          and the shape of `updates` should be equal to :math:`indices.shape[:-1] + input\_x.shape[indices.shape[-1]:]`.

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [0, 0]]), mindspore.int32)
        >>> updates = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> # Next, demonstrate the approximate operation process of this operator:
        >>> # 1, indices[0] = [0, 0], indices[1] = [0, 0]
        >>> # 2, And input_x[0, 0] = -0.1
        >>> # 3, So input_x[indices] = [-0.1, -0.1]
        >>> # 4, Satisfy the above formula: input_x[indices].shape=(2) == updates.shape=(2)
        >>> op = ops.TensorScatterSub()
        >>> # 5, Perform the subtract operation for the first time:
        >>> #      first_input_x = input_x[0][0] - updates[0] = [[-1.1, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> # 6, Perform the subtract operation for the second time:
        >>> #      second_input_x = input_x[0][0] - updates[1] = [[-3.3, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> output = op(input_x, indices, updates)
        >>> print(output)
        [[-3.3000002  0.3        3.6      ]
         [ 0.4        0.5       -3.2      ]]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['input_x', 'indices', 'updates'], outputs=['y'])


class TensorScatterAdd(Primitive):
    """
    Creates a new tensor by adding the values from the positions in `input_x` indicated by
    `indices`, with values from `updates`. When multiple values are given for the same
    index, the updated result will be the sum of all values. This operation is almost
    equivalent to using :class:`mindspore.ops.ScatterNdAdd`, except that the updates are applied on output `Tensor`
    instead of input `Parameter`.

    Refer to :func:`mindspore.ops.tensor_scatter_add` for more details.

    Inputs:
        - **input_x** (Tensor) - The target tensor. The dimension of input_x must be no less than indices.shape[-1].
        - **indices** (Tensor) - The index of input tensor whose data type is int32 or int64.
          The rank must be at least 2.
        - **updates** (Tensor) - The tensor to update the input tensor, has the same type as input,
          and updates. Shape should be equal to indices.shape[:-1] + input_x.shape[indices.shape[-1]:].

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [0, 0]]), mindspore.int32)
        >>> updates = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> # Next, demonstrate the approximate operation process of this operator:
        >>> # 1, indices[0] = [0, 0], indices[1] = [0, 0]
        >>> # 2, And input_x[0, 0] = -0.1
        >>> # 3, So input_x[indices] = [-0.1, -0.1]
        >>> # 4, Satisfy the above formula: input_x[indices].shape=(2) == updates.shape=(2)
        >>> op = ops.TensorScatterAdd()
        >>> # 5, Perform the addition operation for the first time:
        >>> #      first_input_x = input_x[0][0] + updates[0] = [[0.9, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> # 6, Perform the addition operation for the second time:
        >>> #      second_input_x = input_x[0][0] + updates[1] = [[3.1, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> output = op(input_x, indices, updates)
        >>> print(output)
        [[ 3.1  0.3  3.6]
         [ 0.4  0.5 -3.2]]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['input_x', 'indices', 'updates'], outputs=['y'])


class TensorScatterMul(_TensorScatterOp):
    r"""
    Creates a new tensor by multiplying the values from the positions in `input_x` indicated by
    `indices`, with values from `updates`. When multiple values are provided for the same
    index, the result of the update will be to multiply these values respectively.
    The updates are applied on output `Tensor` instead of input `Parameter`.

    .. math::
        output\left [indices  \right ] = input\_x\times  update

    Refer to :func:`mindspore.ops.tensor_scatter_mul` for more details.

    Inputs:
        - **input_x** (Tensor) - The target tensor. The dimension of input_x must be no less than indices.shape[-1].
        - **indices** (Tensor) - The index of input tensor whose data type is int32 or int64.
          The rank must be at least 2.
        - **updates** (Tensor) - The tensor to update the input tensor, has the same type as `input_x`,
          and the shape of `updates` should be equal to :math:`indices.shape[:-1] + input\_x.shape[indices.shape[-1]:]`.

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [0, 0]]), mindspore.int32)
        >>> updates = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> # Next, demonstrate the approximate operation process of this operator:
        >>> # 1, indices[0] = [0, 0], indices[1] = [0, 0]
        >>> # 2, And input_x[0, 0] = -0.1
        >>> # 3, So input_x[indices] = [-0.1, -0.1]
        >>> # 4, Satisfy the above formula: input_x[indices].shape=(2) == updates.shape=(2)
        >>> op = ops.TensorScatterMul()
        >>> # 5, Perform the multiply operation for the first time:
        >>> #      first_input_x = input_x[0][0] * updates[0] = [[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> # 6, Perform the multiply operation for the second time:
        >>> #      second_input_x = input_x[0][0] * updates[1] = [[-0.22, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> output = op(input_x, indices, updates)
        >>> print(output)
        [[-0.22  0.3   3.6  ]
         [ 0.4   0.5   -3.2 ]]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['input_x', 'indices', 'updates'], outputs=['y'])


class TensorScatterDiv(_TensorScatterOp):
    r"""
    Creates a new tensor by dividing the values from the positions in `input_x` indicated by
    `indices`, with values from `updates`. When divided values are provided for the same
    index, the result of the update will be to divided these values respectively. Except that
    the updates are applied on output `Tensor` instead of input `Parameter`.

    Refer to :func:`mindspore.ops.tensor_scatter_div` for more details.

    Inputs:
        - **input_x** (Tensor) - The target tensor. The dimension of input_x must be no less than indices.shape[-1].
        - **indices** (Tensor) - The index of input tensor whose data type is int32 or int64.
          The rank must be at least 2.
        - **updates** (Tensor) - The tensor to update the input tensor, has the same type as input,
          and updates.shape should be equal to :math:`indices.shape[:-1] + input\_x.shape[indices.shape[-1]:]`.

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [0, 0]]), mindspore.int32)
        >>> updates = Tensor(np.array([1.0, 2.0]), mindspore.float32)
        >>> # Next, demonstrate the approximate operation process of this operator:
        >>> # 1, indices[0] = [0, 0], indices[1] = [0, 0]
        >>> # 2, And input_x[0, 0] = -0.1
        >>> # 3, So input_x[indices] = [-0.1, -0.1]
        >>> # 4, Satisfy the above formula: input_x[indices].shape=(2) == updates.shape=(2)
        >>> op = ops.TensorScatterDiv()
        >>> # 5, Perform the division operation for the first time:
        >>> #      first_input_x = input_x[0][0] / updates[0] = [[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> # 6, Perform the division operation for the second time:
        >>> #      second_input_x = input_x[0][0] * updates[1] = [[-0.05, 0.3, 3.6], [0.4, 0.5, -3.2]]
        >>> output = op(input_x, indices, updates)
        >>> print(output)
        [[-0.05  0.3  3.6  ]
         [ 0.4   0.5  -3.2 ]]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['input_x', 'indices', 'updates'], outputs=['y'])


class ListDiff(Primitive):
    r"""
    This function calculates the disparity between two numerical lists.

    It generates a list of all elements that are present in list `x` but not in list `y`.
    The output list `out` retains the same order as the original `x` including duplicate elements.

    Additionally, this class outputs a list `idx` that identifies the position of each element
    in `out` within the original `x`. That is to say:
    :code:`out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]` .

    Args:
        out_idx (:class:`mindspore.dtype`, optional): The dtype of `idx`,
            an optioanal datatype of ``mstype.int32`` and ``mstype.int64`` .
            Default: ``mstype.int32`` .

    Inputs:
        - **x** - Values to keep. A 1-D `Tensor`.
        - **y** - Values to remove. A 1-D `Tensor`. Must have the same type as `x`. 1-D.

    Outputs:
        - **out** - The kept values. A 1-D `Tensor`. Has the same type as `x`.
        - **idx** - The original index of kept values. A 1-D `Tensor` of type `out_idx`.

    Raises:
        ValueError: If `x` or `y` shape is not 1D.
        TypeError: If `x` or `y` is not a Tensor.
        TypeError: If `x` or `y` date type is not int or uint.
        TypeError: If `x` has different data type with `y`.
        TypeError: If attr `out_idx` not in [mstype.int32, mstype.int64].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.arange(1, 7, 1), dtype=mindspore.dtype.int32) # [1, 2, 3, 4, 5, 6]
        >>> y = Tensor([1, 3, 5], dtype=mindspore.dtype.int32)
        >>> op = ops.ListDiff() # out_idx default is mindspore.dtype.int32
        >>> out, idx = op(x, y)
        >>> print(out)
        [2 4 6]
        >>> print(idx)
        [1 3 5]
    """

    @prim_attr_register
    def __init__(self, out_idx=mstype.int32):
        """Initialize ListDiff"""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['out', 'idx'])
        validator.check_value_type("out_idx", out_idx, [mstype.Type], self.name)
        validator.check("out_idx", out_idx, "", [mstype.int32, mstype.int64], validator.IN,
                        self.name, excp_cls=TypeError)
        self.out_idx = out_idx
        self.add_prim_attr('out_idx', out_idx)


class SplitV(Primitive):
    r"""
    Splits the input tensor into `num_split` tensors along the given dimension.

    The `input_x` tensor will be split into sub-tensors with individual shapes given
    by `size_splits` along the split dimension. This requires that `input_x.shape(split_dim)`
    is equal to the sum of `size_splits`.

    The shape of `input_x` is :math:`(x_1, x_2, ..., x_M, ..., x_R)` whose rank
    is `R`. Set the given `split_dim` as M, and :math:`-R \le M < R`. Set the given `num_split`
    as `N`, the given `size_splits` as :math:`(x_{m_1}, x_{m_2}, ..., x_{m_N})`,
    :math:`x_M=\sum_{i=1}^Nx_{m_i}`. The output is a list of tensor objects, for the
    :math:`i`-th tensor, it has the shape of :math:`(x_1, x_2, ..., x_{m_i}, ..., x_R)`.
    :math:`x_{m_i}` is the :math:`M`-th dimension of the :math:`i`-th tensor.
    Then, the shape of the output tensor is

    .. math::

        ((x_1, x_2, ..., x_{m_1}, ..., x_R), (x_1, x_2, ..., x_{m_2}, ..., x_R), ...,
         (x_1, x_2, ..., x_{m_N}, ..., x_R))

    Args:
        size_splits (Union[tuple, list]): A tuple or list of sizes of each output tensor along the split
            dimension, and the sum of these sizes should equal to the dimension of the
            input tensor along `split_dim`. The list may also contain a single instance of
            the value -1, which indicates that the size of that dimension should be inferred.
        split_dim (int): An int indicates the dimension along which to split.
            Must be in the range [-len(input_x.shape), len(input_x.shape)).
        num_split (int): The number of output tensors. Must be positive int.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ...,x_M ..., x_R)`.

    Outputs:
        Tensor, a list of `num_split` Tensor objects with the shape :math:`((x_1, x_2, ..., x_{m_1}, ..., x_R),
        (x_1, x_2, ..., x_{m_2}, ..., x_R), ..., (x_1, x_2, ..., x_{m_N}, ..., x_R))`, :math:`x_M=\sum_{i=1}^Nx_{m_i}`.
        The data type is the same with `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        TypeError: If `size_splits` is not a tuple or a list.
        TypeError: If element of `size_splits` is not an int.
        TypeError: If `split_dim` or `num_split` is not an int.
        ValueError: If rank of the `size_splits` is not equal to `num_split`.
        ValueError: If sum of the `size_splits` is not equal to the dimension of value along `split_dim`.
        ValueError: If `split_dim` is out of the range [-len(input_x.shape), len(input_x.shape)).
        ValueError: If the `num_split` is less than or equal to 0.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.int32)
        >>> op = ops.SplitV(size_splits=[1, -1], split_dim=1, num_split=2)
        >>> output = op(input_x)
        >>> print(output)
        (Tensor(shape=[3, 1], dtype=Int32, value=
        [[1],
         [4],
         [7]]), Tensor(shape=[3, 2], dtype=Int32, value=
        [[2, 3],
         [5, 6],
         [8, 9]]))
        >>> input_x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.int32)
        >>> op = ops.SplitV(size_splits=[2, 1], split_dim=0, num_split=2)
        >>> output = op(input_x)
        >>> print(output)
        (Tensor(shape=[2, 3], dtype=Int32, value=
        [[1, 2, 3],
         [4, 5, 6]]), Tensor(shape=[1, 3], dtype=Int32, value=
        [[7, 8, 9]]))
    """

    @prim_attr_register
    def __init__(self, size_splits, split_dim, num_split):
        """Initialize SplitV"""
        validator.check_value_type("size_splits", size_splits, [tuple, list], self.name)
        for elements_of_size_splits in size_splits:
            validator.check_value_type("elements of size_splits", elements_of_size_splits, [int], self.name)
            if elements_of_size_splits != -1 and elements_of_size_splits < 1:
                raise ValueError(f"For \'{self.name}\', all elements of size_splits must be positive (except at most "
                                 f"one default value -1), but got: {elements_of_size_splits}.")
        validator.check_value_type("split_dim", split_dim, [int], self.name)
        validator.check_value_type("num_split", num_split, [int], self.name)
        validator.check_positive_int(num_split, "num_split", self.name)
        self.init_prim_io_names(inputs=['input_x'], outputs=['output'])


class TensorScatterElements(Primitive):
    """
    Write all elements in `updates` to the index specified by `indices` in `input_x` according to the reduction
    operation specified by `reduction`.
    `axis` controls the direction of the scatter operation.

    Refer to :func:`mindspore.ops.tensor_scatter_elements` for more details.

    .. warning::
        If there are multiple index vectors in `indices` that correspond to the same position,
        the value of that position in the output will be nondeterministic.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        axis (int, optional): Specify which axis to do scatter operation. Default: ``0`` .
        reduction (str, optional): Which reduction operation to scatter, default is ``"none"`` . Other option: "add".

    Inputs:
        - **data** (Tensor) - The target tensor. Its rank must be at least 1.
        - **indices** (Tensor) - The index of `input_x` to do scatter operation whose data type must be int32 or
          int64. It has the same rank as `data`. And accepted range is [-s, s) where s is the size along axis.
        - **updates** (Tensor) - The tensor doing the scatter operation with `data`,
          it has the same type as `data` and the same shape as `indices`.

    Outputs:
        Tensor, has the same shape and type as `data`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> op = ops.TensorScatterElements(0, "none")
        >>> data = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> indices = Tensor(np.array([[1, 0, 2], [0, 2, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[0, 0, 0], [0, 0, 0]]), mindspore.float32)
        >>> output = op(data, indices, updates)
        >>> print(output)
        [[ 0.0  0.0  3.0]
         [ 0.0  5.0  0.0]
         [ 7.0  0.0  0.0]]
        >>> import mindspore as ms
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> op = ops.TensorScatterElements(1, "add")
        >>> data = Tensor(np.array([[1, 2, 3, 4, 5]]), mindspore.float32)
        >>> indices = Tensor(np.array([[2, 4]]), mindspore.int32)
        >>> updates = Tensor(np.array([[8, 8]]), mindspore.float32)
        >>> output = op(data, indices, updates)
        >>> print(output)
        [[ 1  2  11  4  13]]
    """

    @prim_attr_register
    def __init__(self, axis=0, reduction="none"):
        """Initialize TensorScatterElements"""
        validator.check_value_type("axis", axis, [int], self.name)
        validator.check_value_type("reduction", reduction, [str], self.name)
        validator.check_string(reduction, ["none", "add"], "reduction", self.name)
        self.init_prim_io_names(inputs=['data', 'indices', 'updates'], outputs=['y'])
        target = context.get_context("device_target")
        if reduction != 'none' and target.lower() == "ascend":
            raise ValueError(f"For '{self.name}', "
                             f"Currently Ascend device_target only support `reduction`='none', "
                             f"but got {reduction}")


class ExtractVolumePatches(Primitive):
    """
    `ops.ExtractVolumePatches` is deprecated from version 2.3 and will be removed in a future version.

    Supported Platforms:
        Deprecated
    """
    @deprecated("2.3", "ops.ExtractVolumePatches", False)
    @prim_attr_register
    def __init__(self, kernel_size, strides, padding):
        validator.check_value_type("kernel_size", kernel_size, (int, list, tuple), self.name)
        validator.check_value_type("strides", strides, (int, list, tuple), self.name)
        if isinstance(kernel_size, (list, tuple)):
            kernel_size = tuple(kernel_size)
            if len(kernel_size) == 5:
                validator.check_int(kernel_size[0], 1, validator.EQ, "kernel_size[0]", self.name)
                validator.check_int(kernel_size[1], 1, validator.EQ, "kernel_size[1]", self.name)
        if isinstance(strides, (list, tuple)):
            strides = tuple(strides)
            if len(strides) == 5:
                validator.check_int(strides[0], 1, validator.EQ, "strides[0]", self.name)
                validator.check_int(strides[1], 1, validator.EQ, "strides[1]", self.name)
        self.kernel_size = _check_3d_int_or_tuple("kernel_size", kernel_size, self.name,
                                                  allow_five=True, ret_five=True, greater_zero=True)
        self.strides = _check_3d_int_or_tuple("strides", strides, self.name,
                                              allow_five=True, ret_five=True, greater_zero=True)
        self.add_prim_attr("kernel_size", self.kernel_size)
        self.add_prim_attr("strides", self.strides)
        validator.check_value_type("padding_dtype", padding, (str), self.name)
        self.padding = validator.check_string(padding.upper(), ['VALID', 'SAME'], 'padding', self.name)
        self.add_prim_attr("padding", self.padding)


class ScatterAddWithAxis(Primitive):
    """
    'ops.ScatterAddWithAxis' is deprecated from version 2.0 and will be removed in a future version,
    use 'ops.TensorScatterElements' instead.

    Supported Platforms:
        Deprecated

    Examples:
        >>> op = ops.ScatterAddWithAxis(0)
        >>> input_x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> indices = Tensor(np.array([[1, 0, 2], [0, 2, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[1, 1, 1], [1, 1, 1]]), mindspore.float32)
        >>> output = op(input_x, indices, updates)
        >>> print(output)
        [[ 2.  3.  3.]
         [ 5.  5.  7.]
         [ 7.  9.  10.]]
        >>> op = ops.ScatterAddWithAxis(1)
        >>> input_x = Tensor(np.array([[1, 2, 3, 4, 5]]), mindspore.int32)
        >>> indices = Tensor(np.array([[2, 4]]), mindspore.int32)
        >>> updates = Tensor(np.array([[8, 8]]), mindspore.int32)
        >>> output = op(input_x, indices, updates)
        >>> print(output)
        [[ 1  2  11  4  13]]
    """
    __mindspore_signature__ = (
        sig.make_sig('input_x', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('updates', dtype=sig.sig_dtype.T)
    )

    @deprecated("2.0", "ops.TensorScatterElements", False)
    @prim_attr_register
    def __init__(self, axis=0):
        """Initialize ScatterAddWithAxis"""
        validator.check_value_type("axis", axis, [int], self.name)
        self.init_prim_io_names(
            inputs=['input_x', 'indices', 'updates'], outputs=['y'])


class Lstsq(Primitive):
    r"""
    Computes the solutions of the least squares and minimum norm problems of full-rank
    matrix `x` of size :math:`(m \times n)` and matrix `a` of size :math:`(m \times k)`.

    If :math:`m \geq n`, `Lstsq` solves the least-squares problem:

    .. math::

       \begin{array}{ll}
       \min_y & \|xy-a\|_2
       \end{array}

    If :math:`m < n`, `Lstsq` solves the least-norm problem:

    .. math::

       \begin{array}{llll}
       \min_y & \|y\|_2 & \text{subject to} & xy = a
       \end{array}

    Args:
        fast (bool, optional): Solving algorithm. Default: ``True`` .

            - If `fast` is True, then the solution is computed by solving
              the normal equations using Cholesky decomposition.
            - If `fast` is False, an algorithm based on numerically robust
              completed orthogonal decomposition is used.

        l2_regularizer (float, optional): L2 regularization coefficient. Default: ``0.0`` .

    Inputs:
        - **x** (Tensor) - :math:`(m \times n)` matrix `x`. The input tensor whose data type is
          float16, float32 or float64.
        - **a** (Tensor) - :math:`(m \times k)` matrix `a`. The input tensor whose data type is
          float16, float32 or float64.

    Outputs:
        Tensor, the least squares or minimum norm problems solution, which has shape
        :math:`(n \times k)`. The data type is the same with `x`.

    Raises:
        TypeError: If the input `x` or `a` is not a Tensor.
        TypeError: If dtype of `x` or `a` is not one of: float16, float32, float64.
        TypeError: If the dtypes of `x` and `a` are not the same.
        ValueError: If the dimension of `x` is not equal to 2.
        ValueError: If the dimension of `a` is not equal to 2 or 1.
        ValueError: If the length of x_dims[0] is not equal to the length of a_dims[0].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([[2,1,5],[3,5,1],[1,1,1]]),mindspore.float32)
        >>> a = Tensor(np.array([[10,5],[15,8],[7,4]]),mindspore.float32)
        >>> op = ops.Lstsq()
        >>> output = op(x, a)
        >>> print(output)
        [[17.000002  11.000002 ]
         [-6.5000005 -4.500001 ]
         [-3.500002  -2.5000017]]
    """

    @prim_attr_register
    def __init__(self, fast=True, l2_regularizer=0.0):
        """Initialize Lstsq"""
        validator.check_type_name("fast", fast, True, self.name)
        validator.check_type_name("l2_regularizer", l2_regularizer, 0.0, self.name)
        self.fast = fast
        self.l2_regularizer = l2_regularizer


class LowerBound(Primitive):
    """
    Find the index of the lower bound of `values` in sorted sequence `sorted_x` element-wise.

    Args:
        out_type (:class:`mindspore.dtype`, optional): An optional data type of
            ``mindspore.dtype.int32`` and ``mindspore.dtype.int64`` .
            Default: ``mindspore.dtype.int32`` .

    Inputs:
        - **sorted_x** (Tensor) - The input tensor whose dtype is real number and
          the data of each row must be sorted in ascending order. The rank must be 2.
        - **values** (Tensor) - The input tensor whose dtype is the same as `sorted_x`
          and the first dimension of the shape of `values` must be equal to that of
          `sorted_x` . The rank must be 2.

    Outputs:
        Tensor, whose dtype is determined by `out_type` and whose shape is the same
        as that of `values`.

    Raises:
        TypeError: If `sorted_x` is not a Tensor.
        TypeError: If `values` is not a Tensor.
        TypeError: If `out_type` is invalid.
        TypeError: If the type of `sorted_x` is not the same as that of `values`.
        ValueError: If rank of the `sorted_x` is not equal to 2.
        ValueError: If rank of the `values` is not equal to 2.
        ValueError: If the first dimension of the shape of `sorted_x` is not equal to that of `values`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> lowerbound = ops.LowerBound(out_type = mindspore.int32)
        >>> sorted_x = Tensor(np.arange(12).reshape(3, 4).astype(np.int8))
        >>> values = Tensor(np.array([[3], [4], [8]]).astype(np.int8))
        >>> output = lowerbound(sorted_x, values)
        >>> print(output)
        [[3]
         [0]
         [0]]
    """

    @prim_attr_register
    def __init__(self, out_type=mstype.int32):
        """Initialize LowerBound"""
        valid_values = (mstype.int32, mstype.int64)
        validator.check_type_name("out_type", out_type, valid_values, self.name)
        self.init_prim_io_names(inputs=['sorted_x', 'values'], outputs=['y'])


class UpperBound(Primitive):
    """
    Returns a tensor that contains the index for finding the upper bound of the value of
    the input values element in the input sorted_x.

    Args:
        out_type (:class:`mindspore.dtype`, optional): Specified output type.
            Supported types: ``mindspore.dtype.int32`` and ``mindspore.dtype.int64`` .
            Default: ``mindspore.dtype.int32`` .

    Inputs:
        - **sorted_x** (Tensor) - The input tensor whose dtype is real number. The rank must be 2.
          Each row of the `sorted_x` needs to be sorted in ascending order.
        - **values** (Tensor) - The input tensor whose dtype is the same as `sorted_x`. The rank must be 2.
          The shape[0] of the two inputs must be consistent.

    Outputs:
        Tensor, whose dtype is determined by `out_type` and whose shape is consistent with `values`.

    Raises:
        TypeError: If `sorted_x` is not a Tensor.
        TypeError: If `values` is not a Tensor.
        TypeError: If the type of `sorted_x` is not the same as that of `values`.
        ValueError: If rank of the `sorted_x` is not equal to 2.
        ValueError: If rank of the `values` is not equal to 2.
        ValueError: If the number of rows of `sorted_x` is not consistent with that of `values`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> upperbound = ops.UpperBound(out_type = mindspore.int32)
        >>> sorted_x = Tensor(np.arange(12).reshape(3, 4).astype(np.int8))
        >>> values = Tensor(np.array([[3], [6], [9]]).astype(np.int8))
        >>> output = upperbound(sorted_x, values)
        >>> print(output)
        [[4]
         [3]
         [2]]
    """

    @prim_attr_register
    def __init__(self, out_type=mstype.int32):
        """Initialize UpperBound"""
        valid_values = (mstype.int32, mstype.int64)
        validator.check_type_name("out_type", out_type, valid_values, self.name)
        self.init_prim_io_names(inputs=['sorted_x', 'values'], outputs=['y'])


class LogSpace(Primitive):
    r"""
    Generates a 1-D Tensor with a length of steps. The tensor's
    values are uniformly distributed on a logarithmic scale, ranging from
    :math:`base^{start}` to :math:`base^{end}`, including both endpoints.
    The logarithmic scale is based on the specified `base`.

    .. math::
        \begin{aligned}
        &step = (end - start)/(steps - 1)\\
        &output = [base^{start}, base^{start + 1 * step}, ... , base^{start + (steps-2) * step}, base^{end}]
        \end{aligned}

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        steps (int, optional): The steps must be a non-negative integer. Default: ``10`` .
        base (int, optional): The base must be a non-negative integer. Default: ``10`` .
        dtype (mindspore.dtype, optional): The dtype of output, include ``mstype.float16`` ,
            ``mstype.float32`` or ``mstype.float64`` . Default: ``mstype.float32`` .

    Inputs:
        - **start** (Tensor) - Start value of interval, with shape of 0-D,
          dtype is float16, float32 or float64.
        - **end** (Tensor) - End value of interval, with shape of 0-D,
          dtype is float16, float32 or float64.

    Outputs:
        Tensor has the shape as :math:`(step, )`. Its datatype is set by the attr 'dtype'.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `steps` is not an int.
        TypeError: If `base` is not an int.
        TypeError: If `dtype` is not mstype.float16, mstype.float32 or
            mstype.float64.
        ValueError: If `steps` is not a non-negative integer.
        ValueError: If `base` is not a non-negative integer.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> from mindspore import dtype as mstype
        >>> logspace = ops.LogSpace(steps = 10, base = 10, dtype=mstype.float32)
        >>> start = Tensor(1, mstype.float32)
        >>> end = Tensor(10, mstype.float32)
        >>> output = logspace(start, end)
        >>> print(output)
        [1.e+01 1.e+02 1.e+03 1.e+04 1.e+05 1.e+06 1.e+07 1.e+08 1.e+09 1.e+10]
    """

    @prim_attr_register
    def __init__(self, steps=10, base=10, dtype=mstype.float32):
        """Initialize Logspace."""
        validator.check_value_type("steps", steps, [int], self.name)
        validator.check_value_type("base", base, [int], self.name)
        validator.check_non_negative_int(steps, "steps", self.name)
        validator.check_non_negative_int(base, "base", self.name)
        validator.check_value_type("dtype", dtype, [mstype.Type], self.name)
        valid_values = (mstype.float16, mstype.float32, mstype.float64)
        validator.check_type_name("dtype", dtype, valid_values, self.name)
        self.init_prim_io_names(inputs=['start', 'end'], outputs=['y'])


class Tril(Primitive):
    """
    Returns the lower triangular portion of the 2-D matrix or the set of matrices
    in a batch. The remaining elements of the resulting Tensor are assigned a value of 0.
    The lower triangular section of the matrix comprises of the
    elements present on and below the main diagonal.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        diagonal (int, optional): An optional attribute indicates the diagonal to consider, default: ``0`` ,
            indicating the main diagonal.

    Inputs:
        - **x** (Tensor) - The input tensor with shape :math:`(M, N, *)`
          where :math:`*` means any number of additional dimensions.

    Outputs:
        Tensor, the same shape and data type as the input `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `diagonal` is not an int.
        ValueError: If the rank of `x` is less than 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([[ 1,  2,  3,  4],
        ...                      [ 5,  6,  7,  8],
        ...                      [10, 11, 12, 13],
        ...                      [14, 15, 16, 17]]))
        >>> tril = ops.Tril()
        >>> result = tril(x)
        >>> print(result)
        [[ 1  0  0  0]
         [ 5  6  0  0]
         [10 11 12  0]
         [14 15 16 17]]
        >>> x = Tensor(np.array([[ 1,  2,  3,  4],
        ...                      [ 5,  6,  7,  8],
        ...                      [10, 11, 12, 13],
        ...                      [14, 15, 16, 17]]))
        >>> tril = ops.Tril(diagonal=1)
        >>> result = tril(x)
        >>> print(result)
        [[ 1  2  0  0]
         [ 5  6  7  0]
         [10 11 12 13]
         [14 15 16 17]]
        >>> x = Tensor(np.array([[ 1,  2,  3,  4],
        ...                      [ 5,  6,  7,  8],
        ...                      [10, 11, 12, 13],
        ...                      [14, 15, 16, 17]]))
        >>> tril = ops.Tril(diagonal=-1)
        >>> result = tril(x)
        >>> print(result)
        [[ 0  0  0  0]
         [ 5  0  0  0]
         [10 11  0  0]
         [14 15 16  0]]
    """

    @prim_attr_register
    def __init__(self, diagonal=0):
        """Initialize Tril."""
        self.init_prim_io_names(inputs=["x"], outputs=["y"])
        validator.check_value_type("diagonal", diagonal, [int], self.name)


class IndexFill(Primitive):
    """
    Fills the elements under the `dim` dimension of the input Tensor `x` with the input `value`
    by selecting the indices in the order given in `index`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Refer to :func:`mindspore.ops.index_fill` for more details.

    Inputs:
        - **x** (Tensor) - Input tensor.
        - **dim** (Union[int, Tensor]) - Dimension along which to fill the input tensor. Only supports
          a 0-dimensional tensor or an int number.
        - **index** (Tensor) - Indices of the input tensor to fill in.
        - **value** (Union[bool, int, float, Tensor]) - Value to fill the input tensor.

    Outputs:
        Tensor, has the same type and shape as input tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> index_fill = ops.IndexFill()
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32))
        >>> index = Tensor([0, 2], mindspore.int32)
        >>> value = Tensor(-2.0, mindspore.float32)
        >>> y = index_fill(x, 1, index, value)
        >>> print(y)
        [[-2. 2. -2.]
         [-2. 5. -2.]
         [-2. 8. -2.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize IndexFill"""
        self.init_prim_io_names(inputs=['x', 'dim', 'index', 'value'], outputs=['y'])


class IndexPut(Primitive):
    r"""
    According to the index number of `indexes`, replace the value corresponding to `x1` with the value in `x2`.

    Args:
        accumulate (int): If accumulate is 1, the elements in x2 are added to x1,
            else the elements in x2 replace the corresponding element in x1, should be 0 or 1. Default: ``0`` .

    Inputs:
        - **x1** (Tensor) - The assigned target tensor, 1-D or higher dimensional.
        - **x2** (Tensor) - 1-D Tensor of the same type as `x1`. If the size of `x2` is 1,
          it will broadcast to the same size as `x1`.
        - **indices** (tuple[Tensor], list[Tensor]) - the indices of type int32 or int64, used to index into x1.
          The rank of tensors in indices should be 1-D, size of indices should <= x1.rank and the tensors in indices
          should be broadcastable.

    Outputs:
        Tensor, has the same dtype and shape as `x1`.

    Raises:
        TypeError: If the dtype of `x1` is not equal to the dtype of `x2`.
        TypeError: If `indices` is not tuple[Tensor] or list[Tensor].
        TypeError: If the dtype of tensors in `indices` are not int32 or int64.
        TypeError: If the dtype of tensors in `indices` are inconsistent.
        TypeError: If the dtype of `accumulate` are not int.
        ValueError: If rank(x2) is not 1-D.
        ValueError: If size(x2) is not 1 or max size of the tensors in `indices` when rank(x1) == size(indices).
        ValueError: If size(x2) is not 1 or x1.shape[-1] when rank(x1) > size(indices).
        ValueError: If the rank of tensors in `indices` is not 1-D.
        ValueError: If the tensors in `indices` is not be broadcastable.
        ValueError: If size(indices) > rank(x1).
        ValueError: If `accumulate` is not equal to 0 or 1.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x1 = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32))
        >>> x2 = Tensor(np.array([3]).astype(np.int32))
        >>> indices = [Tensor(np.array([0, 0]).astype(np.int32)), Tensor(np.array([0, 1]).astype(np.int32))]
        >>> accumulate = 1
        >>> op = ops.IndexPut(accumulate = accumulate)
        >>> output = op(x1, x2, indices)
        >>> print(output)
         [[4 5 3]
         [4 5 6]]
    """

    @prim_attr_register
    def __init__(self, accumulate=0):
        self.accumulate = accumulate
        validator.check_value_type('accumulate', accumulate, [int], self.name)
        self.init_prim_io_names(inputs=['x1', 'x2', 'indices'], outputs=['y'])


class SegmentMax(Primitive):
    r"""
    Computes the maximum along segments of a Tensor.

    Specifically, it generates a new Tensor `output` such that :math:`output_i=max_j(input\_x_j)`
    in which the maximum value is obtained from all elements corresponding
    to :math:`j` that meets :math:`segment\_ids[j] == i`.
    If a segment contains no elements for a given segment :math:`i`,
    then the corresponding element in the output Tensor is set to zero: :math:`output[i] = 0`.

    Inputs:
        - **input_x** (Tensor) - The input tensor whose dtype is real number and whose rank is not less than 1.
        - **segment_ids** (Tensor) - A 1-D tensor whose dtype is int32 or int64. The size of tensor must be equal to
          the first dimension of the shape of `input_x`. Values must be sorted in ascending order and need not cover
          all values in the full range of valid values, but must be positive integer. Only constant values is allowed.

    Outputs:
        Tensor, whose dtype and the dimension of the shape is the same as `input_x`. The first dimension of the shape
        is equal to the value of the last element of `segment_ids` plus one, and the other dimensions are the same as
        those of `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        TypeError: If `segment_ids` is not a Tensor.
        TypeError: If the dtype of `input_x` is invalid.
        TypeError: If the dtype of `segment_ids` is invalid.
        ValueError: If the rank of `input_x` is less than 1.
        ValueError: If the rank of `segment_ids` is not equal to 1.
        ValueError: If the size of `segment_ids` is not equal to the first dimension of the shape of `input_x`.
        ValueError: If the values of `segment_ids` are negative.
        ValueError: If the values of `segment_ids` are not sorted in ascending order.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mstype.float64)
        >>> segment_ids = Tensor([0, 0, 2], mstype.int64)
        >>> op = ops.SegmentMax()
        >>> output = op(x, segment_ids)
        >>> print(output)
        [[4. 5. 6.]
         [0. 0. 0.]
         [7. 8. 9.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SegmentMax"""
        self.add_prim_attr("max_length", 1000000)
        self.init_prim_io_names(inputs=['input_x', 'segment_ids'], outputs=['output'])


class SegmentMin(Primitive):
    r"""
    Computes the minimum along segments of a Tensor.

    Specifically, it generates a new Tensor `output` such that :math:`output_i=min_j(input\_x_j)`
    in which the minimum value is obtained from all elements corresponding
    to :math:`j` that meets :math:`segment\_ids[j] == i`.
    If a segment contains no elements for a given segment :math:`i`,
    then the corresponding element in the output Tensor is set to zero: :math:`output[i] = 0`.

    Inputs:
        - **input_x** (Tensor) - The input tensor whose dtype is real number and whose rank is not less than 1.
        - **segment_ids** (Tensor) - A 1-D tensor whose dtype is int32 or int64. The size of tensor must be equal to
          the first dimension of the shape of `input_x`. Values must be sorted in ascending order and need not cover
          all values in the full range of valid values, but must be positive integer. Only constant values is allowed.

    Outputs:
        Tensor, whose dtype and the dimension of the shape is the same as `input_x`. The first dimension of the shape
        is equal to the value of the last element of `segment_ids` plus one, and the other dimensions are the same as
        those of `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        TypeError: If `segment_ids` is not a Tensor.
        TypeError: If the dtype of `input_x` is invalid.
        TypeError: If the dtype of `segment_ids` is invalid.
        ValueError: If the rank of `input_x` is less than 1.
        ValueError: If the rank of `segment_ids` is not equal to 1.
        ValueError: If the size of `segment_ids` is not equal to the first dimension of the shape of `input_x`.
        ValueError: If the values of `segment_ids` are negative.
        ValueError: If the values of `segment_ids` are not sorted in ascending order.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mstype.float64)
        >>> segment_ids = Tensor([0, 0, 2], mstype.int64)
        >>> op = ops.SegmentMin()
        >>> output = op(x, segment_ids)
        >>> print(output)
        [[1. 2. 3.]
         [0. 0. 0.]
         [7. 8. 9.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SegmentMin"""
        self.add_prim_attr("max_length", 1000000)
        self.init_prim_io_names(inputs=['input_x', 'segment_ids'], outputs=['output'])


class SegmentSum(Primitive):
    r"""
    Computes the cumulative sum along segments of a Tensor.

    Specifically, it generates a new Tensor `output` such that :math:`output_i = \sum_j input\_x_j`
    in which the cumulative sum is obtained from all elements corresponding
    to :math:`j` that meets :math:`segment\_ids[j] == i`.
    If a segment contains no elements for a given segment :math:`i`,
    then the corresponding element in the output Tensor is set to 0: :math:`output[i] = 0`.

    .. warning::
        If the dtype of `input_x` is complex number, the gradient can not be calculated.

    Inputs:
        - **input_x** (Tensor) - The input tensor whose dtype is real number or complex number and whose rank is not
          less than 1.
        - **segment_ids** (Tensor) - A 1-D tensor whose dtype is int32 or int64. The size of tensor must be equal to
          the first dimension of the shape of `input_x`. Values must be sorted in ascending order and need not cover
          all values in the full range of valid values, but must be positive integer. Only constant values is allowed.

    Outputs:
        Tensor, whose dtype and the dimension of the shape is the same as `input_x`. The first dimension of the shape
        is equal to the value of the last element of `segment_ids` plus one, and the other dimensions are the same as
        those of `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        TypeError: If `segment_ids` is not a Tensor.
        TypeError: If the dtype of `input_x` is invalid.
        TypeError: If the dtype of `segment_ids` is invalid.
        ValueError: If the rank of `input_x` is less than 1.
        ValueError: If the rank of `segment_ids` is not equal to 1.
        ValueError: If the size of `segment_ids` is not equal to the first dimension of the shape of `input_x`.
        ValueError: If the values of `segment_ids` are negative.
        ValueError: If the values of `segment_ids` are not sorted in ascending order.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mstype.float64)
        >>> segment_ids = Tensor([0, 0, 2], mstype.int64)
        >>> op = ops.SegmentSum()
        >>> output = op(x, segment_ids)
        >>> print(output)
        [[5. 7. 9.]
         [0. 0. 0.]
         [7. 8. 9.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SegmentSum"""
        self.add_prim_attr("max_length", 1000000)
        self.init_prim_io_names(inputs=['input_x', 'segment_ids'], outputs=['output'])


class LeftShift(Primitive):
    r"""
    Shift the value of each position of the tensor to the left several bits.
    The inputs are two tensors, dtypes of them must be consistent, and the
    shapes of them could be broadcast.
    The output does not support implicit type conversion.

    .. math::

        \begin{aligned}
        &out_{i} =x_{i} << y_{i}
        \end{aligned}

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **x1** (Tensor) - The target tensor whose dtype supports all int and uint type,
          will be shifted to the left by `x2` in element-wise.
        - **x2** (Tensor) - The tensor must have the same dtype as `x1`.
          And the tensor must have the same shape as `x1` or could be broadcast with `x1`.

    Outputs:
        - **output** (Tensor) - The output tensor, has the same dtype as `x1`.
          And the shape of the output tensor is the same shape as `x1`, or the same shape
          as `x1` and `x2` after broadcasting.

    Raises:
        TypeError: If `x1` or `x2` has wrong type.
        TypeError: If `x1` or `x2` is not tensor.
        ValueError: If `x1` and `x2` could not be broadcast.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> left_shift = ops.LeftShift()
        >>> x1 = Tensor(np.array([1, 2, 3]).astype(np.int8))
        >>> x2 = Tensor(np.array([0, 1, -1]).astype(np.int8))
        >>> output = left_shift(x1, x2)
        >>> print(output)
        [1 4 0]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize LeftShift"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])


class FillDiagonal(Primitive):
    """
    Fills the main diagonal of a Tensor in-place with a specified value and returns the result.
    The input has at least 2 dimensions, and all dimensions of input must be equal in length
    when the dimension of input is greater than 2.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        fill_value (float): The value to fill the diagonal of `input_x`.
        wrap (bool, optional): Controls whether the diagonal elements continue onto the
            remaining rows in case of a tall matrix(A matrix has more rows than columns).
            Examples blow demonstrates how it works on a tall matrix if `wrap` is set ``True`` .
            Default: ``False`` .

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        - **y** (Tensor) - Tensor, has the same shape and data type as the input `input_x`.

    Raises:
        ValueError: If the dimension of `input_x` is not greater than 1.
        ValueError: If the size of each dimension is not equal, when the dimension is greater than 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32))
        >>> fill_value = 9.9
        >>> fill_diagonal = ops.FillDiagonal(fill_value)
        >>> y = fill_diagonal(x)
        >>> print(y)
        [[9.9 2.  3. ]
         [4.  9.9 6. ]
         [7.  8.  9.9]]
        >>> x = Tensor(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]).astype(np.int32))
        >>> fill_value = 9.0
        >>> fill_diagonal = ops.FillDiagonal(fill_value)
        >>> y = fill_diagonal(x)
        >>> print(y)
        [[9 0 0]
         [1 9 1]
         [2 2 9]
         [3 3 3]
         [4 4 4]
         [5 5 5]]
        >>> x = Tensor(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3],
        ...                      [4, 4, 4], [5, 5, 5], [6, 6, 6]]).astype(np.int64))
        >>> fill_value = 9.0
        >>> wrap = True
        >>> fill_diagonal = FillDiagonal(fill_value, wrap)
        >>> y = fill_diagonal(x)
        >>> print(y)
        [[9 0 0]
         [1 9 1]
         [2 2 9]
         [3 3 3]
         [9 4 4]
         [5 9 5]
         [6 6 9]]
    """

    @prim_attr_register
    def __init__(self, fill_value, wrap=False):
        """Initialize FillDiagonal"""
        validator.check_value_type('fill_value', fill_value, [float], self.name)
        self.fill_value = fill_value
        validator.check_value_type('wrap', wrap, [bool], self.name)
        self.init_prim_io_names(inputs=['input_x'], outputs=['y'])


class HammingWindow(Primitive):
    r"""
    Computes the hamming window function with input window length.

    .. math::

        w[n] = \alpha - \beta\ \cos \left( \frac{2 \pi n}{N - 1} \right),

    where :math:`N` is the full window size.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        periodic (bool, optional): a flag determines whether the returned window trims off
            the last duplicate value from the symmetric window. Default: ``True`` .

            - If True, returns a window to be used as periodic function, in above formula,
              :math:`N = \text{length} + 1`.
            - If False, return a symmetric window, :math:`N = \text{length}`.

        alpha (float, optional): The coefficient :math:`\alpha` in the equation above. Default: ``0.54`` .
        beta (float, optional): The coefficient :math:`\beta` in the equation above. Default: ``0.46`` .
        dtype (:class:`mindspore.dtype`, optional): An optional data type of ``mstype.float16`` ,
            ``mstype.float32`` and ``mstype.float64`` . Default: ``mstype.float32``.

    Inputs:
        - **length** (Tensor) - a positive integer tensor controlling the returned window size, must be 1D.

    Outputs:
        Tensor, A 1-D tensor containing the window, whose shape is :math:`(\text{length},)`.

    Raises:
        TypeError: If `length` is not a Tensor.
        TypeError: If dtype of `length` is not integer data type.
        TypeError: If `periodic` is not a bool.
        TypeError: If `alpha` is not a float.
        TypeError: If `beta` is not a float.
        TypeError: If `dtype` is not mindspore.float16, mindspore.float32 or mindspore.float64.
        ValueError: If dimension of `length` is not 1.
        ValueError: If data of `length` is negative.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> # case 1: periodic=True.
        >>> length = Tensor(np.array([6]).astype(np.int32))
        >>> hamming_window = ops.HammingWindow(periodic=True)
        >>> y = hamming_window(length)
        >>> print(y)
        [0.08000001 0.31       0.77000004 1.         0.77000004 0.31      ]
        >>> # case 2: periodic=False.
        >>> length = Tensor(np.array([7]).astype(np.int32))
        >>> hamming_window = ops.HammingWindow(periodic=False)
        >>> y = hamming_window(length)
        >>> print(y)
        [0.08000001 0.31       0.77000004 1.         0.77000004 0.31       0.08000001]
    """

    @prim_attr_register
    def __init__(self, periodic=True, alpha=0.54, beta=0.46, dtype=mstype.float32):
        """Initialize HammingWindow"""
        validator.check_value_type("periodic", periodic, [bool], self.name)
        validator.check_value_type("alpha", alpha, [float], self.name)
        validator.check_value_type("beta", beta, [float], self.name)
        validator.check_value_type("dtype", dtype, [mstype.Type], self.name)
        valid_values = (mstype.float16, mstype.float32, mstype.float64)
        validator.check_type_name("dtype", dtype, valid_values, self.name)
        self.init_prim_io_names(inputs=['length'], outputs=['y'])
        if dtype == mstype.float16:
            self.add_prim_attr('dtype', 1)
        elif dtype == mstype.float32:
            self.add_prim_attr('dtype', 0)
        else:
            self.add_prim_attr('dtype', 11)


class AffineGrid(Primitive):
    r"""
    Creates a 2D or 3D flow field (sampling grid) based on a batch of affine matrices `theta`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Refer to :func:`mindspore.ops.affine_grid` for more details.

    Args:
        align_corners (bool, optional): Geometrically, each pixel of input is viewed as a squqre instead of dot.
            If True, consider extremum -1 and 1 referring to the centers of the pixels rather than pixel corners.
            The default value is ``False`` , extremum -1 and 1 refer to the corners of the pixels, so that sampling is
            irrelevant to resolution of the image. Default: ``False`` .

    Inputs:
        - **theta** (Tensor) - The input tensor of flow field whose dtype is float16, float32.
          Input batch of affine matrices with shape :math:`(N, 2, 3)` for 2D grid or :math:`(N, 3, 4)` for 3D grid.
        - **output_size** (tuple[int]) - The target output image size.
          The value of target output with format :math:`(N, C, H, W)` for 2D grid
          or :math:`(N, C, D, H, W)` for 3D grid.

    Outputs:
        Tensor, a tensor whose data type is same as 'theta', and the shape is :math:`(N, H, W, 2)` for 2D grid
        or :math:`(N, D, H, W, 3)` for 3D grid.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, ops
        >>> affinegrid = ops.AffineGrid(align_corners=False)
        >>> theta = Tensor([[[0.8, 0.5, 0],[-0.5, 0.8, 0]]], mindspore.float32)
        >>> out_size = (1, 3, 2, 3)
        >>> output = affinegrid(theta, out_size)
        >>> print(output)
        [[[[-0.78333336 -0.06666666]
        [-0.25       -0.4       ]
        [ 0.28333336 -0.73333335]]
        [[-0.28333336  0.73333335]
        [ 0.25        0.4       ]
        [ 0.78333336  0.06666666]]]]
    """

    @prim_attr_register
    def __init__(self, align_corners=False):
        """Initialize AffineGrid."""
        validator.check_value_type("align_corners", align_corners, [bool], self.name)
        self.init_prim_io_names(inputs=['theta', 'output_size'], outputs=['y'])


class SegmentMean(Primitive):
    r"""
    Computes the mean along segments of a Tensor.

    Specifically, it generates a new Tensor `output` such that :math:`output_i=mean_j(input\_x_j)`
    in which the mean value is obtained from all elements corresponding
    to :math:`j` that meets :math:`segment\_ids[j] == i`.
    If a segment contains no elements for a given segment :math:`i`,
    then the corresponding element in the output Tensor is set to zero: :math:`output[i] = 0`.

    .. warning::
        If the dtype of `input_x` is complex number, the gradient can not be calculated.

    Inputs:
        - **input_x** (Tensor) - The input tensor whose dtype is real number or complex number and whose rank is not
          less than 1.
        - **segment_ids** (Tensor) - A 1-D tensor whose dtype is int32 or int64. The size of tensor must be equal to
          the first dimension of the shape of `input_x`. Values must be sorted in ascending order and need not cover
          all values in the full range of valid values, but must be positive integer. Only constant values is allowed.

    Outputs:
        Tensor, whose dtype and the dimension of the shape is the same as `input_x`. The first dimension of the shape
        is equal to the value of the last element of `segment_ids` plus one, and the other dimensions are the same as
        those of `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        TypeError: If `segment_ids` is not a Tensor.
        TypeError: If the dtype of `input_x` is invalid.
        TypeError: If the dtype of `segment_ids` is invalid.
        ValueError: If the rank of `input_x` is less than 1.
        ValueError: If the rank of `segment_ids` is not equal to 1.
        ValueError: If the size of `segment_ids` is not equal to the first dimension of the shape of `input_x`.
        ValueError: If the values of `segment_ids` are negative.
        ValueError: If the values of `segment_ids` are not sorted in ascending order.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([[1, 2, 3], [1, 2, 3], [7, 8, 9]], mstype.float64)
        >>> segment_ids = Tensor([0, 0, 2], mstype.int64)
        >>> op = ops.SegmentMean()
        >>> output = op(x, segment_ids)
        >>> print(output)
        [[1. 2. 3.]
         [0. 0. 0.]
         [7. 8. 9.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SegmentMean"""
        self.add_prim_attr("max_length", 1000000)
        self.init_prim_io_names(inputs=['input_x', 'segment_ids'], outputs=['output'])


class SegmentProd(Primitive):
    r"""
    Computes the cumulative product along segments of a Tensor.

    Specifically, it generates a new Tensor `output` such that :math:`output_i = \prod_j input\_x_j`
    in which the cumulative product is obtained from all elements corresponding
    to :math:`j` that meets :math:`segment\_ids[j] == i`.
    If a segment contains no elements for a given segment :math:`i`,
    then the corresponding element in the output Tensor is set to 1: :math:`output[i] = 1`.

    .. warning::
        If the dtype of `input_x` is complex number, the gradient can not be calculated.

    Inputs:
        - **input_x** (Tensor) - The input tensor whose dtype is real number or complex number and whose rank is not
          less than 1.
        - **segment_ids** (Tensor) - A 1-D tensor whose dtype is int32 or int64. The size of tensor must be equal to
          the first dimension of the shape of `input_x`. Values must be sorted in ascending order and need not cover
          all values in the full range of valid values, but must be positive integer. Only constant values is allowed.

    Outputs:
        Tensor, whose dtype and the dimension of the shape is the same as `input_x`. The first dimension of the shape
        is equal to the value of the last element of `segment_ids` plus one, and the other dimensions are the same as
        those of `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        TypeError: If `segment_ids` is not a Tensor.
        TypeError: If the dtype of `input_x` is invalid.
        TypeError: If the dtype of `segment_ids` is invalid.
        ValueError: If the rank of `input_x` is less than 1.
        ValueError: If the rank of `segment_ids` is not equal to 1.
        ValueError: If the size of `segment_ids` is not equal to the first dimension of the shape of `input_x`.
        ValueError: If the values of `segment_ids` are negative.
        ValueError: If the values of `segment_ids` are not sorted in ascending order.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mstype.float64)
        >>> segment_ids = Tensor([0, 0, 2], mstype.int64)
        >>> op = ops.SegmentProd()
        >>> output = op(x, segment_ids)
        >>> print(output)
        [[ 4. 10. 18.]
         [ 1.  1.  1.]
         [ 7.  8.  9.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SegmentProd"""
        self.add_prim_attr("max_length", 1000000)
        self.init_prim_io_names(inputs=['input_x', 'segment_ids'], outputs=['output'])


class PopulationCount(Primitive):
    r"""
    Computes element-wise population count(a.k.a bitsum, bitcount).

    Refer to :func:`mindspore.ops.population_count` for more details.

    Inputs:
        - **input_x** (Tensor) - Tensor of any dimension. The data type must be int16 or uint16 (Ascend).
          The data type must be int8, int16, int32, int64, uint8, uint16, uint32, uint64 (CPU and GPU).

    Outputs:
        Tensor, with the same shape as the input, and the data type is uint8.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor([0, 1, 3], mindspore.int16)
        >>> output = ops.PopulationCount()(input_x)
        >>> print(output)
        [0 1 2]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize PopulationCount"""
        self.init_prim_io_names(inputs=['input'], outputs=['output'])


class TopK(Primitive):
    """
    Finds values and indices of the `k` largest entries along the last dimension.

    .. warning::
        - If sorted is set to False, it will use the aicpu operator, the performance may be reduced. In addition, due to
          different memory layout and traversal methods on different platforms, the display order of calculation results
          may be inconsistent when `sorted` is False.

    If the `input_x` is a one-dimensional Tensor, finds the `k` largest entries in the Tensor,
    and outputs its value and index as a Tensor. values[`k`] is the `k` largest item in `input_x`,
    and its index is indices [`k`].

    For a multi-dimensional matrix,
    calculates the first `k` entries in each row (corresponding vector along the last dimension), therefore:

    .. math::

        values.shape = indices.shape = input.shape[:-1] + [k]

    If the two compared elements are the same, the one with the smaller index value is returned first.

    Args:
        sorted (bool, optional): If ``True`` , the obtained elements will be sorted by the values in descending order.
            If ``False`` , the obtained elements will not be sorted. Default: ``True`` .

    Inputs:
        - **input_x** (Tensor) - Input to be computed, 0-D input is supported on GPU, but not on Ascend or CPU.
          supported dtypes:

          - Ascend: int8, uint8, int32, int64, float16, float32.
          - GPU: float16, float32.
          - CPU: all numeric types.

        - **k** (Union(Tensor, int)) - The number of top elements to be computed along the last dimension.
          If `k` is a Tensor, the supported dtype is int32 and it should be 0-D or 1-D with shape :math:`(1, )` .

    Outputs:
        A tuple consisting of `values` and `indexes`.

        - **values** (Tensor) - The `k` largest elements in each slice of the last dimension.
        - **indices** (Tensor) - The indices of values within the last dimension of input.

    Raises:
        TypeError: If `sorted` is not a bool.
        TypeError: If `input_x` is not a Tensor.
        TypeError: If `k` is not an int.
        TypeError: If dtype of `input_x` is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> import mindspore
        >>> input_x = Tensor([1, 2, 3, 4, 5], mindspore.float16)
        >>> k = 3
        >>> values, indices = ops.TopK(sorted=True)(input_x, k)
        >>> print((values, indices))
        (Tensor(shape=[3], dtype=Float16, value= [ 5.0000e+00,  4.0000e+00,  3.0000e+00]), Tensor(shape=[3],
          dtype=Int32, value= [4, 3, 2]))
    """

    @prim_attr_register
    def __init__(self, sorted=True):
        """Initialize TopK."""
        self.sorted = validator.check_value_type("sorted", sorted, [bool], self.name)
        self.add_prim_attr("sorted", self.sorted)
        self.init_prim_io_names(inputs=['input', 'k'],
                                outputs=['values', 'indices'])


class Bincount(Primitive):
    """
    Counts the number of occurrences of each value in an integer array.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **array** (Tensor) - A Tensor of type int32, whose value can not be less than zero.
        - **size** (Tensor) - A non-negative Tensor of type int32.
        - **weights** (Tensor) - A Tensor with the same shape as array, or a length-0 Tensor, in which case it acts as
          all weights equal to 1. Must be one of the following types: int32, int64, float32, float64.

    Outputs:
        A Tensor. Has the same type as weights.

    Raises:
        TypeError: If dtype of `array` is not int32.
        TypeError: If dtype of `size` is not int32.
        ValueError: If `size` is negative.
        ValueError: If `weights` are empty.
        ValueError: If size of `weights` is not zero and the shape of `weights` is different with the shape of `array`.
        TypeError: If dtype of `weights` is not in int32,int64,float32,float64

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> array = Tensor(np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4]), mindspore.int32)
        >>> size = Tensor(5, mindspore.int32)
        >>> weights = Tensor(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), mindspore.float32)
        >>> bincount = ops.Bincount()
        >>> bins = bincount(array, size, weights)
        >>> print(bins)
        [0. 1. 2. 3. 4.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Bincount"""
        self.init_prim_io_names(inputs=['array', 'size', 'weights'], outputs=['bins'])


class CountNonZero(Primitive):
    """
    Calculates the total number of non-zero entries in the input tensor along the
    specified dimensions.

    Refer to :func:`mindspore.ops.count_nonzero` for more details.

    Args:
        dims (Union[int, tuple(int), list(int)], optional): The dimensions to reduce.
            Default: ``None`` , reduce over all dimensions.

    Inputs:
        - **x** (Tensor) - Input data is used to count non-zero numbers. With shape
          :math:`(N, *)` where :math:`*` means, any number of additional dimensions.

    Outputs:
          Tensor, number of nonzero element across axis specified by `dims`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x = Tensor([[0, 0, 1], [1, 1, 2], [0, 0, 1]], dtype=mindspore.int64)
        >>> countnonzero = ops.CountNonZero(dims=[1])
        >>> y = countnonzero(x)
        >>> print(y)
        [1 3 1]
    """

    @prim_attr_register
    def __init__(self, dims=None):
        dims = [] if dims is None else dims
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        validator.check_value_type('dims', dims, [int, list, tuple], "CountNonZero")
        if isinstance(dims, (list, tuple)):
            for i, each in enumerate(dims):
                validator.check_value_type(f'dims[{i}]', each, [int], "CountNonZero")
        self.dims = dims
        self.add_prim_attr("dims", self.dims)
