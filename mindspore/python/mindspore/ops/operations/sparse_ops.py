# coding: utf-8

# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""Operators for sparse operators."""

from mindspore._checkparam import Validator as validator
from mindspore.common import dtype as mstype
from mindspore.ops import signature as sig
from mindspore.ops.primitive import prim_attr_register, Primitive


class SparseDenseCwiseAdd(Primitive):
    """
    The dense tensor is broadcast into the shape of SparseTensor, and then the corresponding
    positions of the two matrices are added element by element according to the `x1_indices`, then output.
    Note: only dense tensor can be broadcast to SparseTensor.

    Inputs:
        - **x1_indices** (Tensor) -  A 2-D Tensor,N x R matrix with the indices of non-empty values in a SparseTensor,
          possibly not in canonical ordering.Support int64, each element value should be a non-negative number.
          The shape is :math:`(N, R)`.
        - **x1_values** (Tensor) - A 1-D Tensor, N non-empty values corresponding to `x1_indices`.
          The shape should be :math:`(N,)`.
        - **x1_shape**(Tensor) - A Tensor of type int64. 1-D. Shape of the input SparseTensor.
        - **x2** (Tensor) - A R-D tensor, must have the same type as `x1_values`. The dense tensor operand.

    Returns:
        Tensor, a new instance of SparseDenseCwiseAdd. The dtype is same as `x1_values`, and the shape is same with
        the shape of `x1_values`.

    Raises:
        TypeError: If the dtype of `x1_indices` and  dtype of `x1_shape` is not int64.
        TypeError: If the dtype of `x1_values` and  dtype of `x2` is not same.
        ValueError: If the dims of `x1_indices` is not 2.
        ValueError: If the dims of `x1_values` is not 1.
        ValueError: If the dims of `x1_shape` is not 1.
        ValueError: If dense tensor cannot be broabcast to SparseTensor. The size of the trailing axes for `x2` and
             sparse in an operation must either be the same size or size of the trailing axes for `x2` must be 1.
        ValueError: If shape[0] of `x1_indices` is not equal to shape[0] of `x1_values`.
        ValueError: If shape[1] of `x1_indices` is not equal to shape[0] of `x1_shape`.
        ValueError: If `x1_indices` proceed to cross the border the interview.


    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.common.tensor import Tensor
        >>> from mindspore.common import dtype as ms
        >>> from mindspore.ops.operations import sparse_ops as ops
        >>> x1_indices = Tensor([[0, 0], [2, 2]], dtype=ms.int64)
        >>> x1_values = Tensor([1, 2], dtype=ms.int32)
        >>> x1_shape = Tensor([3, 3], dtype=ms.int64)
        >>> x2=Tensor([1,2,3],dtype=ms.int32)
        >>> sparse_dense_cwise_add = ops.SparseDenseCwiseAdd()
        >>> y = sparse_dense_cwise_add(x1_indices, x1_values, x1_shape, x2)
        >>> print(y)
        [2 5]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseDenseCwiseAdd."""
        self.init_prim_io_names(
            inputs=['x1_indices', 'x1_values', 'x1_shape', 'x2'], outputs=['y'])


class SparseDenseCwiseMul(Primitive):
    """
    The dense tensor is broadcast into the shape of SparseTensor, and then the corresponding
    positions of the two matrices are multiplied element by element according to the `x1_indices`, then output.
    Note: only dense tensor can be broadcast to SparseTensor.

    Inputs:
        - **x1_indices** (Tensor) -  A 2-D Tensor,N x R matrix with the indices of non-empty values in a SparseTensor,
          possibly not in canonical ordering.Support int64, each element value should be a non-negative number.
          The shape is :math:`(N, R)`.
        - **x1_values** (Tensor) - A 1-D Tensor, N non-empty values corresponding to `x1_indices`.
          The shape should be :math:`(N,)`.
        - **x1_shape**(Tensor) - A Tensor of type int64. 1-D. Shape of the input SparseTensor.
        - **x2** (Tensor) - A R-D tensor, must have the same type as `x1_values`. The dense tensor operand.

    Returns:
        Tensor, a new instance of SparseDenseCwiseMul. The dtype is same as `x1_values`, and the shape is same with the
        shape of `x1_values`.

    Raises:
        TypeError: If the dtype of `x1_indices` and  dtype of `x1_shape` is not int64.
        TypeError: If the dtype of `x1_values` and  dtype of `x2` is not same.
        ValueError: If the dims of `x1_indices` is not 2.
        ValueError: If the dims of `x1_values` is not 1.
        ValueError: If the dims of `x1_shape` is not 1.
        ValueError: If dense tensor cannot be broabcast to SparseTensor. The size of the trailing axes for `x2` and
             sparse in an operation must either be the same size or size of the trailing axes for `x2` must be 1.
        ValueError: If shape[0] of `x1_indices` is not equal to shape[0] of `x1_values`.
        ValueError: If shape[1] of `x1_indices` is not equal to shape[0] of `x1_shape`.
        ValueError: If `x1_indices` proceed to cross the border the interview.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.common.tensor import Tensor
        >>> from mindspore.common import dtype as ms
        >>> from mindspore.ops.operations import sparse_ops as ops
        >>> x1_indices = Tensor([[0, 0], [2, 2]], dtype=ms.int64)
        >>> x1_values = Tensor([1, 2], dtype=ms.int32)
        >>> x1_shape = Tensor([3, 3], dtype=ms.int64)
        >>> x2=Tensor([1,2,3],dtype=ms.int32)
        >>> sparse_dense_cwise_mul = ops.SparseDenseCwiseMul()
        >>> y = sparse_dense_cwise_mul(x1_indices, x1_values, x1_shape, x2)
        >>> print(y)
        [1 6]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseDenseCwiseMul."""
        self.init_prim_io_names(
            inputs=['x1_indices', 'x1_values', 'x1_shape', 'x2'], outputs=['y'])


class SparseDenseCwiseDiv(Primitive):
    """
    The dense tensor is broadcast into the shape of SparseTensor, and then the corresponding positions elements of
    the dense tensor which non-zeros are divided by SparseTensor element by element according to the `x1_indices`,
    then output.Note: only dense tensor can be broadcast to SparseTensor.

    Inputs:
        - **x1_indices** (Tensor) -  A 2-D Tensor,N x R matrix with the indices of non-empty values in a SparseTensor,
          possibly not in canonical ordering.Support int64, each element value should be a non-negative number.
          The shape is :math:`(N, R)`.
        - **x1_values** (Tensor) - A 1-D Tensor, N non-empty values corresponding to `x1_indices`.
          The shape should be :math:`(N,)`.
        - **x1_shape**(Tensor) - A Tensor of type int64. 1-D. Shape of the input SparseTensor.
        - **x2** (Tensor) - A R-D tensor, must have the same type as `x1_values`. The dense tensor operand.

    Returns:
        Tensor, a new instance of SparseDenseCwiseDiv. The dtype is same as `x1_values`, and the shape is same with
        the shape of `x1_values`.

    Raises:
        TypeError: If the dtype of `x1_indices` and  dtype of `x1_shape` is not int64.
        TypeError: If the dtype of `x1_values` and  dtype of `x2` is not same.
        ValueError: If the dims of `x1_indices` is not 2.
        ValueError: If the dims of `x1_values` is not 1.
        ValueError: If the dims of `x1_shape` is not 1.
        ValueError: If dense tensor cannot be broabcast to SparseTensor. The size of the trailing axes for `x2` and
             sparse in an operation must either be the same size or size of the trailing axes for `x2` must be 1.
        ValueError: If shape[0] of `x1_indices` is not equal to shape[0] of `x1_values`.
        ValueError: If shape[1] of `x1_indices` is not equal to shape[0] of `x1_shape`.
        ValueError: If `x1_indices` proceed to cross the border the interview.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
      >>> from mindspore.common.tensor import Tensor
      >>> from mindspore.common import dtype as ms
      >>> from mindspore.ops.operations import sparse_ops as ops
      >>> x1_indices = Tensor([[0, 0], [2, 2]], dtype=ms.int64)
      >>> x1_values = Tensor([4, 2], dtype=ms.int32)
      >>> x1_shape = Tensor([3, 3], dtype=ms.int64)
      >>> x2=Tensor([1,2,2],dtype=ms.int32)
      >>> sparse_dense_cwise_div = ops.SparseDenseCwiseDiv()
      >>> y = sparse_dense_cwise_div(x1_indices, x1_values, x1_shape, x2)
      >>> print(y)
      [4 1]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseDenseCwiseDiv."""
        self.init_prim_io_names(
            inputs=['x1_indices', 'x1_values', 'x1_shape', 'x2'], outputs=['y'])


class SparseSlice(Primitive):
    r"""
    Slices a SparseTensor based on the `start` and `size`.

    Inputs:
        - **indices** (Tensor) - A 2D Tensor (N x R matrix), the indices of the SparseTensor.
          Support int64, each element value should be a non-negative int number.
          The shape is :math:`(N, R)`.
        - **values** (Tensor) - A 1D Tensor, represents the value corresponding to the position in the `indices`.
          The shape should be :math:`(N,)`.
        - **shape** (Tensor) - A 1D Tensor of type int64 which specifies the shape of sparsetensor,
          represent sparse tensor shape. The shape should be :math:`(R,)`.
        - **start** (Tensor) - A 1D Tensor of type int64, represents the start of the slice.
          The shape should be :math:`(R,)`.
        - **size** (Tensor) - A 1D Tensor of type int64, represents the size of the slice.
          The shape should be :math:`(R,)`.

    Outputs:
        A `SparseTensor` objects resulting from splicing.

        - \*y_indices: A Tensor of type int64.
        - \*y_values: A Tensor. Has the same type as `values`.
        - \*y_shape: A Tensor of type int64. Has the same size as `size`.

    Raises:
        TypeError: If the dtype of `indices`, `shape`, `start`, `size` are not int64.
        ValueError: If `indices` is not 2-D tensor.
        ValueError: If `values`, `start`, `shape` , `size` is not a 1-D tensor.
        ValueError: If the number of `indices` is not corresponding to the number of `values`.
        ValueError: If the shape of `indices[1]` is not corresponding to `shape`.
        ValueError: If the shape of `shape` is not corresponding to `start`.
        ValueError: If the shape of `shape` is not corresponding to `size`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor(np.array([[0, 1], [1, 2], [1, 3], [2, 2]]).astype(np.int64))
        >>> values = Tensor(np.array([1, 2, 3, 4]).astype(np.int64))
        >>> shape = Tensor(np.array([3, 4]).astype(np.int64))
        >>> start = Tensor(np.array([0, 1]).astype(np.int64))
        >>> size = Tensor(np.array([2, 3]).astype(np.int64))
        >>> sparseslice = ops.SparseSlice()
        >>> output = sparseslice(indices, values, shape, start, size)
        >>> print(output[0])
        [[0 0]
         [1 1]
         [1 2]]
        >>> print(output[1])
        [1 2 3]
        >>> print(output[2])
        [2 3]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseSlice."""
        self.init_prim_io_names(inputs=['indices', 'values', 'shape', 'start', 'size'],
                                outputs=['y_indices', 'y_values', 'y_shape'])


class SparseSparseMaximum(Primitive):
    """
    return a sparse tensor representation max element of two input sparse tensor.

    Inputs:
        - **x1_indices**  - A 2-D Tensor, type int64, represents the position of the element in the x1 sparse tensor.
          each element value should be a non-negative int number. the shape of which should be :math:`(n1, m,)`.
        - **x1_values**  - A 1-D Tensor, represents the value corresponding to the position in the `indices`.
          the shape of which should be :math:`(n1,)`.
        - **x1_shape**  - A 1-D Tensor, type int64, which specifies the shape of x1 sparse tensor.
          the shape of which should be :math:`(m,)`.
        - **x2_indices**  - A 2-D Tensor, type int64, represents the position of the element in the x2 sparse tensor.
          each element value should be a non-negative int number. the shape of which should be :math:`(n2, m,)`.
        - **x2_values**  - A 1-D Tensor, represents the value corresponding to the position in the `indices`.
          the shape of which should be :math:`(n2,)`.
        - **x2_shape**  - A 1-D Tensor, type int64, which specifies the shape of x2 sparse tensor.
          the shape of which should be :math:`(m,)`.

    Returns:
        - **y_indices**  - A 2-D Tensor, type int64. It represents the position of the element-wise max of
          two input tensors.
        - **y_values**  - A 1-D Tensor. It represents the value corresponding to the position
          in the `y_indices`. Has the same type as x1_values.

    Raises:
        TypeError: If the dtype of `x1_indices`, `x2_indices`, `x1_indices` and `x2_indices` isn't int64.
        TypeError: If the dtype of `x1_values` and `x2_values` isn't support.
        TypeError: If the dtype of `x1_values` and `x2_values` isn't same.
        TypeError: If the input is not tensor.
        ValueError: If x1_indices.shape[0] and x1_values.shape[0] isn't same.
        ValueError: If x2_indices.shape[0] and x2_values.shape[0] isn't same.
        ValueError: If x1_indices.shape[1] and x1_shape.shape[0] isn't same.
        ValueError: If x2_indices.shape[0] and x2_values.shape[0] isn't same.
        ValueError: If the `x1_shape` and `x2_shape` mismatch with each other.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1_indices = Tensor([[0, 1], [1, 2]])
        >>> x1_values = Tensor([1, 2], dtype=ms.float32)
        >>> x1_shape = Tensor([3, 3])
        >>> x2_indices = Tensor([[0, 1], [1, 1]])
        >>> x2_values = Tensor([3, 4], dtype=ms.float32)
        >>> x2_shape = Tensor([3, 3])
        >>> SparseSparseMaximum = ops.SparseSparseMaximum()
        >>> y_indices, y_values = SparseSparseMaximum(x1_indices, x1_values, x1_shape, x2_indices, x2_values, x2_shape)
        >>> print(y_indices)
        [[0. 1.]
         [1. 1.]
         [1. 2.]]
        >>> print(y_values)
        [3. 4. 2.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseSparseMaximum."""
        self.init_prim_io_names(inputs=['x1_indices', 'x1_values', 'x1_shape', 'x2_indices', 'x2_values', 'x2_shape'],
                                outputs=['y_indices', 'y_values'])


class SetSize(Primitive):
    """
     Number of unique elements along last dimension of input set.

    Args:
        validate_indices (bool): If true, sparse tensor is transposed before multiplication. Default: True.

    Inputs:
        - **set_indices** (Tensor) - A 2-D Tensor, represents the position of the element in the sparse tensor.
          Support int64, each element value should be a non-negative int number. The shape is :math:`(n, 2)`.
        - **set_values** (Tensor) - A 1-D Tensor, represents the value corresponding to the position in
          the `set_indices`. Support int8, int16, int32, int64, uint8, uint16, string, the shape should
          be :math:`(n,)`.
        - **set_shape** (Tensor) - A 1-D Tensor, represents the shape of a SparseTensor,
          Support int64, the shape should be :math:`(n,)`.

    Outputs:
        Tensor. The dtype is int32, and the shape is set_shape[0:-1].

    Raises:
        TypeError: If the type of inputs is not Tensor.
        TypeError: If the type of `set_values` is not one of the following dtype: int8, int16, uint8, uint16,
            int32, int64.
        TypeError: If the type of `validate_indices` is not bool, or the dtype of `set_indices` and `set_shape`
            is ont int64.
        ValueError: If the shape of `set_shape`, shape of `set_indices` and shape of `set_values` don't meet the
            parameter description.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> set_indices = Tensor(np.array([[0, 1], [1, 2]]).astype(np.int64))
        >>> set_values = Tensor(np.array([1, 2]).astype(np.int64))
        >>> set_shape = Tensor(np.array([3, 4]).astype(np.int64))
        >>> setsize = op.SetSize()
        >>> out = setsize(set_indices, set_values, set_shape)
        >>> print(out)
        [1 1 0]
    """

    @prim_attr_register
    def __init__(self, validate_indices=True):
        """Initialize SetSize."""
        self.validate_indices = validate_indices
        validator.check_bool(validate_indices, "validate_indices", self.name)
        self.init_prim_io_names(inputs=['set_indices', 'set_values', 'set_shape'],
                                outputs=['size'])
        self.add_prim_attr("validate_indices", self.validate_indices)
        self.add_prim_attr("max_length", 1000)


class SparseReorder(Primitive):
    """
    Reorders a SparseTensor into the canonical, row-major ordering

    Inputs:
        - **indices** (Tensor) - A 2-D Tensor, represents the position of the element in the sparse tensor.
          Support int64, each element value should be a non-negative int number.The shape is :math:`(n, d)`.
        - **values** (Tensor) - A 1-D Tensor, represents the value corresponding to the position in the `indices`.
          The shape should be :math:`(n,)`.
        - **shape** (Tensor) - A 1-D Tensor, represents the shape corresponding to the position in the `indices`.
          Support int64, each element value should be a non-negative int number.The shape should be :math:`(d,)`.
    Outputs:
        - **y_indices** (Tensor) - Has the same type as "indices".
        - **y_values** (Tensor) -  Has the same type as "values" .

    Raises:
        TypeError: If `indices` or `shape` is not tensor or its dtype is not int64.
        TypeError: If `values` is not tensor or its dtype is incorrect.
        ValueError: If the index exceeds the bounds. (Raise RuntimeError if on GPU Platform)
        ValueError: If the size of `indices` tensor shape is not equal to 2.
        ValueError: If the size of `values` or `shape` tensor shape is not equal to 1.
        ValueError: If `values` the first dimension length is not equal the first dimension length of 'indices'.
        ValueError: If `shape` the first dimension length is not equal the second dimension length of 'indices'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.common.dtype as ms
        >>> from mindspore import Tensor
        >>> import mindspore.ops.operations.sparse_ops as op
        >>> indices = Tensor([[2, 1], [0, 1]], dtype=ms.int64)
        >>> values = Tensor([1, 2], dtype=ms.int16)
        >>> shape = Tensor([3,3], dtype=ms.int64)
        >>> sparse_reorder = op.SparseReorder()
        >>> y_indices,y_values = sparse_reorder(indices, values, shape)
        >>> print(y_indices)
        [[0 1]
         [2 1]]
        >>> print(y_values)
        [2 1]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseReorder."""
        self.init_prim_io_names(inputs=['indices', 'values', 'shape'], outputs=['y_indices', 'y_values'])


class SparseToDense(Primitive):
    """
    Converts a sparse representation into a dense tensor.

    Inputs:
        - **indices** (Tensor) - A 2-D Tensor, represents the position of the element in the sparse tensor.
          Support int32, int64, each element value should be a non-negative int number. The shape is :math:`(n, 2)`.
        - **values** (Tensor) - A 1-D Tensor, represents the value corresponding to the position in the `indices`.
          The shape should be :math:`(n,)`.
        - **sparse_shape** (tuple(int)) - A positive int tuple which specifies the shape of sparse tensor,
          should have 2 elements, represent sparse tensor shape is :math:`(N, C)`.

    Outputs:
        Tensor, converted from sparse tensor. The dtype is same as `values`, and the shape is `sparse_shape`.

    Raises:
        TypeError: If the dtype of `indices` is neither int32 nor int64.
        ValueError: If `sparse_shape`, shape of `indices` and shape of `values` don't meet the parameter description.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]])
        >>> values = Tensor([1, 2], dtype=mindspore.float32)
        >>> sparse_shape = (3, 4)
        >>> sparse_to_dense = ops.SparseToDense()
        >>> out = sparse_to_dense(indices, values, sparse_shape)
        >>> print(out)
        [[0. 1. 0. 0.]
         [0. 0. 2. 0.]
         [0. 0. 0. 0.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseToDense."""
        self.init_prim_io_names(
            inputs=['indices', 'values', 'dense_shape'], outputs=['output'])


class SparseToDenseV2(Primitive):
    """
    Converts a sparse representation into a dense tensor.

    Args:
        validate_indices (bool): If true, indices are checked to make sure they are sorted in
                                 lexicographic order and that there are no repeats. Default: True.

    Inputs:
        - **indices** (Tensor) - A 0D, 1D, or 2D Tensor of type int32 or int64, represents the position
          of the element in the sparse tensor.
        - **output_shape** (Tensor) - A 1D Tensor of the same type as `indices`, represents the shape
          of the dense output tensor.
        - **values** (Tensor) - A 1D Tensor, represents the value corresponding to the position in the `indices`
          or a scalar value to be used for all indices.
        - **default_value** (Tensor) - A 0D Tensor of the same type as `sparse_values`, scalar value to
          set for indices not specified in indices.

    Returns:
        Tensor, converted from sparse tensor. The dtype is same as `values`, and the shape is `output_shape`.

    Raises:
        TypeError: If the dtype of `indices` is neither Int32 nor Int64.
        TypeError: If the dtype of `outputshape` is neither Int32 nor Int64.
        ValueError: If the shape of `output_shape`, shape of `indices`,
            shape of `default_value` and shape of `values` don't meet the parameter description.
        ValueError: If each Element of `output_shape` is not > 0.
        ValueError: If the shape[0] of `indices` don't match with the element of `values`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=ms.int32)
        >>> output_shape = Tensor([3, 4], dtype=ms.int32)
        >>> values = Tensor([1, 2], dtype=ms.float32)
        >>> default_value = Tensor(0, dtype=ms.float32)
        >>> sparse_to_dense_v2 = ops.SparseToDenseV2()
        >>> out = sparse_to_dense_v2(indices, output_shape, values, default_value)
        >>> print(out)
        [[0. 1. 0. 0.]
         [0. 0. 2. 0.]
         [0. 0. 0. 0.]]
    """

    @prim_attr_register
    def __init__(self, validate_indices=True):
        """Initialize SparseToDenseV2."""
        self.add_prim_attr("max_length", 1000000)
        self.validate_indices = validate_indices
        self.add_prim_attr("validate_indices", self.validate_indices)
        self.init_prim_io_names(
            inputs=['indices', 'output_shape', 'values', 'default_value'], outputs=['output'])


class SparseSoftmax(Primitive):
    """
    Similar to softmax but with the catch that the implicitly zero elements do not participate.

    Inputs:
        - **indices** (Tensor) - A 2-D Tensor, represents the position of the element in the sparse tensor.
          Support int64, each element value should be a non-negative int number. The shape is :math:`(n, m)`.
        - **values** (Tensor) - A 1-D Tensor, represents the value corresponding to the position in the `indices`.
          The shape should be :math:`(n,)`.
        - **shape** (Tensor) - A 1-D Tensor, represents the shape of sparse tensor,
          should have 2 or more than 2 elements, represent sparse tensor shape is :math:`(N, ... , C)`.

    Returns:
        Tensor, calculated from sparse tensor. The dtype is same as `values`, and the shape is same as `values`.

    Raises:
        TypeError: If the dtype of `indices` or `shape` is not int64.
        TypeError: If the dtype of `values` is neither float32 nor float64.
        ValueError: If the shape[0] of indices isn't equal to size of values.
        ValueError: If the shape[1] of indices isn't equal to size of shape.
        ValueError: If the indices is not 2D.
        ValueError: If the values is not 1D.
        ValueError: If the shape is not 1D.
        ValueError: If the size of shape < 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0,0], [0,3], [1,2], [1,5], [2,0], [2,5]])
        >>> values = Tensor([1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ], dtype=ms.float64)
        >>> shape = Tensor([6, 6])
        >>> sparsesoftmax = ops.SparseSoftmax()
        >>> out = sparsesoftmax(indices, values, shape)
        >>> print(out)
        [0.26894142 0.73105858 0.26894142 0.73105858 0.26894142 0.73105858]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseSoftmax."""
        self.init_prim_io_names(inputs=['indices', 'values', 'shape'], outputs=['output'])


class SparseTensorDenseAdd(Primitive):
    """
    Add a sparse tensor and a dense tensor to get a dense tensor.

    Inputs:
        - **x1_indices** (Tensor) - A 2-D Tensor, represents the position of the element in the sparse tensor.
          Support int32, int64, each element value should be a non-negative int number. The shape is :math:`(n, ndim)`.
        - **x1_values** (Tensor) - A 1-D Tensor, represents the value corresponding to the position in the `indices`.
          The shape should be :math:`(n,)`.
        - **x1_shape** (tuple(int)) - A positive int tuple which specifies the shape of sparse tensor,
          should have ndim elements, represent sparse tensor shape is :math:`(ndim,)`.
        - **x2** (Tensor) - A dense Tensor, the dtype is same as `values`.

    Outputs:
        Tensor, add result of sparse tensor and dense tensor. The dtype is same as `values`,
        and the shape is `x1_shape`.

    Raises:
        TypeError: If the dtype of `x1_indices` and 'x1_shape' is neither int32 nor int64.
        ValueError: If `x1_shape`, shape of `x1_indices`, shape of `x1_values` and shape
                    of 'x2' don't meet the parameter description.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> import mindspore.ops as ops
        >>> from mindspore.common import dtype as mstype
        >>> x1_indices = Tensor([[0, 0], [0, 1]], dtype=mstype.int64)
        >>> x1_values = Tensor([1, 1], dtype=mstype.float32)
        >>> x1_shape = Tensor([3, 3], dtype=mstype.int64)
        >>> x2= Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=mstype.float32)
        >>> sparse_tensor_dense_add = ops.SparseTensorDenseAdd()
        >>> out = sparse_tensor_dense_add(x1_indices, x1_values, x1_shape, x2)
        >>> print(out)
        [[2. 2. 1.]
         [1. 1. 1.]
         [1. 1. 1.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseTensorDenseAdd."""
        self.init_prim_io_names(
            inputs=['x1_indices', 'x1_values', 'x1_shape', 'x2'], outputs=['y'])


class SparseTensorDenseMatmul(Primitive):
    """
    Multiplies sparse matrix `A` by dense matrix `B`.
    The rank of sparse matrix and dense matrix must be equal to `2`.

    Args:
        adjoint_st (bool): If true, sparse tensor is transposed before multiplication. Default: False.
        adjoint_dt (bool): If true, dense tensor is transposed before multiplication. Default: False.

    Inputs:
        - **indices** (Tensor) - A 2-D Tensor, represents the position of the element in the sparse tensor.
          Support int32, int64, each element value should be a non-negative int number. The shape is :math:`(n, 2)`.
        - **values** (Tensor) - A 1-D Tensor, represents the value corresponding to the position in the `indices`.
          Support float16, float32, float64, int32, int64, complex64, complex128. The shape should be :math:`(n,)`.
        - **sparse_shape** (tuple(int) or (Tensor)) - A positive int tuple or tensor which specifies the shape of
          sparse tensor, and only constant value is allowed when sparse_shape is a tensor, should have 2 elements,
          represent sparse tensor shape is :math:`(N, C)`.
        - **dense** (Tensor) - A 2-D Tensor, the dtype is same as `values`.
          If `adjoint_st` is False and `adjoint_dt` is False, the shape must be :math:`(C, M)`.
          If `adjoint_st` is False and `adjoint_dt` is True, the shape must be :math:`(M, C)`.
          If `adjoint_st` is True and `adjoint_dt` is False, the shape must be :math:`(N, M)`.
          If `adjoint_st` is True and `adjoint_dt` is True, the shape must be :math:`(M, N)`.

    Outputs:
        Tensor, the dtype is the same as `values`.
        If `adjoint_st` is False, the shape is :math:`(N, M)`.
        If `adjoint_st` is True, the shape is :math:`(C, M)`.

    Raises:
        TypeError: If the type of `adjoint_st` or `adjoint_dt` is not bool, or the dtype of `indices`,
            dtype of `values` and dtype of `dense` don't meet the parameter description.
        ValueError: If `sparse_shape`, shape of `indices`, shape of `values`,
            and shape of `dense` don't meet the parameter description.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore.ops import operations as ops
        >>> from mindspore.common import dtype as mstype
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mindspore.int32)
        >>> values = Tensor([1, 2], dtype=mindspore.float32)
        >>> sparse_shape = (3, 4)
        >>> dense = Tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=mindspore.float32)
        >>> sparse_dense_matmul = ops.SparseTensorDenseMatmul()
        >>> out = sparse_dense_matmul(indices, values, sparse_shape, dense)
        >>> print(out)
        [[2. 2.]
         [6. 6.]
         [0. 0.]]
    """

    @prim_attr_register
    def __init__(self, adjoint_st=False, adjoint_dt=False):
        """Initialize SparseTensorDenseMatmul"""
        self.adjoint_st = adjoint_st
        self.adjoint_dt = adjoint_dt
        self.init_prim_io_names(inputs=['indices', 'values', 'sparse_shape', 'dense'],
                                outputs=['output'])
        self.add_prim_attr('adjoint_a', self.adjoint_st)
        self.add_prim_attr('adjoint_b', self.adjoint_dt)
        validator.check_value_type("adjoint_st", adjoint_st, [bool], self.name)
        validator.check_value_type("adjoint_dt", adjoint_dt, [bool], self.name)
        self.set_const_input_indexes([2])


class CSRSparseMatrixToSparseTensor(Primitive):
    """
    Converts a CSR sparse matrix(maybe batched) to its sparse tensor form.

    Inputs:
        - **x_dense_shape** (Tensor) - A 1-D Tensor. It represents the dense form shape of
          the input CSR sparse matrix, the shape of which should be :math:`(2,)` or :math:`(3,)`.
        - **x_batch_pointers** (Tensor) - A 1-D Tensor. Supposing the input CSR sparse matrix is of
          batch size `n`, it should have shape :math:`(n+1,)`, while the `i`-th element of which stores
          acummulated counts of non-zero values of the first `i - 1` batches.
        - **x_row_pointers** (Tensor) - A 1-D Tensor. Supposing the input CSR sparse matrix is of
          batch size `n` and row number `m`, it can be divided into `n` parts, each part of length
          `m + 1`. The `i`-th element of each :math:`(m+1,)` vector stores acummulated counts of
          non-zero values of the first `i - 1` rows in the corresponding batch.
        - **x_col_indices** (Tensor) - A 1-D Tensor. It represents column indices of the non-zero values
          in the input CSR sparse matrix.
        - **x_values** (Tensor) - A 1-D Tensor. It represents all the non-zero values in the
          input CSR sparse matrix.

    Outputs:
        - **indices** (Tensor) - A 2-D Tensor. It represents the position of the non-zero element
          in the sparse tensor.
        - **values** (Tensor) - A 1-D Tensor. It represents the value corresponding to the position
          in the `indices`, the shape of which should be :math:`(N,)`.
        - **dense_shape** (Tensor) - A 1-D Tensor. It represents the dense form shape of
          the sparse tensor. Its shape should be :math:`(2,)` or :math:`(3,)`.

    Raises:
        TypeError: If the dtype of `x_dense_shape` or `x_batch_pointers` or `x_row_pointers` or
                   `x_col_indices` is not int32 or int64.
        TypeError: If the dtype of `x_values` is not float32, float64, complex64 or complex128.
        ValueError: If `x_dense_shape` or `x_batch_pointers` or `x_row_pointers` or `x_values` or
                   `x_dense_shape` is not a 1-D tensor.
        ValueError: If rank of `x_dense_shape` is not 2 or 3.
        ValueError: If shape of `x_col_indices` is not corresponding to shape of `x_values`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.ops.operations.sparse_ops import CSRSparseMatrixToSparseTensor
        >>> x_dense_shape = Tensor(np.array([2, 2, 4]).astype(np.int64))
        >>> x_batch_pointers = Tensor(np.array([0, 3, 6]).astype(np.int64))
        >>> x_row_pointers = Tensor(np.array([0, 1, 3, 0, 1, 3]).astype(np.int64))
        >>> x_col_indices = Tensor(np.array([1, 2, 3, 1, 2, 3]).astype(np.int64))
        >>> x_values = Tensor(np.array([1, 4, 3, 1, 4, 3]).astype(np.float32))
        >>> csr_sparse_matrix_to_sparse_tensor = ops.CSRSparseMatrixToSparseTensor()
        >>> out = csr_sparse_matrix_to_sparse_tensor(x_dense_shape, x_batch_pointers, x_row_pointers,
        ...                                          x_col_indices, x_values)
        >>> print(out[0])
        [[0 0 1]
         [0 1 2]
         [0 1 3]
         [1 0 1]
         [1 1 2]
         [1 1 3]]
        >>> print(out[1])
        [1. 4. 3. 1. 4. 3.]
        >>> print(out[2])
        [2 2 4]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CSRSparseMatrixToSparseTensor."""
        self.init_prim_io_names(inputs=['x_dense_shape', 'x_batch_pointers', 'x_row_pointers',
                                        'x_col_indices', 'x_values'],
                                outputs=['indices', 'values', 'dense_shape'])


class DenseToCSRSparseMatrix(Primitive):
    """
    Converts a dense matrix(maybe batched) to its CSR sparse form.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Inputs:
        - **dense_input** (Tensor) - A 2-D or 3-D Tensor. It represents the input dense matrix.
        - **indices** (Tensor) - A 2-D Tensor. It represents indices of all the nonzero elements.

    Outputs:
        - **y_dense_shape** (Tensor) - A 1-D Tensor. It represents the dense form shape of
          the output CSR sparse matrix, the shape of which should be :math:`(2,)` or :math:`(3,)`.
        - **y_batch_pointers** (Tensor) - A 1-D Tensor. Supposing the output CSR sparse matrix is of
          batch size `n`, it should have shape :math:`(n+1,)`, while the `i`-th element of which stores
          acummulated counts of nonzero values of the first `i - 1` batches.
        - **y_row_pointers** (Tensor) - A 1-D Tensor. Supposing the output CSR sparse matrix is of
          batch size `n` and row number `m`, it can be divided into `n` parts, each part of length
          `m + 1`. The `i`-th element of each :math:`(m+1,)` vector stores acummulated counts of
          nonzero values of the first `i - 1` rows in the corresponding batch.
        - **y_col_indices** (Tensor) - A 1-D Tensor. It represents column indices of the nonzero values
          in the output CSR sparse matrix.
        - **y_values** (Tensor) - A 1-D Tensor. It represents all the nonzero values in the
          output CSR sparse matrix.

    Raises:
        TypeError: If the dtype of `indices` is not int32 or int64.
        TypeError: If the dtype of `dense_input` is not float32, float64, complex64 or complex128.
        ValueError: If either of the inputs is not a tensor.
        ValueError: If rank of `dense_input` is not 2 or 3.
        ValueError: If rank of `indices` is not 2.
        ValueError: If shape[1] of `indices` and rank of `dense_input` is not the same.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([[[1., 0.], [0., 2.]]], dtype=mindspore.float32)
        >>> indices = Tensor([[0, 0, 0], [0, 1, 1]], dtype=mindspore.int32)
        >>> dense_to_csr = ops.DenseToCSRSparseMatrix()
        >>> out = dense_to_csr(x, indices)
        >>> print(out[0])
        [1 2 2]
        >>> print(out[1])
        [0 2]
        >>> print(out[2])
        [0 1 2]
        >>> print(out[3])
        [0 1]
        >>> print(out[4])
        [1. 2.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize DenseToCSRSparseMatrix"""
        self.init_prim_io_names(
            inputs=['dense_input', 'indices'],
            outputs=['y_dense_shape', 'y_batch_pointers', 'y_row_pointers', 'y_col_indices', 'y_values'])


class DenseToDenseSetOperation(Primitive):
    """
    Applies set operation along last dimension of 2 `Tensor` inputs.
    Iterate over groups in set x1 and set x2, applying `ApplySetOperation` to each,
    and outputting the result `SparseTensor`. A "group" is a collection of values
    with the same first n-1 dimensions in x1 and x2.

    Args:
        set_operation (str): The type of set operation, case insensitive. Default:"a-b".
            "a-b": Get the difference set of x1 to x2.
            "b-a": Get the difference set of x2 to x1.
            "intersection": Get the intersection set of x2 to x1.
            "union": Get the union set of x2 to x1.
        validate_indices (bool): Optional attributes for DenseToDenseSetOperation.  Default: True.

    Inputs:
        - **x1** (Tensor) - The input tensor `x1` with rank `n`. 1st `n-1` dimensions must be the same as `x2`.
          Dimension `n` contains values in a set, duplicates are allowed but ignored.
        - **x2** (Tensor) - The input tensor `x2` with rank `n`. 1st `n-1` dimensions must be the same as `x1`.
          Dimension `n` contains values in a set, duplicates are allowed but ignored.

    Outputs:
        - **y_indices** (Tensor) - A 2-D Tensor of type int64, represents the position of the element
          in the sparse tensor.
        - **y_values** (Tensor) - A 1-D Tensor, represents the value corresponding to the position
          in the `y_indices`. The dtype is same as input.
        - **y_shape** (Tensor) - A 1-D Tensor of type int64, represents the shape of sparse tensor.
          `y_shape[0...n-1]` is the same as the 1st `n-1` dimensions of `x1` and `x2`,
          `y_shape[n]` is the max result set size across all `0...n-1` dimensions.

    Raises:
        TypeError: If input `x1` or `x2` is not Tensor.
        TypeError: If the type of `x1` is not the same as `x2`.
        ValueError: If the group shape of `x1` or `x2` mismatch with each other.
        ValueError: If the rank of `x1` or `x2` is less than 2.
        ValueError: If the value of attr set_operation is not a valid value.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x1 = Tensor([[2, 2, 0], [2, 2, 1], [0, 2, 2]], dtype=mstype.int32)
        >>> x2 = Tensor([[2, 2, 1], [0, 2, 0], [0, 1, 1]], dtype=mstype.int32)
        >>> dtod=P.DenseToDenseSetOperation(set_operation="a-b",validate_indices=True)
        >>> res=dtod(x1,x2)
        >>> print(res[0])
        [[0 0]
         [1 0]
         [2 0]]
        >>> print(res[1])
        [0 1 2]
        >>> print(res[2])
        [3 1]
    """

    @prim_attr_register
    def __init__(self, set_operation="a-b", validate_indices=True):
        """Initialize DenseToDenseSetOperation."""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=[
            'y_indices', 'y_values', 'y_shape'])
        validator.check_value_type(
            "set_operation", set_operation, [str], self.name)
        validator.check_value_type(
            "validate_indices", validate_indices, [bool], self.name)


class Sspaddmm(Primitive):
    r"""
    Matrix multiplies a sparse tensor `x2` with a dense tensor `x3`, then adds the sparse tensor `x1`.
    If `x1_shape` is :math:`(s0, s1)`, `x2_shpae` should be :math:`(s0, s2)`, the `x3_shape` should be :math:`(s2, s1)`.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    .. math::
        out =\beta * x1  + \alpha * (x2 @ x3),

    Inputs:
        - **x1_indices** (Tensor) - A 2-D Tensor, represents the position of the element in the sparse tensor.
          Support int32, int64. The shape is :math:`(2, n)`.  If `x1_shape` is :math:`(s0, s1)`, the row index
          value of `x1_indices` should be a non-negative and less than `s0` int number, the col index value of
          `x1_indices` should be a non-negative and less than `s1` int number.
        - **x1_values** (Tensor) - A 1-D Tensor, represents the value corresponding to the position in
          the `x1_indices`. Support float32, float64, int8, int16, int32, int64, uint8. The dtype should be the same as
          `x2_values` and `x3_dense`. The shape should be :math:`(n,)`.
        - **x1_shape** (Tensor) - A 1-D Tensor, specifies the shape of sparse tensor. Support int32, int64,
          have 2 positive int elements, shape is :math:`(2,)`. The dtype should be the same as `x1_indices`.
        - **x2_indices** (Tensor) - A 2-D Tensor, represents the position of the element in the sparse tensor.
          Support int32, int64. The shape is :math:`(2, n)`. If `x2_shape` is :math:`(s0, s2)`, the row index
          value of `x2_indices` should be a non-negative and less than `s0` int number, the col index value of
          `x2_indices` should be a non-negative and less than `s2` int number.
        - **x2_values** (Tensor) - A 1-D Tensor, represents the value corresponding to the position in the `x2_indices`.
          Support float32, float64, int8, int16, int32, int64, uint8. The dtype should be the same as `x1_values`
          and `x3_dense`. The shape should be :math:`(n,)`.
        - **x2_shape** (Tensor) - A 1-D Tensor, specifies the shape of sparse tensor. Support int32,int64,
          have 2 positive int elements, shape is :math:`(2,)`. The dtype is same as `x2_indices`.
        - **x3_dense** (Tensor) - A 2-D Tensor, the dtype should be the same as `x2_values` and `x3_dense`.
        - **alpha** (Tensor) - A 0-D or 1-D Tensor, the weight of x1. If alpha is 1-D tensor,
          the shape should be :math:`()` otherwise the shape is :math:`(1,)`. Support uint8, uint16, uint32, uint64,
          int8, int16, int32, int64, float16, float32, float64. If the dtype of alpha is not the same with expected
          output dtype, alpha value should be convert without overflow.
        - **beta** (Tensor) - A 0-D or 1-D, the weight of x2@x3. If alpha is 1-D tensor,
          the shape should be :math:`()` otherwise the shape is :math:`(1,)`. Support uint8, uint16, uint32, uint64,
          int8, int16, int32, int64, float16, float32, float64. If the `x1_values` dtype is byte, char, short, int,
          long, the dtype of beta doesn't support float16, float32, float64.

    Outputs:
        - **y_indices** (Tensor) - A 2-D Tensor, represents the position of the element in the sparse tensor.
          The dtype is int64, each element value should be a non-negative int number. The shape is :math:`(2, n)`.
        - **y_values** (Tensor) - A 1-D Tensor, represents the value corresponding to the position in the `y_indices`.
          The dtype is the same as `x1_values` . The shape should be :math:`(n,)`.
        - **y_shape** (Tensor) - A 1-D Tensor, A positive int tuple which specifies the shape of sparse tensor.
          The dtype is int64, the values is the same as `x1_shape`.

    Raises:
        TypeError: If dtype of `x1_indices`, `x1_shape` is not the same and neither int32 nor int64.
        TypeError: If dtype of `x2_indices`, `x2_shape` is not the same and not int32 or int64.
        TypeError: If type of `x1_values`, `x2_values`, `x3_dense` is not the same.
        TypeError: If dtype of `x1_values`, `x2_values`, `x3_dense` is not uint8, int8, int16, int32, int64, float32,
                   float64.
        ValueError: If shape of `x1_indices`, `x2_indices` is not (2, n).
        ValueError: If shape of `x1_values`, `x2_values` is not (n,).
        ValueError: If dim0 size of `x1_values` is not the same with dim1 size of `x1_indices`.
        ValueError: If dim0 size of `x2_values` is not the same with dim1 size of `x2_indices`.
        ValueError: If shape of `x1_shape` or shape of `x2_shape` is not (2,).
        ValueError: If dim of `x3_dense` is not 2D.
        ValueError: If dtype of `alpha` is not the same with `x2_values` dtype, and alpha value convert to the
                    `x2_values` dtype overflow.
        TypeError: If dtype of `alpha`, `beta` is not uint8, uint16, uint32, uint64, int8, int16, int32, int64,
                   float16, float32, float64.
        TypeError: If the `x1_values` dtype is byte, char, short, int, long, while the dtype of beta is float16,
                   float32 or float64.
        ValueError: If the shape of `alpha`, `beta` is not () or (1,).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1_indices = Tensor(np.array([[0, 1], [0, 1]]), mstype.int64)
        >>> x1_values = Tensor(np.array([1, 2]), mstype.int32)
        >>> x1_shape = Tensor(np.array([3, 3]), mstype.int64)
        >>> x2_indices = Tensor(np.array([[0, 1], [2, 2]]), mstype.int64)
        >>> x2_values = Tensor(np.array([3, 4]), mstype.int32)
        >>> x2_shape = Tensor(np.array([3, 3]), mstype.int64)
        >>> x3_dense = Tensor(np.array([[1, 2, 3], [1, 3, 2], [3, 2, 1]]), mstype.int32)
        >>> alpha = Tensor(np.array(1), mstype.int32)
        >>> beta = Tensor(np.array(1), mstype.int32)
        >>> sspaddmm = ops.Sspaddmm()
        >>> out_indices, out_values, out_shapes = sspaddmm(x1_indices, x1_values, x1_shape,
        ... x2_indices, x2_values, x2_shape, x3_dense, alpha, beta)
        >>> print(out_indices)
        [[0 1 0 0 0 1 1 1]
         [0 1 0 1 2 0 1 2]]
        >>> print(out_values)
        [ 1  2  9  6  3 12  8  4]
        >>> print(out_shapes)
        [3 3]
    """
    __mindspore_signature__ = (
        sig.make_sig('x1_indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('x1_values', dtype=sig.sig_dtype.T),
        sig.make_sig('x1_shape', dtype=sig.sig_dtype.T2),
        sig.make_sig('x2_indices', dtype=sig.sig_dtype.T3),
        sig.make_sig('x2_values', dtype=sig.sig_dtype.T),
        sig.make_sig('x2_shape', dtype=sig.sig_dtype.T4),
        sig.make_sig('x3_dense', dtype=sig.sig_dtype.T),
        sig.make_sig('alpha', dtype=sig.sig_dtype.T),
        sig.make_sig('beta', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize Sspaddmm."""
        self.init_prim_io_names(inputs=['x1_indices', 'x1_values', 'x1_shape', 'x2_indices', 'x2_values', 'x2_shape',
                                        'x3_dense', 'alpha', 'beta'], outputs=['y_indices', 'y_values', 'y_shape'])


class SparseAddmm(Primitive):
    """
    Multiplies sparse matrix `x1` by dense matrix `x2` * `alpha` and add dense matrix `x3` * `beta`.
    The rank of sparse matrix and dense matrix must equal to `2`. The sparse matrix `x1` is formulated by `x1_indices`,
    `x1_values` and `x1_shape`.

    Inputs:
        - **x1_indices** (Tensor) - A 2-D Tensor, represents the position of the element in the sparse tensor.
          Support int32, int64, each element value should be a non-negative int number. The shape is :math:`(N, 2)`.
        - **x1_values** (Tensor) - A 1-D Tensor, represents the value corresponding to the position in the `indices`.
          Support float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64.
          The shape should be :math:`(N,)`.
        - **x1_shape** (Tensor) - A positive int tuple which specifies the shape of sparse tensor.
          Support int32, int64, should have 2 elements, represent sparse tensor shape is :math:`(Q, P)`.
        - **x2** (Tensor) - A 2-D Dense Tensor, the dtype is same as `values`. The shape should be :math:`(P, M)`.
        - **x3** (Tensor) - A 2-D Dense Tensor, the dtype is same as `values`. The shape should be :math:`(Q, M)`.
        - **alpha** (Tensor) - A 1-D Tensor, the dtype is same as `values`. The shape should be :math:`(1,)`.
        - **beta** (Tensor) - A 1-D Tensor, the dtype is same as `values`. The shape should be :math:`(1,)`.

    Outputs:
        Tensor, the dtype is the same as `x1_values`. The shape is the same as `x3`.

    Raises:
        TypeError: If dtype of `x1_indices`, dtype of `x1_values` and dtype of `dense` don't meet the parameter
                   description.
        ValueError: If shape of `x1_indices`, shape of `x1_values`, shape of `alpha`,
                    and shape of `beta` don't meet the parameter description.
        RuntimeError: If `x1_shape`, shape of `x2`, shape of `x3` don't meet the parameter description.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=ms.int32)
        >>> values = Tensor([1, 2], dtype=ms.float32)
        >>> sparse_shape = Tensor([1, 2], dtype=ms.int32)
        >>> x2_dense = Tensor([[1,1], [2,2], [3,3], [4,4]], dtype=ms.float32)
        >>> x3_dense = Tensor([[2,2], [6,6], [0,0]], dtype=ms.float32)
        >>> alpha = Tensor([1], dtype=ms.float32)
        >>> beta = Tensor([1], dtype=ms.float32)
        >>> sparse_addmm = ops.SparseAddmm()
        >>> out = sparse_addmm(indices, values, sparse_shape, x2_dense, x3_dense, alpha, beta)
        >>> print(out)
        [[4 4]
         [12 12]
         [0 0]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseAddmm"""
        self.init_prim_io_names(inputs=['indices', 'values', 'sparse_shape', 'x2_dense', 'x3_dense', 'alpha', 'beta'],
                                outputs=['output'])


class SparseConcat(Primitive):
    """
    concatenates the input SparseTensor(COO format) along the specified dimension.

    Args:
        concat_dim(Scalar) - A Scalar, decide the dimension to concatenation along.
        The value must be in range [-rank, rank), where rank is the number of dimensions in each input
        SparseTensor. Support int32, int64. Default: 0.

    Inputs:
        - **sp_input_indices** (Tensor) - the list of Tensor which means COOTensor indices, and Need to
          concatenates. Support int64.
        - **sp_input_values** (Tensor) - the list of Tensor which means COOTensor values, and
          need to concatenates.
        - **sp_input_shape** (Tensor) - the list of Tensor which means COOTensor shape, and
          need to concatenates. Support int64.

    Outputs:
        - **output_indices** (Tensor) - the result of concatenates the input SparseTensor along the
          specified dimension. This is the indices of output COOTensor.
        - **output_values** (Tensor) - the result of concatenates the input SparseTensor along the
          specified dimension. This is the values of output COOTensor.
        - **output_shape** (Tensor) - the result of concatenates the input SparseTensor along the
          specified dimension. This is the shape of output COOTensor.

    Raises:
        ValueError: If only one sparse tensor input.
        Error: If input axis value is not in range [-rank, rank).

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> indices0 = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values0 = Tensor([1, 2], dtype=mstype.int32)
        >>> shape0 = Tensor([3, 4], dtype=mstype.int64)
        >>> indices1 = Tensor([[0, 0], [1, 1]], dtype=mstype.int64)
        >>> values1 = Tensor([3, 4], dtype=mstype.int32)
        >>> shape1 = Tensor([3, 4], dtype=mstype.int64)
        >>> sparse_concat = ops.SparseConcat(0)
        >>> out = sparse_concat((indices0, indices1), (values0, values1), (shape0, shape1))
        >>> print(out)
        (Tensor(shape=[4, 2], dtype=Int64, value=
        [[0, 1],
         [1, 2],
         [3, 0],
         [4, 1]]), Tensor(shape=[4], dtype=Int32, value= [1, 2, 3, 4]), Tensor(shape=[2], dtype=Int64, value= [6, 4]))
    """

    @prim_attr_register
    def __init__(self, concat_dim=0):
        """Initialize SparseConcat."""
        self.init_prim_io_names(inputs=['sp_input_indices', 'sp_input_values', 'sp_input_shapes'],
                                outputs=['output_indices', 'output_values', 'output_shape'])
        validator.check_value_type("concat_dim", concat_dim, [int], self.name)


class SparseSegmentSum(Primitive):
    """
    Computes the sum along sparse segments of a tensor.

    Inputs:
        - **x** (Tensor) - A tensor of the first input of SparseSegmentSum.
        - **indices** (Tensor) - 1-D Tensor of type int32 or int 64 with indices into `x`.
          Has same rank as `segment_ids`. The shape should be :math:`(N,)`.
        - **segment_ids** (Tensor) - 1-D Tensor of type int32 or int64 with indices into the output `y`. Values
          should be sorted and can be repeated. The shape should be :math:`(N,)`.

    Outputs:
        A Tensor. Has the same type as `x` .
        Has same shape as `x`, except for dimension 0 which is the number of segments.

    Raises:
        TypeError: If `x` or `indices` or `segment_ids` is not a tensor.
        TypeError: If the dtype of `indices` and `segment_ids` is not int32 or int64.
        ValueError: If dimension size of `x` less than 1.
        ValueError: If any of `indices` and `segment_ids` is not a 1-D tensor.
        ValueError: If shape[0] of `indices` is not corresponding to shape[0] of `segment_ids`.
        ValueError: If indices in `segment_ids` are not contiguous or do not start from 0.
        ValueError: If `segment_ids` is not sorted.
        ValueError: If `indices` is out of range of x's first dimension.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([[0, 1, 2], [1, 2, 3], [3, 6, 7]], dtype=ms.float32)
        >>> indices = Tensor([0, 1, 2], dtype=ms.int32)
        >>> segment_ids = Tensor([0, 1, 1], dtype=ms.int32)
        >>> sparse_segment_sum = ops.SparseSegmentSum()
        >>> out = sparse_segment_sum(x, indices, segment_ids)
        >>> print(out)
        [[ 0. 1. 2.]
         [ 4. 8. 10.]]
    """
    __mindspore_signature__ = (
        sig.make_sig('x', dtype=sig.sig_dtype.T1),
        sig.make_sig('indices', dtype=sig.sig_dtype.T),
        sig.make_sig('segment_ids', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize SparseSegmentSum"""
        self.init_prim_io_names(inputs=['x', 'indices', 'segment_ids'], outputs=['y'])


class SparseSegmentSumWithNumSegments(Primitive):
    """
    Computes the sum along sparse segments of a tensor, but it is allowed to miss id in segment_ids.

    Inputs:
        - **x** (Tensor) - A Tensor of the first input of SparseSegmentSumWithNumSegments.
        - **indices** (Tensor) - 1-D Tensor with indices into `x`. Must be one of the following types: int32, int64.
          Has same rank as `segment_ids`. The shape should be :math:`(N,)`.
        - **segment_ids** (Tensor) - 1-D Tensor with indices into the output `y`. Must be one of the following types:
          int32, int64. Values should be sorted and can be repeated. The shape should be :math:`(N,)`.
        - **num_segments** (Tensor) - Num_segments indicates the size of the output.
          It should be bigger than the largest id of `segment_ids`.

    Outputs:
        A Tensor. Has the same type as `x` .
        Has same shape as `x`, except for dimension 0 which is the value of `num_segments`.

    Raises:
        TypeError: If `x` or `indices` or `segment_ids` or `num_segments` is not a tensor.
        TypeError: If the dtype of `indices` and `segment_ids` and `num_segments` is not int32 or int64.
        ValueError: If dimension size of `x` less than 1.
        ValueError: If any of `indices` and `segment_ids` is not a 1-D tensor.
        ValueError: If rank of `num_segments` is bigger than 1.
        ValueError: If numelements of `num_segments` is not 1.
        ValueError: If shape[0] of `indices` is not corresponding to shape[0] of `segment_ids`.
        ValueError: If `segment_ids` is not sorted.
        ValueError: If the last number of `segment_ids` is bigger than or equal to `num_segments`.
        ValueError: If `indices` is out of range of x's first dimension.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([[0, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0]], dtype=ms.float16)
        >>> indices = Tensor([0, 2, 1], dtype=ms.int32)
        >>> segment_ids = Tensor([0, 0, 2], dtype=ms.int32)
        >>> num_segments = Tensor([4], dtype=ms.int32)
        >>> sparse_segment_sum_with_num_segments = ops.SparseSegmentSumWithNumSegments()
        >>> output = sparse_segment_sum_with_num_segments(x, indices, segment_ids, num_segments)
        >>> print(output)
        [[1. 1. 1. 0.]
         [0. 0. 0. 0.]
         [0. 1. 1. 0.]
         [0. 0. 0. 0.]]
    """
    __mindspore_signature__ = (
        sig.make_sig('x', dtype=sig.sig_dtype.T1),
        sig.make_sig('indices', dtype=sig.sig_dtype.T),
        sig.make_sig('segment_ids', dtype=sig.sig_dtype.T),
        sig.make_sig('num_segments', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize SparseSegmentSumWithNumSegments"""
        self.init_prim_io_names(inputs=['x', 'indices', 'segment_ids', 'num_segments'], outputs=['y'])


class SparseSegmentSqrtN(Primitive):
    """
    Computes the sum along sparse segments of a tensor divided by the sqrt of N.
    N is the size of the segment being reduced.

    Inputs:
        - **x** (Tensor) - A tensor. It's rank must be more than or equal to one.
        - **indices** (Tensor) - 1-D Tensor with indices into `x`. Must be one of the following types: int32, int64.
          Has same rank as segment_ids. The shape should be :math:`(N,)`.
        - **segment_ids** (Tensor) - 1-D Tensor with indices into the output `y`. Must be one of the following
          types: int32, int64. Values should be sorted and can be repeated. The shape should be :math:`(N,)`.

    Outputs:
        A Tensor. Has the same type as `x` .
        Has same shape as `x`, except for dimension 0 which is the number of segments.

    Raises:
        TypeError: If `x` or `indices` or `segment_ids` is not a tensor.
        TypeError: If the dtype of `x` is not any of the following data types: {float16, float32, float64}.
        TypeError: If the dtype of `indices` is not int32 or int64.
        TypeError: If the dtype of `segment_ids` is not int32 or int64.
        ValueError: If dimension size of `x` is less than 1.
        ValueError: If any of `indices` and `segment_ids` is not a 1-D tensor.
        ValueError: If shape[0] of `indices` is not corresponding to shape[0] of `segment_ids`.
        ValueError: If indices in `segment_ids` are not contiguous or do not start from 0.
        ValueError: If `segment_ids` is not sorted.
        ValueError: If `indices` is out of range of x's first dimension.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]).astype(np.float32))
        >>> indices = Tensor(np.array([0,1,2]).astype(np.int32))
        >>> segment_ids = Tensor(np.array([0,1,2]).astype(np.int32))
        >>> sparse_segment_sqrt_n = SparseSegmentSqrtN()
        >>> output = sparse_segment_sqrt_n(x, indices, segment_ids)
        >>> print(output)
        [[ 1.  2.  3.  4.]
        [ 5.  6.  7.  8.]
        [ 9. 10. 11. 12.]]
    """
    __mindspore_signature__ = (
        sig.make_sig('x', dtype=sig.sig_dtype.T1),
        sig.make_sig('indices', dtype=sig.sig_dtype.T),
        sig.make_sig('segment_ids', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize SparseSegmentSqrtN"""
        self.init_prim_io_names(
            inputs=['x', 'indices', 'segment_ids'], outputs=['y'])


class SparseSegmentSqrtNWithNumSegments(Primitive):
    """
    Computes the sum along sparse segments of a tensor divided by the sqrt of N.
    N is the size of the segment being reduced.
    Like SparseSegmentSqrtN, but allows missing ids in segment_ids.
    If an id is missing, the output tensor at that position will be zeroed.

    Inputs:
        - **x** (Tensor) - A Tensor. It's rank must be more than or equal to one.
        - **indices** (Tensor) - 1-D Tensor with indices into `x`. Must be one of the following types: int32, int64.
          Has same rank as segment_ids. The shape should be :math:`(N,)`.
        - **segment_ids** (Tensor) - 1-D Tensor with indices into the output `y`. Must be one of the following
          types: int32, int64. Values should be sorted and can be repeated. The shape should be :math:`(N,)`.
        - **num_segments** (Tensor) - Num_segments indicates the size of the output.
          It should be bigger than the largest id of `segment_ids`.

    Outputs:
        A Tensor. Has the same type as `x` .
        Has same shape as `x`, except for dimension 0 which is the value of `num_segments`.

    Raises:
        TypeError: If `x` or `indices` or `segment_ids` or `num_segments` is not a tensor.
        TypeError: If the dtype of `x` is not any of the following data types: {float16, float32, float64}.
        TypeError: If the dtype of `indices` and `segment_ids` and `num_segments` is not int32 or int64.
        TypeError: If dtype of `segment_ids` and `indices` mismatch.
        TypeError: If dtype of `num_segments` and `indices` mismatch.
        ValueError: If dimension size of `x` is less than 1.
        ValueError: If any of `indices` and `segment_ids` is not a 1-D tensor.
        ValueError: If rank of `num_segments` is bigger than 1.
        ValueError: If numelements of `num_segments` is not 1.
        ValueError: If the first dimension of `indices` is not equal to the first dimension of `segment_ids`.
        ValueError: If `segment_ids` is not sorted.
        ValueError: If the the largest id of `segment_ids` is bigger than or equal to `num_segments`.
        ValueError: If `indices` is out of range of x's first dimension.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([[0, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0]], dtype=ms.float16)
        >>> indices = Tensor([0, 2, 1], dtype=ms.int32)
        >>> segment_ids = Tensor([0, 1, 2], dtype=ms.int32)
        >>> num_segments = Tensor([4], dtype=ms.int32)
        >>> sparse_segment_sqrt_n_with_num_segments = SparseSegmentSqrtNWithNumSegments()
        >>> output = sparse_segment_sqrt_n_with_num_segments(x, indices, segment_ids, num_segments)
        >>> print(output)
        [[0. 1. 0. 0.]
         [1. 0. 1. 0.]
         [0. 1. 1. 0.]
         [0. 0. 0. 0.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseSegmentSqrtNWithNumSegments"""
        self.init_prim_io_names(
            inputs=['x', 'indices', 'segment_ids', 'num_segemnts'], outputs=['y'])


class SparseMatrixNNZ(Primitive):
    r"""
    Count number of the non-zero elements in sparse matrix or sparse matrixs.
    If the sparse matrix input contains batch dimension, then output dimension will be same with the batch dimension.

    Note:
        It is assumed that all the inputs can form a legal CSR sparse matrix, otherwise this operator won't work.

    Inputs:
        - **x_dense_shape** (Tensor) -  A 1-D Tensor. It represents the dense form shape of
          the input CSR sparse matrix, the shape of which should be :math:`(2,)` or :math:`(3,)`.
        - **x_batch_pointers** (Tensor) - A 1-D Tensor. Supposing the input CSR sparse matrix is of
          batch size `n`, it should have shape :math:`(n+1,)`, while the `i`-th element of which stores
          acummulated counts of nonzero values of the first `i - 1` batches.
        - **x_row_pointers** (Tensor) - A 1-D Tensor. Supposing the input CSR sparse matrix is of
          batch size `n` and row number `m`, it can be divided into `n` parts, each part of length
          `m + 1`. The `i`-th element of each :math:`(m+1,)` vector stores acummulated counts of
          nonzero values of the first `i - 1` rows in the corresponding batch.
        - **x_col_indices** (Tensor) - A 1-D Tensor. It represents column indices of the nonzero values
          in the input CSR sparse matrix.
        - **x_values** (Tensor) - A 1-D Tensor. It represents all the nonzero values in the
          input CSR sparse matrix.

    Outputs:
        Tensor, the dtype is int32.
        If there are n batch within input sparse matrix, the shape is :math:`(n,)`.

    Raises:
        TypeError: If the dtype of `x_dense_shape`, `x_batch_pointers`, `x_row_pointers` or `x_col_indices`
            is not int32 or int64, or the dtypes of above inputs are not the same.
        TypeError: If the dtype of `x_values` is not supported.
        TypeError: If any of the inputs is not a tensor.
        ValueError: If any of the inputs is not 1-D.
        ValueError: If `x_values` and `x_col_indices` have different length.
        ValueError: If shape[0] of `x_dense_shape` is not 2 or 3.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> dense_shape = Tensor([2,3], dtype=mstype.int32)
        >>> batch_pointers = Tensor([0,1], dtype=mstype.int32)
        >>> row_pointers = Tensor([0,1,1], dtype=mstype.int32)
        >>> col_indices = Tensor([0], dtype=mstype.int32)
        >>> values = Tensor([99], dtype=mstype.float32)
        >>> sparse_matrix_nnz = ops.SparseMatrixNNZ()
        >>> out = sparse_matrix_nnz(dense_shape, batch_pointers, row_pointers, col_indices, values)
        >>> print(out)
        [1]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseMatrixNNZ"""
        self.init_prim_io_names(
            inputs=['x_dense_shape', 'x_batch_pointers', 'x_row_pointers', 'x_col_indices', 'x_values'], outputs=['y'])


class SparseFillEmptyRows(Primitive):
    r"""
    Fill the blank lines in the input 2D SparseTensor with default values.

    Inputs:
        - **indices** (Tensor) - A 2-D Tensor, represents the position of the element in the sparse tensor.
          Support int64, each element value should be a non-negative int number. The shape is :math:`(n, 2)`.
        - **values** (Tensor) - A 1-D Tensor, represents the value corresponding to the position in the `indices`.
          The shape should be :math:`(n,)`.
        - **dense_shape** (Tensor) - A 1-D Tensor with only two elements, represents the shape of SparseTensor.
          Support int64.
        - **default_value** (Tensor) - A 0-D Tensor of the same type as `values`, scalar value to
          fill the blank lines in the input 2D SparseTensor.

    Outputs:
        - **output_indices** (Tensor) - A 2-D Tensor, represents the position of the element in the sparse tensor
          after being filled. Support int64, each element value should be a non-negative int number.
          The shape is :math:`(m, 2)`, because of being filled, m>=n.
        - **output_values** (Tensor) - A 1-D Tensor. It represents the value corresponding to the position
          in the `output_indices`, the shape of which should be :math:`(m,)`, because of being filled, m>=n.
        - **empty_row_indicator** (Tensor) - A 1-D Tensor. It indicates whether each row is empty.
          Support bool. The shape is :math:`(dense\_shape[0],)`.
        - **reverse_index_map** (Tensor) - A 1-D Tensor. It is the index that means the value here is original
          rather than filled. Support bool. The shape is :math:`(n, 2)`.

    Raises:
        TypeError: If the dtype of `indices` is not int64.
        TypeError: If the dtype of `dense_shape` is not int64.
        TypeError: If the dtype of `values` and the dtype of `default_value` are not same.
        ValueError: If `sparse_shape`, shape of `indices` and shape of `values` don't meet the parameter description.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[1, 0]], dtype=mstype.int64)
        >>> values = Tensor([4], dtype=mstype.float32)
        >>> dense_shape = Tensor([2, 3], dtype=mstype.int64)
        >>> default_value = Tensor(5, dtype=mstype.float32)
        >>> sparsefillemptyrows = ops.SparseFillEmptyRows()
        >>> out = sparsefillemptyrows(indices, values, dense_shape, default_value)
        >>> print(out[0])
        Tensor(shape=[2, 2], dtype=Int64, value=
        [[0, 0],
         [1, 0]])
        >>> print(out[1])
        Tensor(shape=[2], dtype=Float32, value= [ 5.00000000e+00,  4.00000000e+00])
        >>> print(out[2])
        Tensor(shape=[2], dtype=Bool, value= [ True, False])
        >>> print(out[3])
        Tensor(shape=[1], dtype=Int64, value= [1])
    """
    @prim_attr_register
    def __init__(self):
        """Initialize SparseFillEmptyRows."""
        self.init_prim_io_names(inputs=['indices', 'values', 'dense_shape', 'default_value'],
                                outputs=['output_indices', 'output_values', 'empty_row_indicator', 'reverse_index_map'])


class SparseSegmentMeanWithNumSegments(Primitive):
    """
    Compute the mean along sparse segments of a tensor. It is allowed to have missing id in segment_ids.

    Inputs:
        - **x** (Tensor) - A Tensor of the first input of SparseSegmentMeanWithNumSegments.
        - **indices** (Tensor) - 1-D Tensor with indices into `x`. Must be one of the following
          types: int32, int64. Has same rank as `segment_ids`. The shape should be :math:`(N,)`.
        - **segment_ids** (Tensor) - 1-D Tensor with indices into the output `y`. Must be one of the
          following types: int32, int64. Values should be sorted and can be repeated. The shape should
          be :math:`(N,)`.
        - **num_segments** (Tensor) - Num_segments indicates the size of the output.
          It should be bigger than the largest id of `segment_ids`.

    Outputs:
        A Tensor. Has the same type as `x` .
        Has same shape as `x`, except for dimension 0 which is the value of `num_segments`.

    Raises:
        TypeError: If `x` or `indices` or `segment_ids` or `num_segments` is not a tensor.
        TypeError: If dtype of `x` is not in [float16, float32, float64].
        TypeError: If dtype of `indices` is not int32 or int64.
        TypeError: If dtype of `segment_ids` and `indices` mismatch.
        TypeError: If dtype of `num_segments` and `indices` mismatch.
        ValueError: If rank of `x` is less than 1.
        ValueError: If rank of `indices` or `segment_ids` is not 1.
        ValueError: If rank of `num_segments` is bigger than 1.
        ValueError: If numelements of `num_segments` is not 1.
        ValueError: If the first dimension of `indices` is not equal to the first dimension of `segment_ids`.
        ValueError: If `segment_ids` is not sorted.
        ValueError: If the largest id of `segment_ids` is bigger than or equal to `num_segments`.
        ValueError: If `indices` is out of range of x's first dimension.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> import mindspore as ms
        >>> import mindspore.ops.operations.sparse_ops as ops
        >>> x = Tensor([[0, 2, 0, 0], [0, 1, 1, 0], [2, 0, 2, 0]], dtype=ms.float16)
        >>> indices = Tensor([0, 2, 1], dtype=ms.int32)
        >>> segment_ids = Tensor([0, 0, 2], dtype=ms.int32)
        >>> num_segments = Tensor([4], dtype=ms.int32)
        >>> sparse_segment_mean_with_num_segments = ops.SparseSegmentMeanWithNumSegments()
        >>> output = sparse_segment_mean_with_num_segments(x, indices, segment_ids, num_segments)
        >>> print(output)
        [[1. 1. 1. 0.]
         [0. 0. 0. 0.]
         [0. 1. 1. 0.]
         [0. 0. 0. 0.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseSegmentMeanWithNumSegments"""
        self.init_prim_io_names(inputs=['x', 'indices', 'segment_ids', 'num_segments'], outputs=['y'])


class SparseAdd(Primitive):
    """
    Computes the sum of a COOTensor and another COOTensor.

    Inputs:
        - **x1_indices** (Tensor) - represents the first COOTensor's indices.
        - **x1_values** (Tensor) - represents the first COOTensor's values.
        - **x1_shape** (Tensor) - represents the first COOTensor's dense shape.
        - **x2_indices** (Tensor) - represents the second COOTensor's indices.
        - **x2_values** (Tensor) - represents the second COOTensor's values.
        - **x2_shape** (Tensor) - represents the second COOTensor's dense shape.
        - **thresh** (Tensor) - A 0-D Tensor, represents the magnitude threshold that determines if an output
          value/index pair take space. Its dtype should match that of the values if they are real.
          If output's value is less than the `thresh`, it will vanish.

    Outputs:
        - **sum_indices** (Tensor) - this is the indices of the sum.
        - **sum_values** (Tensor) - this is the values of the sum.
        - **sum_shape** (Tensor) - this is the shape of the sum.

    Raises:
        ValueError: If (x1_indices/x2_indices)'s dim is not equal to 2.
        ValueError: If (x1_values/x2_values)'s dim is not equal to 1.
        ValueError: If (x1_shape/x2_shape)'s dim is not equal to 1.
        ValueError: If thresh's dim is not equal to 0.
        TypeError: If (x1_indices/x2_indices)'s type is not equal to int64.
        TypeError: If (x1_shape/x2_shape)'s type is not equal to int64.
        ValueError: If (x1_indices/x2_indices)'s length is not equal to
            (x1_values/x2_values)'s length.
        TypeError: If (x1_values/x2_values)'s type is not equal to anf of
            (int8/int16/int32/int64/float32/float64/complex64/complex128).
        TypeError: If thresh's type is not equal to anf of
            (int8/int16/int32/int64/float32/float64).
        TypeError: If x1_indices's type is not equal to x2_indices's type.
        TypeError: If x1_values's type is not equal to x2_values's type.
        TypeError: If x1_shape's type is not equal to x2_shape's type.
        TypeError: If (x1_values/x2_values)'s type is not matched with thresh's type.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore import dtype as mstype
        >>> from mindspore.ops.operations.sparse_ops import SparseAdd
        >>> indics0 = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values0 = Tensor([1, 2], dtype=mstype.int32)
        >>> shape0 = Tensor([3, 4], dtype=mstype.int64)
        >>> indics1 = Tensor([[0, 0], [1, 1]], dtype=mstype.int64)
        >>> values1 = Tensor([3, 4], dtype=mstype.int32)
        >>> shape1 = Tensor([3, 4], dtype=mstype.int64)
        >>> thres = Tensor(0, dtype=mstype.int32)
        >>> sparse_add = SparseAdd()
        >>> out = sparse_add(indics0, values0, shape0, indics1, values1, shape1, thres)
        >>> print(out)
        (Tensor(shape=[4, 2], dtype=Int64, value=[[0, 0], [0, 1], [1, 1], [1, 2]]),
        Tensor(shape=[4], dtype=Int32, value=[3, 1, 4, 2]),
        Tensor(shape=[2], dtype=Int64, value=[3, 4]))
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(
            inputs=["x1_indices", "x1_values", "x1_shape",
                    "x2_indices", "x2_values", "x2_shape", "thresh"],
            outputs=["sum_indices", "sum_values", "sum_shape"])


class SparseMatrixSoftmax(Primitive):
    """
    Calculates the softmax of a CSRTensorMatrix.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        dtype (dtype.Number) - The valid data type. Only constant value is allowed.

    Inputs:
        - **x_dense_shape** (Tensor) - Input shape of the original Dense matrix.
        - **x_batch_pointers** (Tensor) - The number of rows in the input matrix.
        - **x_row_pointers** (Tensor) - Input the column coordinates of nonzero elements.
        - **x_col_indices** (Tensor) - The number of input nonzero elements up to that line.
        - **x_values** (Tensor) - The value of the input nonzero element.

    Outputs:
        - **y_dense_shape** (Tensor) - Output shape of the original Dense matrix.
        - **y_batch_pointers** (Tensor) - The number of rows in the output matrix.
        - **y_row_pointers** (Tensor) - Output the column coordinates of nonzero elements.
        - **y_col_indices** (Tensor) - The number of output nonzero elements up to that line.
        - **y_values** (Tensor) - The value of the input nonzero element.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.common.dtype as mstype
        >>> from mindspore import Tensor, CSRTensor
        >>> from mindspore.ops.operations.sparse_ops import SparseMatrixSoftmax
        >>> logits_indptr = Tensor([0, 4, 6], dtype=mstype.int32)
        >>> logits_indices = Tensor([0, 2, 3, 4, 3, 4], dtype=mstype.int32)
        >>> logits_values = Tensor([1, 2, 3, 4, 1, 2], dtype=mstype.float32)
        >>> shape = (2, 6)
        >>> logits = CSRTensor(logits_indptr, logits_indices, logits_values, shape)
        >>> net = SparseMatrixSoftmax(mstype.float32)
        >>> logits_pointers =Tensor(logits.values.shape[0], mstype.int32)
        >>> out = net(Tensor(logits.shape, dtype=mstype.int32), logits_pointers,
        ...         logits.indptr, logits.indices, logits.values)
        >>> print(out)
        (Tensor(shape=[2], dtype=Int32, value= [2, 6]),
         Tensor(shape=[], dtype=Int32, value= 6),
         Tensor(shape=[3], dtype=Int32, value= [0, 4, 6]),
         Tensor(shape=[6], dtype=Int32, value= [0, 2, 3, 4, 3, 4]),
         Tensor(shape=[6], dtype=Float32, value= [ 3.20586003e-02,  8.71443152e-02,
         2.36882806e-01,  6.43914223e-01,  2.68941432e-01,  7.31058598e-01]))
    """

    @prim_attr_register
    def __init__(self, dtype):
        '''Initialize for SparseMatrixSoftmax'''
        if not isinstance(dtype, (type(mstype.float32), type(mstype.single), type(mstype.float64),
                                  type(mstype.double))):
            raise TypeError(
                "Only float32 and float64 type data are supported, but got {}".format(dtype))
        self.add_prim_attr("dtype", dtype)
        self.init_prim_io_names(inputs=['x_dense_shape', 'x_batch_pointers', 'x_row_pointers',
                                        'x_col_indices', 'x_values'],
                                outputs=['y_dense_shape', 'y_batch_pointers', 'y_row_pointers', 'y_col_indices',
                                         'y_values'])


class CSRSparseMatrixToDense(Primitive):
    """
    Converts a CSR sparse matrix(maybe batched) to its dense form.

    Note:
        It is assumed that all the inputs can form a legal CSR sparse matrix, otherwise this operator won't work.

    Inputs:
        - **x_dense_shape** (Tensor) - A 1-D Tensor. It represents the dense form shape of
          the input CSR sparse matrix, the shape of which should be :math:`(2,)` or :math:`(3,)`.
        - **x_batch_pointers** (Tensor) - A 1-D Tensor. Supposing the input CSR sparse matrix is of
          batch size `n`, it should have shape :math:`(n+1,)`, while the `i`-th element of which stores
          acummulated counts of nonzero values of the first `i - 1` batches.
        - **x_row_pointers** (Tensor) - A 1-D Tensor. Supposing the input CSR sparse matrix is of
          batch size `n` and row number `m`, it can be divided into `n` parts, each part of length
          `m + 1`. The `i`-th element of each :math:`(m+1,)` vector stores acummulated counts of
          nonzero values of the first `i - 1` rows in the corresponding batch.
        - **x_col_indices** (Tensor) - A 1-D Tensor. It represents column indices of the nonzero values
          in the input CSR sparse matrix.
        - **x_values** (Tensor) - A 1-D Tensor. It represents all the nonzero values in the
          input CSR sparse matrix.

    Outputs:
        Tensor, which is the dense form of the input CSR sparse matrix.
        Its dtype is the same as `x_values`.

    Raises:
        TypeError: If the dtype of `x_dense_shape`, `x_batch_pointers`, `x_row_pointers` or `x_col_indices`
            is not int32 or int64, or the dtypes of above inputs are not the same.
        TypeError: If the dtype of `x_values` is not float32, float64, complex64 or complex128.
        TypeError: If any of the inputs is not a tensor.
        ValueError: If any of the inputs is not 1-D.
        ValueError: If shape[0] of `x_dense_shape` is not 2 or 3.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> dense_shape = Tensor([2, 2], dtype=mindspore.int32)
        >>> batch_pointers = Tensor([0, 1], dtype=mindspore.int32)
        >>> row_pointers = Tensor([0, 1, 1], dtype=mindspore.int32)
        >>> col_indices = Tensor([1], dtype=mindspore.int32)
        >>> values = Tensor([1.], dtype=mindspore.float32)
        >>> csr_to_dense = ops.CSRSparseMatrixToDense()
        >>> out = csr_to_dense(dense_shape, batch_pointers, row_pointers, col_indices, values)
        >>> print(out)
        [[0. 1.]
         [0. 0.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CSRSparseMatrixToDense"""
        self.init_prim_io_names(
            inputs=['x_dense_shape', 'x_batch_pointers',
                    'x_row_pointers', 'x_col_indices', 'x_values'],
            outputs=['y'])


class SparseMatrixTranspose(Primitive):
    r"""
    Return the transpose of sparse matrix or sparse matrixs.
    If the sparse matrix input contains batch dimension, then output dimension will be same with the batch dimension.
    The rank of sparse matrix input must be equal to `2` or `3`.

    Note:
        It is assumed that all the inputs can form a legal CSR sparse matrix, otherwise this operator is not defined.

    Args:
        conjugate (bool): If True, the output sparse tensor is conjugated . Default: False.

    Inputs:
        - **dense_shape** (Tensor) - A 1-D Tensor, represents the shape of input sparse matrix under dense status.
          Support int32, int64. The shape is :math:`(2,)` or :math:`(3,)`.
        - **batch_pointers** (Tensor) - A 1-D Tensor, represents the non-zero elements number in each batch.
          Support int32, int64, takes on values: :math:`(0, nnz[0], nnz[0] + nnz[1], ..., total\_nnz)`.
          If there are `n` batch within input sparse matrix, the shape is :math:`(n+1)`.
        - **row_pointers** (Tensor) - A 1-D Tensor, represents the non-zero elements of each row.
          Support int32, int64, takes on values:
          :math:`(0, num\_rows\{b\}[0], num\_rows\{b\}[0] + num\_rows\{b\}[1], ..., nnz[b])`,
          for :math:`b = 0, ..., n - 1`.
          If there are `n` batch within input sparse matrix and dense shape is :math:`(rows,cols)`,
          the shape is :math:`((rows + 1) * n)`.
          Note: num_rows{0}[0] means the non-zero elements number in the first row of first sparse matrix.
        - **col_indices** (Tensor) - A 1-D Tensor, represents the column values for the given row and column index.
          Support int32, int64. The shape is :math:`(M)`,
          where `M` is the number of non-zero elements in all input sparse matrix.
        - **values** (Tensor) - A 1-D Tensor, represents the actual values for the given row and column index.
          Support BasicType. The shape is :math:`(M)`, where `M` is the number of non-zero elements in all
          input sparse matrix.

    Outputs:
        - **dense_shape** (Tensor) - A 1-D Tensor, represents the shape of output sparse matrix under dense status.
          Support int32, int64. The shape is the same as the input sparse matrix.
        - **batch_pointers** (Tensor) - A 1-D Tensor, which is the same as the input sparse matrix's batch_pointers.
        - **row_pointers** (Tensor) - A 1-D Tensor, represents the non-zero elements of each row of output sparse
          matrix. Support int32, int64, takes on values:
          :math:`(0, num\_rows\{b\}[0], num\_rows\{b\}[0] + num\_rows\{b\}[1], ..., nnz[b])`,
          for :math:`b = 0, ..., n - 1`.
          If there are `n` batch within output sparse matrix and dense shape is :math:`(rows,cols)`,
          the shape is :math:`((rows + 1) * n)`.
          Note: num_rows{0}[0] means the non-zero elements number in the first row of first sparse matrix.
        - **col_indices** (Tensor) - A 1-D Tensor, represents the column values for the given row and column index.
          Support int32, int64. The shape is :math:`(M)`,
          where `M` is the number of non-zero elements in all input sparse matrix.
        - **values** (Tensor) - A 1-D Tensor, which is the same as the input sparse matrix's values.

    Raises:
        TypeError: If dtype of `values` doesn't meet the parameter description.
        TypeError: The data type of `dense_shape, batch_pointers, row_pointers, col_indices` is not int32 or int64.
        ValueError: If rank of `dense_shape` is not 2 or 3.
        TypeError: The input data should have the correct CSR form.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.ops import operations as ops
        >>> dense_shape = Tensor([2,3], dtype=ms.int32)
        >>> batch_pointers = Tensor([0,1], dtype=ms.int32)
        >>> row_pointers = Tensor([0,1,1], dtype=ms.int32)
        >>> col_indices = Tensor([0], dtype=ms.int32)
        >>> values = Tensor([99], dtype=ms.float32)
        >>> sparse_matrix_transpose = ops.SparseMatrixTranspose()
        >>> output = sparse_matrix_transpose(dense_shape, batch_pointers, row_pointers, col_indices, values)
        >>> print(output[0])
        [3 2]
        >>> print(output[1])
        [0 1]
        >>> print(output[2])
        [0 1 1 1]
        >>> print(output[3])
        [0]
        >>> print(output[4])
        [99.]
    """

    @prim_attr_register
    def __init__(self, conjugate=False):
        """Initialize SparseMatrixTranspose"""
        validator.check_value_type("conjugate", conjugate, [bool], self.name)
        self.add_prim_attr("max_length", 100000000)
        self.init_prim_io_names(inputs=['x_dense_shape', 'x_batch_pointers', 'x_row_pointers',
                                        'x_col_indices', 'x_values'],
                                outputs=['y_dense_shape', 'y_batch_pointers', 'y_row_pointers',
                                         'y_col_indices', 'y_values'])


class SparseSparseMinimum(Primitive):
    r"""
    Returns the element-wise min of two SparseTensors.

    Inputs:
        - **x1_indices** (Tensor) - A 2-D Tensor. It represents the position of the non-zero element
          in the first sparse tensor.
        - **x1_values** (Tensor) - A 1-D Tensor. It represents the value corresponding to the position
          in the `x1_indices`, the shape of which should be :math:`(N,)`.
        - **x1_shape** (Tensor) - A 1-D Tensor. It represents the shape of the input sparse tensor,
          the shape of which should be :math:`(N,)`.
        - **x2_indices** (Tensor) - A 2-D Tensor. It represents the position of the non-zero element
          in the second sparse tensor.
        - **x2_values** (Tensor) - A 1-D Tensor. It represents the value corresponding to the position
          in the `x2_indices`, the shape of which should be :math:`(N,)`.
        - **x2_shape** (Tensor) - A 1-D Tensor. It represents the shape of the input sparse tensor,
          the shape of which should be :math:`(N,)`.

    Outputs:
        - **y_indices** (Tensor) - A 2-D Tensor. It represents the position of the element-wise min of
          two input tensors.
        - **y_values** (Tensor) - A 1-D Tensor. It represents the value corresponding to the position
          in the `y_indices`.

    Raises:
        TypeError: The dtype of `x1_indices`, `x1_shape`, `x2_indices` or `x2_shape` is wrong.
        TypeError: The dtype of `x1_values` or `x2_values` is wrong.
        TypeError: If `x1_indices`, `x1_values`, `x1_shape`, `x2_indices`, `x2_values`, `x2_shape`
                    is not a tensor.
        TypeError: If `x1_indices` is not a 2-D tensor.
        TypeError: If `x2_indices` is not a 2-D tensor.
        ValueError: If any of `x1_values` and `x1_shape` is not a 1-D tensor.
        ValueError: If shape[0] of `x1_indices` is not corresponding to shape[0] of `x1_values`.
        ValueError: If shape[1] of `x1_indices` is not corresponding to shape[0] of `x1_shape`.
        ValueError: If any of `x2_values` and `x2_shape` is not a 1-D tensor.
        ValueError: If shape[0] of `x2_indices` is not corresponding to shape[0] of `x2_values`.
        ValueError: If shape[1] of `x2_indices` is not corresponding to shape[0] of `x2_shape`.
        ValueError: If shape[0] of `x1_shape` is not corresponding to shape[0] of `x2_shape`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.ops.operations.sparse_ops import SparseSparseMinimum
        >>> x1_indices = Tensor(np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]]).astype(np.int64))
        >>> x1_values = Tensor([1, 2, 3], dtype=mstype.float32)
        >>> x1_shape = Tensor(np.array([2, 2, 2]).astype(np.int64))
        >>> x2_indices = Tensor(np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]).astype(np.int64))
        >>> x2_values = Tensor([2, 4, 5], dtype=mstype.float32)
        >>> x2_shape = Tensor(np.array([2, 2, 2]).astype(np.int64))
        >>> sparse_sparse_minimum = ops.SparseSparseMinimum()
        >>> out = sparse_sparse_minimum(x1_indices, x1_values, x1_shape, x2_indices, x2_values, x2_shape)
        >>> print(out[0])
        [[0 0 0]
         [0 1 0]
         [0 1 1]
         [1 0 0]]
        >>> print(out[1])
        [1. 2. 0. 0.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseSparseMinimum."""
        self.init_prim_io_names(inputs=['x1_indices', 'x1_values', 'x1_shape', 'x2_indices', 'x2_values', 'x2_shape'],
                                outputs=['y_indices', 'y_values'])


class SparseTensorToCSRSparseMatrix(Primitive):
    """
    Converts a sparse tensor to its CSR sparse matrix(maybe batched) form.

    Inputs:
        - **x_indices** (Tensor) - A 2-D Tensor. It represents the position of the non-zero element
          in the sparse tensor. Support int32, int64.
        - **x_values** (Tensor) - A 1-D Tensor. It represents the value corresponding to the position
          in the `x_indices`, the shape of which should be :math:`(N,)`.
        - **x_dense_shape** (Tensor) - A 1-D Tensor. It represents the dense form shape of
          the input sparse tensor. Its shape should be :math:`(2,)` or :math:`(3,)`. Support int32, int64.
    Outputs:
        - **y_dense_shape** (Tensor) - A 1-D Tensor. It represents the dense form shape of
          the output CSR sparse matrix, the shape of which should be :math:`(2,)` or :math:`(3,)`.
        - **y_batch_pointers** (Tensor) - A 1-D Tensor. Supposing the output CSR sparse matrix is of
          batch size `n`, it should have shape :math:`(n+1,)`, while the `i`-th element of which stores
          acummulated counts of non-zero values of the first `i - 1` batches.
        - **y_row_pointers** (Tensor) - A 1-D Tensor. Supposing the output CSR sparse matrix is of
          batch size `n` and row number `m`, it can be divided into `n` parts, each part of length
          `m + 1`. The `i`-th element of each :math:`(m+1,)` vector stores acummulated counts of
          non-zero values of the first `i - 1` rows in the corresponding batch.
        - **y_col_indices** (Tensor) - A 1-D Tensor. It represents column indices of the non-zero values
          in the output CSR sparse matrix.
        - **y_values** (Tensor) - A 1-D Tensor. It represents all the non-zero values in the
          output CSR sparse matrix.

    Raises:
        TypeError: If the dtype of `x_indices` or `x_dense_shape` is not int32 or int64.
        TypeError: If the dtype of `x_values` is not one of: float32, float64, complex64 or complex128.
        ValueError: If `x_indices` or `x_values` or `x_dense_shape` is not a tensor.
        ValueError: If any of `x_values` and `x_dense_shape` is not a 1-D tensor.
        ValueError: If `x_indices` is not a 2-D tensor.
        ValueError: If shape[0] of `x_indices` is not corresponding to shape[0] of `x_values`.
        ValueError: If shape[1] of `x_indices` is not corresponding to shape[1] of `x_dense_shape`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore.ops.operations.sparse_ops import SparseTensorToCSRSparseMatrix
        >>> x_indices = Tensor(np.array([[0, 0, 1], [0, 1, 2], [0, 1, 3], [1, 0, 1], [1, 1, 2],\
                                         [1, 1, 3]]).astype(np.int64))
        >>> x_values = Tensor(np.array([1, 4, 3, 1, 4, 3]).astype(np.float32))
        >>> x_dense_shape = Tensor(np.array([2, 2, 4]).astype(np.int64))
        >>> sparse_tensor_to_csr_sparse_matrix = SparseTensorToCSRSparseMatrix()
        >>> out = sparse_tensor_to_csr_sparse_matrix(x_indices, x_values, x_dense_shape)
        >>> print(out[0])
        [2 2 4]
        >>> print(out[1])
        [0 3 6]
        >>> print(out[2])
        [0 1 3 0 1 3]
        >>> print(out[3])
        [1 2 3 1 2 3]
        >>> print(out[4])
        [1. 4. 3. 1. 4. 3.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseTensorToCSRSparseMatrix."""
        self.init_prim_io_names(
            inputs=['x_indices', 'x_values', 'x_dense_shape'],
            outputs=['y_dense_shape', 'y_batch_pointers', 'y_row_pointers', 'y_col_indices', 'y_values'])


class SparseMatrixSparseMatMul(Primitive):
    r"""
    Performs a matrix multiplication of a sparse matrix x1 with sparse matrix x2; return a sparse matrix x1*x2.
    Each matrix may be transposed or adjointed (conjugated and transposed),
    according to the Boolean parameters transpose_a,adjoint_a,transpose_b and adjoint_b.
    At most one of transpose_a or adjoint_a may be True. Similarly, at most one of transpose_b or adjoint_b may be True.

    Args:
        transpose_a (bool): If true, sparse tensor x1 is transposed before multiplication. Default: False.
        transpose_b (bool): If true, dense tensor x2 is transposed before multiplication. Default: False.
        adjoint_a (bool): If true, sparse tensor x1 is adjointed before multiplication. Default: False.
        adjoint_b (bool): If true, dense tensor x2 is adjointed before multiplication. Default: False.

    Inputs:
        - **x1_dense_shape** (Tensor) - A 1-D Tensor, represents the shape of input sparse matrix x1 under dense status.
          Support int32, int64. The shape is :math:`(2)` or :math:`(3)`.
        - **x1_batch_pointers** (Tensor) - A 1-D Tensor, represents the non-zero elements number in each batch.
          Support int32, int64, takes on values: :math:`(0, nnz[0], nnz[0] + nnz[1], ..., total\_nnz)`.
          If there are `n` batch within input sparse matrix x1, the shape is :math:`(n+1)`.
        - **x1_row_pointers** (Tensor) - A 1-D Tensor, represents the non-zero elements of each row.
          Support int32, int64, takes on values:
          :math:`(0, num\_rows\{b\}[0], num\_rows\{b\}[0] + num\_rows\{b\}[1], ..., nnz[b])`,
          for :math:`b = 0, ..., n - 1`.
          If there are `n` batch within input sparse matrix x1 and dense shape is :math:`(rows,cols)`,
          the shape is :math:`((rows + 1) * n)`.
          Note: num_rows{0}[0] means the non-zero elements number in the first row of first sparse matrix x1.
        - **x1_col_indices** (Tensor) - A 1-D Tensor, represents the column values for the given row and column index.
          Support int32, int64. The shape is :math:`(M)`,
          where `M` is the number of non-zero elements in  input sparse matrix x1.
        - **x1_values** (Tensor) - A 1-D Tensor, represents the actual values for the given row and column index.
          Support float32, double, complex64, complex128.
          The shape is :math:`(M)`, where `M` is the number of non-zero elements in input sparse matrix x1.

          **x2_dense_shape** (Tensor) - B 1-D Tensor, represents the shape of input sparse matrix x2 under dense status.
          Support int32, int64. The shape is :math:`(2)` or :math:`(3)`.
        - **x2_batch_pointers** (Tensor) - B 1-D Tensor, represents the non-zero elements number in each batch.
          Support int32, int64, takes on values: :math:`(0, nnz[0], nnz[0] + nnz[1], ..., total\_nnz)`.
          If there are `n` batch within input sparse matrix x2, the shape is :math:`(n+1)`.
        - **x2_row_pointers** (Tensor) - B 1-D Tensor, represents the non-zero elements of each row.
          Support int32, int64, takes on values:
          :math:`(0, num\_rows\{b\}[0], num\_rows\{b\}[0] + num\_rows\{b\}[1], ..., nnz[b])`,
          for :math:`b = 0, ..., n - 1`.
          If there are `n` batch within input sparse matrix x2 and dense shape is :math:`(rows,cols)`,
          the shape is :math:`((rows + 1) * n)`.
          Note: num_rows{0}[0] means the non-zero elements number in the first row of sparse matrix x2.
        - **x2_col_indices** (Tensor) - B 1-D Tensor, represents the column values for the given row and column index.
          Support int32, int64. The shape is :math:`(M)`,
          where `M` is the number of non-zero elements in  input sparse matrix x2.
        - **x2_values** (Tensor) - B 1-D Tensor, represents the actual values for the given row and column index.
          Support float32, double, complex64, complex128.
          The shape is :math:`(M)`, where `M` is the number of non-zero elements in input sparse matrix x2.

    Outputs:
        - **y_dense_shape** (Tensor) - B 1-D Tensor, represents the shape of output sparse matrix y under dense status.
          Support int32, int64. The shape is :math:`(2)` or :math:`(3)`.
        - **y_batch_pointers** (Tensor) - B 1-D Tensor, represents the non-zero elements number in each batch.
          Support int32, int64, takes on values: :math:`(0, nnz[0], nnz[0] + nnz[1], ..., total\_nnz)`.
          If there are `n` batch within output sparse matrix y, the shape is :math:`(n+1)`.
        - **y_row_pointers** (Tensor) - B 1-D Tensor, represents the non-zero elements of each row.
          Support int32, int64, takes on values:
          :math:`(0, num\_rows\{b\}[0], num\_rows\{b\}[0] + num\_rows\{b\}[1], ..., nnz[b])`,
          for :math:`b = 0, ..., n - 1`.
          If there are `n` batch within output sparse matrix y and dense shape is :math:`(rows,cols)`,
          the shape is :math:`((rows + 1) * n)`.
          Note: num_rows{0}[0] means the non-zero elements number in the first row of sparse matrix y.
        - **y_col_indices** (Tensor) - B 1-D Tensor, represents the column values for the given row and column index.
          Support int32, int64. The shape is :math:`(M)`,
          where `M` is the number of non-zero elements in  output sparse matrix y.
        - **y_values** (Tensor) - B 1-D Tensor, represents the actual values for the given row and column index.
          Support float32, double, complex64, complex128.
          The shape is :math:`(M)`, where `M` is the number of non-zero elements in output sparse matrix y.

    Raises:
        TypeError: If any dtype of `x1_dense_shape`, `x1_batch_pointers`, `x1_row_pointers`, `x1_col_indices`,
        `x1_values` or `x2_dense_shape`, `x2_batch_pointers`, `x2_row_pointers`, `x2_col_indices`,
        `x2_values` doesn't meet the parameter description.
        ValueError: If rank of `x1_dense_shape` or `x2_dense_shape' is not 2 or 3.

    Supported Platforms:


    Examples:
        >>> from mindspore.ops.operations.sparse_ops import SparseMatrixSparseMatMul
        >>> x1_dense_shape = Tensor([4, 5], dtype=mindspore.int32)
        >>> x1_batch_pointers = Tensor([0, 4], dtype=mindspore.int32)
        >>> x1_row_pointers = Tensor([0, 1, 1, 3, 4], dtype=mindspore.int32)
        >>> x1_col_indices = Tensor([0, 3, 4, 0], dtype=mindspore.int32)
        >>> x1_values = Tensor([1.0, 5.0, -1.0, -2.0], dtype=mindspore.float32)
        >>> x2_dense_shape = Tensor([5, 3], dtype=mindspore.int32)
        >>> x2_batch_pointers = Tensor([0, 3], dtype=mindspore.int32)
        >>> x2_row_pointers = Tensor([0, 1, 1, 3, 3, 3], dtype=mindspore.int32)
        >>> x2_col_indices = Tensor([0, 0, 1], dtype=mindspore.int32)
        >>> x2_values = Tensor([2.0, 7.0, 8.0], dtype=mindspore.float32)
        >>> sparse_matrix_sparse_mat_mul = SparseMatrixSparseMatMul()
        >>> out_dense_shape, out_batch_pointers, out_row_pointers, out_col_indices, out_values =
        ... sparse_matrix_sparse_mat_mul(x1_dense_shape, x1_batch_pointers, x1_row_pointers, x1_col_indices, x1_values,
        ...                              x2_dense_shape, x2_batch_pointers, x2_row_pointers, x2_col_indices, x2_values)
        >>> print(out_dense_shape)
        [4 3]
        >>> print(out_batch_pointers)
        [0 2]
        >>> print(out_row_pointers)
        [0 1 1 1 2]
        >>> print(out_col_indices)
        [0 0]
        >>> print(out_values)
        [ 2. -4.]
    """

    @prim_attr_register
    def __init__(self, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False):
        """Initialize SparseMatrixSparseMatMul"""
        validator.check_value_type(
            "transpose_a", transpose_a, [bool], self.name)
        validator.check_value_type(
            "transpose_b", transpose_b, [bool], self.name)
        validator.check_value_type("adjoint_a", adjoint_b, [bool], self.name)
        validator.check_value_type("adjoint_b", adjoint_b, [bool], self.name)
        self.init_prim_io_names(
            inputs=['x1_dense_shape', 'x1_batch_pointers', 'x1_row_pointers', 'x1_col_indices', 'x1_values',
                    'x2_dense_shape', 'x2_batch_pointers', 'x2_row_pointers', 'x2_col_indices', 'x2_values'],
            outputs=['y_dense_shape', 'y_batch_pointers', 'y_row_pointers', 'y_col_indices', 'y_values'])


class SparseMatrixMatMul(Primitive):
    r"""
    Performs a matrix multiplication of a sparse matrix x1 with dense matrix x2; return a dense matrix x1*x2.
    Each matrix may be transposed or adjointed (conjugated and transposed)
    according to the Boolean parameters transpose_x1, adjoint_x1, transpose_x2 and adjoint_x2.
    At most one of transpose_x1 or adjoint_x1 may be True.
    Similarly, at most one of transpose_x2 or adjoint_x2 may be True.

    Note:
        It is assumed that all the inputs can form a legal CSR sparse matrix, otherwise this operator is not defined.

    Args:
        transpose_x1 (bool): If true, sparse tensor x1 is transposed before multiplication. Default: False.
        transpose_x2 (bool): If true, dense tensor x2 is transposed before multiplication. Default: False.
        adjoint_x1 (bool): If true, sparse tensor x1 is adjointed before multiplication. Default: False.
        adjoint_x2 (bool): If true, dense tensor x2 is adjointed before multiplication. Default: False.
        transpose_output (bool): If true, output x1*x2 is tansposed. Default: False.
        conjugate_output (bool): If true, output x1*x2 is conjugated. Default: False.

    Inputs:
        - **x1_dense_shape** (Tensor) - A 1-D Tensor. It represents the dense form shape of
          the input CSR sparse matrix x1, the shape of which should be :math:`(2,)` or :math:`(3,)`.
        - **x1_batch_pointers** (Tensor) - A 1-D Tensor. Supposing the input CSR sparse matrix x1 is of
          batch size `n`, it should have shape :math:`(n+1,)`, while the `i`-th element of which stores
          acummulated counts of nonzero values of the first `i - 1` batches.
        - **x1_row_pointers** (Tensor) - A 1-D Tensor. Supposing the input CSR sparse matrix x1 is of
          batch size `n` and row number `m`, it can be divided into `n` parts, each part of length
          `m + 1`. The `i`-th element of each :math:`(m+1,)` vector stores acummulated counts of
          nonzero values of the first `i - 1` rows in the corresponding batch.
        - **x1_col_indices** (Tensor) - A 1-D Tensor. It represents column indices of the nonzero values
          in the input CSR sparse matrix x1.
        - **x1_values** (Tensor) - A 1-D Tensor. It represents all the nonzero values
          in the input CSR sparse matrix x1. Support float32, float64, complex64, complex128.
        - **x2_dense** (Tensor) - A 2-D or 3-D Tensor, represents the input dense matrix x2.
          Its dtype is the same as `x1_values`.

    Outputs:
        Tensor, which represents the output dense matrix y.
        Its dtype is the same as `x1_values`.

    Raises:
        TypeError: If the dtype of `x1_dense_shape`, `x1_batch_pointers`, `x1_row_pointers` or `x1_col_indices`
                   is not int32 or int64, or the dtypes of above inputs are not the same.
        TypeError: If the dtype of `x1_values`, `x2_dense` is not supported.
        ValueError: If shape[0] of `x1_dense_shape` or the dimension of `x2_dense` is not 2 or 3.
        ValueError: If shape[0]-1 of `x1_batch_pointers` and shape[0] of `x2_dense` are not the same.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x1_dense_shape = Tensor([4, 5], dtype=ms.int32)
        >>> x1_batch_pointers = Tensor([0, 4], dtype=ms.int32)
        >>> x1_row_pointers = Tensor([0, 1, 1, 3, 4], dtype=ms.int32)
        >>> x1_col_indices = Tensor([0, 3, 4, 0], dtype=ms.int32)
        >>> x1_values = Tensor([1.0, 5.0, -1.0, -2.0], dtype=ms.float32)
        >>> x2_dense = Tensor([[2.0, 0.8, 1.0],[ 2.9, 3.2, 0.0],[7.0, 4.6, 0.2],[3.5, 4.9, 1.4],[4.0, 3.7, 6.9]],
        ... dtype=ms.float32)
        >>> sparse_matrix_mat_mul = ops.SparseMatrixMatMul()
        >>> out = sparse_matrix_mat_mul(x1_dense_shape, x1_batch_pointers, x1_row_pointers, x1_col_indices,
        ... x1_values, x2_dense)
        >>> print(out)
        [[ 2.         0.8        1.       ]
         [ 0.         0.         0.       ]
         [13.5       20.8        0.0999999]
         [-4.        -1.6       -2.       ]]
    """

    @prim_attr_register
    def __init__(self, transpose_x1=False, transpose_x2=False, adjoint_x1=False, adjoint_x2=False,
                 transpose_output=False, conjugate_output=False):
        """Initialize SparseMatrixMatMul"""
        validator.check_value_type(
            "transpose_x1", transpose_x1, [bool], self.name)
        validator.check_value_type(
            "transpose_x2", transpose_x2, [bool], self.name)
        validator.check_value_type("adjoint_x1", adjoint_x1, [bool], self.name)
        validator.check_value_type("adjoint_x2", adjoint_x2, [bool], self.name)
        validator.check_value_type(
            "transpose_output", transpose_output, [bool], self.name)
        validator.check_value_type(
            "conjugate_output", conjugate_output, [bool], self.name)
        self.init_prim_io_names(inputs=['x1_dense_shape', 'x1_batch_pointers', 'x1_row_pointers',
                                        'x1_col_indices', 'x1_values', 'x2_dense'], outputs=['y_dense'])


class SparseMatrixAdd(Primitive):
    """
    Addition of two CSR Tensors : C = alpha * A + beta * B

    Inputs:
        - **x1_dense_shape** (Tensor) - A 1-D Tensor represents the dense form shape of the input CSR sparse matrix.
        - **x1_batch_pointers** (Tensor) - A 1-D Tensor. Supposing the input CSR sparse matrix is of
          batch size `n`, it should have shape :math:`(n+1,)`, while the `i`-th element of which stores
          acummulated counts of non-zero values of the first `i - 1` batches.
        - **x1_row_pointers** (Tensor) - A 1-D Tensor. Supposing the input CSR sparse matrix is of
          batch size `n` and row number `m`, it can be divided into `n` parts, each part of length
          `m + 1`. The `i`-th element of each :math:`(m+1,)` vector stores acummulated counts of
          non-zero values of the first `i - 1` rows in the corresponding batch.
        - **x1_col_indices** (Tensor) - A 1-D Tensor. It represents column indices of the non-zero values
          in the input CSR sparse matrix.
        - **x1_values** (Tensor) - A 1-D Tensor. It represents all the non-zero values in the input CSR sparse matrix.
        - **x2_dense_shape** (Tensor) - A Tensor, same meaning as x1_dense_shape.
        - **x2_batch_pointers** (Tensor) - A Tensor, same meaning as x1_batch_pointers.
        - **x2_row_pointers** (Tensor) - A Tensor, same meaning as x1_row_pointers.
        - **x2_col_indices** (Tensor) - A Tensor, same meaning as x1_col_indices.
        - **x2_values** (Tensor) - A Tensor, same meaning as x1_values.
        - **alpha** (Tensor) - A Tensor.
        - **beta** (Tensor) - A Tensor.

    Outputs:
        - **y1_dense_shape** (Tensor) - A Tensor.
        - **y1_batch_pointers** (Tensor) - A Tensor.
        - **y1_row_pointers** (Tensor) - A Tensor.
        - **y1_col_indices** (Tensor) - A Tensor.
        - **y1_values** (Tensor) - A Tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> import mindspore.common.dtype as mstype
        >>> from mindspore import Tensor
        >>> from mindspore.ops.operations.sparse_ops import SparseMatrixAdd
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.op = SparseMatrixAdd()
        ...
        ...     def construct(self, a_shape, a_batch_pointer, a_indptr, a_indices, a_values,
        ...                   b_shape, b_batch_pointer, b_indptr, b_indices, b_values, alpha, beta):
        ...         return self.op(a_shape, a_batch_pointer, a_indptr, a_indices, a_values,
        ...                        b_shape, b_batch_pointer, b_indptr, b_indices, b_values, alpha, beta)
        >>> a_indptr = Tensor([0, 1, 2], dtype=mstype.int32)
        >>> a_indices = Tensor([0, 1], dtype=mstype.int32)
        >>> a_values = Tensor([1, 2], dtype=mstype.float32)
        >>> a_pointers = Tensor([0, a_values.shape[0]], dtype=mstype.int32)
        >>> shape = Tensor([2, 6], dtype=mstype.int32)
        >>> b_indptr = Tensor([0, 1, 2], dtype=mstype.int32)
        >>> b_indices = Tensor([0, 1], dtype=mstype.int32)
        >>> b_values = Tensor([1, 2], dtype=mstype.float32)
        >>> b_pointers = Tensor([0, b_values.shape[0]], dtype=mstype.int32)
        >>> alpha = Tensor(1, mstype.float32)
        >>> beta = Tensor(1, mstype.float32)
        >>> out = Net()(shape, a_pointers, a_indptr, a_indices, a_values,
        ...             shape, b_pointers, b_indptr, b_indices, b_values, alpha, beta)
        >>> print(out)
        (Tensor(shape=[2], dtype=Int32, value =[2, 6]),
         Tensor(shape[2], dtype=Int32, value = [0, 2]),
         Tensor(shape=[3], dtype=Int32, values = [0, 1, 2]),
         Tensor(shape=[2], dtype=Int32, values = [0, 1]),
         Tensor(shape=[2], dtype=Float32, values = [2.0, 4.0]))
    """

    @prim_attr_register
    def __init__(self):
        '''Initialize for SparseMatrixAdd'''
        self.init_prim_io_names(inputs=['x1_dense_shape', 'x1_batch_pointers', 'x1_row_pointers', 'x1_col_indices',
                                        'x1_values', 'x2_dense_shape', 'x2_batch_pointers', 'x2_row_pointers',
                                        'x2_col_indices', 'x2_values', 'alpha', 'beta'],
                                outputs=['y_dense_shape', 'y_batch_pointers', 'y_row_pointers', 'y_col_indices',
                                         'y_values'])


class SparseSplit(Primitive):
    """
    Split a `SparseTensor` into `num_split` tensors along one dimension.
    If the `shape[split_dim]` is not an integer multiple of `num_split`. Slices
    `[0 : shape[split_dim] % num_split]` gets one extra dimension.

    Args:
        num_split (int): An `int` that is `>= 1`. The number of ways to split. Default: 1.

    Inputs:
        - **split_dim** (Tensor) -A 0-D Tensor of type `int64`.
          The dimension along which to split.  Must be in the range `[0, rank(shape))`.
        - **indices** (Tensor) - A 2-D Tensor of type `int64`, represents the indices of the sparse tensor.
        - **values** (Tensor) - A 1-D Tensor, represents the values of the sparse tensor.
          Support float16, float32, float64, int32, int64, int8, int16, uint8, uint16, uint32,
          uint64, complex64, complex128, bool.
        - **shape** (Tensor) - A 1-D Tensor of type `int64`, represents the shape of the sparse tensor.

    Outputs:
          A tuple of `Tensor` objects (y_indices, y_values, y_shape).
        - **y_indices** (Tensor) - A 2-D Tensor of type `int64`.
        - **y_values** (Tensor) - A 1-D Tensor. The type is the same as input Tensor "values".
        - **y_shape** (Tensor) - A 1-D Tensor of type `int64`.

    Raises:
        TypeError: If the type of `split_dim` or `indices` or `shape` is not int64.
            If the type of `values` is not valid.
            If the type of `num_split` is not int.
        ValueError: If the num_element of `split_dim` is not 1.
            If the rank of `values` or `shape` is not 1.
            If the rank of `indices` is not 1.

    Supported Platforms:

    """

    @prim_attr_register
    def __init__(self, num_split=1):
        """Initialize SparseSplit."""
        self.init_prim_io_names(inputs=['split_dim', 'indices', 'values', 'shape'],
                                outputs=['y_indices', 'y_values', 'y_shape'])
        validator.check_value_type("num_split", num_split, [int], self.name)


class SparseMatrixOrderingAMD(Primitive):
    r"""
    Computes the Approximate Minimum Degree (AMD) ordering of `input`.
    Computes the Approximate Minimum Degree (AMD) ordering for a sparse matrix.

    The returned permutation may be used to permute the rows and columns of the given sparse matrix.
    This typically results in permuted sparse matrix's sparse Cholesky (or other decompositions) in
    having fewer zero fill-in compared to decomposition of the original matrix.

    The input sparse matrix may have rank 2 or rank 3. The output Tensor, representing would then have
    rank 1 or 2 respectively, with the same batch shape as the `input`.

    Each component of the input sparse matrix must represent a square symmetric matrix; only the lower
    triangular part of the matrix is read. The values of the sparse matrix does not affect the returned
    permutation, only the sparsity pattern of the sparse matrix is used. Hence, a single AMD ordering may
    be reused for the Cholesky decompositions of sparse matrices with the same sparsity pattern but
    with possibly different values.

    Each batch component of the output permutation represents a permutation of `N` elements, where
    the input sparse matrix components each have `N` rows. That is, the component contains each of the
    integers :math:`{0, .. N-1}` exactly once. The `i`th element represents the row index that the `i`th
    row maps to.

    Inputs:
        - **x_dense_shape** (Tensor) - A 1-D Tensor. It represents the dense form shape of
          the input CSR sparse matrix x, the shape of which should be :math:`(2,)` or :math:`(3,)`.
        - **x_batch_pointers** (Tensor) - A 1-D Tensor. Supposing the input CSR sparse matrix x is of
          batch size `n`, it should have shape :math:`(n+1,)`, while the `i`-th element of which stores
          acummulated counts of nonzero values of the first `i - 1` batches.
        - **x_row_pointers** (Tensor) - A 1-D Tensor. Supposing the input CSR sparse matrix x is of
          batch size `n` and row number `m`, it can be divided into `n` parts, each part of length
          `m + 1`. The `i`-th element of each :math:`(m+1,)` vector stores acummulated counts of
          nonzero values of the first `i - 1` rows in the corresponding batch.
        - **x_col_indices** (Tensor) - A 1-D Tensor. It represents column indices of the nonzero values
          in the input CSR sparse matrix x.
        - **x_values** (Tensor) - A 1-D Tensor. It represents all the nonzero values in the
          input CSR sparse matrix x.

    Outputs:
        Tensor, the dtype is int32.
        If there are n batch within input sparse matrix, the shape is :math:`(n,)`.

    Raises:
        TypeError: If the dtype of `x_dense_shape` is not int64.
        TypeError: If the dtype of `x_batch_pointers`, `x_row_pointers` or `x_col_indices` is not int32.
        TypeError: If the dtype of `x_values` is not supported.
        TypeError: If any of the inputs is not a tensor.
        ValueError: If any of the inputs is not 1-D.
        ValueError: If `x_values` and `x_col_indices` have different length.
        ValueError: If shape[0] of `x_dense_shape` is not 2 or 3.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.ops.operations.sparse_ops import SparseMatrixOrderingAMD
        >>> dense_shape = Tensor([2, 2], dtype=ms.int64)
        >>> batch_pointers = Tensor([0, 1], dtype=ms.int32)
        >>> row_pointers = Tensor([0, 1, 1], dtype=ms.int32)
        >>> col_indices = Tensor([0], dtype=ms.int32)
        >>> values = Tensor([99], dtype=ms.float32)
        >>> sparse_matrix_ordering_amd = SparseMatrixOrderingAMD()
        >>> output = sparse_matrix_ordering_amd(dense_shape, batch_pointers, row_pointers, col_indices, values)
        >>> print(output)
        [0 1]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseMatrixOrderingAMD."""
        self.init_prim_io_names(inputs=['x_dense_shape', 'x_batch_pointers', 'x_row_pointers',
                                        'x_col_indices', 'x_values'], outputs=['y'])


class SparseReshape(Primitive):
    """
    Reshapes a SparseTensor to represent values in a new dense shape.
    This operation has the same semantics as reshape on the represented dense tensor.
    The `input_indices` are recomputed based on the requested `new_shape`.
    At most one component of `new_shape` can be -1.
    Reshaping does not affect the order of values in the SparseTensor.

    Inputs:
        - **indices** (Tensor) - A 2D Tensor of type int64. The indices of the SparseTensor.
          The shape is :math:`(n, 2)`.
        - **shape** (Tensor) - A 1D Tensor of type int64. The shape of the SparseTensor.
        - **new_shape** (Tensor) - A 1D Tensor of type int64. The requested new dense shape.

    Outputs:
        - **y_indices** (Tensor) - A 2D Tensor of type int64. The indices of the new dense shape.
          The tensor has the same data type and shape as `indices`.
        - **y_shape** (Tensor) - A 1D Tensor of type int64. The shape of the new dense shape.

    Raises:
        TypeError: If the dtype of `indices`, `shape` or `new_shape` is not int64.
        ValueError: If the shape[1] of `indices` is not equal to the first dimension of `shape`.
        ValueError: If `indices` is not a 2D Tensor.
        ValueError: If `shape` is not a 1D Tensor.
        ValueError: If `new_shape` is not a 1D Tensor.
        RuntimeError: If the number of inferred-dims(-1) is larger than 1.
        RuntimeError: If there is any negative value(except -1) in `new_shape`.
        RuntimeError: If the numbers of elements that `shape` and `new_shape` represent are not equal.
        RuntimeError: If inferred-dim(-1) in `new_shape` cannot be correctly inferred.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 0, 0],
        ...                   [0, 0, 1],
        ...                   [0, 1, 0],
        ...                   [1, 0, 0],
        ...                   [1, 2, 3]],
        ...                   dtype=mstype.int64)
        >>> shape = Tensor([2, 3, 6], dtype=mstype.int64)
        >>> new_shape = Tensor([9, -1], dtype=mstype.int64)
        >>> sparse_reshape = sparse_ops.SparseReshape()
        >>> y_indices, y_shape = sparse_reshape(indices, shape, new_shape)
        >>> print(y_indices)
        [[0 0]
         [0 1]
         [1 2]
         [4 2]
         [8 1]]
        >>> print(y_shape)
        [9 4]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseReshape."""
        self.init_prim_io_names(inputs=['indices', 'shape', 'new_shape'], outputs=[
            'y_indices', 'y_shape'])


class SparseCountSparseOutput(Primitive):
    """
    Performs sparse-output bin counting for a sparse tensor input.
    Counts the number of times each value occurs in the input.

    Args:
        binary_output (bool) - If false, output the number of occurrences of each value,
                               if True output 1 for orresponding values. Default False
        minlength(Scalar) - Int type minimum value to count, default -1
        maxlength(Scalar) - Int type maximum value to count, default -1

    Inputs:
        - **indices** (Tensor) - Tensor representing the position of the element in the sparse
          tensor. Support int64, each element value should be a non-negative int number.
        - **values** (Tensor) - 1-D Tensor, represents the value corresponding to the position
          in the `indices`. Support int32, int64
        - **dense_shape** (Tensor) - A positive int tuple which specifies the shape of sparse
          tensor, should have 2 elements, support int64
        - **weights** (Tensor) - A Tensor of the same shape as indices containing per-index
          weight values. Support int32, int64, float32, float64

    Outputs:
        - **output_indices** (Tensor) - contains the indices of the output sparse tensor
        - **output_values** (Tensor) - contains the values of the output sparse tensor
        - **output_dense_shape** (Tensor) - contains the dense shape of the output sparse tensor

    Raises:
        TypeError: If binary_output is not a bool
        TypeError: If minlenght or maxlength are not integers
        TypeError: If dtype of indices and dense_shape is not int64
        TypeError: If dtype of values is neither int32 nor int64
        TypeError: If dtype of weights is not in int32, int64, float32, float64
        ValueError: If number of values does not match first dimension of indices
        ValueError: If number of dense_shape dimensions does not match second dimension of indices
        ValueError: If num dim of dense_shape is < 1
        RunTimeError: If number of weights is not equal to number of values
        RunTimeError: If indexes are not in bounds of the dense shape

    Examples:
        >>> from mindspore.ops.operations.sparse_ops import SparseCountSparseOutput
        >>> indices = Tensor([[1, 2] ,[2, 3], [2, 1], [0, 2]], dtype=mstype.int64)
        >>> values = Tensor([0, 2, 8, 8], dtype=mstype.int64)
        >>> dense_shape = Tensor([4, 4], dtype=mstype.int64)
        >>> weights = Tensor([1, 2, 1, 0], dtype=mstype.int64)
        >>> sparse_count_sparse_output = SparseCountSparseOutput()
        >>> out = sparse_count_sparse_output(indices, values, dense_shape, weights)
        >>> print(out)
        (Tensor(shape=[4, 2], dtype=Int64, value=
        [[0, 8],
         [1, 0],
         [2, 2],
         [2, 8]]), Tensor(shape=[4], dtype=Int64, value= [0, 1, 2, 1]), Tensor(shape=[2], dtype=Int64, value= [4, 9]))

    Supported Platforms:
        ``CPU``

    """

    @prim_attr_register
    def __init__(self, binary_output=False, minlength=-1, maxlength=-1):
        self.init_prim_io_names(
            inputs=["indices", "values", "dense_shape", "weights"],
            outputs=["output_indices", "output_values", "output_shape"])
        validator.check_value_type("binary_output", binary_output, [bool], self.name)
        validator.check_value_type("minlength", minlength, [int], self.name)
        validator.check_value_type("maxlength", maxlength, [int], self.name)


class DenseToSparseSetOperation(Primitive):
    """
    Applies set operation along last dimension of `x1` and `x2`.
    Input `x2` is a SparseTensor represented by `x2_indices`, `x2_values`, and `x2_shape`.
    For `x2` ranked `n`, 1st `n-1` dimensions must be the same as `x1`. Dimension `n` contains values in a set,
    duplicates are allowed but ignored.

    Args:
        set_operation (str): The type of set operation, supports four kinds of inputs, case insensitive. Default: "".
            "a-b": Get the difference set of x1 to x2.
            "b-a": Get the difference set of x2 to x1.
            "intersection": Get the intersection set of x2 to x1.
            "union": Get the union set of x2 to x1.
        validate_indices (bool): Optional attributes for DenseToSparseSetOperation.  Default: True.

    Inputs:
        - **x1** (Tensor) - The input tensor `x1` with rank `n`. 1st `n-1` dimensions must be the same as `x2`.
          Dimension `n` contains values in a set, duplicates are allowed but ignored. Must be one of the
          following types: int8, int16, int32, int64, uint8, uint16.
        - **x2_indices** (Tensor) - A 2-D Tensor, type int64, indices of a SparseTensor.
        - **x2_values** (Tensor) - A 1-D Tensor, must have the same type as x1, values of a SparseTensor. Size
          must be the same as `x2_indices`
        - **x2_shape** (Tensor) - A 1-D Tensor, type int64, shape of a SparseTensor, must have the same size as
          the second dimensions of `x2_indices`

    Outputs:
        y_indices: A Tensor of type int64.
        y_values: A Tensor. Has the same type as x1.
        y_shape: A Tensor of type int64 .

    Raises:
        TypeError: If any input is not Tensor.
        TypeError:If the dtype of `x2_values` is not the same as 'x1'.
        TypeError:If the dtype of `x2_indices` or `x2_shape` is not int64.
        ValueError: If the group shape of `x1` or `x2` mismatch with each other.
        ValueError: If the rank of `x1` is less than 2.
        ValueError: If the rank of `x2_indices` is not equal 2.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x1 = Tensor([[1, 2], [3, 0], [1, 5]], dtype=ms.int64)
        >>> x2_indices = Tensor([[0, 1], [0, 2], [1, 2]], dtype=ms.int64)
        >>> x2_values = Tensor([5, 1, 7],dtype=ms.int64)
        >>> x2_shape = Tensor([3, 3], dtype=ms.int64)
        >>> dense_to_sparse_set_operation = ops.DenseToSparseSetOperation(set_operation='intersection')
        >>> out = dense_to_sparse_set_operation(x1, x2_indices, x2_values, x2_shape)
        >>> print(out)
        (Tensor(shape=[1, 2], dtype=Int64, value=
        [[0, 0]]), Tensor(shape=[1], dtype=Int64, value= [1]), Tensor(shape=[2], dtype=Int64, value= [3, 1]))
    """

    @prim_attr_register
    def __init__(self, set_operation="", validate_indices=True):
        """Initialize DenseToSparseSetOperation."""
        self.init_prim_io_names(inputs=['x1', 'x2_indices', 'x2_values', 'x2_shape'],
                                outputs=['y_indices', 'y_values', 'y_shape'])
        self.set_operation = set_operation
        self.validate_indices = validate_indices
        self.add_prim_attr('set_operation', self.set_operation)
        self.add_prim_attr('validate_indices', self.validate_indices)

        validator.check_value_type("set_operation", set_operation, [str], self.name)
        validator.check_value_type("validate_indices", validate_indices, [bool], self.name)


class RaggedTensorToTensor(Primitive):
    r"""
    Create a dense tensor from a ragged tensor, possibly altering its shape.

    Args:
        row_partition_types(list(str)): A list of `strings`. The types of the row partition tensors.
            At present, these can be:
            "ROW_SPLITS": the row_splits tensor from the ragged tensor.
            "VALUE_ROWIDS": the value_rowids tensor from the ragged tensor.
            "FIRST_DIM_SIZE": if value_rowids is used for the first dimension, then it is preceded by "FIRST_DIM_SIZE".

    Inputs:
        - **shape** (Tensor) - A 1-D `Tensor`. Must be one of the following types: `int64`, `int32`.
          The desired shape of the output tensor.
        - **values** (Tensor) - A 1-D or higher `Tensor` representing the values of the ragged tensor.
        - **default_value** (Tensor) - A `Tensor` representing the default value of the ragged tensor.
          Must have the same type as `values` and less dimension than `values`.
        - **row_partition_tensors** (list(Tensor)) - A list of at least 1 `Tensor` objects with the same
          type in: `int64`, `int32`. The row partition tensor is 0-D, 1-D, 1-D, when the row partition type is
          "FIRST_DIM_SIZE", "VALUE_ROWIDS", "ROW_SPLITS" respectively.

    Outputs:
        A `Tensor`. Has the same type as `values` and the shape is `shape`.

    Raises:
        TypeError: If the type of `shape`, `values` or `default_value` is not Tensor.
        ValueError: If the dimension of `shape` or `values` is not 1.
        ValueError: If the dimension of `default_value` is more than `values`.
        ValueError: If the order or value of `row_partition_types` is not support.
        RuntimeError: If the value of `row_partition_tensors` is not in ascending order
            when the `row_partition_types` is "ROW_SPLITS".
        RuntimeError: If value rowid is not less than first dim size
            when the `row_partition_types` is "FIRST_DIM_SIZE", "VALUE_ROWIDS".
        ValueError: If row partition size plus `values` rank is not equal to `shape` rank.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.ops.operations.sparse_ops import RaggedTensorToTensor
        >>> shape = Tensor([4, 4], mstype.int32)
        >>> values = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], mstype.int64)
        >>> default_value = Tensor(0, dtype=mstype.int64)
        >>> row_partition_tensors_list = []
        >>> row_partition_tensors = Tensor([0, 3, 3, 7, 9], mstype.int32)
        >>> row_partition_tensors_list.append(row_partition_tensors)
        >>> row_partition_types = ["ROW_SPLITS"]
        >>> ragged_tensor_to_tensor = RaggedTensorToTensor(row_partition_types)
        >>> out = ragged_tensor_to_tensor(shape, values, default_value, row_partition_tensors_list)
        >>> print(out)
        [[1 2 3 0]
         [0 0 0 0]
         [4 5 6 7]
         [8 9 0 0]]
    """

    @prim_attr_register
    def __init__(self, row_partition_types):
        """Initialize RaggedTensorToTensor"""
        self.init_prim_io_names(inputs=['shape', 'values', 'default_value', 'row_partition_tensors'],
                                outputs=['result'])
        validator.check_value_type("row_partition_types", row_partition_types, [list], self.name)

        if not row_partition_types:
            raise ValueError(f"For {self.name}, row_partition_types cannot be empty.")

        for i, item in enumerate(row_partition_types):
            validator.check_value_type(f"row_partition_types[{i}]", item, [str], self.name)

        valid_values = ("ROW_SPLITS", "FIRST_DIM_SIZE", "VALUE_ROWIDS")
        if not set(row_partition_types).issubset(valid_values):
            diff = tuple(set(row_partition_types).difference(valid_values))
            raise ValueError(
                f"For {self.name}, row_partition_types only support {valid_values}, "
                f"but got {diff if len(diff) > 1 else repr(diff[0])}.")

        first_element = valid_values[:2]
        if row_partition_types[0] not in first_element:
            raise ValueError(
                f"For {self.name}, the first element of row_partition_types must be in {first_element}, "
                f"but got '{row_partition_types[0]}'.")

        if row_partition_types[0] == "FIRST_DIM_SIZE":
            if set(row_partition_types[1:]) != {"VALUE_ROWIDS"}:
                raise ValueError(
                    f"For {self.name}, 'VALUE_ROWIDS' must be preceded by 'FIRST_DIM_SIZE' in row_partition_types.")
        else:
            if set(row_partition_types) != {"ROW_SPLITS"}:
                raise ValueError(
                    f"For {self.name}, the each element of row_partition_types must be 'ROW_SPLITS' "
                    f"when row_splits tensor.")


class SparseCross(Primitive):
    """
    Generates sparse cross from a list of sparse and dense tensors.

    Args:
        hashed_output (bool): If true, returns the hash of the cross instead of the string. This will allow us
                              avoiding string manipulations.
        num_buckets (int): An int that is >= 0. It is used if "hashed_output" is true.output = hashed_value%num_buckets
                           if num_buckets > 0 else "hashed_value".
        hash_key (int): Specify the hash_key that will be used by the "FingerprintCat64" function to combine the
                        crosses fingerprints.
        out_type (mindspore.dtype): The output data type. Defaults to "int64".
        internal_type (mindspore.dtype): An type int64.

    Inputs:
        - **indices** (list(Tensor)) - A list of Tensor objects with type int64. 2-D.
          Indices of each input SparseTensor.
        - **values** (list(Tensor)) - A list of Tensor objects with types from: int64.
          1-D. values of each SparseTensor.
        - **shapes** (list(Tensor)) - A list with the same length as indices of Tensor objects with type int64.
          1-D. Shapes of each SparseTensor.
        - **dense_inputs** (list(Tensor)) - A list of Tensor objects with types from: int64.
          2-D. Columns represented by dense Tensor.

    Outputs:
        - **output_indices** (Tensor) - A Tensor of type int64. 2-D. Indices of the concatenated SparseTensor.
        - **output_values** (Tensor) - A Tensor of type "out_type". 1-D.
          Non-empty values of the concatenated or hashed SparseTensor.
        - **output_shape** (Tensor) - A Tensor of type int64. 1-D. Shape of the concatenated SparseTensor.

    Raises:
        TypeError: The indices shape rank is not equal to the shape rank.
        TypeError: The indices element number is not equal to the value element number.
        TypeError: The indices shape rank should be 2.
        TypeError: The denses shape rank should be 2.
        TypeError: The shapes rank should be 2.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.ops.operations.sparse_ops import SparseCross
        >>> indice1 = Tensor([[0,0],[1,0],[1,1]], dtype=mstype.int64)
        >>> value1 = Tensor([1, 2, 3], dtype=mstype.int64)
        >>> shape1 = Tensor([2, 2], dtype=mstype.int64)
        >>> dense1 = Tensor([[1],[2]], dtype=mstype.int64)
        >>> indice2 = Tensor([[0,0],[1,0],[1,1]], dtype=mstype.int64)
        >>> value2 = Tensor([1, 2, 3], dtype=mstype.int64)
        >>> shape2 = Tensor([2, 2], dtype=mstype.int64)
        >>> dense2 = Tensor([[1],[2]], dtype=mstype.int64)
        >>> indices = [indice1, indice2]
        >>> values = [value1, value2]
        >>> shapes = [shape1, shape2]
        >>> dense_inputs = [dense1, dense2]
        >>> hashed_output=True
        >>> hash_key= 2
        >>> out_type= mstype.int64
        >>> internal_type = mstype.int64
        >>> num_buckets=0
        >>> sparse_cross = SparseCross(hashed_output, hash_key, out_type, internal_type, num_buckets)
        >>> out = sparse_cross(indices, values, shapes, dense_inputs)
        >>> print(out)
        (Tensor(shape=[5, 2], dtype=Int64, value=
        [[0, 0],
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 3]]), Tensor(shape=[5], dtype=Int64, value= [1350190460805457680, 6319552725219729347,
        4652439303631496997, 7670687697825594049,  174086171018132662]), Tensor(shape=[2], dtype=Int64, value= [2, 4]))
    """

    @prim_attr_register
    def __init__(self, hashed_output, hash_key, out_type, internal_type, num_buckets=0):
        """Initialize SparseCross."""
        self.init_prim_io_names(inputs=["indices", "values", "shapes", "dense_inputs"],
                                outputs=["output_indices", "output_values", "output_shape"])
        validator.check_value_type("hashed_output", hashed_output, [bool], self.name)
        validator.check_value_type("hash_key", hash_key, [int], self.name)
        validator.check_value_type("out_type", out_type, [mstype.Type], self.name)
        validator.check_value_type("internal_type", internal_type, [mstype.Type], self.name)
        validator.check_value_type("num_buckets", num_buckets, [int], self.name)


class RaggedTensorToSparse(Primitive):
    r"""
    Converts a RaggedTensor into a SparseTensor with the same values.

    Args:
        Tsplits(mindspore.dtype): A required attribute, the type of the `rt_nested_splits`. Default: `int64`.

    Inputs:
        - **rt_nested_splits** (list(Tensor)) - A list of at least 1 `Tensor` objects with the same
          type in: `int64`, `int32`. The row_splits for the RaggedTensor.
          Ragged splits is in ascending order, first value of splits must be 0 and final value of splits
          must equal with the length of `rt_dense_values`.
        - **rt_dense_values** (Tensor) - A `Tensor`. The flat_values for the RaggedTensor. The rank of values
          must more than 0.

    Outputs:
        - **sparse_indices** (Tensor) - A `Tensor` of type int64. Contains the indices of the output
          sparse tensor.
        - **sparse_values** (Tensor) - A `Tensor`. Has the same type as rt_dense_values.
          Contains the values of the output sparse tensor.
        - **sparse_dense_shape** (Tensor) - A `Tensor` of type int64. Contains the dense shape of the
          output sparse tensor.

    Raises:
        TypeError: If the type of `Tsplits`, `rt_nested_splits` or `rt_dense_values` is not support.
        RuntimeError: If the order of `rt_nested_splits` is not support.
        RuntimeError: If the first value of `rt_nested_splits` is not 0.
        RuntimeError: If the final value of `rt_nested_splits` is not equal with the length of
            `rt_dense_values`.
        ValueError: If the rank of `rt_dense_values` is not more than 0.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore.ops.operations.sparse_ops import RaggedTensorToSparse
        >>> rt_nested_splits = Tensor([0, 3, 3, 5, 6], mstype.int64)
        >>> rt_dense_values = Tensor([1, 2, 3, 4, 5, 6], mstype.int32)
        >>> rt_nested_splits_list = []
        >>> rt_nested_splits_list.append(rt_nested_splits)
        >>> Tsplits = mstype.int64
        >>> ragged_tensor_to_sparse = RaggedTensorToSparse(Tsplits)
        >>> out = ragged_tensor_to_sparse(rt_nested_splits_list, rt_dense_values)
        >>> print(out)
        (Tensor(shape=[6, 2], dtype=Int64, value=
        [[0, 0],
         [0, 1],
         [0, 2],
         [2, 0],
         [2, 1],
         [3, 0]]),
         Tensor(shape=[6], dtype=Int32, value= [1, 2, 3, 4, 5, 6]),
         Tensor(shape=[2], dtype=Int64, value= [4, 3]))
    """
    @prim_attr_register
    def __init__(self, Tsplits):
        """Initialize RaggedTensorToSparse."""
        self.init_prim_io_names(inputs=['rt_nested_splits', 'rt_dense_values'],
                                outputs=['sparse_indices', 'sparse_values', 'sparse_dense_shape'])
        validator.check_value_type("Tsplits", Tsplits, [mstype.Type], self.name)
        valid_values = {mstype.int64, mstype.int32}
        validator.check_type_name("Tsplits", Tsplits, valid_values, self.name)
