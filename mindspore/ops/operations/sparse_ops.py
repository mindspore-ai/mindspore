# coding: utf-8

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

"""Operators for sparse operators."""

from ..._checkparam import Validator as validator
from ...common import dtype as mstype
from ..primitive import PrimitiveWithInfer, prim_attr_register

class SparseToDense(PrimitiveWithInfer):
    """
    Converts a sparse representation into a dense tensor.

    Inputs:
        - **indices** (Tensor) - The indices of sparse representation.
        - **values** (Tensor) - Values corresponding to each row of indices.
        - **dense_shape** (tuple) - An int tuple which specifies the shape of dense tensor.

    Returns:
        Tensor, the shape of tensor is `dense_shape`.

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]])
        >>> values = Tensor([1, 2], dtype=ms.float32)
        >>> dense_shape = (3, 4)
        >>> out = ops.SparseToDense()(indices, values, dense_shape)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize index_select"""
        self.init_prim_io_names(inputs=['indices', 'values', 'dense_shape'], outputs=['output'])

    def __infer__(self, indices, values, dense_shape):
        validator.check_subclass("indices", indices['dtype'], mstype.tensor, self.name)
        validator.check_subclass("values", values['dtype'], mstype.tensor, self.name)
        out = {'shape': dense_shape['value'],
               'dtype': values['dtype'],
               'value': None}
        return out

class SparseTensorDenseMatmul(PrimitiveWithInfer):
    """
    Multiply SparseTensor(of rank 2) "A" by dense tensor.
    The shape of sparse tensor is :math:`(N, C)`, and the shape of dense tensor is :math:`(C, M)`, then the shape of
    output tensor is :math:`(N, M)`.The output data type is the same as "values".
    tensors.

    Args:
        - *adjoint_st** (Bool) - If true, SparseTensor is transposed before multiplication. Default: False.
        - *adjoint_dt** (Bool) - If true, DenseTensor is transposed before multiplication. Default: False.

    Inputs:
        - **indices** (Tensor) - The indices of sparse representation, support int32/int64.
        - **values** (Tensor) - Values corresponding to each row of indices.
        - **dense_shape** (tuple) - An int tuple which specifies the shape of dense tensor. The dense_shape is :
          math:`(N, C)`. If `adjoint_st` is True, its shape must be :math:`(N, C)` after transpose.
        - **dense** (Tensor) - Dense Matrix. The shape of the tensor is :math:`(C, M)`. If
          `adjoint_dt` is True, its shape must be :math:`(C, M)` after transpose.

    Outputs:
        Tensor, the shape of tensor  is :math:`(N, M)`. The output data type is the same as "values".

    Raises:
        TypeError: If `indices` is neither int32 nor int64.
        TypeError: If 'values' is not boot, uint8-64, int8-64, float16-64.
        TypeError: If 'dense' is not boot, uint8-64, int8-64, float16-64.
        ValueError: If length of shape of `SparseTensor` or `DenseTensor` is not equal to 2

    Supported Platforms:
        ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=ms.int32)
        >>> values = Tensor([1, 2], dtype=ms.float32)
        >>> dense_shape = (3, 4)
        >>> dsMatrix = Tensor([[1,1], [2,2], [3,3 ], [4, 4]], dtype=ms.float32)
        >>> out = ops.SparseTensorDenseMatmul(indices, values, dense_shape, dsMatrix)
    """
    @prim_attr_register
    def __init__(self, adjoint_st=False, adjoint_dt=False):
        """Initialize SparseTensorDenseMatmul"""
        self.adjoint_st = adjoint_st
        self.adjoint_dt = adjoint_dt
        self.init_prim_io_names(inputs=['indices', 'values', 'dense_shape', 'dense'],
                                outputs=['output'])
        self.add_prim_attr('adjoint_st', self.adjoint_st)
        self.add_prim_attr('adjoint_dt', self.adjoint_dt)
        validator.check_value_type("adjoint_st", adjoint_st, [bool], self.name)
        validator.check_value_type("adjoint_dt", adjoint_dt, [bool], self.name)

    def __infer__(self, indices, values, dense_shape, dense):
        validator.check_tensor_dtype_valid('indices', indices['dtype'], [mstype.int32, mstype.int64], self.name)
        valid_types = mstype.number_type + (mstype.bool_,)
        args = {'values': values['dtype'], 'dense': dense['dtype']}
        validator.check_tensors_dtypes_same_and_valid(args, valid_types, self.name)
        a_shape = dense_shape['value']
        b_shape = dense['shape']
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise ValueError('SparseTensorDenseMatmul SparseTensor, DenseTensor should have the same dimension size '
                             + f'and equal to 2, while SparseTensor size is ({len(a_shape)}) and DenseTensor size is '
                             + f'({len(b_shape)}).')
        out_shape = []
        out_shape.append(a_shape[0])
        out_shape.append(b_shape[1])
        out = {'shape': tuple(out_shape),
               'dtype': values['dtype'],
               'value': None}
        return out
