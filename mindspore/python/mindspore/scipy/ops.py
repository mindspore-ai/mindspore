# Copyright 2021-2024 Huawei Technologies Co., Ltd
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
"""Operators for scipy submodule"""
from mindspore import _checkparam as validator
from ..ops import PrimitiveWithInfer, prim_attr_register, Primitive
from ..ops.auto_generate import solve_triangular
from ..common import dtype as mstype


class SolveTriangular():
    """
    Solve linear system,(triangular matrix)
    a * x = b
    """

    def __init__(self, lower: bool = False, unit_diagonal: bool = False, trans: str = 'N'):
        self.lower = lower
        self.unit_diagonal = unit_diagonal
        trans_str_to_int = {'N': 0, 'T': 1, 'C': 2}
        self.trans = trans_str_to_int.get(trans)

    def __call__(self, a, b):
        return solve_triangular(a, b, self.trans, self.lower, self.unit_diagonal)


class Eig(PrimitiveWithInfer):
    """
    Eig decomposition,(generic matrix)
    a * v = w * v
    """

    @prim_attr_register
    def __init__(self, compute_v=True):
        super().__init__(name="Eig")
        self.compute_v = validator.check_value_type("compute_v", compute_v, [bool], self.name)
        self._set_prim_arg("compute_v", compute_v)
        self.io_table = {
            mstype.TensorType(mstype.float32): mstype.complex64,
            mstype.TensorType(mstype.complex64): mstype.complex64,
            mstype.TensorType(mstype.float64): mstype.complex128,
            mstype.TensorType(mstype.complex128): mstype.complex128
        }

    def __infer__(self, a):
        a_dtype = a["dtype"]
        a_shape = tuple(a["shape"])
        validator.check_tensor_dtype_valid("a", a_dtype,
                                           [mstype.float32, mstype.float64, mstype.complex64, mstype.complex128],
                                           self.name)

        output = None
        if self.compute_v:
            output = {
                'shape': (a_shape[:-1], a_shape),
                'dtype': (self.io_table.get(a_dtype), self.io_table.get(a_dtype)),
                'value': None
            }
        else:
            output = {
                'shape': a_shape[:-1],
                'dtype': self.io_table.get(a_dtype),
                'value': None
            }
        return output

    def __call__(self, a):
        return super().__call__(a, self.compute_v)


class LU(PrimitiveWithInfer):
    """
    LU decomposition with partial pivoting
    A = P.L.U
    """

    @prim_attr_register
    def __init__(self):
        super().__init__(name="LU")
        self.init_prim_io_names(inputs=['x'], outputs=['lu', 'pivots', 'permutation'])

    def __infer__(self, x):
        x_shape = list(x['shape'])
        x_dtype = x['dtype']
        k_shape = min(x_shape[-1], x_shape[-2])
        permutation_shape = x_shape[:-2] + [k_shape, k_shape]
        pivots_shape = x_shape[:-2] + [k_shape]
        output = {
            'shape': (x_shape, pivots_shape, permutation_shape),
            'dtype': (x_dtype, mstype.int32, mstype.int32),
            'value': None
        }
        return output


class LinearSumAssignment(Primitive):
    r"""
    Solve the linear sum assignment problem.

    The assignment problem is represented as follows:

    .. math::
        min\sum_{i}^{} \sum_{j}^{} C_{i,j} X_{i,j}

    where :math:`C` is cost matrix, :math:`X_{i,j} = 1` means column :math:`j` is assigned to row :math:`i` .

    Inputs:
        - **cost_matrix** (Tensor) - 2-D cost matrix. Tensor of shape :math:`(M, N)` .
        - **dimension_limit** (Tensor, optional) - A scalar used to limit the actual size of the 2nd dimension of
          ``cost_matrix``. Default is ``Tensor(sys.maxsize)``, which means no limitation. The type is 0-D int64
          Tensor.
        - **maximize** (bool) - Calculate a maximum weight matching if true, otherwise calculate a minimum weight
          matching.

    Outputs:
        A tuple of tensors containing 'row_idx' and 'col_idx'.

        - **row_idx** (Tensor) - Row indices of the problem. If `dimension_limit` is given, -1 would be padded at the
          end. The shape is  :math:`(N, )` , where :math:`N` is the minimum value of `cost_matrix` dimension.
        - **col_idx** (Tensor) - Column indices of the problem. If `dimension_limit` is given, -1 would be padded at
          the end. The shape is  :math:`(N, )` , where :math:`N` is the minimum value of `cost_matrix` dimension.

    Raises:
        TypeError: If the data type of `cost_matrix` is not the type in [float16, float32, float64,
                   int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool]
        TypeError: If the type of `maximize` is not bool.
        TypeError: If the data type of `dimension_limit` is not int64.
        ValueError: If the rank of `cost_matrix` is not 2.
        ValueError: If the number of input args is not 3.


    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.scipy.ops import LinearSumAssignment
        >>> lsap = LinearSumAssignment()
        >>> cost_matrix = Tensor(np.array([[2, 3, 3], [3, 2, 3], [3, 3, 2]])).astype(ms.float64)
        >>> dimension_limit = Tensor(2)
        >>> maximize = False
        >>> a, b = lsap(cost_matrix, dimension_limit, maximize)
        >>> print(a)
        [0 1 -1]
        >>> print(b)
        [0 1 -1]
    """

    @prim_attr_register
    def __init__(self):
        super().__init__(name="LinearSumAssignment")
        self.init_prim_io_names(inputs=['cost_matrix', 'dimension_limit', 'maximize'], outputs=['row_ind', 'col_ind'])
