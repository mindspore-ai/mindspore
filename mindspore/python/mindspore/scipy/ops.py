# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
from ..common import dtype as mstype


class SolveTriangular(Primitive):
    """
    Solve the equation `a x = b` for `x`, assuming a is a triangular matrix.

    Args:
        a (Tensor): A triangular matrix of shape :math:`(..., N, N)`.
        b (Tensor): A Tensor of shape :math:`(M,)` or :math:`(..., N, M)`.
            Right-hand side matrix in :math:`a x = b`.
        lower (bool, optional): Use only data contained in the lower triangle of `a`.
            Default is to use upper triangle.
        trans (0, 1, 2, 'N', 'T', 'C', optional):
            Type of system to solve:
            trans:        system:
                0 or 'N'        a x  = b
                1 or 'T'        a^T x = b
                2 or 'C'        a^H x = b
        unit_diagonal (bool, optional): If True, diagonal elements of :math:`a` are assumed to be 1 and
            will not be referenced.
        overwrite_b (bool, optional): Allow overwriting data in :math:`b` (may enhance performance)
        check_finite (bool, optional): Whether to check that the input matrices contain only finite numbers.
            Disabling may give a performance gain, but may result in problems
            (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns:
        Tensor of shape :math:`(..., M,)` or :math:`(..., M, N)`,
        which is the solution to the system :math:`a x = b`.
        Shape of :math:`x` matches :math:`b`.

    Raises:
        LinAlgError: If :math:`a` is singular

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        Solve the lower triangular system :math:`a x = b`, where:

                 [3  0  0  0]       [4]
            a =  [2  1  0  0]   b = [2]
                 [1  0  1  0]       [4]
                 [1  1  1  1]       [2]

        >>> import numpy as onp
        >>> from mindspore import Tensor
        >>> import mindspore.numpy as mnp
        >>> from mindspore.scipy.ops import SolveTriangular
        >>> a = Tensor(onp.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], onp.float64))
        >>> b = Tensor(onp.array([4, 2, 4, 2], onp.float64))
        >>> solve_triangular = SolveTriangular(lower=True, unit_diagonal=False, trans='N')
        >>> x = solve_triangular(a, b)
        >>> print(x)
        [ 1.33333333 -0.66666667  2.66666667 -1.33333333]
        >>> print(mnp.dot(a, x))  # Check the result
        [4. 2. 4. 2.]
    """

    @prim_attr_register
    def __init__(self, lower: bool = False, unit_diagonal: bool = False, trans: str = 'N'):
        """Initialize SolveTriangular"""
        super(SolveTriangular, self).__init__("SolveTriangular")
        self.lower = validator.check_value_type(
            "lower", lower, [bool], self.name)
        self.unit_diagonal = validator.check_value_type(
            "unit_diagonal", unit_diagonal, [bool], self.name)
        self.trans = validator.check_value_type(
            "trans", trans, [str], self.name)

        self.init_prim_io_names(inputs=['a', 'b'], outputs=['output'])


class Eig(PrimitiveWithInfer):
    """
    Eig decomposition,(generic matrix)
    a * v = w * v
    """

    @prim_attr_register
    def __init__(self, compute_v=True):
        super().__init__(name="Eig")
        self.init_prim_io_names(inputs=['a'], outputs=['w', 'v'])
        self.compute_v = validator.check_value_type("compute_v", compute_v, [bool], self.name)
        self.add_prim_attr('compute_v', self.compute_v)
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

# pylint: disable=C0413,W0611
from .ops_grad import get_bprpo_eigh, get_bprpo_trsm
