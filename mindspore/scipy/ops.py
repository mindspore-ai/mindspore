# Copyright 2021 Huawei Technologies Co., Ltd
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
from ..ops import PrimitiveWithInfer, prim_attr_register
from .._checkparam import Validator as validator
from ..common import dtype as mstype


class SolveTriangular(PrimitiveWithInfer):
    """
    SolveTriangular op frontend implementation.

    Args:
        lower (bool): The input Matrix :math:`A` is lower triangular matrix or not.
        unit_diagonal (bool): If True, diagonal elements of :math:`A` are assumed to be 1 and
            will not be referenced.
        trans (0, 1, 2, 'N', 'T', 'C', optional):
            Type of system to solve:

            ========  =========
            trans     system
            ========  =========
            0 or 'N'  a x  = b
            1 or 'T'  a^T x = b
            2 or 'C'  a^H x = b
            ========  =========

    Inputs:
        - **A** (Tensor) - A triangular matrix of shape :math:`(N, N)`.
        - **b** (Tensor) - A tensor of shape :math:`(M,)` or :math:`(M, N)`. Right-hand side matrix in :math:`A x = b`.

    Returns:
        - **x** (Tensor) - A tensor of shape :math:`(M,)` or :math:`(M, N)`,
            which is the solution to the system :math:`A x = b`.
            Shape of :math:`x` matches :math:`b`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        Solve the lower triangular system :math:`A x = b`, where:

                 [3  0  0  0]       [4]
            A =  [2  1  0  0]   b = [2]
                 [1  0  1  0]       [4]
                 [1  1  1  1]       [2]

        >>> import numpy as onp
        >>> from mindspore.common import Tensor
        >>> import mindspore.numpy as mnp
        >>> from mindspore.scipy.ops import SolveTriangular
        >>> A = Tensor(onp.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], onp.float64))
        >>> b = Tensor(onp.array([4, 2, 4, 2], onp.float64))
        >>> solve_triangular = SolveTriangular(lower=True, unit_diagonal=False, trans='N')
        >>> x = solve_triangular(A, b)
        >>> x
        Tensor(shape=[4], dtype=Float64, value= [ 1.33333333e+00, -6.66666667e-01,  2.66666667e+00, -1.33333333e+00])
        >>> mnp.dot(A, x)  # Check the result
        Tensor(shape=[4], dtype=Float64, value= [ 4.00000000e+00,  2.00000000e+00,  4.00000000e+00,  2.00000000e+00])
    """

    @prim_attr_register
    def __init__(self, lower: bool, unit_diagonal: bool, trans: str):
        """Initialize SolveTriangular"""
        super(SolveTriangular, self).__init__("SolveTriangular")
        self.lower = validator.check_value_type(
            "lower", lower, [bool], self.name)
        self.unit_diagonal = validator.check_value_type(
            "unit_diagonal", unit_diagonal, [bool], self.name)
        self.trans = validator.check_value_type(
            "trans", trans, [str], self.name)

        self.init_prim_io_names(inputs=['A', 'b'], outputs=['output'])

    def __infer__(self, A, b):
        out_shapes = b['shape']
        return {
            'shape': tuple(out_shapes),
            'dtype': A['dtype'],
            'value': None
        }

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid(x_dtype, [mstype.float32, mstype.float64],
                                           self.name, True)
        return x_dtype
