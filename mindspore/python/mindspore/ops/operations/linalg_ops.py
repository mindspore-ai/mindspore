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

"""Operators for linalg."""

from __future__ import absolute_import
from mindspore import _checkparam as Validator
from mindspore.ops.primitive import Primitive
from mindspore.ops.primitive import prim_attr_register


class Geqrf(Primitive):
    r"""
    Decomposes a matrix into the product of an orthogonal matrix `Q` and an upper triangular matrix `R`.
    The process is called QR decomposition: :math:`A = QR`.

    Both `Q` and `R` matrices are stored in the same output tensor `y`.
    The elements of `R` are stored on and above the diagonal, whereas elementary reflectors
    (or Householder vectors) implicitly defining matrix `Q` are stored below the diagonal.

    This function returns two tensors (`y`, `tau`).

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, m, n)`, input must be a matrix greater than or equal to 2D,
          with dtype of float32, float64, complex64, complex128.

    Outputs:
        - **y** (Tensor) - Tensor of shape :math:`(*, m, n)`, has the same dtype as the `x`.
        - **tau** (Tensor) - Tensor of shape :math:`(*, p)` and :math:`p = min(m, n)`, has the same dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If the dtype of `x` is not float32, float64, complex64 or complex128.
        ValueError: If dimension of `x` is less than 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.array([[-2.0, -1.0], [1.0, 2.0]]).astype(np.float32))
        >>> geqrf = ops.Geqrf()
        >>> y, tau = geqrf(input_x)
        >>> print(y)
        [[ 2.236068   1.7888544]
         [-0.236068   1.3416407]]
        >>> print(tau)
        [1.8944271 0.       ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Geqrf"""
        self.init_prim_io_names(inputs=['x'], outputs=['y', 'tau'])


class Svd(Primitive):
    """
    Computes the singular value decompositions of one or more matrices.

    Refer to :func:`mindspore.ops.svd` for more details.

    Args:
        full_matrices (bool, optional): If ``True`` , compute full-sized :math:`U` and :math:`V`. If ``False``,
                                        compute only the leading P singular vectors, with P is the minimum of M and N.
                                        Default: ``False`` .
        compute_uv (bool, optional): If ``True`` , compute the left and right singular vectors.
                                     If ``False`` , compute only the singular values. Default: ``True`` .

    Inputs:
        - **input** (Tensor) - Tensor of the matrices to be decomposed. The shape should be :math:`(*, M, N)`,
          the supported dtype are float32 and float64.

    Outputs:
        - **s**  (Tensor) - Singular values. The shape is :math:`(*, P)`.
        - **u**  (Tensor) - Left singular vectors. If `compute_uv` is ``False`` , u will be zero value.
          The shape is :math:`(*, M, P)`. If `full_matrices` is ``True`` , the shape will be :math:`(*, M, M)`.
        - **v**  (Tensor) - Right singular vectors. If `compute_uv` is ``False`` , v will be zero value.
          The shape is :math:`(*, N, P)`. If `full_matrices` is ``True`` , the shape will be :math:`(*, N, N)`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, set_context
        >>> from mindspore import ops
        >>> set_context(device_target="CPU")
        >>> svd = ops.Svd(full_matrices=True, compute_uv=True)
        >>> a = Tensor(np.array([[1, 2], [-4, -5], [2, 1]]).astype(np.float32))
        >>> s, u, v = svd(a)
        >>> print(s)
        [7.0652843 1.040081 ]
        >>> print(u)
        [[ 0.30821905 -0.48819482 0.81649697]
         [-0.90613353  0.11070572 0.40824813]
         [ 0.2896955   0.8656849  0.4082479 ]]
        >>> print(v)
        [[ 0.63863593 0.769509  ]
         [ 0.769509  -0.63863593]]
    """

    @prim_attr_register
    def __init__(self, full_matrices=False, compute_uv=True):
        super().__init__(name="Svd")
        self.init_prim_io_names(inputs=['a'], outputs=['s', 'u', 'v'])
        self.full_matrices = Validator.check_value_type("full_matrices", full_matrices, [bool], self.name)
        self.compute_uv = Validator.check_value_type("compute_uv", compute_uv, [bool], self.name)
        self.add_prim_attr('full_matrices', self.full_matrices)
        self.add_prim_attr('compute_uv', self.compute_uv)


class Eigh(Primitive):
    """
    Eigh decomposition(Symmetric matrix)
    Ax = lambda * x
    """

    @prim_attr_register
    def __init__(self, compute_eigenvectors=True, lower=True):
        super().__init__(name="Eigh")
        self.init_prim_io_names(inputs=['A'], outputs=['output_w', 'output_v'])
        self.compute_eigenvectors = Validator.check_value_type(
            "compute_eigenvectors", compute_eigenvectors, [bool], self.name)
        self.lower = Validator.check_value_type("lower", lower, [bool], self.lower)
        self.add_prim_attr('lower', self.lower)
        self.add_prim_attr('compute_eigenvectors', self.compute_eigenvectors)
