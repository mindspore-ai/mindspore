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

"""Operators for linalg."""

from ..._checkparam import Validator
from ..primitive import Primitive, prim_attr_register


class Svd(Primitive):
    """
    Refer to :func:`mindspore.ops.svd` for more detail.

    Args:
        full_matrices (bool, optional): If true, compute full-sized :math:`U` and :math:`V`. If false, compute
                                        only the leading P singular vectors. P is the minimum of M and N.
                                        M, N is the row, col of the input matrix. Default: False.
        compute_uv (bool, optional): If true, compute the left and right singular vectors.
                                     If false, compute only the singular values. Default: True.

    Inputs:
        - **a**  (Tensor): Tensor of the matrices to be decomposed. The shape should be :math:`(*, M, N)`.

    Outputs:
        - **s**  (Tensor) - Singular values. The shape is :math:`(*, P)`.
        - **u**  (Tensor) - Left singular vectors. If compute_uv is False, u will be an empty tensor.
          The shape is :math:`(*, M, P)`. If full_matrices is True, the shape will be :math:`(*, M, M)`.
        - **v**  (Tensor) - Right singular vectors. If compute_uv is False, v will be an empty tensor.
          The shape is :math:`(*, P, N)`. If full_matrices is True, the shape will be :math:`(*, N, N)`.

    Raises:
        TypeError: If full_matrices or compute_uv is not the type of bool.
        TypeError: If the rank of input less than 2.
        TypeError: If the type of input is not one of the following dtype: mstype.float32, mstype.float64.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, context
        >>> from mindspore.ops.operations import linalg_ops as linalg
        >>> context.set_context(device_target="CPU")
        >>> svd = linalg.Svd(full_matrices=True, compute_uv=True)
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
