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

"""Operators for linalg function."""

from ..operations import linalg_ops
from .._primitive_cache import _get_cache_prim


def svd(a, full_matrices=False, compute_uv=True):
    """
    Computes the singular value decompositions of one or more matrices.

    If :math:`A` is a matrix, the svd returns the singular values :math:`S`, the left singular vectors :math:`U`
    and the right singular vectors :math:`V`. It meets:

    .. math::
        A=U*diag(S)*V^{T}

    Args:
        a (Tensor): Tensor of the matrices to be decomposed. The shape should be :math:`(*, M, N)`.
        full_matrices (bool, optional): If true, compute full-sized :math:`U` and :math:`V`. If false, compute
                                        only the leading P singular vectors, with P is the minimum of M and N.
                                        Default: False.
        compute_uv (bool, optional): If true, compute the left and right singular vectors.
                                     If false, compute only the singular values. Default: True.

    Returns:
        - **s**  (Tensor) - Singular values. The shape is :math:`(*, P)`.
        - **u**  (Tensor) - Left singular vectors. If `compute_uv` is False, u will not be returned.
          The shape is :math:`(*, M, P)`. If `full_matrices` is True, the shape will be :math:`(*, M, M)`.
        - **v**  (Tensor) - Right singular vectors. If `compute_uv` is False, v will not be returned.
          The shape is :math:`(*, P, N)`. If `full_matrices` is True, the shape will be :math:`(*, N, N)`.

    Raises:
        TypeError: If `full_matrices` or `compute_uv` is not the type of bool.
        TypeError: If the rank of input less than 2.
        TypeError: If the type of input is not one of the following dtype: mstype.float32, mstype.float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, set_context
        >>> from mindspore import ops
        >>> set_context(device_target="CPU")
        >>> a = Tensor(np.array([[1, 2], [-4, -5], [2, 1]]).astype(np.float32))
        >>> s, u, v = ops.svd(a, full_matrices=True, compute_uv=True)
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
    svd_ = _get_cache_prim(linalg_ops.Svd)(full_matrices=full_matrices, compute_uv=compute_uv)

    if compute_uv:
        return svd_(a)

    s, _, _ = svd_(a)
    return s


__all__ = ['svd']

__all__.sort()
