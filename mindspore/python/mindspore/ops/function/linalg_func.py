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

import mindspore.ops as ops
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.function.math_func import _check_input_dtype, _check_attr_dtype
from mindspore._c_expression import Tensor as Tensor_

from ..operations import linalg_ops
from .._primitive_cache import _get_cache_prim


__all__ = ['svd', 'pinv']


def svd(a, full_matrices=False, compute_uv=True):
    """
    Computes the singular value decompositions of one or more matrices.

    If :math:`A` is a matrix, the svd returns the singular values :math:`S`, the left singular vectors :math:`U`
    and the right singular vectors :math:`V`. It meets:

    .. math::
        A=U*diag(S)*V^{T}

    Args:
        a (Tensor): Tensor of the matrices to be decomposed. The shape should be :math:`(*, M, N)`,
          the supported dtype are float32 and float64..
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
          The shape is :math:`(*, N, P)`. If `full_matrices` is True, the shape will be :math:`(*, N, N)`.

    Raises:
        TypeError: If `full_matrices` or `compute_uv` is not the type of bool.
        TypeError: If the rank of input less than 2.
        TypeError: If the type of input is not one of the following dtype: float32, float64.

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


def pinv(x, *, atol=None, rtol=None, hermitian=False):
    """
    Computes the (Moore-Penrose) pseudo-inverse of a matrix.

    Args:
        x (Tensor): A matrix to be calculated. The matrix must be at least two dimensions.
            Only `float32`, `float64` are supported Tensor dtypes.

    Keyword args:
        atol (float, Tensor): absolute tolerance value. Default: None.
        rtol (float, Tensor): relative tolerance value. Default: None.
        hermitian (bool): An optional bool. x is assumed to be symmetric if real. Default: False.

    Outputs:
        Tensor: same type as input. if input x(m, n), output(n, m).

    Raises:
        TypeError: If `hermitian` is not a bool.
        TypeError: If `x` is not a Tensor.
        ValueError: If the dimension of `x` is less than 2.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> x = Tensor([[4., 0.], [0., 5.]], mindspore.float32)
        >>> output = ops.pinv(x)
        >>> print(output)
        [[0.25  0. ]
        [0.  0.2 ]]
    """
    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError("The input x must be tensor")
    if x.shape == ():
        raise TypeError("For pinv, the 0-D input is not supported")
    x_shape = F.shape(x)
    if len(x_shape) < 2:
        raise ValueError("input x should have 2 or more dimensions, " f"but got {len(x_shape)}.")
    x_dtype = _get_cache_prim(P.DType)()(x)
    _check_input_dtype("x", x_dtype, [mstype.float32, mstype.float64], "pinv")
    _check_attr_dtype("hermitian", hermitian, [bool], "pinv")

    if atol is not None:
        if rtol is None:
            rtol = Tensor(0.0)
    else:
        atol = Tensor(0.0)
        if rtol is None:
            rtol = max(ops.shape(x)) * ops.Eps()(Tensor(1.0, x.dtype))

    if not inner.IsInstance()(rtol, mstype.tensor):
        rtol = Tensor(rtol, mstype.float32)
    if not inner.IsInstance()(atol, mstype.tensor):
        atol = Tensor(atol, mstype.float32)

    if not hermitian:
        s, u, v = x.svd()
        max_singular_val = ops.narrow(s, -1, 0, 1)
        threshold = ops.maximum(atol.expand_dims(-1), rtol.expand_dims(-1) * max_singular_val)
        condition = s > threshold
        reciprocal_s_before = Tensor(ops.Reciprocal()(s)).broadcast_to(condition.shape)
        zero = ops.Zeros()(condition.shape, s.dtype)
        s_pseudoinv = ops.select(condition, reciprocal_s_before, zero)
        output = ops.matmul(v * s_pseudoinv.expand_dims(-2), _nd_transpose(ops.Conj()(u)))
    else:
        s, u = _compare_eigh(x)
        s_abs = s.abs()
        max_singular_val = ops.amax(s_abs, -1, True)
        threshold = ops.maximum(atol.expand_dims(-1), rtol.expand_dims(-1) * max_singular_val)
        condition = s_abs > threshold
        reciprocal_s_before = Tensor(ops.Reciprocal()(s))
        zero = ops.Zeros()(condition.shape, s.dtype)
        s_pseudoinv = ops.select(condition, reciprocal_s_before, zero)
        output = ops.matmul(u * s_pseudoinv.expand_dims(-2), _nd_transpose(ops.Conj()(u)))
    return output


def _compare_eigh(x):
    """
    compare eigh
    """
    from mindspore.scipy.ops import Eigh
    s, u = Eigh()(x)
    return s, u


def _nd_transpose(a):
    """
    _nd_transpose
    """
    dims = a.ndim
    if dims < 2:
        raise TypeError("to do _nd_transpose for input a's ndim is not greater or equal to 2d, which is invalid.")
    axes = ops.make_range(0, dims)
    axes = axes[:-2] + (axes[-1],) + (axes[-2],)
    return ops.transpose(a, axes)


__all__.sort()
