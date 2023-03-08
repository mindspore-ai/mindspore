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
from __future__ import absolute_import

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


__all__ = ['svd', 'pinv', 'qr']


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
    r"""
    Computes the (Moore-Penrose) pseudo-inverse of a matrix.
    This function is computed using SVD. If :math:`x=U*S*V^{T}` ,Than the pseudo-inverse of x is:
    :math:`x^{+}=V*S^{+}*U^{T}` , :math:`S^{+}` is the reciprocal of each non-zero element on
    the diagonal of S, and zero remains in place.
    Batch matrices are supported. If x is a batch matrix, the output has the same batch dimension when
    atol or rtol is float.
    If atol or rtol is a Tensor, its shape must be broadcast to the singular value returned by
    `x.svd <https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.svd.html>`_ .
    If x.shape is :math:`(B, M, N)`, and the shape of atol or rtol is :math:`(K, B)`, the output
    shape is :math:`(K, B, N, M)`.
    When the Hermitian is True, temporary support only real domain, x is treated as a real symmetric, so x is
    not checked internally, and only use the lower triangular part in the computations.
    When the singular value of x (or the norm of the eigenvalues when hermitian = True) that are below
    threshold (:math:`Max(atol, \sigma \cdot rtol)`, :math:`\sigma` as the largest singular value or
    characteristic value), it is set to zero, and is not used in the computations.
    If rtol is not specified and x is a matrix of dimensions (M, N), then rtol is set to
    be :math:`rtol=max(M, N)*\varepsilon`, :math:`\varepsilon` is the
    `eps <https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Eps.html>`_ value of x.dtype.
    If rtol is not specified and atol specifies a value larger than zero, rtol is set to zero.

    .. note::
        This function uses
        `svd <https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.svd.html>`_ internally,
        (or `eigh <https://www.mindspore.cn/docs/en/master/api_python/scipy/mindspore.scipy.linalg.eigh.html>`_ ,
        when hermitian = True). So it has the same problem as these functions. For details,
        see the warnings in svd() and eigh().

    Args:
        x (Tensor): A matrix to be calculated. Only `float32`, `float64` are supported Tensor dtypes.
            shape is :math:`(*, M, N)`, * is zero or more batch dimensions.

            - When hermitian is true, batch dimensions are not supported temporarily.

    Keyword args:
        atol (float, Tensor): absolute tolerance value. Default: None.
        rtol (float, Tensor): relative tolerance value. Default: None.
        hermitian (bool): An optional bool. x is assumed to be symmetric if real. Default: False.

    Outputs:
        - **output** (Tensor) - same type as input. Shape is :math:`(*, N, M)`, * is zero or more batch dimensions.

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


def qr(input, mode='reduced'):
    """
    Returns the QR decomposition of one or more matrices.
    If `mode` is 'reduced'(the default), compute the P columns of Q where P is minimum of the 2 innermost dimensions of
    input. If `mode` is 'complete', compute full-sized Q and R.


    Args:
        - **input** (Tensor) - A matrix to be calculated. The matrix must be at least two dimensions.
          types: float16, float32, float64, complex64, complex128.
          Define the shape of input as (..., m, n), p as the minimum values of m and n.
        - **mode** (str, optional) - Whether compute reduce-sized QR decomposition. Default: 'reduced'.


    Returns:
        - **Q** (Tensor) - The orthonormal matrices of input.
          If `some` is false, the shape is :math:`(m, m)`, else the shape is :math:`(m, p)`.
          The dtype of `Q` is same as `input`.
        - **R** (Tensor) - The upper triangular matrices of input.
          If `some` is false, the shape is :math:`(m, n)`, else the shape is :math:`(p, n)`.
          The dtype of `R` is same as `input`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `mode` is neither 'reduced' nor 'complete'.
        ValueError: If the dimension of `input` is less than 2.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> input = Tensor(np.array([[20., -31, 7], [4, 270, -90], [-8, 17, -32]]), mstype.float32)
        >>> Q, R = ops.qr(input)
        >>> print(Q)
        [[-0.912871    0.16366126  0.37400758]
         [-0.18257418 -0.9830709  -0.01544376]
         [ 0.36514837 -0.08238228  0.92729706]]
        >>> print(R)
        [[ -21.908903  -14.788506  -1.6431675]
        [    0.       -271.9031    92.25824  ]
        [    0.          0.       -25.665514 ]]
    """
    if mode not in ('reduced', 'complete'):
        raise TypeError(f"For qr, the arg mode must be 'reduced' or 'complete', but got {mode}.")
    qr_ = _get_cache_prim(P.Qr)(mode == 'complete')
    return qr_(input)


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
