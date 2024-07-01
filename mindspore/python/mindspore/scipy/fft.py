# Copyright 2024 Huawei Technologies Co., Ltd
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
"""fft operations, the function docs are adapted from Scipy API."""
from __future__ import absolute_import
from __future__ import division
__all__ = ['dct', 'idct']
from mindspore.ops.auto_generate import DCT
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.scipy.utils import _raise_value_error


def dct(x, type=2, n=None, axis=-1, norm=None):
    """
    Compute the Discrete Cosine Transform of input tensor x.

    Note:
        - only support type 2 Discrete Cosine Transform currently.
        - `dct` used only for `mindscience` kit.
        - `dct` dost not support Windows platform.

    Args:
        x (Tensor): Tensor to apply Discrete Cosine Transform.
        type (int, optional): Type of the DCT. Optional Value: {1, 2, 3, 4}, see `'A Fast
            Cosine Transform in One and Two Dimensions', by J. Makhoul, IEEE Transactions on
            acoustics, speech and signal processing vol. 28(1), pp. 27-34,
            <https://doi.org/10.1109/TASSP.1980.1163351>`_ . Default: ``2`` .
        n (int, optional): Length of the transform. If n < x.shape[axis],
            x is truncated. If n > x.shape[axis], x is zero-padded. Default: ``n = x.shape[axis]`` .
        axis (int, optional): Axis along which the dct is computed. Default: ``-1`` .
        norm (str, optional): Normalization mode, Optional Value: {"BACKWARD", "FORWARD", "ORTHO"}.
            Default: ``"ORTHO"`` .

    Returns:
        Tensor, the result of Discrete Cosine Transform of x.

    Raises:
        TypeError: If the `x` type is not Tensor.
        TypeError: If `n` or `dim` type is not int.
        ValueError: If `axis` is not in the range of "[ `-x.ndim` , `x.ndim` )".
        ValueError: If `n` is less than 1.
        ValueError: If `norm` is not in {"BACKWARD", "FORWARD", "ORTHO"}.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore.scipy.fft import dct
        >>> from mindspore import Tensor
        >>> x = Tensor([1, 2, 3])
        >>> output = dct(x)
        >>> print(output)
        [ 1.20000000e+01 -3.46410162e+00  3.33066907e-15]
    """
    if type != 2:
        raise NotImplementedError('Only DCT type 2 is implemented')
    if n is None:
        n = -1
    if norm is None:
        norm = "BACKWARD"

    dct_op = _get_cache_prim(DCT)()
    return dct_op(x, type, n, axis, norm, True, False)


def idct(x, type=2, n=None, axis=-1, norm=None):
    """
    Compute the inversed Discrete Cosine Transform of input tensor x.

    Note:
        - only support type 2 inversed Discrete Cosine Transform currently.
        - `norm` only support ``'ORTHO'`` (the transform will be orthogonalized).
        - `idct` used only for `mindscience` kit.
        - `idct` dost not support Windows platform.

    Args:
        x (Tensor): Tensor to apply inversed Discrete Cosine Transform.
        type (int, optional): Type of the inversed DCT. Optional Value: {1, 2, 3, 4}, see `'A Fast
            Cosine Transform in One and Two Dimensions', by J. Makhoul, IEEE Transactions on
            acoustics, speech and signal processing vol. 28(1), pp. 27-34,
            <https://doi.org/10.1109/TASSP.1980.1163351>`_ . Default: ``2`` .
        n (int, optional): Length of the transform. If :math:`n < x.shape[axis]`,
            x is truncated. If :math:`n > x.shape[axis]`, x is zero-padded. Default: ``n = x.shape[axis]`` .
        axis (int, optional): Axis along which the idct is computed. Default: ``-1`` .
        norm (str, optional): Normalization mode,
            only support ``"ORTHO"`` now. Default: ``"ORTHO"`` .

    Returns:
        Tensor, the result of inversed Discrete Cosine Transform of x.

    Raises:
        TypeError: If the `x` type is not Tensor.
        TypeError: If `n` or `dim` type is not int.
        ValueError: If `axis` is not in the range of :math:`[-x.ndim, x.ndim)`.
        ValueError: If `n` is less than 1.
        ValueError: If `norm` is not "ORTHO".

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore.scipy.fft import idct
        >>> from mindspore import Tensor
        >>> x = Tensor([1, 2, 3])
        >>> output = idct(x)
        >>> print(output)
        [ 3.2163087  -1.87213947  0.38788158]
    """
    if type != 2:
        raise NotImplementedError('Only DCT type 2 is implemented')
    if n is None:
        n = -1
    if norm is None:
        norm = "ORTHO"
    norm = norm.upper()
    if norm != "ORTHO":
        _raise_value_error(f'norm should be \"ORTHO\", but got {norm}')

    dct_op = _get_cache_prim(DCT)()
    return dct_op(x, type, n, axis, norm, False, False)
