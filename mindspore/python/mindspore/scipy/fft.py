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

from mindspore.ops.auto_generate import DCT, IDCT, DCTN, IDCTN
from mindspore.ops._primitive_cache import _get_cache_prim

__all__ = ['dct', 'idct', 'dctn', 'idctn']


def dct(x, type=2, n=None, axis=-1, norm=None):
    """
    Compute the Discrete Cosine Transform of input tensor x.

    Note:
        - `dct` used only for `mindscience` kit.
        - `dct` dost not support Windows platform.

    Args:
        x (Tensor): Tensor to apply Discrete Cosine Transform.
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64, complex64, complex128.

        type (int, optional): Type of the DCT. Only type 2 is supported currently, see `'A Fast
            Cosine Transform in One and Two Dimensions', by J. Makhoul, IEEE Transactions on
            acoustics, speech and signal processing vol. 28(1), pp. 27-34,
            <https://doi.org/10.1109/TASSP.1980.1163351>`_ . Default: ``2`` .
        n (int, optional): Length of the transform. If n < x.shape[axis],
            x is truncated. If n > x.shape[axis], x is zero-padded. Default: ``n = x.shape[axis]`` .
            Note that for Ascend backend parameter `n` is required.
        axis (int, optional): Axis along which the dct is computed. Default: ``-1`` .
        norm (string, optional): Normalization mode. Only "ortho" is supported currently.
            Default: ``None`` that means ``"ortho"`` .

    Returns:
        Tensor, The result of `dct()` function. The default is the same shape as `x`.
        If `n` is given, the size of the `axis` is changed to `n`.
        When the `x` is int16, int32, int64, float16, float32, the return value type is float32.
        When the `x` is float64, the return value type is float64.
        When the `x` is complex64/128, the return value type is complex64/128.

    Raises:
        TypeError: If the `x` type is not Tensor.
        TypeError: If the `x` data type is not one of supported dtypes.
        TypeError: If `n` or `axis` type is not int.
        ValueError: If `type` is not `2` .
        ValueError: If `dim` is not in the range of "[ `-x.ndim` , `x.ndim` )".
        ValueError: If `n` is less than 1.
        ValueError: If `norm` is not ``"ortho"`` .

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore.scipy.fft import dct
        >>> from mindspore import Tensor
        >>> x = Tensor([1, 2, 3])
        >>> output = dct(x, 2, 3, -1, "ortho")
        >>> print(output)
        [ 3.4641018e+00 -1.4142137e+00 -5.9604645e-08]
    """
    dct_op = _get_cache_prim(DCT)()
    return dct_op(x, type, n, axis, norm)


def idct(x, type=2, n=None, axis=-1, norm=None):
    """
    Compute the inversed Discrete Cosine Transform of input tensor x.

    Note:
        - `idct` used only for `mindscience` kit.
        - `idct` dost not support Windows platform.

    Args:
        x (Tensor): Tensor to apply inversed Discrete Cosine Transform.
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64, complex64, complex128.

        type (int, optional): Type of the inversed DCT. Only type 2 is supported currently, see `'A Fast
            Cosine Transform in One and Two Dimensions', by J. Makhoul, IEEE Transactions on
            acoustics, speech and signal processing vol. 28(1), pp. 27-34,
            <https://doi.org/10.1109/TASSP.1980.1163351>`_ . Default: ``2`` .
        n (int, optional): Length of the transform. If :math:`n < x.shape[axis]`,
            x is truncated. If :math:`n > x.shape[axis]`, x is zero-padded. Default: ``n = x.shape[axis]`` .
            Note that for Ascend backend parameter `n` is required.
        axis (int, optional): Axis along which the idct is computed. Default: ``-1`` .
        norm (string, optional): Normalization mode. Only "ortho" is supported currently.
            Default: ``None`` that means ``"ortho"`` .

    Returns:
        Tensor, The result of `idct()` function. The default is the same shape as `x`.
        If `n` is given, the size of the `axis` is changed to `n`.
        When the `x` is int16, int32, int64, float16, float32, the return value type is float32.
        When the `x` is float64, the return value type is float64.
        When the `x` is complex64/128, the return value type is complex64/128.

    Raises:
        TypeError: If the `x` type is not Tensor.
        TypeError: If the `x` data type is not one of supported dtypes.
        TypeError: If `n` or `axis` type is not int.
        ValueError: If `type` is not `2` .
        ValueError: If `dim` is not in the range of "[ `-x.ndim` , `x.ndim` )".
        ValueError: If `n` is less than 1.
        ValueError: If `norm` is not ``"ortho"`` .

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore.scipy.fft import idct
        >>> from mindspore import Tensor
        >>> x = Tensor([1, 2, 3])
        >>> output = idct(x, 2, 3, -1, "ortho")
        >>> print(output)
        [ 3.2163088  -1.8721395   0.38788158]
    """

    dct_op = _get_cache_prim(IDCT)()
    return dct_op(x, type, n, axis, norm)


def dctn(x, type=2, s=None, axes=None, norm=None):
    """
    Compute the N dimension Discrete Cosine Transform of input tensor x.

    Note:
        - `dctn` used only for `mindscience` kit.
        - `dctn` dost not support Windows platform.

    Args:
        x (Tensor): Tensor to apply N dimension Discrete Cosine Transform.
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64, complex64, complex128.

        type (int, optional): Type of the DCT. Only type 2 is supported currently, see `'A Fast
            Cosine Transform in One and Two Dimensions', by J. Makhoul, IEEE Transactions on
            acoustics, speech and signal processing vol. 28(1), pp. 27-34,
            <https://doi.org/10.1109/TASSP.1980.1163351>`_ . Default: ``2`` .
        s (tuple[int], optional): Length of the transformed `axes` of the result.
            If given, the input will either be zero-padded or trimmed to this length before computing `dctn`.
            Default: ``None`` , which does not need to process `x`.
            Note that for Ascend backend parameter `s` is required.
        axes (tuple[int], optional): The dimension along which to take the one dimensional `dctn`.
            Default: ``None`` , which means transform the all dimension of `x`,
            or the last `len(s)` dimensions if s is given.
            Note that for Ascend backend parameter `axes` is required.
        norm (string, optional): Normalization mode. Only "ortho" is supported currently.
            Default: ``None`` that means ``"ortho"`` .

    Raises:
        TypeError: If the `x` type is not Tensor.
        TypeError: If the `x` data type is not one of supported dtypes.
        TypeError: If the type/dtype of `s` and `axes` is not int.
        ValueError: If `type` is not `2` .
        ValueError: If `axes` is not in the range of "[ `-x.ndim` , `x.ndim` )".
        ValueError: If `axes` has duplicate values.
        ValueError: If `s` is less than 1.
        ValueError: If `s` and `axes` are given but have different shapes.
        ValueError: If `norm` is not ``"ortho"`` .

    Returns:
        Tensor, The result of `dctn()` function. The default is the same shape as `x`.
        If `s` is given, the size of the `axes[i]` axis is changed to `s[i]`.
        When the `x` is int16, int32, int64, float16, float32, the return value type is float32.
        When the `x` is float64, the return value type is float64.
        When the `x` is complex64/128, the return value type is complex64/128.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore.scipy.fft import dctn
        >>> from mindspore import Tensor
        >>> x = Tensor([[1, 2, 3], [2, 1, 1]])
        >>> output = dctn(x, 2, s=(2, 3), axes=(0,1), norm="ortho")
        >>> print(output)
        [[ 4.082483   -0.49999988  0.28867507]
         [ 0.81649655 -1.4999999  -0.28867507]]
    """
    dct_op = _get_cache_prim(DCTN)()
    return dct_op(x, type, s, axes, norm)


def idctn(x, type=2, s=None, axes=None, norm=None):
    """
    Compute the N dimension inversed Discrete Cosine Transform of input tensor x.

    Note:
        - `dctn` used only for `mindscience` kit.
        - `dctn` dost not support Windows platform.

    Args:
        x (Tensor): Tensor to apply N dimension inversed Discrete Cosine Transform.
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64, complex64, complex128.

        type (int, optional): Type of the IDCT. Only type 2 is supported currently, see `'A Fast
            Cosine Transform in One and Two Dimensions', by J. Makhoul, IEEE Transactions on
            acoustics, speech and signal processing vol. 28(1), pp. 27-34,
            <https://doi.org/10.1109/TASSP.1980.1163351>`_ . Default: ``2`` .
        s (tuple[int], optional): Length of the transformed `axes` of the result.
            If given, the input will either be zero-padded or trimmed to this length before computing `idctn`.
            Default: ``None`` , which does not need to process `x`.
            Note that for Ascend backend parameter `s` is required.
        axes (tuple[int], optional): The dimension along which to take the one dimensional `idctn`.
            Default: ``None`` , which means transform the all dimension of `x`,
            or the last `len(s)` dimensions if s is given.
            Note that for Ascend backend parameter `axes` is required.
        norm (string, optional): Normalization mode. Only "ortho" is supported currently.
            Default: ``None`` that means ``"ortho"`` .

    Raises:
        TypeError: If the `x` type is not Tensor.
        TypeError: If the `x` data type is not one of supported dtypes.
        TypeError: If the type/dtype of `s` and `axes` is not int.
        ValueError: If `type` is not `2` .
        ValueError: If `axes` is not in the range of "[ `-x.ndim` , `x.ndim` )".
        ValueError: If `axes` has duplicate values.
        ValueError: If `s` is less than 1.
        ValueError: If `s` and `axes` are given but have different shapes.
        ValueError: If `norm` is not ``"ortho"`` .

    Returns:
        Tensor, The result of `idctn()` function. The default is the same shape as `x`.
        If `s` is given, the size of the `axes[i]` axis is changed to `s[i]`.
        When the `x` is int16, int32, int64, float16, float32, the return value type is float32.
        When the `x` is float64, the return value type is float64.
        When the `x` is complex64/128, the return value type is complex64/128.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore.scipy.fft import idctn
        >>> from mindspore import Tensor
        >>> x = Tensor([[1, 2, 3], [2, 1, 1]])
        >>> output = idctn(x, 2, s=(2, 3), axes=(0,1), norm="ortho")
        >>> print(output)
        [[ 3.8794453  -1.0846562   0.8794453 ]
         [ 0.66910195 -1.5629487  -0.33089805]]
    """

    dct_op = _get_cache_prim(IDCTN)()
    return dct_op(x, type, s, axes, norm)
