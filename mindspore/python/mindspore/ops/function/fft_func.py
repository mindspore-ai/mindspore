# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Defines Fast Fourier Transform operators with functional form."""
from mindspore.ops.auto_generate import FFTBase, fftshift, ifftshift
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.function.math_func import _check_attr_dtype


def fft(input, n=None, dim=-1, norm=None):
    r"""
    Calculates the one dimensional discrete Fourier transform of `input`.

    Note:
        - `fft` is currently only used in `mindscience` scientific computing scenarios and
          dose not support other usage scenarios.
        - `fft` is not supported on Windows platform yet.

    Args:
        input (Tensor): The input tensor.
        n (int, optional): Length of the transformed `dim` of the result.
            If given, the input will either be zero-padded or trimmed to this length before computing `fft`.
            Default: ``None``.
        dim (int, optional): The dimension along which to take the one dimensional `fft`.
            Default: ``-1``, which means transform the last dimension of `input`.
        norm (string, optional): Normalization mode. Default: ``None`` that means ``"backward"``.
            Three modes are defined as,

            - ``"backward"``(no normalization).
            - ``"forward"`` (normalize by :math:`1/n`).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, The result of `fft()` function.

    Raises:
        TypeError: If the `input` type is not Tensor.
        TypeError: If the `input` data type is not one of: int32, int64, float32, float64, complex64, complex128.
        TypeError: If `n` or `dim` type is not int.
        ValueError: If `dim` is not in the range of "[ `-input.ndim` , `input.ndim` )".
        ValueError: If `n` is less than 1.
        ValueError: If `norm` is none of ``"backward"`` , ``"forward"`` or ``"ortho"``.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> input = Tensor([ 1.6243454+0.j, -0.6117564+0.j, -0.5281718+0.j, -1.0729686+0.j])
        >>> y = ops.fft(input)
        >>> print(y)
        [-0.5885514+0.j          2.1525172-0.46121222j  2.7808986+0.j
         2.1525172+0.46121222j]
    """
    fft_name = "fft"
    fft_op = _get_cache_prim(FFTBase)(fft_mode=fft_name, forward=True)
    if norm is None:
        norm = "backward"
    if n is None:
        n = ()
    else:
        _check_attr_dtype("n", n, [int], fft_name)
    _check_attr_dtype("dim", dim, [int], fft_name)
    return fft_op(input, n, dim, norm)


def ifft(input, n=None, dim=-1, norm=None):
    r"""
    Calculates the inverse of `fft()`.

    Note:
        - `ifft` is currently only used in `mindscience` scientific computing scenarios and
          dose not support other usage scenarios.
        - `ifft` is not supported on Windows platform yet.

    Args:
        input (Tensor): The input tensor.
        n (int, optional): Length of the transformed `dim` of the result.
            If given, the input will either be zero-padded or trimmed to this length before computing the `ifft`.
            Default: ``None``.
        dim (int, optional): The dimension along which to take the one dimensional `ifft`.
            Default: ``-1``, which means transform the last dimension of `input`.
        norm (string, optional): Normalization mode. Default: ``None`` that means ``"backward"``.
            Three modes are defined as,

            - ``"backward"``(no normalization).
            - ``"forward"`` (normalize by :math:`1/n`).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, The result of `ifft()` function.

    Raises:
        TypeError: If the `input` type is not Tensor.
        TypeError: If the `input` data type is not one of: int32, int64, float32, float64, complex64, complex128.
        TypeError: If `n` or `dim` type is not int.
        ValueError: If `dim` is not in the range of "[ `-input.ndim` , `input.ndim` )".
        ValueError: If `n` is less than 1.
        ValueError: If `norm` is none of ``"backward"`` , ``"forward"`` or ``"ortho"``.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> input = Tensor([ 1.6243454+0.j, -0.6117564+0.j, -0.5281718+0.j, -1.0729686+0.j])
        >>> y = ops.ifft(input)
        >>> print(y)
        [-0.14713785+0.j          0.5381293 +0.11530305j  0.69522465+0.j
         0.5381293 -0.11530305j]
    """
    fft_name = "ifft"
    ifft_op = _get_cache_prim(FFTBase)(fft_mode=fft_name, forward=False)
    if norm is None:
        norm = "backward"
    if n is None:
        n = ()
    else:
        _check_attr_dtype("n", n, [int], fft_name)
    _check_attr_dtype("dim", dim, [int], fft_name)
    return ifft_op(input, n, dim, norm)


__all__ = [
    'fftshift',
    'ifftshift',
    'fft',
    'ifft'
]

__all__.sort()
