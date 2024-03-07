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
"""Fast Fourier Transform operations, the function docs are adapted from Numpy API."""
from __future__ import absolute_import
from mindspore import ops

__all__ = ['fftshift', 'ifftshift', 'fft', 'ifft']

def fftshift(x, axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum.

    Refer to :func:`mindspore.ops.fftshift` for more details.
    The difference is that `x` corresponds to `input` and `axes` corresponds to `dim`.

    Args:
        x (Tensor): Input tensor.
        axes (Union[int, list(int), tuple(int)], optional): Axes over which to shift.
            Default is ``None`` , which shifts all axes.

    Returns:
        output (Tensor), the shifted tensor with the same shape and dtype as `x`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> from mindspore import dtype as mstype
        >>> x = np.array([0, 1, 2, 3, 4, -5, -4, -3, -2, -1], dtype=mstype.int32)
        >>> np.fft.fftshift(x)
        Tensor(shape=[10], dtype=Int32, value= [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
    """
    return ops.fftshift(x, axes)


def ifftshift(x, axes=None):
    """
    The inverse of fftshift.

    Refer to :func:`mindspore.ops.ifftshift` for more details.
    The difference is that `x` corresponds to `input` and `axes` corresponds to `dim`.

    Args:
        x (Tensor): Input tensor.
        axes (Union[int, list(int), tuple(int)], optional): Axes over which to shift.
            Default is ``None`` , which shifts all axes.

    Returns:
        output (Tensor), the shifted tensor with the same shape and dtype as `x`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> from mindspore import dtype as mstype
        >>> x = np.array([0, 1, 2, 3, 4, -5, -4, -3, -2, -1], dtype=mstype.int32)
        >>> np.fft.ifftshift(np.fft.fftshift(x))
        Tensor(shape=[10], dtype=Int32, value= [ 0, 1, 2, 3, 4, -5, -4, -3, -2, -1])
    """
    return ops.ifftshift(x, axes)


def fft(a, n=None, axis=-1, norm=None):
    r"""
    Calculates the one dimensional discrete Fourier transform of `a`.

    Refer to :func:`mindspore.ops.fft` for more details.
    The difference is that `a` corresponds to `input` and `axis` corresponds to `dim`.

    Args:
        a (Tensor): The input tensor.
        n (int, optional): Length of the transformed `axis` of the result.
            If given, the input will either be zero-padded or trimmed to this length before computing `fft`.
            Default: ``None``, which does not need to process `a`.
        axis (int, optional): Axis over which to compute the `fft`.
            Default: ``-1``, which means the last axis of `a` is used.
        norm (string, optional): Normalization mode. Default: ``None`` that means ``"backward"``.
            Three modes are defined as,

            - ``"backward"``(no normalization).
            - ``"forward"`` (normalize by :math:`1/n`).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, The result of `fft()` function.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> input = np.array([ 1.6243454, -0.6117564, -0.5281718, -1.0729686])
        >>> np.fft.fft(input)
        Tensor(shape=[4], dtype=Complex64, value= [-0.588551+0j, 2.15252-0.461212j, 2.7809+0j, 2.15252+0.461212j])
    """
    return ops.fft(a, n, axis, norm)


def ifft(a, n=None, axis=-1, norm=None):
    r"""
    Calculates the inverse of `fft()`.

    Refer to :func:`mindspore.ops.ifft` for more details.
    The difference is that `a` corresponds to `input` and `axis` corresponds to `dim`.

    Args:
        a (Tensor): The input tensor.
        n (int, optional): Signal length.
            If given, the input will either be zero-padded or trimmed to this length before computing `ifft`.
            Default: ``None``, which does not need to process `a`.
        axis (int, optional): Axis over which to compute the `ifft`.
            Default: ``-1``, which means the last axis of `a` is used.
        norm (string, optional): Normalization mode. Default: ``None`` that means ``"backward"``.
            Three modes are defined as,

            - ``"backward"``(no normalization).
            - ``"forward"`` (normalize by :math:`1*n`).
            - ``"ortho"`` (normalize by :math:`1*\sqrt{n}`).

    Returns:
        Tensor, The result of `ifft()` function.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> input = np.array([ 1.6243454, -0.6117564, -0.5281718, -1.0729686])
        >>> np.fft.ifft(input)
        Tensor(shape=[4], dtype=Complex64, value= [-0.147138+0j, 0.538129+0.115303j, 0.695225+0j, 0.538129-0.115303j])
    """
    return ops.ifft(a, n, axis, norm)
