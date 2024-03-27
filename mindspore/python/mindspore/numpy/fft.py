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
__all__ = ['fftshift', 'ifftshift', 'fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn']
from mindspore import ops


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
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64, complex64, complex128.

        n (int, optional): Length of the transformed `dim` of the result.
            If given, the size of the `dim` axis will be zero-padded or truncated to `n` before calculating `fft`.
            Default: ``None`` , which does not need to process `a`.
        axis (int, optional): Axis over which to compute the `fft`.
            Default: ``-1`` , which means the last axis of `a` is used.
        norm (string, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as,

            - ``"backward"`` (no normalization).
            - ``"forward"`` (normalize by :math:`1/n`).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, The result of `fft()` function. The default is the same shape as `a`.
        If `n` is given, the size of the `axis` is changed to `n`.
        When the `a` is int16, int32, int64, float16, float32, complex64, the return value type is complex64.
        When the `a` is float64 or complex128, the return value type is complex128.

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
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64, complex64, complex128.

        n (int, optional): Length of the transformed `dim` of the result.
        n (int, optional): Signal length.
            If given, the input will either be zero-padded or trimmed to this length before computing `ifft`.
            Default: ``None`` , which does not need to process `a`.
        axis (int, optional): Axis over which to compute the `ifft`.
            Default: ``-1`` , which means the last axis of `a` is used.
        norm (string, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as,

            - ``"backward"`` (no normalization).
            - ``"forward"`` (normalize by :math:`1*n`).
            - ``"ortho"`` (normalize by :math:`1*\sqrt{n}`).

    Returns:
        Tensor, The result of `ifft()` function. The default is the same shape as `a`.
        If `n` is given, the size of the `axis` is changed to `n`.
        When the `a` is int16, int32, int64, float16, float32, complex64, the return value type is complex64.
        When the `a` is float64 or complex128, the return value type is complex128.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> input = np.array([ 1.6243454, -0.6117564, -0.5281718, -1.0729686])
        >>> np.fft.ifft(input)
        Tensor(shape=[4], dtype=Complex64, value= [-0.147138+0j, 0.538129+0.115303j, 0.695225+0j, 0.538129-0.115303j])
    """
    return ops.ifft(a, n, axis, norm)


def rfft(a, n=None, axis=-1, norm=None):
    r"""
    Calculates the one dimensional discrete Fourier transform for real input `a`.

    Refer to :func:`mindspore.ops.rfft` for more details.
    The difference is that `a` corresponds to `input` and `axis` corresponds to `dim`.

    Args:
        a (Tensor): The input tensor.
        n (int, optional): Number of points along `axis` in the input to use.
            If given, the input will either be zero-padded or trimmed to this length before computing `rfft`.
            Default: ``None``.
        axis (int, optional): Axis over which to compute the `rfft`.
            Default: ``-1``, which means the last axis of `a` is used.
        norm (string, optional): Normalization mode. Default: ``None`` that means ``"backward"``.
            Three modes are defined as,

            - ``"backward"`` (no normalization).
            - ``"forward"`` (normalize by :math:`1/n`).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, the result of `rfft()` function.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore import numpy as mnp
        >>> input = Tensor([1, 2, 3, 4])
        >>> y = mnp.fft.rfft(input)
        >>> print(y)
        [10.+0.j -2.+2.j -2.+0.j]
    """
    return ops.rfft(a, n, axis, norm)


def irfft(a, n=None, axis=-1, norm=None):
    r"""
    Calculates the inverse of `rfft()`.

    Refer to :func:`mindspore.ops.irfft` for more details.
    The difference is that `a` corresponds to `input` and `axis` corresponds to `dim`.

    Args:
        a (Tensor): The input tensor.
        n (int, optional): Length of the transformed `dim` of the result.
            If given, the input will either be zero-padded or trimmed to this length before computing `rfft`.
            If n is not given, it is taken to be :math:`2*(a.shape[axis]-1)`.
            Default: ``None``.
        axis (int, optional): Axis over which to compute the `irfft`.
            Default: ``-1``, which means the last axis of `a` is used.
        norm (string, optional): Normalization mode. Default: ``None`` that means ``"backward"``.
            Three modes are defined as,

            - ``"backward"`` (no normalization).
            - ``"forward"`` (normalize by :math:`1/n`).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, the result of `irfft()` function.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore import numpy as mnp
        >>> input = Tensor([1, 2, 3, 4])
        >>> y = mnp.fft.irfft(input)
        >>> print(y)
        [ 2.5000000e+00 -6.6666669e-01  1.2590267e-15 -1.6666667e-01
        4.2470195e-16 -6.6666669e-01]
    """
    return ops.irfft(a, n, axis, norm)


def fft2(a, s=None, axes=(-2, -1), norm=None):
    r"""
    Calculates the two dimensional discrete Fourier transform of `a`.

    Refer to :func:`mindspore.ops.fft2` for more details.
    The difference is that `a` corresponds to `input` and `axes` corresponds to `dim`.

    Args:
        a (Tensor): The input tensor.
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64, complex64, complex128.

        s (tuple[int], optional): Length of the transformed `axes` of the result.
            If given, the input will either be zero-padded or trimmed to this length before computing `fft2`.
            Default: ``None`` , which does not need to process `a`.
        axes (tuple[int], optional): The dimension along which to take the one dimensional `fft2`.
            Default: ``(-2, -1)`` , which means transform the last two dimension of `a`.
        norm (string, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as,

            - ``"backward"`` (no normalization).
            - ``"forward"`` (normalize by :math:`1/n`).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, The result of `fft2()` function. The default is the same shape as `a`.
        If `s` is given, the size of the `axes[i]` axis is changed to `s[i]`.
        When the `a` is int16, int32, int64, float16, float32, complex64, the return value type is complex64.
        When the `a` is float64 or complex128, the return value type is complex128.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((4, 4))
        >>> np.fft.fft2(a, s=(4, 4), axes=(0, 1), norm="backward")
        Tensor(shape=[4, 4], dtype=Complex64, value=
        [[16+0j, 0+0j, 0+0j, 0+0j],
         [0+0j, 0+0j, 0+0j, 0+0j],
         [0+0j, 0+0j, 0+0j, 0+0j],
         [0+0j, 0+0j, 0+0j, 0+0j]])
    """
    return ops.fft2(a, s, axes, norm)


def ifft2(a, s=None, axes=(-2, -1), norm=None):
    r"""
    Calculates the inverse of `fft2()`.

    Refer to :func:`mindspore.ops.ifft2` for more details.
    The difference is that `a` corresponds to `input` and `axes` corresponds to `dim`.

    Args:
        a (Tensor): The input tensor.
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64, complex64, complex128.

        s (tuple[int], optional): Length of the transformed `axes` of the result.
            If given, the input will either be zero-padded or trimmed to this length before computing `ifft2`.
            Default: ``None`` , which does not need to process `a`.
        axes (tuple[int], optional): The dimension along which to take the one dimensional `ifft2`.
            Default: ``(-2, -1)`` , which means transform the last two dimension of `a`.
        norm (string, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as,

            - ``"backward"`` (no normalization).
            - ``"forward"`` (normalize by :math:`1*n`).
            - ``"ortho"`` (normalize by :math:`1*\sqrt{n}`).

    Returns:
        Tensor, The result of `ifft2()` function. The default is the same shape as `a`.
        If `s` is given, the size of the `axes[i]` axis is changed to `s[i]`.
        When the `a` is int16, int32, int64, float16, float32, complex64, the return value type is complex64.
        When the `a` is float64 or complex128, the return value type is complex128.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((4, 4))
        >>> np.fft.ifft2(a, s=(4, 4), axes=(0, 1), norm="backward")
        Tensor(shape=[4, 4], dtype=Complex64, value=
        [[1+0j, 0+0j, 0+0j, 0+0j],
         [0+0j, 0+0j, 0+0j, 0+0j],
         [0+0j, 0+0j, 0+0j, 0+0j],
         [0+0j, 0+0j, 0+0j, 0+0j]])
    """
    return ops.ifft2(a, s, axes, norm)


def fftn(a, s=None, axes=None, norm=None):
    r"""
    Calculates the N dimensional discrete Fourier transform of `a`.

    Refer to :func:`mindspore.ops.fftn` for more details.
    The difference is that `a` corresponds to `input` and `axis` corresponds to `dim`.

    Args:
        a (Tensor): The input tensor.
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64, complex64, complex128.

        s (tuple[int], optional): Length of the transformed `axes` of the result.
            If given, the input will either be zero-padded or trimmed to this length before computing `fftn`.
            Default: ``None`` , which does not need to process `a`.
        axes (tuple[int], optional): The dimension along which to take the one dimensional `fftn`.
            Default: ``None`` , which means transform the all dimension of `a`,
            or the last `len(s)` dimensions if s is given.
        norm (string, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as,

            - ``"backward"`` (no normalization).
            - ``"forward"`` (normalize by :math:`1/n`).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, The result of `fft()` function. The default is the same shape as `a`.
        If `s` is given, the size of the `axes[i]` axis is changed to `s[i]`.
        When the `a` is int16, int32, int64, float16, float32, complex64, the return value type is complex64.
        When the `a` is float64 or complex128, the return value type is complex128.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((2, 2, 2))
        >>> np.fft.fftn(a, s=(2, 2, 2), axes=(0, 1, 2), norm="backward")
        Tensor(shape=[2, 2, 2], dtype=Complex64, value=
        [[[8+0j, 0+0j],
         [0+0j, 0+0j]],
         [[0+0j, 0+0j],
         [0+0j, 0+0j]]])
    """
    return ops.fftn(a, s, axes, norm)


def ifftn(a, s=None, axes=None, norm=None):
    r"""
    Calculates the inverse of `fftn()`.

    Refer to :func:`mindspore.ops.ifftn` for more details.
    The difference is that `a` corresponds to `input` and `axes` corresponds to `dim`.

    Args:
        a (Tensor): The input tensor.
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64, complex64, complex128.

        s (tuple[int], optional): Length of the transformed `axes` of the result.
            If given, the input will either be zero-padded or trimmed to this length before computing `ifftn`.
            Default: ``None`` , which does not need to process `a`.
        axes (tuple[int], optional): The dimension along which to take the one dimensional `ifftn`.
            Default: ``None`` , which means transform the all dimension of `a`,
            or the last `len(s)` dimensions if s is given.
        norm (string, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as,

            - ``"backward"`` (no normalization).
            - ``"forward"`` (normalize by :math:`1*n`).
            - ``"ortho"`` (normalize by :math:`1*\sqrt{n}`).

    Returns:
        Tensor, The result of `ifftn()` function. The default is the same shape as `a`.
        If `s` is given, the size of the `axes[i]` axis is changed to `s[i]`.
        When the `a` is int16, int32, int64, float16, float32, complex64, the return value type is complex64.
        When the `a` is float64 or complex128, the return value type is complex128.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((2, 2, 2))
        >>> np.fft.ifftn(a, s=(2, 2, 2), axes=(0, 1, 2), norm="backward")
        Tensor(shape=[2, 2, 2], dtype=Complex64, value=
        [[[1+0j, 0+0j],
         [0+0j, 0+0j]],
         [[0+0j, 0+0j],
         [0+0j, 0+0j]]])
    """
    return ops.ifftn(a, s, axes, norm)
