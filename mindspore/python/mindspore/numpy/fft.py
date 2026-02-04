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
__all__ = ['fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
           'hfft', 'ihfft', 'hfft2', 'ihfft2', 'hfftn', 'ihfftn',
           'fftshift', 'ifftshift', 'fftfreq', 'rfftfreq']
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
        >>> out = np.fft.fftshift(x)
        >>> print(out)
        [-5 -4 -3 -2 -1  0  1  2  3  4]
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
        >>> out = np.fft.ifftshift(np.fft.fftshift(x))
        >>> print(out)
        [ 0  1  2  3  4 -5 -4 -3 -2 -1]
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

        n (int, optional): Length of the transformed `axis` of the result.
            If given, the size of the `axis` will be zero-padded or truncated to `n` before calculating `fft`.
            Default: ``None`` , which does not need to process `a`.
        axis (int, optional): The dimension along which to take the one dimensional `fft`.
            Default: ``-1`` , which means transform the last dimension of `a`.
        norm (str, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
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
        >>> import mindspore
        >>> import mindspore.numpy as np
        >>> a = np.array([ 1.6243454, -0.6117564, -0.5281718, -1.0729686])
        >>> out = np.fft.fft(a, n=4, axis=-1, norm="backward")
        >>> print(out)
        [-0.5885514+0.j          2.1525173-0.46121222j  2.7808986+0.j
          2.1525173+0.46121222j]
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

        n (int, optional): Length of the transformed `axis` of the result.
            If given, the size of the `axis` will be zero-padded or truncated to `n` before calculating `ifft`.
            Default: ``None`` , which does not need to process `a`.
        axis (int, optional): The dimension along which to take the one dimensional `ifft`.
            Default: ``-1`` , which means transform the last dimension of `a`.
        norm (str, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as,

            - ``"backward"`` (normalize by :math:`1/n`).
            - ``"forward"`` (no normalization).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, The result of `ifft()` function. The default is the same shape as `a`.
        If `n` is given, the size of the `axis` is changed to `n`.
        When the `a` is int16, int32, int64, float16, float32, complex64, the return value type is complex64.
        When the `a` is float64 or complex128, the return value type is complex128.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.numpy as np
        >>> a = np.array([ 1.6243454, -0.6117564, -0.5281718, -1.0729686])
        >>> out = np.fft.ifft(a, n=4, axis=-1, norm="backward")
        >>> print(out)
        [-0.14713785+0.j          0.5381293 +0.11530305j  0.69522464+0.j
          0.5381293 -0.11530305j]
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
        norm (str, optional): Normalization mode. Default: ``None`` that means ``"backward"``.
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
        >>> import mindspore.numpy as np
        >>> a = np.array([1, 2, 3, 4])
        >>> out = np.fft.rfft(a, n=4, axis=-1, norm='backward')
        >>> print(out)
        [10.+0.j -2.+2.j -2.+0.j]
    """
    return ops.rfft(a, n, axis, norm)


def irfft(a, n=None, axis=-1, norm=None):
    r"""
    Calculates the inverse of `rfft()`.

    Refer to :func:`mindspore.ops.irfft` for more details.
    The difference is that `a` corresponds to `a` and `axis` corresponds to `dim`.

    Args:
        a (Tensor): The input tensor.
        n (int, optional): Length of the transformed `axis` of the result.
            If given, the input will either be zero-padded or trimmed to this length before computing `rfft`.
            If n is not given, it is taken to be :math:`2*(a.shape[axis]-1)`.
            Default: ``None``.
        axis (int, optional): Axis over which to compute the `irfft`.
            Default: ``-1``, which means the last axis of `a` is used.
        norm (str, optional): Normalization mode. Default: ``None`` that means ``"backward"``.
            Three modes are defined as,

            - ``"backward"`` (normalize by :math:`1/n`).
            - ``"forward"`` (no normalization).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, the result of `irfft()` function.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.numpy as np
        >>> a = np.array([1, 2, 3, 4])
        >>> y = np.fft.irfft(a, n=6, axis=-1, norm='backward')
        >>> print(y)
        [ 2.5        -0.6666667   0.         -0.16666667  0.         -0.6666667 ]
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
        norm (str, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as, where :math: `n = prod(s)`

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
        >>> out = np.fft.fft2(a, s=(4, 4), axes=(0, 1), norm="backward")
        >>> print(out)
        [[16.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]]
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
            If given, the `a.shape[axes[i]]` will be zero-padded or truncated to `s[i]` before calculating `ifft2`.
            Default: ``None`` , which does not need to process `a`.
        axes (tuple[int], optional): The dimension along which to take the one dimensional `ifft2`.
            Default: ``(-2, -1)`` , which means transform the last two dimension of `a`.
        norm (str, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as, where :math: `n = prod(s)`

            - ``"backward"`` (normalize by :math:`1/n`).
            - ``"forward"`` (no normalization).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

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
        >>> out = np.fft.ifft2(a, s=(4, 4), axes=(0, 1), norm="backward")
        >>> print(out)
        [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
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
        norm (str, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as, where :math: `n = prod(s)`

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
        >>> out = np.fft.fftn(a, s=(2, 2, 2), axes=(0, 1, 2), norm="backward")
        >>> print(out)
        [[[8.+0.j 0.+0.j]
          [0.+0.j 0.+0.j]]
          [[0.+0.j 0.+0.j]
          [0.+0.j 0.+0.j]]]
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
        norm (str, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as, where :math: `n = prod(s)`

            - ``"backward"`` (normalize by :math:`1/n`).
            - ``"forward"`` (no normalization).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

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
        >>> out = np.fft.ifftn(a, s=(2, 2, 2), axes=(0, 1, 2), norm="backward")
        >>> print(out)
        [[[1.+0.j 0.+0.j]
          [0.+0.j 0.+0.j]]
          [[0.+0.j 0.+0.j]
          [0.+0.j 0.+0.j]]]
    """
    return ops.ifftn(a, s, axes, norm)


def rfft2(a, s=None, axes=(-2, -1), norm=None):
    r"""
    Calculates the two dimensional discrete Fourier transform for real input `a`.

    Refer to :func:`mindspore.ops.rfft2` for more details.
    The difference is that `a` corresponds to `input` and `axes` corresponds to `dim`.

    Args:
        a (Tensor): The input tensor.
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64.

        s (tuple[int], optional): Length of the transformed `axes` of the result.
            If given, the size of the `axes[i]` axis will be zero-padded or truncated to `s[i]`
            before calculating `rfft2`.
            Default: ``None`` , which does not need to process `a`.
        axes (tuple[int], optional): The dimension along which to take the one dimensional `rfft2`.
            Default: ``(-2, -1)`` , which means transform the last two dimension of `a`.
        norm (str, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as, where :math: `n = prod(s)`

            - ``"backward"`` (no normalization).
            - ``"forward"`` (normalize by :math:`1/n`).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, the result of `rfft2()` function, dtype of the result is complex64/128.
        If `s` is given, result.shape[axes[i]] is :math:`s[i]`, while result.shape[axes[-1]] is :math:`s[-1] // 2 + 1`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore import numpy as mnp
        >>> a = mnp.ones((2, 2))
        >>> mnp.fft.rfft2(a, s=(2, 2), axes=(0, 1), norm="backward")
        Tensor(shape=[2, 2], dtype=Complex64, value=
        [[4+0j, 0+0j],
         [0+0j, 0+0j]])
    """
    return ops.rfft2(a, s, axes, norm)


def irfft2(a, s=None, axes=(-2, -1), norm=None):
    r"""
    Calculates the inverse of `rfft2()`.

    Refer to :func:`mindspore.ops.irfft2` for more details.
    The difference is that `a` corresponds to `input` and `axes` corresponds to `dim`.

    Args:
        a (Tensor): The input tensor.
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64, complex64, complex128.

        s (tuple[int], optional): Length of the transformed `axes` of the result.
            If given, the input will either be zero-padded or trimmed to this length before computing `irfft2`.
            Default: ``None`` , the axes[-1] of the `a` will be zero-padded to :math:`2*(a.shape[axes[-1]]-1)`.
        axes (tuple[int], optional): The dimension along which to take the one dimensional `irfft2`.
            Default: ``(-2, -1)`` , which means transform the last two dimension of `a`.
        norm (str, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as, where :math: `n = prod(s)`

            - ``"backward"`` (normalize by :math:`1/n`).
            - ``"forward"`` (no normalization).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, The result of `irfft2()` function, result.shape[axes[i]] is s[i].
        When the `a` is int16, int32, int64, float16, float32, complex64, the return value type is float32.
        When the `a` is float64 or complex128, the return value type is float64.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore import numpy as mnp
        >>> a = mnp.ones((4, 4))
        >>> mnp.fft.irfft2(a, s=(4, 4), axes=(0, 1), norm="backward")
        Tensor(shape=[4, 4], dtype=Float32, value=
        [[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]])
    """
    return ops.irfft2(a, s, axes, norm)


def rfftn(a, s=None, axes=None, norm=None):
    r"""
    Calculates the N dimensional discrete Fourier transform for real input `a`.

    Refer to :func:`mindspore.ops.rfftn` for more details.
    The difference is that `a` corresponds to `input` and `axes` corresponds to `dim`.

    Args:
        a (Tensor): The input tensor.
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64.

        s (tuple[int], optional): Length of the transformed `axes` of the result.
            If given, the input will either be zero-padded or trimmed to this length before computing `rfftn`.
            Default: ``None`` , which does not need to process `a`.
        axes (tuple[int], optional): The dimension along which to take the one dimensional `rfftn`.
            Default: ``None`` , which means transform the all dimension of `a`,
            or the last `len(s)` dimensions if s is given.
        norm (str, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as, where :math: `n = prod(s)`

            - ``"backward"`` (no normalization).
            - ``"forward"`` (normalize by :math:`1/n`).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, the result of `rfftn()` function, dtype of the result is complex64/128.
        If `s` is given, result.shape[axes[i]] is :math:`s[i]`, while result.shape[axes[-1]] is :math:`s[-1] // 2 + 1`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore import numpy as mnp
        >>> a = mnp.ones((2, 2, 2))
        >>> mnp.fft.rfftn(a, s=(2, 2, 2), axes=(0, 1, 2), norm="backward")
        Tensor(shape=[2, 2, 2], dtype=Complex64, value=
        [[[8+0j, 0+0j],
          [0+0j, 0+0j]],
         [[0+0j, 0+0j],
          [0+0j, 0+0j]]])
    """
    return ops.rfftn(a, s, axes, norm)


def irfftn(a, s=None, axes=None, norm=None):
    r"""
    Calculates the inverse of `rfftn()`.

    Refer to :func:`mindspore.ops.irfftn` for more details.
    The difference is that `a` corresponds to `input` and `axes` corresponds to `dim`.

    Args:
        a (Tensor): The input tensor.
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64, complex64, complex128.

        s (tuple[int], optional): Length of the transformed `axes` of the result.
            If given, the input will either be zero-padded or trimmed to this length before computing `irfftn`.
            Default: ``None`` , the axes[-1] of the `a` will be zero-padded to :math:`2*(a.shape[axes[-1]]-1)`.
        axes (tuple[int], optional): The dimension along which to take the one dimensional `irfftn`.
            Default: ``None`` , which means transform the all dimension of `a`,
            or the last `len(s)` dimensions if s is given.
        norm (str, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as, where :math: `n = prod(s)`

            - ``"backward"`` (normalize by :math:`1/n`).
            - ``"forward"`` (no normalization).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, The result of `irfft2()` function, result.shape[axes[i]] is s[i].
        When the `a` is int16, int32, int64, float16, float32, complex64, the return value type is float32.
        When the `a` is float64 or complex128, the return value type is float64.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore import numpy as mnp
        >>> a = mnp.ones((2, 2, 2))
        >>> mnp.fft.irfftn(a, s=(2, 2, 2), axes=(0, 1, 2), norm="backward")
        Tensor(shape=[2, 2, 2], dtype=Float32, value=
        [[[ 1.00000000e+00,  0.00000000e+00],
          [ 0.00000000e+00,  0.00000000e+00]],
         [[ 0.00000000e+00,  0.00000000e+00],
          [ 0.00000000e+00,  0.00000000e+00]]])
    """
    return ops.irfftn(a, s, axes, norm)


def hfft(a, n=None, axis=-1, norm=None):
    r"""
    Calculates the one dimensional discrete Fourier transform of of a Hermitian symmetric `a` signal.

    Refer to :func:`mindspore.ops.hfft` for more details.
    The difference is that `a` corresponds to `input` and `axis` corresponds to `dim`.

    Args:
        a (Tensor): The input tensor.
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64, complex64, complex128.

        n (int, optional): Length of the transformed `axis` of the result.
            If given, the size of the `axis` axis will be zero-padded or truncated to `n` before calculating `hfft`.
            Default: ``None`` , which does not need to process `a`.
        axis (int, optional): The dimension along which to take the one dimensional `hfft`.
            Default: ``-1`` , which means transform the last dimension of `a`.
        norm (str, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as,

            - ``"backward"`` (no normalization).
            - ``"forward"`` (normalize by :math:`1/n`).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, The result of `hfft()` function.
        If `n` is given, result.shape[axis] is :math:`(n - 1) * 2`, otherwise math:`(a.shape[axis] - 1) * 2`.
        When the `a` is int16, int32, int64, float16, float32, complex64, the return value type is float32.
        When the `a` is float64 or complex128, the return value type is float64.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore import numpy as np
        >>> a = np.array([ 1.6243454, -0.6117564, -0.5281718, -1.0729686])
        >>> out = np.fft.hfft(a, n=4, axis=-1, norm="backward")
        >>> print(out)
        [-0.12733912  2.1525173   2.3196864   2.1525173 ]
    """
    return ops.hfft(a, n, axis, norm)


def ihfft(a, n=None, axis=-1, norm=None):
    r"""
    Calculates the inverse of `hfft()`.

    Refer to :func:`mindspore.ops.ihfft` for more details.
    The difference is that `a` corresponds to `input` and `axis` corresponds to `dim`.

    Note:
        - `ihfft` is currently only used in `mindscience` scientific computing scenarios and
        does not support other usage scenarios.
        - `ihfft` is not supported on Windows platform yet.

    Args:
        a (Tensor): The input tensor.
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64.

        n (int, optional): Length of the transformed `axis` of the result.
            If given, the size of the `axis` will be zero-padded or truncated to `n` before calculating `ihfft`.
            Default: ``None`` , which does not need to process `a`.
        axis (int, optional): The dimension along which to take the one dimensional `ihfft`.
            Default: ``-1`` , which means transform the last dimension of `a`.
        norm (str, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as,

            - ``"backward"`` (normalize by :math:`1/n`).
            - ``"forward"`` (no normalization).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, The result of `ihfft()` function.
        If `n` is given, result.shape[axis] is :math:`n // 2 + 1`, otherwise math:`a.shape[axis] // 2 + 1`.
        When the `a` is int16, int32, int64, float16, float32, the return value type is complex64.
        When the `a` is float64, the return value type is complex128.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore import numpy as np
        >>> a = np.array([ 1.6243454, -0.6117564, -0.5281718, -1.0729686])
        >>> out = np.fft.ihfft(a, n=4, axis=-1, norm="backward")
        >>> print(out)
        [-0.14713785-0.j          0.5381293 +0.11530305j  0.69522464-0.j        ]
    """
    return ops.ihfft(a, n, axis, norm)


def hfft2(a, s=None, axes=(-2, -1), norm=None):
    r"""
    Calculates the two dimensional discrete Fourier transform of of a Hermitian symmetric `a` signal.

    Refer to :func:`mindspore.ops.hfft2` for more details.
    The difference is that `a` corresponds to `input` and `axes` corresponds to `dim`.

    Args:
        a (Tensor): The input tensor.
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64, complex64, complex128.

        s (tuple[int], optional): Length of the transformed `axes` of the result.
            If given, the `a.shape[axes[i]]` will be zero-padded or truncated to `s[i]` before calculating `hfft2`.
            Default: ``None`` , which does not need to process `a`.
        axes (tuple[int], optional): The dimension along which to take the one dimensional `hfft2`.
            Default: ``(-2, -1)`` , which means transform the last two dimension of `a`.
        norm (str, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as, where :math: `n = prod(s)`

            - ``"backward"`` (no normalization).
            - ``"forward"`` (normalize by :math:`1/n`).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, The result of `hfft2()` function.
        If `s` is given, result.shape[axes[i]] is s[i], and for the last transformed axes,
        result.shape[axes[-1]] is :math:`(s[-1] - 1) * 2`, otherwise :math:`(a.shape[axes[-1]] - 1) * 2`.
        When the `a` is int16, int32, int64, float16, float32, complex64, the return value type is complex64.
        When the `a` is float64 or complex128, the return value type is complex128.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((4, 4))
        >>> out = np.fft.hfft2(a, s=(4, 4), axes=(0, 1), norm="backward")
        >>> print(out)
        [[16.  0.  0.  0.]
         [ 0.  0.  0.  0.]
         [ 0.  0.  0.  0.]
         [ 0.  0.  0.  0.]]
    """
    return ops.hfft2(a, s, axes, norm)


def ihfft2(a, s=None, axes=(-2, -1), norm=None):
    r"""
    Computes the two dimensional inverse discrete Fourier transform of real `a`.

    Refer to :func:`mindspore.ops.ihfft2` for more details.
    The difference is that `a` corresponds to `input` and `axes` corresponds to `dim`.

    Args:
        a (Tensor): The input tensor.
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64.

        s (tuple[int], optional): Length of the transformed `axes` of the result.
            If given, the `a.shape[axes[i]]` will be zero-padded or truncated to `s[i]` before calculating `ihfft2`.
            Default: ``None`` , which does not need to process `a`.
        axes (tuple[int], optional): The dimension along which to take the one dimensional `ihfft2`.
            Default: ``(-2, -1)`` , which means transform the last two dimension of `a`.
        norm (str, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as, where :math: `n = prod(s)`

            - ``"backward"`` (normalize by :math:`1/n`).
            - ``"forward"`` (no normalization).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, The result of `ihfft2()` function.
        If `s` is given, result.shape[axes[i]] is s[i], and for the last transformed `axes`,
        result.shape[axes[-1]] is :math:`s[-1] // 2 + 1`, otherwise :math:`a.shape[axes[-1]] // 2 + 1`.
        When the `a` is int16, int32, int64, float16, float32, the return value type is complex64.
        When the `a` is float64, the return value type is complex128.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((4, 4))
        >>> out = np.fft.ihfft2(a, s=(4, 4), axes=(0, 1), norm="backward")
        >>> print(out)
        [[1.-0.j 0.-0.j 0.-0.j]
         [0.-0.j 0.-0.j 0.-0.j]
         [0.-0.j 0.-0.j 0.-0.j]
         [0.-0.j 0.-0.j 0.-0.j]]
    """
    return ops.ihfft2(a, s, axes, norm)


def hfftn(a, s=None, axes=None, norm=None):
    r"""
    Calculates the N dimensional discrete Fourier transform of of a Hermitian symmetric `a`.

    Refer to :func:`mindspore.ops.hfftn` for more details.
    The difference is that `a` corresponds to `input` and `axes` corresponds to `dim`.

    Args:
        a (Tensor): The input tensor.
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64, complex64, complex128.

        s (tuple[int], optional): Length of the transformed `axes` of the result.
            If given, the `a.shape[axes[i]]` will be zero-padded or truncated to `s[i]` before calculating `hfftn`.
            Default: ``None`` , which does not need to process `a`.
        axes (tuple[int], optional): The dimension along which to take the one dimensional `hfftn`.
            Default: ``(-2, -1)`` , which means transform the last two dimension of `a`.
        norm (str, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as, where :math: `n = prod(s)`

            - ``"backward"`` (no normalization).
            - ``"forward"`` (normalize by :math:`1/n`).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, The result of `hfftn()` function.
        If `s` is given, result.shape[axes[i]] is s[i], and for the last transformed `axes`,
        result.shape[axes[-1]] is :math:`(s[-1] - 1) * 2`, otherwise :math:`(a.shape[axes[-1]] - 1) * 2`.
        When the `a` is int16, int32, int64, float16, float32, complex64, the return value type is complex64.
        When the `a` is float64 or complex128, the return value type is complex128.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((4, 4))
        >>> out = np.fft.hfftn(a, s=(4, 4), axes=(0, 1), norm="backward")
        >>> print(out)
        [[16.  0.  0.  0.]
         [ 0.  0.  0.  0.]
         [ 0.  0.  0.  0.]
         [ 0.  0.  0.  0.]]
    """
    return ops.hfftn(a, s, axes, norm)


def ihfftn(a, s=None, axes=None, norm=None):
    r"""
    Computes the N dimensional inverse discrete Fourier transform of real `a`.

    Refer to :func:`mindspore.ops.ihfftn` for more details.
    The difference is that `a` corresponds to `input` and `axes` corresponds to `dim`.

    Args:
        a (Tensor): The input tensor.
            Supported dtypes:

            - Ascend/CPU: int16, int32, int64, float16, float32, float64.

        s (tuple[int], optional): Length of the transformed `axes` of the result.
            If given, the `a.shape[axes[i]]` will be zero-padded or truncated to `s[i]` before calculating `ihfftn`.
            Default: ``None`` , which does not need to process `a`.
        axes (tuple[int], optional): The dimension along which to take the one dimensional `ihfftn`.
            Default: ``(-2, -1)`` , which means transform the last two dimension of `a`.
        norm (str, optional): Normalization mode. Default: ``None`` that means ``"backward"`` .
            Three modes are defined as, where :math: `n = prod(s)`

            - ``"backward"`` (normalize by :math:`1/n`).
            - ``"forward"`` (no normalization).
            - ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).

    Returns:
        Tensor, The result of `ihfftn()` function.
        If `s` is given, result.shape[axes[i]] is s[i], and for the last transformed `axes`,
        result.shape[axes[-1]] is :math:`s[-1] // 2 + 1`, otherwise :math:`a.shape[axes[-1]] // 2 + 1`.
        When the `a` is int16, int32, int64, float16, float32, the return value type is complex64.
        When the `a` is float64, the return value type is complex128.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((4, 4))
        >>> out = np.fft.ihfftn(a, s=(4, 4), axes=(0, 1), norm="backward")
        >>> print(out)
        [[16.  0.  0.  0.]
         [ 0.  0.  0.  0.]
         [ 0.  0.  0.  0.]
         [ 0.  0.  0.  0.]]
    """
    return ops.hfftn(a, s, axes, norm)


def fftfreq(n, d=1.0, dtype=None):
    r"""
    Return the Discrete Fourier Transform sample frequencies.

    Args:
        n (int): Window length.
        d (float, optional): Sample spacing (inverse of the sampling rate). Default: ``1.0`` .
        dtype (mindspore.dtype, optional): The dtype of the returned frequencies. Default: ``float32`` .

    Returns:
        Tensor, Array of length ``n`` containing the sample frequencies.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> out = np.fft.fftfreq(n=4, d=1.0)
        >>> print(out)
        [ 0.    0.25 -0.5  -0.25]
    """
    return ops.fftfreq(n, d, dtype)


def rfftfreq(n, d=1.0, dtype=None):
    r"""
    Return the Discrete Fourier Transform sample frequencies (for usage with rfft, irfft).

    Args:
        n (int): Window length.
        d (float, optional): Sample spacing (inverse of the sampling rate). Default: ``1.0`` .
        dtype (mindspore.dtype, optional): The dtype of the returned frequencies. Default: ``float32`` .

    Returns:
        Tensor, Array of length math:`n // 2 + 1` containing the sample frequencies.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> out = np.fft.rfftfreq(n=4, d=1.0)
        >>> print(out)
        [0.   0.25 0.5 ]
    """
    return ops.rfftfreq(n, d, dtype)
