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

"""Defines spectral operators with functional form."""

from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from .._primitive_cache import _get_cache_prim


def blackman_window(window_length, periodic=True, *, dtype=None):
    r"""
    Blackman window function.

    Usually used to extract a finite signal segment for FFT.

    .. math::

        w[n] = 0.42 - 0.5 \cos(\frac{2\pi n}{N - 1}) + 0.08 \cos(\frac{4\pi n}{N - 1})

    where :math:`N` is the full window size, and n is natural number less than :math:`N` :[0, 1, ..., N-1].

    Args:
        window_length (Tensor): The size of window.
        periodic (bool, optional): If ``True`` , return a periodic window. If ``False``, return a symmetric window.
            Default ``True`` .

    Keyword Args:
        dtype (mindspore.dtype, optional): The data type specified. Default ``None`` .

    Returns:
        A 1-D tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> window_length = mindspore.tensor(10)
        >>> output = mindspore.ops.blackman_window(window_length)
        >>> print(output)
        [-2.9802322e-08  4.0212840e-02  2.0077014e-01  5.0978714e-01
          8.4922993e-01  1.0000000e+00  8.4922981e-01  5.0978690e-01
          2.0077008e-01  4.0212870e-02]
    """
    if dtype is None:
        dtype = mstype.float32

    blackman_window_op = _get_cache_prim(P.BlackmanWindow)(periodic, dtype)
    return blackman_window_op(window_length)


def bartlett_window(window_length, periodic=True, *, dtype=None):
    r"""
    Bartlett window function.

    A triangular-shaped weighting function used for smoothing or frequency analysis of signals in digital signal
    processing.

    .. math::

        w[n] = 1 - \left| \frac{2n}{N-1} - 1 \right| = \begin{cases}
        \frac{2n}{N - 1} & \text{if } 0 \leq n \leq \frac{N - 1}{2} \\
        2 - \frac{2n}{N - 1} & \text{if } \frac{N - 1}{2} < n < N \\
        \end{cases},

    where :math:`N` is the full window size, and n is a natural number less than :math:`N` :[0, 1, ..., N-1].

    Args:
        window_length (Tensor): The size of window.
        periodic (bool, optional): If ``True`` , return a periodic window. If ``False``, return a symmetric window.
            Default ``True`` .

    Keyword Args:
        dtype (mindspore.dtype, optional): The data type specified. Default ``None`` .

    Returns:
        A 1-D tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> window_length = mindspore.tensor(5)
        >>> output = mindspore.ops.bartlett_window(window_length)
        >>> print(output)
        [0.  0.4 0.8 0.8 0.4]
        >>> output = mindspore.ops.bartlett_window(window_length, periodic=False)
        >>> print(output)
        [0.  0.5 1.  0.5 0. ]
    """
    if dtype is None:
        dtype = mstype.float32

    bartlett_window_op = _get_cache_prim(P.BartlettWindow)(periodic, dtype)
    return bartlett_window_op(window_length)


__all__ = [
    'blackman_window',
    'bartlett_window',
]

__all__.sort()
