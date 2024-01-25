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
"""Fast Fourier Transform operations, the function docs are adapted from Numpy API."""
from __future__ import absolute_import
from mindspore.ops import function as F


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
        >>> from mindspore.numpy.fft import fftshift
        >>> from mindspore import Tensor
        >>> from mindspore import dtype as mstype
        >>> x = Tensor([0, 1, 2, 3, 4, -5, -4, -3, -2, -1], dtype=mstype.int32)
        >>> fftshift(x)
        Tensor(shape=[10], dtype=Int32, value= [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
    """
    return F.fftshift(x, axes)


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
        >>> from mindspore.numpy.fft import fftshift, ifftshift
        >>> from mindspore import Tensor
        >>> from mindspore import dtype as mstype
        >>> x = Tensor([0, 1, 2, 3, 4, -5, -4, -3, -2, -1], dtype=mstype.int32)
        >>> ifftshift(fftshift(x))
        Tensor(shape=[10], dtype=Int32, value= [ 0, 1, 2, 3, 4, -5, -4, -3, -2, -1])
    """
    return F.ifftshift(x, axes)
