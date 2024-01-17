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
from mindspore.ops.auto_generate import FFTShift
from mindspore.ops._primitive_cache import _get_cache_prim


def fftshift(input, dim=None):
    """
    Shift the zero-frequency component to the center of the spectrum.

    Note:
        - `fftshift` is currently only used in `mindscience` scientific computing scenarios and
          dose not support other usage scenarios.
        - `fftshift` is not supported on Windows platform yet.

    Args:
        input (Tensor): Input tensor.
        dim (Union[int, list(int), tuple(int)], optional): The dimensions which to shift.
            Default is ``None``, which shifts all dimensions.

    Returns:
        output (Tensor), the shifted tensor with the same shape and dtype as `input`.

    Raises:
        TypeError: If `input` is not a tensor.
        TypeError: If the type/dtype of `dim` is not int.
        ValueError: If `dim` is out of the range of `[-input.ndim, input.ndim)`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore.ops import fftshift
        >>> from mindspore import Tensor
        >>> from mindspore import dtype as mstype
        >>> input = Tensor([0, 1, 2, 3, 4, -5, -4, -3, -2, -1], dtype=mstype.int32)
        >>> fftshift(input)
        Tensor(shape=[10], dtype=Int32, value= [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
    """
    if dim == ():
        return input
    if dim is None:
        dim = ()

    fftshift_op = _get_cache_prim(FFTShift)(forward=True)
    return fftshift_op(input, dim)


def ifftshift(input, dim=None):
    """
    The inverse of `fftshift`. Refer to :func:`mindspore.ops.fftshift` for more details.

    Note:
        - `ifftshift` is currently only used in `mindscience` scientific computing scenarios and
          dose not support other usage scenarios.
        - `ifftshift` is not supported on Windows platform yet.

    Args:
        input (Tensor): Input tensor.
        dim (Union[int, list(int), tuple(int)], optional): The dimensions which to shift.
            Default is ``None``, which shifts all dimensions.

    Returns:
        output (Tensor), the shifted tensor with the same shape and dtype as `input`.

    Raises:
        TypeError: If `input` is not a tensor.
        TypeError: If the type/dtype of `dim` is not int.
        ValueError: If `dim` is out of the range of `[-input.ndim, input.ndim)`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore.ops import fftshift, ifftshift
        >>> from mindspore import Tensor
        >>> from mindspore import dtype as mstype
        >>> input = Tensor([0, 1, 2, 3, 4, -5, -4, -3, -2, -1], dtype=mstype.int32)
        >>> ifftshift(fftshift(input))
        Tensor(shape=[10], dtype=Int32, value= [ 0, 1, 2, 3, 4, -5, -4, -3, -2, -1])
    """
    if dim == ():
        return input
    if dim is None:
        dim = ()

    fftshift_op = _get_cache_prim(FFTShift)(forward=False)
    return fftshift_op(input, dim)


__all__ = [
    'fftshift',
    'ifftshift'
]

__all__.sort()
