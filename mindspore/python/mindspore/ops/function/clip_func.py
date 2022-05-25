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

"""Defines clip operators with functional form."""

from ..operations import clip_ops


def clip_by_norm(x, clip_norm, axis=None):
    r"""
    This function is used to clip tensor to a maximum :math:`L_2`-norm. If the :math:`L_2`-norm of the input 'x' is not
    greater than the input `clip_norm`, the output tensor remains unchanged. Otherwise the output tensor will be
    normalized as:

        .. math::
            \text{output}(x) = \frac{\text{clip_norm} * x}{L_2(x)},

        where :math:`L_2(x)` is the :math:`L_2`-norm of :math:`x`.

    Inputs:
        - **x** (Tensor) - Tensor of shape N-D. The type must be float16 or float32.
        - **clip_norm** (Tensor) - A scalar Tensor of shape :math:`()` or :math:`(1)`.
          Or a shape can be broadcast to the shape of `x`. The type must be float16 or float32.
        - **axis** (Union[None, int, tuple(int), list(int)]) - Compute the `L_2`-norm along the specific dimension.
          Default: None, all dimensions to calculate.

    Outputs:
        Tensor, clipped Tensor, whose shape is the same as `x` and type is float32.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.
        TypeError: If dtype of `clip_norm` is neither float16 nor float32.
        TypeError: If `axis` is not one of None, int, tuple(int) and list(int).

    Supported Platforms:
        ``Ascend`` ``GPU`` CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore.ops import functional as F
        >>> x = Tensor(np.random.randint(0, 10, [6, 16]), mindspore.float32)
        >>> clip_norm = Tensor(np.array([10]).astype(np.float32))
        >>> output = F.clip_by_norm(x, clip_norm)
        >>> print(output.shape)
        (6, 16)
    """
    return clip_ops.ClipByNorm(axis=axis)(x, clip_norm)


__all__ = [
    'clip_by_norm'
]
__all__.sort()
