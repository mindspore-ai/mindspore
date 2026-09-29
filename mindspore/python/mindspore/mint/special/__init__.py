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
"""mint module."""
from __future__ import absolute_import

from mindspore.ops.auto_generate import erfc, expm1, exp2, sinc, log1p, round_op
from mindspore.ops.function.nn_func import log_softmax_ext as log_softmax


def round(input):
    r"""
    Round input to the nearest integer.

    .. math::
        out_i \approx input_i

    .. note::
        The input data types supported by the Ascend platform include
        bfloat16, float16, float32, float64, int32, and int64.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor, has the same shape and type as the `input`.

    Raises:
        TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([0.8, 1.5, 2.3, 2.5, -4.5]), mindspore.float32)
        >>> output = mint.special.round(input)
        >>> print(output)
        [ 1.  2.  2.  2. -4.]
    """
    return round_op(input, 0)

__all__ = [
    'erfc',
    'expm1',
    'exp2',
    'round',
    'sinc',
    'log1p',
    'log_softmax',
]
