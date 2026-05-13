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
"""activation layer for mint"""
from __future__ import absolute_import
from __future__ import division

from mindspore import mint
from mindspore.nn.cell import Cell


class Flatten(Cell):
    r"""
    Flatten the input Tensor along dimensions from `start_dim` to `end_dim`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        start_dim (int, optional): The first dimension to flatten. Default: ``1`` .
        end_dim (int, optional): The last dimension to flatten. Default: ``-1`` .

    Inputs:
        - **input** (Tensor) - The input Tensor to be flattened.

    Outputs:
        Tensor. If no dimensions are flattened, returns the original `input`, otherwise return the flattened Tensor.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `start_dim` or `end_dim` is not int.
        ValueError: If `start_dim` is greater than `end_dim` after canonicalized.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> import numpy as np
        >>> input = Tensor(np.array([[[1.2, 1.2], [2.1, 2.1]], [[2.2, 2.2], [3.2, 3.2]]]), mindspore.float32)
        >>> net = mint.nn.Flatten()
        >>> output = net(input)
        >>> print(output)
        [[1.2 1.2 2.1 2.1]
         [2.2 2.2 3.2 3.2]]
        >>> print(f"before flatten the x shape is {input.shape}")
        before flatten the input shape is  (2, 2, 2)
        >>> print(f"after flatten the output shape is {output.shape}")
        after flatten the output shape is (2, 4)
    """

    def __init__(self, start_dim=1, end_dim=-1):
        """Initialize Flatten."""
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def construct(self, input):
        return mint.nn.functional.flatten(input, self.start_dim, self.end_dim)


__all__ = [
    'Flatten',
]
