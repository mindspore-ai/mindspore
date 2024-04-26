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

"""

Math Operators with better performance

"""

from mindspore.ops import auto_generate as P
from mindspore.ops.auto_generate.gen_ops_def import add_ext as add, sub_ext as sub, bmm_ext as bmm


# define Primitive global variables


def baddbmm(input, batch1, batch2, beta=1, alpha=1):
    r"""
    The result is the sum of the input and a batch matrix-matrix product of matrices in batch1 and batch2.
    The formula is defined as follows:

    .. math::
        \text{out}_{i} = \beta \text{input}_{i} + \alpha (\text{batch1}_{i} \mathbin{@} \text{batch2}_{i})

    Args:
        input (Tensor): The input Tensor. When batch1 is a :math:`(C, W, T)` Tensor and batch2 is a
            :math:`(C, T, H)` Tensor, input must be broadcastable with :math:`(C, W, H)` Tensor.
        batch1 (Tensor): :math:`batch1` in the above formula. Must be 3-D Tensor, dtype is same as input.
        batch2 (Tensor): :math:`batch2` in the above formula. Must be 3-D Tensor, dtype is same as input.
        beta (Union[float, int], optional): multiplier for input. Default: ``1`` .
        alpha (Union[float, int], optional): multiplier for :math:`batch1 @ batch2`. Default: ``1`` .
            Arguments beta and alpha must be integers when inputs of type not FloatTensor, otherwise they should
            be a real number.

    Returns:
        Tensor, has the same dtype as input, shape will be :math:`(C, W, H)`.

    Raises:
        TypeError: The type of `input`, `batch1`, `batch2` is not Tensor.
        TypeError: The types of `input`, `batch1`, `batch2` are different.
        TypeError: For inputs of type FloatTensor or DoubleTensor, \
                    arguments beta and alpha not be real numbers, otherwise not be integers.
        TypeError: For Baddbmm, attributes alpha and beta are not real numbers
        ValueError: If `batch1` and `batch2` are not 3-D tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.ones([1, 3, 3]).astype(np.float32))
        >>> batch1 = Tensor(np.ones([1, 3, 4]).astype(np.float32))
        >>> batch2 = Tensor(np.ones([1, 4, 3]).astype(np.float32))
        >>> output = ops.baddbmm(input, batch1, batch2)
        >>> print(output)
        [[[5. 5. 5.]
          [5. 5. 5.]
          [5. 5. 5.]]]
    """
    return P.baddbmm(input, batch1, batch2, beta, alpha)


__all__ = ['baddbmm', 'add', 'sub', 'bmm']
