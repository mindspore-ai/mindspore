# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Create attention paddings from input paddings."""
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor


class CreateAttentionPaddingsFromInputPaddings(nn.Cell):
    """
    Create attention mask according to input mask.

    Args:
        config: Config class.

    Returns:
        Tensor, shape of (N, T, T).
    """

    def __init__(self,
                 config,
                 is_training=True):
        super(CreateAttentionPaddingsFromInputPaddings, self).__init__()

        self.is_training = is_training
        self.input_mask = None
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.batch_matmul = P.BatchMatMul()
        self.multiply = P.Mul()
        self.shape = P.Shape()
        # mask future positions
        ones = np.ones(shape=(config.batch_size, config.seq_length, config.seq_length))
        self.lower_triangle_mask = Tensor(np.tril(ones), dtype=mstype.float32)

    def construct(self, input_mask, mask_future=False):
        """
        Construct network.

        Args:
            input_mask (Tensor): Tensor mask vectors with shape (N, T).
            mask_future (bool): Whether mask future (for decoder training).

        Returns:
            Tensor, shape of (N, T, T).
        """
        input_shape = self.shape(input_mask)
        # Add this for infer as the seq_length will increase.
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)
        if self.is_training:
            input_mask = self.cast(input_mask, mstype.float16)
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)

        attention_mask = self.batch_matmul(mask_left, mask_right)
        if self.is_training:
            attention_mask = self.cast(attention_mask, mstype.float32)

        if mask_future:
            attention_mask = self.multiply(attention_mask, self.lower_triangle_mask)

        return attention_mask
