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
"""Components of model."""
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P


class SaturateCast(nn.Cell):
    """Cast wrapper."""

    def __init__(self, dst_type=mstype.float32):
        super(SaturateCast, self).__init__()
        self.cast = P.Cast()
        self.dst_type = dst_type

    def construct(self, x):
        return self.cast(x, self.dst_type)


class LayerNorm(nn.Cell):
    """
    Do layer norm.

    Args:
        in_channels (int): In channels number of layer norm.
        return_2d (bool): Whether return 2d tensor.

    Returns:
        Tensor, output.
    """

    def __init__(self, in_channels=None, return_2d=False):
        super(LayerNorm, self).__init__()
        self.return_2d = return_2d
        self.layer_norm = nn.LayerNorm((in_channels,))
        self.cast = P.Cast()
        self.get_dtype = P.DType()
        self.reshape = P.Reshape()
        self.get_shape = P.Shape()

    def construct(self, input_tensor):
        """Do layer norm."""
        shape = self.get_shape(input_tensor)
        batch_size = shape[0]
        max_len = shape[1]
        embed_dim = shape[2]

        output = self.reshape(input_tensor, (-1, embed_dim))
        output = self.cast(output, mstype.float32)
        output = self.layer_norm(output)
        output = self.cast(output, self.get_dtype(input_tensor))
        if not self.return_2d:
            output = self.reshape(output, (batch_size, max_len, embed_dim))
        return output
