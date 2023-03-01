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
"""channel shuffle"""
from mindspore.ops import operations as P
from mindspore.nn.cell import Cell

__all__ = ['ChannelShuffle']


class ChannelShuffle(Cell):
    r"""
    Divide the channels of Tensor whose shape is :math:`(*, C , H, W)` into g groups to obtain a Tensor with
    shape :math:`(*, C frac g, g, H, W)`, and transpose along the corresponding axis of :math:`C`, :math:`frac g` and
    :math:`g` to restore Tensor to the original shape.

    Args:
        groups (int): Number of groups to divide channels in. Refer to :math:`g`.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If `groups` is not an int.
        ValueError: If `groups` is less than 1.
        ValueError: If dims of `x` is less than 3.
        ValueError: If number of channels can not be divisible by groups.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> channel_shuffle = nn.ChannelShuffle(2)
        >>> x = Tensor(np.arange(16).astype(np.int32).reshape(1, 4, 2, 2))
        >>> print(x)
        [[[[0 1],
           [2 3]],
          [[4 5],
           [6 7]],
          [[8 9],
           [10 11]],
          [[12 13],
           [14 15]],
         ]]
        >>> output = channel_shuffle(x)
        >>> print(output)
        [[[[0 1],
           [2 3]],
          [[8 9],
           [10 11]],
          [[4 5],
           [6 7]],
          [[12 13],
           [14 15]],
         ]]
    """
    def __init__(self, groups):
        """Initialize ChannelShuffle."""
        super(ChannelShuffle, self).__init__()
        if not isinstance(groups, int):
            raise TypeError("For ChannelShuffle, the param `groups` must be int, but got {}.".format(type(groups)))
        if groups < 1:
            raise ValueError(f"For ChannelShuffle, the param `groups` must be larger than 0, but got {groups}.")

        self.groups = groups
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, x):
        x_shape = self.shape(x)
        n, c = x_shape[0], x_shape[1]
        out = self.reshape(x, (n, self.groups, c // self.groups, -1))
        out = self.transpose(out, (0, 2, 1, 3))
        return self.reshape(out, x_shape)
