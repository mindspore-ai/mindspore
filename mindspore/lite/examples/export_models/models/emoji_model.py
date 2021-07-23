# Copyright 2021 Huawei Technologies Co., Ltd
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
"""emoji_model."""
import mindspore as MS


class GlobalAvgPooling(MS.nn.Cell):
    """
    Global avg pooling definition.
    Args:
    Returns:
        Tensor, output tensor.
    Examples:
        >>> GlobalAvgPooling()
    """

    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.mean = MS.ops.ReduceMean(keep_dims=False)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x


class EmojiModel(MS.nn.Cell):
    """emoji model"""
    def __init__(self, wayc, use_bb, use_head):
        super(EmojiModel, self).__init__()
        self.use_head = use_head
        self.use_bb = use_bb
        if use_bb:
            self.relu = MS.nn.ReLU()
            self.maxpool = MS.nn.MaxPool2d(kernel_size=2, stride=2)
            self.c1 = MS.nn.Conv2d(1, 32, (3, 3), (1, 1), pad_mode='pad', padding=1, dilation=(1, 1), group=1,
                                   has_bias=True)
            self.bn1 = MS.nn.BatchNorm2d(32)
            self.c2 = MS.nn.Conv2d(32, 64, (3, 3), (1, 1), pad_mode='pad', padding=1, dilation=(1, 1), group=1,
                                   has_bias=True)
            self.bn2 = MS.nn.BatchNorm2d(64)
            self.c3 = MS.nn.Conv2d(64, 128, (3, 3), (1, 1), pad_mode='pad', padding=1, dilation=(1, 1), group=1,
                                   has_bias=True)
            self.bn3 = MS.nn.BatchNorm2d(128)
            self.c4 = MS.nn.Conv2d(128, 256, (3, 3), (1, 1), pad_mode='pad', padding=1, dilation=(1, 1), group=1,
                                   has_bias=True)
            self.bn4 = MS.nn.BatchNorm2d(256)
        if use_head:
            self.c5 = MS.nn.Conv2d(256, wayc, (3, 3), (1, 1), pad_mode='pad', padding=1, dilation=(1, 1), group=1,
                                   has_bias=True)
            self.gap = GlobalAvgPooling()

    def construct(self, x):
        """construct"""
        if self.use_bb:
            x = self.c1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.c2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.c3(x)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.c4(x)
            x = self.bn4(x)
            x = self.relu(x)
        if self.use_head:
            x = self.c5(x)
            x = self.gap(x)
        return x
