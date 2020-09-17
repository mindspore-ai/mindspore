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


import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Tensor
import mindspore.common.dtype as mstype

from .base import _conv, _bn
from .resnet50 import ResNet, ResidualBlock
from .fpn import FPN


class ETSNet(nn.Cell):
    def __init__(self, config):
        super(ETSNet, self).__init__()
        self.kernel_num = config.KERNEL_NUM
        self.inference = config.INFERENCE
        if config.INFERENCE:
            self.long_size = config.INFER_LONG_SIZE
        else:
            self.long_size = config.TRAIN_LONG_SIZE

        # backbone
        self.feature_extractor = ResNet(ResidualBlock,
                                        config.BACKBONE_LAYER_NUMS,
                                        config.BACKBONE_IN_CHANNELS,
                                        config.BACKBONE_OUT_CHANNELS)

        # neck
        self.feature_fusion = FPN(config.BACKBONE_OUT_CHANNELS,
                                  config.NECK_OUT_CHANNEL,
                                  self.long_size)

        # head
        self.conv1 = _conv(4 * config.NECK_OUT_CHANNEL,
                           config.NECK_OUT_CHANNEL,
                           kernel_size=3,
                           stride=1,
                           has_bias=True)
        self.bn1 = _bn(config.NECK_OUT_CHANNEL)
        self.relu1 = nn.ReLU()
        self.conv2 = _conv(config.NECK_OUT_CHANNEL,
                           config.KERNEL_NUM,
                           kernel_size=1,
                           has_bias=True)
        self._upsample = P.ResizeBilinear((self.long_size, self.long_size), align_corners=True)

        if self.inference:
            self.one_float32 = Tensor(1.0, mstype.float32)
            self.sigmoid = P.Sigmoid()
            self.greater = P.Greater()
            self.logic_and = P.LogicalAnd()

        print('ETSNet initialized!')

    def construct(self, x):
        c2, c3, c4, c5 = self.feature_extractor(x)

        feature = self.feature_fusion(c2, c3, c4, c5)

        output = self.conv1(feature)
        output = self.relu1(self.bn1(output))
        output = self.conv2(output)
        output = self._upsample(output)

        if self.inference:
            text = output[::, 0:1:1, ::, ::]
            kernels = output[::, 0:7:1, ::, ::]
            score = self.sigmoid(text)
            kernels = self.logic_and(self.greater(kernels, self.one_float32),
                                     self.greater(text, self.one_float32))
            return score, kernels
        return output
