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
"""NetworkInNetwork."""

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


# NiN block
class NiN(nn.Cell):
    """class NiN"""
    def __init__(self, num_classes=10, num_channel=3):
        super().__init__()
        self.size = ops.Size()
        self.block0 = nn.SequentialCell(
            # block 0
            nn.Conv2d(in_channels=num_channel, out_channels=192, kernel_size=5, stride=1, has_bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=160, kernel_size=1, stride=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=96, kernel_size=1, stride=1, has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same'),
            nn.Dropout(p=0.0)
        )
        self.block1 = nn.SequentialCell(
            # block 1
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5, stride=1, has_bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same'),
            nn.Dropout(p=0.0)
        )
        self.block2 = nn.SequentialCell(
            # block 2
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, has_bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=num_classes, kernel_size=1, stride=1, has_bias=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=8, stride=1, pad_mode='valid')
        )
        # flatten
        self.flatten = nn.Flatten()
        self._initialize_weights()

    def _initialize_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, (nn.Conv2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(Tensor(np.random.normal(0, np.sqrt(2. / n),
                                                          m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        Tensor(np.zeros(m.bias.data.shape, dtype="float32")))

    def construct(self, x):
        out = self.block0(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.flatten(out)
        return out
