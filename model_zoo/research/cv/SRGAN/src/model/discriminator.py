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

"""Structure of Discriminator"""

import mindspore.nn as nn
from src.util.util import init_weights

class Discriminator(nn.Cell):
    """Structure of Discriminator"""
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        feature_map_size = int(image_size // 16)
        self.features = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, pad_mode='pad'),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, pad_mode='pad'),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(128, eps=1e-05),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(128, eps=1e-05),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(256, eps=1e-05),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, pad_mode='pad'),      # state size. (256) x 12 x 12
            nn.BatchNorm2d(256, eps=1e-05),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(512, eps=1e-05),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, pad_mode='pad'),      # state size. (512) x 6 x 6
            nn.BatchNorm2d(512, eps=1e-05),
            nn.LeakyReLU(0.2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.SequentialCell(
            nn.Dense(512*feature_map_size*feature_map_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dense(1024, 1),
            nn.Sigmoid()
        )
    def construct(self, x):
        out = self.features(x)
        out = self.flatten(out)
        out = self.classifier(out)
        return out

def get_discriminator(image_size, init_gain):
    """Return discriminator by args."""
    net = Discriminator(image_size)
    init_weights(net, 'normal', init_gain)
    return net
