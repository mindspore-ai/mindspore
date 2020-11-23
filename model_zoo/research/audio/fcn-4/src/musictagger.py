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
'''model'''

from mindspore import nn
from mindspore.ops import operations as P


class MusicTaggerCNN(nn.Cell):
    """
    Music Tagger CNN
    """
    def __init__(self, in_classes, kernel_size, padding, maxpool, has_bias):
        super(MusicTaggerCNN, self).__init__()
        self.in_classes = in_classes
        self.kernel_size = kernel_size
        self.maxpool = maxpool
        self.padding = padding
        self.has_bias = has_bias
        # build model
        self.conv1 = nn.Conv2d(self.in_classes[0], self.in_classes[1],
                               self.kernel_size[0])
        self.conv2 = nn.Conv2d(self.in_classes[1], self.in_classes[2],
                               self.kernel_size[1])
        self.conv3 = nn.Conv2d(self.in_classes[2], self.in_classes[3],
                               self.kernel_size[2])
        self.conv4 = nn.Conv2d(self.in_classes[3], self.in_classes[4],
                               self.kernel_size[3])

        self.bn1 = nn.BatchNorm2d(self.in_classes[1])
        self.bn2 = nn.BatchNorm2d(self.in_classes[2])
        self.bn3 = nn.BatchNorm2d(self.in_classes[3])
        self.bn4 = nn.BatchNorm2d(self.in_classes[4])

        self.pool1 = nn.MaxPool2d(maxpool[0], maxpool[0])
        self.pool2 = nn.MaxPool2d(maxpool[1], maxpool[1])
        self.pool3 = nn.MaxPool2d(maxpool[2], maxpool[2])
        self.pool4 = nn.MaxPool2d(maxpool[3], maxpool[3])
        self.poolreduce = P.ReduceMax(keep_dims=False)
        self.Act = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dense = nn.Dense(2048, 50, activation='sigmoid')
        self.sigmoid = nn.Sigmoid()

    def construct(self, input_data):
        """
        Music Tagger CNN
        """
        x = self.conv1(input_data)
        x = self.bn1(x)
        x = self.Act(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.Act(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.Act(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.Act(x)
        x = self.poolreduce(x, (2, 3))
        x = self.flatten(x)
        x = self.dense(x)

        return x
