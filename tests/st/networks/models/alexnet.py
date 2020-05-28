# Copyright 2019 Huawei Technologies Co., Ltd
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
from mindspore.ops import operations as P


class AlexNet(nn.Cell):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.batch_size = 32
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4, pad_mode="valid")
        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, pad_mode="same")
        self.conv3 = nn.Conv2d(256, 384, 3, stride=1, pad_mode="same")
        self.conv4 = nn.Conv2d(384, 384, 3, stride=1, pad_mode="same")
        self.conv5 = nn.Conv2d(384, 256, 3, stride=1, pad_mode="same")
        self.relu = P.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(66256, 4096)
        self.fc2 = nn.Dense(4096, 4096)
        self.fc3 = nn.Dense(4096, num_classes)

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
