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
""" test_net_infer """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    """ Net definition """

    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal')
        self.bn = nn.BatchNorm2d(64)
        self.fc = nn.Dense(64, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out


def test_net_infer():
    """ test_net_infer """
    Tensor(np.random.randint(0, 255, [1, 3, 224, 224]))
    Net()
