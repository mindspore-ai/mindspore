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
""" test_checkparam """
import numpy as np
import pytest

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.common.tensor import Tensor


class LeNet5(nn.Cell):
    """ LeNet5 definition """

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, pad_mode="valid")
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode="valid")
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 3)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def predict_checke_param(in_str):
    """ predict_checke_param """
    net = LeNet5()  # neural network
    context.set_context(mode=context.GRAPH_MODE)
    model = mindspore.train.Model(net)

    a1, a2, b1, b2, b3, b4 = in_str.strip().split()
    a1 = int(a1)
    a2 = int(a2)
    b1 = int(b1)
    b2 = int(b2)
    b3 = int(b3)
    b4 = int(b4)

    nd_data = np.random.randint(a1, a2, [b1, b2, b3, b4])
    input_data = Tensor(nd_data, mindspore.float32)
    model.predict(input_data)


def test_predict_checke_param_failed():
    """ test_predict_checke_param_failed """
    in_str = "0 255 0 3 32 32"
    with pytest.raises(ValueError):
        predict_checke_param(in_str)
