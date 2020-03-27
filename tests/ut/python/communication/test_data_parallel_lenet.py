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

"""
@File   : test_data_parallel_lenet.py
@Desc   : test data parallel lenet
"""
import os
import numpy as np

import mindspore.nn as nn
import mindspore.context as context
from mindspore.ops import operations as P
from mindspore import Tensor, Model, ParallelMode
from mindspore.nn.optim import Momentum

_current_dir = os.path.dirname(os.path.realpath(__file__)) + "/../test_data"

class LeNet5(nn.Cell):
    """LeNet5 definition"""
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)
        self.flatten = P.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DatasetLenet():
    """DatasetLenet definition"""
    def __init__(self, predict, label, length=3):
        self.predict = predict
        self.label = label
        self.index = 0
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.predict, self.label

    def reset(self):
        self.index = 0


def test_lenet5_train_step_training_pynative():
    """test_lenet5_train_step_training_pynative"""
    context.set_context(mode=context.PYNATIVE_MODE)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                      device_num=8, mirror_mean=True)
    size = 3
    predict = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    DatasetLenet(predict, label, 2)
    network = LeNet5()
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Momentum(network.get_parameters(), learning_rate=0.1, momentum=0.9)
    Model(network=network, loss_fn=loss_fn, optimizer=optimizer)
    context.set_context(mode=context.GRAPH_MODE)
    context.reset_auto_parallel_context()
