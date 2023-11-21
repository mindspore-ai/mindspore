# Copyright 2023 Huawei Technologies Co., Ltd
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
import os
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import ops, context, load_mindir
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import TruncatedNormal
from mindspore.train.serialization import export

def weight_variable():
    return TruncatedNormal(0.02)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


class LeNet5(nn.Cell):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.batch_size = 32
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = ops.Reshape()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.reshape(x, (self.batch_size, -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def _custom_func(mindir_model):
    mindir_model.producer_name = "pandu11111"
    mindir_model.producer_version = "1.0"
    mindir_model.user_info["version"] = "1.0"
    return mindir_model

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_load_mindir_with_custom_func():
    """
    Feature: Load LeNet to MindIR with custom_func
    Description: Test load LeNet to load network into MindIR with self-defined custom_func
    Expectation: load successfully and param compare successfully
    """
    context.set_context(mode=context.GRAPH_MODE)
    network = LeNet5()
    network.set_train()

    inputs = Tensor(np.zeros([32, 1, 32, 32]).astype(np.float32))
    export(network, inputs, file_name="test_lenet_load", file_format='MINDIR', custom_func=_custom_func)
    mindir_name = "test_lenet_load.mindir"
    assert os.path.exists(mindir_name)

    mindir_model = load_mindir("test_lenet_load.mindir")
    assert mindir_model.producer_name == "pandu11111"
    assert mindir_model.producer_version == "1.0"
    assert mindir_model.user_info["version"] == "1.0"
