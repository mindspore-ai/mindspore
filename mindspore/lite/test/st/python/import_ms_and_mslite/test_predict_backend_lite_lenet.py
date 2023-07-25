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
# ============================================================================
"""
######################## LiteInfer test ########################

Note:
    To run this scripts, 'mindspore' and 'mindspore_lite' must be installed.
    mindspore_lite must be cloud inference version.
"""
import os

import numpy as np

import mindspore as ms
from mindspore import context
from mindspore.train import Model
import mindspore.nn as nn
from mindspore.common.initializer import Normal
from lite_infer_predict_utils import predict_backend_lite, _get_max_index_from_res


# pylint: disable=I1101
os.environ['MSLITE_ENABLE_CLOUD_INFERENCE'] = "on"


class LeNet5(nn.Cell):
    """
    Lenet network
    """
    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_top = include_top
        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
            self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
            self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_model():
    """
    create model.
    """
    network = LeNet5(10)
    ms_model = Model(network)
    return ms_model


def test_predict_backend_lite_lenet():
    """
    Feature: test LiteInfer predict.
    Description: test LiteInfer predict.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    fake_input = ms.Tensor(np.ones((1, 1, 32, 32)).astype(np.float32))

    model = create_model()
    res_lite, avg_t_lite = predict_backend_lite(model, fake_input)
    print("Predict using backend lite, res: ", _get_max_index_from_res(res_lite))
    print(f"Predict using backend lite, avg time: {avg_t_lite * 1000} ms")

    assert avg_t_lite > 0.0  # assert predict is ok
