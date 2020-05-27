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
"""
Function:
    test network
Usage:
    python test_predict_save_model.py --path ./
"""

import argparse
import os
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net


class LeNet(nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = P.ReLU()
        self.batch_size = 32

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()
        self.fc1 = nn.Dense(400, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)

    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.reshape(output, (self.batch_size, -1))
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output


parser = argparse.ArgumentParser(description='MindSpore Model Save')
parser.add_argument('--path', default='./lenet_model.ms', type=str, help='model save path')

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    print("test lenet predict start")
    seed = 0
    np.random.seed(seed)
    batch = 1
    channel = 1
    input_h = 32
    input_w = 32
    origin_data = np.random.uniform(low=0, high=255, size=(batch, channel, input_h, input_w)).astype(np.float32)
    origin_data.tofile("lenet_input_data.bin")

    input_data = Tensor(origin_data)
    print(input_data.asnumpy())
    net = LeNet()
    ckpt_file_path = "./tests/ut/python/predict/checkpoint_lenet.ckpt"
    predict_args = parser.parse_args()
    model_path_name = predict_args.path

    is_ckpt_exist = os.path.exists(ckpt_file_path)
    if is_ckpt_exist:
        param_dict = load_checkpoint(ckpoint_file_name=ckpt_file_path)
        load_param_into_net(net, param_dict)
        export(net, input_data, file_name=model_path_name, file_format='LITE')
        print("test lenet predict success.")
    else:
        print("checkpoint file is not exist.")
