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
"""test obfuscate weight"""
import os
import shutil
import numpy as np

from mindspore import nn
from mindspore.common.initializer import TruncatedNormal
from mindspore import save_checkpoint, load_checkpoint
from mindspore import obfuscate_ckpt, load_obf_params_into_net
from mindspore import Tensor
import mindspore.ops as ops
from tests.mark_utils import arg_mark


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    return TruncatedNormal(0.05)


class SubNet(nn.Cell):
    """
    SubNet of lenet.
    """
    def __init__(self):
        super(SubNet, self).__init__()
        self.dense_op = fc_with_initialize(84, 10)

    def construct(self, x):
        return self.dense_op(x)


class LeNet5(nn.Cell):
    """
    Lenet network
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16*5*5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.sub_net = SubNet()
        self.relu = ops.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.sub_net(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_normal_inputs():
    """
    Feature: Test weight obfuscation.
    Description: Test base interfaces for normal inputs.
    Expectation: Success.
    """
    net = LeNet5()
    # generate plain ckpt file
    save_checkpoint(net, './test_net.ckpt')
    input_x = Tensor(np.random.random((32, 1, 32, 32)).astype(np.float32))
    original_predict_result = net(input_x)

    # obfuscate ckpt
    obf_target_modules = ['sub_net', 'dense_op']
    # test_net_obf.ckpt and obf_ratios.npy would be generated and saved under ./obf_files/
    if os.path.exists('./obf_files'):
        shutil.rmtree('./obf_files')
    os.mkdir('./obf_files')
    obf_metadata = obfuscate_ckpt(net, ckpt_files='./', target_modules=obf_target_modules, saved_path='./obf_files')

    # load obf ckpt files
    new_net = LeNet5()
    load_checkpoint('./obf_files/test_net_obf.ckpt', new_net)
    obf_metadata['obf_metadata'] = Tensor(obf_metadata['obf_metadata'])
    obf_net = load_obf_params_into_net(new_net, obf_target_modules)
    obf_predict_result = obf_net(input_x, obf_metadata)

    # compare original predict result and obf net predict result
    assert np.allclose(original_predict_result.numpy(), obf_predict_result.numpy(), rtol=0.01)
