# Copyright 2023-2024 Huawei Technologies Co., Ltd
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
import os
import stat

import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.common.initializer import Normal
from mindspore.train.serialization import load_checkpoint_async


class LeNet5(nn.Cell):

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
        '''
        Forward network.
        '''
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


def remove_ckpt(file_name):
    """remove ckpt."""
    if os.path.exists(file_name) and file_name.endswith(".ckpt"):
        os.chmod(file_name, stat.S_IWRITE)
        os.remove(file_name)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_save_checkpoint(mode):
    """
    Feature: mindspore.save_checkpoint
    Description: Save checkpoint to a specified file.
    Expectation: success
    """
    context.set_context(mode=mode)
    net = LeNet5()
    ms.save_checkpoint(net, "./lenet.ckpt",
                       choice_func=lambda x: x.startswith("conv") and not x.startswith("conv1"))
    output_param_dict = ms.load_checkpoint("./lenet.ckpt")
    assert 'conv2.weight' in output_param_dict
    assert 'conv1.weight' not in output_param_dict
    assert 'fc1.bias' not in output_param_dict

    param_dict = ms.load_checkpoint("./lenet.ckpt")
    ms.save_checkpoint(param_dict, "./lenet_dict.ckpt")
    output_param_dict1 = ms.load_checkpoint("./lenet_dict.ckpt")
    remove_ckpt("./lenet.ckpt")
    remove_ckpt("./lenet_dict.ckpt")
    assert 'conv2.weight' in output_param_dict1
    assert 'conv1.weight' not in output_param_dict1
    assert 'fc1.bias' not in output_param_dict1

    param_list = net.trainable_params()
    ms.save_checkpoint(param_list, "./lenet_list.ckpt",
                       choice_func=lambda x: x.startswith("conv") and not x.startswith("conv1"))
    output_param_dict2 = ms.load_checkpoint("./lenet_list.ckpt")
    remove_ckpt("./lenet_list.ckpt")
    assert 'conv2.weight' in output_param_dict2
    assert 'conv1.weight' not in output_param_dict2
    assert 'fc1.bias' not in output_param_dict2

    empty_list = []
    append_dict = {"lr": 0.01}
    ms.save_checkpoint(empty_list, "./lenet_empty_list.ckpt", append_dict=append_dict)
    output_empty_list = ms.load_checkpoint("./lenet_empty_list.ckpt")
    remove_ckpt("./lenet_empty_list.ckpt")
    assert "lr" in output_empty_list


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_load_checkpoint_async(mode):
    """
    Feature: mindspore.load_checkpoint_async
    Description: load checkpoint async.
    Expectation: success
    """
    context.set_context(mode=mode, device_target="Ascend")
    net = LeNet5()
    ms.save_checkpoint(net, "./lenet.ckpt",
                       choice_func=lambda x: x.startswith("conv") and not x.startswith("conv1"))
    output_param_dict_fu = load_checkpoint_async("./lenet.ckpt")
    output_param_dict = output_param_dict_fu.result()
    remove_ckpt("./lenet.ckpt")

    assert 'conv2.weight' in output_param_dict
    assert 'conv1.weight' not in output_param_dict
    assert 'fc1.bias' not in output_param_dict
