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
import time
import pytest

from safetensors import safe_open

import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.common.initializer import Normal
from mindspore.train.serialization import load_checkpoint_async
from tests.mark_utils import arg_mark


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


class LeNet5Load(nn.Cell):

    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet5Load, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.conv3 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_top = include_top
        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
            self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))


def remove_ckpt(file_name):
    """remove ckpt."""
    if os.path.exists(file_name) and file_name.endswith(".ckpt"):
        os.chmod(file_name, stat.S_IWRITE)
        os.remove(file_name)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_save_checkpoint_async(mode):
    """
    Feature: mindspore.save_checkpoint async
    Description: save checkpoint async.
    Expectation:success
    """
    net = LeNet5()
    ms.save_checkpoint(net, "./lenet.ckpt",
                       choice_func=lambda x: x.startswith("conv") and not x.startswith("conv1"), async_save="process")
    time.sleep(5)
    output_param_dict2 = ms.load_checkpoint("./lenet.ckpt")
    remove_ckpt("./lenet.ckpt")

    assert 'conv2.weight' in output_param_dict2
    assert 'conv1.weight' not in output_param_dict2
    assert 'fc1.bias' not in output_param_dict2

    ms.save_checkpoint(net, "./lenet.ckpt",
                       choice_func=lambda x: x.startswith("conv") and not x.startswith("conv1"), async_save="thread")
    time.sleep(5)
    output_param_dict3 = ms.load_checkpoint("./lenet.ckpt")
    remove_ckpt("./lenet.ckpt")

    assert 'conv2.weight' in output_param_dict3
    assert 'conv1.weight' not in output_param_dict3
    assert 'fc1.bias' not in output_param_dict3


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_load_checkpoint_async_support_sf(mode):
    """
    Feature: mindspore.load_checkpoint_async
    Description: load safetensors async.
    Expectation: success
    """
    context.set_context(mode=mode, device_target="Ascend")
    net = LeNet5()
    ms.save_checkpoint(net, "./lenet_sf.safetensors",
                       choice_func=lambda x: x.startswith("conv") and not x.startswith("conv1"), format="safetensors")
    output_param_dict_fu = load_checkpoint_async("./lenet_sf.safetensors")
    output_param_dict = output_param_dict_fu.result()
    remove_ckpt("./lenet_sf.safetensors")

    assert 'conv2.weight' in output_param_dict
    assert 'conv1.weight' not in output_param_dict
    assert 'fc1.bias' not in output_param_dict


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_save_checkpoint_sf_with_remove_redundancy(mode):
    """
    Feature: mindspore.save_checkpoint
    Description: Save safetensors checkpoint with remove_redundancy.
    Expectation: success
    """
    context.set_context(mode=mode, device_target="Ascend")
    net = LeNet5()
    ms.save_checkpoint(net, "./sf_remove_redundancy_true.safetensors", format="safetensors", remove_redundancy=True)
    ms.save_checkpoint(net, "./sf_remove_redundancy_false.safetensors", format="safetensors", remove_redundancy=False)

    with safe_open("./sf_remove_redundancy_true.safetensors", framework='np') as f:
        assert f.metadata()["remove_redundancy"] == "True"
    with safe_open("./sf_remove_redundancy_false.safetensors", framework='np') as f:
        assert f.metadata()["remove_redundancy"] == "False"

    remove_ckpt("./sf_remove_redundancy_true.safetensors")
    remove_ckpt("./sf_remove_redundancy_false.safetensors")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_load_checkpoint_sf_warnings(mode):
    """
    Feature: mindspore.save_checkpoint
    Description: Monitor warning messages when loading safetensors.
    Expectation: success
    """
    context.set_context(mode=mode, device_target="Ascend")
    net = LeNet5()
    ms.save_checkpoint(net, "./sf_warning.safetensors", format="safetensors")
    param_dict = ms.load_checkpoint("./sf_warning.safetensors", format="safetensors")
    net_load = LeNet5Load()
    param_not_load, ckpt_not_load = ms.load_param_into_net(net_load, param_dict)
    assert set(param_not_load) == {'conv3.weight'}
    assert set(ckpt_not_load) == {'fc3.bias', 'fc3.weight'}
    remove_ckpt("./sf_warning.safetensors")
