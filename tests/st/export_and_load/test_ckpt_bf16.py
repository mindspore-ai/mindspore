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
import os
import stat

import pytest
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.common.initializer import Normal
from tests.mark_utils import arg_mark


class LeNet5(nn.Cell):

    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid', dtype=ms.bfloat16)
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid', dtype=ms.bfloat16)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_top = include_top
        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02), dtype=ms.bfloat16)
            self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02), dtype=ms.bfloat16)
            self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02), dtype=ms.bfloat16)

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


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_save_load_checkpoint_with_bf16_graph(mode):
    """
    Feature: Checkpoint with bfloat16
    Description: Save and load checkpoint with bfloat16.
    Expectation: success
    """
    context.set_context(mode=mode)
    net = LeNet5()
    ms.save_checkpoint(net, "./lenet.ckpt",
                       choice_func=lambda x: x.startswith("conv") and not x.startswith("conv1"))
    output_param_dict = ms.load_checkpoint("./lenet.ckpt")
    assert 'conv2.weight' in output_param_dict
    assert output_param_dict["conv2.weight"].dtype == ms.bfloat16
    assert 'conv1.weight' not in output_param_dict
    assert 'fc1.bias' not in output_param_dict

    param_dict = ms.load_checkpoint("./lenet.ckpt")
    ms.save_checkpoint(param_dict, "./lenet_dict.ckpt")
    output_param_dict1 = ms.load_checkpoint("./lenet_dict.ckpt")
    remove_ckpt("./lenet.ckpt")
    remove_ckpt("./lenet_dict.ckpt")
    assert 'conv2.weight' in output_param_dict1
    assert output_param_dict1["conv2.weight"].dtype == ms.bfloat16
    assert 'conv1.weight' not in output_param_dict1
    assert 'fc1.bias' not in output_param_dict1

    param_list = net.trainable_params()
    ms.save_checkpoint(param_list, "./lenet_list.ckpt",
                       choice_func=lambda x: x.startswith("conv") and not x.startswith("conv1"))
    output_param_dict2 = ms.load_checkpoint("./lenet_list.ckpt")
    remove_ckpt("./lenet_list.ckpt")
    assert 'conv2.weight' in output_param_dict2
    assert output_param_dict2["conv2.weight"].dtype == ms.bfloat16
    assert 'conv1.weight' not in output_param_dict2
    assert 'fc1.bias' not in output_param_dict2

    empty_list = []
    lr_tensor = ms.Tensor([[0.01]], ms.bfloat16)
    lr_parameter = ms.Parameter(lr_tensor)
    append_dict = {"lr": lr_tensor, "lr_parameter": lr_parameter}
    ms.save_checkpoint(empty_list, "./lenet_empty_list.ckpt", append_dict=append_dict)
    output_empty_list = ms.load_checkpoint("./lenet_empty_list.ckpt")
    remove_ckpt("./lenet_empty_list.ckpt")
    assert "lr" in output_empty_list
    assert output_empty_list["lr"].dtype == ms.bfloat16
    assert "lr_parameter" in output_empty_list
    assert output_empty_list["lr_parameter"].dtype == ms.bfloat16



@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_save_load_checkpoint_with_bf16_pynative(mode):
    """
    Feature: Checkpoint with bfloat16
    Description: Save and load checkpoint with bfloat16.
    Expectation: success
    """
    context.set_context(mode=mode)
    net = LeNet5()
    ms.save_checkpoint(net, "./lenet_1.ckpt",
                       choice_func=lambda x: x.startswith("conv") and not x.startswith("conv1"))
    output_param_dict = ms.load_checkpoint("./lenet_1.ckpt")
    assert 'conv2.weight' in output_param_dict
    assert output_param_dict["conv2.weight"].dtype == ms.bfloat16
    assert 'conv1.weight' not in output_param_dict
    assert 'fc1.bias' not in output_param_dict

    param_dict = ms.load_checkpoint("./lenet_1.ckpt")
    ms.save_checkpoint(param_dict, "./lenet_dict_1.ckpt")
    output_param_dict1 = ms.load_checkpoint("./lenet_dict_1.ckpt")
    remove_ckpt("./lenet_1.ckpt")
    remove_ckpt("./lenet_dict_1.ckpt")
    assert 'conv2.weight' in output_param_dict1
    assert output_param_dict1["conv2.weight"].dtype == ms.bfloat16
    assert 'conv1.weight' not in output_param_dict1
    assert 'fc1.bias' not in output_param_dict1

    param_list = net.trainable_params()
    ms.save_checkpoint(param_list, "./lenet_list_1.ckpt",
                       choice_func=lambda x: x.startswith("conv") and not x.startswith("conv1"))
    output_param_dict2 = ms.load_checkpoint("./lenet_list_1.ckpt")
    remove_ckpt("./lenet_list_1.ckpt")
    assert 'conv2.weight' in output_param_dict2
    assert output_param_dict2["conv2.weight"].dtype == ms.bfloat16
    assert 'conv1.weight' not in output_param_dict2
    assert 'fc1.bias' not in output_param_dict2

    empty_list = []
    lr_tensor = ms.Tensor([[0.01]], ms.bfloat16)
    lr_parameter = ms.Parameter(lr_tensor)
    append_dict = {"lr": lr_tensor, "lr_parameter": lr_parameter}
    ms.save_checkpoint(empty_list, "./lenet_empty_list_1.ckpt", append_dict=append_dict)
    output_empty_list = ms.load_checkpoint("./lenet_empty_list_1.ckpt")
    remove_ckpt("./lenet_empty_list_1.ckpt")
    assert "lr" in output_empty_list
    assert output_empty_list["lr"].dtype == ms.bfloat16
    assert "lr_parameter" in output_empty_list
    assert output_empty_list["lr_parameter"].dtype == ms.bfloat16



@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_save_load_checkpoint_with_bf16_pynative_accuracy(mode):
    """
    Feature: the accuracy of Checkpoint with bfloat16
    Description: check the accuracy of checkpoint with bfloat16.
    Expectation: success
    """
    context.set_context(mode=mode)
    net = LeNet5()
    file_name = "./lenet_acc.ckpt"
    ms.save_checkpoint(net, file_name)
    output_param_dict = ms.load_checkpoint(file_name)
    remove_ckpt(file_name)
    for _, param in net.parameters_and_names():
        assert param.name in output_param_dict
        assert np.allclose(param.float().asnumpy(), output_param_dict[param.name].float().asnumpy())


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_list_ckpt_error(mode):
    """
    Feature: save_checkpoint raises list typeerror.
    Description: Whether typeerror of list can be caught.
    Expectation: success.
    """
    x = ms.Parameter(ms.Tensor([1, 2, 3.4]))

    save_obj1 = [{"name": 12, "data": x}]
    with pytest.raises(TypeError):
        ms.save_checkpoint(save_obj1, "./tmp.ckpt")

    save_obj1 = [{"name": "parameter1", "data": 12}]
    with pytest.raises(TypeError):
        ms.save_checkpoint(save_obj1, "./tmp.ckpt")
