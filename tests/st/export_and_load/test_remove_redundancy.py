# Copyright 2024 Huawei Technologies Co., Ltd
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
import pytest
from mindspore import context
import shutil

import mindspore.nn as nn
from mindspore import Parameter, Tensor, save_checkpoint
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import CheckpointConfig


class MyCell(nn.Cell):
    def __init__(self):
        super(MyCell, self).__init__()
        self.param = Parameter(Tensor([1, 2, 3]))

    def construct(self, x):
        return x + self.param


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_remove_redundancy_1_1(mode):
    '''
    Feature: remove_redundancy save ckpt and load ckpt.
    Description: Saving and loading checkpoints with redundancy elimination.
    Expectation: success.
    '''
    for i in range(8):
        os.mkdir(f"device{i}_redundancy11")
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True " \
                    "pytest -s remove_redundancy.py::test_remove_redundancy_save_True_load_True")
    assert ret == 0
    for i in range(8):
        shutil.rmtree(f"device{i}_redundancy11")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_remove_redundancy_0_1(mode):
    '''
    Feature: save full ckpt and remove_redundancy load ckpt.
    Description: Full checkpoint saving and redundant-free checkpoint loading.
    Expectation: success.
    '''
    for i in range(8):
        os.mkdir(f"device{i}_redundancy01")
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True " \
                    "pytest -s remove_redundancy.py::test_remove_redundancy_save_False_load_True")
    assert ret == 0
    for i in range(8):
        shutil.rmtree(f"device{i}_redundancy01")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_save_remove_redundancy_error(mode):
    '''
    Feature: Verify error reporting during redundant-free saving.
    Description: Verify error reporting during redundant-free saving.
    Expectation: success.
    '''
    with pytest.raises(ValueError):
        CheckpointConfig(remove_redundancy="string")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_load_remove_redundancy_error(mode):
    '''
    Feature: Verify error reporting during redundant-free loading.
    Description: Verify error reporting during redundant-free loading.
    Expectation: success.
    '''
    net = MyCell()
    save_checkpoint(net, "./redundancy_error.ckpt")
    param_dict = load_checkpoint("./redundancy_error.ckpt")

    with pytest.raises(ValueError):
        load_checkpoint("./redundancy_error.ckpt", remove_redundancy="string")
    with pytest.raises(ValueError):
        load_param_into_net(net, param_dict, remove_redundancy="string")
