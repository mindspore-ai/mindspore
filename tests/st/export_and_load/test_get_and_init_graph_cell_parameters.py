# Copyright 2021-2023 Huawei Technologies Co., Ltd
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

"""test get and init GraphCell parameters"""

import os
import stat

import numpy as np
import pytest

from mindspore import Tensor, Parameter
from mindspore import context
from mindspore import export, load, save_checkpoint, load_checkpoint
from mindspore import nn


class TestNet(nn.Cell):
    def __init__(self):
        super(TestNet, self).__init__()
        self.flag = False
        self.weight = Parameter(np_param, requires_grad=True)
        self.dense = nn.Dense(3, 4)

    def construct(self, x, y):
        if self.flag:
            ret = self.dense(x * self.weight)
        else:
            ret = x * y * self.weight
        self.weight += 1.0
        return ret


np_a = np.ones((2, 3), np.float32) + 2
np_b = np.ones((2, 3), np.float32) + 3
np_param = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
input_a = Tensor(np_a)
input_b = Tensor(np_b)


def load_mindir_and_update_params(mindir_name, ckpt_name):
    net = TestNet()
    export(net, input_a, input_b, file_name=mindir_name[:-7], file_format='MINDIR')

    load_net = nn.GraphCell(graph=load(mindir_name))
    ret = load_net(input_a, input_b)
    save_checkpoint(load_net, ckpt_name)
    assert np.array_equal(ret.asnumpy(), np_a * np_b * np_param)
    assert np.array_equal(load_net.trainable_params()[0].asnumpy(), np_param + 1.0)

    params_init = load_checkpoint(ckpt_name)
    load_net_with_new_params = nn.GraphCell(graph=load(mindir_name), params_init=params_init)
    return load_net_with_new_params


def get_and_init_graph_cell_parameters():
    mindir_name = f"{context.get_context('mode')}_test_graph_cell_net.mindir"
    ckpt_name = f"{context.get_context('mode')}_test_graph_cell_net.ckpt"
    load_net = load_mindir_and_update_params(mindir_name, ckpt_name)
    ret = load_net(input_a, input_b)
    assert np.array_equal(ret.asnumpy(), np_a * np_b * (np_param + 1.0))
    assert np.array_equal(load_net.trainable_params()[0].asnumpy(), np_param + 2.0)

    if os.path.isfile(mindir_name):
        os.chmod(mindir_name, stat.S_IWUSR)
        os.remove(mindir_name)
    if os.path.isfile(ckpt_name):
        os.chmod(ckpt_name, stat.S_IWUSR)
        os.remove(ckpt_name)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_get_and_init_graph_cell_parameters_in_graph_mode():
    """
    Description: load mind ir and update parameters in graph mode.
    Expectation: generate a graph with updated parameters.
    """
    context.set_context(mode=context.GRAPH_MODE)
    get_and_init_graph_cell_parameters()
