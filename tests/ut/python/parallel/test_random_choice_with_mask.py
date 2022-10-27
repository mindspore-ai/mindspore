# Copyright 2022 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


_input_x = Tensor(np.ones([512, 4]), ms.bool_)


class Net(Cell):
    """
    Create the test net.
    """
    def __init__(self, strategy=None):
        super(Net, self).__init__()
        self.random_choice_with_mask = P.RandomChoiceWithMask().shard(strategy)

    def construct(self, input_x):
        x = self.random_choice_with_mask(input_x)
        return x


def compile_net(net: Cell, *inputs):
    net.set_train()
    _cell_graph_executor.compile(net, *inputs)
    context.reset_auto_parallel_context()


def test_auto_parallel_random_choice_with_mask():
    """
    Feature: test RandomChoiceWithMask auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_context(device_target="GPU")
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net()
    compile_net(net, _input_x)


def test_random_choice_with_mask_wrong_strategy():
    """
    Feature: test RandomChoiceWithMask with illegal strategy
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_context(device_target="GPU")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((8, 1),)
    net = Net(strategy)
    with pytest.raises(RuntimeError):
        compile_net(net, _input_x)
    context.reset_auto_parallel_context()


def test_random_choice_with_mask_not_gpu():
    """
    Feature: RandomChoiceWithMask
    Description: not compile with gpu backend
    Expectation: raise RuntimeError
    """
    context.set_context(device_target="Ascend")
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net()
    with pytest.raises(RuntimeError):
        compile_net(net, _input_x)
    context.reset_auto_parallel_context()
