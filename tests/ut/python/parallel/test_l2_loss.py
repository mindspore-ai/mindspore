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

from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.ops import operations as P

from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

x_ = Tensor(np.random.normal(size=[32, 8, 8]).astype(np.float32))


class Net(Cell):
    def __init__(self, strategy=None):
        super(Net, self).__init__()
        self.l2_loss = P.L2Loss().shard(strategy)

    def construct(self, x):
        return self.l2_loss(x)


def test_l2_loss_auto_parallel():
    """
    Feature: test L2Loss auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net()
    compile_net(net, x_)


def test_l2_loss_model_parallel():
    """
    Feature: test L2Loss model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((2, 2, 2),)
    net = Net(strategy)
    phase = compile_net(net, x_)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('AllReduce-0', ['L2Loss-0'])
    assert validator.check_node_attrs('AllReduce-0', {'op': 'sum'})


def test_l2_loss_data_parallel():
    """
    Feature: test L2Loss data parallel
    Description: data parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    net = Net()
    phase = compile_net(net, x_)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('AllReduce-0', ['L2Loss-0'])
    assert validator.check_node_attrs('AllReduce-0', {'op': 'sum'})
