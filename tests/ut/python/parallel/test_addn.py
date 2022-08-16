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

import pytest
import numpy as np

from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.ops import operations as P

from parallel.utils.utils import compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

x_ = (Tensor(np.random.normal(size=[8, 8, 8])),
      Tensor(np.random.normal(size=[8, 8, 8])),
      Tensor(np.random.normal(size=[8, 8, 8])))


class Net(Cell):
    def __init__(self, strategy=None):
        super(Net, self).__init__()
        self.addn = P.AddN().shard(strategy)

    def construct(self, x):
        return self.addn(x)


def test_addn_auto_parallel():
    """
    Feature: test AddN auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net()
    compile_net(net, x_)


def test_addn_model_parallel():
    """
    Feature: test AddN model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((2, 2, 2), (2, 2, 2), (2, 2, 2))
    net = Net(strategy)
    compile_net(net, x_)


def test_addn_strategy_error():
    """
    Feature: test invalid strategy for AddN
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((2, 2, 2), (2, 2, 2), (2, 2, 1))
    net = Net(strategy)
    with pytest.raises(RuntimeError):
        compile_net(net, x_)
