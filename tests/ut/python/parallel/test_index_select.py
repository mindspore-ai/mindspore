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
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops.auto_generate import IndexSelect
from tests.ut.python.ops.test_math_ops import VirtualLoss
from parallel.utils.utils import ParallelValidator


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad_all(self.network)(x, y)


class Net(nn.Cell):
    def __init__(self, axis=0, strategy1=None, strategy2=None, shape=None):
        super().__init__()
        if shape is None:
            shape = [128, 16]
        self.select = IndexSelect().shard(strategy1)
        self.add = P.Add().shard(strategy2)
        self.param = Tensor(np.ones(shape), dtype=ms.float32)
        self.axis = axis
        self.relu = P.ReLU()
        self.reshape = P.Reshape()

    def construct(self, x, y):
        x = self.relu(x)
        x = self.reshape(x, (-1,))
        out = self.select(self.param, self.axis, x)
        out = self.add(out, y)
        return out


class Net2(nn.Cell):
    def __init__(self, axis=0, strategy1=None, strategy2=None, shape=None):
        super().__init__()
        if shape is None:
            shape = [128, 16]
        self.select = IndexSelect().shard(strategy1)
        self.add = P.Add().shard(strategy2)
        self.param = Tensor(np.ones(shape), dtype=ms.float32)
        self.axis = axis

    def construct(self, x, y):
        out = self.select(self.param, self.axis, x)
        out = self.add(out, y)
        return out


def compile_graph(net, device_num, parallel_mode, x, y, search_mode="dynamic_programming"):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode=parallel_mode,
                                      search_mode=search_mode)
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, x, y)
    return phase


def test_index_select_shard_axis():
    """
    Feature: distribute operator index select in semi auto parallel.
    Description:
    Expectation: compile done without error.
    """
    strategy1 = ((8, 1), (1,))
    strategy2 = ((8, 1), (8, 1))
    net = Net2(0, strategy1, strategy2)
    x = Tensor(np.ones([64]), dtype=ms.int32)
    y = Tensor(np.ones([64, 16]), dtype=ms.float32)
    phase = compile_graph(net, 8, "semi_auto_parallel", x, y)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Add-0', ['ReduceScatter-0'])


def test_index_select_shard_axis_has_reshape():
    """
    Feature: distribute operator index select in semi auto parallel.
    Description:
    Expectation: compile done without error.
    """
    strategy1 = ((8, 1), (1,))
    strategy2 = ((8, 1), (8, 1))
    net = Net(0, strategy1, strategy2)
    x = Tensor(np.ones([8, 8]), dtype=ms.int32)
    y = Tensor(np.ones([64, 16]), dtype=ms.float32)
    phase = compile_graph(net, 8, "semi_auto_parallel", x, y)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Add-0', ['ReduceScatter-0'])
