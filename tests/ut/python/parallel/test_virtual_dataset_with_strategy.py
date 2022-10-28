# Copyright 2021 Huawei Technologies Co., Ltd
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

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops.operations.comm_ops import _VirtualDataset
from tests.ut.python.ops.test_math_ops import VirtualLoss
grad_all = C.GradOperation(get_all=True)

class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        return grad_all(self.network)(x, y, b)


class Net1(nn.Cell):
    def __init__(self, strategy1, strategy2, strategy3):
        super().__init__()
        self.virtual_dataset = _VirtualDataset()
        self.matmul1 = P.MatMul().shard(strategy1)
        self.matmul2 = P.MatMul().shard(strategy2)
        self.gelu = P.GeLU().shard(strategy3)

    def construct(self, x, y, b):
        x, y, b = self.virtual_dataset(x, y, b)
        out = self.gelu(self.matmul1(x, y))
        out = self.matmul2(out, b)
        return out

class Net2(nn.Cell):
    def __init__(self, strategy1, strategy2, strategy3):
        super().__init__()
        self.virtual_dataset = _VirtualDataset()
        self.get_next = P.GetNext([ms.float32, ms.float32, ms.float32], [[128, 32], [32, 64], [64]], 3, "")
        self.matmul1 = P.MatMul().shard(strategy1)
        self.biasadd = P.BiasAdd().shard(strategy2)
        self.gelu = P.GeLU().shard(strategy3)

    def construct(self, a, b, c):
        x, y, b = self.get_next()
        x, y, b = self.virtual_dataset(x, y, b)
        out = self.gelu(self.matmul1(x, y))
        out = self.biasadd(out, b)
        return out

class Net3(nn.Cell):
    def __init__(self, strategy1, strategy2, strategy3):
        super().__init__()
        self.matmul1 = P.MatMul().shard(strategy1)
        self.matmul2 = P.MatMul().shard(strategy2)
        self.gelu = P.GeLU().shard(strategy3)

    def construct(self, x, y, b):
        out = self.gelu(self.matmul1(x, y))
        out = self.matmul2(out, b)
        return out

def compile_net(net, x, y, b):
    net.set_train()
    _cell_graph_executor.compile(net, x, y, b)


def test_virtual_dataset_model_parallel_semi_auto_parallel():
    """
    Feature: distribute operator virtual_dataset in auto parallel.
    Description: virtual_dataset/model_parallel/fully shard/repeat in left.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy0 = ((1, 8), (1, 8), (1, 8))
    context.set_auto_parallel_context(dataset_strategy=strategy0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((2, 2), (2, 2))
    strategy3 = ((2, 4),)
    net = GradWrap(NetWithLoss(Net1(strategy1, strategy2, strategy3)))
    x = Tensor(np.ones([128, 32 // 8]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64 // 8]), dtype=ms.float32)
    b = Tensor(np.ones([64, 2048 // 8]), dtype=ms.float32)
    compile_net(net, x, y, b)

def test_virtual_dataset_model_parallel_auto_parallel():
    """
    Feature: distribute operator virtual_dataset in auto parallel.
    Description: virtual_dataset/model_parallel/fully shard/repeat in left.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy0 = ((1, 8), (1, 8), (1, 8))
    context.set_auto_parallel_context(dataset_strategy=strategy0)
    strategy1 = None
    strategy2 = None
    strategy3 = None
    net = GradWrap(NetWithLoss(Net1(strategy1, strategy2, strategy3)))
    x = Tensor(np.ones([128, 32 // 8]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64 // 8]), dtype=ms.float32)
    b = Tensor(np.ones([64, 2048 // 8]), dtype=ms.float32)
    compile_net(net, x, y, b)

def test_virtual_dataset_model_parallel_semi_auto_parallel_diff_input_dim():
    """
    Feature: distribute operator virtual_dataset in auto parallel.
    Description: virtual_dataset/model_parallel/fully shard/repeat in left.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy0 = ((1, 8), (1, 8), (8,))
    context.set_auto_parallel_context(dataset_strategy=strategy0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((1, 8), (8,))
    strategy3 = ((2, 4),)
    x = Tensor(np.ones([128, 32 // 8]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64 // 8]), dtype=ms.float32)
    b = Tensor(np.ones([64 // 8]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net2(strategy1, strategy2, strategy3)))
    compile_net(net, x, y, b)

def test_virtual_dataset_model_parallel_auto_parallel_diff_input_dim():
    """
    Feature: distribute operator virtual_dataset in auto parallel.
    Description: virtual_dataset/model_parallel/fully shard/repeat in left.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy0 = ((1, 8), (1, 8), (8,))
    context.set_auto_parallel_context(dataset_strategy=strategy0)
    strategy1 = None
    strategy2 = None
    strategy3 = None
    x = Tensor(np.ones([128, 32 // 8]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64 // 8]), dtype=ms.float32)
    b = Tensor(np.ones([64 // 8]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net2(strategy1, strategy2, strategy3)))
    compile_net(net, x, y, b)

def test_virtual_dataset_model_parallel_semi_auto_parallel_diff_input_dim_not_fully_shard():
    """
    Feature: distribute operator virtual_dataset in auto parallel.
    Description: virtual_dataset/model_parallel/not fully shard/repeat in left.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    strategy0 = ((1, 8), (1, 8), (1,))
    context.set_auto_parallel_context(dataset_strategy=strategy0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((1, 8), (8,))
    strategy3 = ((2, 4),)
    x = Tensor(np.ones([128, 32 // 8]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64 // 8]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net2(strategy1, strategy2, strategy3)))
    compile_net(net, x, y, b)

def test_virtual_dataset_model_parallel_auto_parallel_diff_input_dim_not_fully_shard():
    """
    Feature: distribute operator virtual_dataset in auto parallel.
    Description: virtual_dataset/model_parallel/not fully shard/repeat in left.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    strategy0 = ((1, 8), (1, 8), (1,))
    context.set_auto_parallel_context(dataset_strategy=strategy0)
    strategy1 = None
    strategy2 = None
    strategy3 = None
    x = Tensor(np.ones([128, 32 // 8]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64 // 8]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net2(strategy1, strategy2, strategy3)))
    compile_net(net, x, y, b)

def test_virtual_dataset_data_parallel_not_fully_shard_repeat_right():
    """
    Feature: distribute operator virtual_dataset in auto parallel.
    Description: virtual_dataset/data_parallel/not fully shard/repeat in right.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=16, global_rank=0)
    strategy0 = ((4, 1), (4, 1), (4,))
    context.set_auto_parallel_context(dataset_strategy=strategy0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((1, 8), (8,))
    strategy3 = ((2, 4),)
    x = Tensor(np.ones([128 // 4, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32 // 4, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64 // 4]), dtype=ms.float32)
    backbone = Net2(strategy1, strategy2, strategy3)
    backbone.virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    net = GradWrap(NetWithLoss(backbone))
    compile_net(net, x, y, b)

def test_without_virtual_dataset_model_parallel_semi_auto_parallel():
    """
    Feature: distribute operator virtual_dataset in auto parallel.
    Description: virtual_dataset/model_parallel/fully shard/repeat in left.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy0 = ((1, 8), (1, 8), (1, 8))
    context.set_auto_parallel_context(dataset_strategy=strategy0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((2, 2), (2, 2))
    strategy3 = ((2, 4),)
    net = GradWrap(NetWithLoss(Net3(strategy1, strategy2, strategy3)))
    x = Tensor(np.ones([128, 32 // 8]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64 // 8]), dtype=ms.float32)
    b = Tensor(np.ones([64, 2048 // 8]), dtype=ms.float32)
    compile_net(net, x, y, b)

def test_without_virtual_dataset_model_parallel_auto_parallel():
    """
    Feature: distribute operator virtual_dataset in auto parallel.
    Description: virtual_dataset/model_parallel/fully shard/repeat in left.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy0 = ((1, 8), (1, 8), (1, 8))
    context.set_auto_parallel_context(dataset_strategy=strategy0)
    strategy1 = None
    strategy2 = None
    strategy3 = None
    net = GradWrap(NetWithLoss(Net3(strategy1, strategy2, strategy3)))
    x = Tensor(np.ones([128, 32 // 8]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64 // 8]), dtype=ms.float32)
    b = Tensor(np.ones([64, 2048 // 8]), dtype=ms.float32)
    compile_net(net, x, y, b)


if __name__ == '__main__':
    context.reset_auto_parallel_context()
