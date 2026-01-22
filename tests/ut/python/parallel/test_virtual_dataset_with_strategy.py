# Copyright 2021-2025 Huawei Technologies Co., Ltd
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

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.graph.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops.operations.comm_ops import _VirtualDataset
from mindspore.parallel import Layout
from tests.ut.python.ops.test_math_ops import VirtualLoss
from parallel.utils.utils import ParallelValidator

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


class Net4(nn.Cell):
    def __init__(self, strategy1, strategy2, strategy3):
        super().__init__()
        self.virtual_dataset = _VirtualDataset()
        self.matmul1 = P.MatMul().shard(strategy1)
        self.matmul2 = P.MatMul().shard(strategy2)
        self.gelu = P.GeLU().shard(strategy3)

    def construct(self, x, y, b):
        x, y, b = self.virtual_dataset(x, y, b)
        out = self.gelu(self.matmul1(x, y[0]))
        return out


class Net5(nn.Cell):
    def __init__(self, strategy1, strategy2, strategy3):
        super().__init__()
        self.virtual_dataset = _VirtualDataset()
        self.matmul = P.MatMul().shard(strategy1)
        self.add = P.Add().shard(strategy2)
        self.gelu = P.GeLU().shard(strategy3)

    def construct(self, x, y, b):
        out1 = self.matmul(x, y)
        out2 = self.gelu(b[0])
        out = self.add(out1, out2)
        return out


def compile_net(net, x, y, b):
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, x, y, b)
    return phase


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
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation")
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

def test_virtual_dataset_model_parallel_auto_parallel_with_strategy():
    """
    Feature: distribute operator virtual_dataset in auto parallel.
    Description: virtual_dataset/model_parallel/fully shard/repeat in left.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation")
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
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming")
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
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation")
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
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation")
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


def test_list_tensor_input_virtual_dataset_full_batch():
    """
    Feature: distribute operator virtual_dataset in auto parallel.
    Description: virtual_dataset/model_parallel/fully shard/repeat in left.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((2, 2), (2, 2))
    strategy3 = ((2, 4),)
    net = GradWrap(NetWithLoss(Net4(strategy1, strategy2, strategy3)))
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = [Tensor(np.ones([32, 64]), dtype=ms.float32), Tensor(np.ones([32, 64]), dtype=ms.float32)]
    b = Tensor(np.ones([64, 2048]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_list_tensor_input_virtual_dataset_without_full_batch():
    """
    Feature: distribute operator virtual_dataset in auto parallel.
    Description: virtual_dataset/model_parallel/fully shard/repeat in left.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((2, 2), (2, 2))
    strategy3 = ((2, 4),)
    net = GradWrap(NetWithLoss(Net5(strategy1, strategy2, strategy3)))
    x = Tensor(np.ones([128, 256]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = [Tensor(np.ones([1024, 64]), dtype=ms.float32)]
    compile_net(net, x, y, b)


def test_virtual_dataset_model_parallel_semi_auto_parallel_with_layout_1():
    """
    Feature: distribute operator virtual_dataset in auto parallel.
    Description: virtual_dataset/model_parallel/fully shard/repeat in left.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    layout = Layout((4, 2), ("dp", "mp"))
    strategy0 = (layout("dp", "mp"), layout("dp", "mp"), layout("mp", "dp"))
    context.set_auto_parallel_context(dataset_strategy=strategy0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((2, 2), (2, 2))
    strategy3 = ((2, 4),)
    net = GradWrap(NetWithLoss(Net1(strategy1, strategy2, strategy3)))
    x = Tensor(np.ones([128 // 4, 32 // 2]), dtype=ms.float32)
    y = Tensor(np.ones([32 // 4, 64 // 2]), dtype=ms.float32)
    b = Tensor(np.ones([64 // 2, 2048 // 4]), dtype=ms.float32)
    phase = compile_net(net, x, y, b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('MatMul-0', ['Reshape-2', 'StridedSlice-1'])


def test_virtual_dataset_model_parallel_semi_auto_parallel_with_layout_2():
    """
    Feature: distribute operator virtual_dataset in auto parallel.
    Description: virtual_dataset/model_parallel/fully shard/repeat in left.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "mp", "sp"))
    strategy0 = (layout(("dp", "mp"), "sp"), layout(("dp", "mp"), "sp"), layout(("mp", "dp"), "sp"))
    context.set_auto_parallel_context(dataset_strategy=strategy0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((2, 2), (2, 2))
    strategy3 = ((2, 4),)
    net = GradWrap(NetWithLoss(Net1(strategy1, strategy2, strategy3)))
    x = Tensor(np.ones([128 // 4, 32 // 2]), dtype=ms.float32)
    y = Tensor(np.ones([32 // 4, 64 // 2]), dtype=ms.float32)
    b = Tensor(np.ones([64 // 4, 2048 // 2]), dtype=ms.float32)
    phase = compile_net(net, x, y, b)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('MatMul-0', ['Reshape-2', 'StridedSlice-1'])


def test_virtual_dataset_model_parallel_semi_auto_parallel_with_layout_3():
    """
    Feature: distribute operator virtual_dataset in auto parallel.
    Description: virtual_dataset/model_parallel/fully shard/repeat in left.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    layout = Layout((2, 2, 2, 2), ("dp", "mp", "sp", "interleaved_parallel"))
    strategy0 = (layout(("dp", "mp"), "sp"), layout(("dp", "mp"), "sp"), layout(("mp", "dp"), "sp"))
    with pytest.raises(ValueError):
        context.set_auto_parallel_context(dataset_strategy=strategy0)


def test_dataset_strategy_with_layout_using_autoparallel_cell():
    """
    Feature: test AutoParallel(cell).dataset_strategy(config).
    Description: virtual_dataset/model_parallel/fully shard/repeat in left.
    Expectation: compile done without error.
    """
    from mindspore.parallel.auto_parallel import AutoParallel
    from hccl_test.manage.api import Hccl
    from mindspore.nn.utils import no_init_parameters
    hccl = Hccl()
    hccl.rank_id = 0
    hccl.rank_size = 8
    layout = Layout((2, 2, 2), ("dp", "mp", "sp"))
    strategy0 = (layout(("dp", "mp"), "sp"), layout(("dp", "mp"), "sp"), layout(("mp", "dp"), "sp"))
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((2, 2), (2, 2))
    strategy3 = ((2, 4),)
    with no_init_parameters():
        net = GradWrap(NetWithLoss(Net1(strategy1, strategy2, strategy3)))
    parallel_net = AutoParallel(net, parallel_mode="semi_auto")
    parallel_net.dataset_strategy(strategy0)
    x = Tensor(np.ones([128 // 4, 32 // 2]), dtype=ms.float32)
    y = Tensor(np.ones([32 // 4, 64 // 2]), dtype=ms.float32)
    b = Tensor(np.ones([64 // 4, 2048 // 2]), dtype=ms.float32)
    phase = compile_net(parallel_net, x, y, b)
    validator = ParallelValidator(parallel_net, phase)
    assert validator.check_node_inputs_has('MatMul-0', ['Reshape-1', 'Reshape-3', False, False])

if __name__ == '__main__':
    context.reset_auto_parallel_context()
