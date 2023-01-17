# Copyright 2019 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.context import set_auto_parallel_context, reset_auto_parallel_context
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


grad_all = C.GradOperation(get_all=True)


# model_parallel test
def test_six_matmul_save():
    class NetWithLoss(nn.Cell):
        def __init__(self, network):
            super(NetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x1, x6):
            predict = self.network(x1, x6)
            return self.loss(predict)

    class GradWrap(nn.Cell):
        def __init__(self, network):
            super(GradWrap, self).__init__()
            self.network = network

        def construct(self, x1, x6):
            return grad_all(self.network)(x1, x6)

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3, strategy4, strategy5, strategy6):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)
            self.matmul3 = P.MatMul().shard(strategy3)
            self.matmul4 = P.MatMul().shard(strategy4)
            self.matmul5 = P.MatMul().shard(strategy5)
            self.matmul6 = P.MatMul().shard(strategy6)
            self.weight1 = Parameter(Tensor(np.ones([32, 64]), dtype=ms.float32), name="weight1")
            self.weight2 = Parameter(Tensor(np.ones([64, 64]), dtype=ms.float32), name="weight2")
            self.weight3 = Parameter(Tensor(np.ones([64, 128]), dtype=ms.float32), name="weight3")
            self.weight4 = Parameter(Tensor(np.ones([128, 64]), dtype=ms.float32), name="weight4")
            self.weight5 = Parameter(Tensor(np.ones([64, 128]), dtype=ms.float32), name="weight5")
            self.weight6 = Parameter(Tensor(np.ones([32, 128]), dtype=ms.float32), name="weight6")

        def construct(self, x1, x6):
            out = self.matmul1(x1, self.weight1)
            out = self.matmul2(out, self.weight2)
            out = self.matmul3(out, self.weight3)
            out = self.matmul4(out, self.weight4)
            out = self.matmul5(out, self.weight5)
            out = out + self.weight6
            out = self.matmul6(out, x6)
            return out

    reset_auto_parallel_context()
    set_auto_parallel_context(device_num=8, global_rank=0, strategy_ckpt_save_file="./strategy_stage1.ckpt",
                              group_ckpt_save_file="./group_stage1.ckpt", dataset_strategy="full_batch")
    strategy1 = ((8, 1), (1, 1))
    strategy2 = ((1, 8), (8, 1))
    strategy3 = ((2, 2), (2, 2))
    strategy4 = ((1, 1), (1, 8))
    strategy5 = ((4, 2), (2, 1))
    strategy6 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3, strategy4, strategy5, strategy6)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    x1 = Tensor(np.ones([32, 32]), dtype=ms.float32)
    x6 = Tensor(np.ones([128, 32]), dtype=ms.float32)
    net.set_train()
    _cell_graph_executor.compile(net, x1, x6)


# remove matmul2, add matmul7
def six_matmul_load():
    class NetWithLoss(nn.Cell):
        def __init__(self, network):
            super(NetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x1, x6, x7):
            predict = self.network(x1, x6, x7)
            return self.loss(predict)

    class GradWrap(nn.Cell):
        def __init__(self, network):
            super(GradWrap, self).__init__()
            self.network = network

        def construct(self, x1, x6, x7):
            return grad_all(self.network)(x1, x6, x7)

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy3, strategy4, strategy5, strategy6, strategy7):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul3 = P.MatMul().shard(strategy3)
            self.matmul4 = P.MatMul().shard(strategy4)
            self.matmul5 = P.MatMul().shard(strategy5)
            self.matmul6 = P.MatMul().shard(strategy6)
            self.matmul7 = P.MatMul().shard(strategy7)
            self.weight1 = Parameter(Tensor(np.ones([32, 64]), dtype=ms.float32), name="weight1")
            self.weight3 = Parameter(Tensor(np.ones([64, 128]), dtype=ms.float32), name="weight3")
            self.weight4 = Parameter(Tensor(np.ones([128, 64]), dtype=ms.float32), name="weight4")
            self.weight5 = Parameter(Tensor(np.ones([64, 128]), dtype=ms.float32), name="weight5")
            self.weight6 = Parameter(Tensor(np.ones([32, 128]), dtype=ms.float32), name="weight6")

        def construct(self, x1, x6, x7):
            out = self.matmul1(x1, self.weight1)
            out = self.matmul3(out, self.weight3)
            out = self.matmul4(out, self.weight4)
            out = self.matmul5(out, self.weight5)
            out = out + self.weight6
            out = self.matmul6(out, x6)
            out = self.matmul7(out, x7)
            return out

    reset_auto_parallel_context()
    set_auto_parallel_context(device_num=8, global_rank=0, strategy_ckpt_load_file="./strategy_stage1.ckpt",
                              group_ckpt_save_file="./group_stage1.ckpt", dataset_strategy="full_batch")
    strategy1 = ((8, 1), (1, 1))
    strategy3 = ((8, 1), (1, 1))
    strategy4 = ((8, 1), (1, 1))
    strategy5 = ((8, 1), (1, 1))
    strategy6 = ((8, 1), (1, 1))
    strategy7 = ((8, 1), (1, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy3, strategy4, strategy5, strategy6, strategy7)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    x1 = Tensor(np.ones([32, 32]), dtype=ms.float32)
    x6 = Tensor(np.ones([128, 32]), dtype=ms.float32)
    x7 = Tensor(np.ones([32, 32]), dtype=ms.float32)
    net.set_train()
    _cell_graph_executor.compile(net, x1, x6, x7)


# model_parallel test
def test_six_matmul_save_auto():
    class NetWithLoss(nn.Cell):
        def __init__(self, network):
            super(NetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x1, x6):
            predict = self.network(x1, x6)
            return self.loss(predict)

    class GradWrap(nn.Cell):
        def __init__(self, network):
            super(GradWrap, self).__init__()
            self.network = network

        def construct(self, x1, x6):
            return grad_all(self.network)(x1, x6)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul1 = P.MatMul()
            self.matmul2 = P.MatMul()
            self.matmul3 = P.MatMul()
            self.matmul4 = P.MatMul()
            self.matmul5 = P.MatMul()
            self.matmul6 = P.MatMul()
            self.weight1 = Parameter(Tensor(np.ones([32, 64]), dtype=ms.float32), name="weight1")
            self.weight2 = Parameter(Tensor(np.ones([64, 64]), dtype=ms.float32), name="weight2")
            self.weight3 = Parameter(Tensor(np.ones([64, 128]), dtype=ms.float32), name="weight3")
            self.weight4 = Parameter(Tensor(np.ones([128, 64]), dtype=ms.float32), name="weight4")
            self.weight5 = Parameter(Tensor(np.ones([64, 128]), dtype=ms.float32), name="weight5")
            self.weight6 = Parameter(Tensor(np.ones([32, 128]), dtype=ms.float32), name="weight6")

        def construct(self, x1, x6):
            out = self.matmul1(x1, self.weight1)
            out = self.matmul2(out, self.weight2)
            out = self.matmul3(out, self.weight3)
            out = self.matmul4(out, self.weight4)
            out = self.matmul5(out, self.weight5)
            out = out + self.weight6
            out = self.matmul6(out, x6)
            return out

    reset_auto_parallel_context()
    set_auto_parallel_context(device_num=8, global_rank=0, strategy_ckpt_save_file="./strategy_stage1_auto.json")
    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel", dataset_strategy="full_batch")
    x1 = Tensor(np.ones([32, 32]), dtype=ms.float32)
    x6 = Tensor(np.ones([128, 32]), dtype=ms.float32)
    net.set_train()
    _cell_graph_executor.compile(net, x1, x6)


# remove matmul2, add matmul7
def six_matmul_load_auto():
    class NetWithLoss(nn.Cell):
        def __init__(self, network):
            super(NetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x1, x6, x7):
            predict = self.network(x1, x6, x7)
            return self.loss(predict)

    class GradWrap(nn.Cell):
        def __init__(self, network):
            super(GradWrap, self).__init__()
            self.network = network

        def construct(self, x1, x6, x7):
            return grad_all(self.network)(x1, x6, x7)

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy3, strategy4, strategy5):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul3 = P.MatMul().shard(strategy3)
            self.matmul4 = P.MatMul().shard(strategy4)
            self.matmul5 = P.MatMul().shard(strategy5)
            self.matmul6 = P.MatMul()
            self.matmul7 = P.MatMul()
            self.weight1 = Parameter(Tensor(np.ones([32, 64]), dtype=ms.float32), name="weight1")
            self.weight3 = Parameter(Tensor(np.ones([64, 128]), dtype=ms.float32), name="weight3")
            self.weight4 = Parameter(Tensor(np.ones([128, 64]), dtype=ms.float32), name="weight4")
            self.weight5 = Parameter(Tensor(np.ones([64, 128]), dtype=ms.float32), name="weight5")
            self.weight6 = Parameter(Tensor(np.ones([32, 128]), dtype=ms.float32), name="weight6")

        def construct(self, x1, x6, x7):
            out = self.matmul1(x1, self.weight1)
            out = self.matmul3(out, self.weight3)
            out = self.matmul4(out, self.weight4)
            out = self.matmul5(out, self.weight5)
            out = out + self.weight6
            out = self.matmul6(out, x6)
            out = self.matmul7(out, x7)
            return out

    reset_auto_parallel_context()
    set_auto_parallel_context(device_num=8, global_rank=0, strategy_ckpt_load_file="./strategy_stage1_auto.json")
    strategy1 = ((2, 2), (2, 2))
    strategy3 = ((2, 2), (2, 2))
    strategy4 = ((2, 2), (2, 2))
    strategy5 = ((2, 2), (2, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy3, strategy4, strategy5)))
    context.set_auto_parallel_context(parallel_mode="auto_parallel", dataset_strategy="full_batch")
    x1 = Tensor(np.ones([32, 32]), dtype=ms.float32)
    x6 = Tensor(np.ones([128, 32]), dtype=ms.float32)
    x7 = Tensor(np.ones([32, 32]), dtype=ms.float32)
    net.set_train()
    _cell_graph_executor.compile(net, x1, x6, x7)
