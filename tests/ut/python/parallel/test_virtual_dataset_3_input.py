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
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _executor
from mindspore.nn.wrap.cell_wrapper import VirtualDatasetCellTriple
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


# model_parallel test
def test_virtual_dataset_3_input():
    class Net(nn.Cell):
        def __init__(self, strategy0, strategy1, strategy2, strategy3):
            super().__init__()
            self.virtual_dataset = _VirtualDataset().shard(strategy0)
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)
            self.gelu = P.GeLU().shard(strategy3)

        def construct(self, x, y, b):
            x, y, b = self.virtual_dataset(x, y, b)
            out = self.gelu(self.matmul1(x, y))
            out = self.matmul2(out, b)
            return out

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy0 = ((2, 1), (2, 1), (2, 1))
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((2, 2), (2, 2))
    strategy3 = ((2, 4),)
    net = GradWrap(NetWithLoss(Net(strategy0, strategy1, strategy2, strategy3)))
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 2048]), dtype=ms.float32)
    net.set_auto_parallel()
    net.set_train()
    _executor.compile(net, x, y, b)


def test_virtualdataset_cell_3_inputs():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)
            self.gelu = P.GeLU().shard(strategy3)

        def construct(self, x, y, b):
            out = self.gelu(self.matmul1(x, y))
            out = self.matmul2(out, b)
            return out

    context.set_context(save_graphs=False)
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = GradWrap(VirtualDatasetCellTriple(NetWithLoss(Net(None, None, None))))
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 2048]), dtype=ms.float32)
    net.set_auto_parallel()
    net.set_train()
    _executor.compile(net, x, y, b)


if __name__ == '__main__':
    test_virtual_dataset_3_input()
    context.reset_auto_parallel_context()
