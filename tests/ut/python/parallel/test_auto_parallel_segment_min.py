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
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
import mindspore.ops as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, vectors, index):
        predict = self.network(vectors, index)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, vectors, index):
        return grad_all(self.network)(vectors, index)


def test_auto_parallel_unsortedsegmentmin():
    class Net(nn.Cell):
        def __init__(self, num_segments):
            super().__init__()
            self.merge_op = P.UnsortedSegmentMin()
            self.num_segments = num_segments

        def construct(self, vectors, index):
            out = self.merge_op(vectors, index, self.num_segments)
            return out

    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel")

    x = Tensor(np.random.rand(16, 16, 32, 64), dtype=ms.float32)
    indices = Tensor(np.random.randint(16, size=(16,)), ms.int32)

    net = GradWrap(NetWithLoss(Net(16)))
    net.set_train()
    _cell_graph_executor.compile(net, x, indices)
