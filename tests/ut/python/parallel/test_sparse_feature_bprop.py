# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test sparse feature bprop """
import pytest
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import composite as C, operations as P
from mindspore.ops.operations.comm_ops import AllReduce
from mindspore.common.api import _executor
from mindspore.nn import TrainOneStepCell, Adam


grad_all = C.GradOperation(get_all=True)


@pytest.fixture(name="test_context")
def _test_context():
    context.set_context(enable_sparse=True)
    yield
    context.set_context(enable_sparse=False)
    context.reset_auto_parallel_context()


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x):
        return grad_all(self.network)(x)

def test_bprop_with_sparse_feature_allreduce(test_context):
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="hybrid_parallel")

    class Net(nn.Cell):
        def __init__(self, axis=0, shape=None):
            super(Net, self).__init__()
            if shape is None:
                shape = [8, 8]
            self.all_reduce = AllReduce()
            self.gatherv2 = P.SparseGatherV2()
            self.index = Tensor(np.ones(shape), dtype=ms.int32)
            self.axis = axis

        def construct(self, x):
            out = self.all_reduce(x)
            out = self.gatherv2(out, self.index, self.axis)

            return out

    net = GradWrap(Net())
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)

    net.set_train()
    _executor.compile(net, x)


def test_bprop_with_sparse_feature_mirror(test_context):
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")

    class Net(nn.Cell):
        def __init__(self, shape=None):
            super(Net, self).__init__()
            if shape is None:
                shape = [8, 8]
            self.index = Tensor(np.ones(shape), dtype=ms.int32)
            self.embeddinglookup = nn.EmbeddingLookup(64, 64, param_init='ones')
            self.embeddinglookup.embeddinglookup.shard(((1, 1), (8, 1)))

        def construct(self, x, b):
            out = self.embeddinglookup(self.index)

            return out

    _x = Tensor(np.ones([126, 64, 32]), dtype=ms.float32)
    _b = Tensor(np.ones([126, 64, 32]), dtype=ms.float32)

    def compile_net(net):
        optimizer = Adam(net.trainable_params(), learning_rate=0.1, loss_scale=1024.0, weight_decay=0.9)
        train_net = TrainOneStepCell(net, optimizer)
        train_net.set_train()
        _executor.compile(train_net, _x, _b)

    net = Net()
    compile_net(net)


def test_bprop_with_sparse_feature_dataparallel(test_context):
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="data_parallel")

    class Net(nn.Cell):
        def __init__(self, axis=0, shape=None):
            super(Net, self).__init__()
            if shape is None:
                shape = [8, 8]
            weight = Tensor(np.ones([64, 64]), dtype=ms.float32)
            self.weight = Parameter(weight, "w")
            self.index = Tensor(np.ones(shape), dtype=ms.int32)
            self.axis = axis
            self.gatherv2 = P.SparseGatherV2()

        def construct(self, x, b):
            out = self.gatherv2(self.weight, self.index, self.axis)

            return out

    _x = Tensor(np.ones([126, 64, 32]), dtype=ms.float32)
    _b = Tensor(np.ones([126, 64, 32]), dtype=ms.float32)

    def compile_net(net):
        optimizer = Adam(net.trainable_params(), learning_rate=0.1, loss_scale=1024.0, weight_decay=0.9)
        train_net = TrainOneStepCell(net, optimizer)
        train_net.set_train()
        _executor.compile(train_net, _x, _b)

    net = Net()
    compile_net(net)
