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
import glob
import os
import shutil
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.common.parameter import Parameter
from mindspore.graph.api import _cell_graph_executor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.parallel.shard import Layout
from tests.dataset_mock import MindData
from tests.ut.python.ops.test_math_ops import VirtualLoss


class Dataset(MindData):
    def __init__(self, predict, label, length=3):
        super(Dataset, self).__init__(size=length)
        self.predict = predict
        self.label = label
        self.index = 0
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.predict, self.label

    def reset(self):
        self.index = 0


class TransposeNet(nn.Cell):
    def __init__(self, strategy1, strategy2):
        super(TransposeNet, self).__init__()
        self.matmul = P.MatMul().shard(((8, 1), (1, 1)))
        self.matmul_weight = Parameter(Tensor(np.ones([128, 256]), dtype=ms.float32), name="weight")
        self.transpose1 = P.Transpose().shard(strategy1)
        self.transpose2 = P.Transpose().shard(strategy2)

    def construct(self, x):
        x = self.matmul(x, self.matmul_weight)
        x = self.transpose1(x, (1, 0))
        x = self.transpose2(x, (1, 0))
        return x


def transpose_net(strategy1, strategy2):
    return TransposeNet(strategy1=strategy1, strategy2=strategy2)


def transpose_common(strategy1, strategy2):
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, device_num=8,
                                      parameter_broadcast=False)

    predict = Tensor(np.ones([32, 128]), dtype=ms.float32)
    label = Tensor(np.ones([32]), dtype=ms.int32)
    dataset = Dataset(predict, label, 2)
    net = transpose_net(strategy1, strategy2)

    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss.softmax_cross_entropy.shard(((8, 1), (8, 1)))
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    context.set_context(mode=context.GRAPH_MODE)
    model = Model(net, loss, opt)

    model.train(epoch_size, dataset, dataset_sink_mode=False)


def test_transpose1():
    """
    Feature: distribute operator transpose in auto parallel.
    Description: run transpose distribute operator using model.
    Expectation: compile done without error.
    """
    strategy1 = ((1, 8),)
    strategy2 = ((1, 8),)
    transpose_common(strategy1, strategy2)


def test_transpose2():
    """
    Feature: distribute operator transpose in auto parallel.
    Description: run transpose distribute operator using model.
    Expectation: compile done without error.
    """
    strategy1 = ((1, 4),)
    strategy2 = ((1, 8),)
    transpose_common(strategy1, strategy2)

grad_all = C.GradOperation(get_all=True)

class NetWithLoss1(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss1, self).__init__()
        self.loss = VirtualLoss()
        self.network = network
    def construct(self, x):
        predict = self.network(x)
        return self.loss(predict)

class GradWrap1(nn.Cell):
    def __init__(self, network):
        super(GradWrap1, self).__init__()
        self.network = network

    def construct(self, x):
        return grad_all(self.network)(x)


class TestTransposeAlltoAll:
    def setup_method(self):
        self.output_path = './graphs' + self.__str__()
        context.set_context(save_graphs=True, save_graphs_path=self.output_path)

    def teardown_method(self):
        shutil.rmtree(self.output_path)

    def check_config(self, param, pattern, truth_count):
        ir_files = glob.glob(os.path.join(self.output_path, '*_validate*.ir'))
        assert len(ir_files) == 1
        appear_count = 0
        with open(ir_files[0], 'r') as fp:
            for line in fp:
                if param in line and pattern in line:
                    appear_count += 1
        assert appear_count == truth_count
    def test_test34(self):
        """
        Feature: inferring redis op list meeting transpose op
        Description: run transpose distribute operator and inferring alltoall
        Expectation: compile done without error.
        """
        class Net(nn.Cell):
            def __init__(self):
                super().__init__()
                src = Layout((8, 4), ("ep", "train_mp"))
                self.id1 = P.Identity().shard(in_strategy=(src("ep", "None", "None"),))
                self.trans = P.Transpose().shard(in_strategy=(src("ep", "None", "None"),))
                dst = Layout((32,), ("infer_mp",))
                self.id2 = P.Identity().shard(in_strategy=(dst("None", "None", "infer_mp"),))
            def construct(self, x):
                out = self.id1(x)
                out = self.trans(out, (0, 2, 1))
                out = self.id2(out)
                return out
        context.set_auto_parallel_context(device_num=32, global_rank=0,
                                          parallel_mode="semi_auto_parallel",
                                          enable_alltoall=True,
                                          full_batch=True)
        net = GradWrap1(NetWithLoss1(Net()))

        x = Tensor(np.ones([32, 32, 32]), dtype=ms.float32)
        net.set_train()
        _cell_graph_executor.compile(net, x)

        a2a = "AlltoAll"
        rank_list = "rank_list: (0, 4, 8, 12, 16, 20, 24, 28)"
        self.check_config(a2a, rank_list, 2)

    def test_test35(self):
        """
        Feature: inferring redis op list meeting transpose op
        Description: run transpose distribute operator and inferring alltoall
        Expectation: compile done without error.
        """
        class Net(nn.Cell):
            def __init__(self):
                super().__init__()
                src = Layout((8, 4), ("ep", "train_mp"))
                self.id1 = P.Identity().shard(in_strategy=(src("train_mp", "None", "None"),))
                dst = Layout((32,), ("infer_mp",))
                self.id2 = P.Identity().shard(in_strategy=(dst("None", "None", "infer_mp"),))
            def construct(self, x):
                out = self.id1(x)
                out = self.id2(out)
                return out

        context.set_auto_parallel_context(device_num=32, global_rank=0,
                                          parallel_mode="semi_auto_parallel",
                                          enable_alltoall=True,
                                          full_batch=True)
        net = GradWrap1(NetWithLoss1(Net()))

        x = Tensor(np.ones([32, 32, 32]), dtype=ms.float32)
        net.set_train()
        _cell_graph_executor.compile(net, x)

        a2a = "AlltoAll"
        rank_list = "rank_list: (0, 1, 2, 3)"
        self.check_config(a2a, rank_list, 2)

if __name__ == '__main__':
    test_transpose1()
    test_transpose2()
