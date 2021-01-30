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
# ============================================================================
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.common.parameter import ParameterTuple
from mindspore.communication.management import init
from mindspore.nn import Dense, Cell
from mindspore.nn.loss.loss import _Loss
from mindspore.nn.optim import Momentum
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.train import Model
from mindspore.context import ParallelMode

context.set_context(mode=context.GRAPH_MODE)
device_number = 32
batch_size_per_device = 128


class Dataset():
    def __init__(self, predict, length=3):
        self.predict = predict
        self.index = 0
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return (self.predict,)

    def reset(self):
        self.index = 0

    def get_dataset_size(self):
        return 128

    def get_repeat_count(self):
        return 1

    def create_tuple_iterator(self, num_epochs=-1, do_copy=True):
        return self


class GatherV2(_Loss):
    def __init__(self, index_dim, strategy, index_size=16):
        super(GatherV2, self).__init__()
        self.pow = P.Pow()
        emb1_list = 21
        emb2_list = 2
        if index_dim == 1:
            emb_list = list(range(index_size))
            emb1_list = emb_list[0::2]
            emb2_list = emb_list[1::2]
        if index_dim == 2:
            emb_list = np.arange(index_size * 16)
            emb1_list = np.reshape(emb_list[0::2], (int(index_size / 2), 16))
            emb2_list = np.reshape(emb_list[1::2], (int(index_size / 2), 16))
        self.emb1_param = Tensor(emb1_list, dtype=mstype.int32)
        self.emb2_param = Tensor(emb2_list, dtype=mstype.int32)
        self.gatherv2 = P.Gather().shard(strategy).add_prim_attr("data_parallel", True)

    def construct(self, nembeddings):
        emb1 = self.gatherv2(nembeddings, self.emb1_param, 0)
        emb2 = self.gatherv2(nembeddings, self.emb2_param, 0)
        return self.pow((emb1 - emb2), 2.0)


def fc_with_initialize(input_channels, out_channels):
    return Dense(input_channels, out_channels)


class BuildTrainNetwork(nn.Cell):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data):
        embeddings = self.network(input_data)
        loss = self.criterion(embeddings)
        return loss


class TrainOneStepCell(Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.sens = sens

    def construct(self, data):
        weights = self.weights
        loss = self.network(data)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(data, sens)

        return F.depend(loss, self.optimizer(grads))


def net_trains(criterion, rank):
    init()
    lr = 0.1
    momentum = 0.9
    max_epoch = 20
    input_channels = 256
    out_channels = 512
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, device_num=device_number,
                                      global_rank=rank)
    predict = Tensor(np.ones([batch_size_per_device, input_channels]), dtype=ms.float32)
    dataset = Dataset(predict, 4)

    network = fc_with_initialize(input_channels, out_channels)
    network.set_train()

    train_network = BuildTrainNetwork(network, criterion)
    train_network.set_train()
    opt = Momentum(train_network.trainable_params(), lr, momentum)
    train_net = TrainOneStepCell(train_network, opt).set_train()

    model = Model(train_net)
    model.train(max_epoch, dataset, dataset_sink_mode=False)
    context.reset_auto_parallel_context()


def test_auto_batch_parallel():
    gather_v2_strategy = None
    criterion = GatherV2(1, strategy=gather_v2_strategy, index_size=batch_size_per_device * device_number)
    rank = 2
    net_trains(criterion, rank)


def test_2d_index_auto_batch_parallel():
    gather_v2_strategy = None
    criterion = GatherV2(2, strategy=gather_v2_strategy, index_size=batch_size_per_device * device_number)
    rank = 2
    net_trains(criterion, rank)


def test_batch_parallel():
    gather_v2_strategy = ((device_number, 1),)
    criterion = GatherV2(1, strategy=gather_v2_strategy, index_size=batch_size_per_device * device_number)
    rank = 2
    net_trains(criterion, rank)


def test_strategy1():
    gather_v2_strategy = ((16, 2),)
    rank = 2
    criterion = GatherV2(1, strategy=gather_v2_strategy, index_size=batch_size_per_device * device_number)
    net_trains(criterion, rank)


def test_strategy2():
    gather_v2_strategy = ((1, device_number),)
    rank = 2
    criterion = GatherV2(1, strategy=gather_v2_strategy, index_size=batch_size_per_device * device_number)
    net_trains(criterion, rank)


def test_strategy3():
    gather_v2_strategy = ((8, 1),)
    rank = 2
    criterion = GatherV2(1, strategy=gather_v2_strategy, index_size=batch_size_per_device * device_number)
    net_trains(criterion, rank)


class GatherV2Axis1(_Loss):
    def __init__(self, index_dim, strategy, index_size=16):
        super(GatherV2Axis1, self).__init__()
        self.pow = P.Pow()
        emb1_list = 21
        emb2_list = 2
        if index_dim == 1:
            emb_list = list(range(index_size))
            emb1_list = emb_list[0::2]
            emb2_list = emb_list[1::2]
        if index_dim == 2:
            emb_list = np.arange(index_size * index_size)
            emb1_list = np.reshape(emb_list[0::2], (int(index_size / 2), index_size))
            emb2_list = np.reshape(emb_list[1::2], (int(index_size / 2), index_size))
        self.emb1_param = Tensor(emb1_list, dtype=mstype.int32)
        self.emb2_param = Tensor(emb2_list, dtype=mstype.int32)
        self.gatherv2 = P.Gather().shard(strategy)

    def construct(self, nembeddings):
        emb1 = self.gatherv2(nembeddings, self.emb1_param, 1)
        emb2 = self.gatherv2(nembeddings, self.emb2_param, 1)
        return self.pow((emb1 - emb2), 2.0)


def test_axis1_auto_batch_parallel():
    gather_v2_strategy = None
    criterion = GatherV2Axis1(1, strategy=gather_v2_strategy, index_size=512)
    rank = 2
    net_trains(criterion, rank)


def test_axis1_batch_parallel():
    gather_v2_strategy = ((device_number, 1), (1,))
    criterion = GatherV2Axis1(1, strategy=gather_v2_strategy, index_size=512)
    rank = 2
    net_trains(criterion, rank)


def test_axis1_strategy1():
    gather_v2_strategy = ((16, 2), (1,))
    rank = 17
    criterion = GatherV2Axis1(1, strategy=gather_v2_strategy, index_size=512)
    net_trains(criterion, rank)
