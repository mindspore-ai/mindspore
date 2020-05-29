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

""" test Communicate """
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import _executor
from mindspore.communication._comm_helper import Backend
from mindspore.communication.management import HCCL_WORLD_COMM_GROUP, NCCL_WORLD_COMM_GROUP, GlobalComm, init
from mindspore.nn import Dense
from mindspore.nn import Momentum
from mindspore.nn import ReLU
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.ops.operations.comm_ops import AllReduce, AllGather, _AlltoAll, ReduceOp, ReduceScatter
from mindspore.ops.operations.comm_ops import HostAllGather, HostReduceScatter
from mindspore.ops.operations.comm_ops import Broadcast

# pylint: disable=W0212
# W0212: protected-access

tag = 0

init("hccl")


class AllReduceNet(nn.Cell):
    """AllReduceNet definition"""

    def __init__(self, input_channel, out_channel, op):
        super(AllReduceNet, self).__init__()
        self.dense = Dense(input_channel, out_channel)
        self.reduce = AllReduce(op)
        self.relu = ReLU()

    def construct(self, x):
        x = self.dense(x)
        x = self.reduce(x)
        return self.relu(x)


class BroadCastNet(nn.Cell):
    """BroadCastNet definition"""

    def __init__(self, input_channel, out_channel):
        super(BroadCastNet, self).__init__()
        self.dense = Dense(input_channel, out_channel)
        self.broadcast = Broadcast(0)

    def construct(self, x):
        x, = self.broadcast((x,))
        x = self.dense(x)
        return x


class AllGatherNet(nn.Cell):
    """AllGatherNet definition"""

    def __init__(self, input_channel, out_channel):
        super(AllGatherNet, self).__init__()
        self.dense = Dense(input_channel, out_channel)
        if GlobalComm.BACKEND is Backend.HCCL:
            self.allgather = AllGather(group=HCCL_WORLD_COMM_GROUP)
        elif GlobalComm.BACKEND is Backend.NCCL:
            self.allgather = AllGather(group=NCCL_WORLD_COMM_GROUP)
        else:
            self.allgather = AllGather()

        self.relu = ReLU()

    def construct(self, x):
        x = self.dense(x)
        x = self.allgather(x)
        return self.relu(x)


class HostAllGatherNet(nn.Cell):
    """HostAllGatherNet definition"""

    def __init__(self, input_channel, output_channel):
        super(HostAllGatherNet, self).__init__()
        self.dense = Dense(input_channel, output_channel)
        self.hostallgather = HostAllGather((0, 1))
        self.relu = ReLU()

    def construct(self, x):
        x = self.dense(x)
        x = self.hostallgather(x)
        return self.relu(x)


class ReduceScatterNet(nn.Cell):
    """ReduceScatterNet definition"""

    def __init__(self, input_channel, out_channel, op):
        super(ReduceScatterNet, self).__init__()
        self.dense = Dense(input_channel, out_channel)
        self.reducescatter = ReduceScatter(op)
        self.relu = ReLU()

    def construct(self, x):
        x = self.dense(x)
        x = self.reducescatter(x)
        return self.relu(x)


class HostReduceScatterNet(nn.Cell):
    """HostReduceScatterNet definition"""

    def __init__(self, input_channel, out_channel, op):
        super(HostReduceScatterNet, self).__init__()
        self.dense = Dense(input_channel, out_channel)
        self.hostreducescatter = HostReduceScatter(op, (0, 1))
        self.relu = ReLU()

    def construct(self, x):
        x = self.dense(x)
        x = self.hostreducescatter(x)
        return self.relu(x)


class AlltoAllNet(nn.Cell):
    """AlltoAllNet definition"""

    def __init__(self, input_channel, out_channel):
        super(AlltoAllNet, self).__init__()
        self.dense = Dense(input_channel, out_channel)
        self.alltoall = _AlltoAll(1, 0, 1)
        self.relu = ReLU()

    def construct(self, x):
        x = self.dense(x)
        x = self.alltoall(x)
        return self.relu(x)


def run_allreduce(op):
    """run_allreduce"""
    context.set_context(mode=context.GRAPH_MODE)
    input_tensor = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np.float32))
    label_tensor = Tensor(np.array([[1.2], [2.2]], dtype=np.float32))
    network = AllReduceNet(2, 1, op)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Momentum(filter(lambda x: x.requires_grad, network.get_parameters()),
                         learning_rate=0.1,
                         momentum=0.9)
    network = WithLossCell(network, loss_fn)
    network = TrainOneStepCell(network, optimizer)
    _executor.compile(network, input_tensor, label_tensor)


def test_allreduce():
    """test_allreduce"""
    context.set_context(mode=context.GRAPH_MODE)
    run_allreduce(ReduceOp.SUM)
    run_allreduce(ReduceOp.MAX)
    run_allreduce(ReduceOp.MIN)


def test_allgather():
    """test_allgather"""
    context.set_context(mode=context.GRAPH_MODE)
    input_tensor = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np.float32))
    label_tensor = Tensor(np.array([[1.2], [2.2]], dtype=np.float32))
    network = AllGatherNet(2, 1)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Momentum(filter(lambda x: x.requires_grad, network.get_parameters()),
                         learning_rate=0.1,
                         momentum=0.9)
    network = WithLossCell(network, loss_fn)
    network = TrainOneStepCell(network, optimizer)
    _executor.compile(network, input_tensor, label_tensor)


def test_hostallgather():
    """test_hostallgather"""
    context.set_context(mode=context.GRAPH_MODE)
    input_tensor = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np.float32))
    label_tensor = Tensor(np.array([[1.2], [2.2], [3.2], [4.2]], dtype=np.float32))
    network = HostAllGatherNet(2, 1)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Momentum(filter(lambda x: x.requires_grad, network.get_parameters()),
                         learning_rate=0.1,
                         momentum=0.9)
    network = WithLossCell(network, loss_fn)
    network = TrainOneStepCell(network, optimizer)
    _executor.compile(network, input_tensor, label_tensor)


def run_reducescatter(op):
    """run_reducescatter"""
    context.set_context(mode=context.GRAPH_MODE)
    input_tensor = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np.float32))
    label_tensor = Tensor(np.array([[1.2], [2.2]], dtype=np.float32))
    network = ReduceScatterNet(2, 1, op)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Momentum(filter(lambda x: x.requires_grad, network.get_parameters()),
                         learning_rate=0.1,
                         momentum=0.9)
    network = WithLossCell(network, loss_fn)
    network = TrainOneStepCell(network, optimizer)
    _executor.compile(network, input_tensor, label_tensor)


def test_reducescatter():
    """test_reducescatter"""
    context.set_context(mode=context.GRAPH_MODE)
    run_reducescatter(ReduceOp.SUM)


def test_hostreducescatter():
    """test_hostreducescatter"""
    context.set_context(mode=context.GRAPH_MODE)
    input_tensor = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np.float32))
    label_tensor = Tensor(np.array([[1.2]], dtype=np.float32))
    network = HostReduceScatterNet(2, 1, ReduceOp.SUM)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Momentum(filter(lambda x: x.requires_grad, network.get_parameters()),
                         learning_rate=0.1,
                         momentum=0.9)
    network = WithLossCell(network, loss_fn)
    network = TrainOneStepCell(network, optimizer)
    _executor.compile(network, input_tensor, label_tensor)


def test_broadcast():
    """test_broadcast"""
    context.set_context(mode=context.GRAPH_MODE)
    input_tensor_1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np.float32))
    label_tensor = Tensor(np.array([[1.2], [2.2]], dtype=np.float32))
    network = BroadCastNet(2, 1)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Momentum(filter(lambda x: x.requires_grad, network.get_parameters()),
                         learning_rate=0.1,
                         momentum=0.9)
    network = WithLossCell(network, loss_fn)
    network = TrainOneStepCell(network, optimizer)
    _executor.compile(network, input_tensor_1, label_tensor)


def test_alltoall():
    """test_alltoall"""
    context.set_context(mode=context.GRAPH_MODE)
    input_tensor = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np.float32))
    label_tensor = Tensor(np.array([[1.2], [2.2]], dtype=np.float32))
    network = AlltoAllNet(2, 1)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Momentum(filter(lambda x: x.requires_grad, network.get_parameters()),
                         learning_rate=0.1,
                         momentum=0.9)
    network = WithLossCell(network, loss_fn)
    network = TrainOneStepCell(network, optimizer)
    _executor.compile(network, input_tensor, label_tensor)
