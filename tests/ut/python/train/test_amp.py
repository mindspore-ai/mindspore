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
""" auto mixed precision """
import numpy as np
import pytest

import mindspore.context as context
from mindspore import Tensor
from mindspore import amp
from mindspore import nn
from mindspore.communication.management import init
from mindspore.communication._comm_helper import GlobalComm
from mindspore.context import ParallelMode
from mindspore.train import Model
from ....dataset_mock import MindData


def setup_module(module):
    _ = module
    context.set_context(mode=context.GRAPH_MODE)


class Net(nn.Cell):
    def __init__(self, in_features, out_features):
        super(Net, self).__init__()
        self.dense = nn.Dense(in_features, out_features)
        self.loss = nn.MSELoss()

    def construct(self, input_x, label):
        output = self.dense(input_x)
        loss = self.loss(output, label)
        return loss


class NetNoLoss(nn.Cell):
    def __init__(self, in_features, out_features):
        super(NetNoLoss, self).__init__()
        self.dense = nn.Dense(in_features, out_features)

    def construct(self, input_x):
        return self.dense(input_x)


def test_amp_o0():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    net = Net(16, 16)

    optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_network = amp.build_train_network(net, optimizer, level="O0")
    _ = train_network(inputs, label)


def test_amp_o2():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    net = Net(16, 16)

    optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_network = amp.build_train_network(net, optimizer, level="O2")
    _ = train_network(inputs, label)


def test_amp_o2_loss():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    net = NetNoLoss(16, 16)
    loss = nn.MSELoss()
    optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_network = amp.build_train_network(net, optimizer, loss, level="O2")
    _ = train_network(inputs, label)


def test_amp_o0_loss():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    net = NetNoLoss(16, 16)
    loss = nn.MSELoss()
    optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_network = amp.build_train_network(net, optimizer, loss)
    _ = train_network(inputs, label)


class MindDataSet(MindData):
    def __init__(self, dataset_types, dataset_shapes):
        super(MindDataSet, self).__init__(size=2, batch_size=32,
                                          np_types=dataset_types,
                                          output_shapes=dataset_shapes,
                                          input_indexs=(0, 1))

    def __next__(self):
        if self._size < self._iter_num:
            raise StopIteration
        self._iter_num += 1
        lst = []
        for shape_, type_ in zip(self._output_shapes, self._np_types):
            lst.append(Tensor(np.ones(shape_).astype(type_)))
        return tuple(lst)


def test_compile_model_train_O0():
    dataset_types = (np.float32, np.float32)
    dataset_shapes = ((16, 16), (16, 16))

    dataset = MindDataSet(dataset_types, dataset_shapes)

    net = NetNoLoss(16, 16)
    loss = nn.MSELoss()
    optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)

    model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={"acc"}, amp_level="O0")
    model.train(2, dataset, dataset_sink_mode=False)
    with pytest.raises(ValueError):
        # not actual run, the metrics step will fail, check if compile ok.
        model.eval(dataset)


def test_compile_model_train_O2():
    dataset_types = (np.float32, np.float32)
    dataset_shapes = ((16, 16), (16, 16))

    dataset = MindDataSet(dataset_types, dataset_shapes)

    net = NetNoLoss(16, 16)
    loss = nn.MSELoss()
    optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)

    model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={"acc"}, amp_level="O2")
    model.train(2, dataset, dataset_sink_mode=False)
    with pytest.raises(ValueError):
        # not actual run, the metrics step will fail, check if compile ok.
        model.eval(dataset)


def compile_model_train_O2_parallel():
    dataset_types = (np.float32, np.float32)
    dataset_shapes = ((16, 16), (16, 16))
    context.set_context(device_target='Ascend')
    context.set_auto_parallel_context(
        global_rank=0, device_num=8,
        gradients_mean=True, parameter_broadcast=True,
        parallel_mode=ParallelMode.DATA_PARALLEL)

    dataset = MindDataSet(dataset_types, dataset_shapes)

    net = NetNoLoss(16, 16)
    loss = nn.MSELoss()
    optimizer = nn.Momentum(net.trainable_params(), 0.1, 0.9, 0.00004, 1024.0)
    GlobalComm.CHECK_ENVS = False
    init()
    GlobalComm.CHECK_ENVS = True
    model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={"acc"}, amp_level="O2")
    model.train(2, dataset, dataset_sink_mode=False)
