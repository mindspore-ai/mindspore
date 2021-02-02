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
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.nn.optim import Lamb
from mindspore.nn.optim import Momentum, Adam
from mindspore.nn.wrap.cell_wrapper import WithLossCell
from mindspore.nn.wrap.loss_scale import TrainOneStepWithLossScaleCell
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.train import Model
from mindspore.train.loss_scale_manager import FixedLossScaleManager, DynamicLossScaleManager
from ....dataset_mock import MindData

context.set_context(mode=context.GRAPH_MODE)


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


class Net(nn.Cell):
    def __init__(self, in_features, out_features):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.ones([out_features, in_features]).astype(np.float32)), name="weight")
        self.bias = Parameter(Tensor(np.ones([out_features]).astype(np.float32)), name="bias")
        self.matmul = P.MatMul()
        self.add = P.Add()

    def construct(self, input_):
        output = self.add(self.matmul(input_, self.weight), self.bias)
        return output


class NetFP16(nn.Cell):
    def __init__(self, in_features, out_features):
        super(NetFP16, self).__init__()
        self.weight = Parameter(Tensor(np.ones([out_features, in_features]).astype(np.float32)), name="weight")
        self.bias = Parameter(Tensor(np.ones([out_features]).astype(np.float32)), name="bias")
        self.matmul = P.MatMul()
        self.add = P.Add()
        self.cast = P.Cast()

    def construct(self, input_):
        output = self.cast(
            self.add(self.matmul(self.cast(input_, mstype.float16), self.cast(self.weight, mstype.float16)),
                     self.cast(self.bias, mstype.float16)), mstype.float32)
        return output


def get_axis(x):
    shape_op = P.Shape()
    shape = shape_op(x)
    length = F.tuple_len(shape)
    perm = F.make_range(0, length)
    return perm


class MSELoss(nn.Cell):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.reduce_sum = P.ReduceSum()
        self.square = P.Square()
        self.reduce_mean = P.ReduceMean()

    def construct(self, data, label):
        diff = data - label
        return self.reduce_mean(self.square(diff), get_axis(diff))


def test_momentum_compile():
    inputs = Tensor(np.ones([15, 1]).astype(np.float32))
    label = Tensor(np.zeros([15, 1]).astype(np.float32))
    net = Net(1, 1)

    loss = MSELoss()
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepWithLossScaleCell(net_with_loss, optimizer,
                                                  scale_sense=Tensor(np.full((1), 1.0), dtype=mstype.float32))
    train_network.set_train()
    output = train_network(inputs, label)
    print("the result is ", output)


def test_compile_fp16_not_overflow():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    net = NetFP16(16, 16)

    loss = MSELoss()
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepWithLossScaleCell(net_with_loss, optimizer,
                                                  scale_sense=Tensor(np.full((1), 1.0), dtype=mstype.float32))
    train_network.set_train()
    output = train_network(inputs, label)
    print("the result is ", output)


def test_compile_fp16_lr_overflow():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    lr = Tensor(np.ones([1], np.float32) * 0.1)
    net = NetFP16(16, 16)
    loss = MSELoss()
    optimizer = Momentum(net.trainable_params(), learning_rate=lr, momentum=0.9)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepWithLossScaleCell(net_with_loss, optimizer,
                                                  scale_sense=Tensor(np.full((1), np.finfo(np.float32).max),
                                                                     dtype=mstype.float32))
    train_network.set_train()
    output = train_network(inputs, label)
    print("the result is ", output)


def test_compile_fp16_overflow():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    net = NetFP16(16, 16)

    loss = MSELoss()
    optimizer = Lamb(net.trainable_params(), learning_rate=0.01)
    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepWithLossScaleCell(net_with_loss, optimizer,
                                                  scale_sense=Tensor(np.full((1), np.finfo(np.float32).max),
                                                                     dtype=mstype.float32))
    train_network.set_train()
    output = train_network(inputs, label)
    print("the result is ", output)


def test_compile_fp16_lr_overflow_with_lossscale_update():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    lr = Tensor(np.ones([1], np.float32) * 0.1)
    net = NetFP16(16, 16)
    loss = MSELoss()
    optimizer = Momentum(net.trainable_params(), learning_rate=lr, momentum=0.9)

    net_with_loss = WithLossCell(net, loss)
    scale_manager = DynamicLossScaleManager()
    manager = scale_manager.get_update_cell()
    train_network = TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=manager)
    train_network.set_train()
    output = train_network(inputs, label)
    print("the result is ", output)


def test_compile_f16_model_train():
    dataset_types = (np.float32, np.float32)
    dataset_shapes = ((16, 16), (16, 16))

    dataset = MindDataSet(dataset_types, dataset_shapes)
    net = NetFP16(16, 16)
    net.set_train()

    loss = MSELoss()
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    model = Model(net, loss_fn=loss, optimizer=optimizer, metrics=None)
    model.train(2, dataset, dataset_sink_mode=False)


def test_compile_f16_model_train_fixed():
    dataset_types = (np.float32, np.float32)
    dataset_shapes = ((16, 16), (16, 16))

    dataset = MindDataSet(dataset_types, dataset_shapes)
    net = NetFP16(16, 16)
    net.set_train()
    scale_manager = FixedLossScaleManager()
    loss = MSELoss()
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    model = Model(net, loss_fn=loss, optimizer=optimizer, metrics=None, loss_scale_manager=scale_manager)
    model.train(2, dataset)


def test_compile_fp16_lr_overflow_fixed_feed():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    lr = Tensor(np.ones([1], np.float32) * 0.1)
    net = NetFP16(16, 16)
    loss = MSELoss()
    optimizer = Momentum(net.trainable_params(), learning_rate=lr, momentum=0.9)

    net_with_loss = WithLossCell(net, loss)
    scale_manager = FixedLossScaleManager()
    update_cell = scale_manager.get_update_cell()
    train_network = TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=update_cell)
    train_network.set_train()
    output = train_network(inputs, label)
    print("the result is ", output)


def test_compile_fp16_lr_overflow_dynamic_feed():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    lr = Tensor(np.ones([1], np.float32) * 0.1)
    net = NetFP16(16, 16)
    loss = MSELoss()
    optimizer = Momentum(net.trainable_params(), learning_rate=lr, momentum=0.9)

    net_with_loss = WithLossCell(net, loss)
    scale_manager = DynamicLossScaleManager()
    update_cell = scale_manager.get_update_cell()
    train_network = TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=update_cell)
    train_network.set_train()
    output = train_network(inputs, label)
    print("the result is ", output)


def test_compile_fp16_lr_overflow_fixed_graph():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    lr = Tensor(np.ones([1], np.float32) * 0.1)
    net = NetFP16(16, 16)
    loss = MSELoss()
    optimizer = Momentum(net.trainable_params(), learning_rate=lr, momentum=0.9)

    net_with_loss = WithLossCell(net, loss)
    scale_manager = FixedLossScaleManager(drop_overflow_update=True)
    update_cell = scale_manager.get_update_cell()
    train_network = TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=update_cell)
    train_network.set_train()
    output = train_network(inputs, label)
    print("the result is ", output)


def test_compile_fp16_lr_overflow_dynamic_graph():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    lr = Tensor(np.ones([1], np.float32) * 0.1)
    net = NetFP16(16, 16)
    loss = MSELoss()
    optimizer = Momentum(net.trainable_params(), learning_rate=lr, momentum=0.9)

    net_with_loss = WithLossCell(net, loss)
    scale_manager = DynamicLossScaleManager()
    update_cell = scale_manager.get_update_cell()
    train_network = TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=update_cell)
    train_network.set_train()
    output = train_network(inputs, label)
    print("the result is ", output)


def adam_compile(loss_scale=1.0):
    inputs = Tensor(np.ones([15, 1]).astype(np.float32))
    label = Tensor(np.zeros([15, 1]).astype(np.float32))
    net = Net(1, 1)

    loss = MSELoss()
    optimizer = Adam(net.trainable_params(), learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, use_locking=False,
                     use_nesterov=False, weight_decay=0.0, loss_scale=loss_scale)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepWithLossScaleCell(net_with_loss, optimizer,
                                                  scale_sense=Tensor(np.full((1), 1.0), dtype=mstype.float32))
    train_network.set_train()
    output = train_network(inputs, label)
    print("the result is ", output)


def test_adam_compile():
    adam_compile()


def test_adam_loss_scale_compile():
    """ test setting loss_scale to 1e-40 """
    adam_compile(loss_scale=1e-40)
