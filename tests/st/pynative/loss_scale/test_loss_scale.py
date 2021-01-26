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
""" test_loss_scale """
import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor, Parameter
from mindspore.nn.wrap.cell_wrapper import WithLossCell
from mindspore.nn.wrap.loss_scale import TrainOneStepWithLossScaleCell
from mindspore.ops import operations as P
from mindspore.nn.optim import Momentum, RMSProp
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from mindspore.train import Model
from mindspore.nn.optim import Lamb
from mindspore.train.loss_scale_manager import DynamicLossScaleManager

def setup_module():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

class MindData:
    """ Stub for MindData """

    def __init__(self, size=None, batch_size=None, repeat_count=1,
                 np_types=None, output_shapes=None, input_indexes=(), func_name=''):
        self._size = size
        self._batch_size = batch_size
        self._repeat_count = repeat_count
        self._np_types = np_types
        self._output_shapes = output_shapes
        self._input_indexes = input_indexes
        self._func_name = func_name
        self._iter_num = 0

    def get_dataset_size(self):
        return self._size

    def get_repeat_count(self):
        return self._repeat_count

    def get_batch_size(self):
        return self._batch_size

    def output_types(self):
        return self._np_types

    def output_shapes(self):
        return self._output_shapes

    def create_tuple_iterator(self, num_epochs=-1, do_copy=True):
        return self

    @property
    def input_indexes(self):
        return self._input_indexes

    @property
    def func_name(self):
        return self._func_name

    def send(self):
        pass

    def __len__(self):
        return self._size

    def __iter__(self):
        return self

    def __next__(self):
        if self._size < self._iter_num:
            raise StopIteration
        self._iter_num += 1
        next_value = []
        for shape, typ in zip(self._output_shapes, self._np_types):
            next_value.append(Tensor(np.ndarray(shape, typ)))

        return tuple(next_value)

    def next(self):
        return self.__next__()

    def reset(self):
        self._iter_num = 0


class MindDataSet(MindData):
    def __init__(self, dataset_types, dataset_shapes):
        super(MindDataSet, self).__init__(size=2, batch_size=32,
                                          np_types=dataset_types,
                                          output_shapes=dataset_shapes,
                                          input_indexes=(0, 1), func_name='')
    def __next__(self):
        if self._size < self._iter_num:
            raise StopIteration
        self._iter_num += 1
        res = []
        for shape, t in zip(self._output_shapes, self._np_types):
            res.append(Tensor(np.ones(shape).astype(t)))
        return tuple(res)

class NetFP16(nn.Cell):
    def __init__(self, in_features, out_features):
        super(NetFP16, self).__init__()
        self.weight = Parameter(Tensor(np.ones([out_features, in_features]).astype(np.float32)), name="weight")
        self.bias = Parameter(Tensor(np.ones([out_features]).astype(np.float32)), name="bias")
        self.matmul = P.MatMul()
        self.add = P.Add()
        self.cast = P.Cast()

    def construct(self, x):
        output = self.cast(self.add(self.matmul(self.cast(x, mstype.float16),
                                                self.cast(self.weight, mstype.float16)),
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
        self.sum = P.ReduceSum()
        self.square = P.Square()
        self.reduce_mean = P.ReduceMean()

    def construct(self, data, label):
        diff = data - label
        return self.reduce_mean(self.square(diff), get_axis(diff))

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_loss_scale_fp16_lr_overflow():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    lr = Tensor(np.ones([1], np.float32) * 0.1)
    net = NetFP16(16, 16)
    net.set_train()

    loss = MSELoss()
    optimizer = Momentum(net.trainable_params(), learning_rate=lr, momentum=0.9)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepWithLossScaleCell(net_with_loss, optimizer,
                                                  scale_sense=Tensor(np.full((1), np.finfo(np.float32).max),
                                                                     dtype=mstype.float32))
    output_1 = train_network(inputs, label)
    output_2 = train_network(inputs, label)
    assert output_1[0].asnumpy() == output_2[0].asnumpy()
    assert output_1[1].asnumpy() == output_2[1].asnumpy() == True

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_loss_scale_fp16_lr_overflow_set_sense_scale():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    lr = Tensor(np.ones([1], np.float32) * 0.1)
    net = NetFP16(16, 16)
    net.set_train()

    loss = MSELoss()
    optimizer = Momentum(net.trainable_params(), learning_rate=lr, momentum=0.9)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepWithLossScaleCell(net_with_loss, optimizer,
                                                  scale_sense=Tensor(np.full((1), np.finfo(np.float32).max),
                                                                     dtype=mstype.float32))
    output_1 = train_network(inputs, label)

    train_network.set_sense_scale(Tensor(np.full((1), np.finfo(np.float32).max), dtype=mstype.float32))
    output_2 = train_network(inputs, label)
    assert output_1[0].asnumpy() == output_2[0].asnumpy()
    assert output_1[1].asnumpy() == output_2[1].asnumpy() == True

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_loss_scale_fp16_model_train_overflow():
    dataset_types = (np.float32, np.float32)
    dataset_shapes = ((16, 16), (16, 16))
    dataset = MindDataSet(dataset_types, dataset_shapes)

    net = NetFP16(16, 16)
    net.set_train()

    loss = MSELoss()
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    scale_manager = DynamicLossScaleManager(init_loss_scale=16, scale_factor=2, scale_window=2)
    model = Model(net, loss_fn=loss, optimizer=optimizer, metrics=None, loss_scale_manager=scale_manager)
    model.train(2, dataset, dataset_sink_mode=False)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_loss_scale_fp16_opt_rmsprop_overflow():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    net = NetFP16(16, 16)
    net.set_train()

    loss = MSELoss()
    optimizer = RMSProp(net.trainable_params(), learning_rate=0.1)
    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepWithLossScaleCell(net_with_loss, optimizer,
                                                  scale_sense=Tensor(np.full(1, np.finfo(np.float32).max),
                                                                     dtype=mstype.float32))
    output_1 = train_network(inputs, label)
    output_2 = train_network(inputs, label)
    assert output_1[0].asnumpy() == output_2[0].asnumpy()
    assert output_1[1].asnumpy() == output_2[1].asnumpy() == True

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_loss_scale_fp16_overflow():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    net = NetFP16(16, 16)
    net.set_train()

    loss = MSELoss()
    optimizer = Lamb(net.trainable_params(), learning_rate=0.01)
    net_with_loss = WithLossCell(net, loss)
    net_with_loss.set_grad()
    train_network = TrainOneStepWithLossScaleCell(net_with_loss, optimizer,
                                                  scale_sense=Tensor(np.full((1), np.finfo(np.float32).max),
                                                                     dtype=mstype.float32))
    output_1 = train_network(inputs, label)
    output_2 = train_network(inputs, label)
    assert output_1[0].asnumpy() == output_2[0].asnumpy()
    assert output_1[1].asnumpy() == output_2[1].asnumpy() == True
