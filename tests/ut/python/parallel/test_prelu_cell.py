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
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim.momentum import Momentum
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.nn import PReLU
from tests.dataset_mock import MindData


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


context.set_context(mode=context.GRAPH_MODE)


class Dataset(MindData):
    def __init__(self, predict, label, length=3, input_num=2):
        super(Dataset, self).__init__(size=length)
        self.predict = predict
        self.label = label
        self.index = 0
        self.length = length
        self.input_num = input_num

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        if self.input_num == 2:
            return (self.predict, self.label)
        return (self.predict,)

    def reset(self):
        self.index = 0


class PReLUNet(nn.Cell):
    def __init__(self):
        super(PReLUNet, self).__init__()
        self.prelu = PReLU(channel=256)

    def construct(self, x):
        x = self.prelu(x)
        return x


def reshape_common(parallel_mode):
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=8, dataset_strategy="data_parallel")
    predict = Tensor(np.ones([32, 256]), dtype=ms.float32)
    label = Tensor(np.ones([32]), dtype=ms.int32)
    dataset = Dataset(predict, label, 2)
    net = PReLUNet()

    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, loss, opt)
    model.train(epoch_size, dataset, dataset_sink_mode=False)


def test_prelu_cell():
    """
    Feature: distribute operator prelu in auto parallel.
    Description: prelu net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    reshape_common(ParallelMode.SEMI_AUTO_PARALLEL)
