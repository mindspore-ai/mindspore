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

from mindspore.train import Model, ParallelMode
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim.momentum import Momentum
from mindspore import Tensor
import mindspore as ms
import numpy as np
from mindspore.ops import operations as P
import mindspore.nn as nn
from mindspore.common.parameter import Parameter
from tests.dataset_mock import MindData
from mindspore import context
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer

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
            return self.predict, self.label
        else:
            return self.predict,

    def reset(self):
        self.index = 0


class PReLU(nn.Cell):
    def __init__(self, channel=1, w=0.25):
        super(PReLU, self).__init__()
        if isinstance(w, (np.float32, float)):
            tmp = np.empty((channel,), dtype=np.float32)
            tmp.fill(w)
            w = Tensor(tmp)
        elif isinstance(w, list):
            w = Tensor(w)

        if not isinstance(w, Tensor):
            raise TypeError("w only support np.float32, float or Tensor type.")

        self.w = Parameter(initializer(w, [channel, ]), name='a')
        self.prelu = P.PReLU()
        self.relu = P.ReLU().set_strategy(((1,),))
        self.sub = P.Sub().set_strategy(((1,), (1,)))
        self.assign_sub = P.AssignSub().set_strategy(((1,), (1,)))

    def construct(self, x):
        u = self.relu(self.w)
        tmp = self.sub(self.w, u)
        x = F.depend(x, self.assign_sub(self.w, tmp))
        v = self.prelu(x, u)
        return v


class PReLUNet(nn.Cell):
    def __init__(self):
        super(PReLUNet, self).__init__()
        self.prelu = PReLU(channel=256)

    def construct(self, x):
        x = self.prelu(x)
        return x


def prelu_net():
    return PReLUNet()


def reshape_common(parallel_mode):
    batch_size = 32
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=8)
    predict = Tensor(np.ones([32, 256]), dtype=ms.float32)
    label = Tensor(np.ones([32]), dtype=ms.int32)
    dataset = Dataset(predict, label, 2)
    net = prelu_net()

    loss = SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, loss, opt)
    model.train(epoch_size, dataset, dataset_sink_mode=False)


def test_prelu_cell():
    reshape_common(ParallelMode.SEMI_AUTO_PARALLEL)
