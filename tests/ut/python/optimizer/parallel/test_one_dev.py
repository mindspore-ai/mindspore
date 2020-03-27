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

from mindspore.train import Model, ParallelMode
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim.momentum import Momentum
from mindspore import Tensor
import mindspore as ms
import numpy as np
from mindspore.ops import operations as P
import mindspore.nn as nn
from mindspore.common.parameter import Parameter
from .....dataset_mock import MindData
from mindspore import context
from mindspore.parallel._utils import _reset_op_id
from mindspore.common.api import _executor

context.set_context(mode=context.GRAPH_MODE)


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


class AllToAllNet(nn.Cell):
    def __init__(self):
        super(AllToAllNet, self).__init__()
        self.matmul = P.MatMul()
        self.matmul_weight = Parameter(Tensor(np.ones([128, 32]), dtype=ms.float32), name="weight")
        self.transpose1 = P.Transpose()

    def construct(self, x):
        x = self.matmul(x, self.matmul_weight)
        x = self.transpose1(x, (1, 0))
        return x


def all_to_all_net():
    return AllToAllNet()


def all_to_all_common():
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, device_num=1, global_rank=0)
    predict = Tensor(np.ones([32, 128]), dtype=ms.float32)
    label = Tensor(np.ones([32]), dtype=ms.int32)
    dataset = Dataset(predict, label, 2)
    net = all_to_all_net()

    loss = SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, loss, opt)

    model.train(epoch_size, dataset,dataset_sink_mode=False)
    strategys = _executor._get_strategy(model._train_network)
    return strategys


def test_one_dev():

    _reset_op_id()
    strategys = all_to_all_common()
    expect_dict = {'Default/network-_VirtualDatasetCell/_backbone-WithLossCell/_loss_fn-SoftmaxCrossEntropyWithLogits'
                   '/SoftmaxCrossEntropyWithLogits-op18': [[1, 1], [1, 1]],
                   'Default/network-_VirtualDatasetCell/_backbone-WithLossCell/_loss_fn-SoftmaxCrossEntropyWithLogits'
                   '/OneHot-op19': [[1, 1], [], []],
                   'Default/network-_VirtualDatasetCell/_backbone-WithLossCell/_backbone-AllToAllNet/Transpose-op20':
                       [[1, 1]],
                   'Default/network-_VirtualDatasetCell/_backbone-WithLossCell/_backbone-AllToAllNet/MatMul-op21':
                       [[1, 1], [1, 1]]}
    assert (strategys == expect_dict)

