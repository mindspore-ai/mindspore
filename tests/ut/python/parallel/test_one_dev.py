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

import re
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
import mindspore.common.dtype as mstype
from mindspore.common.api import _cell_graph_executor
from mindspore.common.parameter import Parameter
from mindspore.nn.loss.loss import LossBase
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.parallel._utils import _reset_op_id
from mindspore.train import Model
from mindspore.context import ParallelMode
from tests.dataset_mock import MindData

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


class SoftmaxCrossEntropyWithLogits(LossBase):
    def __init__(self,
                 sparse=False,
                 reduction='none'):
        super(SoftmaxCrossEntropyWithLogits, self).__init__(reduction)
        self.sparse = sparse
        self.reduction = reduction
        self.softmax_cross_entropy = P.SoftmaxCrossEntropyWithLogits()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0., mstype.float32)
        self.is_cpugpu = context.get_context('device_target') in ["CPU", "GPU"]

        if self.is_cpugpu:
            self.sparse_softmax_cross_entropy = P.SparseSoftmaxCrossEntropyWithLogits()

    def construct(self, logits, labels):
        if self.is_cpugpu and self.sparse and self.reduction == 'mean':
            x = self.sparse_softmax_cross_entropy(logits, labels)
            return x

        if self.sparse:
            labels = self.one_hot(labels, F.shape(logits)[-1], self.on_value, self.off_value)
        x = self.softmax_cross_entropy(logits, labels)[0]
        return self.get_loss(x)


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

    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, loss, opt)

    model.train(epoch_size, dataset, dataset_sink_mode=False)
    strategys = _cell_graph_executor._get_shard_strategy(model._train_network)
    return strategys


def test_one_dev():
    _reset_op_id()
    strategies = all_to_all_common()
    for (k, v) in strategies.items():
        if re.search('SoftmaxCrossEntropyWithLogits-op', k) is not None:
            assert v == [[1, 1], [1, 1]]
        elif re.search('Transpose-op', k) is not None:
            assert v == [[1, 1]]
        elif re.search('MatMul-op', k) is not None:
            assert v == [[1, 1], [1, 1]]
