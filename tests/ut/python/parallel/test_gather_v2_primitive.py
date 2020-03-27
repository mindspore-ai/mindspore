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
from mindspore.ops import composite as C
from mindspore.common.parameter import ParameterTuple
from mindspore.nn.optim import Momentum
from mindspore.communication.management import init
from mindspore.train import Model, ParallelMode
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.loss.loss import _Loss
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn import Dense, Cell
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


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


class GatherV2(_Loss):
    def __init__(self, batchsize):
        super(GatherV2, self).__init__()
        self.pow = P.Pow()
        emb_list = list(range(batchsize))
        emb1_list = emb_list[0::2]
        emb2_list = emb_list[1::2]
        self.emb1_param = Tensor(emb1_list, dtype=mstype.int32)
        self.emb2_param = Tensor(emb2_list, dtype=mstype.int32)
        self.gatherv2 = P.GatherV2()

    def construct(self, nembeddings):
        emb1 = self.gatherv2(nembeddings, self.emb1_param, 0)
        emb2 = self.gatherv2(nembeddings, self.emb2_param, 0)
        return self.pow((emb1 - emb2), 2.0)


def get_loss(batchsize):
    return GatherV2(batchsize)


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
        self.grad = C.GradOperation('grad',
                                    get_by_list=True,
                                    sens_param=True)
        self.sens = sens

    def construct(self, data):
        weights = self.weights
        loss = self.network(data)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(data, sens)

        return F.depend(loss, self.optimizer(grads))


def test_trains():
    init()
    lr = 0.1
    momentum = 0.9
    max_epoch = 20
    device_number = 32
    batch_size_per_device = 128
    input_channels = 256
    out_channels = 512

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, device_num=device_number)
    predict = Tensor(np.ones([batch_size_per_device, input_channels]), dtype=ms.float32)
    dataset = Dataset(predict, 4)

    network = fc_with_initialize(input_channels, out_channels)
    network.set_train()

    criterion = get_loss(batch_size_per_device * device_number)

    train_network = BuildTrainNetwork(network, criterion)
    train_network.set_train()
    opt = Momentum(train_network.trainable_params(), lr, momentum)
    train_net = TrainOneStepCell(train_network, opt).set_train()

    model = Model(train_net)
    model.train(max_epoch, dataset, dataset_sink_mode=False)
    context.reset_auto_parallel_context()

if __name__ == "__main__":
    test_trains()
