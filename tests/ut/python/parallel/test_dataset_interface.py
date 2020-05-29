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

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import composite as C, functional as F, operations as P
from mindspore.train import Model, ParallelMode
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
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
    def __init__(self, strategy1):
        super(AllToAllNet, self).__init__()
        self.matmul = P.MatMul().set_strategy(((1, 1), (1, 8)))
        self.matmul_weight = Parameter(Tensor(np.ones([128, 256]), dtype=ms.float32), name="weight")
        self.transpose1 = P.Transpose().set_strategy(strategy1)

    def construct(self, x):
        x = self.matmul(x, self.matmul_weight)
        x = self.transpose1(x, (1, 0))
        return x


def all_to_all_net(strategy1):
    return AllToAllNet(strategy1=strategy1)


def loss_scale_manager_common(strategy1):
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, device_num=8)
    predict = Tensor(np.ones([32, 128]), dtype=ms.float32)
    label = Tensor(np.ones([32]), dtype=ms.int32)
    dataset = Dataset(predict, label, 2)
    net = all_to_all_net(strategy1)

    loss = SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
    loss.softmax_cross_entropy.set_strategy(((8, 1), (8, 1)))
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    scale_manager = DynamicLossScaleManager(32, 2, 2000)
    model = Model(net, loss, opt, loss_scale_manager=scale_manager)
    # if no GE exists, outputs = self._train_network(*next_element) outputs inputs tensor.
    try:
        model.train(epoch_size, dataset, dataset_sink_mode=False)
    except TypeError:
        pass
    else:
        assert False


def fixme_test_dataset_interface_sens_scalar():
    # With error: "The type of sens node is not Tensor or Parameter, it is unsupported now."
    strategy1 = ((8, 1),)
    loss_scale_manager_common(strategy1)


class TrainOneStepCell(nn.Cell):

    def __init__(self, network, optimizer):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation('grad', get_by_list=True, sens_param=True)

    def construct(self, data, sens):
        weights = self.weights
        loss = self.network(data)
        grads = self.grad(self.network, weights)(data, sens)
        return F.depend(loss, self.optimizer(grads))


def loss_scale_manager_sens(strategy1, sens):
    learning_rate = 0.1
    momentum = 0.9
    device_num = 8
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, device_num=device_num)
    predict = Tensor(np.ones([32 * device_num, 128]), dtype=ms.float32)
    net = all_to_all_net(strategy1)
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    train_net = TrainOneStepCell(net, opt)
    train_net.set_train()
    train_net(predict, sens)


def test_dataset_interface_sens_shape_not_equal_loss():
    strategy1 = ((8, 1),)
    sens = Tensor(np.ones([256, 1024]), dtype=ms.float32)
    try:
        loss_scale_manager_sens(strategy1, sens)
    except ValueError:
        pass
    except TypeError:
        pass
    except RuntimeError:
        pass


def test_dataset_interface_sens_shape_equal_loss():
    strategy1 = ((4, 2),)
    sens = Tensor(np.ones([256, 256]), dtype=ms.float32)
    loss_scale_manager_sens(strategy1, sens)


def test_input_not_in_parameter_layotu_dict():
    class Net(nn.Cell):
        def __init__(self, strategy1):
            super(Net, self).__init__()
            self.matmul = P.MatMul().set_strategy(((1, 1), (1, 8)))
            self.matmul_weight = Parameter(Tensor(np.ones([128, 256]), dtype=ms.float32), name="weight")
            self.transpose1 = P.Transpose().set_strategy(strategy1)

        def construct(self, x):
            x = self.matmul(x, self.matmul_weight)
            x = self.transpose1(x, (1, 0))
            return x

    strategy1 = ((8, 1),)
    device_num = 8
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, device_num=device_num)
    predict = Tensor(np.ones([32 * device_num, 128]), dtype=ms.float32)
    net = Net(strategy1)
    net.set_train()
    net(predict)
