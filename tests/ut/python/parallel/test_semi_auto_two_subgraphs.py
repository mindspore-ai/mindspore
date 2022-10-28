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
# limitations under the License

import numpy as np

import mindspore as ms
from mindspore import Tensor, Parameter, ParameterTuple, context
from mindspore import nn
from mindspore.common.api import _cell_graph_executor
from mindspore.nn.optim import Adam, FTRL
from mindspore.ops import composite as C
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = P.Mul()
        self.relu = P.ReLU()
        self.param1 = Parameter(Tensor(np.ones([8, 8, 8, 8]).astype(np.float32)), name="wide")
        self.param2 = Parameter(Tensor(np.ones([8, 8, 8, 8]).astype(np.float32)), name="deep")

    def construct(self, x):
        out = self.mul(x, self.param1)
        out = self.mul(out, self.param2)
        out = self.relu(out)
        return out


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.sum = P.ReduceSum(keep_dims=False).shard(((4, 1, 1, 1),))
        self.mean = P.ReduceMean(keep_dims=False).shard(((8, 1, 1, 1),))
        self.net = network

    def construct(self, x):
        net_out = self.net(x)
        loss1 = self.sum(net_out, -1)
        loss2 = self.mean(net_out, -1)
        return loss1, loss2


class IthOutputCell(nn.Cell):
    def __init__(self, network, output_index):
        super(IthOutputCell, self).__init__()
        self.network = network
        self.output_index = output_index

    def construct(self, x1):
        predict = self.network(x1)[self.output_index]
        return predict


class TrainStepWrap(nn.Cell):
    def __init__(self, network, sens=1000.0):
        super(TrainStepWrap, self).__init__()
        self.network = network
        self.network.set_train()
        self.trainable_params = network.trainable_params()
        weights_w = []
        weights_d = []
        for params in self.trainable_params:
            weights_w.append(params)
            weights_d.append(params)

        self.weights_w = ParameterTuple(weights_w)
        self.weights_d = ParameterTuple(weights_d)
        self.optimizer_w = FTRL(learning_rate=1e-2, params=self.weights_w,
                                l1=1e-8, l2=1e-8, initial_accum=1.0)
        self.optimizer_d = Adam(self.weights_d, learning_rate=3.5e-4, eps=1e-8,
                                loss_scale=sens)
        self.hyper_map = C.HyperMap()
        self.grad_w = C.GradOperation(get_by_list=True,
                                      sens_param=True)
        self.grad_d = C.GradOperation(get_by_list=True,
                                      sens_param=True)
        self.sens = sens
        self.loss_net_w = IthOutputCell(network, output_index=0)
        self.loss_net_d = IthOutputCell(network, output_index=1)

    def construct(self, x):
        weights_w = self.weights_w
        weights_d = self.weights_d
        loss_w, loss_d = self.network(x)
        sens_w = P.Fill()(P.DType()(loss_w), P.Shape()(loss_w), self.sens)
        sens_d = P.Fill()(P.DType()(loss_d), P.Shape()(loss_d), self.sens)
        grads_w = self.grad_w(self.loss_net_w, weights_w)(x, sens_w)
        self.optimizer_w(grads_w)
        grads_d = self.grad_d(self.loss_net_d, weights_d)(x, sens_d)
        self.optimizer_d(grads_d)
        return loss_w, loss_d


def test_two_subgraphs():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    net = TrainStepWrap(NetWithLoss(Net()))
    input_x = Tensor(np.ones([8, 8, 8, 8]), dtype=ms.float32)
    net.set_train()
    _cell_graph_executor.compile(net, input_x)
