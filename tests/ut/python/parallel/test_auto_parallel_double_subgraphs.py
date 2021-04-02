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
import re
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore import context, Model
from mindspore.common.api import _executor
from mindspore.nn.optim import Adam, FTRL
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.parallel._utils import _reset_op_id as reset_op_id


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = P.Mul()
        self.relu = P.ReLU()
        self.wd = Parameter(Tensor(np.ones([8, 8, 8, 8]).astype(np.float32)), name="wide")
        self.wt = Parameter(Tensor(np.ones([8, 8, 8, 8]).astype(np.float32)), name="l")

    def construct(self, x):
        out = self.mul(x, self.wd)
        out = self.mul(out, self.wt)
        out = self.relu(out)
        return out


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.sum = P.ReduceSum()
        self.mean = P.ReduceMean()
        self.net = network

    def construct(self, x):
        predict = self.net(x)
        loss1 = self.sum(predict, -1)
        loss2 = self.mean(predict, -1)
        return loss1, loss2


class IthOutputCell(nn.Cell):
    def __init__(self, network, output_index):
        super(IthOutputCell, self).__init__()
        self.network = network
        self.output_index = output_index

    def construct(self, x):
        predict = self.network(x)[self.output_index]
        return predict


class TrainStepWarp(nn.Cell):
    def __init__(self, network, sens=1000.0):
        super(TrainStepWarp, self).__init__()
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
        self.optimizer_w = FTRL(learning_rate=1e-2, params=self.weights_w, l1=1e-8,
                                l2=1e-8, initial_accum=1.0)
        self.optimizer_d = Adam(self.weights_d, learning_rate=3.5e-4, eps=1e-8,
                                loss_scale=sens)
        self.hyper_map = C.HyperMap()
        self.grad_w = C.GradOperation(get_by_list=True, sens_param=True)
        self.grad_d = C.GradOperation(get_by_list=True, sens_param=True)
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
        grads_d = self.grad_d(self.loss_net_d, weights_d)(x, sens_d)
        return F.depend(loss_w, self.optimizer_w(grads_w)), F.depend(loss_d, self.optimizer_d(grads_d))


def test_double_subgraphs():
    _set_multi_subgraphs()
    context.set_context(save_graphs=False)
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net = TrainStepWarp(NetWithLoss(Net()))
    net.set_auto_parallel()

    x = Tensor(np.ones([8, 8, 8, 8]), dtype=ms.float32)
    reset_op_id()
    net.set_train()
    _executor.compile(net, x, phase='train')
    strategies = _executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('ReduceMean-op', k) is not None:
            assert v == [[8, 1, 1, 1]]
        elif re.search('ReLU-op', k) is not None:
            assert v == [[8, 1, 1, 1]]
        elif re.search('Mul-op', k) is not None:
            assert v == [[8, 1, 1, 1], [8, 1, 1, 1]]
        elif re.search('ReduceSum-op', k) is not None:
            assert v == [[8, 1, 1, 1]]


class DatasetLenet():
    def __init__(self, predict, label, length=3):
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
        return self.predict

    def reset(self):
        self.index = 0

    def get_dataset_size(self):
        return 32

    def get_repeat_count(self):
        return 1

    def create_tuple_iterator(self, num_epochs=-1, do_copy=True):
        return self

def test_double_subgraphs_train():
    context.set_context(save_graphs=False)
    context.set_auto_parallel_context(device_num=1, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net = TrainStepWarp(NetWithLoss(Net()))

    batch_ids = np.ones([8, 8, 8, 8]).astype(np.int32)
    ds_train = DatasetLenet(Tensor(batch_ids), None)
    model = Model(net)
    model.train(1, ds_train, dataset_sink_mode=False)
    strategies = _executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('ReduceMean-op', k) is not None:
            assert v == [[1, 1, 1, 1]]
        elif re.search('ReLU-op', k) is not None:
            assert v == [[1, 1, 1, 1]]
        elif re.search('Mul-op', k) is not None:
            assert v == [[1, 1, 1, 1], [1, 1, 1, 1]]
        elif re.search('Cast-op', k) is not None:
            assert v == [[1, 1, 1, 1]]
        elif re.search('ReduceSum-op', k) is not None:
            assert v == [[1, 1, 1, 1]]
