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
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.api import _executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.parallel._utils import _reset_op_id as reset_op_id
from mindspore import context, Tensor, Parameter
from mindspore.parallel import set_algo_parameters
from tests.ut.python.ops.test_math_ops import VirtualLoss

grad_all = C.GradOperation(get_all=True)

class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x):
        predict = self.network(x)
        return self.loss(predict)

class GradWarp(nn.Cell):
    def __init__(self, network):
        super(GradWarp, self).__init__()
        self.network = network

    def construct(self, x):
        return grad_all(self.network)(x)

class Net(nn.Cell):
    def __init__(self, strategy_dict=None):
        super(Net, self).__init__()
        self.mul1 = P.Mul()
        self.mul2 = P.Mul()
        self.mul3 = P.Mul()
        self.mul4 = P.Mul()
        self.relu1 = P.ReLU()
        self.relu2 = P.ReLU()
        self.ba1 = P.BiasAdd()
        self.add = P.Add()
        self.weight = Parameter(Tensor(np.ones([128, 1000]), dtype=ms.float32), name="weight")
        self.bias = Parameter(Tensor(np.ones([1000]), dtype=ms.float32), name="bias")

        if strategy_dict is not None:
            self.mul1.shard(strategy_dict["mul1"])
            self.mul2.shard(strategy_dict["mul2"])
            self.relu1.shard(strategy_dict["relu1"])
            self.relu2.shard(strategy_dict["relu2"])
            self.ba1.shard(strategy_dict["bias_add"])
            self.add.shard(strategy_dict["add"])

    def construct(self, inputs):
        x = self.mul1(inputs, self.weight)
        y = self.relu1(x)
        y = self.mul2(y, self.weight)
        z = self.mul3(x, self.weight)
        z = self.ba1(z, self.bias)
        x = self.add(y, z)
        x = self.mul4(x, self.weight)
        x = self.relu2(x)
        return x

def test_star_strategy_consistency1():
    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    set_algo_parameters(fully_use_devices=False)
    x = Tensor(np.ones([128, 1000]), dtype=ms.float32)
    strategy_dict = {"mul1": ((2, 4), (2, 4)), "mul2": None, "relu1": ((4, 1),), "bias_add": ((8, 1), (1,)),
                     "relu2": ((2, 2),), "add": ((1, 8), (1, 8))}
    net = NetWithLoss(Net(strategy_dict))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    reset_op_id()
    net.set_train()
    _executor.compile(net, x, phase='train')


def test_star_strategy_consistency2():
    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    set_algo_parameters(fully_use_devices=False)
    x = Tensor(np.ones([128, 1000]), dtype=ms.float32)
    strategy_dict = {"mul1": None, "mul2": ((1, 4), (1, 4)), "relu1": ((2, 1),), "bias_add": ((4, 2), (2,)),
                     "relu2": ((2, 2),), "add": ((8, 1), (8, 1))}
    net = NetWithLoss(Net(strategy_dict))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    reset_op_id()
    net.set_train()
    _executor.compile(net, x, phase='train')


def test_star_strategy_consistency3():
    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    set_algo_parameters(fully_use_devices=False)
    x = Tensor(np.ones([128, 1000]), dtype=ms.float32)
    strategy_dict = {"mul1": None, "mul2": None, "relu1": ((8, 1),), "bias_add": ((1, 4), (4,)),
                     "relu2": ((4, 1),), "add": ((2, 2), (2, 2))}
    net = NetWithLoss(Net(strategy_dict))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    reset_op_id()
    net.set_train()
    _executor.compile(net, x, phase='train')


def test_star_strategy_consistency4():
    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    set_algo_parameters(fully_use_devices=False)
    x = Tensor(np.ones([128, 1000]), dtype=ms.float32)
    strategy_dict = {"mul1": ((1, 8), (1, 8)), "mul2": ((1, 4), (1, 4)), "relu1": None, "bias_add": None,
                     "relu2": None, "add": None}
    net = NetWithLoss(Net(strategy_dict))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    reset_op_id()
    with pytest.raises(RuntimeError):
        net.set_train()
        _executor.compile(net, x, phase='train')
