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
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad_all(self.network)(x, y)


class Net(nn.Cell):
    def __init__(self, axis=0, strategy1=None, strategy2=None, shape=None, target=""):
        super().__init__()
        if shape is None:
            shape = [64, 64]
        self.gatherv2 = P.Gather().shard(strategy1).add_prim_attr("primitive_target", target)
        self.mul = P.Mul().shard(strategy2)
        self.index = Tensor(np.ones(shape), dtype=ms.int32)
        self.axis = axis

    def construct(self, x, y):
        out = self.gatherv2(x, self.index, self.axis)
        out = self.mul(out, y)
        return out


def test_gatherv2_semi_auto0():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((1, 8), (1, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2)))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)


def test_gatherv2_semi_auto1():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((8, 1), (1, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2)))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)


def test_gatherv2_semi_auto2():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (1, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2)))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)


def test_gatherv2_semi_auto3():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((1, 8), (1, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(1, strategy1, strategy2)))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)


def test_gatherv2_semi_auto4():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((8, 1), (1, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(1, strategy1, strategy2)))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)


def test_gatherv2_semi_auto5():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (1, 1))
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(1, strategy1, strategy2)))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)


def test_gatherv2_semi_auto6():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(0, None, strategy2)))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 32]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)


def test_gatherv2_semi_auto7():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy2 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(1, None, strategy2)))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)


def test_gatherv2_semi_auto8():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((8,), (1, 1))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2)))
    net.set_auto_parallel()

    x = Tensor(np.ones([64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)


def test_gatherv2_forward_all_reduce():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((8, 1), (1, 1))
    strategy2 = ((2, 4, 1), (2, 4, 1))
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, shape=[2, 64])))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([2, 64, 64]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)


def test_gatherv2_split_axis_0_repeat_calc():
    context.set_auto_parallel_context(device_num=8, global_rank=7, parallel_mode="semi_auto_parallel")
    strategy1 = ((4, 1), (1, 1))
    strategy2 = ((2, 4, 1), (2, 4, 1))
    net = GradWrap(NetWithLoss(Net(0, strategy1, strategy2, shape=[2, 64])))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([2, 64, 64]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)


def test_gatherv2_auto0():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="auto_parallel")
    net = GradWrap(NetWithLoss(Net(0)))
    net.set_auto_parallel()
    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 32]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)


def test_gatherv2_auto1():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="auto_parallel")
    net = GradWrap(NetWithLoss(Net(1)))
    net.set_auto_parallel()
    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)
