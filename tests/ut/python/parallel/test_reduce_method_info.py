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
from mindspore.common.api import _executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


grad_all = C.GradOperation(get_all=True)


class NetWithLossNoBias(nn.Cell):
    def __init__(self, network):
        super(NetWithLossNoBias, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


class GradWrapNoBias(nn.Cell):
    def __init__(self, network):
        super(GradWrapNoBias, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad_all(self.network)(x, y)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        return grad_all(self.network)(x, y, b)


def compile_net_no_bias(net, x, y):
    net.set_auto_parallel()
    net.set_train()
    _executor.compile(net, x, y)


def compile_net(net, x, y, b):
    net.set_auto_parallel()
    net.set_train()
    _executor.compile(net, x, y, b)


# model_parallel test
def test_sum_mul():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.mul1 = P.Mul().shard(strategy1)
            self.reduce_sum = P.ReduceSum(keep_dims=False).shard(strategy2)
            self.mul2 = P.Mul().shard(strategy3)

        def construct(self, x, y, b):
            out = self.mul1(x, y)
            out = self.reduce_sum(out, (1,))
            out = self.mul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((1, 1, 8), (1, 1, 8))
    strategy2 = ((4, 1, 2),)
    strategy3 = ((2, 4), (2, 4))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([128, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_sum_mul2():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.mul1 = P.Mul().shard(strategy1)
            self.reduce_sum = P.ReduceSum(keep_dims=False).shard(strategy2)
            self.mul2 = P.Mul().shard(strategy3)

        def construct(self, x, y, b):
            out = self.mul1(x, y)
            out = self.reduce_sum(out, (0, 1))
            out = self.mul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((1, 1, 4, 2), (1, 1, 4, 2))
    strategy2 = ((2, 4, 1, 1),)
    strategy3 = ((2, 4), (2, 4))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 128, 64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([128, 128, 64, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_sum_mul3():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.mul1 = P.Mul().shard(strategy1)
            self.reduce_sum = P.ReduceSum(keep_dims=False).shard(strategy2)
            self.mul2 = P.Mul().shard(strategy3)

        def construct(self, x, y, b):
            out = self.mul1(x, y)
            out = self.reduce_sum(out, -1)
            out = self.mul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2), (1, 4, 2))
    strategy2 = ((4, 2, 1),)
    strategy3 = ((2, 4), (2, 4))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([128, 32]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_sum_mul4():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.mul1 = P.Mul().shard(strategy1)
            self.reduce_sum = P.ReduceSum(keep_dims=True).shard(strategy2)
            self.mul2 = P.Mul().shard(strategy3)

        def construct(self, x, y, b):
            out = self.mul1(x, y)
            out = self.reduce_sum(out, -1)
            out = self.mul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2), (1, 4, 2))
    strategy2 = ((2, 2, 2),)
    strategy3 = ((4, 2, 1), (4, 2, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([128, 32, 1]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_sum_mul5():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.mul1 = P.Mul().shard(strategy1)
            self.reduce_sum = P.ReduceSum(keep_dims=True).shard(strategy2)

        def construct(self, x, y):
            out = self.mul1(x, y)
            out = self.reduce_sum(out, 0)
            return out

    context.set_auto_parallel_context(device_num=64, global_rank=0)
    strategy1 = ((1, 8, 8), (1, 8, 8))
    strategy2 = ((2, 4, 1),)
    net = GradWrapNoBias(NetWithLossNoBias(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    compile_net_no_bias(net, x, y)


def test_sum_mul6():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.mul1 = P.Mul().shard(strategy1)
            self.reduce_sum = P.ReduceSum(keep_dims=True).shard(strategy2)

        def construct(self, x, y):
            out = self.mul1(x, y)
            out = self.reduce_sum(out, 1)
            return out

    context.set_auto_parallel_context(device_num=64, global_rank=0)
    strategy1 = ((1, 8, 8), (1, 8, 8))
    strategy2 = ((2, 1, 4),)
    net = GradWrapNoBias(NetWithLossNoBias(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    compile_net_no_bias(net, x, y)


def test_sum_mul7():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.mul1 = P.Mul().shard(strategy1)
            self.reduce_sum = P.ReduceSum(keep_dims=True).shard(strategy2)

        def construct(self, x, y):
            out = self.mul1(x, y)
            out = self.reduce_sum(out, (0, 1))
            return out

    context.set_auto_parallel_context(device_num=64, global_rank=0)
    strategy1 = ((1, 8, 8), (1, 8, 8))
    strategy2 = ((2, 4, 1),)
    net = GradWrapNoBias(NetWithLossNoBias(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    compile_net_no_bias(net, x, y)


def test_max_mul():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.mul1 = P.Mul().shard(strategy1)
            self.reduce_max = P.ReduceMax(keep_dims=False).shard(strategy2)
            self.mul2 = P.Mul().shard(strategy3)

        def construct(self, x, y, b):
            out = self.mul1(x, y)
            out = self.reduce_max(out, -1)
            out = self.mul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2), (1, 4, 2))
    strategy2 = ((4, 1, 2),)
    strategy3 = ((2, 4), (2, 4))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([128, 32]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_min_mul():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.mul1 = P.Mul().shard(strategy1)
            self.reduce_min = P.ReduceMin(keep_dims=False).shard(strategy2)
            self.mul2 = P.Mul().shard(strategy3)

        def construct(self, x, y, b):
            out = self.mul1(x, y)
            out = self.reduce_min(out, 0)
            out = self.mul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2), (1, 4, 2))
    strategy2 = ((4, 1, 2),)
    strategy3 = ((2, 4), (2, 4))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([32, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_reduce_mean_mul_float32():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.mul1 = P.Mul().shard(strategy1)
            self.reduce_mean = P.ReduceMean(keep_dims=False).shard(strategy2)
            self.mul2 = P.Mul().shard(strategy3)

        def construct(self, x, y, b):
            out = self.mul1(x, y)
            out = self.reduce_mean(out, 0)
            out = self.mul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2), (1, 4, 2))
    strategy2 = ((4, 1, 2),)
    strategy3 = ((2, 4), (2, 4))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([32, 64]), dtype=ms.float32)

    compile_net(net, x, y, b)


class ArgMaxWithValueNet(nn.Cell):
    def __init__(self, strategy1, strategy2, strategy3):
        super().__init__()
        self.mul1 = P.Mul().shard(strategy1)
        self.arg_max_with_value = P.ArgMaxWithValue(keep_dims=False, axis=-1).shard(strategy2)
        self.mul2 = P.Mul().shard(strategy3)

    def construct(self, x, y, b):
        out = self.mul1(x, y)
        _, out = self.arg_max_with_value(out)
        out = self.mul2(out, b)
        return out


class ArgMinWithValueNet(nn.Cell):
    def __init__(self, strategy1, strategy2, strategy3):
        super().__init__()
        self.mul1 = P.Mul().shard(strategy1)
        self.arg_min_with_value = P.ArgMinWithValue(keep_dims=False, axis=-1).shard(strategy2)
        self.mul2 = P.Mul().shard(strategy3)

    def construct(self, x, y, b):
        out = self.mul1(x, y)
        _, out = self.arg_min_with_value(out)
        out = self.mul2(out, b)
        return out


def gen_inputs_and_compile_net(net):
    x = Tensor(np.ones([128, 64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([128, 64, 64]), dtype=ms.float32)
    b = Tensor(np.ones([128, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def gen_inputs_and_compile_net_no_bias(net):
    x = Tensor(np.ones([128, 64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([128, 64, 64]), dtype=ms.float32)
    compile_net_no_bias(net, x, y)


def tobefixed_test_arg_max_with_value_mul_semi_axis_parallel():
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2), (1, 4, 2))
    strategy2 = ((4, 1, 2),)
    strategy3 = ((2, 4), (2, 4))
    net = GradWrap(NetWithLoss(ArgMaxWithValueNet(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    gen_inputs_and_compile_net(net)


def test_arg_max_with_value_mul_semi():
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2), (1, 4, 2))
    strategy2 = ((4, 1, 1),)
    strategy3 = ((2, 4), (2, 4))
    net = GradWrap(NetWithLoss(ArgMaxWithValueNet(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    gen_inputs_and_compile_net(net)


def test_arg_max_with_value_mul_auto():
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = None
    strategy2 = None
    strategy3 = None
    net = GradWrap(NetWithLoss(ArgMaxWithValueNet(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    gen_inputs_and_compile_net(net)


def test_arg_min_with_value_mul_semi_axis_parallel():
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2), (1, 4, 2))
    strategy2 = ((4, 1, 2),)
    strategy3 = ((2, 4), (2, 4))
    net = GradWrap(NetWithLoss(ArgMinWithValueNet(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    gen_inputs_and_compile_net(net)


def test_arg_min_with_value_mul_semi():
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2), (1, 4, 2))
    strategy2 = ((4, 1, 1),)
    strategy3 = ((2, 4), (2, 4))
    net = GradWrap(NetWithLoss(ArgMinWithValueNet(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    gen_inputs_and_compile_net(net)


def test_arg_min_with_value_mul_auto():
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = None
    strategy2 = None
    strategy3 = None
    net = GradWrap(NetWithLoss(ArgMinWithValueNet(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    gen_inputs_and_compile_net(net)


class ArgMinWithValueNet2(nn.Cell):
    def __init__(self, strategy1, strategy2, strategy3):
        super().__init__()
        self.mul1 = P.Mul().shard(strategy1)
        self.arg_min_with_value = P.ArgMinWithValue(keep_dims=True, axis=-1).shard(strategy2)
        self.relu = P.ReLU().shard(strategy3)

    def construct(self, x, y):
        out = self.mul1(x, y)
        _, out = self.arg_min_with_value(out)
        out = self.relu(out)
        return out


def tobefixed_test_arg_min_with_value_mul_semi_axis_parallel2():
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2), (1, 4, 2))
    strategy2 = ((4, 1, 2),)
    strategy3 = ((2, 4, 1),)
    net = GradWrapNoBias(NetWithLossNoBias(ArgMinWithValueNet2(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    gen_inputs_and_compile_net_no_bias(net)


def test_arg_min_with_value_mul_semi2():
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2), (1, 4, 2))
    strategy2 = ((4, 1, 1),)
    strategy3 = ((2, 4, 1),)
    net = GradWrapNoBias(NetWithLossNoBias(ArgMinWithValueNet2(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    gen_inputs_and_compile_net_no_bias(net)


def test_arg_min_with_value_mul_auto2():
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = None
    strategy2 = None
    strategy3 = None
    net = GradWrapNoBias(NetWithLossNoBias(ArgMinWithValueNet2(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    gen_inputs_and_compile_net_no_bias(net)


def test_cross_batch():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.mul1 = P.Mul().shard(strategy1)
            self.reduce_sum = P.ReduceSum(keep_dims=False).shard(strategy2)
            self.reduce_mean = P.ReduceMean(keep_dims=False).shard(strategy3).add_prim_attr("cross_batch", True)

        def construct(self, x, y):
            out = self.mul1(x, y)
            out = self.reduce_sum(out, -1)
            out = self.reduce_mean(out, 0)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((4, 2), (4, 2))
    strategy2 = ((2, 1),)
    strategy3 = ((8,),)
    net = GradWrapNoBias(NetWithLossNoBias(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    compile_net_no_bias(net, x, y)


def test_cross_batch2():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.mul1 = P.Mul().shard(strategy1)
            self.reduce_mean = P.ReduceMean(keep_dims=False).shard(strategy2)
            self.reduce_sum = P.ReduceSum(keep_dims=False).shard(strategy3).add_prim_attr("cross_batch", True)

        def construct(self, x, y):
            out = self.mul1(x, y)
            out = self.reduce_mean(out, -1)
            out = self.reduce_sum(out, 0)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((4, 2), (4, 2))
    strategy2 = ((2, 1),)
    strategy3 = ((8,),)
    net = GradWrapNoBias(NetWithLossNoBias(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    compile_net_no_bias(net, x, y)


def test_cross_batch_auto():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul1 = P.Mul()
            self.reduce_mean = P.ReduceMean(keep_dims=False)
            self.reduce_sum = P.ReduceSum(keep_dims=False).add_prim_attr("cross_batch", True)

        def construct(self, x, y):
            out = self.mul1(x, y)
            out = self.reduce_mean(out, -1)
            out = self.reduce_sum(out, 0)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = GradWrapNoBias(NetWithLossNoBias(Net()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")

    x = Tensor(np.ones([32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    compile_net_no_bias(net, x, y)


def test_max_empty_tuple():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.mul = P.Mul().shard(strategy1)
            self.reduce_max = P.ReduceMax(keep_dims=False).shard(strategy2)
            self.add = P.Add().shard(strategy3)

        def construct(self, x, y, b):
            out = self.mul(x, y)
            out = self.reduce_max(out)
            out = self.add(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2), (1, 4, 2))
    strategy2 = ((4, 1, 2),)
    strategy3 = ((), (1, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([128, 32]), dtype=ms.float32)

    compile_net(net, x, y, b)
