# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        return grad_all(self.network)(x, y, b)


def compile_net(net, x, y, b):
    net.set_train()
    _cell_graph_executor.compile(net, x, y, b)


# model_parallel test
def test_two_matmul():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul1(x, y)
            out = self.matmul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    compile_net(net, x, y, b)


# model_parallel test
def test_two_matmul_both_a_b_strategy():
    """
    Feature: test method for matmul operator when both strategies are (a,b)
    Description: transpose_b is false, input dimension 2
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul1(x, y)
            out = self.matmul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=32, global_rank=0, gradients_mean=True)
    strategy1 = ((8, 4), (8, 4))
    strategy2 = ((8, 4), (8, 4))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    compile_net(net, x, y, b)


def test_two_matmul_no_need_gather():
    """
    Feature: test method for matmul operator when a/b is equal to 1.
    Description: transpose_b is false, input dimension 2.
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul1(x, y)
            out = self.matmul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((8, 1), (8, 1))
    strategy2 = ((8, 1), (8, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    compile_net(net, x, y, b)


def test_two_matmul_repeated_calculation1():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul1(x, y)
            out = self.matmul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=64, global_rank=5, gradients_mean=True)
    strategy1 = ((2, 4), (4, 8))
    strategy2 = ((1, 1), (1, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_two_matmul_repeated_calculation2():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul1(x, y)
            out = self.matmul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=64, global_rank=15)
    strategy1 = ((2, 4), (4, 8))
    strategy2 = ((2, 2), (2, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_output_strategy_reduce_scatter():
    """
    Feature: test output strategy for matmul operator
    Description: transpose_b is false, set output strategy and use reduce scatter
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, matmul_in_strategy, matmul_out_strategy, mul_strategy):
            super().__init__()
            self.matmul = P.MatMul().shard(matmul_in_strategy, matmul_out_strategy)
            self.mul = P.Mul().shard(mul_strategy)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    matmul_in_strategy = ((2, 2), (2, 2))
    matmul_out_strategy = ((4, 2),)
    mul_strategy = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(matmul_in_strategy, matmul_out_strategy, mul_strategy)))

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([128, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_output_strategy_reduce_scatter_transpose():
    """
    Feature: test output strategy for matmul operator
    Description: transpose_b is true, set output strategy and use reduce scatter
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, matmul_in_strategy, matmul_out_strategy, mul_strategy):
            super().__init__()
            self.matmul = P.MatMul(transpose_b=True).shard(matmul_in_strategy, matmul_out_strategy)
            self.mul = P.Mul().shard(mul_strategy)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    matmul_in_strategy = ((2, 4), (2, 4))
    matmul_out_strategy = ((8, 2),)
    mul_strategy = ((8, 2), (8, 2))
    net = GradWrap(NetWithLoss(Net(matmul_in_strategy, matmul_out_strategy, mul_strategy)))

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 32]), dtype=ms.float32)
    b = Tensor(np.ones([128, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_output_strategy_all_reduce():
    """
    Feature: test output strategy for matmul operator
    Description: transpose_b is false, set output strategy and use all reduce
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, matmul_in_strategy, matmul_out_strategy, mul_strategy):
            super().__init__()
            self.matmul = P.MatMul().shard(matmul_in_strategy, matmul_out_strategy)
            self.mul = P.Mul().shard(mul_strategy)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    matmul_in_strategy = ((2, 2), (2, 2))
    matmul_out_strategy = ((2, 2),)
    mul_strategy = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(matmul_in_strategy, matmul_out_strategy, mul_strategy)))

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([128, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_output_strategy_all_reduce_transpose():
    """
    Feature: test output strategy for matmul operator
    Description: transpose_b is true, set output strategy and use all reduce
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, matmul_in_strategy, matmul_out_strategy, mul_strategy):
            super().__init__()
            self.matmul = P.MatMul(transpose_b=True).shard(matmul_in_strategy, matmul_out_strategy)
            self.mul = P.Mul().shard(mul_strategy)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    matmul_in_strategy = ((2, 2), (2, 2))
    matmul_out_strategy = ((2, 2),)
    mul_strategy = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(matmul_in_strategy, matmul_out_strategy, mul_strategy)))

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 32]), dtype=ms.float32)
    b = Tensor(np.ones([128, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_output_strategy_reduce_scatter_repeat_calc():
    """
    Feature: test output strategy for matmul operator
    Description: transpose_b is false, set output strategy use reduce scatter and repeated calculation
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, matmul_in_strategy, matmul_out_strategy, mul_strategy):
            super().__init__()
            self.matmul = P.MatMul().shard(matmul_in_strategy, matmul_out_strategy)
            self.mul = P.Mul().shard(mul_strategy)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    matmul_in_strategy = ((2, 2), (2, 2))
    matmul_out_strategy = ((4, 2),)
    mul_strategy = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(matmul_in_strategy, matmul_out_strategy, mul_strategy)))

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([128, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_output_strategy_reduce_scatter_transpose_repeat_calc():
    """
    Feature: test output strategy for matmul operator
    Description: transpose_b is true, set output strategy use reduce scatter and repeated calculation
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, matmul_in_strategy, matmul_out_strategy, mul_strategy):
            super().__init__()
            self.matmul = P.MatMul(transpose_b=True).shard(matmul_in_strategy, matmul_out_strategy)
            self.mul = P.Mul().shard(mul_strategy)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=0)
    matmul_in_strategy = ((2, 4), (2, 4))
    matmul_out_strategy = ((8, 2),)
    mul_strategy = ((8, 2), (8, 2))
    net = GradWrap(NetWithLoss(Net(matmul_in_strategy, matmul_out_strategy, mul_strategy)))

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 32]), dtype=ms.float32)
    b = Tensor(np.ones([128, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_output_strategy_all_reduce_repeat_calc():
    """
    Feature: test output strategy for matmul operator
    Description: transpose_b is false, set output strategy use all reduce and repeated calculation
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, matmul_in_strategy, matmul_out_strategy, mul_strategy):
            super().__init__()
            self.matmul = P.MatMul().shard(matmul_in_strategy, matmul_out_strategy)
            self.mul = P.Mul().shard(mul_strategy)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    matmul_in_strategy = ((2, 2), (2, 2))
    matmul_out_strategy = ((2, 2),)
    mul_strategy = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(matmul_in_strategy, matmul_out_strategy, mul_strategy)))

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([128, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_output_strategy_all_reduce_transpose_repeat_calc():
    """
    Feature: test output strategy for matmul operator
    Description: transpose_b is true, set output strategy use all reduce and repeated calculation
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, matmul_in_strategy, matmul_out_strategy, mul_strategy):
            super().__init__()
            self.matmul = P.MatMul(transpose_b=True).shard(matmul_in_strategy, matmul_out_strategy)
            self.mul = P.Mul().shard(mul_strategy)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    matmul_in_strategy = ((2, 2), (2, 2))
    matmul_out_strategy = ((2, 2),)
    mul_strategy = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(matmul_in_strategy, matmul_out_strategy, mul_strategy)))

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([64, 32]), dtype=ms.float32)
    b = Tensor(np.ones([128, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_in_strategy_not_int():
    """
    Feature: the type of in_strategy's value is not int
    Description:
    Expectation: rasise TypeError
    """
    class Net(nn.Cell):
        def __init__(self, matmul_in_strategy, matmul_out_strategy, mul_strategy):
            super().__init__()
            self.matmul = P.MatMul(transpose_b=True).shard(matmul_in_strategy, matmul_out_strategy)
            self.mul = P.Mul().shard(mul_strategy)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    matmul_in_strategy = ((2.0, 2), (2, 2))
    matmul_out_strategy = ((2, 2),)
    mul_strategy = ((4, 2), (4, 2))

    with pytest.raises(TypeError):
        GradWrap(NetWithLoss(Net(matmul_in_strategy, matmul_out_strategy, mul_strategy)))


def test_matmul_out_strategy_not_int():
    """
    Feature: the type of out_strategy's value is not int
    Description:
    Expectation: rasise TypeError
    """
    class Net(nn.Cell):
        def __init__(self, matmul_in_strategy, matmul_out_strategy, mul_strategy):
            super().__init__()
            self.matmul = P.MatMul(transpose_b=True).shard(matmul_in_strategy, matmul_out_strategy)
            self.mul = P.Mul().shard(mul_strategy)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    matmul_in_strategy = ((2, 2), (2, 2))
    matmul_out_strategy = ((2.0, 2),)
    mul_strategy = ((4, 2), (4, 2))

    with pytest.raises(TypeError):
        GradWrap(NetWithLoss(Net(matmul_in_strategy, matmul_out_strategy, mul_strategy)))


def test_matmul_in_strategy_is_none_and_out_strategy_is_not_none():
    """
    Feature: the in_strategy is none and out_strategy is not none
    Description:
    Expectation: rasise ValueError
    """
    class Net(nn.Cell):
        def __init__(self, matmul_in_strategy, matmul_out_strategy, mul_strategy):
            super().__init__()
            self.matmul = P.MatMul(transpose_b=True).shard(matmul_in_strategy, matmul_out_strategy)
            self.mul = P.Mul().shard(mul_strategy)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    matmul_in_strategy = None
    matmul_out_strategy = ((2, 2),)
    mul_strategy = ((4, 2), (4, 2))

    with pytest.raises(ValueError):
        GradWrap(NetWithLoss(Net(matmul_in_strategy, matmul_out_strategy, mul_strategy)))
