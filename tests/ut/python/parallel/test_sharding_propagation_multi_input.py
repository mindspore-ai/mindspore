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
from mindspore import Tensor, context, mint
from mindspore.common import Parameter
from mindspore.graph.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss
import re


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x):
        predict = self.network(x)
        return self.loss(predict)

class NetWithLossSoftmax(nn.Cell):
    def __init__(self, network):
        super(NetWithLossSoftmax, self).__init__()
        self.loss = P.SoftmaxCrossEntropyWithLogits()
        self.network = network

    def construct(self, x, b):
        predict = self.network(x)
        return self.loss(predict, b)[0]

class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x):
        return grad_all(self.network)(x)


class GradWrapTwoInput(nn.Cell):
    def __init__(self, network):
        super(GradWrapTwoInput, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad_all(self.network)(x, y)


class GradWrapThreeInput(nn.Cell):
    def __init__(self, network):
        super(GradWrapThreeInput, self).__init__()
        self.network = network

    def construct(self, x, y, z):
        return grad_all(self.network)(x, y, z)



class NetWithLossTwoInput(nn.Cell):
    def __init__(self, network):
        super(NetWithLossTwoInput, self).__init__()
        self.mean = P.ReduceMean(keep_dims=False)
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.mean(predict, ())


class NetWithLoss3(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss3, self).__init__()
        self.mean = P.ReduceMean(keep_dims=False)
        self.network = network

    def construct(self, x, y, z):
        predict = self.network(x, y, z)
        return self.mean(predict, ())



def compile_graph(net, device_num, x):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode="auto_parallel",
                                      search_mode="sharding_propagation")
    net.set_train()
    _cell_graph_executor.compile(net, x)


def compile_graph_two_input(net, device_num, x, y):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode="auto_parallel",
                                      search_mode="sharding_propagation")
    net.set_train()
    _cell_graph_executor.compile(net, x, y)



def test_mint_func():
    """
    Feature: Sharding propagation for func.
    Description: x->abs->mul y->mul
    Expectation: mul gets right strategy.
    """
    device_num = 8
    class MulNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.gamma = Parameter(Tensor(np.ones([128, 96]), dtype=ms.float32), name="gamma")
            def mul_func(x, y):
                x = mint.abs(x)
                return mint.mul(x, y)
            self.mul_shard = ms.shard(mul_func, in_strategy=((4, 2), (4, 1)))

        def construct(self, input_x, input_y):
            output = self.mul_shard(input_x, input_y)
            print("output:", output)
            return output


    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode="auto_parallel",
                                      search_mode="sharding_propagation")

    net = GradWrapTwoInput(NetWithLossTwoInput(MulNet()))
    net.set_train()

    x = Tensor(np.ones([128, 96]), dtype=ms.float32)
    y = Tensor(np.ones([128, 1]), dtype=ms.float32)


    _cell_graph_executor.compile(net, x, y, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    context._reset_auto_parallel_context()
    for (k, v) in strategies.items():
        print("cnode: {} strategy: {}".format(k, v))
        if re.search('Mul-op0', k) is not None:
            print("check Mul-op0")
            assert v == [[4, 2], [4, 1]]


def test_mint_matmul_layout():
    """
    Feature: Sharding propagation for mint.matmul.
    Description: identity(4, 2)->matmul identity(2, 1)->matmul
    Expectation: matmul gets right strategy.
    """
    device_num = 8
    class MatMulNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.gamma = Parameter(Tensor(np.ones([128, 96]), dtype=ms.float32), name="gamma")
            self.matmul = mint.matmul
            ly = ms.Layout((4, 2), ("axis0", "axis1"))
            self.matmul_shard = ms.shard(self.matmul, in_strategy=(ly("axis0", "axis1"), ly("axis1", "None")))

        def construct(self, input_x, input_y):
            output = self.matmul_shard(input_x, input_y)
            print("output:", output)
            return output

    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode="auto_parallel",
                                      search_mode="sharding_propagation")

    net = GradWrapTwoInput(NetWithLossTwoInput(MatMulNet()))
    net.set_train()

    x = Tensor(np.ones([128, 96]), dtype=ms.float32)
    y = Tensor(np.ones([96, 96]), dtype=ms.float32)


    _cell_graph_executor.compile(net, x, y, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    context._reset_auto_parallel_context()
    for (k, v) in strategies.items():
        print("cnode: {} strategy: {}".format(k, v))
        if re.search('MatMulExt', k) is not None:
            print("check MatMulExt")
            assert v == [[4, 2], [2, 1]]


def test_mint_matmul_layout_222():
    """
    Feature: Sharding propagation for mint.matmul.
    Description: identity(1, 2)->matmul identity(2, 2)->matmul
    Expectation: matmul gets right strategy.
    """
    device_num = 8
    class MatMulNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.gamma = Parameter(Tensor(np.ones([128, 96]), dtype=ms.float32), name="gamma")
            self.matmul = mint.matmul
            ly = ms.Layout((2, 2, 2, 1), ("axis0", "axis1", "axis2", "axis3"))
            self.matmul_shard = ms.shard(self.matmul, in_strategy=(ly("axis3", "axis1"), ly("axis1", "axis0")))

        def construct(self, input_x, input_y):
            output = self.matmul_shard(input_x, input_y)
            print("output:", output)
            return output

    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode="auto_parallel",
                                      search_mode="sharding_propagation")

    net = GradWrapTwoInput(NetWithLossTwoInput(MatMulNet()))
    net.set_train()

    x = Tensor(np.ones([128, 96]), dtype=ms.float32)
    y = Tensor(np.ones([96, 96]), dtype=ms.float32)


    _cell_graph_executor.compile(net, x, y, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    context._reset_auto_parallel_context()
    for (k, v) in strategies.items():
        print("cnode: {} strategy: {}".format(k, v))
        if re.search('MatMulExt', k) is not None:
            print("check MatMulExt")
            assert v == [[1, 2], [2, 2]]

def test_mint_add_layout_222():
    """
    Feature: Sharding propagation for mint.add.
    Description: identity(2, 1)->add identity(2, 1)->add
    Expectation: add gets right strategy.
    """
    device_num = 8
    class MatMulNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.gamma = Parameter(Tensor(np.ones([128, 96]), dtype=ms.float32), name="gamma")
            self.add = mint.add
            ly = ms.Layout((2, 1, 2, 2), ("axis0", "axis1", "axis2", "axis3"))
            self.add_shard = ms.shard(self.add, in_strategy=(ly("axis3", "axis1"), ly("axis3", "axis1")))

        def construct(self, input_x, input_y):
            output = self.add_shard(input_x, input_y)
            print("output:", output)
            return output

    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode="auto_parallel",
                                      search_mode="sharding_propagation")

    net = GradWrapTwoInput(NetWithLossTwoInput(MatMulNet()))
    net.set_train()

    x = Tensor(np.ones([96, 96]), dtype=ms.float32)
    y = Tensor(np.ones([96, 96]), dtype=ms.float32)


    _cell_graph_executor.compile(net, x, y, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    context._reset_auto_parallel_context()
    for (k, v) in strategies.items():
        print("cnode: {} strategy: {}".format(k, v))
        if re.search('AddExt', k) is not None:
            print("check AddExt")
            assert v == [[2, 1], [2, 1]]


def test_mint_add_layout_broadcast():
    """
    Feature: Sharding propagation for mint.add.
    Description: identity(4, 2)->add identity(1, 2)->add
    Expectation: add gets right strategy.
    """
    device_num = 8
    class AddNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.gamma = Parameter(Tensor(np.ones([128, 96]), dtype=ms.float32), name="gamma")
            self.add = mint.add
            ly = ms.Layout((4, 2), ("axis0", "axis1"))
            self.add_shard = ms.shard(self.add, in_strategy=(ly("axis0", "axis1"), ly("None", "axis1")))

        def construct(self, input_x, input_y):
            output = self.add_shard(input_x, input_y)
            print("output:", output)
            return output

    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode="auto_parallel",
                                      search_mode="sharding_propagation")

    net = NetWithLossTwoInput(AddNet())
    net.set_train()

    x = Tensor(np.ones([128, 96]), dtype=ms.float32)
    y = Tensor(np.ones([1, 96]), dtype=ms.float32)


    _cell_graph_executor.compile(net, x, y, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    context._reset_auto_parallel_context()
    for (k, v) in strategies.items():
        print("cnode: {} strategy: {}".format(k, v))
        if re.search('AddExt', k) is not None:
            print("check Add-op0")
            assert v == [[4, 2], [1, 2]]


def test_mint_flash_attention_score_BSH_layout():
    """
    Feature: Sharding propagation for flash_attention_score.
    Description: identity(2, 1, 4)->flash_attention_score identity(2, 1, 4)->flash_attention_score
    Expectation: flash_attention_score gets right strategy.
    """
    device_num = 8
    class FlashAttentionScoreNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.gamma = Parameter(Tensor(np.ones([128, 96]), dtype=ms.float32), name="gamma")
            self.num_head = 4
            ly = ms.Layout((2, 1, 4), ("axis0", "axis1", "axis2"))
            fas_layout = (ly("axis0", "axis1", "axis2"), ly("axis0", "axis1", "axis2"), ly("axis0", "axis1", "axis2"))
            def fas_func(x, y, z):
                return ms.ops.flash_attention_score(x, y, z, self.num_head)
            self.flash_attention_score_shard = ms.shard(fas_func, in_strategy=fas_layout)

        def construct(self, q, k, v):
            output = self.flash_attention_score_shard(q, k, v)
            print("output:", output)
            return output

    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode="auto_parallel",
                                      search_mode="sharding_propagation")

    net = GradWrapThreeInput(NetWithLoss3(FlashAttentionScoreNet()))
    net.set_train()

    q = Tensor(np.ones([4, 8, 96]), dtype=ms.bfloat16)
    k = Tensor(np.ones([4, 8, 96]), dtype=ms.bfloat16)
    v = Tensor(np.ones([4, 8, 96]), dtype=ms.bfloat16)


    _cell_graph_executor.compile(net, q, k, v, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    context._reset_auto_parallel_context()
    for (k, v) in strategies.items():
        print("cnode: {} strategy: {}".format(k, v))
        if re.search('FlashAttentionScore', k) is not None:
            print("check FlashAttentionScore")
            assert v == [[2, 1, 4], [2, 1, 4], [2, 1, 4]]



def test_relu_relu_matmul_layout():
    """
    Feature: Sharding propagation for relu, relu, matmul, add net.
    Description: relu(4, 2)->matmul relu(2, 1)->matmul matmul->add relu(4, 2)->add
    Expectation: matmul, add get right strategy.
    """
    device_num = 8
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.rma_net = RMANet()
            ly = ms.Layout((4, 2), ("axis0", "axis1"))
            in_layout = (ly("axis0", "axis1"), ly("axis1", "None"))
            gamma_layout = ly("axis0", "axis1")
            beta_layout = ly("axis0", "axis1")
            self.rma_net_shard = ms.shard(self.rma_net, in_strategy=in_layout,
                                          parameter_plan={"self.rma_net.gamma": gamma_layout,
                                                          "self.rma_net.beta": beta_layout})
        def construct(self, x, y):
            return self.rma_net_shard(x, y)

    class RMANet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.gamma = Parameter(Tensor(np.ones([128, 96]), dtype=ms.float32), name="gamma")
            self.beta = Parameter(Tensor(np.ones([128, 96]), dtype=ms.float32), name="beta")
            self.add1 = mint.add
            self.add2 = mint.add
            self.relu1 = ms.ops.relu
            self.relu2 = ms.ops.relu
            self.relu3 = ms.ops.relu
            self.matmul = mint.matmul

        def construct(self, x, y):
            out0 = self.add1(x, self.gamma)
            out1 = self.relu1(out0)
            out2 = self.relu2(y)
            out3 = self.matmul(out1, out2)
            out4 = self.relu3(self.beta)
            out = self.add2(out3, out4)
            return out

    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode="auto_parallel",
                                      search_mode="sharding_propagation")

    x = Tensor(np.ones([128, 96]), dtype=ms.float32)
    y = Tensor(np.ones([96, 1]), dtype=ms.float32)


    net = GradWrapTwoInput(NetWithLossTwoInput(Net()))

    net.set_train()
    _cell_graph_executor.compile(net, x, y, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    context._reset_auto_parallel_context()
    for (k, v) in strategies.items():
        print("cnode: {} strategy: {}".format(k, v))
        if re.search('AddExt-op0', k) is not None:
            print("check AddExt-op0")
            assert v == [[4, 2], [4, 2]]
        if re.search('MatMulExt-op0', k) is not None:
            print("check MatMulExt")
            assert v == [[4, 2], [2, 1]]
        elif re.search('AddExt-op1', k) is not None:
            print("check AddExt-op1")
            assert v == [[4, 1], [4, 2]]


def test_mint_waitting():
    """
    Feature: Sharding propagation for relu, relu, matmul, add net.
    Description: relu(4, 2)->matmul relu(2, 1)->matmul matmul->add
    Expectation: matmul, add get right strategy.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.rmasd_net = RMASDNet()
            ly = ms.Layout((4, 2), ("axis0", "axis1"))
            in_layout = (ly("axis0", "axis1"), ly("axis0", "axis1"))
            alpha_layout = ly("axis0", "axis1")
            self.rmasd_net_shard = ms.shard(self.rmasd_net, in_strategy=in_layout,
                                            parameter_plan={"self.rmasd_net.alpha": alpha_layout})

        def construct(self, x, y):
            return self.rmasd_net_shard(x, y)


    class RMASDNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.beta = Parameter(Tensor(np.ones([96, 1]), dtype=ms.float32), name="beta")
            self.alpha = Parameter(Tensor(np.ones([128, 96]), dtype=ms.float32), name="alpha")
            self.sub = mint.sub
            self.add = mint.add
            self.relu1 = ms.ops.relu
            self.relu2 = ms.ops.relu

            self.matmul = mint.matmul
            self.div = mint.div

        def construct(self, x, y):
            out3 = self.add(x, y)
            out4 = self.relu1(self.beta)
            out5 = self.matmul(out3, out4)
            out6 = self.relu2(self.alpha)
            out = self.div(out5, out6)
            return out


    device_num = 8
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode="auto_parallel",
                                      search_mode="sharding_propagation")
    x = Tensor(np.ones([128, 96]), dtype=ms.float32)
    y = Tensor(np.ones([128, 96]), dtype=ms.float32)

    net = GradWrapTwoInput(NetWithLossTwoInput(Net()))

    net.set_train()
    _cell_graph_executor.compile(net, x, y, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    context._reset_auto_parallel_context()
    for (k, v) in strategies.items():
        print("cnode: {} strategy: {}".format(k, v))
        if re.search('AddExt-op0', k) is not None:
            print("check AddExt-op0")
            assert v == [[4, 2], [4, 2]]
        elif re.search('MatMulExt-op0', k) is not None:
            print("check MatMulExt-op0")
            assert v == [[4, 2], [2, 1]]
        elif re.search('Div-op0', k) is not None:
            print("check Div-op0")
            assert v == [[4, 1], [4, 2]]

def test_mint_rma_softmaxcrossentropywithlogits():
    """
    Feature: Sharding propagation for add, relu, matmul net, and use softmaxcrossentropywithlogits.
    Description: To test whether Virtual dataset be inserted correctly.
    Expectation: matmul, add get right strategy without compile error.
    """
    class Net(nn.Cell):
        def __init__(self, in_layout, gamma_layout, use_shard=False, matmul_strategy=False, relu_strategy=False):
            super(Net, self).__init__()
            if matmul_strategy and relu_strategy:
                self.rma_net = RMANet(matmul_strategy=matmul_strategy, relu_strategy=relu_strategy)
            elif matmul_strategy and not relu_strategy:
                self.rma_net = RMANet(matmul_strategy=matmul_strategy)
            elif relu_strategy and not matmul_strategy:
                self.rma_net = RMANet(relu_strategy=relu_strategy)
            else:
                self.rma_net = RMANet()
            self.use_shard = use_shard
            if self.use_shard:
                self.rma_net_shard = ms.shard(self.rma_net, in_strategy=in_layout,
                                              parameter_plan={"self.rma_net.gamma": gamma_layout})
        def construct(self, x):
            if self.use_shard:
                return self.rma_net_shard(x)
            return self.rma_net(x)

    class RMANet(nn.Cell):
        def __init__(self, matmul_strategy=False, relu_strategy=False):
            super().__init__()
            self.gamma = Parameter(Tensor(np.ones([16, 16]), dtype=ms.float32), name="gamma")
            self.beta = Parameter(Tensor(np.ones([16, 16]), dtype=ms.float32), name="beta")
            self.add = mint.add
            if not matmul_strategy:
                self.matmul = mint.matmul
            else:
                self.matmul = ms.shard(mint.matmul, in_strategy=matmul_strategy)
            if not relu_strategy:
                self.relu = mint.nn.ReLU()
            else:
                self.relu = ms.shard(mint.nn.ReLU(), in_strategy=relu_strategy)

        def construct(self, x):
            out0 = self.add(x, self.gamma)
            out1 = self.relu(self.beta)
            out2 = self.matmul(out0, out1)
            return out2

    device_num = 8
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode="auto_parallel",
                                      search_mode="sharding_propagation")

    x = Tensor(np.ones([16, 16]), dtype=ms.float32)
    y = Tensor(np.ones([16, 16]), dtype=ms.float32)
    ly = ms.Layout((4, 2), ("axis0", "axis1"))
    in_layout = (ly("axis0", "axis1"),)
    gamma_layout = ly("axis0", "axis1")

    net = GradWrapTwoInput(NetWithLossSoftmax(Net(in_layout=in_layout, gamma_layout=gamma_layout, use_shard=True)))
    net.set_train()
    _cell_graph_executor.compile(net, x, y, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    context._reset_auto_parallel_context()
    for (k, v) in strategies.items():
        print("cnode: {} strategy: {}".format(k, v))
        if re.search('AddExt-op0', k) is not None:
            print("check AddExt-op0")
            assert v == [[4, 2], [4, 2]]
        elif re.search('MatMulExt-op0', k) is not None:
            print("check MatMulExt-op0")
            assert v == [[4, 2], [2, 1]]


def test_mint_relu_layout_propagate_back():
    """
    Feature: Sharding propagation for mint.nn.ReLU.
    Description: identity(2, 1)->relu
    Expectation: relu gets right strategy.
    """
    device_num = 8
    class MatMulReLUNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul = mint.matmul
            self.relu = mint.nn.ReLU()
            ly = ms.Layout((4, 2), ("axis0", "axis1"))
            self.matmul_shard = ms.shard(self.matmul, in_strategy=(ly("axis0", "axis1"), ly("axis1", "None")))

        def construct(self, input_x, input_y):
            out1 = self.relu(input_y)
            out2 = self.matmul_shard(input_x, out1)
            return out2

    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode="auto_parallel",
                                      search_mode="sharding_propagation")

    net = GradWrapTwoInput(NetWithLossTwoInput(MatMulReLUNet()))
    net.set_train()

    x = Tensor(np.ones([128, 96]), dtype=ms.float32)
    y = Tensor(np.ones([96, 96]), dtype=ms.float32)


    _cell_graph_executor.compile(net, x, y, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    context._reset_auto_parallel_context()
    for (k, v) in strategies.items():
        print("cnode: {} strategy: {}".format(k, v))
        if re.search('ReLU-op0', k) is not None:
            print("check ReLU-op0")
            assert v == [[2, 1]]
