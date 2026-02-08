# Copyright 2025 Huawei Technologies Co., Ltd
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

"""test inplace augassign op."""

import pytest
from mindspore import nn
from mindspore import Tensor, context
from mindspore.graph.api import _cell_graph_executor
from mindspore.ops import composite as C
from tests.ut.python.ops.test_math_ops import VirtualLoss

grad_all = C.GradOperation(get_all=True)


def compile_net(net, x):
    net.set_train()
    _cell_graph_executor.compile(net, x)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x,):
        predict = self.network(x)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def construct(self, x):
        return grad_all(self.network)(x)


class InplaceAddExtNet(nn.Cell):
    def construct(self, x):
        x += Tensor(2.0)
        return x


class InplaceAddsExtNet(nn.Cell):
    def construct(self, x):
        x += 2.0
        return x


class InplaceSubExtNet(nn.Cell):
    def construct(self, x):
        x -= Tensor(2.0)
        return x


class InplaceSubScalarNet(nn.Cell):
    def construct(self, x):
        x -= 2.0
        return x


class InplaceMulNet(nn.Cell):
    def construct(self, x):
        x *= Tensor(2.0)
        return x


class InplaceMulsNet(nn.Cell):
    def construct(self, x):
        x *= 2.0
        return x


class InplaceDivNet(nn.Cell):
    def construct(self, x):
        x /= Tensor(2.0)
        return x


class InplaceDivsNet(nn.Cell):
    def construct(self, x):
        x /= 2.0
        return x


class InplaceFloorDivideNet(nn.Cell):
    def construct(self, x):
        x //= Tensor(2.0)
        return x


class InplaceFloorDividesNet(nn.Cell):
    def construct(self, x):
        x //= 2.0
        return x


@pytest.mark.parametrize("network",
                         [InplaceAddExtNet, InplaceAddsExtNet, InplaceSubExtNet, InplaceSubScalarNet, InplaceMulNet,
                          InplaceMulsNet, InplaceDivNet, InplaceDivsNet, InplaceFloorDivideNet, InplaceFloorDividesNet])
def test_inplace_augassign(network):
    """
    Feature: distribute augassign operators in auto parallel.
    Description: augassign net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

    context.set_auto_parallel_context(device_num=64, global_rank=15)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = GradWrap(NetWithLoss(network()))

    x = Tensor(5.0)
    compile_net(net, x)
