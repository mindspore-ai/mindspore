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

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.ops import composite as C
from mindspore.ops import operations as P
context.set_context(mode=context.GRAPH_MODE)
grad_by_list = C.GradOperation(get_by_list=True)

class Net(nn.Cell):
    def __init__(self, channel=1, w=0.25, strategy1=None, strategy2=None):
        super().__init__()
        self.norm = P.L2Normalize().shard(strategy1)
        self.prelu = P.PReLU().shard(strategy2)
        self.w = Parameter(initializer(w, [channel,]), name='w')

    def construct(self, data):
        x = self.norm(data)
        x = self.prelu(x, self.w)
        return x

class NetWithLoss(nn.Cell):
    def __init__(self, network, types, shapes, output_num, strategy3=None, strategy4=None, axis=-1):
        super(NetWithLoss, self).__init__()
        self.get_next = P.GetNext(types, shapes, output_num, "")
        self.one_hot = P.OneHot(axis=axis).shard(strategy3)
        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self.loss = P.SoftmaxCrossEntropyWithLogits().shard(strategy4)
        self.network = network

    def construct(self):
        data, label = self.get_next()
        predict = self.network(data)
        label = self.one_hot(label, 64, self.on_value, self.off_value)
        return self.loss(predict, label)[0]


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())

    def construct(self):
        return grad_by_list(self.network, self.weights)()


def compile_net(net):
    _cell_graph_executor.compile(net)

def test_get_next_single():
    """
    Feature: test get next ops
    Description: standalone, getnext-norm-prelu-loss.
    Expectation: compile well done.
    """
    net = GradWrap(NetWithLoss(Net(), [ms.float32, ms.int32], [[32, 64], [32]], 2))
    _cell_graph_executor.compile(net)

def test_get_next_semi_auto_parallel():
    """
    Feature: test get next ops
    Description: semi-auto parallel, getnext-norm-prelu-loss.
    Expectation: compile well done.
    """
    context.set_auto_parallel_context(device_num=4, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    network = Net(strategy1=((1, 4),), strategy2=((4, 1), (1,)))
    strategy3 = ((4, 1), (), ())
    strategy4 = ((4, 1), (4, 1))
    net_with_loss = NetWithLoss(network, [ms.float32, ms.int32], [[32, 64], [32]], 2, strategy3=strategy3,
                                strategy4=strategy4)
    net = GradWrap(net_with_loss)
    compile_net(net)


def test_get_next_semi_auto_parallel1():
    """
    Feature: test get next ops
    Description: semi-auto parallel, getnext-norm-prelu-loss.
    Expectation: compile well done.
    """
    context.set_auto_parallel_context(device_num=4, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    network = Net(strategy1=((1, 4),), strategy2=((4, 1), (1,)))
    strategy3 = ((1, 4), (), ())
    strategy4 = ((4, 1), (4, 1))
    net_with_loss = NetWithLoss(network, [ms.float32, ms.int32], [[32, 64], [32]], 2, strategy3=strategy3,
                                strategy4=strategy4)
    net = GradWrap(net_with_loss)
    compile_net(net)


def test_get_next_auto_parallel():
    """
    Feature: test get next ops
    Description: auto parallel, getnext-norm-prelu-loss.
    Expectation: compile well done.
    """
    context.set_auto_parallel_context(device_num=4, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    network = Net()
    net_with_loss = NetWithLoss(network, [ms.float32, ms.int32], [[32, 64], [32]], 2)
    net = GradWrap(net_with_loss)
    compile_net(net)


def test_only_one_get_next():
    """
    Feature: test get next ops
    Description: semi-auto parallel, only getnext.
    Expectation: compile well done.
    """
    class Net1(nn.Cell):
        def __init__(self):
            super().__init__()
            self.get_next = P.GetNext([ms.float32, ms.int32], [[32, 64], [32]], 2, "")

        def construct(self):
            return self.get_next()

    context.set_auto_parallel_context(device_num=4, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = Net1()
    net.set_train()
    compile_net(net)
