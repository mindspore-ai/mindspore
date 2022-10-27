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
from mindspore.common.api import _cell_graph_executor
from mindspore.common.parameter import Parameter
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

    def construct(self, x):
        predict = self.network(x)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x):
        return grad_all(self.network)(x)

def compile_net(net, input_data, dev_num=8, parallel_mode="semi_auto_parallel"):
    context.set_auto_parallel_context(device_num=dev_num, global_rank=0)
    context.set_auto_parallel_context(parallel_mode=parallel_mode)
    net.set_train()
    _cell_graph_executor.compile(net, input_data)

def test_reshape_unexpand():
    """
    Feature: distribute operator reshape which cannot do normal redistribution in semi auto parallel.
    Description: rreshape-weight net in auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reshape = P.Reshape()
            self.mul = P.Mul().shard(((1, 8), (1, 1, 8)))
            self.mul_weight = Parameter(Tensor(np.ones([96, 128]), dtype=ms.float32), name="weight")

        def construct(self, x):
            weight = self.reshape(self.mul_weight, (1, 128, 96))
            out = self.mul(x, weight)
            return out

    x = Tensor(np.ones([128, 96]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_net(net, x)

def test_reshape_unexpand_1():
    """
    Feature: distribute operator reshape which cannot do normal redistribution in semi auto parallel.
    Description: weight-reshape net in auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reshape = P.Reshape()
            self.mul = P.Mul().shard(((1, 1, 8), (1, 8)))
            self.mul_weight = Parameter(Tensor(np.ones([128, 96]), dtype=ms.float32), name="weight")

        def construct(self, data):
            x = self.reshape(self.mul_weight, (1, 128, 96))
            out = self.mul(x, self.mul_weight)
            return out

    x = Tensor(np.ones([128, 96]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_net(net, x)

def test_reshape_unexpand_2():
    """
    Feature: distribute operator reshape which cannot do normal redistribution in semi auto parallel.
    Description: weight-reshape net in auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reshape = P.Reshape()
            self.mul = P.Mul().shard(((1, 4, 2), (4, 2)))
            self.mul_weight = Parameter(Tensor(np.ones([128, 96]), dtype=ms.float32), name="weight")

        def construct(self, data):
            x = self.reshape(self.mul_weight, (1, 128, 96))
            out = self.mul(x, self.mul_weight)
            return out

    x = Tensor(np.ones([128, 96]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_net(net, x)

def test_reshape_unexpand_3():
    """
    Feature: distribute operator reshape which cannot do normal redistribution in semi auto parallel.
    Description: relu-reshape-relu net in auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reshape = P.Reshape()
            self.relu1 = P.ReLU().shard(((4, 1),))
            self.relu2 = P.ReLU().shard(((1, 4),))

        def construct(self, data):
            x = self.relu1(data)
            x = self.reshape(x, (3, 4))
            x = self.relu2(x)
            return x

    x = Tensor(np.ones([4, 3]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_net(net, x, dev_num=4)

def test_reshape_unexpand_4():
    """
    Feature: distribute operator reshape which cannot do normal redistribution in semi auto parallel.
    Description: relu-reshape-relu net in auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reshape = P.Reshape()
            self.relu1 = P.ReLU().shard(((4, 1),))
            self.relu2 = P.ReLU().shard(((1, 2, 2),))

        def construct(self, data):
            x = self.relu1(data)
            x = self.reshape(x, (3, 2, 2))
            x = self.relu2(x)
            return x

    x = Tensor(np.ones([4, 3]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_net(net, x, dev_num=4)

def test_reshape_unexpand_5():
    """
    Feature: distribute operator reshape which cannot do normal redistribution in semi auto parallel.
    Description: relu-reshape-relu net in auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reshape = P.Reshape()
            self.relu1 = P.ReLU().shard(((2, 2, 1),))
            self.relu2 = P.ReLU().shard(((1, 4),))

        def construct(self, data):
            x = self.relu1(data)
            x = self.reshape(x, (3, 4))
            x = self.relu2(x)
            return x

    x = Tensor(np.ones([2, 2, 3]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_net(net, x, dev_num=4)

def test_reshape_unexpand_6():
    """
    Feature: distribute operator reshape which cannot do normal redistribution in semi auto parallel.
    Description: weight-reshape net in auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reshape = P.Reshape()
            self.relu1 = P.ReLU().shard(((2, 1),))
            self.relu2 = P.ReLU().shard(((1, 1, 4),))

        def construct(self, data):
            x = self.relu1(data)
            x = self.reshape(x, (1, 3, 4))
            x = self.relu2(x)
            return x

    x = Tensor(np.ones([4, 3]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_net(net, x, dev_num=4)

def test_reshape_unexpand_7():
    """
    Feature: distribute operator reshape which cannot do normal redistribution in auto parallel.
    Description: reshape as net output in auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        def __init__(self, in_channel=3, out_channel=8, axis=1, input_shape=(32, 4, 110, -1),
                     mul_size=(32, 1, 220, 220)):
            super().__init__()
            mul_np = np.full(mul_size, 0.5, dtype=np.float32)
            self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
            self.mul = P.Mul()
            self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                  kernel_size=5, has_bias=True, weight_init='ones',
                                  bias_init='ones', pad_mode='valid')
            self.conv.conv2d.shard(((8, 1, 1, 1), (1, 1, 1, 1)))
            self.softmax = nn.Softmax(axis=axis)
            self.relu = nn.ReLU()
            self.reshape = P.Reshape()
            self.input_shape = input_shape

        def construct(self, inputs):
            x = self.conv(inputs)
            x = self.softmax(x)
            x = self.relu(x)
            x = self.mul(x, self.mul_weight)
            x = self.reshape(x, self.input_shape)
            return x

    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    x = Tensor(np.ones([32, 3, 224, 224]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_net(net, x, parallel_mode="auto_parallel")

def test_reshape_unexpand_8():
    """
    Feature: distribute operator reshape which cannot do normal redistribution in auto parallel.
    Description: weight-reshape net in auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reshape = P.Reshape()
            self.mul = P.Mul().shard(((1, 4, 2), (4, 2)))
            self.mul_weight = Parameter(Tensor(np.ones([128, 96]), dtype=ms.float32), name="weight")

        def construct(self, data):
            x = self.reshape(self.mul_weight, (1, 128, 96))
            out = self.mul(x, self.mul_weight)
            return out

    x = Tensor(np.ones([128, 96]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_net(net, x, parallel_mode="auto_parallel")
