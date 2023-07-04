# Copyright 2023 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np
import mindspore as ms
import mindspore.context as context
from mindspore import Tensor, Parameter
import mindspore.nn as nn
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Momentum
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(nn.Cell):
    def __init__(self, weight1, strategy1=None, strategy2=None, is_parameter=True):
        super(Net, self).__init__()
        self.shape = (8, 48, 64)
        self.broadcast = P.BroadcastTo(self.shape).shard(strategy1)
        self.mul = P.Mul().shard(strategy2)
        if is_parameter:
            self.weight1 = Parameter(weight1, "w1")
        else:
            self.weight1 = weight1

    def construct(self, x):
        out = self.broadcast(self.weight1)
        out = self.mul(x, out)
        return out


class MatMulNet(nn.Cell):
    def __init__(self, weight1, strategy1=None, strategy2=None, strategy3=None, is_parameter=True):
        super(MatMulNet, self).__init__()
        self.shape = (8, 64, 64)
        self.broadcast = P.BroadcastTo(self.shape).shard(strategy1)
        self.matmul = P.BatchMatMul().shard(strategy2)
        self.mul = P.Mul().shard(strategy3)
        if is_parameter:
            self.weight1 = Parameter(weight1, "w1")
        else:
            self.weight1 = weight1

    def construct(self, x1, x2):
        out = self.broadcast(x2)
        out = self.matmul(x1, out)
        out = self.mul(out, self.weight1)
        return out


class TrainOneStep(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0, return_grad=False):
        super(TrainOneStep, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.return_grad = return_grad
        if return_grad:
            self.weights_name = [i.name for i in self.optimizer.parameters]
        self.reducer_flag = False
        self.grad_reducer = DistributedGradReducer(self.weights)

    def construct(self, *inputs):
        loss = self.network(*inputs)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        if self.return_grad:
            grad_with_param_name = {}
            for index, value in enumerate(grads):
                grad_with_param_name[self.weights_name[index]] = value
            return loss, grad_with_param_name
        return loss


_w1 = Tensor(np.ones([1, 48, 64]), dtype=ms.float32)
_x1 = Tensor(np.ones([8, 48, 64]), dtype=ms.float32)
_x2 = Tensor(np.ones([64, 64]), dtype=ms.float32)


def compile_net(net):
    context.set_context(mode=context.GRAPH_MODE)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStep(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x1)
    context.reset_auto_parallel_context()


def test_grad_reducer_in_semi_auto_parallel():
    """
    Feature: test grad reducer in semi_auto_parallel mode
    Description:
    Expectation: raise runtime error
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2),)
    strategy2 = ((1, 4, 2), (1, 4, 2))
    net = Net(_w1, strategy1, strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_grad_reducer_in_auto_parallel():
    """
    Feature: test grad reducer in auto_parallel mode
    Description:
    Expectation: raise runtime error
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2),)
    strategy2 = ((1, 4, 2), (1, 4, 2))
    net = Net(_w1, strategy1, strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)
