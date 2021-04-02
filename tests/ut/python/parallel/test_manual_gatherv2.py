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

import numpy as np
import pytest
import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer

class Net(Cell):
    def __init__(self,
                 strategy1=None,
                 strategy2=None,
                 strategy3=None,
                 axis=0,
                 init_flag=True,
                 split_tuple=(4, 4),
                 split_string="manual_split",
                 param_shape=(8, 8)):
        super().__init__()
        self.gatherv2 = P.Gather().shard(strategy1)
        self.gatherv2.add_prim_attr(split_string, split_tuple)
        self.mul = P.Mul().shard(strategy2)
        self.reshape = P.Reshape()
        self.matmul = P.MatMul().shard(strategy3)
        self.matmul.add_prim_attr("forward_reduce_scatter", True)
        if init_flag:
            self.param = Parameter(initializer("ones", param_shape, ms.float32), name="gatherv2_param")
        else:
            self.param = Parameter(Tensor(np.ones(param_shape), dtype=ms.float32), name="gatherv2_param")
        self.mul_weight = Parameter(initializer("ones", (8, 8, 8), ms.float32), name="mul_weight")
        self.matmul_weight = Parameter(initializer("ones", (64, 16), ms.float32), name="matmul_weight")
        self.axis = axis

    def construct(self, x, b):
        out = self.gatherv2(self.param, x, self.axis)
        out = self.mul(out, self.mul_weight)
        out = self.reshape(out, (8, 64))
        out = self.matmul(out, self.matmul_weight)
        return out


_x = Tensor(np.ones([8, 8]), dtype=ms.int32)
_b = Tensor(np.ones([64, 8]), dtype=ms.float32)


def compile_net(net):
    context.set_context(save_graphs=False)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    train_net.set_train()
    _executor.compile(train_net, _x, _b, auto_parallel_mode=True)
    context.reset_auto_parallel_context()


def test_normal_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    strategy1 = ((2, 1), (1, 2))
    strategy2 = ((1, 2, 1), (1, 2, 1))
    strategy3 = ((1, 2), (2, 1))
    net = Net(strategy1, strategy2, strategy3)
    compile_net(net)


def test_normal_split2():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    strategy1 = ((4, 1), (1, 4))
    strategy2 = ((1, 4, 1), (1, 4, 1))
    strategy3 = ((1, 4), (4, 1))
    net = Net(strategy1, strategy2, strategy3, split_tuple=(10, 20, 30, 4), param_shape=(64, 8))
    compile_net(net)


def test_normal_split3():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=17)
    strategy1 = ((4, 8), (1, 4))
    strategy2 = ((1, 4, 8), (1, 4, 8))
    strategy3 = ((1, 32), (32, 1))
    net = Net(strategy1, strategy2, strategy3, split_tuple=(10, 20, 30, 4), param_shape=(64, 8))
    compile_net(net)


def test_normal_split_with_offset():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    strategy1 = ((2, 1), (1, 2))
    strategy2 = ((1, 2, 1), (1, 2, 1))
    strategy3 = ((1, 2), (2, 1))
    net = Net(strategy1, strategy2, strategy3, split_string="manual_split_with_offset", split_tuple=((4, 0), (4, 4)))
    compile_net(net)


def test_auto_parallel_error():
    context.set_context(save_graphs=False)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=2, global_rank=0)
    net = Net()
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_axis_error():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    strategy1 = ((2, 1), (1, 2))
    strategy2 = ((1, 2, 1), (1, 2, 1))
    strategy3 = ((1, 2), (2, 1))
    net = Net(strategy1, strategy2, strategy3, axis=1)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_strategy_error():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((4, 1), (8, 1))
    strategy2 = ((1, 2, 1), (1, 2, 1))
    strategy3 = ((1, 2), (2, 1))
    net = Net(strategy1, strategy2, strategy3)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_strategy_error2():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((4, 1), (1, 8))
    strategy2 = ((1, 2, 1), (1, 2, 1))
    strategy3 = ((1, 2), (2, 1))
    net = Net(strategy1, strategy2, strategy3)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_strategy_error3():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 1), (1, 2))
    strategy2 = ((1, 2, 1), (1, 2, 1))
    strategy3 = ((1, 2), (2, 1))
    net = Net(strategy1, strategy2, strategy3)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_strategy_error4():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    strategy1 = ((2, 8), (1, 2))
    strategy2 = ((1, 2, 1), (1, 2, 1))
    strategy3 = ((1, 2), (2, 1))
    net = Net(strategy1, strategy2, strategy3)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_strategy_error5():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    strategy1 = ((4, 1), (1, 4))
    strategy2 = ((1, 2, 1), (1, 2, 1))
    strategy3 = ((1, 2), (2, 1))
    net = Net(strategy1, strategy2, strategy3)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_split_tuple_error():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    strategy1 = ((2, 1), (1, 2))
    strategy2 = ((1, 2, 1), (1, 2, 1))
    strategy3 = ((1, 2), (2, 1))
    net = Net(strategy1, strategy2, strategy3, split_tuple=((5, 0), (5, 5)))
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_parameter_use_tensor_error():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    strategy1 = ((2, 1), (1, 2))
    strategy2 = ((1, 2, 1), (1, 2, 1))
    strategy3 = ((1, 2), (2, 1))
    net = Net(strategy1, strategy2, strategy3, init_flag=False)
    with pytest.raises(RuntimeError):
        compile_net(net)
