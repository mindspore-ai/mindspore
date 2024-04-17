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

import numpy as np
import mindspore as ms
from mindspore import context, Tensor
from mindspore.common.api import _cell_graph_executor
from mindspore.common.parameter import Parameter
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P
from mindspore.nn import Conv2d
from mindspore.nn import Dense
from mindspore.nn import Softmax
from mindspore.nn import ReLU
from mindspore.common.initializer import initializer
from mindspore.common import dtype as mstype

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class UnknownOp_Net(Cell):
    def __init__(self, seed=0, seed2=0):
        super().__init__()
        mul_np = np.full((1, 1), 0.1, dtype=np.float32)
        self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
        self.seed = seed
        self.seed2 = seed2
        self.cast = P.Cast()
        self.mul = P.Mul()
        self.random_choice_with_mask = P.RandomChoiceWithMask(count=256, seed=self.seed,
                                                              seed2=self.seed2)

    def construct(self, input_a, label):
        x, _ = self.random_choice_with_mask(input_a)
        x = self.cast(x, ms.float32)
        x = self.mul(x, self.mul_weight)
        return x

class ParallelStrategySearchNet(Cell):
    def __init__(self, in_channel,
                 out_channel,
                 axis,
                 input_shape,
                 mul_size,
                 test_size,
                 prelu_size,
                 transpose_b,
                 matmul_size,
                 num_class):
        super().__init__()
        mul_np = np.full(mul_size, 0.5, dtype=np.float32)
        self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
        bias_np = np.full((12,), 7.1, dtype=np.float32)
        self.bias = Parameter(Tensor(bias_np), name="bias")
        prelu_np = np.full(prelu_size, 0.8, dtype=np.float32)
        self.prelu_weight = Parameter(Tensor(prelu_np), name="prelu_weight")
        matmul_np = np.full(matmul_size, 1.1, dtype=np.float32)
        self.matmul_weight = Parameter(Tensor(matmul_np), name="matmul_weight")
        self.mul = P.Mul()
        self.conv = Conv2d(in_channels=in_channel, out_channels=out_channel,
                           kernel_size=5, has_bias=True,
                           weight_init='ones', bias_init='ones',
                           pad_mode='valid')
        self.scalar = 0.5
        self.parameter = Parameter(initializer(0.5, test_size, dtype=mstype.float32), name='parameter')
        self.tensor = Tensor(np.full(test_size, 0.05, dtype=np.float32))
        self.softmax = Softmax(axis=axis)
        self.relu = ReLU()
        self.relu.relu.add_prim_attr("primitive_target", "CPU")
        self.reshape = P.Reshape()
        self.input_shape = input_shape
        self.equal = P.Equal()
        self.cast = P.Cast()
        self.concat = P.Concat(axis=1)
        self.reduce_sum = P.ReduceSum()
        self.bias_add = P.BiasAdd()
        self.cos = P.Cos()
        self.prelu = P.PReLU()
        self.matmul = P.MatMul(transpose_b=transpose_b)
        self.l2norm = P.L2Normalize(axis=(1 - axis))
        self.tensoradd = P.TensorAdd()
        self.strided_slice = P.StridedSlice()
        self.dense = Dense(in_channels=6,
                           out_channels=num_class,
                           weight_init='ones',
                           bias_init='ones',
                           has_bias=True)

    def construct(self, inputs):
        x = self.conv(inputs)
        x = self.softmax(x)
        x = self.relu(x)
        x = self.mul(x, self.mul_weight)
        x = self.reshape(x, self.input_shape)
        y = self.parameter * self.tensor * self.scalar
        z = self.equal(self.parameter, self.scalar)
        z = self.cast(z, mstype.float16)
        z = self.cast(z, mstype.float32)
        x = self.concat((x, y, z))
        x = self.reduce_sum(x, (2, 3))
        x = self.bias_add(x, self.bias)
        y = self.cos(x)
        y = self.prelu(y, self.prelu_weight)
        z = self.matmul(x, self.matmul_weight)
        z = self.l2norm(z)
        x = self.tensoradd(y, z)
        x = self.strided_slice(x, (0, 0), (32, 6), (1, 1))
        x = self.dense(x)
        return x

def compile_unknownop_net(net):
    inputs_np = Tensor(np.ones([32, 4, 12]).astype(np.bool_))
    label_ = Tensor(np.random.randn(1, 1).astype(np.float32))
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, inputs_np, label_)
    context.reset_auto_parallel_context()

def compile_parallelstrategysearch_net(net):
    inputs_np = Tensor(np.random.randn(32, 3, 224, 224).astype(np.float32))
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, inputs_np)
    context.reset_auto_parallel_context()

def test_auto_parallel_unknownop_rec():
    """
    Feature: test unknownop net of auto parallel
    Description: using recursive algorithm
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="recursive_programming")
    net = UnknownOp_Net(1, 1)
    compile_unknownop_net(net)

def test_auto_parallel_parallelstrategysearch_net():
    """
    Feature: test parallelstrategysearch net of auto parallel
    Description: using recursive algorithm
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="recursive_programming")
    net = ParallelStrategySearchNet(
        in_channel=3,
        out_channel=8,
        axis=0,
        input_shape=(32, 4, 110, -1),
        mul_size=(32, 1, 220, 220),
        test_size=(32, 4, 110, 880),
        prelu_size=(12,),
        transpose_b=False,
        matmul_size=(12, 12),
        num_class=65536,
    )
    compile_parallelstrategysearch_net(net)
