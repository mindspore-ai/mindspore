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
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops import operations as P
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class ReshapeNet(Cell):
    def __init__(self, weight, dst_shape, strategy1=None, strategy2=None):
        super().__init__()
        self.add = P.Add().shard(strategy1)
        self.weight = Parameter(weight, "w1")
        self.reshape = P.Reshape()
        self.relu = P.ReLU().shard(strategy2)
        self.dst_shape = dst_shape

    def construct(self, x):
        w = self.reshape(self.weight, self.dst_shape)
        out = self.add(x, w)
        out = self.relu(out)
        return out

class ReshapeStraNet(Cell):
    def __init__(self, weight, dst_shape, strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.add = P.Add().shard(strategy1)
        self.weight = Parameter(weight, "w1")
        self.reshape = P.Reshape().shard(strategy3)
        self.relu = P.ReLU().shard(strategy2)
        self.dst_shape = dst_shape

    def construct(self, x):
        w = self.reshape(self.weight, self.dst_shape)
        out = self.add(x, w)
        out = self.relu(out)
        return out

def test_reshape_parameter_scalar():
    """
    Feature: reshape the scalar parameter
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1), (1, 1))
    strategy2 = ((1, 1),)
    input_x = Tensor(np.ones([32, 64]), dtype=ms.float32)
    weight = Tensor(1, dtype=ms.float32)
    dst_shape = (1, 1)
    net = ReshapeNet(weight, dst_shape, strategy1, strategy2)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Add-0', ['Reshape-0'])
    assert validator.check_parameter_shape('w1', [])


def test_reshape_parameter_first_dim_can_not_div_by_dev_num():
    """
    Feature: reshape the parameter and its first dimension can not divisor by dev num
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1), (1, 1))
    strategy2 = ((1, 1),)
    input_x = Tensor(np.ones([32, 64]), dtype=ms.float32)
    weight = Tensor(np.ones([1]), dtype=ms.float32)
    dst_shape = (1, 1)
    net = ReshapeNet(weight, dst_shape, strategy1, strategy2)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Add-0', ['Reshape-0'])
    assert validator.check_parameter_shape('w1', [1])


def test_reshape_parameter_first_dim_can_div_by_dev_num():
    """
    Feature: reshape the parameter and its first dimension can divisor by dev num
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1), (1, 1))
    strategy2 = ((1, 1),)
    input_x = Tensor(np.ones([32, 64]), dtype=ms.float32)
    weight = Tensor(np.ones([32 * 64]), dtype=ms.float32)
    dst_shape = (32, 64)
    net = ReshapeNet(weight, dst_shape, strategy1, strategy2)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Add-0', ['AllGather-0'])
    assert validator.check_parameter_shape('w1', [4 * 64])

def test_reshape_with_stra_parameter_scalar():
    """
    Feature: reshape the scalar parameter while reshape is assigned with strategy
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1), (1, 1))
    strategy2 = ((1, 1),)
    strategy3 = ((1,),)
    input_x = Tensor(np.ones([32, 64]), dtype=ms.float32)
    weight = Tensor(1, dtype=ms.float32)
    dst_shape = (1, 1)
    net = ReshapeStraNet(weight, dst_shape, strategy1, strategy2, strategy3)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Add-0', ['Reshape-0'])
    assert validator.check_parameter_shape('w1', [])


def test_reshape_with_stra_parameter_first_dim_can_not_div_by_dev_num():
    """
    Feature: reshape the parameter and its first dimension can not divisor by dev num
             while reshape is assigned with strategy
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1), (1, 1))
    strategy2 = ((1, 1),)
    strategy3 = ((1,),)
    input_x = Tensor(np.ones([32, 64]), dtype=ms.float32)
    weight = Tensor(np.ones([1]), dtype=ms.float32)
    dst_shape = (1, 1)
    net = ReshapeStraNet(weight, dst_shape, strategy1, strategy2, strategy3)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Add-0', ['Reshape-0'])
    assert validator.check_parameter_shape('w1', [1])


def test_reshape_with_stra_parameter_first_dim_can_div_by_dev_num():
    """
    Feature: reshape the parameter and its first dimension can divisor by dev num
             while reshape is assigned with strategy
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1), (1, 1))
    strategy2 = ((1, 1),)
    strategy3 = ((4,),)
    input_x = Tensor(np.ones([32, 64]), dtype=ms.float32)
    weight = Tensor(np.ones([32 * 64]), dtype=ms.float32)
    dst_shape = (32, 64)
    net = ReshapeStraNet(weight, dst_shape, strategy1, strategy2, strategy3)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Add-0', ['AllGather-0'])
    assert validator.check_parameter_shape('w1', [8 * 64])
