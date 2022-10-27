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
import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from parallel.utils.utils import ParallelValidator

class Net(Cell):
    def __init__(self, strategy1, strategy2, split_tuple, param_shape, mul_weight_shape):
        super().__init__()
        self.gatherv2 = P.Gather().shard(strategy1)
        self.gatherv2.add_prim_attr("manual_split", split_tuple)
        self.mul = P.Mul().shard(strategy2)
        self.param = Parameter(initializer("ones", param_shape, ms.float32), name="gather_param")
        self.mul_weight = Parameter(initializer("ones", mul_weight_shape, ms.float32), name="mul_weight")

    def construct(self, x, b):
        out = self.gatherv2(self.param, x, 0)
        out = self.mul(out, self.mul_weight)
        return out


def compile_net(net, x):
    net.set_train()
    b = Tensor(np.ones([64, 8]), dtype=ms.float32)
    phase, _ = _cell_graph_executor.compile(net, x, b)
    context.reset_auto_parallel_context()
    return phase


def test_field_split_dim_1x1():
    """
    Feature: test field split
    Description: param dim is 1, indices dim is 1
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=1)
    strategy1 = ((8,), (8,))
    strategy2 = ((8,), (1,))
    split_tuple = (5, 6, 7, 8, 9, 10, 11, 8)
    param_shape = (64)
    mul_weight_shape = (1)
    x = Tensor(np.ones([16 // 8]), dtype=ms.int32)
    net = Net(strategy1, strategy2, split_tuple, param_shape, mul_weight_shape)
    phase = compile_net(net, x)

    validator = ParallelValidator(net, phase)
    # check layout, dev-matrix/tensor-map/slice_shape/field_size/uniform_split/opt_shard_group
    gather_param_layout = ([8], [0], [6], 0, False, '')
    assert validator.check_parameter_layout('gather_param', gather_param_layout)

    # check inputs
    sub_expect_inputs = ['TupleGetItem', 'value=5']
    assert validator.check_node_inputs_fuzzy_match('Sub-0', sub_expect_inputs)


def test_field_split_dim_2x1():
    """
    Feature: test field split
    Description: param dim is 2, indices dim is 1
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=2)
    strategy1 = ((4, 2), (4,))
    strategy2 = ((4, 2), (1,))
    split_tuple = (10, 20, 30, 4)
    param_shape = (64, 32)
    mul_weight_shape = (1)
    x = Tensor(np.ones([16 // 8]), dtype=ms.int32)
    net = Net(strategy1, strategy2, split_tuple, param_shape, mul_weight_shape)
    phase = compile_net(net, x)

    validator = ParallelValidator(net, phase)
    # check layout, dev-matrix/tensor-map/slice_shape/field_size/uniform_split/opt_shard_group
    gather_param_layout = ([4, 2], [1, 0], [20, 16], 0, False, '')
    assert validator.check_parameter_layout('gather_param', gather_param_layout)

    # check inputs
    sub_expect_inputs = ['Reshape', 'value=10']
    assert validator.check_node_inputs_fuzzy_match('Sub-0', sub_expect_inputs)


def test_field_split_dim_3x1():
    """
    Feature: test field split
    Description: param dim is 3, indices dim is 1
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=2)
    strategy1 = ((4, 2, 1), (4,))
    strategy2 = ((4, 2, 1), (1,))
    split_tuple = (10, 20, 30, 4)
    param_shape = (64, 32, 16)
    mul_weight_shape = (1)
    x = Tensor(np.ones([16 // 8]), dtype=ms.int32)
    net = Net(strategy1, strategy2, split_tuple, param_shape, mul_weight_shape)
    phase = compile_net(net, x)

    validator = ParallelValidator(net, phase)
    # check layout, dev-matrix/tensor-map/slice_shape/field_size/uniform_split/opt_shard_group
    gather_param_layout = ([4, 2], [1, 0, -1], [20, 16, 16], 0, False, '')
    assert validator.check_parameter_layout('gather_param', gather_param_layout)

    # check inputs
    sub_expect_inputs = ['Reshape', 'value=10']
    assert validator.check_node_inputs_fuzzy_match('Sub-0', sub_expect_inputs)
