# Copyright 2024 Huawei Technologies Co., Ltd
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

import mindspore.common.dtype as mstype
from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.ops.auto_generate.gen_ops_def import GroupNorm
from parallel.utils.utils import compile_net
from parallel.utils.utils import ParallelValidator


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class GroupNormDistNet(Cell):
    def __init__(self, strategy=None):
        super(GroupNormDistNet, self).__init__()
        if strategy:
            self.group_norm = GroupNorm().shard(strategy)
        else:
            self.group_norm = GroupNorm()
        self.eps = 1e-5

    def construct(self, x, num_groups, gamma, beta):
        out = self.group_norm(x, num_groups, gamma, beta, self.eps)[0]
        return out


def test_group_norm_parallel():
    """
    Feature: test GroupNorm parallel.
    Description: test GroupNorm parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel",
                                      device_num=4, global_rank=0)

    num_channels = 16
    batch_size = 8
    x = Tensor(np.random.rand(batch_size, num_channels, 2, 2).astype(np.float32))
    gamma = Tensor(np.ones(shape=(num_channels,)), dtype=mstype.float32)
    beta = Tensor(np.ones(shape=(num_channels,)), dtype=mstype.float32)
    num_groups = 4
    strategy = ((4, 1, 1, 1), (1,), (1,))
    net = GroupNormDistNet(strategy)
    phase = compile_net(net, x, num_groups, gamma, beta)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('GroupNorm-0', ['StridedSlice-0', num_groups])


def test_group_norm_parallel_input_with_rank3():
    """
    Feature: test GroupNorm parallel with input rank3(N,C,D).
    Description: test GroupNorm parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel",
                                      device_num=4, global_rank=0)

    num_channels = 16
    batch_size = 8
    x = Tensor(np.random.rand(batch_size, num_channels, 2).astype(np.float32))
    gamma = Tensor(np.ones(shape=(num_channels,)), dtype=mstype.float32)
    beta = Tensor(np.ones(shape=(num_channels,)), dtype=mstype.float32)
    num_groups = 4
    strategy = ((4, 1, 1), (1,), (1,))
    net = GroupNormDistNet(strategy)
    phase = compile_net(net, x, num_groups, gamma, beta)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('GroupNorm-0', ['StridedSlice-0', num_groups])


def test_group_norm_parallel_with_repeated_calc():
    """
    Feature: test GroupNorm parallel with repeated calculate.
    Description: test GroupNorm parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel",
                                      device_num=8, global_rank=0)

    num_channels = 16
    batch_size = 8
    x = Tensor(np.random.rand(batch_size, num_channels, 2, 2).astype(np.float32))
    gamma = Tensor(np.ones(shape=(num_channels,)), dtype=mstype.float32)
    beta = Tensor(np.ones(shape=(num_channels,)), dtype=mstype.float32)
    num_groups = 4
    strategy = ((4, 1, 1, 1), (1,), (1,))
    net = GroupNormDistNet(strategy)
    phase = compile_net(net, x, num_groups, gamma, beta)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('GroupNorm-0', ['StridedSlice-0', num_groups])
