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
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops.auto_generate import KVCacheScatterUpdate
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class ScattercNet(Cell):
    def __init__(self, strategy):
        super(ScattercNet, self).__init__()
        self.scatter = KVCacheScatterUpdate().shard(strategy)

    def construct(self, var, indices, update, axis=-1, reduce="update"):
        return self.scatter(var, indices, update, axis, reduce)

def test_scatter_4d():
    """
    Feature: test KVCacheScatterUpdate auto parallel
    Description: auto parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 8, 1, 1), (1,), (1, 8, 1, 1))
    net = ScattercNet(strategy)
    var_shape = [1, 64, 128, 128]
    var = Parameter(Tensor(np.random.uniform(low=1, high=10, size=var_shape).astype(np.float32)), "var")
    indices_shape = [1]
    indices = Tensor(np.random.randint(low=1, high=10, size=indices_shape).astype(np.int64))
    updates_shape = [1, 64, 128, 1]
    updates = Tensor(np.random.uniform(low=1, high=10, size=updates_shape).astype(np.float32))

    net.set_inputs(var, indices, updates)

    phase = compile_net(net, var, indices, updates)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('var', [1, 8, 128, 128])


def test_scatter_3d():
    """
    Feature: test KVCacheScatterUpdate auto parallel
    Description: auto parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 1, 8), (1,), (1, 1, 8))
    net = ScattercNet(strategy)
    var_shape = [1, 128, 128]
    var = Parameter(Tensor(np.random.uniform(low=1, high=10, size=var_shape).astype(np.float32)), "var")
    indices_shape = [1]
    indices = Tensor(np.random.randint(low=1, high=10, size=indices_shape).astype(np.int64))
    updates_shape = [1, 128, 128]
    updates = Tensor(np.random.uniform(low=1, high=10, size=updates_shape).astype(np.float32))

    net.set_inputs(var, indices, updates)

    phase = compile_net(net, var, indices, updates)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('var', [1, 128, 16])
