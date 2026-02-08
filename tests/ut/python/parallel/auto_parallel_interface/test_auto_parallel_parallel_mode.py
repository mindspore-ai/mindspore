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

import numpy as np
import pytest
import math

from hccl_test.manage.api import Hccl
import mindspore as ms
from mindspore import ops, nn, context, Tensor
from mindspore.graph.api import _cell_graph_executor
from mindspore.common.initializer import initializer, HeUniform
from mindspore.nn.utils import no_init_parameters
from mindspore.parallel.auto_parallel import AutoParallel



class ParallelNetwork(nn.Cell):
    """ParallelNetwork"""

    def __init__(self, strategy=None):
        super().__init__()
        self.flatten = ops.Flatten()
        self.fc1_weight = ms.Parameter(initializer(HeUniform(math.sqrt(5)), shape=[
            16, 10], dtype=ms.float32), name="fc1")
        self.matmul1 = ops.MatMul().shard(strategy)
        self.relu1 = ops.ReLU()

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu1(x)
        return x


def init_hccl(global_rank, device_num):
    hccl = Hccl()
    hccl.rank_id = global_rank
    hccl.rank_size = device_num


def setup_function():
    context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")


def compile_net(parallel_mode, strategy):
    init_hccl(0, 8)
    with no_init_parameters():
        net = ParallelNetwork(strategy)
    parallel_net = AutoParallel(net, parallel_mode=parallel_mode)
    inputs = Tensor(np.random.randn(32, 16).astype(np.float32), ms.float32)
    _ = _cell_graph_executor.compile(parallel_net, inputs)


def test_valid_parallel_mode():
    """
    Feature: Test AutoParallel use valid parallel_mode "semi_auto"
    Description: Enable parallel mode "semi_auto", and shard tensor in model_parallel dimension
    Expectation: Compile success
    """
    parallel_mode = "semi_auto"
    strategy = ((1, 2), (2, 1))
    compile_net(parallel_mode, strategy)


def test_invalid_parallel_mode():
    """
    Feature: Test AutoParallel use invalid parallel_mode "stand_alone"
    Description: AutoParallel only supports "semi_auto", "sharding_propagation" and "recursive_programming"
    Expectation: Compile with error
    """
    parallel_mode = "stand_alone"
    strategy = None
    with pytest.raises(ValueError):
        compile_net(parallel_mode, strategy)
