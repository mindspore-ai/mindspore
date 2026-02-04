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

import re
import os
import numpy as np
import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore.graph.api import _cell_graph_executor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore.nn.utils import no_init_parameters
from mindspore.common.initializer import initializer
from hccl_test.manage.api import Hccl


def test_save_and_load_operator_strategy_file():
    """
    Feature: test AutoParallel(cell).save_operator_strategy_file(file_path) and
             AutoParallel(cell).load_operator_strategy_file(file_path)
    Description: the sharding strategy would be saved or loaded.
    Expectation: when using AutoParallel(cell).save_operator_strategy_file(file_path), the config json
             file requires to be generated;
    when using AutoParallel(cell).load_operator_strategy_file(file_path) the strategy requires to
             be loaded the same as the SAVEd strategy .
    """

    class NetForSaveAndLoad(Cell):
        def __init__(self, w1_size, w2_size, in_strategy1, in_strategy2):
            super().__init__()
            self.matmul1 = P.MatMul(transpose_b=True).shard(
                in_strategy=in_strategy1)
            self.matmul2 = P.MatMul(transpose_b=True).shard(
                in_strategy=in_strategy2)
            self.add1 = P.Add()
            self.add2 = P.Add()
            self.add3 = P.Add()
            self.mul_weight1 = Parameter(initializer("ones", w1_size), "w1")
            self.mul_weight2 = Parameter(initializer("ones", w2_size), "w2")

        def construct(self, x, b1, b2, b3):
            out = self.add1(x, b1)
            out = self.matmul1(out, self.mul_weight1)
            out = self.add2(out, b2)
            out = self.matmul2(out, self.mul_weight2)
            out = self.add3(out, b3)
            return out

    def compile_and_get_strategies(in_strategy1, in_strategy2, mode):
        x = Tensor(np.ones([64, 32]), dtype=ms.float32)
        b1 = Tensor(np.ones([64, 32]), dtype=ms.float32)
        w1_size = [8, 32]
        b2 = Tensor(np.ones([64, 8]), dtype=ms.float32)
        w2_size = [16, 8]
        b3 = Tensor(np.ones([64, 16]), dtype=ms.float32)

        with no_init_parameters():
            net = NetForSaveAndLoad(w1_size, w2_size, in_strategy1, in_strategy2)

        net.set_train()
        parallel_net = AutoParallel(net, parallel_mode='sharding_propagation')
        if mode == "save":
            parallel_net.save_operator_strategy_file("/tmp/strategy.json")
        if mode == "load":
            parallel_net.load_operator_strategy_file("/tmp/strategy.json")
        _cell_graph_executor.compile(
            parallel_net, x, b1, b2, b3, phase='train')
        strategies = _cell_graph_executor._get_shard_strategy(parallel_net)
        return strategies

    def assert_sharding_strategy(dp1, mp1, dp2, mp2, strategies):
        for (k, v) in strategies.items():
            if re.search('Add-op0', k) is not None:
                assert v == [[dp2, mp2], [dp2, mp2]]
            if re.search('Add-op1', k) is not None:
                assert v == [[dp2, mp2], [dp2, mp2]]
            if re.search('Add-op2', k) is not None:
                assert v == [[dp1, mp1], [dp1, mp1]]
            if re.search('MatMul-op0', k) is not None:
                assert v == [[dp2, mp2], [1, mp2]]
            if re.search('MatMul-op1', k) is not None:
                assert v == [[dp1, mp1], [1, mp1]]

    hccl = Hccl
    hccl.rank_id = 0
    hccl.rank_size = 8
    _dp1 = 4
    _mp1 = 2
    _dp2 = 2
    _mp2 = 4

    _in_strategy1 = ((_dp1, _mp1), (1, _mp1))
    _in_strategy2 = ((_dp2, _mp2), (1, _mp2))
    if os.path.exists("/tmp/strategy.json"):
        os.remove("/tmp/strategy.json")
    _strategies = compile_and_get_strategies(
        _in_strategy1, _in_strategy2, "save")
    _strategies = compile_and_get_strategies(
        _in_strategy1, _in_strategy2, "save")
    assert os.path.exists("/tmp/strategy.json")
    assert_sharding_strategy(_dp1, _mp1, _dp2, _mp2, _strategies)
    _in_strategy1 = None
    _in_strategy2 = None
    _strategies = compile_and_get_strategies(
        _in_strategy1, _in_strategy2, "load")
    assert_sharding_strategy(_dp1, _mp1, _dp2, _mp2, _strategies)
    os.remove("/tmp/strategy.json")
