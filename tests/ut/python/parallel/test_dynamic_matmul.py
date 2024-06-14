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

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops.auto_generate import MatMulExt
from parallel.utils.utils import ParallelValidator, compile_net


class MatMulExtNet(Cell):
    def __init__(self, strategy1=None, strategy2=None):
        super().__init__()
        self.w = Parameter(Tensor(np.ones([8, 16]), dtype=ms.float32), "w1")
        self.matmul = MatMulExt().shard(strategy1)
        self.relu = P.ReLU().shard(strategy2)
    def construct(self, x):
        out = self.matmul(x, self.w)
        out = self.relu(out)
        return out


def test_matmul_ext_dynamic():
    """
    Feature: test dynamic shape for matmul ext
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((2, 1),)
    net = MatMulExtNet(strategy1, strategy2)
    input_x = Tensor(shape=[None, 8], dtype=ms.float32)
    net.set_inputs(input_x)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('ReLU-0', ['AllReduce-0'])
    assert validator.check_node_inputs('AllReduce-0', ['MatMulExt-0'])
    assert validator.check_parameter_shape("w1", [2, 16])
