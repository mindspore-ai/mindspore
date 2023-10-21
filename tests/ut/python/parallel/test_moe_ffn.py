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
from mindspore.ops.operations._inner_ops import MoeFFN
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

class MoeFFNNet(Cell):
    def __init__(self, strategy):
        super(MoeFFNNet, self).__init__()
        self.moe_ffn = MoeFFN("fastgelu").shard(strategy)

    def construct(self, x, expert_tokens, w1, bias1, w2):
        return self.moe_ffn(x, expert_tokens, w1, bias1, w2)


def test_moeffn_net():
    """
    Feature: test moe_ffn auto parallel
    Description: auto parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 1,), (1,), (1, 1, 8,), (1, 8,), (1, 8, 1,))
    net = MoeFFNNet(strategy)
    input_x = Tensor(np.ones([1024, 512]), dtype=ms.float16)
    expert_tokens = Tensor(np.ones([16]), dtype=ms.int64)
    weight1 = Parameter(Tensor(np.ones([16, 512, 2048]), dtype=ms.float16), "w1")
    bias1 = Parameter(Tensor(np.ones([16, 2048]), dtype=ms.float16), "bias1")
    weight2 = Parameter(Tensor(np.ones([16, 2048, 512]), dtype=ms.float16), "w2")
    net.set_inputs(input_x, expert_tokens, weight1, bias1, weight2)

    phase = compile_net(net, input_x, expert_tokens, weight1, bias1, weight2)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [16, 512, 256])
    assert validator.check_parameter_shape('bias1', [16, 256])
    assert validator.check_parameter_shape('w2', [16, 256, 512])
