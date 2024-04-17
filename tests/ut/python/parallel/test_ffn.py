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
from mindspore.ops.operations._inner_ops import FFN
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class FFNNet(Cell):
    def __init__(self, strategy):
        super(FFNNet, self).__init__()
        self.ffn = FFN("fastgelu", 1).shard(strategy)

    def construct(self, x, w1, w2, expert_tokens, bias1, bias2):
        if bias2 is not None:
            return self.ffn(x, w1, w2, expert_tokens, bias1, bias2)
        if bias1 is not None:
            return self.ffn(x, w1, w2, expert_tokens, bias1)
        if expert_tokens is not None:
            return self.ffn(x, w1, w2, expert_tokens)
        return self.ffn(x, w1, w2)


def test_ffn_net_with_moe_with_bias1_bias2():
    """
    Feature: test moe_ffn auto parallel
    Description: auto parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 1,), (1, 1, 8,), (1, 8, 1,), (1,), (1, 8,), (1, 1,))
    net = FFNNet(strategy)
    input_x = Tensor(np.ones([1024, 512]), dtype=ms.float16)
    weight1 = Parameter(Tensor(np.ones([16, 512, 2048]), dtype=ms.float16), "w1")
    weight2 = Parameter(Tensor(np.ones([16, 2048, 512]), dtype=ms.float16), "w2")
    expert_tokens = Tensor(np.ones([16]), dtype=ms.int64)
    bias1 = Parameter(Tensor(np.ones([16, 2048]), dtype=ms.float16), "bias1")
    bias2 = Parameter(Tensor(np.ones([16, 512]), dtype=ms.float16), "bias2")
    net.set_inputs(input_x, weight1, weight2, expert_tokens, bias1, bias2)

    phase = compile_net(net, input_x, weight1, weight2, expert_tokens, bias1, bias2)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [16, 512, 256])
    assert validator.check_parameter_shape('w2', [16, 256, 512])
    assert validator.check_parameter_shape('bias1', [16, 256])
    assert validator.check_parameter_shape('bias2', [16, 512])


def test_ffn_net_with_moe_with_bias1_bias2_large_x():
    """
    Feature: test moe_ffn auto parallel
    Description: auto parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 1, 1, 1, 1,), (1, 1, 8,), (1, 8, 1,), (1,), (1, 8,), (1, 1,))
    net = FFNNet(strategy)
    input_x = Tensor(np.ones([1, 2, 2, 21024, 512]), dtype=ms.float16)
    weight1 = Parameter(Tensor(np.ones([16, 512, 2048]), dtype=ms.float16), "w1")
    weight2 = Parameter(Tensor(np.ones([16, 2048, 512]), dtype=ms.float16), "w2")
    expert_tokens = Tensor(np.ones([16]), dtype=ms.int64)
    bias1 = Parameter(Tensor(np.ones([16, 2048]), dtype=ms.float16), "bias1")
    bias2 = Parameter(Tensor(np.ones([16, 512]), dtype=ms.float16), "bias2")
    net.set_inputs(input_x, weight1, weight2, expert_tokens, bias1, bias2)

    phase = compile_net(net, input_x, weight1, weight2, expert_tokens, bias1, bias2)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [16, 512, 256])
    assert validator.check_parameter_shape('w2', [16, 256, 512])
    assert validator.check_parameter_shape('bias1', [16, 256])
    assert validator.check_parameter_shape('bias2', [16, 512])


def test_ffn_net_with_moe_bias1():
    """
    Feature: test moe_ffn auto parallel
    Description: auto parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 1,), (1, 1, 8,), (1, 8, 1,), (1,), (1, 8,))
    net = FFNNet(strategy)
    input_x = Tensor(np.ones([1024, 512]), dtype=ms.float16)
    weight1 = Parameter(Tensor(np.ones([16, 512, 2048]), dtype=ms.float16), "w1")
    weight2 = Parameter(Tensor(np.ones([16, 2048, 512]), dtype=ms.float16), "w2")
    expert_tokens = Tensor(np.ones([16]), dtype=ms.int64)
    bias1 = Parameter(Tensor(np.ones([16, 2048]), dtype=ms.float16), "bias1")
    bias2 = None
    net.set_inputs(input_x, weight1, weight2, expert_tokens, bias1, bias2)

    phase = compile_net(net, input_x, weight1, weight2, expert_tokens, bias1, bias2)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [16, 512, 256])
    assert validator.check_parameter_shape('w2', [16, 256, 512])
    assert validator.check_parameter_shape('bias1', [16, 256])


def test_ffn_net_with_moe_bias2():
    """
    Feature: test moe_ffn auto parallel
    Description: auto parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 1,), (1, 1, 8,), (1, 8, 1,), (1,), (1, 1,))
    net = FFNNet(strategy)
    input_x = Tensor(np.ones([1024, 512]), dtype=ms.float16)
    weight1 = Parameter(Tensor(np.ones([16, 512, 2048]), dtype=ms.float16), "w1")
    weight2 = Parameter(Tensor(np.ones([16, 2048, 512]), dtype=ms.float16), "w2")
    expert_tokens = Tensor(np.ones([16]), dtype=ms.int64)
    bias1 = None
    bias2 = Parameter(Tensor(np.ones([16, 512]), dtype=ms.float16), "bias2")
    net.set_inputs(input_x, weight1, weight2, expert_tokens, bias1, bias2)

    phase = compile_net(net, input_x, weight1, weight2, expert_tokens, bias1, bias2)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [16, 512, 256])
    assert validator.check_parameter_shape('w2', [16, 256, 512])
    assert validator.check_parameter_shape('bias2', [16, 512])


def test_ffn_net_with_moe():
    """
    Feature: test moe_ffn auto parallel
    Description: auto parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 1,), (1, 1, 8,), (1, 8, 1,), (1,))
    net = FFNNet(strategy)
    input_x = Tensor(np.ones([1024, 512]), dtype=ms.float16)
    weight1 = Parameter(Tensor(np.ones([16, 512, 2048]), dtype=ms.float16), "w1")
    weight2 = Parameter(Tensor(np.ones([16, 2048, 512]), dtype=ms.float16), "w2")
    expert_tokens = Tensor(np.ones([16]), dtype=ms.int64)
    bias1 = None
    bias2 = None
    net.set_inputs(input_x, weight1, weight2, expert_tokens, bias1, bias2)

    phase = compile_net(net, input_x, weight1, weight2, expert_tokens, bias1, bias2)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [16, 512, 256])
    assert validator.check_parameter_shape('w2', [16, 256, 512])


def test_ffn_net_with_bias1_bias2():
    """
    Feature: test moe_ffn auto parallel
    Description: auto parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 1,), (1, 8,), (8, 1,), (8,), (1,))
    net = FFNNet(strategy)
    input_x = Tensor(np.ones([1024, 512]), dtype=ms.float16)
    weight1 = Parameter(Tensor(np.ones([512, 2048]), dtype=ms.float16), "w1")
    weight2 = Parameter(Tensor(np.ones([2048, 512]), dtype=ms.float16), "w2")
    expert_tokens = None
    bias1 = Parameter(Tensor(np.ones([2048]), dtype=ms.float16), "bias1")
    bias2 = Parameter(Tensor(np.ones([512]), dtype=ms.float16), "bias2")
    net.set_inputs(input_x, weight1, weight2, expert_tokens, bias1, bias2)

    phase = compile_net(net, input_x, weight1, weight2, expert_tokens, bias1, bias2)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [512, 256])
    assert validator.check_parameter_shape('w2', [256, 512])
    assert validator.check_parameter_shape('bias1', [256])
    assert validator.check_parameter_shape('bias2', [512])


def test_ffn_net_with_bias1():
    """
    Feature: test moe_ffn auto parallel
    Description: auto parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 1,), (1, 8,), (8, 1,), (8,))
    net = FFNNet(strategy)
    input_x = Tensor(np.ones([1024, 512]), dtype=ms.float16)
    weight1 = Parameter(Tensor(np.ones([512, 2048]), dtype=ms.float16), "w1")
    weight2 = Parameter(Tensor(np.ones([2048, 512]), dtype=ms.float16), "w2")
    expert_tokens = None
    bias1 = Parameter(Tensor(np.ones([2048]), dtype=ms.float16), "bias1")
    bias2 = None
    net.set_inputs(input_x, weight1, weight2, expert_tokens, bias1, bias2)

    phase = compile_net(net, input_x, weight1, weight2, expert_tokens, bias1, bias2)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [512, 256])
    assert validator.check_parameter_shape('w2', [256, 512])
    assert validator.check_parameter_shape('bias1', [256])


def test_ffn_net_with_bias2():
    """
    Feature: test moe_ffn auto parallel
    Description: auto parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 1,), (1, 8,), (8, 1,), (1,))
    net = FFNNet(strategy)
    input_x = Tensor(np.ones([1024, 512]), dtype=ms.float16)
    weight1 = Parameter(Tensor(np.ones([512, 2048]), dtype=ms.float16), "w1")
    weight2 = Parameter(Tensor(np.ones([2048, 512]), dtype=ms.float16), "w2")
    expert_tokens = None
    bias1 = None
    bias2 = Parameter(Tensor(np.ones([512]), dtype=ms.float16), "bias2")
    net.set_inputs(input_x, weight1, weight2, expert_tokens, bias1, bias2)

    phase = compile_net(net, input_x, weight1, weight2, expert_tokens, bias1, bias2)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [512, 256])
    assert validator.check_parameter_shape('w2', [256, 512])
    assert validator.check_parameter_shape('bias2', [512])


def test_ffn_net():
    """
    Feature: test moe_ffn auto parallel
    Description: auto parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 1,), (1, 8,), (8, 1,))
    net = FFNNet(strategy)
    input_x = Tensor(np.ones([1024, 512]), dtype=ms.float16)
    weight1 = Parameter(Tensor(np.ones([512, 2048]), dtype=ms.float16), "w1")
    weight2 = Parameter(Tensor(np.ones([2048, 512]), dtype=ms.float16), "w2")
    expert_tokens = None
    bias1 = None
    bias2 = None
    net.set_inputs(input_x, weight1, weight2, expert_tokens, bias1, bias2)

    phase = compile_net(net, input_x, weight1, weight2, expert_tokens, bias1, bias2)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [512, 256])
    assert validator.check_parameter_shape('w2', [256, 512])
