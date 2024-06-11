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
from mindspore.ops.operations._inner_ops import FFN
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

class FFNA16W8Net(Cell):
    def __init__(self, strategy):
        super().__init__()
        self.ffn = FFN('fastgelu', 1).shard(strategy)

    def construct(self, x, w1, w2, expert_tokens, bias1, bias2, antiquant_scale1,
                  antiquant_scale2, antiquant_offset1, antiquant_offset2):
        out = self.ffn(x, w1, w2, expert_tokens, bias1, bias2, None, None, None, None, antiquant_scale1,
                       antiquant_scale2, antiquant_offset1, antiquant_offset2)
        return out


class FFNA8W8Net(Cell):
    def __init__(self, strategy):
        super().__init__()
        self.ffn = FFN('fastgelu', 1).shard(strategy)

    def construct(self, x, w1, w2, expert_tokens, bias1, bias2, scale,
                  offset, dequant_scale1, dequant_scale2):
        out = self.ffn(x, w1, w2, expert_tokens, bias1, bias2, scale,
                       offset, dequant_scale1, dequant_scale2)
        return out

def test_ffn_a16w8():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((1, 1), (1, 1, 4), (1, 4, 1), (1,), (1, 4), (1, 1), (1, 4), (1, 1), (1, 4), (1, 1))

    net = FFNA16W8Net(strategy)

    hidden_size = 5120
    ffn_hidden_size = 2560
    expert_num = 8
    batch_seq = 5
    x = Parameter(Tensor(np.ones([batch_seq, hidden_size]), dtype=ms.float16), "x")
    w1 = Parameter(Tensor(np.ones([expert_num, hidden_size, ffn_hidden_size]), dtype=ms.int8), "w1")
    w2 = Parameter(Tensor(np.ones([expert_num, ffn_hidden_size, hidden_size]), dtype=ms.int8), "w2")
    expert_tokens = Parameter(Tensor([1, 1, 1, 1, 1, 0, 0, 0], dtype=ms.int64), "expert_tokens")
    bias1 = Parameter(Tensor(np.ones([expert_num, ffn_hidden_size]), dtype=ms.float16), "bias1")
    bias2 = Parameter(Tensor(np.ones([expert_num, hidden_size]), dtype=ms.float16), "bias2")
    antiquant_scale1 = Parameter(Tensor(np.ones([expert_num, ffn_hidden_size]), dtype=ms.float16), "antiquant_scale1")
    antiquant_scale2 = Parameter(Tensor(np.ones([expert_num, hidden_size]), dtype=ms.float16), "antiquant_scale2")
    antiquant_offset1 = Parameter(Tensor(np.ones([expert_num, ffn_hidden_size]), dtype=ms.float16), "antiquant_offset1")
    antiquant_offset2 = Parameter(Tensor(np.ones([expert_num, hidden_size]), dtype=ms.float16), "antiquant_offset2")
    net.set_inputs(x, w1, w2, expert_tokens, bias1, bias2, antiquant_scale1,
                   antiquant_scale2, antiquant_offset1, antiquant_offset2)

    phase = compile_net(net, x, w1, w2, expert_tokens, bias1, bias2, antiquant_scale1,
                        antiquant_scale2, antiquant_offset1, antiquant_offset2)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("x", [batch_seq, hidden_size])
    assert validator.check_parameter_shape("w1", [expert_num, hidden_size, ffn_hidden_size / 4])
    assert validator.check_parameter_shape("w2", [expert_num, ffn_hidden_size / 4, hidden_size])
    assert validator.check_parameter_shape("expert_tokens", [expert_num])
    assert validator.check_parameter_shape("bias1", [expert_num, ffn_hidden_size / 4])
    assert validator.check_parameter_shape("bias2", [expert_num, hidden_size])
    assert validator.check_parameter_shape("antiquant_scale1", [expert_num, ffn_hidden_size / 4])
    assert validator.check_parameter_shape("antiquant_scale2", [expert_num, hidden_size])
    assert validator.check_parameter_shape("antiquant_offset1", [expert_num, ffn_hidden_size / 4])
    assert validator.check_parameter_shape("antiquant_offset2", [expert_num, hidden_size])


def test_ffn_a8w8():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((1, 1), (1, 1, 4), (1, 4, 1), (1,), (1, 4), (1, 1), (1,), (1,), (1, 4), (1, 1))

    net = FFNA8W8Net(strategy)

    hidden_size = 5120
    ffn_hidden_size = 2560
    expert_num = 8
    batch_seq = 5
    x = Parameter(Tensor(np.ones([batch_seq, hidden_size]), dtype=ms.int8), "x")
    w1 = Parameter(Tensor(np.ones([expert_num, hidden_size, ffn_hidden_size]), dtype=ms.int8), "w1")
    w2 = Parameter(Tensor(np.ones([expert_num, ffn_hidden_size, hidden_size]), dtype=ms.int8), "w2")
    expert_tokens = Parameter(Tensor([1, 1, 1, 1, 1, 0, 0, 0], dtype=ms.int64), "expert_tokens")
    bias1 = Parameter(Tensor(np.ones([expert_num, ffn_hidden_size]), dtype=ms.int32), "bias1")
    bias2 = Parameter(Tensor(np.ones([expert_num, hidden_size]), dtype=ms.int32), "bias2")
    scale = Parameter(Tensor(np.ones([expert_num]), dtype=ms.float16), "scale")
    offset = Parameter(Tensor(np.ones([expert_num]), dtype=ms.float16), "offset")
    dequant_scale1 = Parameter(Tensor(np.ones([expert_num, ffn_hidden_size]), dtype=ms.uint64), "dequant_scale1")
    dequant_scale2 = Parameter(Tensor(np.ones([expert_num, hidden_size]), dtype=ms.uint64), "dequant_scale2")
    net.set_inputs(x, w1, w2, expert_tokens, bias1, bias2, scale,
                   offset, dequant_scale1, dequant_scale2)

    phase = compile_net(net, x, w1, w2, expert_tokens, bias1, bias2, scale,
                        offset, dequant_scale1, dequant_scale2)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("x", [batch_seq, hidden_size])
    assert validator.check_parameter_shape("w1", [expert_num, hidden_size, ffn_hidden_size / 4])
    assert validator.check_parameter_shape("w2", [expert_num, ffn_hidden_size / 4, hidden_size])
    assert validator.check_parameter_shape("expert_tokens", [expert_num])
    assert validator.check_parameter_shape("bias1", [expert_num, ffn_hidden_size / 4])
    assert validator.check_parameter_shape("bias2", [expert_num, hidden_size])
    assert validator.check_parameter_shape("scale", [expert_num])
    assert validator.check_parameter_shape("offset", [expert_num])
    assert validator.check_parameter_shape("dequant_scale1", [expert_num, ffn_hidden_size / 4])
    assert validator.check_parameter_shape("dequant_scale2", [expert_num, hidden_size])
