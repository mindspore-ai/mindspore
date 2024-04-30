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
from mindspore.ops.auto_generate import MoeFinalizeRouting
from parallel.utils.utils import ParallelValidator, compile_net

# MoeFinalizeRouting has 7 inputs and 1 outputs (token_num is  bs*seq)
# expanded_x:            2D Tensor (token_num * top_k, hidden)
# skip1:                 2D Tensor (token_num, hidden)
# skip2(optional):       2D Tensor (token_num, hidden)
# bias:                  2D Tensor (expert_num, hidden)
# scales:                2D Tensor (token_num, top_k)
# expanded_row_idx:      1D Tensor (token_num * top_k)
# expanded_expert_idx:   2D Tensor (token_num, top_k)
# ------------------------------
# y:                     2D Tensor (token_num, hidden)

class MoeFinalizeRoutingNet(Cell):
    def __init__(self, strategy):
        super().__init__()
        self.moefzr = MoeFinalizeRouting().shard(strategy)

    def construct(self, x, skip1, skip2, bias, scale, expanded_src_to_dst, expert_for_source_row):
        out = self.moefzr(x, skip1, skip2, bias, scale, expanded_src_to_dst, expert_for_source_row)
        return out


def test_moe_finalize_routing_case0():
    """
    Feature: Test moe_finalize_routing auto parallel
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=1, global_rank=0)
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)

    top_k = 2
    token_num = 6
    hidden = 4
    expert_num = 8
    strategy = ((1, 1), (1, 1), (1, 1), (1, 1), (1,), (1, 1))
    moefzr_net = MoeFinalizeRoutingNet(strategy)

    # tensor input
    x = Tensor(np.ones([token_num * top_k, hidden]), dtype=ms.float16)
    skip1 = Tensor(np.ones([token_num, hidden]), dtype=ms.float16)
    skip2 = None
    bias = Parameter(Tensor(np.ones([expert_num, hidden]), dtype=ms.float16), "bias")
    scale = Tensor(np.ones([token_num, top_k]), dtype=ms.float16)
    expanded_src_to_dst = np.arange(token_num * top_k).astype(np.int32)
    np.random.shuffle(expanded_src_to_dst)
    expanded_src_to_dst = Tensor(expanded_src_to_dst, dtype=ms.int32)
    expert_for_source_row = Tensor(np.random.randint(low=0, high=expert_num, size=(token_num, top_k)).astype(np.int32))

    moefzr_net.set_inputs(x, skip1, skip2, bias, scale, expanded_src_to_dst, expert_for_source_row)
    phase = compile_net(moefzr_net, x, skip1, skip2, bias, scale, expanded_src_to_dst, expert_for_source_row)

    validator = ParallelValidator(moefzr_net, phase)
    assert validator.check_parameter_shape('bias', [expert_num, hidden])
