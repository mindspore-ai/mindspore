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
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.context import set_auto_parallel_context
import mindspore.ops.operations.nn_ops as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


def generate_inputs(B, N, S, D, input_layout='BSH'):
    padding_mask = None
    actual_seq_lengths = None
    ret_inputs = None
    if input_layout == 'BSH':
        H = N * D
        query = Tensor(np.ones((B, S, H), dtype=np.float16))
        key = Tensor(np.ones((B, S, H), dtype=np.float16))
        value = Tensor(np.ones((B, S, H), dtype=np.float16))
        attn_mask = Tensor(np.ones((B, 1, S, S), dtype=np.float16))
        ret_inputs = (query, key, value, attn_mask, padding_mask, actual_seq_lengths)
    elif input_layout == 'BNSD':
        query = Tensor(np.ones((B, N, S, D), dtype=np.float16))
        key = Tensor(np.ones((B, N, S, D), dtype=np.float16))
        value = Tensor(np.ones((B, N, S, D), dtype=np.float16))
        attn_mask = Tensor(np.ones((B, 1, S, S), dtype=np.float16))
        ret_inputs = (query, key, value, attn_mask, padding_mask, actual_seq_lengths)
    else:
        print("unsupported input layout ", input_layout)
    return ret_inputs


def compile_net(net, *inputs):
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, *inputs)
    context.reset_auto_parallel_context()
    return phase


class Net(nn.Cell):
    def __init__(self, num_heads, scale_value=1.0, pre_tokens=2147483547, next_tokens=0, input_layout='BSH',
                 num_key_value_heads=0, dp=None, mp=None, strategy=None):
        super(Net, self).__init__()
        self.fa_op = P.PromptFlashAttention(num_heads=num_heads, scale_value=scale_value, pre_tokens=pre_tokens,
                                            next_tokens=next_tokens, input_layout=input_layout,
                                            num_key_value_heads=num_key_value_heads)
        stra = strategy
        if dp is not None and mp is not None:
            if input_layout == 'BSH':
                stra = ((dp, 1, mp), (dp, 1, mp), (dp, 1, mp), (dp, 1, 1, 1))
            if input_layout == 'BNSD':
                stra = ((dp, mp, 1, 1), (dp, mp, 1, 1), (dp, mp, 1, 1), (dp, 1, 1, 1))
        self.fa_op.shard(stra)

    def construct(self, query, key, value, attn_mask, padding_mask, actual_seq_lengths):
        return self.fa_op(query, key, value, attn_mask, None, actual_seq_lengths)


@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD'])
def test_self_attention_standalone(input_layout):
    """
    Feature: test PromptFlashAttention standalone
    Description: standalone
    Expectation: compile success
    """
    context.reset_auto_parallel_context()
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    B, N, S, D = 8, 16, 1024, 128
    inputs = generate_inputs(B, N, S, D, input_layout=input_layout)
    net = Net(N, input_layout=input_layout)
    compile_net(net, *inputs)


@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD'])
def test_prompt_flash_attention_semi_auto_parallel(input_layout):
    """
    Feature: test PromptFlashAttention semi parallel
    Description: semi parallel
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    dp = 2
    mp = 4
    B, N, S, D = 8, 16, 1024, 128
    inputs = generate_inputs(B, N, S, D, input_layout=input_layout)
    net = Net(N, dp=dp, mp=mp, input_layout=input_layout)
    compile_net(net, *inputs)


@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD'])
def test_prompt_flash_attention_auto_parallel(input_layout):
    """
    Feature: test PromptFlashAttention auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=8,
                                      global_rank=0)
    dp = 2
    mp = 4
    B, N, S, D = 8, 16, 1024, 128
    inputs = generate_inputs(B, N, S, D, input_layout=input_layout)
    net = Net(N, dp=dp, mp=mp, input_layout=input_layout)
    compile_net(net, *inputs)


@pytest.mark.parametrize('input_layout', ['BSH', 'BNSD'])
def test_prompt_flash_attention_strategy_error(input_layout):
    """
    Feature: test invalid strategy for PromptFlashAttention
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((1, 2, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1, 1))
    B, N, S, D = 8, 16, 1024, 128
    net = Net(N, strategy=strategy, input_layout=input_layout)
    with pytest.raises(RuntimeError):
        inputs = generate_inputs(B, N, S, D, input_layout=input_layout)
        compile_net(net, *inputs)
    context.reset_auto_parallel_context()
