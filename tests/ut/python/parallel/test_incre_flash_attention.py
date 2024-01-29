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
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.context import set_auto_parallel_context
from mindspore.ops import operations as OPS
import mindspore.ops.operations.nn_ops as P


def setup_function():
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(dataset_strategy="full_batch")


def generate_inputs(
        B,
        N,
        S,
        D,
        num_key_value_heads,
        input_layout,
        attn_mask=None,
        actual_seq_lengths=None,
):
    assert input_layout == "BSH" or input_layout == "BNSD"
    q_shape = (B, 1, N * D) if input_layout == "BSH" else (B, N, 1, D)
    kv_shape = (
        (B, S, num_key_value_heads * D)
        if input_layout == "BSH"
        else (B, num_key_value_heads, S, D)
    )
    attn_mask_shape = (B, 1, S) if input_layout == "BSH" else (B, 1, 1, S)
    dtype = np.float16
    query = Tensor(np.ones(q_shape, dtype=dtype))
    key = Tensor(np.ones(kv_shape, dtype=dtype))
    value = Tensor(np.ones(kv_shape, dtype=dtype))
    attn_mask = Tensor(np.ones(attn_mask_shape, dtype=dtype)) if attn_mask else None
    actual_seq_lengths = (
        Tensor(np.ones((B,), dtype=np.int64)) if actual_seq_lengths else None
    )
    padding_mask = None
    return [query, key, value, padding_mask, attn_mask, actual_seq_lengths]


def compile_net(net, inputs):
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, *inputs)
    context.reset_auto_parallel_context()
    return phase


class Net(nn.Cell):
    def __init__(
            self,
            num_heads,
            scale_value=1.0,
            input_layout="BSH",
            num_key_value_heads=0,
            strategy=None,
    ):
        super(Net, self).__init__()
        self.fa_op = P.IncreFlashAttention(
            num_heads=num_heads,
            scale_value=scale_value,
            input_layout=input_layout,
            num_key_value_heads=num_key_value_heads,
        )
        stra_q = None
        if strategy:
            stra_q = (strategy[0],)
        self.square = OPS.Square().shard(stra_q)
        self.fa_op.shard(strategy)

    def construct(
            self,
            query,
            key,
            value,
            padding_mask=None,
            attn_mask=None,
            actual_seq_lengths=None,
            dequant_scale1=None,
            quant_scale1=None,
            dequant_scale2=None,
            quant_scale2=None,
            quant_offset2=None,
            antiquant_scale=None,
            antiquant_offset=None,
            block_table=None,
    ):
        out = self.fa_op(
            query,
            [key],
            [value],
            attn_mask,
            actual_seq_lengths,
            padding_mask,
            dequant_scale1,
            quant_scale1,
            dequant_scale2,
            quant_scale2,
            quant_offset2,
            antiquant_scale,
            antiquant_offset,
            block_table,
        )
        return self.square(out)


@pytest.mark.parametrize("input_layout", ["BSH", "BNSD"])
def test_self_attention_standalone(input_layout):
    """
    Feature: test increFlashAttention standalone
    Description: standalone
    Expectation: compile success
    """
    context.reset_auto_parallel_context()

    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    B, N, S, D, num_key_value_heads = 8, 18, 1, 57, 18
    inputs = generate_inputs(B, N, S, D, num_key_value_heads, input_layout)
    net = Net(N, input_layout=input_layout, num_key_value_heads=num_key_value_heads)
    compile_net(net, inputs)


@pytest.mark.parametrize("input_layout", ["BSH", "BNSD"])
@pytest.mark.parametrize("strategys", [(4, 2), (2, 2)])
def test_incre_flash_attention_semi_auto_parallel(input_layout, strategys):
    """
    Feature: test increFlashAttention semi parallel
    Description: semi parallel
    Expectation: compile success
    """
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    B, N, S, D, num_key_value_heads = 4, 32, 1024, 128, 16
    dp = strategys[0]
    mp = strategys[1]
    inputs = generate_inputs(
        B,
        N,
        S,
        D,
        num_key_value_heads,
        input_layout,
        attn_mask=True,
        actual_seq_lengths=True,
    )
    qkv_stra = (dp, mp, 1, 1) if input_layout == "BNSD" else (dp, 1, mp)
    attn_stra = (dp, 1, 1, 1) if input_layout == "BNSD" else (dp, 1, 1)
    strategies = (qkv_stra, qkv_stra, qkv_stra, attn_stra, (1,))
    net = Net(
        N,
        input_layout=input_layout,
        strategy=strategies,
        num_key_value_heads=num_key_value_heads,
    )
    compile_net(net, inputs)
