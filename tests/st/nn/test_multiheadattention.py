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
# ============================================================================

import numpy as np
import pytest

import mindspore as ms
from mindspore import context
from mindspore.nn import MultiheadAttention
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('dtype', [ms.float16, ms.float32, ms.float64])
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_multihead_attention_cpu_gpu(dtype, mode):
    """
    Feature: MultiheadAttention
    Description: Verify the result of MultiheadAttention
    Expectation: success
    """
    context.set_context(mode=mode)
    embed_dim, num_heads = 128, 8
    seq_length, batch_size = 10, 8
    query = ms.Tensor(np.random.randn(seq_length, batch_size, embed_dim), ms.float32)
    key = ms.Tensor(np.random.randn(seq_length, batch_size, embed_dim), ms.float32)
    value = ms.Tensor(np.random.randn(seq_length, batch_size, embed_dim), ms.float32)
    multihead_attn = MultiheadAttention(embed_dim, num_heads)
    attn_output, attn_output_weights = multihead_attn(query, key, value)
    assert attn_output.shape == (10, 8, 128)
    assert attn_output_weights.shape == (8, 10, 10)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [ms.float16, ms.float32])
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_multihead_attention_ascend(dtype, mode):
    """
    Feature: MultiheadAttention
    Description: Verify the result of MultiheadAttention
    Expectation: success
    """
    context.set_context(mode=mode)
    embed_dim, num_heads = 128, 8
    seq_length, batch_size = 10, 8
    query = ms.Tensor(np.random.randn(seq_length, batch_size, embed_dim), ms.float32)
    key = ms.Tensor(np.random.randn(seq_length, batch_size, embed_dim), ms.float32)
    value = ms.Tensor(np.random.randn(seq_length, batch_size, embed_dim), ms.float32)
    multihead_attn = MultiheadAttention(embed_dim, num_heads)
    attn_output, attn_output_weights = multihead_attn(query, key, value)
    assert attn_output.shape == (10, 8, 128)
    assert attn_output_weights.shape == (8, 10, 10)
