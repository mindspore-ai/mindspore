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
# ============================================================================
import os
import numpy as np
import pytest

import mindspore.ops as ops
import mindspore.nn as nn
import mindspore as ms
from mindspore import context, Tensor
from mindspore.common.np_dtype import bfloat16

def get_ms_dtype(query_dtype):
    if query_dtype == np.float32:
        ms_dtype = ms.float32
    elif query_dtype == np.float16:
        ms_dtype = ms.float16
    elif query_dtype == bfloat16:
        ms_dtype = ms.bfloat16
    return ms_dtype


class RotaryEmbedding(nn.Cell):
    # cosFormat=0  shape是[maxSeqLen, headDim]，    cos/sin不交替
    # cosFormat=1  shape是[maxSeqLen, headDim]，    cos/sin交替
    # cosFormat=2  shape是[batch*seqLen, headDim]， cos/sin不交替
    # cosFormat=3  shape是[batch*seqLen, headDim]， cos/sin交替
    def __init__(self, dim, base=10000, max_seq_len=2048, cos_dtype=np.float32, cos_format=0):
        super(RotaryEmbedding, self).__init__()
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) * (1 / dim)))
        t = np.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = np.outer(t, inv_freq)
        if cos_format == 0 or cos_format == 2:
            emb = np.concatenate((freqs, freqs), axis=-1)
        else:
            freqs = np.expand_dims(freqs, 2)
            emb = np.concatenate((freqs, freqs), axis=-1)
            emb = emb.reshape(max_seq_len, dim)
        self.cos_np = np.cos(emb).astype(cos_dtype)
        self.sin_np = np.sin(emb).astype(cos_dtype)
        self.cos = Tensor(np.cos(emb), dtype=get_ms_dtype(cos_dtype))
        self.sin = Tensor(np.sin(emb), dtype=get_ms_dtype(cos_dtype))
        self.apply_rotary_pos_emb = ops.ApplyRotaryPosEmb(cos_format)
        self.dim = dim
        self.cos_format = cos_format

    def construct(self, query, key, position_ids):
        if self.cos_format == 2 or self.cos_format == 3:
            batch, seq_len, _ = query.shape
            if seq_len == 1:
                freqs_cos = ops.gather(self.cos, position_ids, 0)
                freqs_sin = ops.gather(self.sin, position_ids, 0)
            else:
                freqs_cos = ops.tile(ops.gather(self.cos, position_ids, 0), (batch, 1))
                freqs_sin = ops.tile(ops.gather(self.sin, position_ids, 0), (batch, 1))
            query_embed, key_embed = self.apply_rotary_pos_emb(query, key, freqs_cos, freqs_sin, position_ids)
        else:
            query_embed, key_embed = self.apply_rotary_pos_emb(query, key, self.cos, self.sin, position_ids)
        return query_embed, key_embed

    def rotate_half1(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return np.concatenate((-x2, x1), axis=-1)

    def cal_truth_numpy(self, query, key, position_ids, query_dtype, cos_format):
        if query.shape[2] == 1:
            cos1 = np.expand_dims(self.cos_np[position_ids, :], axis=[1, 2])
            sin1 = np.expand_dims(self.sin_np[position_ids, :], axis=[1, 2])
        else:
            cos1 = np.expand_dims(self.cos_np[position_ids, :], axis=[0, 1])
            sin1 = np.expand_dims(self.sin_np[position_ids, :], axis=[0, 1])
        if cos_format == 1 or cos_format == 3:
            tmp_shape = cos1.shape
            cos1 = cos1.reshape((-1, tmp_shape[-1] // 2, 2)).transpose((0, 2, 1)).reshape(tmp_shape)
            sin1 = sin1.reshape((-1, tmp_shape[-1] // 2, 2)).transpose((0, 2, 1)).reshape(tmp_shape)
        query = query.astype(query_dtype).astype(np.float32)
        key = key.astype(query_dtype).astype(np.float32)
        cos1 = cos1.astype(query_dtype).astype(np.float32)
        sin1 = sin1.astype(query_dtype).astype(np.float32)
        query_embed = (query * cos1) + (self.rotate_half1(query) * sin1)
        key_embed = (key * cos1) + (self.rotate_half1(key) * sin1)
        query_embed = query_embed.astype(np.float32)
        key_embed = key_embed.astype(np.float32)
        return query_embed, key_embed

def run(net, seqLen, batch, num_head_q, num_head_k, hidden_dim, max_seq_len, query_dtype, pos_dtype, ndim=3,
        cos_format=0):
    if ndim == 3:
        query = np.random.rand(batch, seqLen, num_head_q * hidden_dim).astype(np.float32)
        key = np.random.rand(batch, seqLen, num_head_k * hidden_dim).astype(np.float32)
    else:
        query = np.random.rand(batch, seqLen, num_head_q, hidden_dim).astype(np.float32)
        key = np.random.rand(batch, seqLen, num_head_k, hidden_dim).astype(np.float32)
    if seqLen == 1:
        position_ids = np.random.randint(0, max_seq_len, [batch], dtype=pos_dtype)
    else:
        position_ids = np.arange(seqLen).astype(pos_dtype)
    query_tmp = Tensor(query, dtype=get_ms_dtype(query_dtype))
    key_tmp = Tensor(key, dtype=get_ms_dtype(query_dtype))
    position_ids_tmp = Tensor(position_ids)
    query_embed1, key_embed1 = net(query_tmp, key_tmp, position_ids_tmp)
    query_embed1 = query_embed1.astype(ms.float32).asnumpy()
    key_embed1 = key_embed1.astype(ms.float32).asnumpy()
    query1 = query.reshape((batch, seqLen, num_head_q, hidden_dim)).transpose((0, 2, 1, 3))
    key1 = key.reshape((batch, seqLen, num_head_k, hidden_dim)).transpose((0, 2, 1, 3))
    if cos_format == 1 or cos_format == 3:
        tmp_shape1, tmp_shape2 = query1.shape, key1.shape
        query1 = query1.reshape(-1, hidden_dim // 2, 2).transpose((0, 2, 1)).reshape(tmp_shape1)
        key1 = key1.reshape(-1, hidden_dim // 2, 2).transpose((0, 2, 1)).reshape(tmp_shape2)
    query_embed2, key_embed2 = net.cal_truth_numpy(query1, key1, position_ids, query_dtype, cos_format)
    query_embed2 = query_embed2.transpose((0, 2, 1, 3)).reshape(query.shape)
    key_embed2 = key_embed2.transpose((0, 2, 1, 3)).reshape(key.shape)
    if cos_format == 1 or cos_format == 3:
        tmp_shape1, tmp_shape2 = query_embed2.shape, key_embed2.shape
        query_embed2 = query_embed2.reshape(-1, 2, hidden_dim // 2).transpose((0, 2, 1)).reshape(tmp_shape1)
        key_embed2 = key_embed2.reshape(-1, 2, hidden_dim // 2).transpose((0, 2, 1)).reshape(tmp_shape2)
    np.testing.assert_allclose(query_embed1, query_embed2, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(key_embed1, key_embed2, rtol=1e-2, atol=1e-2)


def _test_rope(query_dtype, cos_dtype, cos_format, batch_size, seq_len, num_head):
    ndim = 3
    hidden_dim = 128
    base = 10000
    max_seq_len = 4096
    np.random.seed(0)
    if "ASCEND_HOME_PATH" not in os.environ:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    ms.set_context(device_target="Ascend", mode=context.GRAPH_MODE)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    net = RotaryEmbedding(hidden_dim, base, max_seq_len, cos_dtype, cos_format)
    run(net, seq_len, batch_size, num_head, num_head, hidden_dim, max_seq_len, query_dtype, np.int32, ndim, cos_format)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('query_dtype', [np.float16])
@pytest.mark.parametrize('cos_dtype', [np.float16, np.float32])
@pytest.mark.parametrize('cos_format', [2])
@pytest.mark.parametrize('batch_size', [16])
@pytest.mark.parametrize('seq_len', [256])
@pytest.mark.parametrize('num_head', [32])
def test_rope_float16(query_dtype, cos_dtype, cos_format, batch_size, seq_len, num_head):
    """
    Feature: test ROPE op in kbk enabling infer_boost
    Description: test ROPE op in float16.
    Expectation: the result is correct
    """
    _test_rope(query_dtype, cos_dtype, cos_format, batch_size, seq_len, num_head)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('query_dtype', [bfloat16])
@pytest.mark.parametrize('cos_dtype', [bfloat16, np.float32])
@pytest.mark.parametrize('cos_format', [2])
@pytest.mark.parametrize('batch_size', [16])
@pytest.mark.parametrize('seq_len', [256])
@pytest.mark.parametrize('num_head', [32])
def test_rope_bfloat16(query_dtype, cos_dtype, cos_format, batch_size, seq_len, num_head):
    """
    Feature: test ROPE op in kbk enabling infer_boost
    Description: test ROPE op in bfloat16.
    Expectation: the result is correct
    """
    _test_rope(query_dtype, cos_dtype, cos_format, batch_size, seq_len, num_head)
