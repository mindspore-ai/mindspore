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
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore as ms
from mindspore import context, Tensor
import numpy as np


class RotaryEmbedding(nn.Cell):
    def __init__(self, dim, base=10000, max_seq_len=2048):
        super(RotaryEmbedding, self).__init__()
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) * (1 / dim)))
        t = np.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1).astype(np.float16)
        self.cos = Tensor(np.cos(emb))
        self.sin = Tensor(np.sin(emb))
        self.apply_rotary_pos_emb = ops.ApplyRotaryPosEmb()

    def construct(self, query, key, position_ids):
        # get shape
        qshape = query.shape  # [batch, numHeadQ, seqLen, hiddenDim]
        kshape = key.shape  # [batch, numHeadK, seqLen, hiddenDim]

        # get query/key
        if qshape[2] != 1:
            if qshape[1] != 1:
                query = query.permute(0, 2, 1, 3)
            if kshape[1] != 1:
                key = key.permute(0, 2, 1, 3)
        query = query.reshape((qshape[0] * qshape[2], qshape[1] * qshape[3]))
        key = key.reshape((kshape[0] * kshape[2], kshape[1] * kshape[3]))

        # get cos/sin
        cos = ops.index_select(self.cos, axis=0, index=position_ids)
        sin = ops.index_select(self.sin, axis=0, index=position_ids)
        if qshape[2] != 1 and qshape[0] != 1:
            cos = ops.tile(cos, (qshape[0], 1))
            sin = ops.tile(sin, (qshape[0], 1))

        # get seqLen
        seqLen = ops.full([qshape[0]], qshape[2], dtype=ms.int32)

        # call kernel
        query_embed, key_embed = self.apply_rotary_pos_emb(query, key, cos, sin, seqLen)

        # get query_embed/key_embed
        if qshape[1] == 1 or qshape[2] == 1:
            query_embed = query_embed.reshape(qshape)
        else:
            query_embed = query_embed.reshape((qshape[0], qshape[2], qshape[1], qshape[3])).permute(0, 2, 1, 3)
        if kshape[1] == 1 or kshape[2] == 1:
            key_embed = key_embed.reshape(kshape)
        else:
            key_embed = key_embed.reshape((kshape[0], kshape[2], kshape[1], kshape[3])).permute(0, 2, 1, 3)
        return query_embed, key_embed

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return np.concatenate((-x2, x1), axis=-1)

    def CalTruth(self, query, key, position_ids):
        cos = self.cos.asnumpy()
        sin = self.sin.asnumpy()
        if query.shape[2] == 1:
            cos1 = np.expand_dims(cos[position_ids, :], axis=[1, 2])  # [bs, 1, 1, dim]
            sin1 = np.expand_dims(sin[position_ids, :], axis=[1, 2])  # [bs, 1, 1, dim]
        else:
            cos1 = np.expand_dims(cos[position_ids, :], axis=[0, 1])  # [1, 1, seq_len, dim]
            sin1 = np.expand_dims(sin[position_ids, :], axis=[0, 1])  # [1, 1, seq_len, dim]
        query_embed = (query * cos1) + (self.rotate_half(query) * sin1)
        key_embed = (key * cos1) + (self.rotate_half(key) * sin1)
        return query_embed, key_embed


def run(net, seqLen, batch, numHeadQ, numHeadK, hiddenDim, maxSeqLen):
    query = np.random.rand(batch, numHeadQ, seqLen, hiddenDim).astype(np.float16)
    key = np.random.rand(batch, numHeadK, seqLen, hiddenDim).astype(np.float16)
    if seqLen == 1:
        position_ids = np.random.randint(0, maxSeqLen, [batch], dtype=np.int64)
    else:
        position_ids = np.arange(seqLen).astype(np.int64)

    query_embed1, key_embed1 = net(Tensor(query), Tensor(key), Tensor(position_ids))
    query_embed1 = query_embed1.asnumpy()
    key_embed1 = key_embed1.asnumpy()
    print("query_embed1.shape", query_embed1.shape)
    print("key_embed1.shape", key_embed1.shape)

    query_embed2, key_embed2 = net.CalTruth(query, key, position_ids)
    print("query_embed2.shape", query_embed2.shape)
    print("key_embed2.shape", key_embed2.shape)

    np.testing.assert_allclose(query_embed1, query_embed2)
    np.testing.assert_allclose(key_embed1, key_embed2)
    print("run rope success")


def test_apply_rotary_pos_emb():
    """
    Feature: test ApplyRotaryPosEmb operator in graph mode.
    Description: test ApplyRotaryPosEmb.
    Expectation: the result is correct
    """
    hiddenDim = 128
    base = 10000
    maxSeqLen = 4096
    ms.set_context(device_target="Ascend", mode=context.GRAPH_MODE)
    net = RotaryEmbedding(hiddenDim, base, maxSeqLen)
    np.random.seed(0)
    run(net, 512, 1, 8, 2, hiddenDim, maxSeqLen)
    run(net, 10, 23, 8, 2, hiddenDim, maxSeqLen)
    run(net, 1, 4, 8, 2, hiddenDim, maxSeqLen)
    run(net, 1, 233, 8, 2, hiddenDim, maxSeqLen)
    run(net, 1024, 1, 11, 1, hiddenDim, maxSeqLen)
    run(net, 1, 4, 11, 1, hiddenDim, maxSeqLen)
