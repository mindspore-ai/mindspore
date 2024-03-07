# pylint: disable=C0330, C0326
#
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
"""
Test PagedAttentionMask plugin custom ops.
"""
import os
import math
import random
import logging
import numpy as np
import mindspore_lite as mslite
from mindspore import nn
from mindspore import Tensor, context, export
from mindspore.ops.auto_generate.gen_ops_prim import PagedAttentionMask

MAX_SEQ_LEN = 1024


class PagedAttentionMaskNet(nn.Cell):
    """
    A single op network of PagedAttentionMask.
    """

    def __init__(self):
        super().__init__()
        self.n_head_no_use = 40
        self.head_dim_no_use = 128
        self.scale_value_no_use = 1 / math.sqrt(self.head_dim_no_use)
        self.n_kv_head_no_use = 40
        self.paged_attention_mask = PagedAttentionMask(
            self.n_head_no_use, self.scale_value_no_use, self.n_kv_head_no_use
        )

    def construct(
        self, query, key_cache, value_cache, block_tables, context_lens, alibi_mask
    ):
        return self.paged_attention_mask(
            query, key_cache, value_cache, block_tables, context_lens, alibi_mask
        )


def export_model() -> str:
    """
    Export model with fixed shape.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    num_tokens = 2
    num_head = 32
    head_dim = 128
    kv_head = 16
    num_blocks = 64
    block_size = 128
    max_num_blocks_per_batch = 8

    q = Tensor(np.ones((num_tokens, num_head, head_dim), dtype=np.float16))
    key_cache = Tensor(
        np.ones((num_blocks, block_size, kv_head, head_dim), dtype=np.float16)
    )
    value_cache = Tensor(
        np.ones((num_blocks, block_size, kv_head, head_dim), dtype=np.float16)
    )
    block_tables = Tensor(
        np.ones((num_tokens, max_num_blocks_per_batch), dtype=np.int32)
    )
    context_len = Tensor(np.ones((num_tokens,), dtype=np.int32))
    alibi_mask = Tensor(
        np.ones((num_tokens, num_head, 1, max_num_blocks_per_batch), dtype=np.float16)
    )

    file_name = "paged_attention_mask"
    net = PagedAttentionMaskNet()
    export(
        net,
        q,
        key_cache,
        value_cache,
        block_tables,
        context_len,
        alibi_mask,
        file_name=file_name,
        file_format="MINDIR",
    )
    model_name = file_name + ".mindir"
    assert os.path.exists(model_name)
    return model_name


def group_matmul(head, kv_head, a, b):
    """
    Calculte a group(for all heads) of MatMul.
    """
    group_num = head // kv_head
    score = None
    for i in range(kv_head):
        group_score = np.matmul(
            a[i * group_num : (i + 1) * group_num, :, :].astype(np.float32),
            b[i : (i + 1), :, :].astype(np.float32),
        ).astype(np.float16)
        if score is None:
            score = group_score
        else:
            score = np.concatenate((score, group_score), 0)
    print(score.shape)
    return score


def ref_masked_attention(
    query,  # (1, num_heads, head_size)
    key,  # (context_len, kv_heads, head_size)
    value,
    scale: float,
    alibi_bias,
):
    """
    Implement masked attention with numpy.
    """
    # Q * K.T
    query = query * scale
    query = np.transpose(query, (1, 0, 2))  # 转置-> num_head, seqlen, head_size
    key = np.transpose(key, (1, 2, 0))  # 转置 -> kv_heads, head_size, context_len
    sim = group_matmul(query.shape[0], key.shape[0], query, key)
    sim = sim + alibi_bias

    # softmax
    row_max = np.max(sim, axis=-1, keepdims=True)
    sim -= row_max
    sim = sim.astype("float32")
    sim = np.exp(sim)
    row_sum = np.sum(sim, axis=-1, keepdims=True)
    p = sim / row_sum
    p = p.astype("float16")
    # P * V
    value = np.transpose(value, (1, 0, 2))  # 转置-> kv_heads, seqlen, head_size
    out = group_matmul(query.shape[0], key.shape[0], p, value)
    out = np.transpose(out, (1, 0, 2))  # 转置-> seqlen, num_head, head_size
    return out


def ref_single_query_cached_kv_attention(output, paged_input) -> None:
    """
    Implement single query attention with numpy.
    """
    query, key_cache, value_cache, block_tables, context_lens, alibi_mask = paged_input
    num_heads = query.shape[1]
    kv_heads = value_cache.shape[2]
    head_size = value_cache.shape[3]
    block_size = value_cache.shape[1]

    num_input_tokens = query.shape[0]
    for i in range(num_input_tokens):
        q = np.expand_dims(query[i], 0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        # 读取不同content_len的key和value，拼接在一起。
        keys = []
        values = []
        for j in range(context_len):  # 单个序列总的block个数
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, block_offset, :, :]
            k = k.reshape(kv_heads, head_size)
            keys.append(k)  # 读取key的内容

            v = value_cache[block_number, block_offset, :, :]
            v = v.reshape(kv_heads, head_size)
            values.append(v)  # 读取value的内容
        keys = np.stack(np.array(keys), axis=0)
        values = np.stack(np.array(values), axis=0)
        print(
            f"query.shape: {q.shape}, {q.dtype}, keys.shape: {keys.shape}, "
            f"context_len: {context_len}, keyblocknum: {(context_len + block_size - 1) // block_size}, "
            f"tail: {context_len % block_size}, alibi_bias.shape: {alibi_mask[i].shape}"
        )
        scale = 1.0 / (head_size**0.5)  # 1/sqrt(d)

        out = ref_masked_attention(
            q, keys, values, scale, alibi_mask[i, :, :, :context_len]
        )  # 计算attention

        out = out.reshape(num_heads, head_size)  # 2D输出
        output[i] = out


def create_golden_data(num_tokens=2, kv_heads=16, block_size=128, num_blocks=64):
    """
    Create golden data for PagedAttentionMask op.
    """
    num_heads = 32
    head_size = 128
    dtype = "float16"
    query = np.random.uniform(
        -1.0, 1.0, size=(num_tokens, num_heads, head_size)
    ).astype(dtype)

    # key value cache: (num_blocks, block_size, num_heads, head_size)
    key_cache = np.random.uniform(
        -1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size)
    ).astype(dtype)
    value_cache = np.random.uniform(
        -1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size)
    ).astype(dtype)

    context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_tokens)]
    # context_lens = [1024] * num_tokens # 每个batch对应的seqlen
    _ = [
        print(f"context_len: {x} % {block_size} == 1")
        for x in context_lens
        if x % block_size == 1
    ]
    max_context_len = max(context_lens)

    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []  # （num_tokens, max_num_blocks_per_seq）
    for i in range(num_tokens):
        n_block = (context_lens[i] + block_size - 1) // block_size
        print(f"n_block {i} = {n_block}")
        n_pad_block = max_num_blocks_per_seq - n_block
        block_table = [
            random.randint(0, num_blocks - 1)
            for _ in range(n_block)  # 给方块里面的每一个Block都分配了显存
        ]
        if n_pad_block != 0:
            block_table = block_table + ([-1] * n_pad_block)
        print(f"block table {i} = {block_table}")
        block_tables.append(block_table)

    context_lens = np.array(context_lens).astype(np.int32)
    block_tables = np.array(block_tables).astype(np.int32)

    # alibi mask
    alibi_slopes = np.random.random(num_heads).astype(np.float16)
    alibi_mask = np.zeros((num_tokens, num_heads, 1, max_context_len), dtype=np.float16)
    for i, context_len in enumerate(context_lens):
        position_ids = np.arange(context_len).astype(np.int32)
        alibi_bias = (position_ids - context_len + 1).astype(
            np.float16
        )  # -context_len+1, -context_len+2,..,0
        alibi_bias = alibi_slopes.reshape(-1, 1, 1) * alibi_bias.reshape(
            1, 1, -1
        )  # (head_num, 1, context)
        alibi_mask[i, :, :, :context_len] = alibi_bias
    print(f"alibi_mask.shape = {alibi_mask.shape}")

    paged_input = [
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        alibi_mask,
    ]
    ref_output = np.zeros_like(query)

    # 计算输出
    ref_single_query_cached_kv_attention(ref_output, paged_input)

    print(f"==> query shape: {query.shape}, data: \n{query}")
    print(f"==> key_cache shape: {key_cache.shape}")
    print(f"==> value_cache shape: {value_cache.shape}")
    print(f"==> block_tables shape: {block_tables.shape}, data: \n{block_tables}")
    print(f"==> context_lens shape: {context_lens.shape}, data: \n{context_lens}")
    print(f"==> alibi_mask shape: {alibi_mask.shape}, data: \n{alibi_mask}")
    print("data generate done!")
    ref_outputs = [ref_output]
    return paged_input, ref_outputs


def do_mslite_infer(model_file, in_tensors):
    """
    Do model inference with mslite.
    """
    print(model_file)
    lite_context = mslite.Context()
    lite_context.target = ["ascend"]
    lite_context.ascend.device_id = 2
    lite_context.ascend.provider = "ge"
    lite_context.ascend.rank_id = 0
    model = mslite.Model()

    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, "ascend_akg.ini")
    print(f"Use config file: {config_path}")
    model.build_from_file(
        model_file, mslite.ModelType.MINDIR, lite_context, config_path=config_path
    )

    outputs = model.predict(in_tensors)
    np_output: list[np.ndarray] = []
    for output in outputs:
        np_output.append(output.get_data_to_numpy())
        print("outputs' shape: ", np_output[-1].shape)
    print("finish------------------")
    return np_output


def inference_model(mindir_model: str):
    """
    Inference model.
    """
    inputs, ref_outputs = create_golden_data()

    # 运行昇腾算子
    in_tensors = [mslite.Tensor(x) for x in inputs]
    ascend_outputs = do_mslite_infer(mindir_model, in_tensors)

    for i, ascend_output in enumerate(ascend_outputs):
        is_close = np.allclose(ref_outputs[i], ascend_output, rtol=1e-3, atol=1e-03)
        logging.info("ref_outputs %d:\n%s", i, ref_outputs[i])
        logging.info("ascend_outputs %d:\n%s", i, ascend_output)
        logging.info("ascend output %d is equal to ref output: %s", i, is_close)
        assert is_close


def test_paged_attention_mask_fixed_shape():
    """
    Test PagedAttentionMask of fixed shape.
    """
    model_path = export_model()
    print(f"paged_attention_fixed_shape st : export success to path: {model_path}")
    logging.info(
        "paged_attention_mask_fixed_shape st : export success to path: %s", model_path
    )

    model_path = "paged_attention_mask.mindir"

    inference_model(model_path)
    print("paged_attention_mask_fixed_shape st : inference success.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
        filename="./test.log",
        filemode="w",
    )
    test_paged_attention_mask_fixed_shape()
