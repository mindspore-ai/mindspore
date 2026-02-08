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
"""Test FlashAttentionScore's precision with auto parallsel"""
import pytest
import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.ops as P
from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.ops.composite import GradOperation
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspore import Layout
from mindspore.communication.management import init


class Net(Cell):
    """Define flash_attention_score net."""
    def __init__(self, head_num, keep_prob=0.9, input_layout="BSH", sparse_mode=0, use_mqa=False,
                 with_real_shift=True, dp=None, mp=None, sp=1):
        super().__init__()
        self.reshape = P.Reshape()
        self.drop_gen_mask = P.DropoutGenMask()
        self.keep_prob = Tensor(keep_prob, ms.float16)
        compressed_mask_mode = [2, 3, 4, 5, 6, 7, 8]
        self.head_num = head_num
        self.input_layout = input_layout
        pre_tokens = 2147483647 if sparse_mode not in compressed_mask_mode else 512
        next_tokens = 2147483647 if sparse_mode not in compressed_mask_mode else 0
        self.fa_op = FlashAttentionScore(head_num=head_num,
                                         keep_prob=keep_prob,
                                         pre_tokens=pre_tokens,
                                         next_tokens=next_tokens,
                                         input_layout=input_layout,
                                         sparse_mode=sparse_mode)
        self.attn_mask = Tensor(np.ones((2048, 2048), dtype=np.uint8))
        if dp is not None and mp is not None:
            kv_head_stra = 1 if use_mqa else mp
            if input_layout == "BSH":
                stra = ((dp, sp, mp), (dp, 1, kv_head_stra), (dp, 1, kv_head_stra))
            elif input_layout == "SBH":
                stra = ((sp, dp, mp), (1, dp, kv_head_stra), (1, dp, kv_head_stra))
            elif input_layout == "BNSD":
                stra = ((dp, mp, sp, 1), (dp, kv_head_stra, 1, 1), (dp, kv_head_stra, 1, 1))
            elif input_layout == "BSND":
                stra = ((dp, sp, mp, 1), (dp, 1, kv_head_stra, 1), (dp, 1, kv_head_stra, 1))
            elif input_layout == "TND":
                stra = ((dp * sp, mp, 1), (dp, kv_head_stra, 1), (dp, kv_head_stra, 1))
            else:
                raise ValueError("input_layout is invalid.")
            if with_real_shift:
                stra += ((dp, mp, sp, 1),)
            if keep_prob < 1.0:
                stra += ((dp, mp, sp, 1),)
            if sparse_mode not in compressed_mask_mode:
                stra += ((dp, 1, sp, 1),)
            else:
                stra += ((1, 1),)
            if input_layout == "TND":
                stra += ((dp * sp,),)
                stra += ((dp,),)
            self.fa_op.shard(stra)

            if input_layout == "TND":
                layout = Layout(device_matrix=(dp, sp, mp), alias_name=("dp", "sp", "mp"))
                kv_head_map_name = "None" if use_mqa else "mp"
                self.fa_op.shard(in_strategy=(layout(("dp", "sp"), "mp", "None"),
                                              layout("dp", kv_head_map_name, "None"),
                                              layout("dp", kv_head_map_name, "None"),
                                              layout("None", "None"),
                                              layout("dp"),
                                              layout("dp")))

    def construct(self, query, key, value, real_shift, attn_mask, actual_seq_qlen=None, actual_seq_kvlen=None):
        """forward"""
        drop_mask_bits = None
        if self.input_layout != "TND":
            if self.input_layout == "BSH":
                bsz, seq_len, _ = query.shape
            elif self.input_layout == "SBH":
                seq_len, bsz, _ = query.shape
            elif self.input_layout == "BNSD":
                bsz, _, seq_len, _ = query.shape
            elif self.input_layout == "BSND":
                bsz, seq_len, _, _ = query.shape
            else:
                raise ValueError("input_layout is invalid.")
            if self.keep_prob < 1.0:
                drop_mask_bits = self.reshape(self.drop_gen_mask((bsz, self.head_num, seq_len, seq_len),
                                                                 self.keep_prob),
                                              (bsz, self.head_num, seq_len, 128))
        _, _, _, out = self.fa_op(query, key, value, real_shift, drop_mask_bits, None, self.attn_mask, None,
                                  actual_seq_qlen, actual_seq_kvlen)
        return out


class Grad(Cell):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.grad = GradOperation(get_all=True, sens_param=True)

    def construct(self, *inputs):
        return self.grad(self.network)(*inputs)


def generate_inputs(B, N1, N2, S1, S2, D, input_layout, dtype, return_tensor=True):
    """Generate inputs for flash_attention_score"""
    min_value = -1
    max_value = 1
    if input_layout == "BSH":
        query = np.random.uniform(min_value, max_value, [B, S1, N1 * D])
        key = np.random.uniform(min_value, max_value, [B, S2, N2 * D])
        value = np.random.uniform(min_value, max_value, [B, S2, N2 * D])
    elif input_layout == "BNSD":
        query = np.random.uniform(min_value, max_value, [B, N1, S1, D])
        key = np.random.uniform(min_value, max_value, [B, N2, S2, D])
        value = np.random.uniform(min_value, max_value, [B, N2, S2, D])
    elif input_layout == "SBH":
        query = np.random.uniform(min_value, max_value, [S1, B, N1 * D])
        key = np.random.uniform(min_value, max_value, [S2, B, N2 * D])
        value = np.random.uniform(min_value, max_value, [S2, B, N2 * D])
    elif input_layout == "BSND":
        query = np.random.uniform(min_value, max_value, [B, S1, N1, D])
        key = np.random.uniform(min_value, max_value, [B, S2, N2, D])
        value = np.random.uniform(min_value, max_value, [B, S2, N2, D])
    elif input_layout == "TND":
        query = np.random.uniform(min_value, max_value, [B * S1, N1, D])
        key = np.random.uniform(min_value, max_value, [B * S2, N2, D])
        value = np.random.uniform(min_value, max_value, [B * S2, N2, D])
    else:
        raise ValueError("input_layout is invalid.")
    real_shift = None
    attn_mask = np.triu(np.ones([B, 1, S1, S2]))
    prefix = None
    if return_tensor:
        return Tensor(query, dtype=dtype), Tensor(key, dtype=dtype), Tensor(value, dtype=dtype), real_shift, \
               Tensor(attn_mask, dtype=mstype.uint8), prefix
    return query, key, value, real_shift, attn_mask, prefix


@pytest.mark.parametrize("eod_index_list", [[145, 301, 488, 560, 890, 1024, 1024],
                                            [1024, 1024, 1024, 1024, 1024, 1024, 1024]])
def test_flash_attention_score_tnd(eod_index_list):
    """
    Feature: Test the precision for TND.
    Description: Test function flash attention score forward and backward.
    Expectation: The result of TND and BSH is equal.
    """
    mode = ms.context.GRAPH_MODE
    dtype = mstype.float16
    input_layout = "TND"
    context.set_context(jit_level='O0')
    context.set_context(mode=mode)
    init()
    B, N, S, D = 1, 8, 1024, 128
    dp = 1
    mp = 2
    sp = 4
    use_mqa = 0
    sparse_mode = 3
    query, key, value, _, _, _ = generate_inputs(B, N, N, S, S, D, input_layout, dtype)
    real_shift = None
    actual_seq_qlen = Tensor(eod_index_list, mstype.int64)
    actual_seq_kvlen = Tensor(eod_index_list, mstype.int64)

    attn_mask = Tensor(np.triu(np.ones((2048, 2048), np.uint8), 1))

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    standalone_net = Net(N, input_layout=input_layout, use_mqa=use_mqa, keep_prob=1.0, sparse_mode=sparse_mode)
    standalone_out = standalone_net(query, key, value, real_shift, attn_mask, actual_seq_qlen, actual_seq_kvlen)

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, dataset_strategy="full_batch",
                                      parallel_mode="semi_auto_parallel")
    parallel_net = Net(N, input_layout=input_layout, use_mqa=use_mqa, keep_prob=1.0, sparse_mode=sparse_mode, dp=dp,
                       mp=mp, sp=sp)
    parallel_out = parallel_net(query, key, value, real_shift, attn_mask, actual_seq_qlen, actual_seq_kvlen)

    atol, rtol = 1e-3, 1e-3
    np.allclose(standalone_out.asnumpy(), parallel_out.asnumpy(), atol, rtol)

def test_flash_attention_score_load_balance():
    """
    Feature: Test the precision for load balance.
    Description: Test function flash attention score forward and backward with load balance.
    Expectation: The result of standalone and parallel is equal.
    """
    mode = ms.context.GRAPH_MODE
    dtype = mstype.float16
    input_layout = "BNSD"
    context.set_context(jit_level='O0')
    context.set_context(mode=mode)
    init()
    B, N, S, D = 1, 8, 1024, 128
    dp = 1
    mp = 1
    sp = 8
    use_mqa = 0
    sparse_mode = 3
    query, key, value, _, _, _ = generate_inputs(B, N, N, S, S, D, input_layout, dtype)
    real_shift = None

    attn_mask = Tensor(np.triu(np.ones((2048, 2048), np.uint8), 1))

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    standalone_net = Net(N, input_layout=input_layout, use_mqa=use_mqa, keep_prob=1.0, sparse_mode=sparse_mode,
                         with_real_shift=False)
    standalone_out = standalone_net(query, key, value, real_shift, attn_mask)

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, dataset_strategy="full_batch",
                                      parallel_mode="semi_auto_parallel")
    parallel_net = Net(N, input_layout=input_layout, use_mqa=use_mqa, keep_prob=1.0, sparse_mode=sparse_mode,
                       with_real_shift=False, dp=dp, mp=mp, sp=sp)
    parallel_out = parallel_net(query, key, value, real_shift, attn_mask)

    atol, rtol = 1e-3, 1e-3
    np.allclose(standalone_out.asnumpy(), parallel_out.asnumpy(), atol, rtol)
