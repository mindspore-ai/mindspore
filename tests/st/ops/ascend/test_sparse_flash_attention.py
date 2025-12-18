# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test cases for sparse_flash_attention operator."""
import torch
import random
import math
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, ops
from mindspore import context
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.pynative.utils import allclose_nparray
from tests.st.utils import test_utils
from tests.st.utils.test_utils import double_golden_compare

ms.context.set_context(device_target="Ascend", deterministic='ON', pynative_synchronize=False)
torch.use_deterministic_algorithms(True)

# Tolerance constants for different dtypes
FP16_RTOL = 1e-3
FP16_ATOL = 1e-3
BF16_RTOL = 4e-3
BF16_ATOL = 4e-3

# ACLNN Interface Requirements:
# - sparse_block_size: range [1, 128], must be power of 2
# - layout_query: "BSND" or "TND"
# - layout_kv: "BSND", "TND", or "PA_BSND"
# - sparse_mode: 0=full, 3=rightDownCausal
# - attention_mode: ONLY supports 2 (MLA-absorb mode), requires query_rope and key_rope
# - D of query/key/value equals 512, D of query_rope/key_rope equals 64
# - N1 supports 1/2/4/8/16/32/64/128, N2 only supports 1
# - query/key/value dtype: float16 or bfloat16
# - pre_tokens/next_tokens: only support default value (2^63-1)

def tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdim=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    softmax_res = y.div(x_sum)
    return softmax_res, x_max, x_sum

def simpleSoftmax(x, x_max, x_sum):
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    softmax_res = y.div(x_sum)
    return softmax_res

def tsoftmax_grad(p, dp, out, out_grad):
    muls = out_grad * out
    muls_res = muls.sum(dim=-1, keepdim=True)
    sub_res = dp - muls_res
    res = sub_res * p
    return res

def get_tnd_idx(actual_q_len, t_idx):
    b_idx = 0
    while t_idx >= actual_q_len[b_idx]:
        b_idx += 1
    if b_idx == 0:
        s1_offset = 0
    else:
        s1_offset = actual_q_len[b_idx - 1]
    s1_idx = t_idx - s1_offset
    return b_idx, s1_idx

def get_currentB_index(BS_index, actual_seq_len):
    for i, x in enumerate(actual_seq_len):
        if BS_index < x:
            return i
    raise RuntimeError("BS_index is greater than max(actual_seq_len).")

def cal_attenmask(topk, s1_idx, selected_block_size, actual_sel_s2, G, cur_s2, cur_s1):
    atten_msk = torch.zeros(actual_sel_s2)
    sparse_tail_idx = math.ceil(cur_s2 / selected_block_size)
    sparse_tail_seq_len = cur_s2 % selected_block_size
    if sparse_tail_seq_len == 0:
        sparse_tail_seq_len = selected_block_size
    # sparseMode=2
    # threshold = s1_idx + 1
    # sparseMode=3
    delta_s = cur_s2 - cur_s1
    threshold = delta_s + s1_idx + 1
    s_idx = 0
    for sparse_id in topk:
        if sparse_id == -1:
            break
        begin_idx = sparse_id * selected_block_size
        block_len = selected_block_size if sparse_id != sparse_tail_idx - 1 else sparse_tail_seq_len
        end_idx = begin_idx + block_len
        if begin_idx < threshold and end_idx <= threshold:
            s_idx += block_len
            continue
        if end_idx > threshold:
            local_offset = 0 if threshold <= begin_idx else threshold - begin_idx
            mask_begin = s_idx + local_offset
            mask_end = s_idx + block_len
            atten_msk[mask_begin:mask_end] = 1
        s_idx += block_len
    atten_msk = atten_msk.repeat(G, 1)
    return atten_msk


class SparseFlashAttentionRef:
    def __init__(self, query, key, value, out, dy, topkIndices, attenMsk, actualSeqLenQ,
                 actualSeqLenKV, scaleValue=1.0, blockCount=16, blockSize=64, sparseMode=0,
                 q_rope=None, k_rope=None, params=None):
        self.query = query
        self.key = key
        self.value = value
        self.out = out
        self.dy = dy
        self.topkIndices = topkIndices
        self.attenMsk = attenMsk
        self.actualSeqLenQ = actualSeqLenQ
        self.actualSeqLenKV = actualSeqLenKV
        self.q_rope = q_rope if q_rope.numel() != 0 else None
        self.k_rope = k_rope if k_rope.numel() != 0 else None
        self.scaleValue = scaleValue
        self.blockCount = blockCount
        self.blockSize = blockSize
        self.sparseMode = sparseMode
        self.dtype = query.dtype
        self.softmaxMax = (torch.rand(query.shape[0], query.shape[1], 8).to(self.dtype))
        self.softmaxSum = self.softmaxMax
        self.orgDtype = params.dtype

    def gather_kv(self, cur_k, cur_v, topk, cur_s2):
        actual_sel_s2 = 0
        s2_sparse = []
        for sparse_id in topk:
            if sparse_id == -1:
                break
            begin_idx = sparse_id * self.blockSize
            end_idx = begin_idx + self.blockSize \
                if begin_idx + self.blockSize <= cur_s2 else cur_s2
            s2_sparse.extend(np.arange(begin_idx, end_idx))
            actual_sel_s2 += end_idx - begin_idx
        k_cal, v_cal = cur_k[s2_sparse, :], cur_v[s2_sparse, :]
        return k_cal, v_cal, actual_sel_s2

    def scatter_kv(self, dkv_out, dkv, topk, cur_s2):
        dkv_start = 0
        for sparse_id in topk:
            if sparse_id == -1:
                break
            begin_idx = sparse_id * self.blockSize
            end_idx = begin_idx + self.blockSize \
                if begin_idx + self.blockSize <= cur_s2 else cur_s2
            dkv_end = dkv_start + (end_idx - begin_idx)
            dkv_out[begin_idx:end_idx, :] += dkv[dkv_start:dkv_end, :]
            dkv_start = dkv_end
        return dkv_out

    def forward(self):
        query = self.query.float()
        key = self.key.float()
        value = self.value.float()
        topk_indices = self.topkIndices
        actual_q_len = self.actualSeqLenQ
        actual_kv_len = self.actualSeqLenKV
        scaleValue = self.scaleValue
        selected_block_size = self.blockSize
        atten_enable = self.sparseMode == 3
        T1, N1, D_qk = query.shape
        _, N2, D_v = value.shape
        G = N1 // N2
        # reshape
        query = query.reshape(T1, N2, G, D_qk)
        if self.q_rope is not None and self.k_rope is not None:
            q_rope = self.q_rope.float()
            k_rope = self.k_rope.float()
            q_rope = q_rope.reshape(T1, N2, G, q_rope.shape[-1])
            query = torch.cat([query, q_rope], dim=-1)
            key = torch.cat([key, k_rope], dim=-1)
            D_qk += q_rope.shape[-1]
        x_max_out = torch.zeros(T1, N2, G, 1).to(torch.float)
        x_sum_out = torch.zeros(T1, N2, G, 1).to(torch.float)
        # Output tensor for attention: [T1, N2, G, D_v]
        out_tnd = torch.zeros(T1, N2, G, D_v).to(torch.float)
        for i in range(T1):
            b_idx, s1_idx = get_tnd_idx(actual_q_len, i)
            for n2_idx in range(N2):
                topk = topk_indices[i][n2_idx]
                q_cal = query[i][n2_idx]  # [G, D_qk]
                s2_start = 0 if b_idx == 0 else actual_kv_len[b_idx - 1]
                s2_end = actual_kv_len[b_idx]
                cur_s2 = actual_kv_len[b_idx] if b_idx == 0 else \
                    (actual_kv_len[b_idx] - actual_kv_len[b_idx - 1])
                cur_s1 = actual_q_len[b_idx] if b_idx == 0 else \
                    (actual_q_len[b_idx] - actual_q_len[b_idx - 1])
                cur_k = key[s2_start:s2_end, n2_idx, :]
                cur_v = value[s2_start:s2_end, n2_idx, :]
                k_cal, v_cal, actual_sel_s2 = self.gather_kv(cur_k, cur_v, topk, cur_s2)
                if actual_sel_s2 > 0:
                    qk = torch.matmul(q_cal, k_cal.permute(1, 0)).mul(scaleValue)
                    if atten_enable:
                        atten_msk_cal = cal_attenmask(
                            topk, s1_idx, selected_block_size, actual_sel_s2, G, cur_s2, cur_s1)
                        qk = qk + atten_msk_cal * (-2e35)
                    softmax_res, x_max, x_sum = tsoftmax(qk)
                    x_max_out[i][n2_idx] = x_max
                    x_sum_out[i][n2_idx] = x_sum
                    # Compute attention output: softmax @ V
                    attn_out = torch.matmul(softmax_res, v_cal)  # [G, D_v]
                    out_tnd[i][n2_idx] = attn_out
        x_max_out = x_max_out.expand(T1, N2, G, 8)
        x_sum_out = x_sum_out.expand(T1, N2, G, 8)
        self.softmaxMax = x_max_out.reshape(T1, N2 * G, 8).float()
        self.softmaxSum = x_sum_out.reshape(T1, N2 * G, 8).float()
        # Reshape output to [T1, N1, D_v]
        self.out = out_tnd.reshape(T1, N1, D_v).to(self.dtype)
        return self.out, self.softmaxMax, self.softmaxSum

    def backward(self):
        query = self.query.float()
        key = self.key.float()
        value = self.value.float()
        out = self.out.float()
        dy = self.dy.float()
        softmaxMax = self.softmaxMax.float()
        softmaxSum = self.softmaxSum.float()
        topk_indices = self.topkIndices
        actual_q_len = self.actualSeqLenQ
        actual_kv_len = self.actualSeqLenKV
        # param
        scaleValue = self.scaleValue
        selected_block_size = self.blockSize
        atten_enable = self.sparseMode == 3
        # shape
        T1, N1, D_qk = query.shape
        _, N2, D_v = value.shape
        G = N1 // N2
        org_Dqk = D_qk
        S2 = max(self.actualSeqLenKV)
        # reshape
        query = query.reshape(T1, N2, G, D_qk)
        out = out.reshape(T1, N2, G, D_v)
        dy = dy.reshape(T1, N2, G, D_v)
        softmaxMax = softmaxMax.reshape(T1, N2, G, 8)
        softmaxSum = softmaxSum.reshape(T1, N2, G, 8)
        if self.q_rope is not None and self.k_rope is not None:
            q_rope = self.q_rope.float()
            k_rope = self.k_rope.float()
            q_rope = q_rope.reshape(T1, N2, G, q_rope.shape[-1])
            query = torch.cat([query, q_rope], dim=-1)
            key = torch.cat([key, k_rope], dim=-1)
            D_qk += q_rope.shape[-1]
        dq_out = torch.zeros(T1, N2, G, D_qk).to(torch.float)
        dk_out = torch.zeros(S2, N2, D_qk).to(torch.float)
        dv_out = torch.zeros(S2, N2, D_v).to(torch.float)
        for i in range(T1):
            b_idx, s1_idx = get_tnd_idx(actual_q_len, i)
            for n2_idx in range(N2):
                print(f"\r cal: {i}/{T1-1} {n2_idx}/{N2-1}......", end="")
                # gather
                topk = topk_indices[i][n2_idx]
                q_cal = query[i][n2_idx]
                out_cal = out[i][n2_idx]
                dy_cal = dy[i][n2_idx]
                s2_start = 0 if b_idx == 0 else actual_kv_len[b_idx - 1]
                s2_end = actual_kv_len[b_idx]
                cur_s2 = actual_kv_len[b_idx] if b_idx == 0 else \
                    (actual_kv_len[b_idx] - actual_kv_len[b_idx - 1])
                cur_s1 = actual_q_len[b_idx] if b_idx == 0 else \
                    (actual_q_len[b_idx] - actual_q_len[b_idx - 1])
                cur_k = key[s2_start:s2_end, n2_idx, :]
                cur_v = value[s2_start:s2_end, n2_idx, :]
                k_cal, v_cal, actual_sel_s2 = self.gather_kv(cur_k, cur_v, topk, cur_s2)
                if actual_sel_s2 > 0:
                    # fag cal
                    qk = torch.matmul(q_cal, k_cal.permute(1, 0)).mul(scaleValue)
                    if atten_enable:
                        atten_msk_cal = cal_attenmask(
                            topk, s1_idx, selected_block_size, actual_sel_s2, G, cur_s2, cur_s1)
                        qk = qk + atten_msk_cal * (-2e35)
                    x_max = softmaxMax[i][n2_idx][:, [0]].reshape(-1, 1)
                    x_sum = softmaxSum[i][n2_idx][:, [0]].reshape(-1, 1)
                    softmax_res = simpleSoftmax(qk, x_max, x_sum)
                    dp = torch.matmul(dy_cal, v_cal.permute(1, 0))
                    softmax_grad_res = tsoftmax_grad(softmax_res, dp, out_cal, dy_cal)
                    softmax_grad_res = softmax_grad_res.to(self.orgDtype).to(torch.float32)
                    dq = torch.matmul(softmax_grad_res, k_cal)
                    dk = torch.matmul(softmax_grad_res.permute(1, 0), q_cal)
                    dv = torch.matmul(
                        softmax_res.permute(1, 0).to(self.orgDtype).to(torch.float32), dy_cal)
                    # scatter
                    dq_out[i][n2_idx] = dq
                    dk_out[s2_start:s2_end, n2_idx, :] = self.scatter_kv(
                        dk_out[s2_start:s2_end, n2_idx, :], dk, topk, cur_s2)
                    dv_out[s2_start:s2_end, n2_idx, :] = self.scatter_kv(
                        dv_out[s2_start:s2_end, n2_idx, :], dv, topk, cur_s2)
        dq_out = dq_out * scaleValue
        dk_out = dk_out * scaleValue
        dq_out = dq_out.reshape(T1, N1, D_qk)
        if self.q_rope is not None and self.k_rope is not None:
            dq_rope = dq_out[:, :, org_Dqk:]
            dk_rope = dk_out[:, :, org_Dqk:]
            dk = dk_out[:, :, :org_Dqk]
            dq = dq_out[:, :, :org_Dqk]
        else:
            dq = dq_out
            dk = dk_out
            dq_rope = None
            dk_rope = None
        return dq, dk, dv_out, dq_rope, dk_rope


@test_utils.run_with_cell
def sparse_flash_attention_forward_func(query, key, value, sparse_indices, scale_value,
                                        block_table=None, actual_seq_lengths_query=None,
                                        actual_seq_lengths_kv=None, query_rope=None, key_rope=None,
                                        sparse_block_size=1, layout_query="BSND", layout_kv="BSND",
                                       sparse_mode=3, pre_tokens=9223372036854775807,
                                       next_tokens=9223372036854775807, attention_mode=2,
                                       return_softmax_lse=False):
    """MindSpore forward function wrapper"""
    return ops.sparse_flash_attention(query, key, value, sparse_indices, scale_value,
                                      block_table=block_table,
                                      actual_seq_lengths_query=actual_seq_lengths_query,
                                      actual_seq_lengths_kv=actual_seq_lengths_kv,
                                      query_rope=query_rope, key_rope=key_rope,
                                      sparse_block_size=sparse_block_size,
                                      layout_query=layout_query, layout_kv=layout_kv,
                                      sparse_mode=sparse_mode, pre_tokens=pre_tokens,
                                      next_tokens=next_tokens, attention_mode=attention_mode,
                                      return_softmax_lse=return_softmax_lse)


@test_utils.run_with_cell
def sparse_flash_attention_backward_func(query, key, value, sparse_indices, scale_value,
                                         block_table=None, actual_seq_lengths_query=None,
                                         actual_seq_lengths_kv=None, query_rope=None, key_rope=None,
                                         sparse_block_size=1, layout_query="BSND", layout_kv="BSND",
                                         sparse_mode=3, pre_tokens=9223372036854775807,
                                         next_tokens=9223372036854775807, attention_mode=2,
                                         return_softmax_lse=False):
    """MindSpore backward function wrapper.
    has_aux=True: only compute gradient w.r.t. attention_out (first output)
    """
    grad_fn = ms.grad(sparse_flash_attention_forward_func, grad_position=(0, 1, 2), has_aux=True)
    grads, _ = grad_fn(
        query, key, value, sparse_indices, scale_value,
        block_table, actual_seq_lengths_query, actual_seq_lengths_kv,
        query_rope, key_rope, sparse_block_size, layout_query, layout_kv,
        sparse_mode, pre_tokens, next_tokens, attention_mode, return_softmax_lse)
    return grads


@ms.jit
def sparse_flash_attention_forward_func_jit(query, key, value, sparse_indices, scale_value,
                                            block_table, actual_seq_lengths_query,
                                            actual_seq_lengths_kv, query_rope, key_rope,
                                            sparse_block_size, layout_query, layout_kv,
                                            sparse_mode, pre_tokens, next_tokens,
                                            attention_mode, return_softmax_lse):
    return ops.sparse_flash_attention(query, key, value, sparse_indices, scale_value,
                                      block_table=block_table,
                                      actual_seq_lengths_query=actual_seq_lengths_query,
                                      actual_seq_lengths_kv=actual_seq_lengths_kv,
                                      query_rope=query_rope, key_rope=key_rope,
                                      sparse_block_size=sparse_block_size,
                                      layout_query=layout_query, layout_kv=layout_kv,
                                      sparse_mode=sparse_mode, pre_tokens=pre_tokens,
                                      next_tokens=next_tokens, attention_mode=attention_mode,
                                      return_softmax_lse=return_softmax_lse)


@ms.jit
def sparse_flash_attention_backward_func_jit(query, key, value, sparse_indices, scale_value,
                                             block_table, actual_seq_lengths_query,
                                             actual_seq_lengths_kv, query_rope, key_rope,
                                             sparse_block_size, layout_query, layout_kv,
                                             sparse_mode, pre_tokens, next_tokens,
                                             attention_mode, return_softmax_lse):
    grad_fn = ms.grad(sparse_flash_attention_forward_func_jit, grad_position=(0, 1, 2), has_aux=True)
    grads, _ = grad_fn(
        query, key, value, sparse_indices, scale_value,
        block_table, actual_seq_lengths_query, actual_seq_lengths_kv,
        query_rope, key_rope, sparse_block_size, layout_query, layout_kv,
        sparse_mode, pre_tokens, next_tokens, attention_mode, return_softmax_lse)
    return grads


def _ms_forward_backward(context_mode, query_np, key_np, value_np, sparse_indices_np,
                         scale_value, actual_seq_lengths_query_list, actual_seq_lengths_kv_list,
                         query_rope_np=None, key_rope_np=None, sparse_block_size=1,
                         layout_query="BSND", layout_kv="BSND", sparse_mode=3,
                         attention_mode=2, dtype=ms.float16, return_softmax_lse=True):
    """Run forward and backward in MindSpore with numpy inputs"""
    context.set_context(mode=context_mode)
    query = Tensor(query_np, dtype=dtype)
    key = Tensor(key_np, dtype=dtype)
    value = Tensor(value_np, dtype=dtype)
    sparse_indices = Tensor(sparse_indices_np, dtype=ms.int32)
    query_rope = Tensor(query_rope_np, dtype=dtype) if query_rope_np is not None else None
    key_rope = Tensor(key_rope_np, dtype=dtype) if key_rope_np is not None else None

    # Convert list to Tensor (int32) for actual_seq_lengths, aligned with ACLNN interface
    actual_seq_lengths_query = Tensor(actual_seq_lengths_query_list, dtype=ms.int32) \
        if actual_seq_lengths_query_list is not None else None
    actual_seq_lengths_kv = Tensor(actual_seq_lengths_kv_list, dtype=ms.int32) \
        if actual_seq_lengths_kv_list is not None else None

    f_out = sparse_flash_attention_forward_func(
        query, key, value, sparse_indices, scale_value,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        query_rope=query_rope, key_rope=key_rope,
        sparse_block_size=sparse_block_size,
        layout_query=layout_query, layout_kv=layout_kv,
        sparse_mode=sparse_mode, attention_mode=attention_mode,
        return_softmax_lse=return_softmax_lse)

    b_out = sparse_flash_attention_backward_func(
        query, key, value, sparse_indices, scale_value,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        query_rope=query_rope, key_rope=key_rope,
        sparse_block_size=sparse_block_size,
        layout_query=layout_query, layout_kv=layout_kv,
        sparse_mode=sparse_mode, attention_mode=attention_mode,
        return_softmax_lse=return_softmax_lse)

    return f_out, b_out


def get_qkv_bsnd(B, S1, S2, N1, N2, D, dtype=np.float16, seed=42):
    """Generate random query, key, value tensors for BSND layout.

    Args:
        B: Batch size
        S1: Query sequence length
        S2: Key/value sequence length
        N1: Query head number
        N2: Key/value head number
        D: Head dimension
        dtype: Data type (np.float16 or 'bf16')
        seed: Random seed
    """
    np.random.seed(seed)
    if dtype == 'bf16':
        query_np = torch.randn(B, S1, N1, D, dtype=torch.bfloat16).to(torch.float32).numpy()
        key_np = torch.randn(B, S2, N2, D, dtype=torch.bfloat16).to(torch.float32).numpy()
        value_np = key_np.copy()
    else:
        query_np = np.random.uniform(-10, 10, (B, S1, N1, D)).astype(dtype)
        key_np = np.random.uniform(-5, 10, (B, S2, N2, D)).astype(dtype)
        value_np = key_np.copy()
    return query_np, key_np, value_np


def get_sparse_indices(B, S1, N2, sparse_count, max_idx, seed=42):
    """Generate sparse indices tensor.

    Args:
        B: Batch size
        S1: Query sequence length
        N2: Key head number
        sparse_count: Number of sparse indices per position
        max_idx: Maximum index value
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    idxs = random.sample(range(max_idx), min(sparse_count, max_idx))
    if len(idxs) < sparse_count:
        idxs = idxs + [0] * (sparse_count - len(idxs))
    sparse_indices = np.array([idxs for _ in range(B * S1 * N2)]).reshape((B, S1, N2, sparse_count))
    return sparse_indices.astype(np.int32)


def get_rope_tensors(B, S1, S2, N1, N2, D_rope, dtype=np.float16, seed=42):
    """Generate query_rope and key_rope tensors.

    Args:
        B: Batch size
        S1: Query sequence length
        S2: Key/value sequence length
        N1: Query head number
        N2: Key/value head number
        D_rope: RoPE dimension (typically 64)
        dtype: Data type
        seed: Random seed
    """
    np.random.seed(seed)
    if dtype == 'bf16':
        query_rope = torch.randn(B, S1, N1, D_rope, dtype=torch.bfloat16).to(torch.float32).numpy()
        key_rope = torch.randn(B, S2, N2, D_rope, dtype=torch.bfloat16).to(torch.float32).numpy()
    else:
        query_rope = np.random.uniform(-10, 10, (B, S1, N1, D_rope)).astype(dtype)
        key_rope = np.random.uniform(-10, 10, (B, S2, N2, D_rope)).astype(dtype)
    return query_rope, key_rope


def get_torch_tensors(query_np, key_np, value_np, sparse_indices_np,
                      query_rope_np=None, key_rope_np=None,
                      dtype='fp16', target_dtype='fp32'):
    """Convert numpy arrays to PyTorch tensors.

    Args:
        query_np: Query numpy array
        key_np: Key numpy array
        value_np: Value numpy array
        sparse_indices_np: Sparse indices numpy array
        query_rope_np: Query RoPE numpy array (optional)
        key_rope_np: Key RoPE numpy array (optional)
        dtype: Source dtype ('fp16' or 'bf16')
        target_dtype: Target dtype for tensors ('fp32', 'fp16', or 'bf16')
    """
    if target_dtype == 'fp32':
        torch_dtype = torch.float32
        pt_query = torch.tensor(query_np, dtype=torch_dtype)
        pt_key = torch.tensor(key_np, dtype=torch_dtype)
        pt_value = torch.tensor(value_np, dtype=torch_dtype)
        pt_query_rope = torch.tensor(query_rope_np, dtype=torch_dtype) if query_rope_np is not None else None
        pt_key_rope = torch.tensor(key_rope_np, dtype=torch_dtype) if key_rope_np is not None else None
    elif target_dtype == 'fp16':
        pt_query = torch.from_numpy(query_np).to(torch.float16)
        pt_key = torch.from_numpy(key_np).to(torch.float16)
        pt_value = torch.from_numpy(value_np).to(torch.float16)
        pt_query_rope = torch.from_numpy(query_rope_np).to(torch.float16) if query_rope_np is not None else None
        pt_key_rope = torch.from_numpy(key_rope_np).to(torch.float16) if key_rope_np is not None else None
    elif target_dtype == 'bf16':
        pt_query = torch.from_numpy(query_np).to(torch.bfloat16)
        pt_key = torch.from_numpy(key_np).to(torch.bfloat16)
        pt_value = torch.from_numpy(value_np).to(torch.bfloat16)
        pt_query_rope = torch.from_numpy(query_rope_np).to(torch.bfloat16) if query_rope_np is not None else None
        pt_key_rope = torch.from_numpy(key_rope_np).to(torch.bfloat16) if key_rope_np is not None else None
    else:
        torch_dtype = torch.float32
        pt_query = torch.tensor(query_np, dtype=torch_dtype)
        pt_key = torch.tensor(key_np, dtype=torch_dtype)
        pt_value = torch.tensor(value_np, dtype=torch_dtype)
        pt_query_rope = torch.tensor(query_rope_np, dtype=torch_dtype) if query_rope_np is not None else None
        pt_key_rope = torch.tensor(key_rope_np, dtype=torch_dtype) if key_rope_np is not None else None

    pt_sparse_indices = torch.tensor(sparse_indices_np, dtype=torch.int32)

    return pt_query, pt_key, pt_value, pt_sparse_indices, pt_query_rope, pt_key_rope


def gather_kv_torch(k_tensor, v_tensor, sparse_indices, sparse_block_size,
                    batch, n2_idx, s1_idx, cur_actual_seq_kv):
    """Gather K/V tensors based on sparse indices (PyTorch version)."""
    s2_sparse = []
    for sparse_id in sparse_indices:
        if sparse_id == -1:
            break
        begin_idx = int(sparse_id) * sparse_block_size
        end_idx = min(begin_idx + sparse_block_size, cur_actual_seq_kv)
        s2_sparse.extend(range(begin_idx, end_idx))

    if len(s2_sparse) == 0:
        return None, None

    k_sparse = k_tensor[batch, n2_idx, s2_sparse, :]
    v_sparse = v_tensor[batch, n2_idx, s2_sparse, :]
    return k_sparse, v_sparse


def apply_causal_mask(res, cur_actual_seq_q, cur_actual_seq_kv, topk_indices, s1_idx, sparse_blocksize):
    """Apply causal mask to attention scores."""
    sparse_tail_idx = int(np.ceil(cur_actual_seq_kv / sparse_blocksize))
    sparse_tail_seq_len = cur_actual_seq_kv % sparse_blocksize
    if sparse_tail_seq_len == 0:
        sparse_tail_seq_len = sparse_blocksize

    delta_s = cur_actual_seq_kv - cur_actual_seq_q
    threshold = delta_s + s1_idx + 1

    s_idx = 0
    for sparse_id in topk_indices:
        if sparse_id == -1:
            break
        begin_idx = int(sparse_id) * sparse_blocksize
        block_len = sparse_blocksize if int(sparse_id) != sparse_tail_idx - 1 else sparse_tail_seq_len
        end_idx = begin_idx + block_len

        if begin_idx < threshold <= end_idx:
            local_offset = 0 if threshold <= begin_idx else threshold - begin_idx
            mask_begin = s_idx + local_offset
            mask_end = s_idx + block_len
            res[:, mask_begin:mask_end] = -1e12
        elif end_idx > threshold:
            res[:, s_idx:s_idx + block_len] = -1e12
        s_idx += block_len

    return res


def torch_cpu_generate_golden(query, key, value, sparse_indices, scale_value, sparse_block_size,
                               actual_seq_lengths_query, actual_seq_lengths_kv,
                               query_rope=None, key_rope=None, sparse_mode=3,
                               return_softmax_lse=True):
    """Generate golden reference output using torch CPU."""
    B = query.shape[0]
    S1 = query.shape[1]
    N1 = query.shape[2]
    _ = query.shape[3]  # D (head dimension)
    N2 = key.shape[2]
    G = N1 // N2

    dtype = query.dtype

    # Concatenate RoPE if provided
    if query_rope is not None:
        q_tensor = torch.cat([query, query_rope], dim=-1)
        k_tensor = torch.cat([key, key_rope], dim=-1)
    else:
        q_tensor = query
        k_tensor = key
    v_tensor = value

    # Transform to BNSD format
    q_bnsd = q_tensor.permute(0, 2, 1, 3)  # [B, N1, S1, D+D_rope]
    k_bnsd = k_tensor.permute(0, 2, 1, 3)  # [B, N2, S2, D+D_rope]
    v_bnsd = v_tensor.permute(0, 2, 1, 3)  # [B, N2, S2, D]
    sparse_bnsd = sparse_indices.permute(0, 2, 1, 3)  # [B, N2, S1, sparse_count]

    # Output tensors
    out_shape = list(q_bnsd.shape)
    if query_rope is not None:
        out_shape[-1] = out_shape[-1] - query_rope.shape[-1]
    y = torch.zeros(out_shape, dtype=torch.float32)

    # softmax_max/sum shape: [B, N2, S1, G]
    softmax_max = torch.zeros(B, N2, S1, G, dtype=torch.float32)
    softmax_sum = torch.zeros(B, N2, S1, G, dtype=torch.float32)

    for batch in range(B):
        cur_seq_q = actual_seq_lengths_query[batch]
        cur_seq_kv = actual_seq_lengths_kv[batch]

        for n2_idx in range(N2):
            for s1_idx in range(cur_seq_q):
                q_curr = q_bnsd[batch, n2_idx * G:(n2_idx + 1) * G, s1_idx, :].to(torch.float32)
                cur_sparse_indices = sparse_bnsd[batch, n2_idx, s1_idx, :]

                k_sparse, v_sparse = gather_kv_torch(
                    k_bnsd, v_bnsd, cur_sparse_indices.numpy(),
                    sparse_block_size, batch, n2_idx, s1_idx, cur_seq_kv)

                if k_sparse is None:
                    continue

                k_sparse = k_sparse.to(torch.float32)
                v_sparse = v_sparse.to(torch.float32)

                # QK^T
                mm1_res = torch.matmul(q_curr, k_sparse.T)
                # Scale
                scale_res = mm1_res * scale_value

                # Apply mask if sparse_mode == 3
                if sparse_mode == 3:
                    scale_res = apply_causal_mask(
                        scale_res, cur_seq_q, cur_seq_kv,
                        cur_sparse_indices.numpy(), s1_idx, sparse_block_size)

                # Softmax
                softmax_res, x_max, x_sum = tsoftmax(scale_res)

                # Store softmax_max and softmax_sum if needed
                if return_softmax_lse:
                    softmax_max[batch, n2_idx, s1_idx, :] = x_max.squeeze(-1)
                    softmax_sum[batch, n2_idx, s1_idx, :] = x_sum.squeeze(-1)

                # Attention output
                mm2_res = torch.matmul(softmax_res, v_sparse)
                y[batch, n2_idx * G:(n2_idx + 1) * G, s1_idx, :] = mm2_res

    # Transform back to BSND
    attention_out = y.permute(0, 2, 1, 3).to(dtype)

    return attention_out, softmax_max, softmax_sum


def torch_cpu_forward_backward(query, key, value, sparse_indices, scale_value, sparse_block_size,
                                actual_seq_lengths_query, actual_seq_lengths_kv,
                                query_rope=None, key_rope=None, sparse_mode=3,
                                return_softmax_lse=True):
    """Torch CPU forward and backward computation (original implementation, kept for reference)"""
    query = query.clone().detach().requires_grad_(True)
    key = key.clone().detach().requires_grad_(True)
    value = value.clone().detach().requires_grad_(True)
    if query_rope is not None:
        query_rope = query_rope.clone().detach().requires_grad_(True)
        key_rope = key_rope.clone().detach().requires_grad_(True)

    attention_out, softmax_max, softmax_sum = torch_cpu_generate_golden(
        query, key, value, sparse_indices, scale_value, sparse_block_size,
        actual_seq_lengths_query, actual_seq_lengths_kv,
        query_rope, key_rope, sparse_mode, return_softmax_lse)

    # Only use attention_out for gradient computation
    # softmax_max/sum are intermediate values and don't contribute to gradients in actual implementation
    loss = attention_out.sum()
    loss.backward()

    forward_out = (attention_out.detach(), softmax_max.detach(), softmax_sum.detach())
    backward_out = (query.grad.detach(), key.grad.detach(), value.grad.detach())

    return forward_out, backward_out


def sfa_ref_forward_backward(query, key, value, sparse_indices, scale_value, sparse_block_size,
                         actual_seq_lengths_query, actual_seq_lengths_kv,
                         query_rope=None, key_rope=None, sparse_mode=3,
                         return_softmax_lse=True):
    """Forward using torch_cpu_generate_golden, backward using SparseFlashAttentionRef.

    Args:
        query: [B, S1, N1, D] BSND format
        key: [B, S2, N2, D] BSND format
        value: [B, S2, N2, D] BSND format
        sparse_indices: [B, S1, N2, sparse_count]
        scale_value: attention scale
        sparse_block_size: block size
        actual_seq_lengths_query: list of actual query sequence lengths
        actual_seq_lengths_kv: list of actual kv sequence lengths
        query_rope: RoPE for query (optional)
        key_rope: RoPE for key (optional)
        sparse_mode: 0=full, 3=causal
        return_softmax_lse: whether to return softmax lse

    Returns:
        forward_out: (attention_out, softmax_max, softmax_sum) from torch_cpu_generate_golden
        backward_out: (dq, dk, dv) from SparseFlashAttentionRef.backward()
    """
    # Forward pass: use torch_cpu_generate_golden
    attention_out, softmax_max, softmax_sum = torch_cpu_generate_golden(
        query, key, value, sparse_indices, scale_value, sparse_block_size,
        actual_seq_lengths_query, actual_seq_lengths_kv,
        query_rope, key_rope, sparse_mode, return_softmax_lse)

    # Backward pass: use SparseFlashAttentionRef
    dtype = query.dtype
    B = query.shape[0]
    S1 = query.shape[1]
    N1 = query.shape[2]
    D = query.shape[3]
    N2 = key.shape[2]
    S2 = key.shape[1]

    # Convert BSND to TND layout
    T1 = sum(actual_seq_lengths_query)
    T2 = sum(actual_seq_lengths_kv)

    query_tnd = torch.zeros(T1, N1, D, dtype=dtype)
    key_tnd = torch.zeros(T2, N2, D, dtype=dtype)
    value_tnd = torch.zeros(T2, N2, D, dtype=dtype)
    # Convert attention_out (BSND) to TND for SparseFlashAttentionRef
    out_tnd = torch.zeros(T1, N1, D, dtype=dtype)

    # Convert sparse_indices: [B, S1, N2, sparse_count] -> [T1, N2, sparse_count]
    sparse_count = sparse_indices.shape[-1]
    sparse_indices_tnd = torch.zeros(T1, N2, sparse_count, dtype=torch.int64)

    t1_offset = 0
    t2_offset = 0
    for b in range(B):
        s1 = actual_seq_lengths_query[b]
        s2 = actual_seq_lengths_kv[b]
        query_tnd[t1_offset:t1_offset + s1] = query[b, :s1]
        key_tnd[t2_offset:t2_offset + s2] = key[b, :s2]
        value_tnd[t2_offset:t2_offset + s2] = value[b, :s2]
        out_tnd[t1_offset:t1_offset + s1] = attention_out[b, :s1]
        sparse_indices_tnd[t1_offset:t1_offset + s1] = sparse_indices[b, :s1]
        t1_offset += s1
        t2_offset += s2

    # Convert actual_seq_lengths to cumulative format
    actual_seq_q_cum = []
    actual_seq_kv_cum = []
    cum_q, cum_kv = 0, 0
    for b in range(B):
        cum_q += actual_seq_lengths_query[b]
        cum_kv += actual_seq_lengths_kv[b]
        actual_seq_q_cum.append(cum_q)
        actual_seq_kv_cum.append(cum_kv)

    # Handle RoPE
    if query_rope is not None:
        D_rope = query_rope.shape[-1]
        query_rope_tnd = torch.zeros(T1, N1, D_rope, dtype=dtype)
        key_rope_tnd = torch.zeros(T2, N2, D_rope, dtype=dtype)
        t1_offset = 0
        t2_offset = 0
        for b in range(B):
            s1 = actual_seq_lengths_query[b]
            s2 = actual_seq_lengths_kv[b]
            query_rope_tnd[t1_offset:t1_offset + s1] = query_rope[b, :s1]
            key_rope_tnd[t2_offset:t2_offset + s2] = key_rope[b, :s2]
            t1_offset += s1
            t2_offset += s2
    else:
        query_rope_tnd = torch.empty(0)
        key_rope_tnd = torch.empty(0)

    dy_tnd = torch.ones(T1, N1, D, dtype=dtype)

    # Create SparseFlashAttentionRef instance with attention_out from torch_cpu_generate_golden
    sfa_ref = SparseFlashAttentionRef(
        query=query_tnd,
        key=key_tnd,
        value=value_tnd,
        out=out_tnd,
        dy=dy_tnd,
        topkIndices=sparse_indices_tnd,
        attenMsk=None,
        actualSeqLenQ=actual_seq_q_cum,
        actualSeqLenKV=actual_seq_kv_cum,
        scaleValue=scale_value,
        blockCount=sparse_count,
        blockSize=sparse_block_size,
        sparseMode=sparse_mode,
        q_rope=query_rope_tnd,
        k_rope=key_rope_tnd,
        params=query
    )

    # Forward pass (only to compute softmax_max/sum for backward)
    _, _, _ = sfa_ref.forward()

    # Backward pass
    dq_tnd, dk_tnd, dv_tnd, _, _ = sfa_ref.backward()

    # Convert backward gradients from TND to BSND format
    dq = torch.zeros(B, S1, N1, D, dtype=torch.float32)
    dk = torch.zeros(B, S2, N2, D, dtype=torch.float32)
    dv = torch.zeros(B, S2, N2, D, dtype=torch.float32)

    t1_offset = 0
    t2_offset = 0
    for b in range(B):
        s1 = actual_seq_lengths_query[b]
        s2 = actual_seq_lengths_kv[b]
        dq[b, :s1] = dq_tnd[t1_offset:t1_offset + s1]
        dk[b, :s2] = dk_tnd[t2_offset:t2_offset + s2]
        dv[b, :s2] = dv_tnd[t2_offset:t2_offset + s2]
        t1_offset += s1
        t2_offset += s2

    forward_out = (attention_out, softmax_max, softmax_sum)
    backward_out = (dq, dk, dv)

    return forward_out, backward_out


def sfa_ref_backward(query, key, value, sparse_indices, scale_value, sparse_block_size,
                     actual_seq_lengths_query, actual_seq_lengths_kv,
                     query_rope=None, key_rope=None, sparse_mode=3,
                     attention_out=None):
    """Use SparseFlashAttentionRef class for backward computation.

    Args:
        query: [B, S1, N1, D] or [T1, N1, D] (TND layout)
        key: [B, S2, N2, D] or [T2, N2, D] (TND layout)
        value: [B, S2, N2, D] or [T2, N2, D] (TND layout)
        sparse_indices: sparse attention indices
        scale_value: attention scale
        sparse_block_size: block size for sparse attention
        actual_seq_lengths_query: list of actual query sequence lengths
        actual_seq_lengths_kv: list of actual kv sequence lengths
        query_rope: RoPE for query (optional)
        key_rope: RoPE for key (optional)
        sparse_mode: 0=full, 3=causal
        attention_out: forward attention output for backward computation

    Returns:
        backward_out: (dq, dk, dv, dq_rope, dk_rope) gradients
    """
    dtype = query.dtype
    B = len(actual_seq_lengths_query)

    # Convert BSND to TND layout for SparseFlashAttentionRef
    if query.dim() == 4:
        # BSND layout: [B, S1, N1, D] -> [T1, N1, D]
        T1 = sum(actual_seq_lengths_query)
        T2 = sum(actual_seq_lengths_kv)
        N1 = query.shape[2]
        N2 = key.shape[2]
        D = query.shape[3]

        # Gather valid sequences into TND format
        query_tnd = torch.zeros(T1, N1, D, dtype=dtype)
        key_tnd = torch.zeros(T2, N2, D, dtype=dtype)
        value_tnd = torch.zeros(T2, N2, D, dtype=dtype)
        out_tnd = torch.zeros(T1, N1, D, dtype=dtype)
        dy_tnd = torch.ones(T1, N1, D, dtype=dtype)  # gradient of loss w.r.t output

        # Convert sparse_indices to TND format: [B, S1, N2, sparse_count] -> [T1, N2, sparse_count]
        sparse_count = sparse_indices.shape[-1]
        sparse_indices_tnd = torch.zeros(T1, N2, sparse_count, dtype=torch.int64)

        t1_offset = 0
        t2_offset = 0
        for b in range(B):
            s1 = actual_seq_lengths_query[b]
            s2 = actual_seq_lengths_kv[b]
            query_tnd[t1_offset:t1_offset + s1] = query[b, :s1]
            key_tnd[t2_offset:t2_offset + s2] = key[b, :s2]
            value_tnd[t2_offset:t2_offset + s2] = value[b, :s2]
            out_tnd[t1_offset:t1_offset + s1] = attention_out[b, :s1]
            sparse_indices_tnd[t1_offset:t1_offset + s1] = sparse_indices[b, :s1]
            t1_offset += s1
            t2_offset += s2

        # Convert actual_seq_lengths to cumulative format for TND
        actual_seq_q_cum = []
        actual_seq_kv_cum = []
        cum_q, cum_kv = 0, 0
        for b in range(B):
            cum_q += actual_seq_lengths_query[b]
            cum_kv += actual_seq_lengths_kv[b]
            actual_seq_q_cum.append(cum_q)
            actual_seq_kv_cum.append(cum_kv)

        # Handle RoPE
        if query_rope is not None:
            D_rope = query_rope.shape[-1]
            query_rope_tnd = torch.zeros(T1, N1, D_rope, dtype=dtype)
            key_rope_tnd = torch.zeros(T2, N2, D_rope, dtype=dtype)
            t1_offset = 0
            t2_offset = 0
            for b in range(B):
                s1 = actual_seq_lengths_query[b]
                s2 = actual_seq_lengths_kv[b]
                query_rope_tnd[t1_offset:t1_offset + s1] = query_rope[b, :s1]
                key_rope_tnd[t2_offset:t2_offset + s2] = key_rope[b, :s2]
                t1_offset += s1
                t2_offset += s2
        else:
            query_rope_tnd = torch.empty(0)
            key_rope_tnd = torch.empty(0)
    else:
        # Already TND layout
        query_tnd = query
        key_tnd = key
        value_tnd = value
        out_tnd = attention_out
        dy_tnd = torch.ones_like(attention_out)
        sparse_indices_tnd = sparse_indices
        actual_seq_q_cum = list(np.cumsum(actual_seq_lengths_query))
        actual_seq_kv_cum = list(np.cumsum(actual_seq_lengths_kv))
        query_rope_tnd = query_rope if query_rope is not None else torch.empty(0)
        key_rope_tnd = key_rope if key_rope is not None else torch.empty(0)

    # Create SparseFlashAttentionRef instance
    sfa_ref = SparseFlashAttentionRef(
        query=query_tnd,
        key=key_tnd,
        value=value_tnd,
        out=out_tnd,
        dy=dy_tnd,
        topkIndices=sparse_indices_tnd,
        attenMsk=None,
        actualSeqLenQ=actual_seq_q_cum,
        actualSeqLenKV=actual_seq_kv_cum,
        scaleValue=scale_value,
        blockCount=sparse_indices.shape[-1],
        blockSize=sparse_block_size,
        sparseMode=sparse_mode,
        q_rope=query_rope_tnd,
        k_rope=key_rope_tnd,
        params=query  # for dtype reference
    )

    # Run forward to get softmax_max/sum
    _, _, _ = sfa_ref.forward()

    # Run backward
    dq, dk, dv, dq_rope, dk_rope = sfa_ref.backward()

    return (dq, dk, dv, dq_rope, dk_rope)


# Backward comparison modes
BACKWARD_MODE_SKIP = 0        # Skip backward comparison
BACKWARD_MODE_ORIGINAL = 1    # Use original torch_cpu_forward_backward
BACKWARD_MODE_SFA_REF = 2     # Use SparseFlashAttentionRef


def compare_backward_gradients(ms_b_out, torch_b_out_fp32, torch_b_out, backward_mode=BACKWARD_MODE_ORIGINAL):
    """Compare backward gradients with selectable mode.

    Args:
        ms_b_out: MindSpore backward outputs (dq, dk, dv)
        torch_b_out_fp32: Torch fp32 backward outputs for golden comparison
        torch_b_out: Torch same-dtype backward outputs
        backward_mode: comparison mode
            - BACKWARD_MODE_SKIP (0): skip backward comparison
            - BACKWARD_MODE_ORIGINAL (1): use double_golden_compare
            - BACKWARD_MODE_SFA_REF (2): use SparseFlashAttentionRef based comparison

    Returns:
        bool: True if comparison passes or skipped
    """
    if backward_mode == BACKWARD_MODE_SKIP:
        print("[INFO] Backward comparison skipped (mode=0)")
        return True

    if backward_mode == BACKWARD_MODE_ORIGINAL:
        # Original double golden comparison
        result = True
        result = result and double_golden_compare(torch_b_out_fp32[0], torch_b_out[0], ms_b_out[0])
        result = result and double_golden_compare(torch_b_out_fp32[1], torch_b_out[1], ms_b_out[1])
        result = result and double_golden_compare(torch_b_out_fp32[2], torch_b_out[2], ms_b_out[2])
        return result

    if backward_mode == BACKWARD_MODE_SFA_REF:
        # SparseFlashAttentionRef based comparison - compare ms_b_out with torch_b_out directly
        result = True
        result = result and double_golden_compare(torch_b_out_fp32[0], torch_b_out[0], ms_b_out[0])
        result = result and double_golden_compare(torch_b_out_fp32[1], torch_b_out[1], ms_b_out[1])
        result = result and double_golden_compare(torch_b_out_fp32[2], torch_b_out[2], ms_b_out[2])
        return result

    raise ValueError(f"Unknown backward_mode: {backward_mode}")


def allclose_nparray_tuple(ms_out, expected_out, loss):
    for ms_out_item, expected_out_item in zip(ms_out, expected_out):
        if ms_out_item.dtype == ms.bfloat16:
            allclose_nparray(ms_out_item.float().asnumpy(), expected_out_item.float().numpy(), loss, loss)
        else:
            allclose_nparray(ms_out_item.asnumpy(), expected_out_item.numpy(), loss, loss)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_sparse_flash_attention_functional_general(context_mode, dtype):
    """
    Feature: sparse_flash_attention operator forward/backward accuracy comparison
    Description: Compare MindSpore sparse_flash_attention results with torch_cpu (tolerance-based)
    Expectation: CPU comparison meets tolerance standards
    """
    # Test parameters aligned with ACLNN interface
    B, S1, S2, N1, N2, DN, DR = 4, 1, 8192, 128, 1, 512, 64
    S2_ACT = 4096
    sparse_block_count = 2048
    scale_value = 0.041666666666666664
    sparse_block_size = 1
    attention_mode = 2

    if dtype == 'fp16':
        query_np, key_np, value_np = get_qkv_bsnd(B, S1, S2, N1, N2, DN, dtype=np.float16)
        query_rope_np, key_rope_np = get_rope_tensors(B, S1, S2, N1, N2, DR, dtype=np.float16)
        ms_dtype = ms.float16
    else:
        query_np, key_np, value_np = get_qkv_bsnd(B, S1, S2, N1, N2, DN, dtype='bf16')
        query_rope_np, key_rope_np = get_rope_tensors(B, S1, S2, N1, N2, DR, dtype='bf16')
        ms_dtype = ms.bfloat16

    sparse_indices_np = get_sparse_indices(B, S1, N2, sparse_block_count, S2_ACT - S1 + 1, seed=42)
    actual_seq_q = [S1] * B
    actual_seq_kv = [S2_ACT] * B

    # MindSpore execution with return_softmax_lse=True for full validation
    ms_f_out, ms_b_out = _ms_forward_backward(
        context_mode, query_np, key_np, value_np, sparse_indices_np,
        scale_value, actual_seq_q, actual_seq_kv,
        query_rope_np, key_rope_np, sparse_block_size,
        "BSND", "BSND", 3, attention_mode, ms_dtype, return_softmax_lse=True)

    # Torch CPU same dtype for forward and backward comparison
    pt_query, pt_key, pt_value, pt_sparse_idx, pt_qr, pt_kr = get_torch_tensors(
        query_np, key_np, value_np, sparse_indices_np, query_rope_np, key_rope_np, dtype, target_dtype=dtype)
    torch_f_out, torch_b_out = sfa_ref_forward_backward(
        pt_query, pt_key, pt_value, pt_sparse_idx, scale_value, sparse_block_size,
        list(actual_seq_q), list(actual_seq_kv), pt_qr, pt_kr, sparse_mode=3)

    # Torch CPU fp32 for backward double golden comparison
    pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_sparse_idx_fp32, pt_qr_fp32, pt_kr_fp32 = get_torch_tensors(
        query_np, key_np, value_np, sparse_indices_np, query_rope_np, key_rope_np, dtype, target_dtype='fp32')
    _, torch_b_out_fp32 = sfa_ref_forward_backward(
        pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_sparse_idx_fp32, scale_value, sparse_block_size,
        list(actual_seq_q), list(actual_seq_kv), pt_qr_fp32, pt_kr_fp32, sparse_mode=3)

    # Compare all forward outputs using allclose_nparray (same dtype)
    if dtype == 'bf16':
        ms_attention_np = ms_f_out[0].float().asnumpy()
        ms_softmax_max_np = ms_f_out[1].float().asnumpy()
        ms_softmax_sum_np = ms_f_out[2].float().asnumpy()
    else:
        ms_attention_np = ms_f_out[0].asnumpy()
        ms_softmax_max_np = ms_f_out[1].asnumpy()
        ms_softmax_sum_np = ms_f_out[2].asnumpy()

    torch_attention_np = torch_f_out[0].float().numpy()
    torch_softmax_max_np = torch_f_out[1].float().numpy()
    torch_softmax_sum_np = torch_f_out[2].float().numpy()

    rtol, atol = (BF16_RTOL, BF16_ATOL) if dtype == 'bf16' else (FP16_RTOL, FP16_ATOL)
    allclose_nparray(torch_attention_np, ms_attention_np, rtol, atol)
    allclose_nparray(torch_softmax_max_np, ms_softmax_max_np, rtol, atol)
    allclose_nparray(torch_softmax_sum_np, ms_softmax_sum_np, rtol, atol)

    # Compare backward gradients using double_golden_compare (torch fp32, torch dtype, ms dtype)
    assert double_golden_compare(torch_b_out_fp32[0], torch_b_out[0], ms_b_out[0])
    assert double_golden_compare(torch_b_out_fp32[1], torch_b_out[1], ms_b_out[1])
    assert double_golden_compare(torch_b_out_fp32[2], torch_b_out[2], ms_b_out[2])


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sparse_flash_attention_constraints_validation(context_mode):
    """
    Feature: SparseFlashAttention ACLNN constraints validation
    Description: Verify all test cases conform to ACLNN interface constraints
    Expectation: All parameters meet ACLNN interface requirements
    """
    B, S1, S2, N1, N2, DN, DR = 4, 1, 8192, 128, 1, 512, 64
    sparse_block_count = 2048
    scale_value = 1.0 / (DN ** 0.5)
    sparse_block_size = 1  # Must be in [1, 128] and power of 2

    query_np, key_np, value_np = get_qkv_bsnd(B, S1, S2, N1, N2, DN, dtype=np.float16)
    query_rope_np, key_rope_np = get_rope_tensors(B, S1, S2, N1, N2, DR, dtype=np.float16)
    sparse_indices_np = get_sparse_indices(B, S1, N2, sparse_block_count, S2, seed=42)
    actual_seq_q = [S1] * B
    actual_seq_kv = [S2] * B

    # Full validation with RoPE
    ms_f_out, ms_b_out = _ms_forward_backward(
        context_mode, query_np, key_np, value_np, sparse_indices_np,
        scale_value, actual_seq_q, actual_seq_kv,
        query_rope_np, key_rope_np, sparse_block_size,
        "BSND", "BSND", 3, attention_mode=2,
        dtype=ms.float16, return_softmax_lse=True)

    # Torch CPU fp16 for forward and backward comparison
    pt_query, pt_key, pt_value, pt_sparse_idx, pt_qr, pt_kr = get_torch_tensors(
        query_np, key_np, value_np, sparse_indices_np, query_rope_np, key_rope_np, 'fp16', target_dtype='fp16')
    torch_f_out, torch_b_out = sfa_ref_forward_backward(
        pt_query, pt_key, pt_value, pt_sparse_idx, scale_value, sparse_block_size,
        list(actual_seq_q), list(actual_seq_kv), pt_qr, pt_kr, sparse_mode=3)

    # Torch CPU fp32 for backward double golden comparison
    pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_sparse_idx_fp32, pt_qr_fp32, pt_kr_fp32 = get_torch_tensors(
        query_np, key_np, value_np, sparse_indices_np, query_rope_np, key_rope_np, 'fp16', target_dtype='fp32')
    _, torch_b_out_fp32 = sfa_ref_forward_backward(
        pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_sparse_idx_fp32, scale_value, sparse_block_size,
        list(actual_seq_q), list(actual_seq_kv), pt_qr_fp32, pt_kr_fp32, sparse_mode=3)

    # Compare all forward outputs
    allclose_nparray(torch_f_out[0].float().numpy(), ms_f_out[0].asnumpy(), FP16_RTOL, FP16_ATOL)
    allclose_nparray(torch_f_out[1].float().numpy(), ms_f_out[1].asnumpy(), FP16_RTOL, FP16_ATOL)
    allclose_nparray(torch_f_out[2].float().numpy(), ms_f_out[2].asnumpy(), FP16_RTOL, FP16_ATOL)

    # Compare backward gradients using double_golden_compare
    assert double_golden_compare(torch_b_out_fp32[0], torch_b_out[0], ms_b_out[0])
    assert double_golden_compare(torch_b_out_fp32[1], torch_b_out[1], ms_b_out[1])
    assert double_golden_compare(torch_b_out_fp32[2], torch_b_out[2], ms_b_out[2])


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sparse_flash_attention_exact_match(context_mode):
    """
    Feature: SparseFlashAttention exact match verification
    Description: Verify MindSpore implementation matches torch_cpu (tolerance-based)
    Expectation: CPU comparison meets tolerance standards
    """
    # Parameters aligned with ACLNN interface
    B, S1, S2, N1, N2, DN, DR = 4, 1, 8192, 128, 1, 512, 64
    S2_ACT = 4096
    sparse_block_count = 2048
    scale_value = 0.041666666666666664
    sparse_block_size = 1
    attention_mode = 2

    np.random.seed(42)
    random.seed(42)

    query_np, key_np, value_np = get_qkv_bsnd(B, S1, S2, N1, N2, DN, dtype=np.float16)
    query_rope_np, key_rope_np = get_rope_tensors(B, S1, S2, N1, N2, DR, dtype=np.float16)
    sparse_indices_np = get_sparse_indices(B, S1, N2, sparse_block_count, S2_ACT - S1 + 1, seed=42)
    actual_seq_q = [S1] * B
    actual_seq_kv = [S2_ACT] * B

    # return_softmax_lse=True for full validation
    ms_f_out, ms_b_out = _ms_forward_backward(
        context_mode, query_np, key_np, value_np, sparse_indices_np,
        scale_value, actual_seq_q, actual_seq_kv,
        query_rope_np, key_rope_np, sparse_block_size,
        "BSND", "BSND", 3, attention_mode, ms.float16, return_softmax_lse=True)

    # Torch CPU fp16 for forward and backward comparison
    pt_query, pt_key, pt_value, pt_sparse_idx, pt_qr, pt_kr = get_torch_tensors(
        query_np, key_np, value_np, sparse_indices_np, query_rope_np, key_rope_np, 'fp16', target_dtype='fp16')
    torch_f_out, torch_b_out = sfa_ref_forward_backward(
        pt_query, pt_key, pt_value, pt_sparse_idx, scale_value, sparse_block_size,
        list(actual_seq_q), list(actual_seq_kv), pt_qr, pt_kr, sparse_mode=3)

    # Torch CPU fp32 for backward double golden comparison
    pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_sparse_idx_fp32, pt_qr_fp32, pt_kr_fp32 = get_torch_tensors(
        query_np, key_np, value_np, sparse_indices_np, query_rope_np, key_rope_np, 'fp16', target_dtype='fp32')
    _, torch_b_out_fp32 = sfa_ref_forward_backward(
        pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_sparse_idx_fp32, scale_value, sparse_block_size,
        list(actual_seq_q), list(actual_seq_kv), pt_qr_fp32, pt_kr_fp32, sparse_mode=3)

    # Compare all forward outputs using allclose_nparray
    allclose_nparray(torch_f_out[0].float().numpy(), ms_f_out[0].asnumpy(), FP16_RTOL, FP16_ATOL)
    allclose_nparray(torch_f_out[1].float().numpy(), ms_f_out[1].asnumpy(), FP16_RTOL, FP16_ATOL)
    allclose_nparray(torch_f_out[2].float().numpy(), ms_f_out[2].asnumpy(), FP16_RTOL, FP16_ATOL)

    # Compare backward gradients using double_golden_compare
    assert double_golden_compare(torch_b_out_fp32[0], torch_b_out[0], ms_b_out[0])
    assert double_golden_compare(torch_b_out_fp32[1], torch_b_out[1], ms_b_out[1])
    assert double_golden_compare(torch_b_out_fp32[2], torch_b_out[2], ms_b_out[2])


@pytest.mark.skip(reason="inf input not fully supported")
@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sparse_flash_attention_inf_inputs(context_mode):
    """
    Feature: Robustness with special values (inf)
    Description: Forward run with an inf value in input
    Expectation: Produces valid output tensors with expected shapes
    """
    B, S1, S2, N1, N2, DN, DR = 4, 1, 8192, 128, 1, 512, 64
    sparse_block_count = 2048
    scale_value = 1.0 / (DN ** 0.5)

    query_np, key_np, value_np = get_qkv_bsnd(B, S1, S2, N1, N2, DN, dtype=np.float16)
    query_np[0, 0, 0, 0] = np.inf  # Inject inf
    query_rope_np, key_rope_np = get_rope_tensors(B, S1, S2, N1, N2, DR, dtype=np.float16)

    sparse_indices_np = get_sparse_indices(B, S1, N2, sparse_block_count, S2, seed=42)
    actual_seq_q = [S1] * B
    actual_seq_kv = [S2] * B

    ms_f_out, _ = _ms_forward_backward(
        context_mode, query_np, key_np, value_np, sparse_indices_np,
        scale_value, actual_seq_q, actual_seq_kv,
        query_rope_np, key_rope_np, sparse_block_size=1,
        layout_query="BSND", layout_kv="BSND", sparse_mode=3,
        attention_mode=2, dtype=ms.float16)

    assert ms_f_out[0].shape == (B, S1, N1, DN)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_flash_attention_dynamic_shape_TEST_OP():
    """
    Feature: Dynamic-shape verification via TEST_OP
    Description: Run two representative input sets to exercise dynamic behavior
    Expectation: Executes successfully with correct output shapes
    """
    # Fixed dimensions (must be consistent between cases)
    N1, N2, D, D_rope = 128, 1, 512, 64
    S1 = 1  # Query sequence length (fixed)
    sparse_count = 2048  # Sparse indices count (fixed)
    scale = 1.0 / (D ** 0.5)

    # Case 1: Large batch, large S2
    B1, S2_1 = 4, 8192
    query1, key1, value1 = get_qkv_bsnd(B1, S1, S2_1, N1, N2, D, dtype=np.float16, seed=1)
    query_rope1, key_rope1 = get_rope_tensors(B1, S1, S2_1, N1, N2, D_rope, dtype=np.float16)
    sparse_idx1 = get_sparse_indices(B1, S1, N2, sparse_count, S2_1, seed=1)
    actual_seq_q1 = Tensor([S1] * B1, dtype=ms.int32)
    actual_seq_kv1 = Tensor([S2_1] * B1, dtype=ms.int32)

    # Case 2: Small batch, small S2
    B2, S2_2 = 2, 4096
    query2, key2, value2 = get_qkv_bsnd(B2, S1, S2_2, N1, N2, D, dtype=np.float16, seed=2)
    query_rope2, key_rope2 = get_rope_tensors(B2, S1, S2_2, N1, N2, D_rope, dtype=np.float16)
    sparse_idx2 = get_sparse_indices(B2, S1, N2, sparse_count, S2_2, seed=2)
    actual_seq_q2 = Tensor([S1] * B2, dtype=ms.int32)
    actual_seq_kv2 = Tensor([S2_2] * B2, dtype=ms.int32)

    TEST_OP(
        sparse_flash_attention_forward_func,
        [[Tensor(query1, ms.float16), Tensor(key1, ms.float16), Tensor(value1, ms.float16),
          Tensor(sparse_idx1, ms.int32), scale, None, actual_seq_q1, actual_seq_kv1,
          Tensor(query_rope1, ms.float16), Tensor(key_rope1, ms.float16),
          1, "BSND", "BSND", 3, (1 << 63) - 1, (1 << 63) - 1, 2, False],
         [Tensor(query2, ms.float16), Tensor(key2, ms.float16), Tensor(value2, ms.float16),
          Tensor(sparse_idx2, ms.int32), scale, None, actual_seq_q2, actual_seq_kv2,
          Tensor(query_rope2, ms.float16), Tensor(key_rope2, ms.float16),
          1, "BSND", "BSND", 3, (1 << 63) - 1, (1 << 63) - 1, 2, False]],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=["EmptyTensor", "ScalarTensor"],
        case_config={"disable_input_check": True, "disable_grad": True, "ignore_output_index": [1, 2]},
    )


@pytest.mark.skip(reason="attention_mode=0 not supported, only attention_mode=2 (MLA-absorb) is supported")
@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_sparse_flash_attention_attention_mode_0(context_mode, dtype):
    """
    Feature: sparse_flash_attention with attention_mode=0 (normal mode)
    Description: Test forward/backward with attention_mode=0 and RoPE
    Expectation: SKIP - attention_mode=0 is not supported, only attention_mode=2 is supported
    """
    B, S1, S2, N1, N2, DN, DR = 4, 1, 4096, 64, 1, 512, 64
    sparse_block_count = 1024
    scale_value = 1.0 / (DN ** 0.5)
    attention_mode = 0  # normal mode - NOT SUPPORTED

    if dtype == 'fp16':
        query_np, key_np, value_np = get_qkv_bsnd(B, S1, S2, N1, N2, DN, dtype=np.float16)
        query_rope_np, key_rope_np = get_rope_tensors(B, S1, S2, N1, N2, DR, dtype=np.float16)
        ms_dtype = ms.float16
    else:
        query_np, key_np, value_np = get_qkv_bsnd(B, S1, S2, N1, N2, DN, dtype='bf16')
        query_rope_np, key_rope_np = get_rope_tensors(B, S1, S2, N1, N2, DR, dtype='bf16')
        ms_dtype = ms.bfloat16

    sparse_indices_np = get_sparse_indices(B, S1, N2, sparse_block_count, S2, seed=42)
    actual_seq_q = [S1] * B
    actual_seq_kv = [S2] * B

    # return_softmax_lse=True for full validation
    ms_f_out, ms_b_out = _ms_forward_backward(
        context_mode, query_np, key_np, value_np, sparse_indices_np,
        scale_value, actual_seq_q, actual_seq_kv,
        query_rope_np, key_rope_np, sparse_block_size=1,
        layout_query="BSND", layout_kv="BSND", sparse_mode=3,
        attention_mode=attention_mode, dtype=ms_dtype, return_softmax_lse=True)

    # Torch CPU same dtype for forward and backward comparison
    pt_query, pt_key, pt_value, pt_sparse_idx, pt_qr, pt_kr = get_torch_tensors(
        query_np, key_np, value_np, sparse_indices_np, query_rope_np, key_rope_np, dtype, target_dtype=dtype)
    torch_f_out, torch_b_out = sfa_ref_forward_backward(
        pt_query, pt_key, pt_value, pt_sparse_idx, scale_value, 1,
        list(actual_seq_q), list(actual_seq_kv), pt_qr, pt_kr, sparse_mode=3)

    # Torch CPU fp32 for backward double golden comparison
    pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_sparse_idx_fp32, pt_qr_fp32, pt_kr_fp32 = get_torch_tensors(
        query_np, key_np, value_np, sparse_indices_np, query_rope_np, key_rope_np, dtype, target_dtype='fp32')
    _, torch_b_out_fp32 = sfa_ref_forward_backward(
        pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_sparse_idx_fp32, scale_value, 1,
        list(actual_seq_q), list(actual_seq_kv), pt_qr_fp32, pt_kr_fp32, sparse_mode=3)

    # Compare all forward outputs
    rtol, atol = (BF16_RTOL, BF16_ATOL) if dtype == 'bf16' else (FP16_RTOL, FP16_ATOL)
    if dtype == 'bf16':
        allclose_nparray(torch_f_out[0].float().numpy(), ms_f_out[0].float().asnumpy(), rtol, atol)
        allclose_nparray(torch_f_out[1].float().numpy(), ms_f_out[1].float().asnumpy(), rtol, atol)
        allclose_nparray(torch_f_out[2].float().numpy(), ms_f_out[2].float().asnumpy(), rtol, atol)
    else:
        allclose_nparray(torch_f_out[0].float().numpy(), ms_f_out[0].asnumpy(), rtol, atol)
        allclose_nparray(torch_f_out[1].float().numpy(), ms_f_out[1].asnumpy(), rtol, atol)
        allclose_nparray(torch_f_out[2].float().numpy(), ms_f_out[2].asnumpy(), rtol, atol)

    # Compare backward gradients using double_golden_compare
    assert double_golden_compare(torch_b_out_fp32[0], torch_b_out[0], ms_b_out[0])
    assert double_golden_compare(torch_b_out_fp32[1], torch_b_out[1], ms_b_out[1])
    assert double_golden_compare(torch_b_out_fp32[2], torch_b_out[2], ms_b_out[2])


@pytest.mark.skip(reason="temporarily skipped")
@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('n1', [1, 8, 32, 64, 128])
def test_sparse_flash_attention_different_n1(context_mode, n1):
    """
    Feature: sparse_flash_attention with different N1 values
    Description: Test with N1 = 1/8/32/64/128 (ACLNN supports 1/2/4/8/16/32/64/128) with RoPE
    Expectation: Executes successfully with correct output shapes
    """
    B, S1, S2, N2, DN, DR = 2, 1, 2048, 1, 512, 64
    sparse_block_count = 512
    scale_value = 1.0 / (DN ** 0.5)

    query_np, key_np, value_np = get_qkv_bsnd(B, S1, S2, n1, N2, DN, dtype=np.float16)
    query_rope_np, key_rope_np = get_rope_tensors(B, S1, S2, n1, N2, DR, dtype=np.float16)
    sparse_indices_np = get_sparse_indices(B, S1, N2, sparse_block_count, S2, seed=42)
    actual_seq_q = [S1] * B
    actual_seq_kv = [S2] * B

    # return_softmax_lse=True for full validation
    ms_f_out, ms_b_out = _ms_forward_backward(
        context_mode, query_np, key_np, value_np, sparse_indices_np,
        scale_value, actual_seq_q, actual_seq_kv,
        query_rope_np, key_rope_np, sparse_block_size=1,
        layout_query="BSND", layout_kv="BSND", sparse_mode=3,
        attention_mode=2, dtype=ms.float16, return_softmax_lse=True)

    # Torch CPU fp16 for forward and backward comparison
    pt_query, pt_key, pt_value, pt_sparse_idx, pt_qr, pt_kr = get_torch_tensors(
        query_np, key_np, value_np, sparse_indices_np, query_rope_np, key_rope_np, 'fp16', target_dtype='fp16')
    torch_f_out, torch_b_out = sfa_ref_forward_backward(
        pt_query, pt_key, pt_value, pt_sparse_idx, scale_value, 1,
        list(actual_seq_q), list(actual_seq_kv), pt_qr, pt_kr, sparse_mode=3)

    # Torch CPU fp32 for backward double golden comparison
    pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_sparse_idx_fp32, pt_qr_fp32, pt_kr_fp32 = get_torch_tensors(
        query_np, key_np, value_np, sparse_indices_np, query_rope_np, key_rope_np, 'fp16', target_dtype='fp32')
    _, torch_b_out_fp32 = sfa_ref_forward_backward(
        pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_sparse_idx_fp32, scale_value, 1,
        list(actual_seq_q), list(actual_seq_kv), pt_qr_fp32, pt_kr_fp32, sparse_mode=3)

    # Compare all forward outputs
    allclose_nparray(torch_f_out[0].float().numpy(), ms_f_out[0].asnumpy(), FP16_RTOL, FP16_ATOL)
    allclose_nparray(torch_f_out[1].float().numpy(), ms_f_out[1].asnumpy(), FP16_RTOL, FP16_ATOL)
    allclose_nparray(torch_f_out[2].float().numpy(), ms_f_out[2].asnumpy(), FP16_RTOL, FP16_ATOL)

    # Compare backward gradients using double_golden_compare
    assert double_golden_compare(torch_b_out_fp32[0], torch_b_out[0], ms_b_out[0])
    assert double_golden_compare(torch_b_out_fp32[1], torch_b_out[1], ms_b_out[1])
    assert double_golden_compare(torch_b_out_fp32[2], torch_b_out[2], ms_b_out[2])


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('sparse_block_size', [4, 8, 16, 32, 64, 128])
def test_sparse_flash_attention_different_sparse_block_size(context_mode, sparse_block_size):
    """
    Feature: sparse_flash_attention with different sparse_block_size values
    Description: Test with sparse_block_size = 1/2/4/8/16/32/64/128 (must be power of 2) with RoPE
    Expectation: Executes successfully with correct output shapes
    """
    B, S1, S2, N1, N2, DN, DR = 2, 1, 2048, 64, 1, 512, 64
    # Adjust sparse_block_count based on sparse_block_size to ensure valid indices
    sparse_block_count = min(512, S2 // sparse_block_size)
    scale_value = 1.0 / (DN ** 0.5)

    query_np, key_np, value_np = get_qkv_bsnd(B, S1, S2, N1, N2, DN, dtype=np.float16)
    query_rope_np, key_rope_np = get_rope_tensors(B, S1, S2, N1, N2, DR, dtype=np.float16)
    sparse_indices_np = get_sparse_indices(B, S1, N2, sparse_block_count, S2 // sparse_block_size, seed=42)
    actual_seq_q = [S1] * B
    actual_seq_kv = [S2] * B

    # return_softmax_lse=True for full validation
    ms_f_out, ms_b_out = _ms_forward_backward(
        context_mode, query_np, key_np, value_np, sparse_indices_np,
        scale_value, actual_seq_q, actual_seq_kv,
        query_rope_np, key_rope_np, sparse_block_size=sparse_block_size,
        layout_query="BSND", layout_kv="BSND", sparse_mode=3,
        attention_mode=2, dtype=ms.float16, return_softmax_lse=True)

    # Torch CPU fp16 for forward and backward comparison
    pt_query, pt_key, pt_value, pt_sparse_idx, pt_qr, pt_kr = get_torch_tensors(
        query_np, key_np, value_np, sparse_indices_np, query_rope_np, key_rope_np, 'fp16', target_dtype='fp16')
    torch_f_out, torch_b_out = sfa_ref_forward_backward(
        pt_query, pt_key, pt_value, pt_sparse_idx, scale_value, sparse_block_size,
        list(actual_seq_q), list(actual_seq_kv), pt_qr, pt_kr, sparse_mode=3)

    # Torch CPU fp32 for backward double golden comparison
    pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_sparse_idx_fp32, pt_qr_fp32, pt_kr_fp32 = get_torch_tensors(
        query_np, key_np, value_np, sparse_indices_np, query_rope_np, key_rope_np, 'fp16', target_dtype='fp32')
    _, torch_b_out_fp32 = sfa_ref_forward_backward(
        pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_sparse_idx_fp32, scale_value, sparse_block_size,
        list(actual_seq_q), list(actual_seq_kv), pt_qr_fp32, pt_kr_fp32, sparse_mode=3)

    # Compare all forward outputs
    allclose_nparray(torch_f_out[0].float().numpy(), ms_f_out[0].asnumpy(), FP16_RTOL, FP16_ATOL)
    allclose_nparray(torch_f_out[1].float().numpy(), ms_f_out[1].asnumpy(), FP16_RTOL, FP16_ATOL)
    allclose_nparray(torch_f_out[2].float().numpy(), ms_f_out[2].asnumpy(), FP16_RTOL, FP16_ATOL)

    # Compare backward gradients using double_golden_compare
    assert double_golden_compare(torch_b_out_fp32[0], torch_b_out[0], ms_b_out[0])
    assert double_golden_compare(torch_b_out_fp32[1], torch_b_out[1], ms_b_out[1])
    assert double_golden_compare(torch_b_out_fp32[2], torch_b_out[2], ms_b_out[2])


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('n1', [2, 4, 16])
def test_sparse_flash_attention_different_n1_extra(context_mode, n1):
    """
    Feature: sparse_flash_attention with N1 = 2/4/16
    Description: Test with N1 = 2/4/16 (supplement to cover all ACLNN supported N1 values) with RoPE
    Expectation: Executes successfully with correct output
    """
    B, S1, S2, N2, DN, DR = 2, 1, 2048, 1, 512, 64
    sparse_block_count = 512
    scale_value = 1.0 / (DN ** 0.5)

    query_np, key_np, value_np = get_qkv_bsnd(B, S1, S2, n1, N2, DN, dtype=np.float16)
    query_rope_np, key_rope_np = get_rope_tensors(B, S1, S2, n1, N2, DR, dtype=np.float16)
    sparse_indices_np = get_sparse_indices(B, S1, N2, sparse_block_count, S2, seed=42)
    actual_seq_q = [S1] * B
    actual_seq_kv = [S2] * B

    # return_softmax_lse=True for full validation
    ms_f_out, ms_b_out = _ms_forward_backward(
        context_mode, query_np, key_np, value_np, sparse_indices_np,
        scale_value, actual_seq_q, actual_seq_kv,
        query_rope_np, key_rope_np, sparse_block_size=1,
        layout_query="BSND", layout_kv="BSND", sparse_mode=3,
        attention_mode=2, dtype=ms.float16, return_softmax_lse=True)

    # Torch CPU fp16 for forward and backward comparison
    pt_query, pt_key, pt_value, pt_sparse_idx, pt_qr, pt_kr = get_torch_tensors(
        query_np, key_np, value_np, sparse_indices_np, query_rope_np, key_rope_np, 'fp16', target_dtype='fp16')
    torch_f_out, torch_b_out = sfa_ref_forward_backward(
        pt_query, pt_key, pt_value, pt_sparse_idx, scale_value, 1,
        list(actual_seq_q), list(actual_seq_kv), pt_qr, pt_kr, sparse_mode=3)

    # Torch CPU fp32 for backward double golden comparison
    pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_sparse_idx_fp32, pt_qr_fp32, pt_kr_fp32 = get_torch_tensors(
        query_np, key_np, value_np, sparse_indices_np, query_rope_np, key_rope_np, 'fp16', target_dtype='fp32')
    _, torch_b_out_fp32 = sfa_ref_forward_backward(
        pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_sparse_idx_fp32, scale_value, 1,
        list(actual_seq_q), list(actual_seq_kv), pt_qr_fp32, pt_kr_fp32, sparse_mode=3)

    # Compare all forward outputs
    allclose_nparray(torch_f_out[0].float().numpy(), ms_f_out[0].asnumpy(), FP16_RTOL, FP16_ATOL)
    allclose_nparray(torch_f_out[1].float().numpy(), ms_f_out[1].asnumpy(), FP16_RTOL, FP16_ATOL)
    allclose_nparray(torch_f_out[2].float().numpy(), ms_f_out[2].asnumpy(), FP16_RTOL, FP16_ATOL)

    # Compare backward gradients using double_golden_compare
    assert double_golden_compare(torch_b_out_fp32[0], torch_b_out[0], ms_b_out[0])
    assert double_golden_compare(torch_b_out_fp32[1], torch_b_out[1], ms_b_out[1])
    assert double_golden_compare(torch_b_out_fp32[2], torch_b_out[2], ms_b_out[2])


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sparse_flash_attention_sparse_mode_0(context_mode):
    """
    Feature: sparse_flash_attention with sparse_mode=0 (full mode)
    Description: Test with sparse_mode=0 instead of default sparse_mode=3
    Expectation: Executes successfully with correct output
    """
    B, S1, S2, N1, N2, DN, DR = 4, 1, 4096, 64, 1, 512, 64
    sparse_block_count = 1024
    scale_value = 1.0 / (DN ** 0.5)

    query_np, key_np, value_np = get_qkv_bsnd(B, S1, S2, N1, N2, DN, dtype=np.float16)
    query_rope_np, key_rope_np = get_rope_tensors(B, S1, S2, N1, N2, DR, dtype=np.float16)
    sparse_indices_np = get_sparse_indices(B, S1, N2, sparse_block_count, S2, seed=42)
    actual_seq_q = [S1] * B
    actual_seq_kv = [S2] * B

    # return_softmax_lse=True for full validation with sparse_mode=0
    ms_f_out, _ = _ms_forward_backward(
        context_mode, query_np, key_np, value_np, sparse_indices_np,
        scale_value, actual_seq_q, actual_seq_kv,
        query_rope_np, key_rope_np, sparse_block_size=1,
        layout_query="BSND", layout_kv="BSND", sparse_mode=0,
        attention_mode=2, dtype=ms.float16, return_softmax_lse=True)

    # Torch CPU fp16 for forward comparison (sparse_mode=0)
    pt_query, pt_key, pt_value, pt_sparse_idx, pt_qr, pt_kr = get_torch_tensors(
        query_np, key_np, value_np, sparse_indices_np, query_rope_np, key_rope_np, 'fp16', target_dtype='fp16')
    torch_f_out, _ = sfa_ref_forward_backward(
        pt_query, pt_key, pt_value, pt_sparse_idx, scale_value, 1,
        list(actual_seq_q), list(actual_seq_kv), pt_qr, pt_kr, sparse_mode=0)

    # Compare all forward outputs
    allclose_nparray(torch_f_out[0].float().numpy(), ms_f_out[0].asnumpy(), FP16_RTOL, FP16_ATOL)
    allclose_nparray(torch_f_out[1].float().numpy(), ms_f_out[1].asnumpy(), FP16_RTOL, FP16_ATOL)
    allclose_nparray(torch_f_out[2].float().numpy(), ms_f_out[2].asnumpy(), FP16_RTOL, FP16_ATOL)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sparse_flash_attention_layout_tnd(context_mode):
    """
    Feature: sparse_flash_attention with TND layout
    Description: Test with layout_query="TND" and layout_kv="TND" (no torch comparison)
    Expectation: Executes successfully with correct output shapes
    Note: TND layout test only validates execution, torch golden uses BSND
    """
    B, S1, S2, N1, N2, DN, DR = 2, 1, 2048, 64, 1, 512, 64
    T1 = B * S1  # Total sequence length for query
    T2 = B * S2  # Total sequence length for key/value
    sparse_block_count = 512
    scale_value = 1.0 / (DN ** 0.5)

    np.random.seed(42)
    # TND layout: (T, N, D) where T = B * S
    query_np = np.random.uniform(-10, 10, (T1, N1, DN)).astype(np.float16)
    key_np = np.random.uniform(-5, 10, (T2, N2, DN)).astype(np.float16)
    value_np = key_np.copy()
    query_rope_np = np.random.uniform(-10, 10, (T1, N1, DR)).astype(np.float16)
    key_rope_np = np.random.uniform(-10, 10, (T2, N2, DR)).astype(np.float16)
    # sparse_indices for TND: (T1, N2, sparse_count)
    sparse_indices_np = get_sparse_indices(
        1, T1, N2, sparse_block_count, S2, seed=42).reshape((T1, N2, sparse_block_count))
    actual_seq_q = [S1] * B
    actual_seq_kv = [S2] * B

    context.set_context(mode=context_mode)
    query = Tensor(query_np, dtype=ms.float16)
    key = Tensor(key_np, dtype=ms.float16)
    value = Tensor(value_np, dtype=ms.float16)
    sparse_indices = Tensor(sparse_indices_np, dtype=ms.int32)
    query_rope = Tensor(query_rope_np, dtype=ms.float16)
    key_rope = Tensor(key_rope_np, dtype=ms.float16)
    actual_seq_lengths_query = Tensor(actual_seq_q, dtype=ms.int32)
    actual_seq_lengths_kv = Tensor(actual_seq_kv, dtype=ms.int32)

    # Only validate execution (TND layout)
    f_out = sparse_flash_attention_forward_func(
        query, key, value, sparse_indices, scale_value,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        query_rope=query_rope, key_rope=key_rope,
        sparse_block_size=1,
        layout_query="TND", layout_kv="TND",
        sparse_mode=3, attention_mode=2, return_softmax_lse=True)

    # Verify output shapes for TND layout
    assert f_out[0].shape == (T1, N1, DN), f"attention_out shape mismatch: {f_out[0].shape}"


@pytest.mark.skip(reason="attention_mode=2 requires query_rope and key_rope")
@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_sparse_flash_attention_no_rope(context_mode, dtype):
    """
    Feature: sparse_flash_attention operator without RoPE
    Description: Test forward/backward without query_rope and key_rope
    Expectation: SKIP - attention_mode=2 requires query_rope and key_rope
    """
    B, S1, S2, N1, N2, D = 4, 1, 4096, 64, 1, 512
    sparse_block_count = 1024
    scale_value = 1.0 / (D ** 0.5)

    if dtype == 'fp16':
        query_np, key_np, value_np = get_qkv_bsnd(B, S1, S2, N1, N2, D, dtype=np.float16)
        ms_dtype = ms.float16
    else:
        query_np, key_np, value_np = get_qkv_bsnd(B, S1, S2, N1, N2, D, dtype='bf16')
        ms_dtype = ms.bfloat16

    sparse_indices_np = get_sparse_indices(B, S1, N2, sparse_block_count, S2, seed=42)
    actual_seq_q = [S1] * B
    actual_seq_kv = [S2] * B

    # Only validate MindSpore execution (no torch comparison for no-RoPE scenario)
    ms_f_out, ms_b_out = _ms_forward_backward(
        context_mode, query_np, key_np, value_np, sparse_indices_np,
        scale_value, actual_seq_q, actual_seq_kv,
        query_rope_np=None, key_rope_np=None, sparse_block_size=1,
        layout_query="BSND", layout_kv="BSND", sparse_mode=3,
        attention_mode=2, dtype=ms_dtype, return_softmax_lse=True)

    # Verify output shapes only (no torch comparison for no-RoPE scenario)
    assert ms_f_out[0].shape == (B, S1, N1, D), f"attention_out shape mismatch: {ms_f_out[0].shape}"
    assert ms_b_out[0].shape == query_np.shape, f"query grad shape mismatch: {ms_b_out[0].shape}"
    assert ms_b_out[1].shape == key_np.shape, f"key grad shape mismatch: {ms_b_out[1].shape}"
    assert ms_b_out[2].shape == value_np.shape, f"value grad shape mismatch: {ms_b_out[2].shape}"


@pytest.mark.skip(reason="empty tensor not supported")
@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sparse_flash_attention_empty_input_tensor(context_mode):
    """
    Feature: Empty input tensor validation
    Description: Forward run with empty tensor (S=0)
    Expectation: Produces valid output tensors with expected shapes
    """
    B, S1, S2, N1, N2, D = 4, 0, 0, 128, 1, 512
    query_np = np.zeros((B, S1, N1, D), dtype=np.float16)
    key_np = np.zeros((B, S2, N2, D), dtype=np.float16)
    value_np = key_np.copy()
    sparse_indices_np = np.zeros((B, S1, N2, 1), dtype=np.int32)

    actual_seq_q = [0] * B
    actual_seq_kv = [0] * B

    ms_f_out, _ = _ms_forward_backward(
        context_mode, query_np, key_np, value_np, sparse_indices_np,
        0.1, actual_seq_q, actual_seq_kv, dtype=ms.float16)

    assert ms_f_out[0].shape == (B, S1, N1, D)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sparse_flash_attention_exception_invalid_layout(context_mode):
    """
    Feature: Parameter validation - invalid layout
    Description: Test with invalid layout_query value
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    B, S1, S2, N1, N2, D = 2, 1, 1024, 8, 1, 512
    sparse_block_count = 128

    query = Tensor(np.random.randn(B, S1, N1, D).astype(np.float16))
    key = Tensor(np.random.randn(B, S2, N2, D).astype(np.float16))
    value = key.copy()
    sparse_indices = Tensor(np.random.randint(0, S2, (B, S1, N2, sparse_block_count)).astype(np.int32))

    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.sparse_flash_attention(
            query, key, value, sparse_indices, 0.1,
            layout_query="INVALID", layout_kv="BSND")
        y[0].asnumpy()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sparse_flash_attention_exception_dtype_mismatch(context_mode):
    """
    Feature: Dtype validation - query/key/value dtype mismatch
    Description: query, key, value have different dtypes
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    B, S1, S2, N1, N2, D = 2, 1, 1024, 8, 1, 512
    sparse_block_count = 128

    query = Tensor(np.random.randn(B, S1, N1, D).astype(np.float16))
    key = Tensor(np.random.randn(B, S2, N2, D).astype(np.float32))  # mismatch
    value = key.copy()
    sparse_indices = Tensor(np.random.randint(0, S2, (B, S1, N2, sparse_block_count)).astype(np.int32))

    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.sparse_flash_attention(
            query, key, value, sparse_indices, 0.1,
            layout_query="BSND", layout_kv="BSND")
        y[0].asnumpy()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sparse_flash_attention_exception_dim_mismatch_bsnd(context_mode):
    """
    Feature: Shape validation - wrong dimension for BSND layout
    Description: query has wrong rank for BSND layout
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    B, S1, S2, N1, N2, D = 2, 1, 1024, 8, 1, 512
    sparse_block_count = 128

    # 3D query for BSND layout (should be 4D)
    query = Tensor(np.random.randn(S1, N1, D).astype(np.float16))
    key = Tensor(np.random.randn(B, S2, N2, D).astype(np.float16))
    value = key.copy()
    sparse_indices = Tensor(np.random.randint(0, S2, (B, S1, N2, sparse_block_count)).astype(np.int32))

    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.sparse_flash_attention(
            query, key, value, sparse_indices, 0.1,
            layout_query="BSND", layout_kv="BSND")
        y[0].asnumpy()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sparse_flash_attention_exception_dim_mismatch_tnd(context_mode):
    """
    Feature: Shape validation - wrong dimension for TND layout
    Description: query has wrong rank for TND layout
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    B, S1, S2, N1, N2, D = 2, 1, 1024, 8, 1, 512
    sparse_block_count = 128

    # 4D query for TND layout (should be 3D)
    query = Tensor(np.random.randn(B, S1, N1, D).astype(np.float16))
    key = Tensor(np.random.randn(B, S2, N2, D).astype(np.float16))
    value = key.copy()
    sparse_indices = Tensor(np.random.randint(0, S2, (B, S1, N2, sparse_block_count)).astype(np.int32))

    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.sparse_flash_attention(
            query, key, value, sparse_indices, 0.1,
            layout_query="TND", layout_kv="TND")
        y[0].asnumpy()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('invalid_param', [
    ("sparse_block_size", 129),   # Out of range [1, 128]
    ("sparse_block_size", 0),     # Must be >= 1
    ("sparse_block_size", 3),     # Not power of 2
])
def test_sparse_flash_attention_exception_invalid_parameters(context_mode, invalid_param):
    """
    Feature: Parameter validation
    Description: Test various invalid parameter combinations
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    param_name, param_value = invalid_param

    B, S1, S2, N1, N2, D = 2, 1, 1024, 8, 1, 512
    sparse_block_count = 128
    scale_value = 1.0 / (D ** 0.5)
    sparse_block_size = 1

    if param_name == "sparse_block_size":
        sparse_block_size = param_value

    query = Tensor(np.random.randn(B, S1, N1, D).astype(np.float16))
    key = Tensor(np.random.randn(B, S2, N2, D).astype(np.float16))
    value = key.copy()
    sparse_indices = Tensor(np.random.randint(0, S2, (B, S1, N2, sparse_block_count)).astype(np.int32))

    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.sparse_flash_attention(
            query, key, value, sparse_indices, scale_value,
            sparse_block_size=sparse_block_size,
            layout_query="BSND", layout_kv="BSND")
        y[0].asnumpy()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sparse_flash_attention_exception_empty_sparse_indices(context_mode):
    """
    Feature: Sparse indices validation
    Description: Test with empty sparse_indices tensor
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    B, S1, S2, N1, N2, D = 2, 1, 1024, 8, 1, 512

    query = Tensor(np.random.randn(B, S1, N1, D).astype(np.float16))
    key = Tensor(np.random.randn(B, S2, N2, D).astype(np.float16))
    value = key.copy()
    # Empty sparse_indices (sparse_count = 0)
    sparse_indices = Tensor(np.zeros((B, S1, N2, 0), dtype=np.int32))

    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.sparse_flash_attention(
            query, key, value, sparse_indices, 0.1,
            layout_query="BSND", layout_kv="BSND")
        y[0].asnumpy()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sparse_flash_attention_return_softmax_lse_false(context_mode):
    """
    Feature: sparse_flash_attention with return_softmax_lse=False
    Description: When return_softmax_lse=False, softmax_max and softmax_sum outputs are not meaningful
    Expectation: Only attention_out is validated, softmax_max/sum are ignored
    """
    B, S1, S2, N1, N2, DN, DR = 4, 1, 4096, 64, 1, 512, 64
    sparse_block_count = 1024
    scale_value = 1.0 / (DN ** 0.5)

    query_np, key_np, value_np = get_qkv_bsnd(B, S1, S2, N1, N2, DN, dtype=np.float16)
    query_rope_np, key_rope_np = get_rope_tensors(B, S1, S2, N1, N2, DR, dtype=np.float16)
    sparse_indices_np = get_sparse_indices(B, S1, N2, sparse_block_count, S2, seed=42)
    actual_seq_q = [S1] * B
    actual_seq_kv = [S2] * B

    # return_softmax_lse=False, only validate attention_out
    ms_f_out, _ = _ms_forward_backward(
        context_mode, query_np, key_np, value_np, sparse_indices_np,
        scale_value, actual_seq_q, actual_seq_kv,
        query_rope_np, key_rope_np, sparse_block_size=1,
        layout_query="BSND", layout_kv="BSND", sparse_mode=3,
        attention_mode=2, dtype=ms.float16, return_softmax_lse=False)

    # Torch CPU fp16 for forward comparison
    pt_query, pt_key, pt_value, pt_sparse_idx, pt_qr, pt_kr = get_torch_tensors(
        query_np, key_np, value_np, sparse_indices_np, query_rope_np, key_rope_np, 'fp16', target_dtype='fp16')
    torch_f_out, _ = sfa_ref_forward_backward(
        pt_query, pt_key, pt_value, pt_sparse_idx, scale_value, 1,
        list(actual_seq_q), list(actual_seq_kv), pt_qr, pt_kr, sparse_mode=3,
        return_softmax_lse=False)

    # Only compare attention_out (softmax_max/sum not meaningful when return_softmax_lse=False)
    allclose_nparray(torch_f_out[0].float().numpy(), ms_f_out[0].asnumpy(), FP16_RTOL, FP16_ATOL)
