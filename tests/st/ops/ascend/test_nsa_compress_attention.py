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
"""Test cases for nsa_compress_attention operator."""
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, ops
from mindspore import context
from tests.mark_utils import arg_mark
from tests.st.utils.test_utils import double_golden_compare
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.pynative.utils import allclose_nparray
from tests.st.utils import test_utils
import torch

ms.context.set_context(device_target="Ascend", deterministic='ON', pynative_synchronize=False)
torch.use_deterministic_algorithms(True)

try:
    from einops import rearrange
except ImportError as exc:
    raise ValueError("einops not found") from exc

TOPK_CHECK_OUT_SUFFIX = 2


# ACLNN Interface Requirements:
# - compress_block_size: 16-aligned, range 16-128
# - compress_stride: 16-aligned, range 16-64
# - select_block_size: 16-aligned, range 16-128
# - select_block_count: range 1-32
# - D must be multiple of 16
# - cmp_skv[i] <= 14000
# - compress_block_size >= compress_stride
# - select_block_size >= compress_block_size
# - select_block_size % compress_stride == 0
# - Head dimensions: qD == kD && kD >= vD
# - Head counts: qN >= kN && qN % kN == 0


@test_utils.run_with_cell
def nsa_compress_attention_forward_func(query, key, value, scale_value, head_num, compress_block_size,
                                        compress_stride, select_block_size, select_block_count,
                                        topk_mask=None, atten_mask=None, actual_seq_qlen=None,
                                        actual_cmp_seq_kvlen=None, actual_sel_seq_kvlen=None):
    """MindSpore forward function wrapper"""
    return ops.nsa_compress_attention(query, key, value, scale_value, head_num, compress_block_size,
                                     compress_stride, select_block_size, select_block_count,
                                     topk_mask=topk_mask, atten_mask=atten_mask,
                                     actual_seq_qlen=actual_seq_qlen,
                                     actual_cmp_seq_kvlen=actual_cmp_seq_kvlen,
                                     actual_sel_seq_kvlen=actual_sel_seq_kvlen)


@test_utils.run_with_cell
def nsa_compress_attention_backward_func(query, key, value, scale_value, head_num, compress_block_size,
                                         compress_stride, select_block_size, select_block_count,
                                         topk_mask=None, atten_mask=None, actual_seq_qlen=None,
                                         actual_cmp_seq_kvlen=None, actual_sel_seq_kvlen=None):
    """MindSpore backward function wrapper"""
    return ms.grad(nsa_compress_attention_forward_func, (0, 1, 2))(
        query, key, value, scale_value, head_num, compress_block_size, compress_stride,
        select_block_size, select_block_count, topk_mask, atten_mask,
        actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen)


@ms.jit
def nsa_compress_attention_forward_func_jit(query, key, value, scale_value, head_num, compress_block_size,
                                           compress_stride, select_block_size, select_block_count,
                                           topk_mask, atten_mask, actual_seq_qlen, actual_cmp_seq_kvlen,
                                           actual_sel_seq_kvlen):
    return ops.nsa_compress_attention(query, key, value, scale_value, head_num, compress_block_size,
                                     compress_stride, select_block_size, select_block_count,
                                     topk_mask=topk_mask, atten_mask=atten_mask, actual_seq_qlen=actual_seq_qlen,
                                     actual_cmp_seq_kvlen=actual_cmp_seq_kvlen,
                                     actual_sel_seq_kvlen=actual_sel_seq_kvlen)


@ms.jit
def nsa_compress_attention_backward_func_jit(query, key, value, scale_value, head_num, compress_block_size,
                                            compress_stride, select_block_size, select_block_count,
                                            topk_mask, atten_mask, actual_seq_qlen, actual_cmp_seq_kvlen,
                                            actual_sel_seq_kvlen):
    return ms.grad(nsa_compress_attention_forward_func_jit, (0, 1, 2))(
        query, key, value, scale_value, head_num, compress_block_size,
        compress_stride, select_block_size, select_block_count,
        topk_mask, atten_mask, actual_seq_qlen, actual_cmp_seq_kvlen,
        actual_sel_seq_kvlen
    )


def _ms_forward_backward(context_mode, query_np, key_np, value_np, topk_mask_np, atten_mask_np,
                         scale_value, head_num, compress_block_size, compress_stride,
                         select_block_size, select_block_count, actual_seq_qlen,
                         actual_cmp_seq_kvlen, actual_sel_seq_kvlen, dtype=ms.float16):
    """Run forward and backward in MindSpore with numpy inputs"""
    context.set_context(mode=context_mode)
    query = Tensor(query_np, dtype=dtype)
    key = Tensor(key_np, dtype=dtype)
    value = Tensor(value_np, dtype=dtype)
    topk_mask = Tensor(topk_mask_np, dtype=ms.bool_) if topk_mask_np is not None else None
    atten_mask = Tensor(atten_mask_np, dtype=ms.bool_) if atten_mask_np is not None else None

    f_out = nsa_compress_attention_forward_func(
        query, key, value, scale_value, head_num, compress_block_size,
        compress_stride, select_block_size, select_block_count,
        topk_mask=topk_mask, atten_mask=atten_mask,
        actual_seq_qlen=actual_seq_qlen, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen,
        actual_sel_seq_kvlen=actual_sel_seq_kvlen)

    b_out = nsa_compress_attention_backward_func(
        query, key, value, scale_value, head_num, compress_block_size,
        compress_stride, select_block_size, select_block_count,
        topk_mask=topk_mask, atten_mask=atten_mask,
        actual_seq_qlen=actual_seq_qlen, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen,
        actual_sel_seq_kvlen=actual_sel_seq_kvlen)

    return f_out, b_out


def get_cu_seqlens(seqlens_list):
    """Get cumulative sequence lengths from prefix-sum inputs.

    The incoming list is already a prefix-sum per test usage (e.g., [s1, s1+s2, ...]).
    We simply prepend a leading zero to form the cumulative index array expected by slicing.
    """
    cu = torch.zeros(len(seqlens_list) + 1, dtype=torch.int32)
    if len(seqlens_list) > 0:
        cu[1:] = torch.as_tensor(seqlens_list, dtype=torch.int32)
    return cu


def broadcast_kv(n1, n2, kv_tensor, dtype):
    """Broadcast KV tensor from n2 to n1"""
    factor = n1 // n2
    kv_shape = kv_tensor.shape
    B = kv_shape[0]
    S = kv_shape[2]
    D = kv_shape[3]
    kv_res = torch.zeros([B, n1, S, D]).to(dtype)
    for i in range(n1):
        j = i // factor
        kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
    return kv_res


def tsoftmax(x):
    """Compute softmax with max and sum"""
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    ans = y.div(x_sum)
    return ans, x_max, x_sum


def tforward(q, k, v, atten_mask, scale):
    """Forward computation with native fp16 support."""
    qk = torch.matmul(q.to(torch.float32), k.permute(0, 1, 3, 2).to(torch.float32)).mul(scale).to(q.dtype)
    if atten_mask.numel() == 1 and atten_mask.item() == 0:
        pass
    else:
        if len(atten_mask.shape) == 2:
            atten_mask = atten_mask.unsqueeze(0).unsqueeze(0)
        qk = qk + atten_mask.bool() * (-40000.0)
    softmax_res, x_max, x_sum = tsoftmax(qk)
    y = torch.matmul(softmax_res.to(torch.float32), v.to(torch.float32)).to(q.dtype)
    return y, softmax_res, x_max, x_sum


def compute_impscore(softmax_res, topk_mask, sel_kv_len, m, n, n2, g):
    """Compute importance score for top-k selection with native fp16 support."""
    s1 = int(softmax_res.shape[-2])
    s2 = int(softmax_res.shape[-1])
    outer_loop = sel_kv_len
    inner_loop = m + n - 1
    trans = softmax_res.permute(0, 1, 3, 2)
    sum_res = torch.zeros([int(softmax_res.shape[0]), int(softmax_res.shape[1]),
                          outer_loop, s1], dtype=softmax_res.dtype)
    for i in range(outer_loop, 0, -1):
        dst_idx = i - 1
        for j in range(inner_loop):
            if j < inner_loop / 2:
                times = j + 1
            else:
                times = inner_loop - j
            times = min(times, n)
            if m * i - j >= s2 or m * i - j < 0:
                continue
            sum_res[:, :, dst_idx] += trans[:, :, m * i - j] * times
    trans_back = sum_res.permute(0, 1, 3, 2)
    trans_back = trans_back.reshape(1, n2, g, s1, -1)
    trans_back = trans_back.sum(dim=2)
    if topk_mask is None:
        res = trans_back
    else:
        res = trans_back.masked_fill(topk_mask.to(torch.bool), value=23333.0)
    return res


def torch_cpu_generate_golden(q, k, v, scale_value, head_num, compress_block_size, compress_stride,
                               select_block_size, select_block_count, topk_mask, atten_mask,
                               actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen):
    """Generate golden reference output using torch CPU with native fp16 support."""
    S1 = q.shape[0]
    N1 = q.shape[1]
    D2 = v.shape[2]
    N2 = v.shape[1]
    B = len(actual_cmp_seq_kvlen)
    m = select_block_size // compress_stride
    n = compress_block_size // compress_stride
    G = N1 // N2

    dtype = q.dtype
    atten_out = torch.zeros(S1, N1, D2, dtype=dtype)
    topk_indices = torch.zeros(S1, N2, select_block_count, dtype=torch.int32)
    topk_indices_with_n = torch.zeros(S1, N2, select_block_count + TOPK_CHECK_OUT_SUFFIX, dtype=torch.int32)
    softmax_max = torch.empty(0, dtype=dtype)
    softmax_sum = torch.empty(0, dtype=dtype)

    cu_seqlens_q = get_cu_seqlens(actual_seq_qlen)
    cu_seqlens_k = get_cu_seqlens(actual_cmp_seq_kvlen)

    for i in range(B):
        qi = q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]]
        ki = k[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
        vi = v[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
        qi = rearrange(qi, 's n d -> 1 n s d')
        ki = rearrange(ki, 's n d -> 1 n s d')
        vi = rearrange(vi, 's n d -> 1 n s d')

        if N1 != N2:
            ki = broadcast_kv(N1, N2, ki, ki.dtype)
            vi = broadcast_kv(N1, N2, vi, vi.dtype)

        cur_seq_q_len = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
        cur_cmp_seq_k_len = cu_seqlens_k[i + 1] - cu_seqlens_k[i]
        cur_sel_seq_k = actual_sel_seq_kvlen[i]

        if atten_mask is None:
            atten_maski = torch.tensor([0])
        else:
            atten_maski = atten_mask[:cur_seq_q_len, :cur_cmp_seq_k_len]

        outi_golden, softmax_resi, x_maxi, x_sumi = tforward(qi, ki, vi, atten_maski, scale_value)

        if topk_mask is None:
            topk_maski = None
        else:
            if (topk_mask.shape[0] != cur_seq_q_len) or (topk_mask.shape[1] != cur_sel_seq_k):
                topk_maski = torch.zeros(cur_seq_q_len, cur_sel_seq_k, dtype=torch.bool)
                diag_len = min(cur_seq_q_len, cur_sel_seq_k)
                if diag_len > 0:
                    idx = torch.arange(diag_len)
                    topk_maski[idx, idx] = True
                topk_maski[:, 0] = True
            else:
                topk_maski = topk_mask
        importance_score_i = compute_impscore(softmax_resi, topk_maski,
                                            actual_sel_seq_kvlen[i], m, n, N2, G)
        _, topki_total = torch.sort(importance_score_i, descending=True, stable=True, dim=-1)
        topki = topki_total[:, :, :, :select_block_count]
        topki_with_n = topki_total[:, :, :, :select_block_count + TOPK_CHECK_OUT_SUFFIX]

        atten_out[cu_seqlens_q[i]:cu_seqlens_q[i + 1]] = rearrange(outi_golden, '1 n s d -> s n d')
        topk_indices[cu_seqlens_q[i]:cu_seqlens_q[i + 1]] = rearrange(topki, '1 n s d -> s n d')
        topk_indices_with_n[cu_seqlens_q[i]:cu_seqlens_q[i + 1]] = rearrange(topki_with_n, '1 n s d -> s n d')

        x_maxi = x_maxi.broadcast_to(1, N1, cur_seq_q_len, 8).contiguous().view(-1)
        x_sumi = x_sumi.broadcast_to(1, N1, cur_seq_q_len, 8).contiguous().view(-1)
        softmax_max = torch.cat([softmax_max, x_maxi], dim=0)
        softmax_sum = torch.cat([softmax_sum, x_sumi], dim=0)

    softmax_max = softmax_max.reshape(S1, N1, 8)
    softmax_sum = softmax_sum.reshape(S1, N1, 8)

    return atten_out, topk_indices.to(torch.int32), softmax_max, softmax_sum, topk_indices_with_n.to(torch.int32)


def torch_cpu_forward_backward(query, key, value, scale_value, head_num, compress_block_size,
                                compress_stride, select_block_size, select_block_count,
                                topk_mask=None, atten_mask=None, actual_seq_qlen=None,
                                actual_cmp_seq_kvlen=None, actual_sel_seq_kvlen=None):
    """Torch CPU forward and backward computation"""
    query = query.clone().detach().requires_grad_(True)
    key = key.clone().detach().requires_grad_(True)
    value = value.clone().detach().requires_grad_(True)

    attention_out, topk_indices_out, softmax_max_out, softmax_sum_out, \
        topk_indices_with_n_out = torch_cpu_generate_golden(
            query, key, value, scale_value, head_num, compress_block_size,
            compress_stride, select_block_size, select_block_count, topk_mask,
            atten_mask, actual_seq_qlen, actual_cmp_seq_kvlen,
            actual_sel_seq_kvlen)

    loss = attention_out.sum()
    loss.backward()

    forward_out = (attention_out.detach(), topk_indices_out.detach(),
                  softmax_max_out.detach(), softmax_sum_out.detach(), topk_indices_with_n_out.detach())
    backward_out = (query.grad.detach(), key.grad.detach(), value.grad.detach())

    return forward_out, backward_out


def allclose_nparray_tuple(ms_out, expected_out, loss):
    for ms_out_item, expected_out_item in zip(ms_out, expected_out):
        if ms_out_item.dtype == ms.bfloat16:
            allclose_nparray(ms_out_item.float().asnumpy(), expected_out_item.float().numpy(), loss, loss)
        else:
            allclose_nparray(ms_out_item.asnumpy(), expected_out_item.numpy(), loss, loss)


def get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float16, seed=42):
    """Generate random query, key, value tensors.

    Args:
        T1: Query sequence length
        T2: Key/value sequence length
        N1: Query head number
        N2: Key/value head number
        D1: Query/key head dimension
        D2: Value head dimension
        dtype: Data type (np.float16 or np.float32)
        seed: Random seed (default: 42)
    """
    np.random.seed(seed)
    if dtype == 'bf16':
        query_np = torch.randn(T1, N1, D1, dtype=torch.bfloat16).to(torch.float32).numpy()
        key_np = torch.randn(T2, N2, D1, dtype=torch.bfloat16).to(torch.float32).numpy()
        value_np = torch.randn(T2, N2, D2, dtype=torch.bfloat16).to(torch.float32).numpy()
    else:
        query_np = np.random.randn(T1, N1, D1).astype(dtype)
        key_np = np.random.randn(T2, N2, D1).astype(dtype)
        value_np = np.random.randn(T2, N2, D2).astype(dtype)
    return query_np, key_np, value_np


def get_topk_mask(S1, S2):
    """Generate topk mask tensor with diagonal and first column set to True."""
    topk_mask = np.eye(S1, S2, dtype=np.bool_)
    topk_mask[:, 0] = True
    return topk_mask


def get_atten_mask(S1, S2):
    """Generate attention mask tensor (upper triangular)."""
    atten_mask = np.triu(np.ones((S1, S2), dtype=np.bool_), k=1)
    return atten_mask


def get_top_k_mask(S1, S2):
    """Alias for get_topk_mask for backward compatibility"""
    return get_topk_mask(S1, S2)


def get_torch_tensors(query_np, key_np, value_np, topk_mask_np=None,
                      atten_mask_np=None, dtype='fp16', target_dtype='fp32'):
    """Convert numpy arrays to PyTorch tensors.

    Args:
        query_np: Query numpy array
        key_np: Key numpy array
        value_np: Value numpy array
        topk_mask_np: TopK mask numpy array (optional)
        atten_mask_np: Attention mask numpy array (optional)
        dtype: Source dtype ('fp16' or 'bf16') - used to determine conversion method
        target_dtype: Target dtype for tensors ('fp32', 'fp16', or 'bf16')
    """
    if target_dtype == 'fp32':
        torch_dtype = torch.float32
        pt_query = torch.tensor(query_np, dtype=torch_dtype) if query_np is not None else None
        pt_key = torch.tensor(key_np, dtype=torch_dtype) if key_np is not None else None
        pt_value = torch.tensor(value_np, dtype=torch_dtype) if value_np is not None else None
    elif target_dtype == 'fp16':
        pt_query = torch.from_numpy(query_np).to(torch.float16) if query_np is not None else None
        pt_key = torch.from_numpy(key_np).to(torch.float16) if key_np is not None else None
        pt_value = torch.from_numpy(value_np).to(torch.float16) if value_np is not None else None
    elif target_dtype == 'bf16':
        pt_query = torch.from_numpy(query_np).to(torch.bfloat16) if query_np is not None else None
        pt_key = torch.from_numpy(key_np).to(torch.bfloat16) if key_np is not None else None
        pt_value = torch.from_numpy(value_np).to(torch.bfloat16) if value_np is not None else None
    else:
        torch_dtype = torch.float32
        pt_query = torch.tensor(query_np, dtype=torch_dtype) if query_np is not None else None
        pt_key = torch.tensor(key_np, dtype=torch_dtype) if key_np is not None else None
        pt_value = torch.tensor(value_np, dtype=torch_dtype) if value_np is not None else None

    pt_topk_mask = torch.tensor(topk_mask_np, dtype=torch.bool) if topk_mask_np is not None else None
    pt_atten_mask = torch.tensor(atten_mask_np, dtype=torch.bool) if atten_mask_np is not None else None

    return pt_query, pt_key, pt_value, pt_topk_mask, pt_atten_mask


def check_topk_in_range(golden_topk_with_n, actual_topk):
    """
    Check if all indices in actual_topk are within the top(k+2) range from golden_topk_with_n.

    Args:
        golden_topk_with_n: CPU top(k+2) indices, shape [S1, N2, k+2]
        actual_topk: MindSpore topk indices, shape [S1, N2, k]

    Returns:
        bool: True if all actual_topk indices are in golden_topk_with_n, False otherwise
    """
    golden_topk_with_n_np = (golden_topk_with_n.cpu().numpy()
                             if isinstance(golden_topk_with_n, torch.Tensor)
                             else golden_topk_with_n)
    actual_topk_np = actual_topk.asnumpy() if isinstance(actual_topk, ms.Tensor) else actual_topk

    if golden_topk_with_n_np.shape[:2] != actual_topk_np.shape[:2]:
        print(f"Shape mismatch: golden_topk_with_n={golden_topk_with_n_np.shape}, actual_topk={actual_topk_np.shape}")
        return False

    S1, N2 = actual_topk_np.shape[:2]
    k_plus_2 = golden_topk_with_n_np.shape[2]

    for s in range(S1):
        for n in range(N2):
            golden_set = set(golden_topk_with_n_np[s, n, :])
            actual_indices = actual_topk_np[s, n, :]
            for idx in actual_indices:
                if idx not in golden_set:
                    print(f"Index {idx} at position [{s}, {n}] not in top{k_plus_2} set: {golden_set}")
                    return False
    return True


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=158.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
@pytest.mark.parametrize('use_masks', [True, False])
def test_nsa_compress_attention_functional_general(context_mode, dtype, use_masks):
    """
    Feature: nsa_compress_attention operator forward/backward accuracy comparison
    Description: Compare MindSpore nsa_compress_attention results with torch_cpu (tolerance-based)
    Expectation: CPU comparison meets tolerance standards
    """

    T1, T2 = 2633, 4072
    N1, N2 = 7, 7
    D1, D2 = 8, 7

    if dtype == 'fp16':
        query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float16)
    else:  # bf16
        query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype='bf16')

    scale_value = 0.3
    head_num = 7
    compress_block_size = 32
    compress_stride = 16
    select_block_size = 64
    select_block_count = 16

    S1, S2 = 1384, 567
    if use_masks:
        topk_mask_np = get_topk_mask(S1, S2)
        atten_mask_np = get_atten_mask(1384, 2266)
    else:
        topk_mask_np = None
        atten_mask_np = None

    actual_seq_qlen = [1384, 2633]
    actual_cmp_seq_kvlen = [2266, 4072]
    actual_sel_seq_kvlen = [567, 1019]

    ms_dtype = ms.float16 if dtype == 'fp16' else ms.bfloat16
    pt_dtype = torch.float16 if dtype == 'fp16' else torch.bfloat16
    ms_f_out, ms_b_out = _ms_forward_backward(
        context_mode, query_np, key_np, value_np, topk_mask_np, atten_mask_np,
        scale_value, head_num, compress_block_size, compress_stride,
        select_block_size, select_block_count, actual_seq_qlen,
        actual_cmp_seq_kvlen, actual_sel_seq_kvlen, ms_dtype)

    pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_topk_mask, pt_atten_mask = get_torch_tensors(
        query_np, key_np, value_np, topk_mask_np, atten_mask_np, dtype, target_dtype='fp32')
    torch_f_out_fp32, torch_b_out_fp32 = torch_cpu_forward_backward(
        pt_query_fp32, pt_key_fp32, pt_value_fp32, scale_value, head_num, compress_block_size,
        compress_stride, select_block_size, select_block_count,
        topk_mask=pt_topk_mask, atten_mask=pt_atten_mask,
        actual_seq_qlen=actual_seq_qlen, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen,
        actual_sel_seq_kvlen=actual_sel_seq_kvlen)

    pt_query_gpu, pt_key_gpu, pt_value_gpu, _, _ = get_torch_tensors(
        query_np, key_np, value_np, topk_mask_np, atten_mask_np, dtype, target_dtype=dtype)
    torch_f_out_gpu, torch_b_out_gpu = torch_cpu_forward_backward(
        pt_query_gpu, pt_key_gpu, pt_value_gpu, scale_value, head_num, compress_block_size,
        compress_stride, select_block_size, select_block_count,
        topk_mask=pt_topk_mask, atten_mask=pt_atten_mask,
        actual_seq_qlen=actual_seq_qlen, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen,
        actual_sel_seq_kvlen=actual_sel_seq_kvlen)

    golden_attention = torch_f_out_fp32[0]
    gpu_attention = torch_f_out_gpu[0]
    actual_attention = ms_f_out[0]
    assert double_golden_compare(golden_attention, gpu_attention, actual_attention)

    golden_topk_with_n = torch_f_out_fp32[4]
    actual_topk = ms_f_out[1]
    assert check_topk_in_range(golden_topk_with_n, actual_topk), "topk_indices_out not in top(k+2) range"

    golden_softmax_max = torch_f_out_fp32[2].to(pt_dtype)
    gpu_softmax_max = torch_f_out_gpu[2].to(pt_dtype)
    actual_softmax_max = ms_f_out[2].to(ms_dtype)
    assert double_golden_compare(golden_softmax_max, gpu_softmax_max, actual_softmax_max)

    golden_softmax_sum = torch_f_out_fp32[3].to(pt_dtype)
    gpu_softmax_sum = torch_f_out_gpu[3].to(pt_dtype)
    actual_softmax_sum = ms_f_out[3].to(ms_dtype)
    assert double_golden_compare(golden_softmax_sum, gpu_softmax_sum, actual_softmax_sum)

    golden_grad_query = torch_b_out_fp32[0]
    gpu_grad_query = torch_b_out_gpu[0]
    actual_grad_query = ms_b_out[0]
    assert double_golden_compare(golden_grad_query, gpu_grad_query, actual_grad_query)

    golden_grad_key = torch_b_out_fp32[1]
    gpu_grad_key = torch_b_out_gpu[1]
    actual_grad_key = ms_b_out[1]
    assert double_golden_compare(golden_grad_key, gpu_grad_key, actual_grad_key)

    golden_grad_value = torch_b_out_fp32[2]
    gpu_grad_value = torch_b_out_gpu[2]
    actual_grad_value = ms_b_out[2]
    assert double_golden_compare(golden_grad_value, gpu_grad_value, actual_grad_value)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential", mem_peak=158.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_attention_pta_constraints_validation(context_mode):
    """
    Feature: NSACompressAttention ACLNN constraints validation
    Description: Verify all test cases conform to ACLNN interface constraints, compare with torch_cpu
    Expectation: All parameters meet ACLNN interface requirements; matches CPU within tolerance
    """

    T1, T2 = 2633, 4072
    N1, N2 = 7, 7
    D1, D2 = 8, 7

    # Generate input tensors
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float16)
    ms_dtype = ms.float16
    pt_dtype = torch.float16

    scale_value = 0.3
    head_num = N1

    compress_block_size = 32  # Supported range: 16-128, aligned to 16
    compress_stride = 16      # Supported range: 16-64, aligned to 16
    select_block_size = 64    # Supported range: 16-128, aligned to 16
    select_block_count = 16   # Supported range: 1-32

    S1, S2 = 1384, 567
    topk_mask_np = get_topk_mask(S1, S2)
    atten_mask_np = get_atten_mask(1384, 2266)

    actual_seq_qlen = [1384, 2633]
    actual_cmp_seq_kvlen = [2266, 4072]
    actual_sel_seq_kvlen = [567, 1019]

    ms_f_out, ms_b_out = _ms_forward_backward(
        context_mode, query_np, key_np, value_np, topk_mask_np, atten_mask_np,
        scale_value, head_num, compress_block_size, compress_stride,
        select_block_size, select_block_count, actual_seq_qlen,
        actual_cmp_seq_kvlen, actual_sel_seq_kvlen, ms_dtype)

    dtype_str = 'fp16'
    pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_topk_mask, pt_atten_mask = get_torch_tensors(
        query_np, key_np, value_np, topk_mask_np, atten_mask_np, dtype_str, target_dtype='fp32')
    torch_f_out_fp32, torch_b_out_fp32 = torch_cpu_forward_backward(
        pt_query_fp32, pt_key_fp32, pt_value_fp32, scale_value, head_num, compress_block_size,
        compress_stride, select_block_size, select_block_count,
        topk_mask=pt_topk_mask, atten_mask=pt_atten_mask,
        actual_seq_qlen=actual_seq_qlen, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen,
        actual_sel_seq_kvlen=actual_sel_seq_kvlen)

    pt_query_gpu, pt_key_gpu, pt_value_gpu, _, _ = get_torch_tensors(
        query_np, key_np, value_np, topk_mask_np, atten_mask_np, dtype_str, target_dtype=dtype_str)
    torch_f_out_gpu, torch_b_out_gpu = torch_cpu_forward_backward(
        pt_query_gpu, pt_key_gpu, pt_value_gpu, scale_value, head_num, compress_block_size,
        compress_stride, select_block_size, select_block_count,
        topk_mask=pt_topk_mask, atten_mask=pt_atten_mask,
        actual_seq_qlen=actual_seq_qlen, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen,
        actual_sel_seq_kvlen=actual_sel_seq_kvlen)

    golden_attention = torch_f_out_fp32[0]
    gpu_attention = torch_f_out_gpu[0]
    actual_attention = ms_f_out[0]
    assert double_golden_compare(golden_attention, gpu_attention, actual_attention)

    golden_topk_with_n = torch_f_out_fp32[4]
    actual_topk = ms_f_out[1]
    assert check_topk_in_range(golden_topk_with_n, actual_topk), "topk_indices_out not in top(k+2) range"

    golden_softmax_max = torch_f_out_fp32[2].to(pt_dtype)
    gpu_softmax_max = torch_f_out_gpu[2].to(pt_dtype)
    actual_softmax_max = ms_f_out[2].to(ms_dtype)
    assert double_golden_compare(golden_softmax_max, gpu_softmax_max, actual_softmax_max)

    golden_softmax_sum = torch_f_out_fp32[3].to(pt_dtype)
    gpu_softmax_sum = torch_f_out_gpu[3].to(pt_dtype)
    actual_softmax_sum = ms_f_out[3].to(ms_dtype)
    assert double_golden_compare(golden_softmax_sum, gpu_softmax_sum, actual_softmax_sum)

    golden_grad_query = torch_b_out_fp32[0]
    gpu_grad_query = torch_b_out_gpu[0]
    actual_grad_query = ms_b_out[0]
    assert double_golden_compare(golden_grad_query, gpu_grad_query, actual_grad_query)

    golden_grad_key = torch_b_out_fp32[1]
    gpu_grad_key = torch_b_out_gpu[1]
    actual_grad_key = ms_b_out[1]
    assert double_golden_compare(golden_grad_key, gpu_grad_key, actual_grad_key)

    golden_grad_value = torch_b_out_fp32[2]
    gpu_grad_value = torch_b_out_gpu[2]
    actual_grad_value = ms_b_out[2]
    assert double_golden_compare(golden_grad_value, gpu_grad_value, actual_grad_value)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential", mem_peak=158.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_attention_pta_exact_match(context_mode):
    """
    Feature: NSACompressAttention exact match verification
    Description: Verify MindSpore implementation matches torch_cpu (tolerance-based)
    Expectation: CPU comparison meets tolerance standards
    """

    T1, T2 = 2633, 4072
    N1, N2 = 7, 7
    D1, D2 = 8, 7

    # Generate input tensors
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float16)
    ms_dtype = ms.float16
    pt_dtype = torch.float16

    scale_value = 0.3
    head_num = N1

    compress_block_size = 32  # Supported range: 16-128, aligned to 16
    compress_stride = 16      # Supported range: 16-64, aligned to 16
    select_block_size = 64    # Supported range: 16-128, aligned to 16
    select_block_count = 16   # Supported range: 1-32

    S1, S2 = 1384, 567
    topk_mask_np = get_topk_mask(S1, S2)
    atten_mask_np = get_atten_mask(1384, 2266)

    actual_seq_qlen = [1384, 2633]
    actual_cmp_seq_kvlen = [2266, 4072]
    actual_sel_seq_kvlen = [567, 1019]

    ms_f_out, ms_b_out = _ms_forward_backward(
        context_mode, query_np, key_np, value_np, topk_mask_np, atten_mask_np,
        scale_value, head_num, compress_block_size, compress_stride,
        select_block_size, select_block_count, actual_seq_qlen,
        actual_cmp_seq_kvlen, actual_sel_seq_kvlen, ms_dtype)

    dtype_str = 'fp16'
    pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_topk_mask, pt_atten_mask = get_torch_tensors(
        query_np, key_np, value_np, topk_mask_np, atten_mask_np, dtype_str, target_dtype='fp32')
    torch_f_out_fp32, torch_b_out_fp32 = torch_cpu_forward_backward(
        pt_query_fp32, pt_key_fp32, pt_value_fp32, scale_value, head_num, compress_block_size,
        compress_stride, select_block_size, select_block_count,
        topk_mask=pt_topk_mask, atten_mask=pt_atten_mask,
        actual_seq_qlen=actual_seq_qlen, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen,
        actual_sel_seq_kvlen=actual_sel_seq_kvlen)

    pt_query_gpu, pt_key_gpu, pt_value_gpu, _, _ = get_torch_tensors(
        query_np, key_np, value_np, topk_mask_np, atten_mask_np, dtype_str, target_dtype=dtype_str)
    torch_f_out_gpu, torch_b_out_gpu = torch_cpu_forward_backward(
        pt_query_gpu, pt_key_gpu, pt_value_gpu, scale_value, head_num, compress_block_size,
        compress_stride, select_block_size, select_block_count,
        topk_mask=pt_topk_mask, atten_mask=pt_atten_mask,
        actual_seq_qlen=actual_seq_qlen, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen,
        actual_sel_seq_kvlen=actual_sel_seq_kvlen)

    golden_attention = torch_f_out_fp32[0]
    gpu_attention = torch_f_out_gpu[0]
    actual_attention = ms_f_out[0]
    assert double_golden_compare(golden_attention, gpu_attention, actual_attention)

    golden_topk_with_n = torch_f_out_fp32[4]
    actual_topk = ms_f_out[1]
    assert check_topk_in_range(golden_topk_with_n, actual_topk), "topk_indices_out not in top(k+2) range"

    golden_softmax_max = torch_f_out_fp32[2].to(pt_dtype)
    gpu_softmax_max = torch_f_out_gpu[2].to(pt_dtype)
    actual_softmax_max = ms_f_out[2].to(ms_dtype)
    assert double_golden_compare(golden_softmax_max, gpu_softmax_max, actual_softmax_max)

    golden_softmax_sum = torch_f_out_fp32[3].to(pt_dtype)
    gpu_softmax_sum = torch_f_out_gpu[3].to(pt_dtype)
    actual_softmax_sum = ms_f_out[3].to(ms_dtype)
    assert double_golden_compare(golden_softmax_sum, gpu_softmax_sum, actual_softmax_sum)

    golden_grad_query = torch_b_out_fp32[0]
    gpu_grad_query = torch_b_out_gpu[0]
    actual_grad_query = ms_b_out[0]
    assert double_golden_compare(golden_grad_query, gpu_grad_query, actual_grad_query)

    golden_grad_key = torch_b_out_fp32[1]
    gpu_grad_key = torch_b_out_gpu[1]
    actual_grad_key = ms_b_out[1]
    assert double_golden_compare(golden_grad_key, gpu_grad_key, actual_grad_key)

    golden_grad_value = torch_b_out_fp32[2]
    gpu_grad_value = torch_b_out_gpu[2]
    actual_grad_value = ms_b_out[2]
    assert double_golden_compare(golden_grad_value, gpu_grad_value, actual_grad_value)


@pytest.mark.skip(reason="not supported")
@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=0.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_attention_inf_inputs_forward_backward(context_mode):
    """
    Feature: Robustness with special values (inf)
    Description: Forward and backward run with an inf value in input, compare with torch_cpu
    Expectation: Produces valid output tensors with expected shapes; matches CPU within tolerance
    """
    T1, T2 = 2633, 4072
    N1, N2 = 7, 7
    D1, D2 = 8, 7

    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float16)
    query_np[0, 0, 0] = np.inf

    scale_value = 0.3
    head_num = N1

    compress_block_size = 32
    compress_stride = 16
    select_block_size = 64
    select_block_count = 16

    S1, S2 = 1384, 567
    topk_mask_np = get_topk_mask(S1, S2)
    atten_mask_np = get_atten_mask(1384, 2266)

    actual_seq_qlen = [1384, 2633]
    actual_cmp_seq_kvlen = [2266, 4072]
    actual_sel_seq_kvlen = [567, 1019]

    ms_dtype = ms.float16
    pt_dtype = torch.float16
    ms_f_out, ms_b_out = _ms_forward_backward(
        context_mode, query_np, key_np, value_np, topk_mask_np, atten_mask_np,
        scale_value, head_num, compress_block_size, compress_stride,
        select_block_size, select_block_count, actual_seq_qlen,
        actual_cmp_seq_kvlen, actual_sel_seq_kvlen, ms_dtype)

    dtype_str = 'fp16'
    pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_topk_mask, pt_atten_mask = get_torch_tensors(
        query_np, key_np, value_np, topk_mask_np, atten_mask_np, dtype_str, target_dtype='fp32')
    torch_f_out_fp32, torch_b_out_fp32 = torch_cpu_forward_backward(
        pt_query_fp32, pt_key_fp32, pt_value_fp32, scale_value, head_num, compress_block_size,
        compress_stride, select_block_size, select_block_count,
        topk_mask=pt_topk_mask, atten_mask=pt_atten_mask,
        actual_seq_qlen=actual_seq_qlen, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen,
        actual_sel_seq_kvlen=actual_sel_seq_kvlen)

    pt_query_gpu, pt_key_gpu, pt_value_gpu, _, _ = get_torch_tensors(
        query_np, key_np, value_np, topk_mask_np, atten_mask_np, dtype_str, target_dtype=dtype_str)
    torch_f_out_gpu, torch_b_out_gpu = torch_cpu_forward_backward(
        pt_query_gpu, pt_key_gpu, pt_value_gpu, scale_value, head_num, compress_block_size,
        compress_stride, select_block_size, select_block_count,
        topk_mask=pt_topk_mask, atten_mask=pt_atten_mask,
        actual_seq_qlen=actual_seq_qlen, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen,
        actual_sel_seq_kvlen=actual_sel_seq_kvlen)

    golden_attention = torch_f_out_fp32[0]
    gpu_attention = torch_f_out_gpu[0]
    actual_attention = ms_f_out[0]
    assert double_golden_compare(golden_attention, gpu_attention, actual_attention)

    golden_topk_with_n = torch_f_out_fp32[4]
    actual_topk = ms_f_out[1]
    assert check_topk_in_range(golden_topk_with_n, actual_topk), "topk_indices_out not in top(k+2) range"

    golden_softmax_max = torch_f_out_fp32[2].to(pt_dtype)
    gpu_softmax_max = torch_f_out_gpu[2].to(pt_dtype)
    actual_softmax_max = ms_f_out[2].to(ms_dtype)
    assert double_golden_compare(golden_softmax_max, gpu_softmax_max, actual_softmax_max)

    golden_softmax_sum = torch_f_out_fp32[3].to(pt_dtype)
    gpu_softmax_sum = torch_f_out_gpu[3].to(pt_dtype)
    actual_softmax_sum = ms_f_out[3].to(ms_dtype)
    assert double_golden_compare(golden_softmax_sum, gpu_softmax_sum, actual_softmax_sum)

    golden_grad_query = torch_b_out_fp32[0]
    gpu_grad_query = torch_b_out_gpu[0]
    actual_grad_query = ms_b_out[0]
    assert double_golden_compare(golden_grad_query, gpu_grad_query, actual_grad_query)

    golden_grad_key = torch_b_out_fp32[1]
    gpu_grad_key = torch_b_out_gpu[1]
    actual_grad_key = ms_b_out[1]
    assert double_golden_compare(golden_grad_key, gpu_grad_key, actual_grad_key)

    golden_grad_value = torch_b_out_fp32[2]
    gpu_grad_value = torch_b_out_gpu[2]
    actual_grad_value = ms_b_out[2]
    assert double_golden_compare(golden_grad_value, gpu_grad_value, actual_grad_value)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="unessential", mem_peak=208.0)
def test_nsa_compress_attention_dynamic_shape_TEST_OP():
    """
    Feature: Dynamic-shape verification via TEST_OP
    Description: Run two representative input sets to exercise dynamic behavior
    Expectation: Executes successfully with correct output shapes; no crash
    """
    T1, T2 = 2633, 4072
    N1, N2 = 7, 7
    D1, D2 = 8, 7
    query1, key1, value1 = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float16)
    scale_value1 = 0.3
    head_num1 = 7
    compress_block_size1 = 32
    compress_stride1 = 16
    select_block_size1 = 64
    select_block_count1 = 16
    S1, S2 = 1384, 567
    topk_mask1 = Tensor(get_topk_mask(S1, S2), dtype=ms.bool_)
    atten_mask1 = Tensor(get_atten_mask(1384, 2266), dtype=ms.bool_)
    actual_seq_qlen1 = [1384, 2633]
    actual_cmp_seq_kvlen1 = [2266, 4072]
    actual_sel_seq_kvlen1 = [567, 1019]

    T1, T2 = 256, 256
    N1, N2 = 8, 8
    D1, D2 = 128, 128
    query2, key2, value2 = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float16)
    scale_value2 = 1.0 / (D1 ** 0.5)
    head_num2 = 8
    compress_block_size2 = 32
    compress_stride2 = 16
    select_block_size2 = 64
    select_block_count2 = 16
    topk_mask2 = Tensor(get_topk_mask(256, 256), dtype=ms.bool_)
    atten_mask2 = Tensor(get_atten_mask(256, 256), dtype=ms.bool_)

    actual_seq_qlen2 = [256]
    actual_cmp_seq_kvlen2 = [256]
    actual_sel_seq_kvlen2 = [256]

    TEST_OP(
        nsa_compress_attention_forward_func,
        [[Tensor(query1, ms.float16), Tensor(key1, ms.float16), Tensor(value1, ms.float16),
          scale_value1, head_num1, compress_block_size1, compress_stride1, select_block_size1,
          select_block_count1, topk_mask1, atten_mask1, actual_seq_qlen1, actual_cmp_seq_kvlen1, actual_sel_seq_kvlen1],
         [Tensor(query2, ms.float16), Tensor(key2, ms.float16), Tensor(value2, ms.float16),
          scale_value2, head_num2, compress_block_size2, compress_stride2, select_block_size2,
          select_block_count2, topk_mask2, atten_mask2, actual_seq_qlen2, actual_cmp_seq_kvlen2,
          actual_sel_seq_kvlen2]],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=["EmptyTensor", "ScalarTensor"],
        case_config={"disable_input_check": True},
    )


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=52.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_nsa_compress_attention_no_mask(context_mode, dtype):
    """
    Feature: nsa_compress_attention operator forward/backward accuracy comparison without masks
    Description: Compare MindSpore nsa_compress_attention results with torch_cpu (tolerance-based)
    Expectation: CPU comparison meets tolerance standards
    """

    T1, T2 = 256, 256
    N1, N2 = 8, 8
    D1, D2 = 128, 128

    if dtype == 'fp16':
        query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float16)
        ms_dtype = ms.float16
        pt_dtype = torch.float16
    else:  # bf16
        query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype='bf16')
        ms_dtype = ms.bfloat16
        pt_dtype = torch.bfloat16

    scale_value = 1.0 / (D1 ** 0.5)
    head_num = 8
    compress_block_size = 32
    compress_stride = 16
    select_block_size = 64
    select_block_count = 16

    actual_seq_qlen = [256]
    actual_cmp_seq_kvlen = [256]
    actual_sel_seq_kvlen = [256]

    ms_f_out, ms_b_out = _ms_forward_backward(
        context_mode, query_np, key_np, value_np, None, None,
        scale_value, head_num, compress_block_size, compress_stride,
        select_block_size, select_block_count, actual_seq_qlen,
        actual_cmp_seq_kvlen, actual_sel_seq_kvlen, ms_dtype)

    pt_query_fp32, pt_key_fp32, pt_value_fp32, _, _ = get_torch_tensors(
        query_np, key_np, value_np, None, None, dtype, target_dtype='fp32')
    torch_f_out_fp32, torch_b_out_fp32 = torch_cpu_forward_backward(
        pt_query_fp32, pt_key_fp32, pt_value_fp32, scale_value, head_num, compress_block_size,
        compress_stride, select_block_size, select_block_count,
        topk_mask=None, atten_mask=None,
        actual_seq_qlen=actual_seq_qlen, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen,
        actual_sel_seq_kvlen=actual_sel_seq_kvlen)

    pt_query_gpu, pt_key_gpu, pt_value_gpu, _, _ = get_torch_tensors(
        query_np, key_np, value_np, None, None, dtype, target_dtype=dtype)
    torch_f_out_gpu, torch_b_out_gpu = torch_cpu_forward_backward(
        pt_query_gpu, pt_key_gpu, pt_value_gpu, scale_value, head_num, compress_block_size,
        compress_stride, select_block_size, select_block_count,
        topk_mask=None, atten_mask=None,
        actual_seq_qlen=actual_seq_qlen, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen,
        actual_sel_seq_kvlen=actual_sel_seq_kvlen)

    golden_attention = torch_f_out_fp32[0]
    gpu_attention = torch_f_out_gpu[0]
    actual_attention = ms_f_out[0]
    assert double_golden_compare(golden_attention, gpu_attention, actual_attention)

    golden_topk_with_n = torch_f_out_fp32[4]
    actual_topk = ms_f_out[1]
    assert check_topk_in_range(golden_topk_with_n, actual_topk), "topk_indices_out not in top(k+2) range"

    golden_softmax_max = torch_f_out_fp32[2].to(pt_dtype)
    gpu_softmax_max = torch_f_out_gpu[2].to(pt_dtype)
    actual_softmax_max = ms_f_out[2].to(ms_dtype)
    assert double_golden_compare(golden_softmax_max, gpu_softmax_max, actual_softmax_max)

    golden_softmax_sum = torch_f_out_fp32[3].to(pt_dtype)
    gpu_softmax_sum = torch_f_out_gpu[3].to(pt_dtype)
    actual_softmax_sum = ms_f_out[3].to(ms_dtype)
    assert double_golden_compare(golden_softmax_sum, gpu_softmax_sum, actual_softmax_sum)

    golden_grad_query = torch_b_out_fp32[0]
    gpu_grad_query = torch_b_out_gpu[0]
    actual_grad_query = ms_b_out[0]
    assert double_golden_compare(golden_grad_query, gpu_grad_query, actual_grad_query)

    golden_grad_key = torch_b_out_fp32[1]
    gpu_grad_key = torch_b_out_gpu[1]
    actual_grad_key = ms_b_out[1]
    assert double_golden_compare(golden_grad_key, gpu_grad_key, actual_grad_key)

    golden_grad_value = torch_b_out_fp32[2]
    gpu_grad_value = torch_b_out_gpu[2]
    actual_grad_value = ms_b_out[2]
    assert double_golden_compare(golden_grad_value, gpu_grad_value, actual_grad_value)

@pytest.mark.skip(reason="empty input tensor not supported torch cpu")
@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
    mem_peak=0.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_attention_empty_input_tensor_forward_backward(context_mode):
    """
    Feature: Empty input tensor validation
    Description: Forward and backward run with empty tensor (T=0), compare with torch_cpu
    Expectation: Produces valid output tensors with expected shapes; matches CPU within tolerance
    """
    T1, T2 = 0, 0
    N1, N2 = 4, 4
    D1, D2 = 32, 32

    query_np = np.zeros((T1, N1, D1), dtype=np.float16)
    key_np = np.zeros((T2, N2, D1), dtype=np.float16)
    value_np = np.zeros((T2, N2, D2), dtype=np.float16)

    scale_value = 1.0 / (D1 ** 0.5)
    head_num = N1

    compress_block_size = 32
    compress_stride = 16
    select_block_size = 64
    select_block_count = 16

    topk_mask_np = None
    atten_mask_np = None

    actual_seq_qlen = [0]
    actual_cmp_seq_kvlen = [0]
    actual_sel_seq_kvlen = [0]

    ms_dtype = ms.float16
    pt_dtype = torch.float16
    ms_f_out, ms_b_out = _ms_forward_backward(
        context_mode, query_np, key_np, value_np, topk_mask_np, atten_mask_np,
        scale_value, head_num, compress_block_size, compress_stride,
        select_block_size, select_block_count, actual_seq_qlen,
        actual_cmp_seq_kvlen, actual_sel_seq_kvlen, ms_dtype)

    dtype_str = 'fp16'
    pt_query_fp32, pt_key_fp32, pt_value_fp32, pt_topk_mask, pt_atten_mask = get_torch_tensors(
        query_np, key_np, value_np, topk_mask_np, atten_mask_np, dtype_str, target_dtype='fp32')
    torch_f_out_fp32, torch_b_out_fp32 = torch_cpu_forward_backward(
        pt_query_fp32, pt_key_fp32, pt_value_fp32, scale_value, head_num, compress_block_size,
        compress_stride, select_block_size, select_block_count,
        topk_mask=pt_topk_mask, atten_mask=pt_atten_mask,
        actual_seq_qlen=actual_seq_qlen, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen,
        actual_sel_seq_kvlen=actual_sel_seq_kvlen)

    pt_query_gpu, pt_key_gpu, pt_value_gpu, _, _ = get_torch_tensors(
        query_np, key_np, value_np, topk_mask_np, atten_mask_np, dtype_str, target_dtype=dtype_str)
    torch_f_out_gpu, torch_b_out_gpu = torch_cpu_forward_backward(
        pt_query_gpu, pt_key_gpu, pt_value_gpu, scale_value, head_num, compress_block_size,
        compress_stride, select_block_size, select_block_count,
        topk_mask=pt_topk_mask, atten_mask=pt_atten_mask,
        actual_seq_qlen=actual_seq_qlen, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen,
        actual_sel_seq_kvlen=actual_sel_seq_kvlen)

    golden_attention = torch_f_out_fp32[0]
    gpu_attention = torch_f_out_gpu[0]
    actual_attention = ms_f_out[0]
    assert double_golden_compare(golden_attention, gpu_attention, actual_attention)

    golden_topk_with_n = torch_f_out_fp32[4]
    actual_topk = ms_f_out[1]
    assert check_topk_in_range(golden_topk_with_n, actual_topk), "topk_indices_out not in top(k+2) range"

    golden_softmax_max = torch_f_out_fp32[2].to(pt_dtype)
    gpu_softmax_max = torch_f_out_gpu[2].to(pt_dtype)
    actual_softmax_max = ms_f_out[2].to(ms_dtype)
    assert double_golden_compare(golden_softmax_max, gpu_softmax_max, actual_softmax_max)

    golden_softmax_sum = torch_f_out_fp32[3].to(pt_dtype)
    gpu_softmax_sum = torch_f_out_gpu[3].to(pt_dtype)
    actual_softmax_sum = ms_f_out[3].to(ms_dtype)
    assert double_golden_compare(golden_softmax_sum, gpu_softmax_sum, actual_softmax_sum)

    golden_grad_query = torch_b_out_fp32[0]
    gpu_grad_query = torch_b_out_gpu[0]
    actual_grad_query = ms_b_out[0]
    assert double_golden_compare(golden_grad_query, gpu_grad_query, actual_grad_query)

    golden_grad_key = torch_b_out_fp32[1]
    gpu_grad_key = torch_b_out_gpu[1]
    actual_grad_key = ms_b_out[1]
    assert double_golden_compare(golden_grad_key, gpu_grad_key, actual_grad_key)

    golden_grad_value = torch_b_out_fp32[2]
    gpu_grad_value = torch_b_out_gpu[2]
    actual_grad_value = ms_b_out[2]
    assert double_golden_compare(golden_grad_value, gpu_grad_value, actual_grad_value)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
    mem_peak=0.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("invalid_param", [
    ("head_num", 0),
    ("compress_block_size", 18),
    ("compress_stride", 15),
    ("select_block_size", 15),
    ("select_block_count", 33),
    ("d_not_multiple_16", 5),
])
def test_nsa_compress_attention_exception_invalid_parameters(context_mode, invalid_param):
    """
    Feature: Parameter validation
    Description: Test various invalid parameter combinations
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    param_name, param_value = invalid_param

    T, N, D = 64, 4, 32
    head_num = 4
    compress_block_size = 32
    compress_stride = 16
    select_block_size = 64
    select_block_count = 16
    scale_value = 1.0 / (D ** 0.5)
    actual_seq_qlen = [T]
    actual_cmp_seq_kvlen = [T]
    actual_sel_seq_kvlen = [T]

    if param_name == "head_num":
        head_num = param_value
    elif param_name == "compress_block_size":
        compress_block_size = param_value
    elif param_name == "compress_stride":
        compress_stride = param_value
    elif param_name == "select_block_size":
        select_block_size = param_value
    elif param_name == "select_block_count":
        select_block_count = param_value
    elif param_name == "d_not_multiple_16":
        D = param_value

    rng = np.random.default_rng(100)
    query_np = rng.standard_normal((T, N, D)).astype(np.float16)
    key_np = rng.standard_normal((T, N, D)).astype(np.float16)
    value_np = rng.standard_normal((T, N, D)).astype(np.float16)

    query = Tensor(query_np, dtype=ms.float16)
    key = Tensor(key_np, dtype=ms.float16)
    value = Tensor(value_np, dtype=ms.float16)

    maxSq, maxSkv, maxSelSkv = T, T, T // 2
    atten_mask = Tensor(np.ones((maxSq, maxSkv), dtype=np.bool_))
    topk_mask = Tensor(np.ones((maxSq, maxSelSkv), dtype=np.bool_))

    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.nsa_compress_attention(query, key, value, scale_value, head_num, compress_block_size,
                                      compress_stride, select_block_size, select_block_count,
                                      topk_mask, atten_mask, actual_seq_qlen, actual_cmp_seq_kvlen,
                                      actual_sel_seq_kvlen)
        y[0].asnumpy()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
    mem_peak=0.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_attention_exception_dtype_mismatch(context_mode):
    """
    Feature: Dtype validation and consistency
    Description: query, key, value have different dtypes
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    T, N, D = 64, 4, 32
    head_num = 4
    compress_block_size = 32  # Supported range: 16-128, aligned to 16
    compress_stride = 16      # Supported range: 16-64, aligned to 16
    select_block_size = 64    # Supported range: 16-128, aligned to 16
    select_block_count = 16   # Supported range: 1-32
    scale_value = 1.0 / (D ** 0.5)  # Standard scaling: 1/sqrt(D)
    actual_seq_qlen = [32, 48, 64]
    actual_cmp_seq_kvlen = [32, 48, 64]
    actual_sel_seq_kvlen = [16, 24, 32]

    rng = np.random.default_rng(104)
    query_np = rng.standard_normal((T, N, D)).astype(np.float16)
    key_np = rng.standard_normal((T, N, D)).astype(np.float32)  # mismatch
    value_np = rng.standard_normal((T, N, D)).astype(np.float16)

    query = Tensor(query_np, dtype=ms.float16)
    key = Tensor(key_np, dtype=ms.float32)  # mismatch
    value = Tensor(value_np, dtype=ms.float16)

    maxSq = T  # max query sequence length
    maxSkv = T  # max key/value sequence length
    maxSelSkv = T // 2  # max select sequence length
    atten_mask = Tensor(np.ones((maxSq, maxSkv), dtype=np.bool_))
    topk_mask = Tensor(np.ones((maxSq, maxSelSkv), dtype=np.bool_))

    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.nsa_compress_attention(query, key, value, scale_value, head_num, compress_block_size,
                                      compress_stride, select_block_size, select_block_count,
                                      topk_mask, atten_mask, actual_seq_qlen, actual_cmp_seq_kvlen,
                                      actual_sel_seq_kvlen)
        y[0].asnumpy()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
    mem_peak=0.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("invalid_seq_len", [
    ("actual_seq_qlen", None, [64], [64]),
    ("actual_cmp_seq_kvlen", [64], None, [64]),
    ("actual_sel_seq_kvlen", [0], [64], None),
    ("seq_len_not_match_q", [63], [0], [64]),
    ("seq_len_not_match_kv", [64], [0], [0]),
])
def test_nsa_compress_attention_exception_invalid_seq_len(context_mode, invalid_seq_len):
    """
    Feature: Sequence length validation
    Description: Test None or mismatched sequence lengths
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    _, seq_qlen, seq_kvlen, seq_sel_kvlen = invalid_seq_len

    T, N, D = 64, 4, 32
    head_num = 4
    compress_block_size = 32
    compress_stride = 16
    select_block_size = 64
    select_block_count = 16
    scale_value = 1.0 / (D ** 0.5)

    rng = np.random.default_rng(105)
    query_np = rng.standard_normal((T, N, D)).astype(np.float16)
    key_np = rng.standard_normal((T, N, D)).astype(np.float16)
    value_np = rng.standard_normal((T, N, D)).astype(np.float16)

    query = Tensor(query_np, dtype=ms.float16)
    key = Tensor(key_np, dtype=ms.float16)
    value = Tensor(value_np, dtype=ms.float16)

    maxSq, maxSkv, maxSelSkv = T, T, T // 2
    atten_mask = Tensor(np.ones((maxSq, maxSkv), dtype=np.bool_))
    topk_mask = Tensor(np.ones((maxSq, maxSelSkv), dtype=np.bool_))

    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.nsa_compress_attention(query, key, value, scale_value, head_num, compress_block_size,
                                      compress_stride, select_block_size, select_block_count,
                                      topk_mask, atten_mask, seq_qlen, seq_kvlen, seq_sel_kvlen)
        y[0].asnumpy()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=1.00
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_attention_exception_shape_mismatch(context_mode):
    """
    Feature: Input shape validation
    Description: query/key/value have inconsistent shapes
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    T, N, D = 64, 4, 32
    head_num = 4
    compress_block_size = 32
    compress_stride = 16
    select_block_size = 64
    select_block_count = 16
    scale_value = 1.0 / (D ** 0.5)
    actual_seq_qlen = [T]
    actual_cmp_seq_kvlen = [T]
    actual_sel_seq_kvlen = [T]

    rng = np.random.default_rng(110)
    query_np = rng.standard_normal((T, N, D, 3)).astype(np.float16)
    key_np = rng.standard_normal((T + 1, N, D, 1)).astype(np.float16)
    value_np = rng.standard_normal((T, N, D, 2)).astype(np.float16)

    query = Tensor(query_np, dtype=ms.float16)
    key = Tensor(key_np, dtype=ms.float16)
    value = Tensor(value_np, dtype=ms.float16)

    maxSq, maxSkv, maxSelSkv = T, T, T // 2
    atten_mask = Tensor(np.ones((maxSq, maxSkv), dtype=np.bool_))
    topk_mask = Tensor(np.ones((maxSq, maxSelSkv), dtype=np.bool_))

    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.nsa_compress_attention(query, key, value, scale_value, head_num, compress_block_size,
                                      compress_stride, select_block_size, select_block_count,
                                      topk_mask, atten_mask, actual_seq_qlen, actual_cmp_seq_kvlen,
                                      actual_sel_seq_kvlen)
        y[0].asnumpy()
