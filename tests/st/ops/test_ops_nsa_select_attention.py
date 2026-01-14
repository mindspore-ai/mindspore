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
""" test_ops_nsa_select_attention.py """
# pylint: disable=unused-import
# pylint: disable=E1121
"""
Tests for NsaSelectAttention operation.
"""
import pytest
import torch
import hashlib
import numpy as np
import mindspore as ms
from einops import rearrange
from mindspore import ops, mint
from mindspore.graph.api import _pynative_executor
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP


_env_has_torch_npu = True
_ci_environment = True
try:
    import torch_npu
    from torch_npu import npu_nsa_select_attention
    torch.npu.set_device(f'npu:{ms.get_context("device_id")}')
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    print("torch_npu is found")
except ImportError:
    print("torch_npu or torch_npu.npu_nsa_select_attention not found, benchmark will use torch cpu instead")
    _env_has_torch_npu = False
torch.use_deterministic_algorithms(True)


@test_utils.run_with_cell
def nsa_select_attention_forward_func(query, key, value, topk_indices, scale_value, head_num, select_block_size,
                                      select_block_count, atten_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None):
    return ops.nsa_select_attention(query, key, value, topk_indices, scale_value, head_num, select_block_size,
                                    select_block_count, atten_mask=atten_mask, actual_seq_qlen=actual_seq_qlen,
                                    actual_seq_kvlen=actual_seq_kvlen)


@test_utils.run_with_cell
def nsa_select_attention_backward_func(query, key, value, topk_indices, scale_value, head_num, select_block_size,
                                       select_block_count, atten_mask=None, actual_seq_qlen=None,
                                       actual_seq_kvlen=None):
    return ms.grad(nsa_select_attention_forward_func, (0, 1, 2))(query, key, value, topk_indices, scale_value, head_num,
                                                                 select_block_size, select_block_count, atten_mask,
                                                                 actual_seq_qlen, actual_seq_kvlen)


def torch_npu_nsa_select_attention_forward_backward(query, key, value, topk_indices, scale_value, head_num,
                                                    select_block_size, select_block_count, atten_mask=None,
                                                    actual_seq_qlen=None, actual_seq_kvlen=None, skip_grad=False):
    query = query.npu()
    key = key.npu()
    value = value.npu()
    topk_indices = topk_indices.npu()
    if atten_mask is not None:
        atten_mask = atten_mask.npu()

    if not skip_grad:
        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True

    attention_out, softmax_max, softmax_sum = torch_npu.npu_nsa_select_attention(query, key, value, topk_indices,
                                                                                 scale_value, head_num,
                                                                                 select_block_size, select_block_count,
                                                                                 atten_mask=atten_mask,
                                                                                 actual_seq_qlen=actual_seq_qlen,
                                                                                 actual_seq_kvlen=actual_seq_kvlen)
    if not skip_grad:
        attention_out.backward(gradient=torch.ones_like(attention_out))
        return attention_out.detach(), softmax_max.detach(), softmax_sum.detach(), \
            query.grad.detach(), key.grad.detach(), value.grad.detach()

    return attention_out, softmax_max, softmax_sum, None, None, None


class TorchCpuNsaSelectAttention():
    def __init__(self, query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count,
                 atten_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None):
        self.query = query
        self.key = key
        self.value = value
        self.topk_indices = topk_indices
        self.scale_value = scale_value
        self.head_num = head_num
        self.select_block_size = select_block_size
        self.select_block_count = select_block_count
        self.atten_mask = atten_mask
        self.actual_seq_qlen = actual_seq_qlen
        self.actual_seq_kvlen = actual_seq_kvlen
        self.sparse_mode = 2
        self.attention_out = None
        self.softmax_max = None
        self.softmax_sum = None

    def softmax_torch(self, x):
        x_max = torch.max(x, dim=-1, keepdims=True)[0]
        x_sub = x.sub(x_max)
        y = torch.exp(x_sub)
        x_sum = y.sum(dim=-1, keepdims=True)
        ans = torch.softmax(x, dim=-1)
        return ans, x_max, x_sum

    def get_currentB_index(self, BS_index, actual_seq_len):
        for i in range(actual_seq_len.size(0)):
            if BS_index < actual_seq_len[i]:
                return i
        raise RuntimeError("BS_index is greater than max(actual_seq_len).")

    def simple_softmax(self, x, x_max, x_sum):
        x_sub = x.sub(x_max)
        y = torch.exp(x_sub)
        softmax_res = y.div(x_sum)
        return softmax_res

    def tsoftmax_grad(self, p, dp, out, outGrad):
        muls = outGrad * out
        muls_res = muls.sum(dim=-1, keepdims=True)
        sub_res = dp - muls_res
        res = sub_res * p
        return res

    def forward(self):
        ori_dtype = self.query.dtype
        query_float = self.query.float()
        key_float = self.key.float()
        value_float = self.value.float()

        block_size = self.select_block_size
        block_count = self.select_block_count
        actual_seq_qlen = torch.tensor(self.actual_seq_qlen, dtype=torch.int64)
        actual_seq_kvlen = torch.tensor(self.actual_seq_kvlen, dtype=torch.int64)

        BS1, N1, QKD = query_float.shape
        BS2, N2, VD = value_float.shape
        G = N1 // N2
        N1 = N2 * G
        key_reshaped = key_float.view(BS2 // block_size, block_size, N2, QKD)
        value_reshaped = value_float.view(BS2 // block_size, block_size, N2, VD)
        query_reshaped = query_float.reshape(BS1, N2 * G, QKD).view(BS1, N2, G, QKD)

        # output
        output = torch.zeros(BS1, N2, G, VD, dtype=torch.float32, device=query_reshaped.device)
        softmax_max = torch.zeros(BS1, N2, G, 1, dtype=torch.float32, device=query_reshaped.device)
        softmax_sum = torch.zeros(BS1, N2, G, 1, dtype=torch.float32, device=query_reshaped.device)

        for bs1_index in range(BS1):
            b_index = self.get_currentB_index(bs1_index, actual_seq_qlen)
            # s1_index = bs1_index if b_index == 0 else bs1_index - actual_seq_qlen[b_index - 1]
            start_BS2_index = 0 if b_index == 0 else actual_seq_kvlen[b_index - 1]
            start_block_index = start_BS2_index // block_size
            current_S2_size = actual_seq_kvlen[b_index] - start_BS2_index
            current_block_count = current_S2_size // block_size
            end_block_index = start_block_index + current_block_count
            if start_BS2_index % block_size != 0:
                raise RuntimeError(f"当前golden只支持S2 {block_size}对齐.")

            for n2_index in range(N2):
                topk_index_ = self.topk_indices[bs1_index, n2_index, :]
                # (block_count, block_size, QKD)
                selectedK = torch.index_select(key_reshaped[start_block_index:end_block_index, :, n2_index, :], 0,
                                               topk_index_)
                selectedK = selectedK.reshape(block_count * block_size, QKD)
                # (G, block_count * block_size)
                qk_ = torch.matmul(query_reshaped[bs1_index, n2_index, :, :], selectedK.transpose(-1, -2))
                qk_ = torch.mul(qk_, self.scale_value)
                softmax_res, x_max, x_sum = self.softmax_torch(qk_)
                # (block_count, block_size, VD)
                selectedV = torch.index_select(value_reshaped[start_block_index:end_block_index, :, n2_index, :], 0,
                                               topk_index_)
                selectedV = selectedV.reshape(block_count * block_size, VD)
                # (G, VD)
                out_ = torch.matmul(softmax_res.to(ori_dtype).float(), selectedV.to(ori_dtype).float())

                output[bs1_index, n2_index] = out_
                softmax_max[bs1_index, n2_index] = x_max
                softmax_sum[bs1_index, n2_index] = x_sum

        attention_out = output.reshape(BS1, N1, VD)
        softmax_max = softmax_max.reshape(BS1, N1, 1)
        softmax_sum = softmax_sum.reshape(BS1, N1, 1)

        softmax_max_8 = softmax_max.broadcast_to(BS1, N1, 8)
        softmax_sum_8 = softmax_sum.broadcast_to(BS1, N1, 8)

        self.attention_out = attention_out.to(ori_dtype).detach()
        self.softmax_max = softmax_max_8.detach()
        self.softmax_sum = softmax_sum_8.detach()

        return self.attention_out, self.softmax_max, self.softmax_sum

    def backward(self):
        ori_dtype = self.query.dtype
        query = self.query.float()
        key = self.key.float()
        value = self.value.float()
        out = self.attention_out.float()
        grad = torch.ones_like(out)
        softmax_max = self.softmax_max.float()
        softmax_sum = self.softmax_sum.float()
        topk_indices = self.topk_indices
        actual_q_len = self.actual_seq_qlen
        actual_kv_len = self.actual_seq_kvlen

        # param
        scaleValue = self.scale_value
        selected_block_count = self.select_block_count
        selected_block_size = self.select_block_size
        select_s2 = selected_block_size * selected_block_count
        atten_enable = bool(self.sparse_mode == 2)

        # shape
        T1, N1, D_qk = query.shape
        T2, N2, D_v = value.shape
        G = N1 // N2
        # S1 = max(self.actual_seq_qlen)
        S2 = max(self.actual_seq_kvlen)
        B = len(self.actual_seq_qlen)
        # reshape
        query = query.reshape(T1, N2, G, D_qk)
        key = key.reshape(T2, N2, 1, D_qk)
        value = value.reshape(T2, N2, 1, D_v)
        out = out.reshape(T1, N2, G, D_v)
        grad = grad.reshape(T1, N2, G, D_v)
        softmax_max = softmax_max.reshape(T1, N2, G, 8)
        softmax_sum = softmax_sum.reshape(T1, N2, G, 8)


        dq_out = torch.zeros(T1, N2, G, D_qk).to(torch.float)
        x_max_out = torch.zeros(T1, N2, G, 1).to(torch.float)
        x_sum_out = torch.zeros(T1, N2, G, 1).to(torch.float)
        dk_out = torch.zeros(B, N2, S2, D_qk).reshape(B, N2, -1, selected_block_size, D_qk).to(torch.float)
        dv_out = torch.zeros(B, N2, S2, D_v).reshape(B, N2, -1, selected_block_size, D_v).to(torch.float)

        k_tmp = torch.zeros(B, S2, N2, D_qk).to(torch.float)
        v_tmp = torch.zeros(B, S2, N2, D_v).to(torch.float)

        start_kv_idx = 0
        for b_idx in range(B):
            end_kv_idx = actual_kv_len[b_idx]
            batch_kv_len = end_kv_idx - start_kv_idx
            k_tmp[b_idx, :batch_kv_len] = key[start_kv_idx:end_kv_idx, :, 0]
            v_tmp[b_idx, :batch_kv_len] = value[start_kv_idx:end_kv_idx, :, 0]
            start_kv_idx = end_kv_idx

        k_tmp = rearrange(k_tmp, 'b s n d ->  b n s d').reshape(B, N2, -1, selected_block_size, D_qk)
        v_tmp = rearrange(v_tmp, 'b s n d ->  b n s d').reshape(B, N2, -1, selected_block_size, D_v)

        for i in range(T1):
            b_idx, s1_idx = get_tnd_idx(actual_q_len, i)

            for n2_idx in range(N2):
                # gather
                topk = topk_indices[i][n2_idx]
                q_cal = query[i][n2_idx]
                out_cal = out[i][n2_idx]
                grad_cal = grad[i][n2_idx]
                k_cal = torch.index_select(k_tmp[b_idx][n2_idx], 0, topk).reshape(select_s2, D_qk)
                v_cal = torch.index_select(v_tmp[b_idx][n2_idx], 0, topk).reshape(select_s2, D_v)

                if atten_enable:
                    if s1_idx < select_s2:
                        atten_msk_cal = torch.ones(select_s2)
                        atten_msk_cal[0:s1_idx + 1] = 0
                    else:
                        atten_msk = torch.ones(S2)
                        atten_msk[0:select_s2 + 1] = 0
                        atten_msk = atten_msk.reshape(-1, selected_block_size)
                        atten_msk_cal = torch.index_select(atten_msk, 0, topk).reshape(select_s2)
                    atten_msk_cal = atten_msk_cal.repeat(G, 1)
                # fag cal
                qk = torch.matmul(q_cal, k_cal.permute(1, 0)).mul(scaleValue)
                if atten_enable:
                    qk = qk + atten_msk_cal * (-2e35)

                x_max = softmax_max[i][n2_idx][:, [0]].reshape(-1, 1)
                x_sum = softmax_sum[i][n2_idx][:, [0]].reshape(-1, 1)
                softmax_res = self.simple_softmax(qk, x_max, x_sum)

                dp = torch.matmul(grad_cal, v_cal.permute(1, 0))
                softmax_grad_res = self.tsoftmax_grad(softmax_res, dp, out_cal, grad_cal)
                dq = torch.matmul(softmax_grad_res, k_cal)
                dk = torch.matmul(softmax_grad_res.permute(1, 0), q_cal)
                dv = torch.matmul(softmax_res.permute(1, 0), grad_cal)
                dk = dk.reshape(selected_block_count, selected_block_size, D_qk)
                dv = dv.reshape(selected_block_count, selected_block_size, D_v)

                #scatter
                dq_out[i][n2_idx] = dq
                x_max_out[i][n2_idx] = x_max
                x_sum_out[i][n2_idx] = x_sum
                for kk in range(selected_block_count):
                    dk_out[b_idx][n2_idx][topk[kk]] += dk[kk]
                    dv_out[b_idx][n2_idx][topk[kk]] += dv[kk]


        dq_out = dq_out * scaleValue
        dk_out = dk_out * scaleValue

        dk_out = dk_out.reshape(B, N2, S2, D_qk)
        dv_out = dv_out.reshape(B, N2, S2, D_v)
        dk_out = rearrange(dk_out, 'b n s d ->  b s n d')
        dv_out = rearrange(dv_out, 'b n s d ->  b s n d')

        dk_out_continuous = torch.zeros(T2, N2, D_qk).to(torch.float)
        dv_out_continuous = torch.zeros(T2, N2, D_v).to(torch.float)

        start_kv_idx = 0
        for b_idx in range(B):
            end_kv_idx = actual_kv_len[b_idx]
            batch_kv_len = end_kv_idx - start_kv_idx
            dk_out_continuous[start_kv_idx:end_kv_idx, :] = dk_out[b_idx, :batch_kv_len, :]
            dv_out_continuous[start_kv_idx:end_kv_idx, :] = dv_out[b_idx, :batch_kv_len, :]
            start_kv_idx = end_kv_idx

        dq_out = dq_out.reshape(T1, N1, D_qk)
        dk_out = dk_out_continuous.reshape(T2, N2, D_qk)
        dv_out = dv_out_continuous.reshape(T2, N2, D_v)

        x_max_out = x_max_out.expand(T1, N2, G, 8).reshape(T1, N2 * G, 8)
        x_sum_out = x_sum_out.expand(T1, N2, G, 8).reshape(T1, N2 * G, 8)

        return dq_out.to(ori_dtype), dk_out.to(ori_dtype), dv_out.to(ori_dtype)


def torch_cpu_nsa_select_attention_forward_backward(query, key, value, topk_indices, scale_value, head_num,
                                                    select_block_size, select_block_count, atten_mask=None,
                                                    actual_seq_qlen=None, actual_seq_kvlen=None):
    cpu_golden_exec = TorchCpuNsaSelectAttention(query, key, value, topk_indices, scale_value, head_num,
                                                 select_block_size, select_block_count, atten_mask, actual_seq_qlen,
                                                 actual_seq_kvlen)
    attention_out, softmax_max, softmax_sum = cpu_golden_exec.forward()
    dq, dk, dv = cpu_golden_exec.backward()
    return attention_out, softmax_max, softmax_sum, dq, dk, dv


def torch_npu_cpu_nsa_select_attention_forward_backward(query, key, value, topk_indices, scale_value, head_num,
                                                        select_block_size, select_block_count, atten_mask=None,
                                                        actual_seq_qlen=None, actual_seq_kvlen=None):

    attention_out, softmax_max, softmax_sum, _, _, _ = \
        torch_npu_nsa_select_attention_forward_backward(query, key, value, topk_indices, scale_value, head_num,
                                                        select_block_size, select_block_count, atten_mask,
                                                        actual_seq_qlen, actual_seq_kvlen, skip_grad=True)

    cpu_golden_exec = TorchCpuNsaSelectAttention(query, key, value, topk_indices, scale_value, head_num,
                                                 select_block_size, select_block_count, atten_mask, actual_seq_qlen,
                                                 actual_seq_kvlen)
    cpu_golden_exec.attention_out = attention_out.cpu()
    cpu_golden_exec.softmax_max = softmax_max.cpu()
    cpu_golden_exec.softmax_sum = softmax_sum.cpu()
    dq, dk, dv = cpu_golden_exec.backward()
    return attention_out, softmax_max, softmax_sum, dq, dk, dv


def torch_golden_expect_forward_backward(query, key, value, topk_indices, scale_value, head_num, select_block_size,
                                         select_block_count, atten_mask=None, actual_seq_qlen=None,
                                         actual_seq_kvlen=None):
    if _env_has_torch_npu:
        if _ci_environment:
            golden_func = torch_npu_cpu_nsa_select_attention_forward_backward
        else:
            golden_func = torch_npu_nsa_select_attention_forward_backward
    else:
        golden_func = torch_cpu_nsa_select_attention_forward_backward

    return golden_func(query, key, value, topk_indices, scale_value, head_num, select_block_size,
                       select_block_count, atten_mask, actual_seq_qlen, actual_seq_kvlen)


def set_context_mode(mode):
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'kbk':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O2')


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


def get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32):

    query_np = np.random.randn(T1, N1, D1).astype(dtype)
    key_np = np.random.randn(T2, N2, D1).astype(dtype)
    value_np = np.random.randn(T2, N2, D2).astype(dtype)
    return query_np, key_np, value_np


def get_topk_indices(query_np, value_np, select_block_size, select_block_count, actual_seq_qlen):
    topk_indices = np.zeros((query_np.shape[0], value_np.shape[1], select_block_count)).astype(np.int32)
    for i in range(query_np.shape[0]):
        _, s1_idx = get_tnd_idx(actual_seq_qlen, i)
        for j in range(value_np.shape[1]):
            if s1_idx < select_block_count * select_block_size:
                topk_indices[i][j] = np.arange(select_block_count)
            else:
                topk_indices[i][j] = np.random.permutation(select_block_count)
                idx = int(np.random.uniform(0, select_block_count))
                topk_indices[i][j][idx] = (s1_idx + select_block_size - 1) // select_block_size
    return topk_indices


def get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype):
    if dtype == 'fp16':
        ms_query = ms.Tensor(query_np, dtype=ms.float16) if query_np is not None else None
        ms_key = ms.Tensor(key_np, dtype=ms.float16) if key_np is not None else None
        ms_value = ms.Tensor(value_np, dtype=ms.float16) if value_np is not None else None
    else:
        ms_query = ms.Tensor(query_np, dtype=ms.bfloat16) if query_np is not None else None
        ms_key = ms.Tensor(key_np, dtype=ms.bfloat16) if key_np is not None else None
        ms_value = ms.Tensor(value_np, dtype=ms.bfloat16) if value_np is not None else None
    ms_topk_indices = ms.Tensor(topk_indices_np) if topk_indices_np is not None else None

    return ms_query, ms_key, ms_value, ms_topk_indices


def get_torch_tensors(query_np, key_np, value_np, topk_indices_np, dtype):
    if dtype == 'fp16':
        pt_query = torch.tensor(query_np, dtype=torch.float16) if query_np is not None else None
        pt_key = torch.tensor(key_np, dtype=torch.float16) if key_np is not None else None
        pt_value = torch.tensor(value_np, dtype=torch.float16) if value_np is not None else None
    else:
        pt_query = torch.tensor(query_np, dtype=torch.bfloat16) if query_np is not None else None
        pt_key = torch.tensor(key_np, dtype=torch.bfloat16) if key_np is not None else None
        pt_value = torch.tensor(value_np, dtype=torch.bfloat16) if value_np is not None else None
    pt_topk_indices = torch.tensor(topk_indices_np) if topk_indices_np is not None else None

    return pt_query, pt_key, pt_value, pt_topk_indices


def compare_results(torch_out, ms_out, grad=False):
    def compare_with_torch(ms_out, torch_out):
        def cpu_cmp(ms_out, torch_out):
            # CPU scenario: use single_golden_compare for accuracy verification
            # Refer to nsa_compress: use the last dimension size as ksize
            for _, (golden_t, actual_t) in enumerate(zip(torch_out, ms_out)):
                ksize = int(actual_t.shape[-1]) if actual_t.shape else 1
                assert test_utils.single_golden_compare(golden_t, actual_t, ksize)
        def npu_cmp(ms_out, torch_out):
            def get_ndarray_md5(arr):
                arr_bytes = np.ascontiguousarray(arr).tobytes()
                return hashlib.md5(arr_bytes).hexdigest()

            ms_out_np = [out.float().asnumpy() if out.dtype == ms.bfloat16 else out.asnumpy() for out in ms_out]
            pt_out_np = [
                out.float().cpu().numpy() if out.dtype == torch.bfloat16 else out.cpu().numpy()
                for out in torch_out
            ]
            assert get_ndarray_md5(ms_out_np[0]) == get_ndarray_md5(pt_out_np[0])
            assert get_ndarray_md5(ms_out_np[1]) == get_ndarray_md5(pt_out_np[1])
            assert get_ndarray_md5(ms_out_np[2]) == get_ndarray_md5(pt_out_np[2])

        if _env_has_torch_npu:
            if _ci_environment and grad:
                cpu_cmp(ms_out, torch_out)
            else:
                npu_cmp(ms_out, torch_out)
        else:
            cpu_cmp(ms_out, torch_out)

    print(f"comparing mindspore with torch {'npu' if _env_has_torch_npu else 'cpu'}"
          f"{'' if not _ci_environment else '_ci'} {('forward' if not grad else 'backward')} results...")
    # compare mindspore with torch
    compare_with_torch(ms_out, torch_out)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=582.00)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_ops_nsa_select_attention_normal(mode, dtype):
    """
    Feature: pyboost function.
    Description: test function nsa_select_attention.
    Expectation: expect correct result.
    """
    set_context_mode(mode)

    T1, T2, N1, N2, D1, D2 = 1, 1088, 115, 115, 192, 128
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = N1
    select_block_count = 16
    select_block_size = 64
    actual_seq_qlen = [1]
    actual_seq_kvlen = [1088]
    topk_indices_np = get_topk_indices(query_np, value_np, select_block_size, select_block_count, actual_seq_qlen)

    pt_query, pt_key, pt_value, pt_topk_indices = get_torch_tensors(query_np, key_np, value_np, topk_indices_np, dtype)
    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    # torch nsa_select_attention forward and backward
    torch_out = torch_golden_expect_forward_backward(pt_query, pt_key, pt_value, pt_topk_indices, scale_value, head_num,
                                                     select_block_size, select_block_count, None, actual_seq_qlen,
                                                     actual_seq_kvlen)
    # mindspore nsa_select_attention forward
    ms_out = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices, scale_value, head_num,
                                               select_block_size, select_block_count, None, actual_seq_qlen,
                                               actual_seq_kvlen)
    # compare mindspore with torch forward
    compare_results(torch_out[:3], ms_out, grad=False)
    # mindspore nsa_select_attention backward
    ms_out_grad = nsa_select_attention_backward_func(ms_query, ms_key, ms_value, ms_topk_indices, scale_value, head_num,
                                                     select_block_size, select_block_count, None, actual_seq_qlen,
                                                     actual_seq_kvlen)
    # compare mindspore with torch backward
    compare_results(torch_out[3:], ms_out_grad, grad=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=208.00)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['fp16'])
def test_ops_nsa_select_attention_empty_tensor(mode, dtype):
    """
    Feature: pyboost function.
    Description: test function nsa_select_attention with various empty tensor inputs.

    This test covers 14 different empty tensor scenarios to validate the algorithm's
    robustness in handling edge cases where different dimensions (T, N, D) are zero.

    Supported empty tensor scenarios (should succeed):
    - Test case 1: T1=0 (Empty query sequence)
    - Test case 2: T1=0, T2=0 (All sequences empty)
    - Test case 3: All dimensions = 0 (Completely empty tensors)
    - Test case 5: N1=N2=0 (Empty head dimension)
    - Test case 8: T1=0, T2>0 (Mixed empty sequences)
    - Test case 13: Only N1=N2=0 (Only head count is zero)

    Unsupported empty tensor scenarios (should raise RuntimeError):
    - Test case 4: T1>0, T2=0 (Non-empty query with empty key/value)
    - Test case 6: D1=0 (Empty query/key feature dimension)
    - Test case 7: D2=0 (Empty value feature dimension)
    - Test case 9: Asymmetric head configurations (N1>0, N2=0)
    - Test case 10: D1=D2=0 (Zero feature dimensions with large sequences)
    - Test case 11: Only D1=0 (Only query/key feature dimension is zero)
    - Test case 12: Only D2=0 (Only value feature dimension is zero)
    - Test case 14: Only T2=0 (Only key/value sequence length is zero)

    Expectation: Supported scenarios return correct empty tensor shapes,
    unsupported scenarios raise RuntimeError with appropriate error messages.
    """
    set_context_mode(mode)

    # Standard parameters following API constraints
    select_block_size = 64  # Must be 64 per API specification
    select_block_count = 16  # Must be 16 when select_block_size=64

    # Test case 1: Empty query sequence (T_1 = 0)
    T1, T2, N1, N2, D1, D2 = 0, 1024, 4, 4, 192, 128
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = N1
    actual_seq_qlen = [0]
    actual_seq_kvlen = [1024]
    # For empty query, topk_indices shape should be (0, N2, select_block_count)
    topk_indices_np = np.array([]).reshape(0, 4, select_block_count).astype(np.int32)

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    ms_out = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices, scale_value, head_num,
                                               select_block_size, select_block_count, None, actual_seq_qlen,
                                               actual_seq_kvlen)
    _pynative_executor.sync()
    assert ms_out[0].shape == (0, 4, 128)  # Empty output with correct shape
    assert ms_out[1].shape == (0, 4, 8)
    assert ms_out[2].shape == (0, 4, 8)

    # Test case 2: All sequences empty (T1=0, T2=0)
    T1, T2, N1, N2, D1, D2 = 0, 0, 2, 2, 192, 128
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = N1
    actual_seq_qlen = [0]
    actual_seq_kvlen = [0]
    # For both empty sequences, select_block_count should be 0
    empty_select_block_count = 0
    topk_indices_np = np.array([]).reshape(0, 2, empty_select_block_count).astype(np.int32)

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    ms_out_all = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices, scale_value, head_num,
                                                   select_block_size, empty_select_block_count, None, actual_seq_qlen,
                                                   actual_seq_kvlen)
    _pynative_executor.sync()
    assert ms_out_all[0].shape == (0, 2, 128)  # Empty batch with valid head and value dims
    assert ms_out_all[1].shape == (0, 2, 8)   # Softmax max/sum keep last dim as 8
    assert ms_out_all[2].shape == (0, 2, 8)

    # Test case 3: All dimensions completely empty (all dimensions = 0)
    T1, T2, N1, N2, D1, D2 = 0, 0, 0, 0, 0, 0
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32)
    scale_value = 1.0 if D1 == 0 else round(1.0 / (D1 ** 0.5), 6)  # Handle D1=0 case
    head_num = 0
    actual_seq_qlen = [0]
    actual_seq_kvlen = [0]
    empty_select_block_count = 0
    topk_indices_np = np.array([]).reshape(0, 0, empty_select_block_count).astype(np.int32)

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    ms_out_completely = nsa_select_attention_forward_func(
        ms_query, ms_key, ms_value, ms_topk_indices,
        scale_value, head_num, select_block_size, empty_select_block_count,
        None, actual_seq_qlen, actual_seq_kvlen)
    _pynative_executor.sync()
    assert ms_out_completely[0].shape == (0, 0, 0)  # All dimensions empty
    assert ms_out_completely[1].shape == (0, 0, 8)  # Softmax max/sum keep last dim as 8
    assert ms_out_completely[2].shape == (0, 0, 8)

    # Test case 4: Non-empty query but empty key/value sequence (T1>0, T2=0) - Should fail
    T1, T2, N1, N2, D1, D2 = 2, 0, 3, 3, 192, 128
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = N1
    actual_seq_qlen = [2]
    actual_seq_kvlen = [0]
    empty_select_block_count = 0
    topk_indices_np = np.array([]).reshape(2, 3, empty_select_block_count).astype(np.int32)

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    with pytest.raises((RuntimeError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices, scale_value, head_num,
                                              select_block_size, empty_select_block_count, None, actual_seq_qlen,
                                              actual_seq_kvlen)
        _pynative_executor.sync()

    # Test case 5: Non-empty sequence but empty head dimension (N1=0, N2=0)
    T1, T2, N1, N2, D1, D2 = 3, 1024, 0, 0, 192, 128
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = 0
    actual_seq_qlen = [3]
    actual_seq_kvlen = [1024]
    # For empty heads, select_block_count can be standard 16
    topk_indices_np = np.array([]).reshape(3, 0, select_block_count).astype(np.int32)

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    ms_out_no_heads = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices, scale_value,
                                                        head_num, select_block_size, select_block_count, None,
                                                        actual_seq_qlen, actual_seq_kvlen)
    _pynative_executor.sync()
    assert ms_out_no_heads[0].shape == (3, 0, 128)  # No heads means empty head dimension
    assert ms_out_no_heads[1].shape == (3, 0, 8)
    assert ms_out_no_heads[2].shape == (3, 0, 8)

    # Test case 6: Empty query/key feature dimension (D1=0) but other dims valid - Should fail
    T1, T2, N1, N2, D1, D2 = 1, 1024, 2, 2, 0, 128
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32)
    scale_value = 1.0 if D1 == 0 else round(1.0 / (D1 ** 0.5), 6)  # Handle D1=0 case
    head_num = N1
    actual_seq_qlen = [1]
    actual_seq_kvlen = [1024]
    # Can still generate standard topk_indices despite D1=0
    topk_indices_np = get_topk_indices(np.random.randn(1, 2, 192).astype(np.float32), value_np,
                                       select_block_size, select_block_count, actual_seq_qlen)

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    with pytest.raises((RuntimeError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices, scale_value, head_num,
                                              select_block_size, select_block_count, None, actual_seq_qlen,
                                              actual_seq_kvlen)
        _pynative_executor.sync()

    # Test case 7: Empty value feature dimension (D2=0) but other dims valid - Should fail
    T1, T2, N1, N2, D1, D2 = 1, 1024, 3, 3, 192, 0
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = N1
    actual_seq_qlen = [1]
    actual_seq_kvlen = [1024]
    topk_indices_np = get_topk_indices(query_np, np.random.randn(1024, 3, 128).astype(np.float32),
                                       select_block_size, select_block_count, actual_seq_qlen)

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    with pytest.raises((RuntimeError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                              scale_value, head_num, select_block_size, select_block_count,
                                              None, actual_seq_qlen, actual_seq_kvlen)
        _pynative_executor.sync()

    # Test case 8: Mixed zero dimensions - T1=0 but T2>0, different from case 1
    T1, T2, N1, N2, D1, D2 = 0, 1024, 6, 6, 192, 128
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = N1
    actual_seq_qlen = [0]
    actual_seq_kvlen = [1024]
    # For empty query with non-empty kv, use standard select_block_count
    topk_indices_np = np.array([]).reshape(0, 6, select_block_count).astype(np.int32)

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    ms_out_mixed = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                                     scale_value, head_num, select_block_size, select_block_count,
                                                     None, actual_seq_qlen, actual_seq_kvlen)
    _pynative_executor.sync()
    assert ms_out_mixed[0].shape == (0, 6, 128)  # Empty query batch but full feature dims
    assert ms_out_mixed[1].shape == (0, 6, 8)
    assert ms_out_mixed[2].shape == (0, 6, 8)

    # Test case 9: Asymmetric head configurations (N1>0, N2=0) - Should fail
    query_np = np.random.randn(3, 8, 192).astype(np.float32)  # 8 query heads
    key_np = np.array([]).reshape(1024, 0, 192).astype(np.float32)
    value_np = np.array([]).reshape(1024, 0, 128).astype(np.float32)  # 0 kv heads
    scale_value = round(1.0 / (192 ** 0.5), 6)  # Ensure float32 precision for D1=192
    head_num = 0  # Since N2=0, head_num should be 0
    actual_seq_qlen = [3]
    actual_seq_kvlen = [1024]
    topk_indices_np = np.array([]).reshape(3, 0, select_block_count).astype(np.int32)

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    with pytest.raises((RuntimeError)):
        # Note: This should result in no attention computation due to N2=0
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                              scale_value, head_num, select_block_size, select_block_count,
                                              None, actual_seq_qlen, actual_seq_kvlen)
        _pynative_executor.sync()

    # Test case 10: Large sequences with zero feature dimensions (D1=D2=0) - Should fail
    T1, T2, N1, N2, D1, D2 = 100, 1024, 4, 4, 0, 0
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32)
    scale_value = 1.0 if D1 == 0 else round(1.0 / (D1 ** 0.5), 6)  # Handle D1=0 case
    head_num = N1
    actual_seq_qlen = [100]
    actual_seq_kvlen = [1024]
    # Generate standard topk_indices despite zero feature dimensions
    topk_indices_np = get_topk_indices(np.random.randn(100, 4, 192).astype(np.float32),
                                       np.random.randn(1024, 4, 128).astype(np.float32),
                                       select_block_size, select_block_count, actual_seq_qlen)

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    with pytest.raises((RuntimeError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                              scale_value, head_num, select_block_size, select_block_count,
                                              None, actual_seq_qlen, actual_seq_kvlen)
        _pynative_executor.sync()

    # Test case 11: Only D1=0, all others non-zero - Should fail
    T1, T2, N1, N2, D1, D2 = 2, 1024, 3, 3, 0, 128
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32)
    scale_value = 1.0 if D1 == 0 else round(1.0 / (D1 ** 0.5), 6)  # Handle D1=0 case
    head_num = N1
    actual_seq_qlen = [2]
    actual_seq_kvlen = [1024]
    # Generate standard topk_indices
    topk_indices_np = get_topk_indices(np.random.randn(2, 3, 192).astype(np.float32), value_np,
                                       select_block_size, select_block_count, actual_seq_qlen)

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    with pytest.raises((RuntimeError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                              scale_value, head_num, select_block_size, select_block_count,
                                              None, actual_seq_qlen, actual_seq_kvlen)
        _pynative_executor.sync()

    # Test case 12: Only D2=0, all others non-zero - Should fail
    T1, T2, N1, N2, D1, D2 = 4, 1024, 2, 2, 192, 0
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = N1
    actual_seq_qlen = [4]
    actual_seq_kvlen = [1024]
    topk_indices_np = get_topk_indices(query_np, np.random.randn(1024, 2, 128).astype(np.float32),
                                       select_block_size, select_block_count, actual_seq_qlen)

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    with pytest.raises((RuntimeError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                              scale_value, head_num, select_block_size, select_block_count,
                                              None, actual_seq_qlen, actual_seq_kvlen)
        _pynative_executor.sync()

    # Test case 13: Only N1=N2=0, other dimensions non-zero (different config from case 5)
    T1, T2, N1, N2, D1, D2 = 8, 1024, 0, 0, 192, 128
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = 0
    actual_seq_qlen = [8]
    actual_seq_kvlen = [1024]
    topk_indices_np = np.array([]).reshape(8, 0, select_block_count).astype(np.int32)

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    ms_out_only_heads_zero = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                                               scale_value, head_num, select_block_size,
                                                               select_block_count, None, actual_seq_qlen,
                                                               actual_seq_kvlen)
    _pynative_executor.sync()
    assert ms_out_only_heads_zero[0].shape == (8, 0, 128)  # Sequence length preserved but no heads
    assert ms_out_only_heads_zero[1].shape == (8, 0, 8)
    assert ms_out_only_heads_zero[2].shape == (8, 0, 8)

    # Test case 14: Only T2=0, other dimensions non-zero - Should fail
    T1, T2, N1, N2, D1, D2 = 5, 0, 1, 1, 192, 128
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = N1
    actual_seq_qlen = [5]
    actual_seq_kvlen = [0]
    empty_select_block_count = 0
    topk_indices_np = np.array([]).reshape(5, 1, empty_select_block_count).astype(np.int32)

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    with pytest.raises((RuntimeError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                              scale_value, head_num, select_block_size, empty_select_block_count,
                                              None, actual_seq_qlen, actual_seq_kvlen)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_ops_nsa_select_attention_invalid_constant_scalar_tensors(mode):
    """
    Feature: pyboost function.
    Description: test function nsa_select_attention with scalar constant tensors (shape=[]).

    q, k, v, and topk_indices are all scalar tensors (shape=[]), which violate API shape requirements.

    Expectation: expect ValueError/RuntimeError/TypeError for invalid scalar tensor inputs.
    """
    set_context_mode(mode)

    # Constant scalar tensors (shape=[])
    ms_query = ms.Tensor(1.0, dtype=ms.float16)
    ms_key = ms.Tensor(1.0, dtype=ms.float16)
    ms_value = ms.Tensor(1.0, dtype=ms.float16)
    ms_topk_indices = ms.Tensor(0, dtype=ms.int32)

    # Any valid-looking scalar params here still must fail due to invalid input ranks
    with pytest.raises((ValueError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                              1.0, 1, 64, 1, None, [1], [64])
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=584.00)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['bf16'])
def test_ops_nsa_select_attention_non_contiguous(mode, dtype):
    """
    Feature: pyboost function.
    Description: test function nsa_select_attention with non-contiguous tensors created by transpose+permute operations.

    Tests nsa_select_attention with tensors that are made non-contiguous through:
    1. Initial numpy transpose operations: query(0,2,1), key(0,2,1), value(1,0,2), topk_indices(0,2,1)
    2. Additional MindSpore/PyTorch permute operations: (0,2,1) for all tensors
    This ensures all input tensors have is_contiguous() == False before computation.

    Test dimensions: T1=2, T2=192, N1=4, N2=2, D1=192, D2=128, scale=1/sqrt(192)

    Expectation: expect correct forward/backward results with non-contiguous tensors and match torch precision.
    """
    set_context_mode(mode)

    T1, T2, N1, N2, D1, D2 = 1, 1088, 115, 115, 192, 128
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = N1
    select_block_count = 16
    select_block_size = 64
    actual_seq_qlen = [1]
    actual_seq_kvlen = [1088]
    topk_indices_np = get_topk_indices(query_np, value_np, select_block_size, select_block_count, actual_seq_qlen)

    # Create tensors with shapes that allow proper non-contiguous operations
    query_np = query_np.transpose(0, 2, 1).copy()
    key_np = key_np.transpose(0, 2, 1).copy()
    value_np = value_np.transpose(0, 2, 1).copy()
    topk_indices_np = topk_indices_np.transpose(0, 2, 1).copy()

    pt_query, pt_key, pt_value, pt_topk_indices = get_torch_tensors(query_np, key_np, value_np, topk_indices_np, dtype)
    pt_query_permute = torch.permute(pt_query, (0, 2, 1))
    pt_key_permute = torch.permute(pt_key, (0, 2, 1))
    pt_value_permute = torch.permute(pt_value, (0, 2, 1))
    pt_topk_indices_permute = torch.permute(pt_topk_indices, (0, 2, 1))

    assert not pt_query_permute.is_contiguous()
    assert not pt_key_permute.is_contiguous()
    assert not pt_value_permute.is_contiguous()
    assert not pt_topk_indices_permute.is_contiguous()

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)
    ms_query_permute = ms.ops.permute(ms_query, (0, 2, 1))
    ms_key_permute = ms.ops.permute(ms_key, (0, 2, 1))
    ms_value_permute = ms.ops.permute(ms_value, (0, 2, 1))
    ms_topk_indices_permute = ms.ops.permute(ms_topk_indices, (0, 2, 1))

    assert not ms_query_permute.is_contiguous()
    assert not ms_key_permute.is_contiguous()
    assert not ms_value_permute.is_contiguous()
    assert not ms_topk_indices_permute.is_contiguous()

    # PyTorch nsa_select_attention forward and backward
    torch_out = torch_golden_expect_forward_backward(pt_query_permute, pt_key_permute, pt_value_permute,
                                                     pt_topk_indices_permute, scale_value, head_num,
                                                     select_block_size, select_block_count, None, actual_seq_qlen,
                                                     actual_seq_kvlen)

    ms_out = nsa_select_attention_forward_func(ms_query_permute, ms_key_permute, ms_value_permute,
                                               ms_topk_indices_permute, scale_value, head_num,
                                               select_block_size, select_block_count, None, actual_seq_qlen,
                                               actual_seq_kvlen)
    # Compare MindSpore with PyTorch forward results
    compare_results(torch_out[:3], ms_out, grad=False)

    # MindSpore nsa_select_attention backward
    ms_out_grad = nsa_select_attention_backward_func(ms_query_permute, ms_key_permute, ms_value_permute,
                                                     ms_topk_indices_permute, scale_value, head_num,
                                                     select_block_size, select_block_count, None, actual_seq_qlen,
                                                     actual_seq_kvlen)
    # Compare MindSpore with PyTorch backward results
    compare_results(torch_out[3:], ms_out_grad, grad=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', [ms.float32, ms.float64, ms.int32, ms.int64, ms.uint8, ms.bool_])
def test_ops_nsa_select_attention_invalid_dtype(mode, dtype):
    """
    Feature: pyboost function.
    Description: test function nsa_select_attention with invalid data types.

    Tests that unsupported data types (float32, float64, int32, int64, uint8, bool_)
    are properly rejected. Only float16 and bfloat16 are supported according to API doc.

    Expectation: expect correct error message for unsupported data types.
    """
    set_context_mode(mode)

    T1, T2, N1, N2, D1, D2 = 1, 64, 1, 1, 192, 128
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = N1
    select_block_count = 16
    select_block_size = 64
    actual_seq_qlen = [1]
    actual_seq_kvlen = [64]

    topk_indices_np = get_topk_indices(query_np, value_np, select_block_size, select_block_count, actual_seq_qlen)

    ms_query = ms.Tensor(query_np, dtype=dtype)
    ms_key = ms.Tensor(key_np, dtype=dtype)
    ms_value = ms.Tensor(value_np, dtype=dtype)
    ms_topk_indices = ms.Tensor(topk_indices_np)

    with pytest.raises((TypeError, RuntimeError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                              scale_value, head_num, select_block_size, select_block_count,
                                              None, actual_seq_qlen, actual_seq_kvlen)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_ops_nsa_select_attention_invalid_select_block_size(mode):
    """
    Feature: pyboost function.
    Description: test function nsa_select_attention with invalid select_block_size.

    Tests that select_block_size must be 64 according to API specification.
    Any other value should raise an error.

    Expectation: expect correct error message for non-64 select_block_size.
    """
    set_context_mode(mode)

    T1, T2, N1, N2, D1, D2 = 1, 64, 1, 1, 192, 128
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = N1
    select_block_count = 15
    select_block_size = 63
    actual_seq_qlen = [1]
    actual_seq_kvlen = [64]

    # Generate proper topk_indices for valid scenario (before testing invalid block_size)
    topk_indices_np = get_topk_indices(query_np, value_np, select_block_size, select_block_count, actual_seq_qlen)

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, 'fp16')

    # Test with invalid select_block_size (128 instead of required 64)
    with pytest.raises((ValueError, RuntimeError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                              scale_value, head_num, select_block_size, select_block_count,
                                              None, actual_seq_qlen, actual_seq_kvlen)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'],
        level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=2.0)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_ops_nsa_select_attention_invalid_head_num(mode):
    """
    Feature: pyboost function.
    Description: test function nsa_select_attention with invalid head_num.

    Tests that head_num parameter must match the N1 dimension of query tensor.
    According to API spec, head_num should equal query.shape[1].

    Expectation: expect correct error message for head_num that doesn't match query dimension.
    """
    set_context_mode(mode)

    T1, T2, N1, N2, D1, D2 = 1, 64, 1, 1, 192, 128
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    select_block_count = 16
    select_block_size = 64
    actual_seq_qlen = [1]
    actual_seq_kvlen = [64]

    topk_indices_np = get_topk_indices(query_np, value_np, select_block_size, select_block_count, actual_seq_qlen)

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, 'fp16')

    # Test with invalid head_num (2 doesn't match query N1 dimension which is 1)
    with pytest.raises((ValueError, RuntimeError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices, scale_value, 2,
                                              select_block_size, select_block_count, None, actual_seq_qlen,
                                              actual_seq_kvlen)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'],
        level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=340.0)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['fp16'])
@pytest.mark.parametrize('scale_value', [-0.0077, 0.0, 100.0])
def test_ops_nsa_select_attention_scale_value_variants(mode, dtype, scale_value):
    """
    Feature: pyboost function.
    Description: test function nsa_select_attention with various scale_value.

    Tests that the algorithm correctly handles different scale_value including
    negative, zero, and positive values. Compares results with torch for precision.

    Expectation: expect correct result and match torch precision.
    """
    set_context_mode(mode)
    T1, T2, N1, N2, D1, D2 = 1, 1088, 115, 115, 192, 128
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2)
    head_num = N1
    select_block_count = 16
    select_block_size = 64
    actual_seq_qlen = [1]
    actual_seq_kvlen = [1088]
    topk_indices_np = get_topk_indices(query_np, value_np, select_block_size, select_block_count, actual_seq_qlen)

    pt_query, pt_key, pt_value, pt_topk_indices = get_torch_tensors(query_np, key_np, value_np, topk_indices_np, dtype)
    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    # PyTorch nsa_select_attention forward and backward
    torch_out = torch_golden_expect_forward_backward(pt_query, pt_key, pt_value, pt_topk_indices, scale_value,
                                                     head_num, select_block_size, select_block_count, None,
                                                     actual_seq_qlen, actual_seq_kvlen)
    # MindSpore nsa_select_attention forward
    ms_out = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                               scale_value, head_num, select_block_size, select_block_count,
                                               None, actual_seq_qlen, actual_seq_kvlen)
    # Compare MindSpore with PyTorch forward
    compare_results(torch_out[:3], ms_out, grad=False)

    # MindSpore nsa_select_attention backward
    ms_out_grad = nsa_select_attention_backward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                                     scale_value, head_num, select_block_size, select_block_count,
                                                     None, actual_seq_qlen, actual_seq_kvlen)
    # Compare MindSpore with PyTorch backward
    compare_results(torch_out[3:], ms_out_grad, grad=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_ops_nsa_select_attention_shape_mismatch(mode):
    """
    Feature: pyboost function.
    Description: test function nsa_select_attention with query/key feature dimension mismatch.

    Tests that query and key tensors must have the same feature dimension (D1).
    Query shape: (T1, N1, D1=192), Key shape: (T2, N2, D1=128) - D1 dimension mismatch.

    Expectation: expect correct error message for D1 dimension mismatch between query and key.
    """
    set_context_mode(mode)

    query_np = np.random.randn(1, 1, 192).astype(np.float32)
    key_np = np.random.randn(64, 1, 128).astype(np.float32)  # Different dimension
    value_np = np.random.randn(64, 1, 128).astype(np.float32)
    topk_indices_np = np.array([[[0]]]).astype(np.int32)

    ms_query = ms.Tensor(query_np, dtype=ms.float16)
    ms_key = ms.Tensor(key_np, dtype=ms.float16)  # Shape mismatch with query
    ms_value = ms.Tensor(value_np, dtype=ms.float16)
    ms_topk_indices = ms.Tensor(topk_indices_np)

    with pytest.raises((ValueError, RuntimeError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                              1.0, 1, 64, 1, None, [1], [64])
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'],
        level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('value', [np.inf, -np.inf, np.nan])
def test_ops_nsa_select_attention_qkv_inf_nan_values(mode, value):
    """
    Feature: pyboost function.
    Description: test function nsa_select_attention with special values (inf, -inf, nan) in Q/K/V tensors.

    Tests that the algorithm correctly handles special floating-point values when all query, key, and value
    tensors are filled with the same special value (positive infinity, negative infinity, or NaN).
    Compares MindSpore results with PyTorch reference implementation for consistency.

    Expectation: expect correct computation and match torch precision for inf/nan value handling.
    """

    if not _env_has_torch_npu or _ci_environment:
        print("torch_npu is not installed, skip test because of cpu golden expect will raise IndexError.")
        return

    set_context_mode(mode)

    T1, T2, N1, N2, D1, D2 = 1, 64, 1, 1, 192, 128
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = N1
    select_block_count = 16
    select_block_size = 64
    actual_seq_qlen = [1]
    actual_seq_kvlen = [64]

    topk_indices_np = get_topk_indices(query_np, value_np, select_block_size, select_block_count, actual_seq_qlen)

    # Add inf values to different tensors
    query_np.fill(value)
    key_np.fill(value)
    value_np.fill(value)

    pt_query, pt_key, pt_value, pt_topk_indices = get_torch_tensors(query_np, key_np, value_np, topk_indices_np, 'fp16')
    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, 'fp16')

    # PyTorch nsa_select_attention forward and backward
    torch_out = torch_golden_expect_forward_backward(pt_query, pt_key, pt_value, pt_topk_indices,
                                                     scale_value, head_num, select_block_size, select_block_count,
                                                     None, actual_seq_qlen, actual_seq_kvlen)
    # MindSpore nsa_select_attention forward
    ms_out = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                               scale_value, head_num, select_block_size, select_block_count,
                                               None, actual_seq_qlen, actual_seq_kvlen)
    # Compare MindSpore with PyTorch forward
    compare_results(torch_out[:3], ms_out, grad=False)

    # MindSpore nsa_select_attention backward
    ms_out_grad = nsa_select_attention_backward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                                     scale_value, head_num, select_block_size, select_block_count,
                                                     None, actual_seq_qlen, actual_seq_kvlen)
    # Compare MindSpore with PyTorch backward
    compare_results(torch_out[3:], ms_out_grad, grad=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.skip(reason="invalid topk_indices will cause stream error, remove this mark while testing locally.")
def test_ops_nsa_select_attention_topk_indices_boundary(mode):
    """
    Feature: pyboost function.
    Description: test function nsa_select_attention with negative topk_indices values.

    Tests that topk_indices must contain non-negative values. The test generates valid
    topk_indices using normal parameters (T1=1, T2=1088, N1=115, N2=115) and then
    converts all indices to negative values to trigger boundary condition validation.

    Expectation: expect correct error handling for negative topk_indices values.
    """
    set_context_mode(mode)

    T1, T2, N1, N2, D1, D2 = 1, 1088, 115, 115, 192, 128
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = N1
    select_block_count = 16
    select_block_size = 64
    actual_seq_qlen = [1]
    actual_seq_kvlen = [1088]
    topk_indices_np = get_topk_indices(query_np, value_np, select_block_size, select_block_count, actual_seq_qlen)
    topk_indices_np = -topk_indices_np

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, 'fp16')

    with pytest.raises((ValueError, RuntimeError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices, scale_value, head_num,
                                              select_block_size, select_block_count, None, actual_seq_qlen,
                                              actual_seq_kvlen)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'],
        level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_ops_nsa_select_attention_mixed_dtype(mode):
    """
    Feature: pyboost function.
    Description: test function nsa_select_attention with inconsistent tensor data types.

    Tests that query, key, and value tensors must have consistent data types.
    Query: float16, Key: bfloat16 (different), Value: float16 - dtype inconsistency.
    Only float16 or bfloat16 are supported, and all must be the same type.

    Expectation: expect TypeError/RuntimeError for mixed float16 and bfloat16 data types.
    """
    set_context_mode(mode)

    query_np = np.random.randn(1, 1, 192).astype(np.float32)
    key_np = np.random.randn(64, 1, 192).astype(np.float32)
    value_np = np.random.randn(64, 1, 128).astype(np.float32)
    topk_indices_np = np.array([[[0]]]).astype(np.int32)

    # Mixed fp16 and bf16
    ms_query = ms.Tensor(query_np, dtype=ms.float16)
    ms_key = ms.Tensor(key_np, dtype=ms.bfloat16)  # Different dtype
    ms_value = ms.Tensor(value_np, dtype=ms.float16)
    ms_topk_indices = ms.Tensor(topk_indices_np)

    with pytest.raises((TypeError, RuntimeError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                              1.0, 1, 64, 1, None, [1], [64])
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', ['kbk', 'pynative'])
def test_ops_nsa_select_attention_default_params(mode):
    """
    Feature: pyboost function.
    Description: test function nsa_select_attention with None required parameters.

    Tests that required parameters actual_seq_qlen and actual_seq_kvlen cannot be None.
    According to API specification, these parameters are required for proper computation.

    Expectation: expect ValueError/RuntimeError when actual_seq_qlen or actual_seq_kvlen is None.
    """
    set_context_mode(mode)

    T1, T2, N1, N2, D1, D2 = 1, 1088, 115, 115, 192, 128
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = N1
    select_block_count = 16
    select_block_size = 64
    actual_seq_qlen = [1]
    actual_seq_kvlen = [1088]
    topk_indices_np = get_topk_indices(query_np, value_np, select_block_size, select_block_count, actual_seq_qlen)

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, 'fp16')

    with pytest.raises((ValueError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices, scale_value, head_num,
                                              select_block_size, select_block_count, None, None, None)
        _pynative_executor.sync()

    with pytest.raises((ValueError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices, scale_value, head_num,
                                              select_block_size, select_block_count, None, actual_seq_qlen, None)
        _pynative_executor.sync()

    with pytest.raises((ValueError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices, scale_value, head_num,
                                              select_block_size, select_block_count, None, None, actual_seq_kvlen)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_ops_nsa_select_attention_large_scale_precision(mode, dtype):
    """
    Feature: pyboost function.
    Description: test function nsa_select_attention with large scale dimensions for precision validation.

    Tests large-scale input scenarios with torch precision alignment:
    - Large sequence lengths (T1=128, T2=2048)
    - High head count (N1=N2=512, within 768 limit)
    - High-dimensional features (D1=512, D2=256)
    - Compares forward and backward results with PyTorch reference

    Expectation: expect correct computation results and precise alignment with torch reference.
    """

    if not _env_has_torch_npu or _ci_environment:
        print("torch_npu is not installed, skip test because of cpu golden expect running too slow.")
        return

    set_context_mode(mode)

    # Large scale dimensions for precision stress testing
    T1_large, T2_large, N1_large, N2_large, D1_large, D2_large = 128, 2048, 512, 512, 512, 256
    query_large, key_large, value_large = get_qkv(T1_large, T2_large, N1_large, N2_large, D1_large, D2_large,
                                                  dtype=np.float32)
    # Ensure scale_value precision is within float32 range
    scale_value_large = round(1.0 / (D1_large ** 0.5), 6)
    head_num_large = query_large.shape[1]  # 512 heads (within 768 limit)
    select_block_count = 16  # Must be 16 per API constraint
    select_block_size = 64   # Must be 64 per API constraint
    actual_seq_qlen_large = [128]   # Full query sequence length
    actual_seq_kvlen_large = [2048] # Large 64-aligned KV sequence (2048 % 64 == 0)

    # Generate proper topk_indices with correct shape (128, 512, 16)
    topk_indices_large = get_topk_indices(query_large, value_large, select_block_size, select_block_count,
                                          actual_seq_qlen_large)

    pt_query_large, pt_key_large, pt_value_large, pt_topk_indices_large = get_torch_tensors(
        query_large, key_large, value_large, topk_indices_large, dtype)
    ms_query_large, ms_key_large, ms_value_large, ms_topk_indices_large = get_ms_tensors(
        query_large, key_large, value_large, topk_indices_large, dtype)

    # PyTorch nsa_select_attention forward and backward
    torch_out_large = torch_golden_expect_forward_backward(pt_query_large, pt_key_large, pt_value_large,
                                                           pt_topk_indices_large, scale_value_large, head_num_large,
                                                           select_block_size, select_block_count,
                                                           None, actual_seq_qlen_large, actual_seq_kvlen_large)

    # MindSpore nsa_select_attention forward
    ms_out_large = nsa_select_attention_forward_func(ms_query_large, ms_key_large, ms_value_large,
                                                     ms_topk_indices_large, scale_value_large, head_num_large,
                                                     select_block_size, select_block_count,
                                                     None, actual_seq_qlen_large, actual_seq_kvlen_large)
    # Compare MindSpore with PyTorch forward results
    compare_results(torch_out_large[:3], ms_out_large, grad=False)

    # MindSpore nsa_select_attention backward
    ms_out_grad_large = nsa_select_attention_backward_func(ms_query_large, ms_key_large, ms_value_large,
                                                           ms_topk_indices_large, scale_value_large, head_num_large,
                                                           select_block_size, select_block_count,
                                                           None, actual_seq_qlen_large, actual_seq_kvlen_large)
    if _env_has_torch_npu:
        print("aclnnNsaSelectedAttentionGrad does not support deterministic calculations in this case, "
              "So skip backward comparison.")
        return

    # Compare MindSpore with PyTorch backward results
    compare_results(torch_out_large[3:], ms_out_grad_large, grad=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=962.00)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['fp16'])
def test_ops_nsa_select_attention_batch_processing(mode, dtype):
    """
    Feature: pyboost function.
    Description: test function nsa_select_attention with multiple batch sizes for precision validation.

    Tests batch processing with different sequence lengths and compares results with PyTorch:
    - Multi-batch scenario: batch0=4 query tokens, batch1=6 query tokens
    - Large-scale KV lengths: batch0=1088 tokens, batch1=1088 tokens (matching normal test scale)
    - Head count: 115 heads (matching normal test configuration)
    - Validates forward and backward computation precision with PyTorch reference

    Expectation: expect correct batch processing and precise alignment with torch reference.
    """
    set_context_mode(mode)

    # Multi-batch scenario with different sequence lengths (referencing normal test scale)
    T1, T2, N1, N2, D1, D2 = 10, 2176, 115, 115, 192, 128  # 10 query, 2176 kv tokens (1088*2)
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = N1
    select_block_size = 64   # Must be 64 per API constraint
    select_block_count = 16  # Must be 16 per API constraint

    # Two batches: first batch has 4 query tokens, second has 6 query tokens
    actual_seq_qlen = [4, 10]  # Cumulative: batch0=4 tokens, batch1=6 tokens
    # Two batches: first batch has 1088 kv tokens, second has 1088 kv tokens (matching normal test scale)
    actual_seq_kvlen = [1088, 2176]  # Cumulative: batch0=1088 tokens, batch1=1088 tokens

    topk_indices_np = get_topk_indices(query_np, value_np, select_block_size, select_block_count, actual_seq_qlen)

    pt_query, pt_key, pt_value, pt_topk_indices = get_torch_tensors(query_np, key_np, value_np, topk_indices_np, dtype)
    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    # PyTorch nsa_select_attention forward and backward
    torch_out = torch_golden_expect_forward_backward(pt_query, pt_key, pt_value, pt_topk_indices,
                                                     scale_value, head_num, select_block_size, select_block_count,
                                                     None, actual_seq_qlen, actual_seq_kvlen)

    # MindSpore nsa_select_attention forward
    ms_out = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices, scale_value, head_num,
                                               select_block_size, select_block_count, None, actual_seq_qlen,
                                               actual_seq_kvlen)

    # Compare MindSpore with PyTorch forward results
    compare_results(torch_out[:3], ms_out, grad=False)

    if not _env_has_torch_npu:
        print("aclnnNsaSelectedAttentionGrad does not support deterministic calculations in this case, "
              "So skip backward comparison to avoid random failure.")
        return

    # MindSpore nsa_select_attention backward
    ms_out_grad = nsa_select_attention_backward_func(ms_query, ms_key, ms_value, ms_topk_indices, scale_value,
                                                     head_num, select_block_size, select_block_count, None,
                                                     actual_seq_qlen, actual_seq_kvlen)

    # Compare MindSpore with PyTorch backward results
    compare_results(torch_out[3:], ms_out_grad, grad=True)


@arg_mark(plat_marks=['platform_ascend910b'],
        level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_ops_nsa_select_attention_invalid_parameter_types(mode):
    """
    Feature: pyboost function.
    Description: test function nsa_select_attention with invalid parameter data types.

    Tests that the algorithm properly validates parameter data types according to API specification:
    - scale_value must be float type
    - head_num must be int type
    - select_block_size must be int type
    - select_block_count must be int type

    Expectation: expect TypeError/ValueError for parameters with invalid data types.
    """
    set_context_mode(mode)

    query_np = np.random.randn(1, 1, 192).astype(np.float32)
    key_np = np.random.randn(64, 1, 192).astype(np.float32)
    value_np = np.random.randn(64, 1, 128).astype(np.float32)
    topk_indices_np = np.array([[[0]]]).astype(np.int32)

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, 'fp16')

    # Test invalid scale_value type (string)
    with pytest.raises((TypeError, ValueError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                              "invalid_scale", 1, 64, 1, None, [1], [64])
        _pynative_executor.sync()

    # Test invalid head_num type (float)
    with pytest.raises((TypeError, ValueError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                              1.0, 1.5, 64, 1, None, [1], [64])
        _pynative_executor.sync()

    # Test invalid select_block_size type (string)
    with pytest.raises((TypeError, ValueError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                              1.0, 1, "64", 1, None, [1], [64])
        _pynative_executor.sync()

    # Test invalid select_block_count type (list)
    with pytest.raises((TypeError, ValueError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                              1.0, 1, 64, [1], None, [1], [64])
        _pynative_executor.sync()

    # Test invalid scale_value type (list)
    with pytest.raises((TypeError, ValueError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                              [1.0], 1, 64, 1, None, [1], [64])
        _pynative_executor.sync()

    # Test invalid head_num type (string)
    with pytest.raises((TypeError, ValueError)):
        _ = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                              1.0, "1", 64, 1, None, [1], [64])
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=90.0)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_ops_nsa_select_attention_large_seq_len_precision(mode, dtype):
    """
    Feature: pyboost function.
    Description: test function nsa_select_attention with large sequence lengths for precision validation.

    Tests large sequence length scenarios and compares results with PyTorch:
    - Large sequence lengths: T1=64 query tokens, T2=2048 KV tokens
    - Moderate head count: N1=N2=8 heads
    - Standard feature dimensions: D1=192, D2=128
    - Validates forward and backward computation precision with PyTorch reference

    Expectation: expect correct handling of large sequence data and precise alignment with torch reference.
    """
    set_context_mode(mode)

    # Test with relatively large sequence lengths
    T1, T2, N1, N2, D1, D2 = 64, 2048, 8, 8, 192, 128  # 64 query, 2048 kv tokens
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = N1
    select_block_size = 64   # Must be 64 per API constraint
    select_block_count = 16  # Must be 16 per API constraint
    actual_seq_qlen = [64]
    actual_seq_kvlen = [2048]

    topk_indices_np = get_topk_indices(query_np, value_np, select_block_size, select_block_count, actual_seq_qlen)

    pt_query, pt_key, pt_value, pt_topk_indices = get_torch_tensors(query_np, key_np, value_np, topk_indices_np, dtype)
    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    # PyTorch nsa_select_attention forward and backward
    torch_out = torch_golden_expect_forward_backward(pt_query, pt_key, pt_value, pt_topk_indices,
                                                     scale_value, head_num, select_block_size, select_block_count,
                                                     None, actual_seq_qlen, actual_seq_kvlen)

    # MindSpore nsa_select_attention forward
    ms_out = nsa_select_attention_forward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                               scale_value, head_num, select_block_size, select_block_count,
                                               None, actual_seq_qlen, actual_seq_kvlen)

    # Compare MindSpore with PyTorch forward results
    compare_results(torch_out[:3], ms_out, grad=False)

    # MindSpore nsa_select_attention backward
    ms_out_grad = nsa_select_attention_backward_func(ms_query, ms_key, ms_value, ms_topk_indices,
                                                     scale_value, head_num, select_block_size, select_block_count,
                                                     None, actual_seq_qlen, actual_seq_kvlen)

    if _env_has_torch_npu:
        print("aclnnNsaSelectedAttentionGrad does not support deterministic calculations in this case, "
              "So skip backward comparison.")
        return

    # Compare MindSpore with PyTorch backward results
    compare_results(torch_out[3:], ms_out_grad, grad=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=508.00)
def test_ops_nsa_select_attention_test_op():
    """
    Feature: pyboost function.
    Description: test function nsa_select_attention with test_op.
    Expectation: expect correct result.
    """
    def nsa_select_attention_func(query, key, value, topk_indices, scale_value, head_num, select_block_size,
                                  select_block_count, atten_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None):
        return ops.nsa_select_attention(query, key, value, topk_indices, scale_value, head_num, select_block_size,
                                        select_block_count, atten_mask=atten_mask, actual_seq_qlen=actual_seq_qlen,
                                        actual_seq_kvlen=actual_seq_kvlen)

    T1, T2, N1, N2, D1, D2 = 1, 1088, 115, 115, 192, 128
    query_np1, key_np1, value_np1 = get_qkv(T1, T2, N1, N2, D1, D2)
    scale_value1 = round(1.0 / (D1 ** 0.5), 6)
    head_num1 = N1
    select_block_count1 = 16
    select_block_size1 = 64
    actual_seq_qlen1 = [1]
    actual_seq_kvlen1 = [1088]
    topk_indices_np1 = get_topk_indices(query_np1, value_np1, select_block_size1, select_block_count1, actual_seq_qlen1)
    ms_query1, ms_key1, ms_value1, ms_topk_indices1 = get_ms_tensors(query_np1, key_np1, value_np1, topk_indices_np1,
                                                                     'bf16')

    T1, T2, N1, N2, D1, D2 = 1, 64, 1, 1, 192, 128
    query_np2, key_np2, value_np2 = get_qkv(T1, T2, N1, N2, D1, D2, dtype=np.float32)
    scale_value2 = round(1.0 / (D1 ** 0.5), 6)
    head_num2 = N1
    select_block_count2 = 16
    select_block_size2 = 64
    actual_seq_qlen2 = [1]
    actual_seq_kvlen2 = [64]

    topk_indices_np2 = get_topk_indices(query_np2, value_np2, select_block_size2, select_block_count2, actual_seq_qlen2)
    ms_query2, ms_key2, ms_value2, ms_topk_indices2 = get_ms_tensors(query_np2, key_np2, value_np2, topk_indices_np2,
                                                                     'bf16')

    input1 = [ms_query1, ms_key1, ms_value1, ms_topk_indices1, scale_value1, head_num1, select_block_size1,
              select_block_count1, None, actual_seq_qlen1, actual_seq_kvlen1]
    input2 = [ms_query2, ms_key2, ms_value2, ms_topk_indices2, scale_value2, head_num2, select_block_size2,
              select_block_count2, None, actual_seq_qlen2, actual_seq_kvlen2]
    TEST_OP(
        nsa_select_attention_func,
        [input1, input2],
        disable_mode=['GRAPH_MODE_GE'],
        disable_case=['ScalarTensor', 'Deterministic'],
        case_config={
            'disable_input_check': True,
            'disable_resize': True,
            'all_dim_zero': True,
        },
    )


def nsa_select_attention_memory_case(TND_params, actual_seq_qkvlen, dtype, grad_mark):
    def mindspore_forward_func(query, key, value, topk_indices, scale_value, head_num, select_block_size,
                               select_block_count, atten_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None):
        query = mint.abs(query)
        key = mint.abs(key)
        value = mint.abs(value)
        topk_indices = mint.abs(topk_indices)
        attention_out, softmax_max, softmax_sum = ops.nsa_select_attention(query, key, value, topk_indices, scale_value,
                                                                           head_num, select_block_size,
                                                                           select_block_count, atten_mask=atten_mask,
                                                                           actual_seq_qlen=actual_seq_qlen,
                                                                           actual_seq_kvlen=actual_seq_kvlen)
        attention_out = mint.abs(attention_out).sum()
        softmax_max = mint.abs(softmax_max).sum()
        softmax_sum = mint.abs(softmax_sum).sum()
        return attention_out

    def torch_npu_forward_func(query, key, value, topk_indices, scale_value, head_num, select_block_size,
                               select_block_count, atten_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None):
        query = torch.abs(query)
        key = torch.abs(key)
        value = torch.abs(value)
        topk_indices = torch.abs(topk_indices)
        attention_out, softmax_max, softmax_sum = torch_npu.npu_nsa_select_attention(query, key, value, topk_indices,
                                                                                     scale_value, head_num,
                                                                                     select_block_size,
                                                                                     select_block_count,
                                                                                     atten_mask=atten_mask,
                                                                                     actual_seq_qlen=actual_seq_qlen,
                                                                                     actual_seq_kvlen=actual_seq_kvlen)
        attention_out = torch.abs(attention_out).sum()
        softmax_max = torch.abs(softmax_max).sum()
        softmax_sum = torch.abs(softmax_sum).sum()
        return attention_out

    if not _env_has_torch_npu or _ci_environment:
        print("torch_npu is not installed, skip device memory test")
        return

    T1, T2, N1, N2, D1, D2 = TND_params[0], TND_params[1], TND_params[2], TND_params[3], TND_params[4], TND_params[5]
    query_np, key_np, value_np = get_qkv(T1, T2, N1, N2, D1, D2)
    scale_value = round(1.0 / (D1 ** 0.5), 6)  # Ensure float32 precision
    head_num = N1
    select_block_count = 16
    select_block_size = 64
    actual_seq_qlen = actual_seq_qkvlen[0]
    actual_seq_kvlen = actual_seq_qkvlen[1]
    topk_indices_np = get_topk_indices(query_np, value_np, select_block_size, select_block_count, actual_seq_qlen)

    print(f"\nStart {dtype} grad_position={grad_mark} memory test...\n"
          f"T1={T1}, T2={T2}, N1={N1}, N2={N2}, D1={D1}, D2={D2}\n"
          f"actual_seq_qlen={actual_seq_qlen}, actual_seq_kvlen={actual_seq_kvlen}")

    ms.runtime.reset_max_memory_allocated()
    ms_init_memory = ms.runtime.max_memory_allocated()
    print(f"ms_init_memory = {ms_init_memory}")

    ms_query, ms_key, ms_value, ms_topk_indices = get_ms_tensors(query_np, key_np, value_np, topk_indices_np, dtype)

    _ = mindspore_forward_func(ms_query, ms_key, ms_value, ms_topk_indices, scale_value, head_num,
                               select_block_size, select_block_count, None, actual_seq_qlen, actual_seq_kvlen)
    _pynative_executor.sync()
    ms_forward_memory = ms.runtime.max_memory_allocated()
    print(f"ms_forward_memory = {ms_forward_memory}")

    grad_postion = []
    if 'q' in grad_mark:
        grad_postion.append(0)
    if 'k' in grad_mark:
        grad_postion.append(1)
    if 'v' in grad_mark:
        grad_postion.append(2)
    grad_postion = tuple(grad_postion)
    _ = ms.grad(mindspore_forward_func, grad_position=grad_postion)(ms_query, ms_key, ms_value,
                                                                    ms_topk_indices, scale_value, head_num,
                                                                    select_block_size, select_block_count,
                                                                    None, actual_seq_qlen, actual_seq_kvlen)
    _pynative_executor.sync()
    ms_backward_memory = ms.runtime.max_memory_allocated()
    print(f"ms_backward_memory = {ms_backward_memory}")

    ms_forward_memory_diff = ms_forward_memory - ms_init_memory
    ms_backward_memory_diff = ms_backward_memory - ms_init_memory

    torch_npu.npu.reset_max_memory_allocated()
    pt_init_memory = torch_npu.npu.max_memory_allocated()
    print(f"pt_init_memory = {pt_init_memory}")
    pt_query, pt_key, pt_value, pt_topk_indices = get_torch_tensors(query_np, key_np, value_np, topk_indices_np, dtype)
    pt_query, pt_key, pt_value, pt_topk_indices = pt_query.npu(), pt_key.npu(), pt_value.npu(), pt_topk_indices.npu()

    if 'q' in grad_mark:
        pt_query.requires_grad = True
    if 'k' in grad_mark:
        pt_key.requires_grad = True
    if 'v' in grad_mark:
        pt_value.requires_grad = True

    pt_out = torch_npu_forward_func(pt_query, pt_key, pt_value, pt_topk_indices, scale_value, head_num,
                                    select_block_size, select_block_count, None, actual_seq_qlen, actual_seq_kvlen)
    torch.npu.synchronize()
    pt_forward_memory = torch_npu.npu.max_memory_allocated()
    print(f"pt_forward_memory = {pt_forward_memory}")

    pt_out.backward()
    torch.npu.synchronize()
    pt_backward_memory = torch_npu.npu.max_memory_allocated()
    print(f"pt_backward_memory = {pt_backward_memory}")

    pt_forward_memory_diff = pt_forward_memory - pt_init_memory
    pt_backward_memory_diff = pt_backward_memory - pt_init_memory

    print(f"ms_forward_memory_diff = {ms_forward_memory_diff}")
    print(f"ms_backward_memory_diff = {ms_backward_memory_diff}")
    print(f"pt_forward_memory_diff = {pt_forward_memory_diff}")
    print(f"pt_backward_memory_diff = {pt_backward_memory_diff}")

    assert ms_forward_memory_diff <= pt_forward_memory_diff
    assert ms_backward_memory_diff <= pt_backward_memory_diff


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_ops_nsa_select_attention_device_memory_large_size():
    """
    Feature: device memory comparison.
    Description: Compare device memory usage (forward/backward) between MindSpore and torch_npu
    for nsa_select_attention using dtype=float16 with gradients w.r.t. Q only.
    Expectation: MindSpore max memory increase is less than or equal to torch_npu.
    """
    TND_params = [10, 2176, 115, 115, 192, 128]
    actual_seq_qkvlen = [[4, 10], [1088, 2176]]
    nsa_select_attention_memory_case(TND_params, actual_seq_qkvlen, 'fp16', 'q')
    nsa_select_attention_memory_case(TND_params, actual_seq_qkvlen, 'fp16', 'k')
    nsa_select_attention_memory_case(TND_params, actual_seq_qkvlen, 'fp16', 'v')
    nsa_select_attention_memory_case(TND_params, actual_seq_qkvlen, 'bf16', 'qkv')


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_ops_nsa_select_attention_device_memory_small_size():
    """
    Feature: device memory comparison.
    Description: Compare device memory usage (forward/backward) between MindSpore and torch_npu
    for nsa_select_attention using dtype=float16 with gradients w.r.t. Q only.
    Expectation: MindSpore max memory increase is less than or equal to torch_npu.
    """
    TND_params = [1, 1088, 115, 115, 192, 128]
    actual_seq_qkvlen = [[1], [1088]]
    nsa_select_attention_memory_case(TND_params, actual_seq_qkvlen, 'fp16', 'q')
    nsa_select_attention_memory_case(TND_params, actual_seq_qkvlen, 'fp16', 'k')
    nsa_select_attention_memory_case(TND_params, actual_seq_qkvlen, 'fp16', 'v')
    nsa_select_attention_memory_case(TND_params, actual_seq_qkvlen, 'bf16', 'qkv')
