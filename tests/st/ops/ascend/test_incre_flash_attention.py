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
from tests.mark_utils import arg_mark
import numpy as np
import ctypes
import pytest
import math
import torch
import mindspore
import mindspore.context as context
from mindspore.nn import Cell
from mindspore import Tensor
from mindspore import mutable
import mindspore.ops as ops
from mindspore.ops.function.nn_func import incre_flash_attention

RANDOM_MAX = 1


def conv_float_to_u32(data_f):
    fp = ctypes.pointer(ctypes.c_float(data_f))
    cp = ctypes.cast(fp, ctypes.POINTER(ctypes.c_uint))
    data_hex = cp.contents.value
    result = (data_hex // 8192) * 8192
    return result


def trans_19bit(deqscale):
    res_19bit = np.zeros(deqscale.shape[0], dtype=np.uint64)

    for idx, scale in enumerate(deqscale):
        val = np.bitwise_and(((conv_float_to_u32(scale) >> 13) << 13),
                             0xffffffff)
        val = val.astype(np.uint64)
        res_19bit[idx] = val
    return res_19bit


def ceil_div(a, b):
    return int((a + b - 1) / b)


def gen_deqscale():
    temp_deqscale = np.random.uniform(1, 2, (1,))
    temp_deqscale = temp_deqscale.astype(np.float32)
    deqscale = trans_19bit(temp_deqscale)
    return deqscale


def dequant(x, deqscale, relu_weight):
    deqscale = np.frombuffer(deqscale, dtype=np.float32)
    deqscale = deqscale[:1][0]
    if relu_weight is None:
        relu_weight = deqscale
    else:
        relu_weight = np.full(deqscale.shape, relu_weight)
    res_cal = np.zeros(x.shape, np.float16)
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    x[n, c, h, w] = x[n, c, h, w].astype(np.float32)
                    if x[n, c, h, w] >= 0:
                        res_cal[n, c, h, w] = x[n, c, h, w] * deqscale
                    else:
                        res_cal[n, c, h, w] = x[n, c, h, w] * relu_weight
    return res_cal


def gen_quant_param(max_num, min_num):
    max0 = abs(max_num)
    min0 = abs(min_num)
    max_scale = max if max0 > min0 else min0

    temp_qscale = np.zeros(1).astype(np.float32)
    temp_offset = np.zeros(1).astype(np.float32)
    temp_qscale[0] = (127 / max_scale)
    temp_offset[0] = 0
    temp_qscale = temp_qscale.astype(np.float16).astype(np.float32)
    temp_offset = temp_offset.astype(np.float16).astype(np.float32)
    return temp_qscale, temp_offset


def s8_saturation(inputdata):
    if inputdata > 127:
        inputdata = 127
    elif inputdata < -128:
        inputdata = -128
    outdata = np.int8(inputdata)
    return outdata


def s9_saturation(inputdata):
    if inputdata > 255:
        inputdata = 255
    elif inputdata < -256:
        inputdata = -256
    return inputdata


def quant(x, qscale, qoffset):
    s8_res_cal = np.zeros(x.shape, np.int8)
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    s8_res_cal[n, c, h, w] = s8_saturation(
                        np.round(
                            s9_saturation(
                                np.half(x[n, c, h, w]) * np.half(qscale))) +
                        np.half(qoffset))

    return s8_res_cal


def softmax(x):
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y / x_sum
    ans = ans.astype(np.float16)
    return ans


def change_data_bf16(input_np):
    input_ms = Tensor(input_np, mindspore.bfloat16)
    out_np = input_ms.float().asnumpy()
    return out_np


def get_slopes(n_heads):
    n = 2**math.floor(math.log2(n_heads))
    m_0 = 2.0**(-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_hat_0 = 2.0**(-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return m


def gen_ifa_golden(shape,
                   actual_seq_length=None,
                   input_layout="BSH",
                   p_a_s_control=None,
                   dtype=mindspore.float16,
                   atten_mask_shape=None,
                   page_attention_flag=False,
                   block_size=0,
                   pse_shift_1n1s=True,
                   atten_mask_bool=True):
    in_dtype = np.float16
    out_dtype = np.float16
    if p_a_s_control is None:
        p_a_s_control = [True, True, True]

    if atten_mask_shape is None:
        atten_mask_shape = [1, 1, 1, 1]

    matmul_dtype = np.int32 if in_dtype == np.int8 else in_dtype

    B, N, S, D, kvN = shape
    if kvN == 0:
        kvN = N
    if D == 0:
        scalar = 1.0
    else:
        scalar = 1 / math.sqrt(D)
    dropscalar, _ = -10000, 1
    # 计算真实的actual seq length
    if not p_a_s_control[2]:
        actual_seq_length = None

    if actual_seq_length is None:
        actual_shape = [B]
        actual_seq_length = np.random.uniform(S, S, actual_shape).astype(
            np.int64).tolist()
    actual_seq_length_real = actual_seq_length
    if len(actual_seq_length) == 1:
        actual_seq_length_real = [actual_seq_length[0]] * B

    q_shape = [B, 1, N * D]
    kv_shape = [B, S, kvN * D]
    mask_shape = [B, 1, 1, S]

    if page_attention_flag:
        kv_shape = [B, max(actual_seq_length_real), kvN * D]

    p_shape = [1, N, 1, S] if pse_shift_1n1s else [B, N, 1, S]
    y = np.zeros([B, N, 1, D]).astype(np.float16)

    q = np.random.uniform(-1 * RANDOM_MAX, RANDOM_MAX,
                          q_shape).astype(in_dtype)
    k = np.random.uniform(-1 * RANDOM_MAX, RANDOM_MAX,
                          kv_shape).astype(in_dtype)
    v = np.random.uniform(-1 * RANDOM_MAX, RANDOM_MAX,
                          kv_shape).astype(in_dtype)

    if not p_a_s_control[1]:
        if atten_mask_bool:
            masks = np.zeros(mask_shape).astype(np.uint8).astype(np.bool_)
        else:
            masks = np.zeros(mask_shape).astype(np.uint8).astype(np.float16)
    else:
        if atten_mask_bool:
            masks = np.random.uniform(0, 2, mask_shape).astype(
                np.uint8).astype(np.bool_)
        else:
            masks = np.random.uniform(0, 2, mask_shape).astype(
                np.uint8).astype(np.float16)

    # 生成pse_shift
    pse_shift = np.zeros(p_shape)
    if p_a_s_control[0]:
        maya = get_slopes(p_shape[1])
        maya = maya.numpy()
        pse_shift = np.zeros(p_shape)
        for n in range(p_shape[1]):
            alibi_biases = np.zeros([1, p_shape[-1]])
            for x in range(0, p_shape[-1]):
                alibi_biases[0, x] = -1 * x
            pse_shift[:, n:n + 1, :, :] = alibi_biases * maya[n]
    pse_shift = pse_shift.astype(np.float16)
    if dtype == mindspore.bfloat16:
        q = change_data_bf16(q)
        k = change_data_bf16(k)
        v = change_data_bf16(v)

    q_right = q
    k_right = k
    v_right = v
    masks_right = Tensor(masks, mindspore.float16)
    if len(atten_mask_shape) == 2:
        squeeze = ops.Squeeze((1, 2))
        masks_right = squeeze(masks_right)
    elif len(atten_mask_shape) == 3:
        squeeze = ops.Squeeze(1)
        masks_right = squeeze(masks_right)
    masks_right = masks_right.asnumpy()
    # Paged Attention 场景，生成block_table, k_cache, v_cache
    block_table = None
    k_cache = None
    v_cache = None
    if page_attention_flag:
        blockNum = 0
        maxBlockNumPerSeq = ceil_div(max(actual_seq_length_real),
                                     block_size)  # TODO:更大的maxBlockNumPerSeq
        blockNumPerBlock = []
        for actual_seq in actual_seq_length_real:
            blockNumPerBlock.append(ceil_div(actual_seq, block_size))
            blockNum += ceil_div(actual_seq, block_size)
        block_idx_list = np.arange(0, blockNum, 1)
        block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)
        block_idx = 0

        block_table = [-1] * maxBlockNumPerSeq
        block_table = np.tile(block_table, (B, 1)).astype(np.int32)
        block_table_batch_idx = 0
        for idx in blockNumPerBlock:
            for j in range(idx):
                block_table[block_table_batch_idx][j] = (
                    block_idx_list[block_idx])
                block_idx += 1
            block_table_batch_idx += 1
        # k/v padding to blocksize
        k_tensor_bsh = np.zeros(
            (B, maxBlockNumPerSeq * block_size, kvN * D)).astype(in_dtype)
        v_tensor_bsh = np.zeros(
            (B, maxBlockNumPerSeq * block_size, kvN * D)).astype(in_dtype)
        k_tensor_bsh[:, :max(actual_seq_length_real), :] = k[:, :, :]
        v_tensor_bsh[:, :max(actual_seq_length_real), :] = v[:, :, :]

        # kv_shape [B, S, kvN * D] -> kv_cache [batch_num, block_size, h]
        k_cache = np.zeros((blockNum, block_size, kvN * D)).astype(in_dtype)
        v_cache = np.zeros((blockNum, block_size, kvN * D)).astype(in_dtype)
        for b in range(B):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * block_size
                if kv_cache_blk_id == -1:
                    continue
                else:
                    k_cache[kv_cache_blk_id,
                            0:block_size, :] = k_tensor_bsh[b, block_offset:(
                                block_offset + block_size), :]
                    v_cache[kv_cache_blk_id,
                            0:block_size, :] = v_tensor_bsh[b, block_offset:(
                                block_offset + block_size), :]

    deq_scale1 = gen_deqscale()
    deq_scale2 = gen_deqscale()
    softmax_total = np.zeros([B, N, 1, S]).astype(np.float16)

    new_q_shape = [B, 1, N, D]
    new_kv_shape = [B, S, kvN, D]
    if page_attention_flag:
        new_kv_shape = [B, max(actual_seq_length_real), kvN, D]
    q = q.reshape(new_q_shape).transpose(0, 2, 1, 3)  # 最终CPU计算，使用的是B N S D格式
    v = v.reshape(new_kv_shape).transpose(0, 2, 1, 3)
    k = k.reshape(new_kv_shape).transpose(0, 2, 1, 3)

    if input_layout == "BNSD":
        q_right = q
        k_right = k
        v_right = v

    nnumofqinonegroup = N // kvN

    max_num = np.finfo(np.float16).min
    min_num = np.finfo(np.float16).max

    for bidx in range(B):
        s_value = actual_seq_length_real[bidx]
        for nidx in range(N):
            kvnidx = nidx // nnumofqinonegroup
            q_cur = q[bidx:(bidx + 1), nidx:(nidx + 1), :, :]
            k_cur = k.transpose(0, 1, 3, 2)[bidx:(bidx + 1),
                                            kvnidx:(kvnidx + 1), :, :s_value]
            qk_cur = np.matmul(q_cur, k_cur, dtype=matmul_dtype)
            if in_dtype == np.int8:
                qk_cur = dequant(qk_cur, deq_scale1[0], None)
            qk_cur = qk_cur * scalar
            # add pse_shift
            if pse_shift_1n1s:
                # pse_shift shape is 1n1s
                pse_cur = pse_shift[:, nidx:nidx + 1, :, :s_value]
            else:
                # pse_shift shape is bn1s
                pse_cur = pse_shift[bidx:bidx + 1, nidx:nidx + 1, :, :s_value]
            qk_cur = qk_cur + pse_cur
            qk_cur = qk_cur + dropscalar * masks[bidx:
                                                 (bidx + 1), :, :, :s_value]

            softmax_res = softmax(qk_cur.astype(np.float16))
            softmax_total[bidx:(bidx + 1),
                          nidx:(nidx + 1), :, :s_value] = softmax_res

            if in_dtype == np.int8:
                max_num = max(max_num, softmax_res.max())
                min_num = min(min_num, softmax_res.min())

    quant_scale1, quant_scale2, quant_offset2 = np.zeros([1]).astype(
        np.float16), np.zeros([1]).astype(np.float16), np.zeros([1]).astype(
            np.float16)

    if in_dtype == np.int8:
        quant_scale1, _ = gen_quant_param(max_num, min_num)

    for bidx in range(B):
        s_value = actual_seq_length_real[bidx]
        for nidx in range(N):
            kvnidx = nidx // nnumofqinonegroup
            v_cur = v[bidx:(bidx + 1), kvnidx:(kvnidx + 1), :s_value, :]
            softmax_res = softmax_total[bidx:(bidx + 1),
                                        nidx:(nidx + 1), :, :s_value]
            if in_dtype == np.int8:
                softmax_res = quant(softmax_res, quant_scale1, 0)

            mm2_res = np.matmul(softmax_res, v_cur, dtype=matmul_dtype)

            if in_dtype == np.int8:
                mm2_res = dequant(mm2_res, deq_scale2, None)
            y[bidx:(bidx + 1), nidx:(nidx + 1), :, :] = mm2_res

    if out_dtype == np.int8:
        max_num = y.max()
        min_num = y.min()
        quant_scale2, quant_offset2 = gen_quant_param(max_num, min_num)
        y = quant(y, quant_scale2, quant_offset2)
    y = y.transpose(0, 2, 1, 3).reshape(q_right.shape)
    return q_right, k_right, v_right, masks_right, y, block_table, k_cache, v_cache, pse_shift, actual_seq_length_real


class IncreFlashAttentionFunc(Cell):

    def __init__(self,
                 num_heads,
                 input_layout,
                 scale_value,
                 num_key_value_heads,
                 kv_multy_control=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale_value = scale_value
        self.input_layout = input_layout
        self.num_key_value_heads = num_key_value_heads
        self.kv_multy_control = kv_multy_control
        self.ifa = incre_flash_attention

    def construct(self,
                  query,
                  key_i,
                  value_i,
                  attn_mask,
                  actual_seq_lengths,
                  pse_shift,
                  dequant_scale1,
                  quant_scale1,
                  dequant_scale2,
                  quant_scale2,
                  quant_offset2,
                  antiquant_scale=None,
                  antiquant_offset=None,
                  block_table=None,
                  block_size=0):
        out = self.ifa(query, key_i, value_i, attn_mask, actual_seq_lengths,
                       pse_shift, dequant_scale1, quant_scale1, dequant_scale2,
                       quant_scale2, quant_offset2, antiquant_scale,
                       antiquant_offset, block_table, self.num_heads,
                       self.input_layout, self.scale_value,
                       self.num_key_value_heads, block_size, 1)  # 0, 1
        return out


def net_forward_test(inputs: list):
    B, N, S, D, data_format, num_key_value_heads = inputs[0], inputs[1], inputs[2], inputs[3], \
        inputs[4], inputs[5]
    H = N * D

    if D == 0:
        scale_value = 1.0
    else:
        scale_value = 1.0 / math.sqrt(D)

    shape = [B, N, S, D, num_key_value_heads]

    actual_seq_lengths = []
    for _ in range(B):
        actual_seq_lengths.append(S)

    p_a_s_control = [inputs[6], inputs[7], inputs[8]]
    kv_multy = inputs[9]
    data_type = inputs[10]
    page_attention_flag = inputs[11]
    block_size = inputs[12]
    pse_shift_1n1s = inputs[13]
    atten_mask_bool = inputs[14]
    kv_list = inputs[15]
    atten_mask_shape = [B, S]

    if data_type == np.float16:
        ms_dtype = mindspore.float16
    else:
        ms_dtype = mindspore.bfloat16

    q_all, k_all, v_all, mask_all, _, block_table_all, k_cache, v_cache, \
        pse_shift, actual_seq_length_real = gen_ifa_golden(shape, actual_seq_lengths, data_format,
                                                           p_a_s_control, ms_dtype, atten_mask_shape,
                                                           page_attention_flag, block_size, pse_shift_1n1s,
                                                           atten_mask_bool)

    query = Tensor(q_all, dtype=ms_dtype)

    if page_attention_flag:
        if kv_list:
            key = mutable([
                Tensor(k_cache, dtype=ms_dtype),
            ])
            value = mutable([
                Tensor(v_cache, dtype=ms_dtype),
            ])
        else:
            key = mutable((Tensor(k_cache, dtype=ms_dtype),))
            value = mutable((Tensor(v_cache, dtype=ms_dtype),))
    else:
        if kv_list:
            key = mutable([
                Tensor(k_all, dtype=ms_dtype),
            ])
            value = mutable([
                Tensor(v_all, dtype=ms_dtype),
            ])
        else:
            key = mutable((Tensor(k_all, dtype=ms_dtype),))
            value = mutable((Tensor(v_all, dtype=ms_dtype),))

    pse_shift = Tensor(pse_shift, dtype=mindspore.float16)

    if atten_mask_bool:
        attn_mask = Tensor(mask_all, dtype=mindspore.bool_)
    else:
        attn_mask = Tensor(mask_all, dtype=mindspore.float16)

    block_table = None if block_table_all is None else Tensor(
        block_table_all, dtype=mindspore.int32)
    actual_seq_lengths = Tensor(actual_seq_length_real, dtype=mindspore.int64)

    if not p_a_s_control[0]:
        pse_shift = None
    if not p_a_s_control[1]:
        attn_mask = None
    if not p_a_s_control[2]:
        actual_seq_lengths = None

    net = IncreFlashAttentionFunc(N, data_format, scale_value,
                                  num_key_value_heads, kv_multy)

    ifa_result = net(query, key, value, attn_mask, actual_seq_lengths,
                     pse_shift, None, None, None, None, None, None, None,
                     block_table, block_size)
    if data_format == 'BSH':
        assert ifa_result.shape == (B, 1, H)
    else:
        assert ifa_result.shape == (B, N, 1, D)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_incre_flash_attention_bsh_fwd():
    """
    Feature: Test functional ifa operator.
    Description: bsh mode for ifa test.
    Expectation: Assert result compare with expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    # B, N, S, D, input_layerout, kvN, p_a_s_control[0], p_a_s_control[1], p_a_s_control[2], kv_multy(no used),
    # data_type, page_attention_flag, block_size, pse_shift_1n1s, atten_mask_bool, kv_list
    inputs = [
        1, 5, 4096, 128, "BSH", 1, True, True, True, False, np.float16, False,
        0, True, True, False
    ]
    net_forward_test(inputs)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_incre_flash_attention_bnsd_fwd():
    """
    Feature: Test functional ifa operator.
    Description: bnsd mode for ifa test.
    Expectation: Assert result compare with expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    # B, N, S, D, input_layerout, kvN, p_a_s_control[0], p_a_s_control[1], p_a_s_control[2], kv_multy(no used),
    # data_type, page_attention_flag, block_size, pse_shift_1n1s, atten_mask_bool, kv_list
    inputs = [
        1, 5, 4096, 128, "BNSD", 1, True, True, True, False, np.float16, False,
        0, True, True, False
    ]
    net_forward_test(inputs)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_incre_flash_attention_bsh_fwd_paged_attention():
    """
    Feature: Test functional ifa operator of paged attention.
    Description: bsh mode for ifa test of paged attention.
    Expectation: Assert result compare with expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    # B, N, S, D, input_layerout, kvN, p_a_s_control[0], p_a_s_control[1], p_a_s_control[2], kv_multy(no used),
    # data_type, page_attention_flag, block_size, pse_shift_1n1s, atten_mask_bool, kv_list
    inputs = [
        1, 5, 4096, 128, "BSH", 1, True, True, True, False, np.float16, True,
        128, True, True, False
    ]
    net_forward_test(inputs)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_incre_flash_attention_bnsd_fwd_paged_attention():
    """
    Feature: Test functional ifa operator of paged attention.
    Description: bnsd mode for ifa test of paged attention.
    Expectation: Assert result compare with expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    # B, N, S, D, input_layerout, kvN, p_a_s_control[0], p_a_s_control[1], p_a_s_control[2], kv_multy(no used),
    # data_type, page_attention_flag, block_size, pse_shift_1n1s, atten_mask_bool, kv_list
    inputs = [
        1, 5, 4096, 128, "BNSD", 1, True, True, True, False, np.float16, True,
        128, True, True, False
    ]
    net_forward_test(inputs)
