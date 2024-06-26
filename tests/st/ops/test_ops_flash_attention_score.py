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
import pytest
import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.ops.composite import GradOperation
from mindspore.ops import flash_attention_score
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def flash_attention_score_forward_func(query, key, value, head_num, real_shift=None, drop_mask=None,
                                       padding_mask=None, attn_mask=None, prefix=None, actual_seq_qlen=None,
                                       actual_seq_kvlen=None, keep_prob=0.9, input_layout='BSH', pre_tokens=65536,
                                       next_tokens=65536, scalar_value=1.0, inner_precise=0, sparse_mode=0):
    return flash_attention_score(query, key, value, head_num, real_shift, drop_mask, padding_mask, attn_mask,
                                 prefix, actual_seq_qlen, actual_seq_kvlen, keep_prob, scalar_value, pre_tokens,
                                 next_tokens, inner_precise, input_layout, sparse_mode)

class Grad(Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.network = network
        self.grad = GradOperation(get_all=True, sens_param=True)

    def construct(self, *inputs):
        return self.grad(self.network)(*inputs)


def tsoftmax(x):
    x_max = np.max(x, axis=-1, keepdims=True)[0]
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = np.sum(y, axis=-1, keepdims=True)
    ans = y / x_sum
    return ans, x_max, x_sum


def tsoftmax_grad(dp, softmax_res):
    muls = dp * softmax_res
    muls_r = np.sum(muls, axis=-1, keepdims=True)
    sub_r = dp - muls_r
    res = sub_r * softmax_res
    return res


def fas_forward(q, k, v, drop_mask, atten_mask, pse, scale, keep_prob):
    if pse is None:
        qk = np.multiply(np.matmul(q, k.transpose(0, 1, 3, 2)), scale)
    else:
        qk = np.multiply((np.matmul(q, k.transpose(0, 1, 3, 2)) + pse), scale)
    softmax_res, x_max, x_sum = tsoftmax(qk)
    if drop_mask is None:
        drop_res = softmax_res
    else:
        drop_res = softmax_res * drop_mask * (1.0 / (keep_prob))
    y = np.matmul(drop_res, v)
    return y, softmax_res, x_max, x_sum


def fas_backward(dx, q, k, v, softmax_res, drop_mask, pse, scale, keep_prob):
    dp = np.matmul(dx, v.transpose(0, 1, 3, 2))
    if drop_mask is None:
        drop_res = softmax_res.transpose(0, 1, 3, 2)
        dp_drop = dp
    else:
        drop_res = softmax_res * drop_mask*(1.0 / (keep_prob)).transpose(0, 1, 3, 2)
        dp_drop = dp * drop_mask * (1.0 / (keep_prob))
    dv = np.matmul(drop_res, dx)
    softmax_grad_res = (tsoftmax_grad(dp_drop, softmax_res) * scale)
    dq = np.matmul(softmax_grad_res, k)
    dk = np.matmul(softmax_grad_res.transpose(0, 1, 3, 2), q)
    return dq, dk, dv


def flash_attention_score_func(query, key, value, head_num, actual_seq_qlen=(4, 8), actual_seq_kvlen=(4, 8),
                               input_layout='TND', real_shift=None, drop_mask=None, padding_mask=None,
                               attn_mask=None, prefix=None, keep_prob=1.0, scalar_value=1.0, pre_tokens=65536,
                               next_tokens=65536, inner_precise=0, sparse_mode=0):
    return flash_attention_score(query, key, value, head_num, real_shift, drop_mask, padding_mask, attn_mask,
                                 prefix, actual_seq_qlen, actual_seq_kvlen, keep_prob, scalar_value, pre_tokens,
                                 next_tokens, inner_precise, input_layout, sparse_mode)

def generate_inputs(B, N1, N2, S1, S2, D, input_layout, dtype, return_tensor=True):
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
        raise ValueError(f"input_layout is invalid.")
    real_shift = None
    attn_mask = np.triu(np.ones([B, 1, S1, S2]))
    prefix = None
    if return_tensor:
        return Tensor(query, dtype=dtype), Tensor(key, dtype=dtype), Tensor(value, dtype=dtype), real_shift, \
        Tensor(attn_mask, dtype=mstype.uint8), prefix
    return query, key, value, real_shift, attn_mask, prefix

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [mstype.float16, mstype.bfloat16])
def test_ops_flash_attention_score(mode, dtype):
    """
    Feature: Pyboost function.
    Description: Test function flash attention score forward and backward.
    Expectation: Correct result.
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=mode)
    input_layout = "BNSD"
    N1 = 1
    head_num = N1
    padding_mask = None
    scale_value = 1.0
    keep_prob = 1.0
    pre_tokens = 65536
    next_tokens = 65536
    inner_precise = 0
    sparse_mode = 0
    query = np.array([[[[-0.22983274, 0.52305974, -0.24619523, -0.68436081, -0.6159003,
                         -0.29813264, -0.30243467, -0.86161009, 0.3531674, 0.29330656,
                         0.49375594, 0.18899898, 0.27588086, -0.07713666, 0.02064493,
                         -0.54287771]]]])
    key = np.array([[[[0.54226112, -0.77507697, -0.43685387, 0.92910044, 0.18111083,
                       0.87586272, 0.12202536, -0.04840969, -0.68629483, 0.67655794,
                       -0.47323952, 0.94174782, -0.51611285, -0.2766026, 0.35866267,
                       0.26934939]]]])
    value = np.array([[[[-0.42552355, 0.64300023, -0.60108409, 0.45352306, -0.3664402,
                         -0.42133802, 0.26913729, 0.23181166, 0.02401428, -0.5889721,
                         -0.05283948, 0.0316619, 0.95860307, -0.30445275, -0.83552493,
                         0.38094309]]]])
    dx = np.array([[[[-0.16176246, -0.91536644, -0.72173913, 0.31794797, 0.28690627,
                      -0.93322098, -0.09320809, -0.87175356, 0.998217, 0.82003949,
                      -0.74785121, 0.42856346, -0.90514853, 0.67772169, -0.51628007,
                      0.80124134]]]])

    real_shift = None
    attn_mask = None
    prefix = None

    ms_query = Tensor(query, dtype=dtype)
    ms_key = Tensor(key, dtype=dtype)
    ms_value = Tensor(value, dtype=dtype)
    ms_dx = Tensor(dx, dtype=dtype)
    drop_mask = None

    golden_y, softmax_res, _, _ = fas_forward(query, key, value, None, None, None, scale_value, keep_prob)
    dq, dk, dv = fas_backward(dx, query, key, value, softmax_res, None, None, scale_value, keep_prob)

    actual_output = flash_attention_score_forward_func(ms_query, ms_key, ms_value, head_num, real_shift, drop_mask,
                                                       padding_mask, attn_mask, prefix, None, None, keep_prob,
                                                       input_layout, pre_tokens, next_tokens, scale_value,
                                                       inner_precise, sparse_mode)

    flash_attention_score_backward = Grad(flash_attention_score_forward_func)
    actual_grad = flash_attention_score_backward(ms_query, ms_key, ms_value, head_num, real_shift, drop_mask,
                                                 padding_mask, attn_mask, prefix, None, None, keep_prob,
                                                 input_layout, pre_tokens, next_tokens, scale_value, inner_precise,
                                                 sparse_mode, ms_dx)

    if dtype == mstype.float16:
        y_diff = actual_output.asnumpy() - golden_y
        dq_diff = actual_grad[0].asnumpy() - dq
        dk_diff = actual_grad[1].asnumpy() - dk
        dv_diff = actual_grad[2].asnumpy() - dv

        expect_y_diff = np.array([[[[-1.3559375000027352e-05, 6.6176250000049563e-05,
                                     9.8712499999598080e-06, 9.0221250000022124e-05,
                                     -1.4878125000006293e-05, -4.8698750000020219e-05,
                                     -9.4321249999973489e-05, -1.3656250000271264e-07,
                                     3.0539843750010398e-06, 1.0491249999999841e-04,
                                     1.3552265625001170e-05, -1.5171484374999766e-05,
                                     -1.0697625000000155e-04, 9.3906249999942126e-06,
                                     7.5711250000054520e-05, -8.3714999999984219e-05]]]])
        expect_dq_diff = np.array([[[[-5.960464477539063e-08, 5.960464477539063e-08,
                                      0.000000000000000e+00, -5.960464477539063e-08,
                                      -0.000000000000000e+00, -5.960464477539063e-08,
                                      -0.000000000000000e+00, 0.000000000000000e+00,
                                      5.960464477539063e-08, -5.960464477539063e-08,
                                      0.000000000000000e+00, -5.960464477539063e-08,
                                      5.960464477539063e-08, 0.000000000000000e+00,
                                      -0.000000000000000e+00, -0.000000000000000e+00]]]])
        expect_dk_diff = np.array([[[[0.000000000000000e+00, -5.960464477539063e-08,
                                      0.000000000000000e+00, 5.960464477539063e-08,
                                      5.960464477539063e-08, 0.000000000000000e+00,
                                      0.000000000000000e+00, 5.960464477539063e-08,
                                      -0.000000000000000e+00, -0.000000000000000e+00,
                                      -0.000000000000000e+00, -0.000000000000000e+00,
                                      -0.000000000000000e+00, 0.000000000000000e+00,
                                      -0.000000000000000e+00, 5.960464477539063e-08]]]])
        expect_dv_diff = np.array([[[[1.9295937499996807e-05, -1.6090374999999657e-04,
                                      5.9442500000006504e-05, -7.6876250000024182e-05,
                                      -4.1035625000018783e-05, 1.1551124999997775e-04,
                                      7.4064062499934469e-06, 1.7152875000003842e-04,
                                      -1.7012500000002095e-04, -2.1527124999998204e-04,
                                      -1.9566500000001152e-04, -9.6663125000007177e-05,
                                      -1.2490750000004880e-04, 1.2684999999956759e-05,
                                      1.6678874999997984e-04, 2.8191249999975909e-05]]]])
    elif dtype == mstype.bfloat16:
        y_diff = actual_output.astype(mstype.float64).asnumpy() - golden_y
        dq_diff = actual_grad[0].astype(mstype.float64).asnumpy() - dq
        dk_diff = actual_grad[1].astype(mstype.float64).asnumpy() - dk
        dv_diff = actual_grad[2].astype(mstype.float64).asnumpy() - dv

        expect_y_diff = np.array([[[[-2.5770000000002735e-04, 1.5310200000000496e-03,
                                     -4.7841000000004019e-04, -3.9805999999997788e-04,
                                     -7.4730000000000629e-04, -5.3698000000002022e-04,
                                     3.9396000000002651e-04, -3.6634750000000271e-04,
                                     3.3571562500001040e-05, -8.7165000000000159e-04,
                                     1.0510500000000117e-04, 7.6381250000000234e-05,
                                     -1.5718200000000015e-03, -2.3475000000000579e-04,
                                     -4.1256999999994548e-04, -8.3714999999984219e-05]]]])
        expect_dq_diff = np.array([[[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]])
        expect_dk_diff = np.array([[[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]])
        expect_dv_diff = np.array([[[[-3.4691500000000319e-04, 1.3039400000000034e-03,
                                      -9.1711999999999350e-04, 4.1140499999997582e-04,
                                      2.0310499999998122e-04, -3.7277000000002225e-04,
                                      -5.3628750000006553e-05, 6.5981000000003842e-04,
                                      1.7829999999999790e-03, 2.7301000000001796e-04,
                                      1.7574599999999885e-03, -8.2908500000000718e-04,
                                      -1.1014700000000488e-03, -1.9404400000000432e-03,
                                      6.5506999999997984e-04, -4.6009000000002409e-04]]]])
    else:
        raise ValueError(f"dtype is invalid.")

    np.testing.assert_allclose(y_diff, expect_y_diff, rtol=1e-4)
    np.testing.assert_allclose(dq_diff, expect_dq_diff, rtol=1e-4)
    np.testing.assert_allclose(dk_diff, expect_dk_diff, rtol=1e-4)
    np.testing.assert_allclose(dv_diff, expect_dv_diff, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('input_layout', ["BSH", "BNSD", "SBH", "BSND", "TND"])
def test_ops_flash_attention_score_dynamic(input_layout):
    """
    Feature: Pyboost function.
    Description: Test function flash attention score dynamic.
    Expectation: Correct result.
    """
    dtype = mstype.float16
    B1 = 1
    N1 = 4
    S1 = 8
    D1 = 128
    head_num1 = N1
    query1, key1, value1, _, _, _ = generate_inputs(B1, N1, N1, S1, S1, D1, input_layout, dtype)
    actual_seq_qlen1 = (2, 8)
    actual_seq_kvlen1 = (2, 8)
    B2 = 4
    N2 = 8
    S2 = 2
    D2 = 128
    head_num2 = N2
    query2, key2, value2, _, _, _ = generate_inputs(B2, N2, N2, S2, S2, D2, input_layout, dtype)
    actual_seq_qlen2 = (2, 8)
    actual_seq_kvlen2 = (2, 8)
    TEST_OP(flash_attention_score_func, \
            [[query1, key1, value1, head_num1, actual_seq_qlen1, actual_seq_kvlen1, input_layout], \
             [query2, key2, value2, head_num2, actual_seq_qlen2, actual_seq_kvlen2, input_layout]], \
             '', disable_input_check=True, disable_yaml_check=True, disable_mode=['GRAPH_MODE'], ignore_output_index=2)


def generate_unpad_full_attn_mask(batch, seq_len, actual_seq_qlen, actual_seq_kvlen):
    attn_mask = np.ones([batch, 1, seq_len, seq_len], np.uint8)
    pre_query_index, pre_key_index = 0, 0
    for cur_query_index, cur_key_index in zip(actual_seq_qlen, actual_seq_kvlen):
        sub_query_len = cur_query_index - pre_query_index
        sub_key_len = cur_key_index - pre_key_index
        if sub_query_len > sub_key_len:
            raise ValueError(f"For each sample, the length of query must be greater than key, "
                             f"but got {sub_query_len} and {sub_key_len}")
        diff_len = sub_key_len - sub_query_len
        attn_mask[:, :, pre_query_index: cur_query_index, pre_key_index: pre_key_index + diff_len] = 0
        attn_mask[:, :, pre_query_index: cur_query_index, pre_key_index + diff_len: cur_key_index] = \
            np.triu(np.ones((sub_query_len, sub_query_len), np.uint8), 1)
        pre_query_index = cur_query_index
        pre_key_index = cur_key_index
    return Tensor(attn_mask)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [mstype.float16, mstype.bfloat16])
def test_ops_flash_attention_score_tnd(mode, dtype):
    """
    Feature: Test the precision for TND.
    Description: Test function flash attention score forward and backward.
    Expectation: The result of TND and BSH is equal.
    """
    B, N, S, D = 1, 8, 1024, 128
    T = B * S
    H = N * D
    sample_num = 4
    query, key, value, _, _, _ = generate_inputs(B, N, N, S, S, D, "BSH", dtype, return_tensor=False)
    actual_seq_qlen = tuple(range(S // sample_num, S + 1, S // sample_num))
    actual_seq_kvlen = tuple(range(S // sample_num, S + 1, S // sample_num))
    full_attn_mask_tensor = generate_unpad_full_attn_mask(B, S, actual_seq_qlen, actual_seq_kvlen)
    context.set_context(jit_level='O0')
    context.set_context(mode=mode)
    input_layout = "BSH"
    head_num = N
    padding_mask = None
    scale_value = 1.0
    keep_prob = 1.0
    pre_tokens = 65536
    next_tokens = 0
    inner_precise = 0
    sparse_mode = 0

    dx = np.random.uniform(-1, 1, [B, S, N * D])
    real_shift = None
    prefix = None
    drop_mask = None

    bsh_ms_query = Tensor(query, dtype=dtype)
    bsh_ms_key = Tensor(key, dtype=dtype)
    bsh_ms_value = Tensor(value, dtype=dtype)
    bsh_ms_dx = Tensor(dx, dtype=dtype)

    bsh_output_tensor = flash_attention_score_forward_func(bsh_ms_query, bsh_ms_key, bsh_ms_value, head_num, real_shift,
                                                           drop_mask, padding_mask, full_attn_mask_tensor, prefix, None,
                                                           None, keep_prob, input_layout, pre_tokens, next_tokens,
                                                           scale_value, inner_precise, sparse_mode)

    flash_attention_score_backward = Grad(flash_attention_score_forward_func)
    bsh_grad_out = flash_attention_score_backward(bsh_ms_query, bsh_ms_key, bsh_ms_value, head_num, real_shift,
                                                  drop_mask, padding_mask, full_attn_mask_tensor, prefix, None, None,
                                                  keep_prob, input_layout, pre_tokens, next_tokens, scale_value,
                                                  inner_precise, sparse_mode, bsh_ms_dx)
    bsh_dq_tensor, bsh_dk_tensor, bsh_dv_tensor = bsh_grad_out[0], bsh_grad_out[1], bsh_grad_out[2]

    T = B * S
    tnd_ms_query = Tensor(query.reshape((T, N, D)), dtype=dtype)
    tnd_ms_key = Tensor(key.reshape((T, N, D)), dtype=dtype)
    tnd_ms_value = Tensor(value.reshape((T, N, D)), dtype=dtype)
    tnd_ms_dx = Tensor(dx.reshape((T, N, D)), dtype=dtype)
    attn_mask = Tensor(np.triu(np.ones((2048, 2048), np.uint8), 1))
    tnd_output_tensor = flash_attention_score_forward_func(tnd_ms_query, tnd_ms_key, tnd_ms_value, head_num, real_shift,
                                                           drop_mask, padding_mask, attn_mask, prefix, actual_seq_qlen,
                                                           actual_seq_kvlen, keep_prob, "TND", pre_tokens, next_tokens,
                                                           scale_value, inner_precise, 3)

    flash_attention_score_backward = Grad(flash_attention_score_forward_func)
    tnd_grad_out = flash_attention_score_backward(tnd_ms_query, tnd_ms_key, tnd_ms_value, head_num, real_shift,
                                                  drop_mask, padding_mask, attn_mask, prefix, actual_seq_qlen,
                                                  actual_seq_kvlen, keep_prob, "TND", pre_tokens, next_tokens,
                                                  scale_value, inner_precise, 3, tnd_ms_dx)
    tnd_dq_tensor, tnd_dk_tensor, tnd_dv_tensor = tnd_grad_out[0], tnd_grad_out[1], tnd_grad_out[2]

    bsh_output, bsh_dq, bsh_dk, bsh_dv = \
        bsh_output_tensor.astype(mstype.float32).asnumpy(), \
        bsh_dq_tensor.astype(mstype.float32).asnumpy(), \
        bsh_dk_tensor.astype(mstype.float32).asnumpy(), \
        bsh_dv_tensor.astype(mstype.float32).asnumpy()
    tnd_output, tnd_dq, tnd_dk, tnd_dv = \
        tnd_output_tensor.astype(mstype.float32).asnumpy(), \
        tnd_dq_tensor.astype(mstype.float32).asnumpy(), \
        tnd_dk_tensor.astype(mstype.float32).asnumpy(), \
        tnd_dv_tensor.astype(mstype.float32).asnumpy()

    rtol, atol = 1e-2, 1e-2
    assert np.allclose(bsh_output, tnd_output.reshape((B, S, H)), rtol, atol)
    assert np.allclose(bsh_dq, tnd_dq.reshape((B, S, H)), rtol, atol)
    assert np.allclose(bsh_dk, tnd_dk.reshape((B, S, H)), rtol, atol)
    assert np.allclose(bsh_dv, tnd_dv.reshape((B, S, H)), rtol, atol)
