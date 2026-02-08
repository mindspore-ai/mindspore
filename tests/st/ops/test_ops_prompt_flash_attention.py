import pytest
import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops.function.nn_func import prompt_flash_attention
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def prompt_flash_attention_forward_func(query, key, value, attn_mask, actual_seq_lengths,
                                        actual_seq_lengths_kv, pse_shift,
                                        deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, num_heads,
                                        scale_value=1.0, pre_tokens=2147483647, next_tokens=0, input_layout='BSH',
                                        num_key_value_heads=0, sparse_mode=0, inner_precise=1):
    return prompt_flash_attention(query, key, value, attn_mask, actual_seq_lengths, actual_seq_lengths_kv, pse_shift,
                                  deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, num_heads,
                                  scale_value, pre_tokens, next_tokens, input_layout,
                                  num_key_value_heads, sparse_mode, inner_precise)


@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [mstype.float16, mstype.bfloat16])
@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
def test_ops_prompt_flash_attention_normal(mode, dtype, input_layout):
    """
    Feature: Pyboost function.
    Description: Test function flash attention score forward and backward.
    Expectation: Correct result.
    """
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode, device_target="Ascend")
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
    if input_layout == "BSH":
        query = np.squeeze(query, axis=0)
        key = np.squeeze(key, axis=0)
        value = np.squeeze(value, axis=0)
    ms_query = Tensor(query, dtype=dtype)
    ms_key = Tensor(key, dtype=dtype)
    ms_value = Tensor(value, dtype=dtype)
    attn_mask = None
    actual_seq_lengths = None
    actual_seq_lengths_kv = None
    num_heads = 1
    scale_value = 1.0
    pre_tokens = 2147483647
    next_tokens = 0
    input_layout = input_layout
    num_key_value_heads = 1
    sparse_mode = 0
    inner_precise = 1
    actual_output = prompt_flash_attention_forward_func(ms_query, ms_key, ms_value, attn_mask,
                                                        actual_seq_lengths, actual_seq_lengths_kv, None,
                                                        None, None, None, None, None, num_heads,
                                                        scale_value, pre_tokens, next_tokens, input_layout,
                                                        num_key_value_heads, sparse_mode, inner_precise)
    rtol = atol = 1e-3
    if dtype == mstype.bfloat16:
        rtol = atol = 4e-3
        actual_output = actual_output.float()

    expect_output = np.array([[[[-0.4255, 0.643, -0.601, 0.4536, -0.3665, -0.4214, 0.269,\
                            0.2318, 0.02402, -0.589, -0.05283, 0.03165, 0.9585, -0.3044, -0.8354, 0.3809]]]])
    if input_layout == "BSH":
        expect_output = np.squeeze(expect_output, axis=0)
    np.testing.assert_allclose(actual_output.asnumpy(), expect_output, rtol=rtol, atol=atol)


def generate_inputs(b, n, kv_n, q_s, kv_s, q_h, kv_h, d, input_layout, dtype, return_tensor=True):
    min_value = -1
    max_value = 1
    if input_layout == "BSH":
        q_shape = [b, q_s, q_h]
        kv_shape = [b, kv_s, kv_h]
    elif input_layout == "BNSD":
        q_shape = [b, n, q_s, d]
        kv_shape = [b, kv_n, kv_s, d]
    else:
        raise ValueError(f"input_layout is invalid.")
    query = np.random.uniform(min_value, max_value, q_shape)
    key = np.random.uniform(min_value, max_value, kv_shape)
    value = np.random.uniform(min_value, max_value, kv_shape)
    if return_tensor:
        return Tensor(query, dtype=dtype), Tensor(key, dtype=dtype), Tensor(value, dtype=dtype)
    return query, key, value



def prompt_flash_attention_func(query, key, value, num_heads, num_key_value_heads, actual_seq_lengths=None,
                                actual_seq_lengths_kv=None, input_layout='BSH', pse_shift=None,
                                attn_mask=None, deq_scale1=None, quant_scale1=None, deq_scale2=None,
                                quant_scale2=None, quant_offset2=None, scale_value=1.0,
                                pre_tokens=2147483647, next_tokens=0, sparse_mode=0, inner_precise=1):
    return prompt_flash_attention(query, key, value, attn_mask, actual_seq_lengths, actual_seq_lengths_kv, pse_shift,
                                  deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, num_heads,
                                  scale_value, pre_tokens, next_tokens, input_layout,
                                  num_key_value_heads, sparse_mode, inner_precise)


@arg_mark(plat_marks=['platform_ascend910b'],
        level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=20.0)
@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
def test_ops_prompt_flash_attention_dynamic(input_layout):
    """
    Feature: Pyboost function.
    Description: Test function prmopt_flash_attention dynamic.
    Expectation: Correct result.
    """
    dtype = mstype.float16
    b = 1
    n, kv_n, d = (5, 5, 16)
    q_s, kv_s = (1, 1)
    q_h = n * d
    num_key_value_heads = kv_n
    kv_h = q_h if num_key_value_heads == 0 else num_key_value_heads * d
    head_num1 = n
    query1, key1, value1 = generate_inputs(b, n, kv_n, q_s, kv_s, q_h, kv_h, d, input_layout, dtype)
    actual_seq_qlen1 = (q_s,)
    actual_seq_kvlen1 = (kv_s,)

    b2 = 1
    n2, kv_n2, d2 = (40, 40, 128)
    q_s2, kv_s2 = (4, 4)
    q_h2 = n2 * d2
    num_key_value_heads2 = kv_n2
    kv_h2 = q_h2 if num_key_value_heads2 == 0 else num_key_value_heads2 * d2
    head_num2 = n2
    query2, key2, value2 = generate_inputs(b2, n2, kv_n2, q_s2, kv_s2, q_h2, kv_h2, d2, input_layout, dtype)
    actual_seq_qlen2 = (q_s2,)
    actual_seq_kvlen2 = (kv_s2,)

    TEST_OP(prompt_flash_attention_func,
            [[query1, key1, value1, head_num1, num_key_value_heads, actual_seq_qlen1,
              actual_seq_kvlen1, input_layout],
             [query2, key2, value2, head_num2, num_key_value_heads2, actual_seq_qlen2,
              actual_seq_kvlen2, input_layout]],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_grad': True})


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=26.0)
@pytest.mark.parametrize('input_layout', ["BSH"])
def test_ops_prompt_flash_attention_dynamic_all_params(input_layout):
    """
    Feature: Pyboost function.
    Description: Test function prmopt_flash_attention dynamic.
    Expectation: Correct result.
    """
    dtype = mstype.int8
    b = 1
    n, kv_n, d = (5, 5, 16)
    q_s, kv_s = (1, 1)
    q_h = n * d
    num_key_value_heads = kv_n
    kv_h = q_h if num_key_value_heads == 0 else num_key_value_heads * d
    head_num1 = n
    query1, key1, value1 = generate_inputs(b, n, kv_n, q_s, kv_s, q_h, kv_h, d, input_layout, dtype)
    actual_seq_qlen1 = (q_s,)
    actual_seq_kvlen1 = (kv_s,)

    pse_shift1 = Tensor(np.random.randn(b, n, q_s, kv_s), ms.float16)
    attn_mask1 = Tensor(np.random.randn(q_s, kv_s), ms.bool_)
    deq_scale11 = Tensor(np.array([3]), ms.uint64)
    quant_scale11 = Tensor(np.array([1.7]), ms.float32)
    deq_scale21 = Tensor(np.array([5]), ms.uint64)
    quant_scale21 = Tensor(np.array([1.7]), ms.float32)
    quant_offset21 = Tensor(np.array([1.7]), ms.float32)
    scale_value1 = 1.0
    pre_tokens1 = 2147483647
    next_tokens1 = 0
    sparse_mode1 = 0
    inner_precise1 = 1

    b2 = 2
    n2, kv_n2, d2 = (40, 40, 128)
    q_s2, kv_s2 = (4, 4)
    q_h2 = n2 * d2
    num_key_value_heads2 = kv_n2
    kv_h2 = q_h2 if num_key_value_heads2 == 0 else num_key_value_heads2 * d2
    head_num2 = n2
    query2, key2, value2 = generate_inputs(b2, n2, kv_n2, q_s2, kv_s2, q_h2, kv_h2, d2, input_layout, dtype)
    actual_seq_qlen2 = (q_s2, 1)
    actual_seq_kvlen2 = (kv_s2, 1)

    pse_shift2 = Tensor(np.random.randn(b2, n2, q_s2, kv_s2), ms.float16)
    attn_mask2 = Tensor(np.random.randn(2048, 2048), ms.bool_)
    deq_scale12 = Tensor(np.array([4]), ms.uint64)
    quant_scale12 = Tensor(np.array([1.778]), ms.float32)
    deq_scale22 = Tensor(np.array([7]), ms.uint64)
    quant_scale22 = Tensor(np.array([1.7123]), ms.float32)
    quant_offset22 = Tensor(np.array([1.714]), ms.float32)
    scale_value2 = 1.012
    pre_tokens2 = 214748354
    next_tokens2 = 12127
    sparse_mode2 = 2
    inner_precise2 = 0

    TEST_OP(prompt_flash_attention_func,
            [[query1, key1, value1, head_num1, num_key_value_heads, actual_seq_qlen1,
              actual_seq_kvlen1, input_layout, pse_shift1, attn_mask1, deq_scale11, quant_scale11, deq_scale21,
              quant_scale21, quant_offset21, scale_value1, pre_tokens1, next_tokens1, sparse_mode1, inner_precise1],
             [query2, key2, value2, head_num2, num_key_value_heads2, actual_seq_qlen2,
              actual_seq_kvlen2, input_layout, pse_shift2, attn_mask2, deq_scale12, quant_scale12, deq_scale22,
              quant_scale22, quant_offset22, scale_value2, pre_tokens2, next_tokens2, sparse_mode2, inner_precise2]],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_grad': True})
