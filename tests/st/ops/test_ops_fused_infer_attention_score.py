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
import math
import pytest
import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor, context
from mindspore.common.dtype import _dtype_to_nptype
from mindspore.ops import fused_infer_attention_score
from mindspore.ops.function.nn_func import prompt_flash_attention
from mindspore.ops.function.nn_func import incre_flash_attention
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def pfa_forward_func(query, key, value, num_heads, input_layout, inner_precise=0, **kwargs):
    return prompt_flash_attention(query, key, value, num_heads=num_heads, input_layout=input_layout,
                                  inner_precise=inner_precise, **kwargs)


@test_utils.run_with_cell
def ifa_forward_func(query, key, value, num_heads, input_layout, inner_precise=0, **kwargs):
    return incre_flash_attention(query, [key], [value], num_heads=num_heads, input_layout=input_layout,
                                 inner_precise=inner_precise, **kwargs)


@test_utils.run_with_cell
def fias_forward_func(query, key, value, num_heads, input_layout, inner_precise=0, **kwargs):
    return fused_infer_attention_score(query, [key], [value], num_heads=num_heads, input_layout=input_layout,
                                       inner_precise=inner_precise, **kwargs)


def generate_inputs(B, N1, N2, S1, S2, D, input_layout, dtype, return_tensor=True):
    np_dtype = _dtype_to_nptype(dtype)  # pylint:disable=protected-access
    if input_layout == "BSH":
        query = np.random.rand(B, S1, N1 * D).astype(np_dtype)
        key = np.random.rand(B, S2, N2 * D).astype(np_dtype)
        value = np.random.rand(B, S2, N2 * D).astype(np_dtype)
    elif input_layout == "BNSD" or input_layout == "BNSD_BSND":
        query = np.random.rand(B, N1, S1, D).astype(np_dtype)
        key = np.random.rand(B, N2, S2, D).astype(np_dtype)
        value = np.random.rand(B, N2, S2, D).astype(np_dtype)
    elif input_layout == "BSND":
        query = np.random.rand(B, S1, N1, D).astype(np_dtype)
        key = np.random.rand(B, S2, N2, D).astype(np_dtype)
        value = np.random.rand(B, S2, N2, D).astype(np_dtype)
    else:
        raise ValueError(f"input_layout {input_layout} is invalid.")
    if return_tensor:
        return Tensor(query, dtype=dtype), Tensor(key, dtype=dtype), Tensor(value, dtype=dtype)
    return query, key, value


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=48.0)
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
@test_utils.run_test_with_On
def test_ops_fused_infer_attention_score_in_pfa_branch(mode, input_layout):
    """
    Feature: Pyboost function.
    Description: Test function fused infer attention score forward, in prompt flash attention branch.
    Expectation: Correct result.
    """
    context.set_context(mode=mode)
    dtype = mstype.float16
    np.random.seed(968941859)
    B1, N1, N2, S1, D1 = 1, 10, 5, 1024, 32
    scale_value = 1 / math.sqrt(D1)
    q, k, v = generate_inputs(B1, N1, N2, S1, S1, D1, input_layout, dtype)
    expect_out = pfa_forward_func(q, k, v, num_heads=N1, input_layout=input_layout, num_key_value_heads=N2,
                                  scale_value=scale_value)
    actual_out, _ = fias_forward_func(q, k, v, num_heads=N1, input_layout=input_layout, num_key_value_heads=N2,
                                      scale=scale_value)
    np.testing.assert_allclose(actual_out.asnumpy(), expect_out.asnumpy(), rtol=5e-3, atol=5e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=30.0)
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
@test_utils.run_test_with_On
def test_ops_fused_infer_attention_score_in_ifa_branch(mode, input_layout):
    """
    Feature: Pyboost function.
    Description: Test function fused infer attention score forward, in incre flash attention branch.
    Expectation: Correct result.
    """
    context.set_context(mode=mode)
    dtype = mstype.float16
    np.random.seed(968941859)
    B1, N1, N2, S1, S2, D1 = 1, 5, 5, 1, 4096, 128  # ifa needs S1 dimension to be set as 1
    scale_value = 1 / math.sqrt(D1)
    q, k, v = generate_inputs(B1, N1, N2, S1, S2, D1, input_layout, dtype)
    expect_out = ifa_forward_func(q, k, v, num_heads=N1, input_layout=input_layout, num_key_value_heads=N2,
                                  scale_value=scale_value)
    actual_out, _ = fias_forward_func(q, k, v, num_heads=N1, input_layout=input_layout, num_key_value_heads=N2,
                                      scale=scale_value)
    np.testing.assert_allclose(actual_out.asnumpy(), expect_out.asnumpy(), rtol=5e-3, atol=5e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=22.0)
@pytest.mark.parametrize('input_layout', ["BSH", "BNSD", "BSND", "BNSD_BSND"])
def test_ops_fused_infer_attention_score_dynamic(input_layout):
    """
    Feature: Pyboost function.
    Description: Test function fused infer attention score dynamic.
    Expectation: Correct result.
    """
    dtype = mstype.float16
    np.random.seed(968941859)
    B1, N1, S1, D1 = 1, 4, 8, 128
    head_num1 = N1
    query1, key1, value1 = generate_inputs(B1, N1, N1, S1, S1, D1, input_layout, dtype)
    B2, N2, S2, D2 = 4, 8, 2, 128
    head_num2 = N2
    query2, key2, value2 = generate_inputs(B2, N2, N2, S2, S2, D2, input_layout, dtype)
    TEST_OP(fias_forward_func,
            [[query1, key1, value1, head_num1, input_layout],
             [query2, key2, value2, head_num2, input_layout]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_grad': True,
                         'ignore_output_index': 1})
