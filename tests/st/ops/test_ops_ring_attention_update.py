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
import pytest
import numpy as np
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark

import mindspore as ms
from mindspore import Tensor, context
from mindspore import ops

def gen_inputs(layout, S, B, H, T, N, D, dtype=np.float32):
    if layout == 'SBH':
        prev_attn_out = np.random.uniform(-1.0, 1.0, size=(S, B, H)).astype(dtype)
        prev_softmax_max = np.random.uniform(-1.0, 1.0, size=(B, N, S, 8)).astype(np.float32)
        prev_softmax_sum = np.random.uniform(-1.0, 1.0, size=(B, N, S, 8)).astype(np.float32)
        cur_attn_out = np.random.uniform(-1.0, 1.0, size=(S, B, H)).astype(dtype)
        cur_softmax_max = np.random.uniform(-1.0, 1.0, size=(B, N, S, 8)).astype(np.float32)
        cur_softmax_sum = np.random.uniform(-1.0, 1.0, size=(B, N, S, 8)).astype(np.float32)
    elif layout == 'TND':
        prev_attn_out = np.random.uniform(-1.0, 1.0, size=(T, N, D)).astype(dtype)
        prev_softmax_max = np.random.uniform(-1.0, 1.0, size=(T, N, 8)).astype(np.float32)
        prev_softmax_sum = np.random.uniform(-1.0, 1.0, size=(T, N, 8)).astype(np.float32)
        cur_attn_out = np.random.uniform(-1.0, 1.0, size=(T, N, D)).astype(dtype)
        cur_softmax_max = np.random.uniform(-1.0, 1.0, size=(T, N, 8)).astype(np.float32)
        cur_softmax_sum = np.random.uniform(-1.0, 1.0, size=(T, N, 8)).astype(np.float32)
    else:
        raise ValueError(f'For RingAttentionUpdate, layout must be SBH/TND.')

    return prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out, cur_softmax_max, cur_softmax_sum


def get_expected_res(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                     cur_attn_out, cur_softmax_max, cur_softmax_sum,
                     actual_seq_qlen=None, layout='SBH'):
    # Update softmax_max
    softmax_max = np.maximum(prev_softmax_max, cur_softmax_max)
    prev_scale = np.exp(prev_softmax_max - softmax_max)
    cur_scale = np.exp(cur_softmax_max - softmax_max)

    # Update softmax_sum
    prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
    cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
    softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled

    # Compute output scaling factors
    prev_out_scale = prev_softmax_sum_scaled / softmax_sum
    cur_out_scale = cur_softmax_sum_scaled / softmax_sum

    # Handle different layouts
    if layout == 'SBH':
        n = prev_out_scale.shape[1]
        h = prev_attn_out.shape[-1]
        d = h // n

        prev_out_scale = np.repeat(prev_out_scale[..., 0, np.newaxis], d, axis=3)
        prev_out_scale = prev_out_scale.transpose(2, 0, 1, 3).reshape(prev_attn_out.shape)

        cur_out_scale = np.repeat(cur_out_scale[..., 0, np.newaxis], d, axis=3)
        cur_out_scale = cur_out_scale.transpose(2, 0, 1, 3).reshape(cur_attn_out.shape)

    elif layout == 'TND':
        n = prev_out_scale.shape[1]
        h = prev_attn_out.shape[-1]
        prev_out_scale = np.repeat(prev_out_scale[..., 0, np.newaxis], h, axis=2)
        prev_out_scale = prev_out_scale.reshape(prev_attn_out.shape)
        cur_out_scale = np.repeat(cur_out_scale[..., 0, np.newaxis], h, axis=2)
        cur_out_scale = cur_out_scale.reshape(cur_attn_out.shape)

    else:
        raise ValueError(f"RingAttentionUpdate unsupported layout: {layout}")

    # Update output
    attn_out = prev_attn_out * prev_out_scale + cur_attn_out * cur_out_scale
    return attn_out.astype(prev_attn_out.dtype), softmax_max, softmax_sum



@test_utils.run_with_cell
def ring_attention_update_forward_func(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                                       cur_attn_out, cur_softmax_max, cur_softmax_sum,
                                       actual_seq_qlen=None, layout='SBH'):
    return ops.ring_attention_update(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                                     cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen, layout)


def set_mode(mode):
    """
    set mode
    """
    if mode == "KBK":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential", mem_peak=20.01
)
@pytest.mark.parametrize("mode", ["KBK", "PYBOOST"])
def test_ring_attention_update(mode):
    """
    Feature: ring_attention_update
    Description: test ops.ring_attention_update
    Expectation: expect correct result.
    """
    set_mode(mode)
    S, B, H, T, N, D = 4, 6, 16, 5, 8, 7
    input_np = gen_inputs('SBH', S, B, H, T, N, D)
    input_ms = [Tensor(item) for item in input_np]
    ac = Tensor([1, 2, 3, 4, 5, 6], ms.int64)
    attn_out_ms, softmax_max_ms, softmax_sum_ms = ring_attention_update_forward_func(*input_ms, ac, 'SBH')
    attn_out_expect, softmax_max_expect, softmax_sum_expect = get_expected_res(*input_np)

    np.allclose(attn_out_ms.asnumpy(), attn_out_expect)
    np.allclose(softmax_max_ms.asnumpy(), softmax_max_expect)
    np.allclose(softmax_sum_ms.asnumpy(), softmax_sum_expect)

    S, B, H, T, N, D = 4, 6, 16, 24, 8, 64
    input_np = gen_inputs('TND', S, B, H, T, N, D)
    input_ms_1 = [Tensor(item) for item in input_np]
    ac = Tensor([0, 2, 3, 4, 5, 6, 24], ms.int64)
    attn_out_ms_1, softmax_max_ms_1, softmax_sum_ms_1 = ring_attention_update_forward_func(*input_ms_1, ac, 'TND')
    attn_out_expect, softmax_max_expect, softmax_sum_expect = get_expected_res(*input_np, layout="TND")

    np.allclose(attn_out_ms_1.asnumpy(), attn_out_expect)
    np.allclose(softmax_max_ms_1.asnumpy(), softmax_max_expect)
    np.allclose(softmax_sum_ms_1.asnumpy(), softmax_sum_expect)


def ring_attention_update_func(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                               cur_attn_out, cur_softmax_max, cur_softmax_sum,
                               actual_seq_qlen=None, layout="SBH"):
    return ops.ring_attention_update(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                                     cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen, layout)

@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=60.01
)
def test_permute_dynamic():
    """
    Feature: Ops
    Description: test op ring_attention_update dynamic shape
    Expectation: expect correct result.
    """
    S, B, H, T, N, D = 4, 6, 16, 5, 8, 7
    input_np = gen_inputs('SBH', S, B, H, T, N, D)
    input_ms_1 = [Tensor(item) for item in input_np]

    S, B, H, T, N, D = 64, 64, 512, 24, 8, 64
    input_np = gen_inputs('SBH', S, B, H, T, N, D)
    input_ms_2 = [Tensor(item) for item in input_np]

    TEST_OP(
        ring_attention_update_func,
        [input_ms_1, input_ms_2],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['EmptyTensor', 'ScalarTensor'],
        case_config={'disable_input_check': True,
                     'disable_grad': True}
    )
