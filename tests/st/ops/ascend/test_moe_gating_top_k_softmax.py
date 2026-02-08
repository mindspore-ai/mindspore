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
import numpy as np
import pytest
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor, jit
from mindspore.ops.auto_generate import MoeGatingTopKSoftmax

def softmax_func(x, axis=None):
    is_fp16 = x.dtype == np.float16
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y / x_sum
    if is_fp16:
        ans = ans.astype(np.float16)
        x_max = x_max.astype(np.float16)
        x_sum = x_sum.astype(np.float16)
    return ans, x_max, x_sum

def generate_expect_output(x, finished_optional, k):
    num_expert = x.shape[-1]
    softmax, _, _, = softmax_func(x, -1)
    expert_idx = np.argsort(-softmax, axis=-1, kind='stable')
    expert_idx = expert_idx[:, :k]
    y = np.take_along_axis(softmax, expert_idx, axis=-1)
    if finished_optional is not None:
        finished_optional = finished_optional.reshape(finished_optional.shape[0], 1)
        finished_optional = np.tile(finished_optional, (1, k))
        expert_idx = np.where(finished_optional, num_expert, expert_idx)
    row_idx = np.arange(y.shape[0] * y.shape[1]).reshape(y.shape[1], y.shape[0]).transpose(1, 0)
    if x.dtype == np.float16:
        y = y.astype(np.float16)
    return y, expert_idx.astype(np.int32), row_idx.astype(np.int32)

@test_utils.run_with_cell
def moe_gating_topk_softmax_forward_func(x, finished, k):
    net = MoeGatingTopKSoftmax()
    return net(x, finished, k)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
@pytest.mark.parametrize('support_type', [mstype.float32, mstype.float16, mstype.bfloat16])
def test_moe_gating_top_k_softmax_case0(mode, support_type):
    """
    Feature: Test the MoeGatingTopKSoftmax calculate
    Description: Test the MoeGatingTopKSoftmax ops in Ascend backend
    Expectation: The result match to the expect value.
    """
    n = 10
    k = 2
    col = 200
    x = Tensor(np.random.uniform(-1, 1, size=(n, col)).astype(np.float32), dtype=support_type)
    finished = Tensor(np.random.uniform(-1, 1, size=(n,)).astype(bool))

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        y_ms, expert_idx_ms, row_idx_ms = moe_gating_topk_softmax_forward_func(x, finished, k)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE)
        y_ms, expert_idx_ms, row_idx_ms = (jit(moe_gating_topk_softmax_forward_func, jit_level="O0"))(x, finished, k)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        y_ms, expert_idx_ms, row_idx_ms = (jit(moe_gating_topk_softmax_forward_func, backend="GE"))(x, finished, k)

    if support_type == mstype.bfloat16:
        y, expert_idx, row_idx = \
            generate_expect_output(x.float().asnumpy(), finished.asnumpy(), k)
        np.testing.assert_allclose(y, y_ms.float().asnumpy(), rtol=5e-3, atol=5e-3)
    else:
        y, expert_idx, row_idx = generate_expect_output(x.asnumpy(), finished.asnumpy(), k)
        np.testing.assert_allclose(y, y_ms.asnumpy(), rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(expert_idx, expert_idx_ms.asnumpy())
    np.testing.assert_allclose(row_idx, row_idx_ms.asnumpy())

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=14.0)
def test_moe_gating_top_k_softmax_dynamic():
    """
    Feature: Test the MoeGatingTopKSoftmax calculate with dynamic shape.
    Description: Test the MoeGatingTopKSoftmax ops in Ascend backend
    Expectation: The result match to the expect value.
    """
    x1 = np.random.uniform(-1, 1, size=(430, 200)).astype(np.float16)
    finished1 = np.random.uniform(-1, 1, size=(430,)).astype(bool)
    k1 = 4

    x2 = np.random.uniform(-1, 1, size=(2, 520, 5120)).astype(np.float16)
    finished2 = np.random.uniform(-1, 1, size=(2, 520,)).astype(bool)
    k2 = 5

    TEST_OP(moe_gating_topk_softmax_forward_func,
            [[Tensor(x1), Tensor(finished1), k1], [Tensor(x2), Tensor(finished2), k2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_grad': True})
