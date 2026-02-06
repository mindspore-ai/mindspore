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
from mindspore.ops.auto_generate import cross_entropy_loss_op as cross_entropy_loss


def cross_entropy_loss_expect(input, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    N, C = input.shape
    log_probs = input - np.log(np.sum(np.exp(input), axis=1, keepdims=True))
    valid_mask = target != ignore_index
    loss = -log_probs[np.arange(N), target]

    if label_smoothing > 0.0:
        smooth_loss = -np.sum(log_probs, axis=1) / C
        loss = (1 - label_smoothing) * loss + label_smoothing * smooth_loss
    if weight is not None:
        loss *= weight[target]
    loss = loss[valid_mask]
    if reduction == 'sum':
        return np.sum(loss)
    elif reduction == 'mean':
        weight_sum = np.sum(weight[target][valid_mask]) if weight is not None else valid_mask.sum()
        return np.sum(loss) / weight_sum
    else:
        output = np.zeros(N)
        output[valid_mask] = loss
        return output


def cross_entropy_loss_grad_expect(input, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    N, C = input.shape
    exp_logits = np.exp(input - np.max(input, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    target_one_hot = np.zeros_like(probs)
    valid_mask = target != ignore_index
    target_one_hot[np.arange(N), target] = 1.0
    
    if label_smoothing > 0.0:
        target_one_hot = target_one_hot * (1 - label_smoothing) + label_smoothing / C
    grad = (probs - target_one_hot) * valid_mask[:, None] 
    if weight is not None:
        grad *= weight[target].reshape(N, 1)
    
    if reduction == 'sum':
        return grad
    elif reduction == 'mean':
        valid_count = np.sum(valid_mask)
        return grad / valid_count
    else:
        return grad


@test_utils.run_with_cell
def cross_entropy_loss_forward_func(logits, target, weight=None, reduction='mean', ignore_index=-100,
                                    label_smoothing=0.0):
    return cross_entropy_loss(logits, target, weight, reduction, ignore_index, label_smoothing)


@test_utils.run_with_cell
def cross_entropy_loss_backward_func(logits, target, weight=None, reduction='mean', ignore_index=-100,
                                     label_smoothing=0.0):
    return ms.grad(cross_entropy_loss_forward_func, (0))(
        logits, target, weight, reduction, ignore_index, label_smoothing
    )


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
@pytest.mark.parametrize("reduction", ["none", "sum"])
@pytest.mark.parametrize("dtype", [ms.float16, ms.bfloat16, ms.float32])
def test_cross_entropy_loss_normal(mode, reduction, dtype):
    """
    Feature: cross_entropy_loss
    Description: test ops.cross_entropy_loss
    Expectation: expect correct result.
    """
    set_mode(mode)
    loss = {ms.float16: 1e-3,
            ms.bfloat16: 4e-3,
            ms.float32: 1e-4}[dtype]
    N, C = 7, 16
    logits_np = np.random.randn(N, C)
    target_np = np.random.randint(0, C, (N, ))
    weight_np = np.random.randn(C,)
    logits_ms = Tensor(logits_np, dtype)
    target_ms = Tensor(target_np, ms.int64)
    weight_ms = Tensor(weight_np, ms.float32)
    actual_loss = cross_entropy_loss_forward_func(logits_ms, target_ms, weight_ms, reduction=reduction)
    expect_loss = cross_entropy_loss_expect(logits_np, target_np, weight_np, reduction=reduction)
    assert np.allclose(actual_loss[0].asnumpy() if dtype != ms.bfloat16 else actual_loss[0].float().asnumpy(), expect_loss, loss, loss)
    actual_grad = cross_entropy_loss_backward_func(logits_ms, target_ms, weight_ms, reduction=reduction)
    expect_grad = cross_entropy_loss_grad_expect(logits_np, target_np, weight_np, reduction=reduction)
    assert np.allclose(actual_grad.asnumpy() if dtype != ms.bfloat16 else actual_grad.float().asnumpy(), expect_grad, loss, loss)


def cross_entropy_loss_func(logits, target, weight=None, reduction='none', ignore_index=-100,
                                    label_smoothing=0.0):
    return cross_entropy_loss(logits, target, weight, reduction, ignore_index, label_smoothing)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=20.01
)
def test_cross_entropy_loss_dynamic():
    """
    Feature: Ops
    Description: test op cross_entropy_loss dynamic shape
    Expectation: expect correct result.
    """
    dtype=ms.float32
    logits_ms1 = Tensor(np.random.randn(7, 10), dtype)
    target_ms1 = Tensor(np.random.randint(0, 10, size=7), ms.int64)
    weight_ms1 = Tensor(np.random.randn(10), ms.float32)

    input_case1 = [logits_ms1, target_ms1, weight_ms1]

    # test case 2
    logits_ms2 = Tensor(np.random.randn(13, 256), dtype)
    target_ms2 = Tensor(np.random.randint(0, 256, size=13), ms.int64)
    weight_ms2 = Tensor(np.random.randn(256), ms.float32)

    input_case2 = [logits_ms2, target_ms2, weight_ms2]
    TEST_OP(
        cross_entropy_loss_func,
        [input_case1, input_case2],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['ScalarTensor'],
        case_config={'disable_input_check': True,
                     'ignore_output_index': [2, 3],
                     'all_dim_zero': True,
                     'deterministic_use_origin_inputs': True}
    )
