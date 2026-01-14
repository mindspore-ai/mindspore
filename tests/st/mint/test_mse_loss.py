# Copyright 2024 Huawei Technomse_lossies Co., Ltd
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
from mindspore import mint, context, Tensor
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def get_broadcast_index(x_shape, y_shape):
    x_size = len(x_shape)
    y_size = len(y_shape)
    n = max(len(x_shape), len(y_shape))
    x_bdi = []
    y_bdi = []

    for i in range(n, 0, -1):
        x_i = 1 if x_size < i else x_shape[x_size - i]
        y_i = 1 if y_size < i else y_shape[y_size - i]
        reduce_index = n - i
        if x_i == y_i:
            continue
        elif x_i == 1:
            x_bdi.append(reduce_index)
        elif y_i == 1:
            y_bdi.append(reduce_index)
    return x_bdi, y_bdi


def unbroadcast(grad_x, grad_y, x_shape, y_shape):
    keepdim_x = grad_x.ndim == len(x_shape)
    keepdim_y = grad_y.ndim == len(y_shape)
    x_bdi, y_bdi = get_broadcast_index(x_shape, y_shape)

    x_grad_unbroadcasted = np.sum(grad_x, axis=tuple(x_bdi), keepdims=keepdim_x)
    y_grad_unbroadcasted = np.sum(grad_y, axis=tuple(y_bdi), keepdims=keepdim_y)

    x_grad_unbroadcasted = np.reshape(x_grad_unbroadcasted, x_shape)
    y_grad_unbroadcasted = np.reshape(y_grad_unbroadcasted, y_shape)

    return x_grad_unbroadcasted, y_grad_unbroadcasted


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(x_np, target_np, reduction, dtype):
    y = np.square(target_np - x_np).astype(dtype)
    if reduction == 'mean':
        return y.mean(axis=None)
    if reduction == 'sum':
        return y.sum(axis=None)
    return y

def generate_expect_backward_output(x_np, target_np, reduction, dtype):
    cof = 2.
    if reduction == 'mean':
        cof = cof/(x_np - target_np).size

    grad_input = cof * (x_np - target_np).astype(dtype)
    grad_target = cof * (target_np - x_np).astype(dtype)

    return  unbroadcast(grad_input, grad_target, x_np.shape, target_np.shape)

@test_utils.run_with_cell
def mse_loss_forward_func(x, target, reduction):
    return mint.nn.functional.mse_loss(x, target, reduction)

@test_utils.run_with_cell
def mse_loss_backward_func(x, target, reduction):
    return ms.ops.grad(mse_loss_forward_func, (0, 1))(x, target, reduction)



@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_mse_loss_forward(mode, reduction):
    """
    Feature: test mse_loss operator
    Description: test mse_loss run by pyboost
    Expectation: success
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

    x_np = generate_random_input((8, 1, 6, 1), np.float32)
    x_tensor = Tensor(x_np, ms.float32)
    target_np = generate_random_input((7, 1, 5), np.float32)
    target_tensor = Tensor(target_np, ms.float32)

    output = mse_loss_forward_func(x_tensor, target_tensor, reduction)
    expect = generate_expect_forward_output(x_np, target_np, reduction, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)

    output_backward = mse_loss_backward_func(x_tensor, target_tensor, reduction)
    expect_backward = generate_expect_backward_output(x_np, target_np, reduction, np.float32)
    np.testing.assert_allclose(output_backward[0].asnumpy(), expect_backward[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(output_backward[1].asnumpy(), expect_backward[1], rtol=1e-4, atol=1e-4)



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize("reduction", ["mean", "none", "sum"])
def test_mse_loss_dynamic_shape_testop(reduction):
    """
    Feature: Test mse_loss with dynamic shape in graph mode using TEST_OP.
    Description: call ops.mse_loss with valid input and index.
    Expectation: return the correct value.
    """

    x1 = generate_random_input((3, 4, 5), np.float32)
    target1 = generate_random_input((3, 1, 5), np.float32)
    reduction1 = reduction
    x2 = generate_random_input((3, 7, 1, 3), np.float32)
    target2 = generate_random_input((1, 8, 1), np.float32)
    reduction2 = reduction

    TEST_OP(mint.nn.functional.mse_loss,
            [[ms.Tensor(x1), ms.Tensor(target1), reduction1],
             [ms.Tensor(x2), ms.Tensor(target2), reduction2]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True})
