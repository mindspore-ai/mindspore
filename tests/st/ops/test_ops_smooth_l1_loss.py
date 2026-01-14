# Copyright 2024 Huawei Technocasties Co., Ltd
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
from mindspore import ops
from mindspore.mint.nn.functional import smooth_l1_loss
import mindspore as ms
import tests.st.utils.test_utils as test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x_np, target_np, reduction, beta):
    result = np.zeros_like(x_np, dtype=x_np.dtype)
    for index, _ in np.ndenumerate(x_np):
        abs_diff = np.abs(x_np[index] - target_np[index])
        if abs_diff < beta:
            result[index] = 0.5 * np.square(target_np[index] - x_np[index]).astype(x_np.dtype) / beta
        else:
            result[index] = abs_diff - 0.5 * beta
    if reduction == 'mean':
        return result.mean(axis=None)
    if reduction == 'sum':
        return result.sum(axis=None)
    return result


def generate_expect_backward_output(reduction):
    input_grad_sum_none = np.array([-0.4213, 0.0380, 0.2522, 1.0000, 0.1349, -0.1500, 1.0000, 0.1703,
                                    0.0570, 0.3627, 0.0601, -1.0000, 0.1111, -0.3720, -1.0000, 0.2863,
                                    -0.5312, 0.9887, 0.1827, -0.6996])
    input_grad_mean = np.array([-0.0211, 0.0019, 0.0126, 0.0500, 0.0067, -0.0075, 0.0500, 0.0085,
                                0.0029, 0.0181, 0.0030, -0.0500, 0.0056, -0.0186, -0.0500, 0.0143,
                                -0.0266, 0.0494, 0.0091, -0.0350])
    output = {"mean": [input_grad_mean, -input_grad_mean],
              "sum": [input_grad_sum_none, -input_grad_sum_none],
              "none": [input_grad_sum_none, -input_grad_sum_none]}
    return output[reduction]


@test_utils.run_with_cell
def smooth_l1_loss_forward_func(inputx, target, reduction="mean", beta=1.0):
    return smooth_l1_loss(inputx, target, reduction, beta)


@test_utils.run_with_cell
def smooth_l1_loss_backward_func(inputx, target, reduction="mean", beta=1.0):
    grad_op = ms.grad(smooth_l1_loss_forward_func, (0, 1))
    return grad_op(inputx, target, reduction, beta)


@test_utils.run_with_cell
def smooth_l1_loss_vmap_func(inputx, target, reduction="mean", beta=1.0):
    return ops.vmap(smooth_l1_loss_forward_func, in_axes=(0, 0, None, None),
                    out_axes=(0))(inputx, target, reduction, beta)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize("mode", ["pynative", "KBK", "graph"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_smooth_l1_loss_normal(mode, reduction):
    """
    Feature: pyboost function.
    Description: test function smooth_l1_loss forward and backward.
    Expectation: expect correct result.
    """
    inputx_f = generate_random_input((2, 3, 4, 5), np.float32)
    target_f = generate_random_input((2, 3, 4, 5), np.float32)
    beta_f = 6.7

    np.random.seed(42)
    inputx_b = generate_random_input((20,), np.float32)
    target_b = generate_random_input((20,), np.float32)
    beta_b = 2.3

    expect_forward = generate_expect_forward_output(inputx_f, target_f, reduction, beta_f)
    expect_backward = generate_expect_backward_output(reduction)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = smooth_l1_loss_forward_func(ms.Tensor(inputx_f), ms.Tensor(target_f), reduction, beta_f)
        output_backward = smooth_l1_loss_backward_func(ms.Tensor(inputx_b), ms.Tensor(target_b), reduction, beta_b)

    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op_froward = ms.jit(smooth_l1_loss_forward_func, jit_level="O0")
        output_forward = op_froward(ms.Tensor(inputx_f), ms.Tensor(target_f), reduction, beta_f)
        op_backward = ms.jit(smooth_l1_loss_backward_func, jit_level="O0")
        output_backward = op_backward(ms.Tensor(inputx_b), ms.Tensor(target_b), reduction, beta_b)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op_froward = ms.jit(smooth_l1_loss_forward_func, backend="GE")
        output_forward = op_froward(ms.Tensor(inputx_f), ms.Tensor(target_f), reduction, beta_f)
        op_backward = ms.jit(smooth_l1_loss_backward_func, backend="GE")
        output_backward = op_backward(ms.Tensor(inputx_b), ms.Tensor(target_b), reduction, beta_b)

    np.testing.assert_allclose(output_forward.asnumpy(), expect_forward, rtol=1e-3)
    np.testing.assert_allclose(output_backward[0].asnumpy(), expect_backward[0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(output_backward[1].asnumpy(), expect_backward[1], rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize("mode", ["pynative", "KBK", "graph"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_smooth_l1_loss_bf16(mode, reduction):
    """
    Feature: pyboost function.
    Description: test function smooth_l1_loss forward(bf16).
    Expectation: expect correct result.
    """
    inputx = generate_random_input((5, 6, 7, 8), np.float32)
    target = generate_random_input((5, 6, 7, 8), np.float32)
    inputx_tensor = ms.Tensor(inputx, dtype=ms.bfloat16)
    target_tensor = ms.Tensor(target, dtype=ms.bfloat16)
    beta = 1.2
    expect = generate_expect_forward_output(inputx, target, reduction, beta)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = smooth_l1_loss_forward_func(inputx_tensor, target_tensor, reduction, beta)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op_froward = ms.jit(smooth_l1_loss_forward_func, jit_level="O0")
        output = op_froward(inputx_tensor, target_tensor, reduction, beta)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op_froward = ms.jit(smooth_l1_loss_forward_func, backend="GE")
        output = op_froward(inputx_tensor, target_tensor, reduction, beta)
    np.testing.assert_allclose(output.float().asnumpy(), expect, rtol=4e-3, atol=1e-2)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows',
                      'cpu_macos'], level_mark='level1', card_mark='onecard',
                      essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize("mode", ["pynative", "KBK", "graph"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_smooth_l1_loss_vmap(mode, reduction):
    """
    Feature: pyboost function.
    Description: test function smooth_l1_loss vmap feature.
    Expectation: expect correct result.
    """
    inputx = generate_random_input((2, 3, 4, 5), np.float32)
    target = generate_random_input((2, 3, 4, 5), np.float32)
    beta = 2.3
    expect = []
    for s_x_np, s_y_np in zip(inputx, target):
        np_out = generate_expect_forward_output(s_x_np, s_y_np, reduction, beta)
        expect.append(np_out)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = smooth_l1_loss_vmap_func(ms.Tensor(inputx), ms.Tensor(target), reduction, beta)

    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op_froward = ms.jit(smooth_l1_loss_vmap_func, jit_level="O0")
        output = op_froward(ms.Tensor(inputx), ms.Tensor(target), reduction, beta)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op_froward = ms.jit(smooth_l1_loss_vmap_func, backend="GE")
        output = op_froward(ms.Tensor(inputx), ms.Tensor(target), reduction, beta)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows',
                      'cpu_macos'], level_mark='level1', card_mark='onecard',
                      essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_smooth_l1_loss_dynamic_shape(reduction):
    """
    Feature: pyboost function.
    Description: test function smooth_l1_loss forward with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = ms.Tensor(np.random.rand(7, 8, 9).astype(np.float32))
    target1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    beta1 = 1.1

    x2 = ms.Tensor(np.random.rand(9, 8).astype(np.float32))
    target2 = ms.Tensor(generate_random_input((9, 8), np.float32))
    beta2 = 2.1

    TEST_OP(smooth_l1_loss_forward_func, [[x1, target1, reduction, beta1], [x2, target2, reduction, beta2]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True})


def ops_smooth_l1_loss_binary_compare(input_binary_data, output_binary_data, reduction="mean", beta=1.0):
    output = smooth_l1_loss_forward_func(ms.Tensor(input_binary_data[0]), ms.Tensor(input_binary_data[1]), reduction,
                                         beta)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-4, 1e-4)
    grads = smooth_l1_loss_backward_func(ms.Tensor(input_binary_data[0]), ms.Tensor(input_binary_data[1]), reduction,
                                         beta)
    np.allclose(grads[0].asnumpy(), output_binary_data[1], 1e-4, 1e-4)
    np.allclose(grads[1].asnumpy(), output_binary_data[2], 1e-4, 1e-4)


@ops_binary_cases(OpsBinaryCase(input_info=[((1, 576, 128, 16, 2), np.float32), ((1, 576, 128, 16, 2), np.float32)],
                                output_info=[((1, 576, 128, 16, 2), np.float32), ((1, 576, 128, 16, 2), np.float32),
                                             ((1, 576, 128, 16, 2), np.float32)],
                                extra_info='auto_drive'))
def ops_smooth_l1_loss_binary_case1(input_binary_data=None, output_binary_data=None):
    ops_smooth_l1_loss_binary_compare(input_binary_data, output_binary_data, 'none', 1.0)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=3072.0)
@pytest.mark.parametrize("mode", ["pynative", "kbk", "ge"])
def test_ops_smooth_l1_loss_binary_cases(mode):
    """
    Feature: pyboost function.
    Description: test function smooth_l1_loss forward and backward with binary data.
    Expectation: expect correct result.
    """
    if mode == "kbk":
        if ms.context.get_context("device_target") != "Ascend":
            return
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    elif mode == 'ge':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O2')
    else:
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

    ops_smooth_l1_loss_binary_case1()
