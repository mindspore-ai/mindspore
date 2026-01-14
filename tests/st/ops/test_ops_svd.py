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
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.common.random_generator import generate_numpy_ndarray_by_randn
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor

def svd_forward_expect(x, full_matrices, compute_uv):
    return np.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)

@test_utils.run_with_cell
def svd_dyn_forward_func(x, full_matrices, compute_uv):
    return ops.Svd(full_matrices, compute_uv)(x)

@test_utils.run_with_cell
def svd_forward_func(x, full_matrices, compute_uv):
    return ops.svd(x, full_matrices, compute_uv)

@test_utils.run_with_cell
def svd_backward_func(x, full_matrices, compute_uv):
    return ms.grad(svd_forward_func, (0))(x, full_matrices, compute_uv)

@test_utils.run_with_cell
def svd_vmap_func(x, full_matrices, compute_uv):
    return ops.vmap(svd_forward_func, in_axes=(0, None, None), out_axes=0)(x, full_matrices, compute_uv)

def set_context_mode(mode):
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'kbk':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O2')

def get_svd_input_ndarray(shape, dtype):
    prod = 1
    for dim in shape:
        prod *= dim
    if dtype == np.complex64:
        real_np = np.arange(prod).reshape(*shape).astype(np.float32)
        image_np = np.arange(prod).reshape(*shape).astype(np.float32)
        x_np = (real_np + 1j*image_np).astype(np.complex64)
    elif dtype == np.complex128:
        real_np = np.arange(prod).reshape(*shape).astype(np.float64)
        image_np = np.arange(prod).reshape(*shape).astype(np.float64)
        x_np = (real_np + 1j*image_np).astype(np.complex128)
    else:
        x_np = np.arange(prod).reshape(*shape).astype(dtype)
    return x_np

def get_loss(dtype):
    loss = {
        np.float32: 1e-4,
        np.float64: 1e-5,
        np.complex64: 1e-4,
        np.complex128: 1e-5,
    }
    return loss[dtype]

def svd_normal_testcase(shape, dtype, expect_grad, full_matrices, compute_uv):
    x_np = get_svd_input_ndarray(shape, dtype)
    out = svd_forward_func(Tensor(x_np), full_matrices, compute_uv)
    expect = svd_forward_expect(x_np, full_matrices, compute_uv)
    loss = get_loss(dtype)
    if compute_uv:
        np.testing.assert_allclose(np.abs(out[0].asnumpy()), np.abs(expect[1]), loss, loss)
        np.testing.assert_allclose(np.abs(out[1].asnumpy()), np.abs(expect[0]), loss, loss)
        np.testing.assert_allclose(np.abs(out[2].asnumpy()), np.abs(expect[2]), loss, loss)
    else:
        np.testing.assert_allclose(np.abs(out.asnumpy()), np.abs(expect), loss, loss)
    if dtype not in (np.complex64, np.complex128) and ms.context.get_context("device_target") != "GPU":
        grad = svd_backward_func(Tensor(x_np), full_matrices, compute_uv)
        np.testing.assert_allclose(grad.asnumpy(), expect_grad, loss, loss)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential',
          mem_peak=20.00)
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex64, np.complex128])
def test_ops_svd_normal(mode, dtype):
    """
    Feature: Svd
    Description: test cases for svd: m >= n
    Expectation: the result match to numpy
    """
    if (dtype == np.complex64 or dtype == np.complex128) and ms.context.get_context("device_target") == "GPU":
        return
    if dtype == np.float64 and ms.context.get_context("device_target") == "Ascend":
        return
    set_context_mode(mode)
    expect_grad_1 = np.array([[-0.65930605, 0.63138646],
                              [0.04559251, 0.57554727],
                              [0.75049107, 0.51970808]]).astype(dtype)
    expect_grad_2 = np.array([[-0.49018779, 0.89098777],
                              [0.03374469, 0.71657469],
                              [0.55767716, 0.54216162]]).astype(dtype)
    expect_grad_3 = np.array([[-0.08193948, 0.48273947],
                              [-0.78275192, 1.53307129],
                              [0.96592546, 0.13391332]]).astype(dtype)
    svd_normal_testcase((3, 2), dtype, expect_grad_1, False, False)
    svd_normal_testcase((3, 2), dtype, expect_grad_1, True, False)
    svd_normal_testcase((3, 2), dtype, expect_grad_2, False, True)
    svd_normal_testcase((3, 2), dtype, expect_grad_3, True, True)

    expect_grad_4 = np.array([[[-0.65930605, 0.63138646],
                               [0.04559251, 0.57554727],
                               [0.75049107, 0.51970808]],
                              [[-0.31005466, 0.85860319],
                               [0.29660608, 0.49533642],
                               [0.90326682, 0.13206965]]]).astype(dtype)
    expect_grad_5 = np.array([[[-0.49018779, 0.89098777],
                               [0.03374469, 0.71657469],
                               [0.55767716, 0.54216162]],
                              [[-0.27928889, 0.9346323],
                               [0.27658402, 0.52720084],
                               [0.83245693, 0.11976939]]]).astype(dtype)
    expect_grad_6 = np.array([[[-0.08193948, 0.48273947],
                               [-0.78275192, 1.53307129],
                               [0.96592546, 0.13391332]],
                              [[0.12895937, 0.52638404],
                               [-0.5399125, 1.34369737],
                               [1.24070519, -0.28847888]]]).astype(dtype)
    svd_normal_testcase((2, 3, 2), dtype, expect_grad_4, False, False)
    svd_normal_testcase((2, 3, 2), dtype, expect_grad_4, True, False)
    svd_normal_testcase((2, 3, 2), dtype, expect_grad_5, False, True)
    svd_normal_testcase((2, 3, 2), dtype, expect_grad_6, True, True)


def svd_self_testcase(shape, dtype, full_matrices):

    def matrix_diag(diagonal, shape, dtype):
        assist_matrix = Tensor(np.zeros(shape).astype(dtype))
        return ops.MatrixSetDiagV3()(assist_matrix, diagonal, Tensor(0, ms.int32))

    x = Tensor(generate_numpy_ndarray_by_randn(shape, dtype, "svd 'x'"))
    s, u, v = svd_forward_func(x, full_matrices, True)
    matmul_op = ops.MatMul()
    if len(shape) > 2:
        matmul_op = ops.BatchMatMul()
    transpose_op = ops.Transpose()
    perm = [i for i in range(len(shape))]
    perm[-2], perm[-1] = perm[-1], perm[-2]

    if not full_matrices:
        new_shape = list(shape)
        p = min(shape[-1], shape[-2])
        new_shape[-1], new_shape[-2] = p, p
        shape = tuple(new_shape)
    output = matmul_op(u, matmul_op(matrix_diag(s, shape, dtype), transpose_op(v, tuple(perm))))

    loss = get_loss(dtype)
    np.testing.assert_allclose(output.asnumpy(), x.asnumpy(), loss, loss)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential',
          mem_peak=4.00)
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
@pytest.mark.parametrize('dtype', [np.float32])
def test_ops_svd_selfcases(mode, dtype):
    """
    Feature: Svd
    Description: test cases for svd
    Expectation: the result match to ms results
    """
    set_context_mode(mode)
    svd_self_testcase((2, 3), dtype, True)
    svd_self_testcase((3, 2), dtype, False)
    svd_self_testcase((5, 3, 3), dtype, True)
    svd_self_testcase((5, 5, 3, 2), dtype, True)
    svd_self_testcase((5, 5, 3, 2), dtype, False)


def svd_vmap_testcase(shape, dtype, full_matrices, compute_uv):
    x_np = generate_numpy_ndarray_by_randn(shape, dtype, "svd 'x'")
    out = svd_vmap_func(Tensor(x_np), full_matrices, compute_uv)
    expect = svd_forward_expect(x_np, full_matrices, compute_uv)
    loss = get_loss(dtype)
    if compute_uv:
        np.testing.assert_allclose(np.abs(out[0].asnumpy()), np.abs(expect[1]), loss, loss)
        np.testing.assert_allclose(np.abs(out[1].asnumpy()), np.abs(expect[0]), loss, loss)
        np.testing.assert_allclose(np.abs(out[2].asnumpy()), np.abs(expect[2]), loss, loss)
    else:
        np.testing.assert_allclose(np.abs(out.asnumpy()), np.abs(expect), loss, loss)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'kbk', 'ge'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_ops_svd_vmap(mode, dtype):
    """
    Feature: Svd
    Description: test cases for svd vmap
    Expectation: the result match to numpy
    """
    set_context_mode(mode)
    svd_vmap_testcase((2, 3, 2), dtype, False, False)
    svd_vmap_testcase((2, 3, 2), dtype, True, False)
    svd_vmap_testcase((2, 3, 2), dtype, False, True)
    svd_vmap_testcase((2, 3, 2), dtype, True, True)
    svd_vmap_testcase((5, 5, 3, 2), dtype, False, False)
    svd_vmap_testcase((5, 5, 3, 2), dtype, True, False)
    svd_vmap_testcase((5, 5, 3, 2), dtype, False, True)
    svd_vmap_testcase((5, 5, 3, 2), dtype, True, True)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=36.0)
def test_ops_svd_dynamic():
    """
    Feature: Ops
    Description: test op rotated_iou dynamic shape
    Expectation: expect correct result.
    """
    inputs1 = [Tensor(get_svd_input_ndarray((3, 4), np.float32)), False, True]
    inputs2 = [Tensor(get_svd_input_ndarray((2, 3, 4), np.float32)), True, True]
    TEST_OP(svd_dyn_forward_func, [inputs1, inputs2],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_nontensor_dynamic_type': 'BOTH'})

    inputs3 = [Tensor(get_svd_input_ndarray((3, 4), np.float32)), True, True]
    inputs4 = [Tensor(get_svd_input_ndarray((2, 3, 4), np.float32)), False, True]
    TEST_OP(svd_dyn_forward_func, [inputs3, inputs4],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_nontensor_dynamic_type': 'BOTH'})

    inputs1 = [Tensor(get_svd_input_ndarray((2, 3, 3), np.float32)), False, True]
    inputs2 = [Tensor(get_svd_input_ndarray((2, 3), np.float32)), True, True]
    TEST_OP(svd_dyn_forward_func, [inputs1, inputs2],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_nontensor_dynamic_type': 'BOTH'})

    inputs3 = [Tensor(get_svd_input_ndarray((2, 3, 3), np.float32)), True, True]
    inputs4 = [Tensor(get_svd_input_ndarray((2, 3), np.float32)), False, True]
    TEST_OP(svd_dyn_forward_func, [inputs3, inputs4],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_nontensor_dynamic_type': 'BOTH'})

    inputs1 = [Tensor(get_svd_input_ndarray((4, 3, 2), np.float32)), False, True]
    inputs2 = [Tensor(get_svd_input_ndarray((4, 3), np.float32)), True, True]
    TEST_OP(svd_dyn_forward_func, [inputs1, inputs2],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_nontensor_dynamic_type': 'BOTH'})

    inputs3 = [Tensor(get_svd_input_ndarray((4, 3, 2), np.float32)), True, True]
    inputs4 = [Tensor(get_svd_input_ndarray((4, 3), np.float32)), False, True]
    TEST_OP(svd_dyn_forward_func, [inputs3, inputs4],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_nontensor_dynamic_type': 'BOTH'})

    inputs5 = [Tensor(get_svd_input_ndarray((2, 3, 4), np.float32)), False, False]
    inputs6 = [Tensor(get_svd_input_ndarray((2, 3), np.float32)), True, False]
    TEST_OP(svd_dyn_forward_func, [inputs5, inputs6],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_nontensor_dynamic_type': 'BOTH',
                         'ignore_output_index': [1, 2]})

    inputs7 = [Tensor(get_svd_input_ndarray((2, 3, 4), np.float32)), True, False]
    inputs8 = [Tensor(get_svd_input_ndarray((2, 3), np.float32)), False, False]
    TEST_OP(svd_dyn_forward_func, [inputs7, inputs8],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_nontensor_dynamic_type': 'BOTH',
                         'ignore_output_index': [1, 2]})
