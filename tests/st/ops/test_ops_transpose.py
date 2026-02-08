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
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, jit
from mindspore.mint import transpose
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def transpose_forward_func(x, dim0, dim1):
    return transpose(x, dim0, dim1)


@test_utils.run_with_cell
def transpose_backward_func(x, dim0, dim1):
    return ms.grad(transpose_forward_func, (0))(x, dim0, dim1)


@jit(backend="ms_backend")
@test_utils.run_with_cell
def transpose_vmap_func(x, dim0, dim1):
    return ops.vmap(transpose_forward_func, in_axes=(0, None, None), out_axes=0)(x, dim0, dim1)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_transpose_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function transpose forward and backward.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    # forward
    x = ms.Tensor(np.arange(12).reshape(2, 3, 2).astype(np.float32))
    output_f1 = transpose_forward_func(x, 0, 1)
    expect_f1 = np.array([[[0., 1.], [6., 7.]], [[2., 3.], [8., 9.]], [[4., 5.], [10., 11.]]]).astype(np.float32)
    np.testing.assert_allclose(output_f1.asnumpy(), expect_f1, rtol=1e-3)

    # backward
    output_b = transpose_backward_func(x, 0, 1)
    expect_b = np.ones((2, 3, 2)).astype(np.float32)
    np.testing.assert_allclose(output_b.asnumpy(), expect_b, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_transpose_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function transpose vmap feature.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.arange(12).reshape(2, 3, 2).astype(np.float32))
    output = transpose_vmap_func(x, 0, 1)
    expect = np.array([[[0, 2, 4], [1, 3, 5]], [[6, 8, 10], [7, 9, 11]]]).astype(np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
def test_ops_transpose_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function transpose with dynamic shape and dynamic rank.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3, 1, 2), np.float32)
    dim0_0 = 0
    dim1_0 = 1
    x2 = generate_random_input((3, 1, 2, 1, 4), np.float32)
    dim0_1 = 2
    dim1_1 = 3
    TEST_OP(transpose_forward_func, [[ms.Tensor(x1), dim0_0, dim1_0], [ms.Tensor(x2), dim0_1, dim1_1]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['ScalarTensor'])
