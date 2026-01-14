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
import mindspore as ms
import tests.st.utils.test_utils as test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape):
    return np.random.randint(0, 10, shape)


def forward_expect_func(x, return_inverse=False, return_counts=False):
    return np.unique(x, False, return_inverse, return_counts)


@test_utils.run_with_cell
def unique_forward_func(x, is_sorted=True, return_inverse=False, return_counts=False, dim=None):
    return x.unique(is_sorted, return_inverse, return_counts, dim)


@test_utils.run_with_cell
def unique_forward_func_dynamic(x, is_sorted=True, dim=1):
    return x.unique(is_sorted, True, True, dim)


@arg_mark(
    plat_marks=['platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_unique_forward_dim_None(mode):
    """
    Feature: pyboost function.
    Description: test function Tensor.unique forward dim None.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x_np = generate_random_input((5, 6, 7))
    x = ms.Tensor(x_np)

    expect_out1 = forward_expect_func(x_np)
    expect_out2, expect_inverse2 = forward_expect_func(x_np, True, False)
    expect_inverse2 = expect_inverse2.reshape(5, 6, 7)
    expect_out3, expect_counts3 = forward_expect_func(x_np, False, True)
    expect_out4, expect_inverse4, expect_counts4 = forward_expect_func(x_np, True, True)
    expect_inverse4 = expect_inverse4.reshape(5, 6, 7)

    out1 = unique_forward_func(x)
    out2, inverse2 = unique_forward_func(x, True, True, False, None)
    out3, counts3 = unique_forward_func(x, True, False, True, None)
    out4, inverse4, counts4 = unique_forward_func(x, True, True, True, None)

    np.testing.assert_allclose(out1.asnumpy(), expect_out1, rtol=1e-3)
    np.testing.assert_allclose(out2.asnumpy(), expect_out2, rtol=1e-3)
    np.testing.assert_allclose(inverse2.asnumpy(), expect_inverse2, rtol=1e-3)
    np.testing.assert_allclose(out3.asnumpy(), expect_out3, rtol=1e-3)
    np.testing.assert_allclose(counts3.asnumpy(), expect_counts3, rtol=1e-3)
    np.testing.assert_allclose(out4.asnumpy(), expect_out4, rtol=1e-3)
    np.testing.assert_allclose(inverse4.asnumpy(), expect_inverse4, rtol=1e-3)
    np.testing.assert_allclose(counts4.asnumpy(), expect_counts4, rtol=1e-3)


@arg_mark(
    plat_marks=['platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_unique_forward_with_dim(mode):
    """
    Feature: pyboost function.
    Description: test function Tensor.unique forward dim not None.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = ms.Tensor([[1, 3, 2, 3, 4, 5, 3, 2],
                   [2, 4, 3, 2, 2, 5, 2, 5],
                   [1, 3, 2, 3, 4, 5, 3, 2]])

    expect_out1 = np.array([[1, 2, 2, 3, 3, 4, 5],
                            [2, 3, 5, 2, 4, 2, 5],
                            [1, 2, 2, 3, 3, 4, 5]])
    expect_inverse1 = np.array([0, 4, 1, 3, 5, 6, 3, 2])
    expect_counts1 = np.array([1, 1, 1, 2, 1, 1, 1])

    expect_out2 = np.array([[1, 3, 2, 3, 4, 5, 3, 2],
                            [2, 4, 3, 2, 2, 5, 2, 5]])
    expect_inverse2 = np.array([0, 1, 0])
    expect_counts2 = np.array([2, 1])

    out1, inverse1, counts1 = unique_forward_func(x, True, True, True, 1)
    out2, inverse2, counts2 = unique_forward_func(x, True, True, True, 0)

    np.testing.assert_allclose(out1.asnumpy(), expect_out1, rtol=1e-3)
    np.testing.assert_allclose(inverse1.asnumpy(), expect_inverse1, rtol=1e-3)
    np.testing.assert_allclose(counts1.asnumpy(), expect_counts1, rtol=1e-3)
    np.testing.assert_allclose(out2.asnumpy(), expect_out2, rtol=1e-3)
    np.testing.assert_allclose(inverse2.asnumpy(), expect_inverse2, rtol=1e-3)
    np.testing.assert_allclose(counts2.asnumpy(), expect_counts2, rtol=1e-3)


@arg_mark(
    plat_marks=['platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
def test_tensor_unique_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function Tensor.unique forward with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = ms.Tensor(generate_random_input((7, 8, 9)))
    sorted1 = True
    dim1 = 0

    x2 = ms.Tensor(generate_random_input((8, 9)))
    sorted2 = False
    dim2 = 1

    test_cell = test_utils.to_cell_obj(unique_forward_func_dynamic)
    TEST_OP(test_cell, [[x1, sorted1, dim1], [x2, sorted2, dim2]],
            disable_mode=["GRAPH_MODE_GE"],
            disable_case=['ScalarTensor'],
            case_config={'disable_grad': True})
