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
from mindspore.mint import unique_consecutive
import mindspore as ms
import tests.st.utils.test_utils as test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark

def generate_random_input(shape):
    return np.random.randint(0, 5, shape)

def unique_consecutive_numpy_impl(arr, return_inverse=False, return_counts=False, axis=None):
    if axis is None:
        arr = arr.flatten()
        diff = np.ones(len(arr), dtype=bool)
        diff[1:] = arr[1:] != arr[:-1]
        unique = arr[diff]

        if not return_inverse and not return_counts:
            return unique

        inverse_indices = np.cumsum(diff) - 1
        if return_counts:
            counts = np.diff(np.append(np.where(diff)[0], len(arr)))

        result = [unique]
        if return_inverse:
            result.append(inverse_indices)
        if return_counts:
            result.append(counts)
    else:
        arr = np.moveaxis(arr, axis, 0)
        shape = arr.shape
        arr_flat = arr.reshape(shape[0], -1)

        diff = np.ones(arr_flat.shape[0], dtype=bool)
        diff[1:] = (arr_flat[1:] != arr_flat[:-1]).any(axis=1)
        unique = arr[diff]

        if not return_inverse and not return_counts:
            return np.moveaxis(unique, 0, axis)

        inverse_indices = np.cumsum(diff) - 1
        inverse_indices_full_shape = inverse_indices.reshape((inverse_indices.shape[0],))

        if return_counts:
            counts = np.diff(np.append(np.where(diff)[0], len(arr_flat)))

        result = [np.moveaxis(unique, 0, axis)]
        if return_inverse:
            result.append(inverse_indices_full_shape)
        if return_counts:
            result.append(counts)

    return tuple(result)

def forward_expect_func(inputx, return_inverse=False, return_counts=False, dim=None):
    return unique_consecutive_numpy_impl(inputx, return_inverse, return_counts, dim)

@test_utils.run_with_cell
def unique_consecutive_forward_func(inputx, return_inverse=False, return_counts=False, dim=None):
    return unique_consecutive(inputx, return_inverse=return_inverse, return_counts=return_counts, dim=dim)

@test_utils.run_with_cell
def unique_consecutive_forward_func_dynamic(inputx, dim=1):
    return unique_consecutive(inputx, True, True, dim)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize("mode", ["pynative", "KBK", "O2"])
def test_ops_unique_consecutive_forward(mode):
    """
    Feature: pyboost function.
    Description: test function unique_consecutive forward dim None.
    Expectation: expect correct result.
    """
    inputx_np = generate_random_input((2, 3, 4))
    inputx = ms.Tensor(inputx_np)

    expect_out1 = forward_expect_func(inputx_np)
    expect_out2, expect_inverse2 = forward_expect_func(inputx_np, True, False)
    expect_inverse2 = expect_inverse2.reshape(2, 3, 4)
    expect_out3, expect_counts3 = forward_expect_func(inputx_np, False, True)
    expect_out4, expect_inverse4, expect_counts4 = forward_expect_func(inputx_np, True, True)
    expect_inverse4 = expect_inverse4.reshape(2, 3, 4)
    expect_out5, expect_inverse5, expect_counts5 = forward_expect_func(inputx_np, True, True, 1)
    expect_out6, expect_inverse6, expect_counts6 = forward_expect_func(inputx_np, True, True, 0)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        out1 = unique_consecutive_forward_func(inputx)
        out2, inverse2 = unique_consecutive_forward_func(inputx, True, False, None)
        out3, counts3 = unique_consecutive_forward_func(inputx, False, True, None)
        out4, inverse4, counts4 = unique_consecutive_forward_func(inputx, True, True, None)
        out5, inverse5, counts5 = unique_consecutive_forward_func(inputx, True, True, 1)
        out6, inverse6, counts6 = unique_consecutive_forward_func(inputx, True, True, 0)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op = ms.jit(unique_consecutive_forward_func, jit_level="O0")
        out1 = op(inputx)
        out2, inverse2 = op(inputx, True, False, None)
        out3, counts3 = op(inputx, False, True, None)
        out4, inverse4, counts4 = op(inputx, True, True, None)
        out5, inverse5, counts5 = op(inputx, True, True, 1)
        out6, inverse6, counts6 = op(inputx, True, True, 0)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op = ms.jit(unique_consecutive_forward_func, backend="GE")
        out1 = unique_consecutive_forward_func(inputx)
        out2, inverse2 = unique_consecutive_forward_func(inputx, True, False, None)
        out3, counts3 = unique_consecutive_forward_func(inputx, False, True, None)
        out4, inverse4, counts4 = unique_consecutive_forward_func(inputx, True, True, None)
        out5, inverse5, counts5 = unique_consecutive_forward_func(inputx, True, True, 1)
        out6, inverse6, counts6 = unique_consecutive_forward_func(inputx, True, True, 0)

    np.testing.assert_allclose(out1.asnumpy(), expect_out1, rtol=1e-3)
    np.testing.assert_allclose(out2.asnumpy(), expect_out2, rtol=1e-3)
    np.testing.assert_allclose(inverse2.asnumpy(), expect_inverse2, rtol=1e-3)
    np.testing.assert_allclose(out3.asnumpy(), expect_out3, rtol=1e-3)
    np.testing.assert_allclose(counts3.asnumpy(), expect_counts3, rtol=1e-3)
    np.testing.assert_allclose(out4.asnumpy(), expect_out4, rtol=1e-3)
    np.testing.assert_allclose(inverse4.asnumpy(), expect_inverse4, rtol=1e-3)
    np.testing.assert_allclose(counts4.asnumpy(), expect_counts4, rtol=1e-3)
    np.testing.assert_allclose(out5.asnumpy(), expect_out5, rtol=1e-3)
    np.testing.assert_allclose(inverse5.asnumpy(), expect_inverse5, rtol=1e-3)
    np.testing.assert_allclose(counts5.asnumpy(), expect_counts5, rtol=1e-3)
    np.testing.assert_allclose(out6.asnumpy(), expect_out6, rtol=1e-3)
    np.testing.assert_allclose(inverse6.asnumpy(), expect_inverse6, rtol=1e-3)
    np.testing.assert_allclose(counts6.asnumpy(), expect_counts6, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
def test_ops_unique_consecutive_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function unique_consecutive forward with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = ms.Tensor(generate_random_input((7, 8, 9)))
    dim1 = 0

    x2 = ms.Tensor(generate_random_input((8, 9)))
    dim2 = 1

    test_cell = test_utils.to_cell_obj(unique_consecutive_forward_func_dynamic)
    TEST_OP(test_cell, [[x1, dim1], [x2, dim2]],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_grad': True})
