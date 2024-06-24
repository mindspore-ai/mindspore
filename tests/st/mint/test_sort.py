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
from tests.mark_utils import arg_mark
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, mint, jit, JitConfig
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_expect_backward_output(values, indices, x, dim):
    ones = np.ones_like(values)
    grad_output = np.zeros_like(x)
    np.put_along_axis(grad_output, indices, ones, dim)
    return grad_output


def sort_forward_func(x, dim, descending, stable):
    return mint.sort(x, dim=dim, descending=descending, stable=stable)


def sort_backward_func(x, dim, descending, stable):
    return ops.grad(sort_forward_func, (0, 1, 2, 3))(x, dim=dim, descending=descending, stable=stable)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_sort_std(descending, mode):
    """
    Feature: Test sort with standard forward, backward feature.
    Description: call mint.sort with valid input and index.
    Expectation: return the correct value.
    """
    x_numpy = np.array([[[1, 2, 3, 4], [8, 7, 2, 0], [9, 4, 1, 8]],
                        [[5, 4, 1, 8], [2, 9, 0, 7], [6, 1, 7, 4]]]).astype(np.float32)
    x = ms.Tensor(x_numpy)

    expect_indices_list = [
        np.array([[[0, 1, 2, 3], [3, 2, 1, 0], [2, 1, 3, 0]],
                  [[2, 1, 0, 3], [2, 0, 3, 1], [1, 3, 0, 2]]]),
        np.array([[[0, 0, 2, 1], [1, 2, 1, 0], [2, 1, 0, 2]],
                  [[1, 2, 1, 2], [0, 0, 0, 1], [2, 1, 2, 0]]]),
        np.array([[[0, 0, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1]],
                  [[1, 1, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]]])
    ]
    for dim, expected_indices in zip([-1, 1, -3], expect_indices_list):
        expected_output = np.sort(x_numpy, dim)

        if descending:
            if dim == -1:
                expected_output = expected_output[:, :, ::-1]
                expected_indices = expected_indices[:, :, ::-1]
            elif dim == 1:
                expected_output = expected_output[:, ::-1, :]
                expected_indices = expected_indices[:, ::-1, :]
            elif dim == -3:
                expected_output = expected_output[::-1, :, :]
                expected_indices = expected_indices[::-1, :, :]

        expected_grad = generate_expect_backward_output(expected_output, expected_indices, x, dim)

        if mode == 'pynative':
            ms.context.set_context(mode=ms.PYNATIVE_MODE)
            output, indices = sort_forward_func(x, dim, descending, False)
            ms_grad = sort_backward_func(x, dim, descending, False)
        else:
            output, indices = (jit(sort_forward_func, jit_config=JitConfig(jit_level="O0")))(x, dim, descending, False)
            ms_grad = (jit(sort_backward_func, jit_config=JitConfig(jit_level="O0")))(x, dim, descending, False)

        np.testing.assert_array_equal(output.asnumpy(), expected_output)
        np.testing.assert_array_equal(indices.asnumpy(), expected_indices)
        np.testing.assert_array_equal(ms_grad.asnumpy(), expected_grad)


def sort_forward_func_dyn(x, dim):
    return mint.sort(x, dim=dim, descending=True, stable=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sort_dynamic_shape():
    """
    Feature: Test sort with dynamic shape.
    Description: call mint.sort with valid input and index.
    Expectation: return the correct value.
    """
    x1 = np.array([[[1, 2, 3, 4], [8, 7, 2, 0], [9, 4, 1, 8]],
                   [[5, 4, 1, 8], [2, 9, 0, 7], [6, 1, 7, 4]]]).astype(np.float32)
    tensor_1 = ms.Tensor(x1)
    dim_1 = 1
    x2 = np.array([1, 0, 3, 4]).astype(np.float32)
    tensor_2 = ms.Tensor(x2)
    dim_2 = 0

    TEST_OP(sort_forward_func_dyn, [[tensor_1, dim_1], [tensor_2, dim_2]], 'sort_ext',
            disable_mode=['GRAPH_MODE'], disable_yaml_check=True)
