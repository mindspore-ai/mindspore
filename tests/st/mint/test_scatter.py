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
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import mint, Tensor, jit, context, ops


@test_utils.run_with_cell
def scatter_forward_func(x, dim, index, src):
    return mint.scatter(x, dim, index, src)

@test_utils.run_with_cell
def scatter_backward_func(x, dim, index, src):
    return ops.grad(scatter_forward_func, (0, 3))(x, dim, index, src)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.05)
@pytest.mark.parametrize('mode', ['GE', 'pynative', 'KBK'])
def test_scatter_forward_backward(mode):
    """
    Feature: Scatter
    Description: test op Scatter
    Expectation: expect correct result.
    """
    # scatter forward
    input_x = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    index = Tensor(np.array([[0, 1], [0, 1], [0, 1]]), dtype=ms.int64)
    src = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=ms.float32)
    dim = 1
    expect = np.array([[1., 2., 0., 0., 0.],
                       [4., 5., 0., 0., 0.],
                       [7., 8., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.]])
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        out = scatter_forward_func(input_x, dim, index, src)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        out = (jit(scatter_forward_func, jit_level="O0"))(input_x, dim, index, src)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        out = scatter_forward_func(input_x, dim, index, src)
    assert np.allclose(out.asnumpy(), expect)

    # scatter value forward
    input_x = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    index = Tensor(np.array([[0, 1], [0, 1], [0, 1]]), dtype=ms.int64)
    src = 3.
    dim = 1
    expect = np.array([[3., 3., 0., 0., 0.],
                       [3., 3., 0., 0., 0.],
                       [3., 3., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.]])
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        out = scatter_forward_func(input_x, dim, index, src)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        out = (jit(scatter_forward_func, jit_level="O0"))(input_x, dim, index, src)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        out = scatter_forward_func(input_x, dim, index, src)
    assert np.allclose(out.asnumpy(), expect)

    # scatter backward
    input_x = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    index = Tensor(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), dtype=ms.int64)
    src = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=ms.float32)
    dim = 1
    expect_input_grad = np.array([[0., 0., 0., 1., 1.],
                                  [0., 0., 0., 1., 1.],
                                  [0., 0., 0., 1., 1.],
                                  [1., 1., 1., 1., 1.],
                                  [1., 1., 1., 1., 1.]])
    expect_src_grad = np.array([[1., 1., 1.],
                                [1., 1., 1.],
                                [1., 1., 1.]])
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        input_grad, src_grad = scatter_backward_func(input_x, dim, index, src)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        input_grad, src_grad = \
            (jit(scatter_backward_func, jit_level="O0"))(input_x, dim, index, src)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        input_grad, src_grad = scatter_backward_func(input_x, dim, index, src)
    assert np.allclose(input_grad.asnumpy(), expect_input_grad)
    assert np.allclose(src_grad.asnumpy(), expect_src_grad)

    # scatter value backward
    input_x = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    index = Tensor(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), dtype=ms.int64)
    src = 2.
    dim = 1
    expect_input_grad = np.array([[0., 0., 0., 1., 1.],
                                  [0., 0., 0., 1., 1.],
                                  [0., 0., 0., 1., 1.],
                                  [1., 1., 1., 1., 1.],
                                  [1., 1., 1., 1., 1.]])
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        input_grad = scatter_backward_func(input_x, dim, index, src)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        input_grad = (jit(scatter_backward_func, jit_level="O0"))(input_x, dim, index, src)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        input_grad = scatter_backward_func(input_x, dim, index, src)
    assert np.allclose(input_grad.asnumpy(), expect_input_grad)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=3072.00)
def test_f_scatter_dynamic():
    """
    Feature: test dynamicscatter.
    Description: test auto grad of op Scatter.
    Expectation: expect correct result.
    """
    def scatter(input_x, dim, index, src):
        op = ops.auto_generate.Scatter()
        return op(input_x, dim, index, src)

    input_1 = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    index_1 = Tensor(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), dtype=ms.int64)
    src_1 = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=ms.float32)
    dim_1 = 1
    input_2 = Tensor(np.ones((3, 4, 5)), dtype=ms.float32)
    index_2 = Tensor(np.array([[[0, 1], [1, 0], [1, 1]], [[0, 1], [1, 0], [0, 0]]]), dtype=ms.int64)
    src_2 = Tensor(np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]), dtype=ms.float32)
    dim_2 = 0
    # dynamic string is not supported
    TEST_OP(scatter,
            [[input_1, dim_1, index_1, src_1],
             [input_2, dim_2, index_2, src_2]],
            disable_case=['EmptyTensor',
                          'ScalarTensor'])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=3072.01)
def test_f_scatter_scalar_value_dynamic():
    """
    Feature: test dynamicscatter.
    Description: test auto grad of op Scatter.
    Expectation: expect correct result.
    """
    def scatter(input_x, dim, index, src):
        op = ops.auto_generate.ScatterValue()
        return op(input_x, dim, index, src)

    input_1 = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    index_1 = Tensor(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), dtype=ms.int64)
    src_1 = 2.
    dim_1 = 1
    input_2 = Tensor(np.zeros((3, 4, 5)), dtype=ms.float32)
    index_2 = Tensor(np.array([[[0, 1], [1, 2], [2, 2]], [[0, 1], [1, 2], [2, 2]]]), dtype=ms.int64)
    src_2 = 3.
    dim_2 = 0
    TEST_OP(scatter,
            [[input_1, dim_1, index_1, src_1],
             [input_2, dim_2, index_2, src_2]],
            disable_case=['EmptyTensor',
                          'ScalarTensor'])
