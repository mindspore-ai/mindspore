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
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP

import mindspore as ms
from mindspore import mint, Tensor, jit, context, JitConfig, ops


@test_utils.run_with_cell
def scatter_forward_func(x, dim, index, src):
    return mint.scatter(x, dim, index, src)

@test_utils.run_with_cell
def scatter_backward_func(x, dim, index, src):
    return ops.grad(scatter_forward_func, (0, 3))(x, dim, index, src)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize("mode", ['GE', 'pynative', 'KBK'])
def test_scatter_forward(mode):
    """
    Feature: Scatter
    Description: test op Scatter
    Expectation: expect correct result.
    """
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
        out = (jit(scatter_forward_func, jit_config=JitConfig(jit_level="O0")))(input_x, dim, index, src)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        out = scatter_forward_func(input_x, dim, index, src)
    assert np.allclose(out.asnumpy(), expect)

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize("mode", ['pynative', 'KBK'])
def test_scatter_scalar_value_forward(mode):
    """
    Feature: Scatter
    Description: test op Scatter
    Expectation: expect correct result.
    """
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
        out = (jit(scatter_forward_func, jit_config=JitConfig(jit_level="O0")))(input_x, dim, index, src)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        out = scatter_forward_func(input_x, dim, index, src)
    assert np.allclose(out.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize("mode", ['GE', 'pynative', 'KBK'])
def test_scatter_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op Scatter.
    Expectation: expect correct result.
    """
    input_x = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    index = Tensor(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), dtype=ms.int64)
    src = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=ms.float32)
    dim = 1
    expect_input_grad = np.array([[1., 2., 3., 1., 1.],
                                  [4., 5., 6., 1., 1.],
                                  [7., 8., 9., 1., 1.],
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
            (jit(scatter_backward_func, jit_config=JitConfig(jit_level="O0")))(input_x, dim, index, src)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        input_grad, src_grad = scatter_backward_func(input_x, dim, index, src)
    assert np.allclose(input_grad.asnumpy(), expect_input_grad)
    assert np.allclose(src_grad.asnumpy(), expect_src_grad)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize("mode", ['pynative', 'KBK'])
def test_scatter_scalar_value_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op Scatter.
    Expectation: expect correct result.
    """
    input_x = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    index = Tensor(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), dtype=ms.int64)
    src = 2.
    dim = 1
    expect_input_grad = np.array([[2., 2., 2., 1., 1.],
                                  [2., 2., 2., 1., 1.],
                                  [2., 2., 2., 1., 1.],
                                  [1., 1., 1., 1., 1.],
                                  [1., 1., 1., 1., 1.]])
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        input_grad = scatter_backward_func(input_x, dim, index, src)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        input_grad = (jit(scatter_backward_func, jit_config=JitConfig(jit_level="O0")))(input_x, dim, index, src)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        input_grad = scatter_backward_func(input_x, dim, index, src)
    assert np.allclose(input_grad.asnumpy(), expect_input_grad)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_f_scatter_dynamic(mode):
    """
    Feature: test dynamicscatter.
    Description: test auto grad of op Scatter.
    Expectation: expect correct result.
    """
    scatter = ops.auto_generate.Scatter()
    input_1 = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    index_1 = Tensor(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), dtype=ms.int64)
    src_1 = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=ms.float32)
    dim_1 = 1
    reduce_1 = 1 # "add"
    input_2 = Tensor(np.ones((3, 4, 5)), dtype=ms.float32)
    index_2 = Tensor(np.array([[[0, 1], [1, 0], [1, 1]], [[0, 1], [1, 0], [0, 0]]]), dtype=ms.int64)
    src_2 = Tensor(np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]), dtype=ms.float32)
    dim_2 = 0
    reduce_2 = 2 # "multiply"
    TEST_OP(scatter, [[input_1, dim_1, index_1, src_1, reduce_1],
                      [input_2, dim_2, index_2, src_2, reduce_2]], mode=mode, grad=False)
    TEST_OP(scatter, [[input_1, dim_1, index_1, src_1, reduce_1],
                      [input_2, dim_2, index_2, src_2, reduce_2]], mode=mode, grad=True)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_f_scatter_scalar_value_dynamic(mode):
    """
    Feature: test dynamicscatter.
    Description: test auto grad of op Scatter.
    Expectation: expect correct result.
    """
    scatter = ops.auto_generate.ScatterValue()
    input_1 = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    index_1 = Tensor(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), dtype=ms.int64)
    src_1 = 2.
    dim_1 = 1
    reduce_1 = 1 # "add"
    input_2 = Tensor(np.zeros((3, 4, 5)), dtype=ms.float32)
    index_2 = Tensor(np.array([[[0, 1], [1, 2], [2, 2]], [[0, 1], [1, 2], [2, 2]]]), dtype=ms.int64)
    src_2 = 3.
    dim_2 = 0
    reduce_2 = 2 # "multiply"
    TEST_OP(scatter, [[input_1, dim_1, index_1, src_1, reduce_1],
                      [input_2, dim_2, index_2, src_2, reduce_2]], mode=mode, grad=False)
