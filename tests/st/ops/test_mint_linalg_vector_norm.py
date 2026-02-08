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

import numpy as np
import pytest

import mindspore as ms
from mindspore import ops, Tensor, mint

import tests.st.utils.test_utils as test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def vector_norm_forward_func(x, p=2, dim=None, keepdim=False):
    return ops.vector_norm(x, ord=p, axis=dim, keepdims=keepdim)

@test_utils.run_with_cell
def vector_norm_backward_func(x, p, dim, keepdim):
    return ms.grad(vector_norm_forward_func, (0))(x, p, dim, keepdim)

@test_utils.run_with_cell
def linalg_vector_norm_forward_func(x, p=2, dim=None, keepdim=False):
    return mint.linalg.vector_norm(x, ord=p, dim=dim, keepdim=keepdim)

@test_utils.run_with_cell
def linalg_vector_norm_backward_func(x, p, dim, keepdim):
    return ms.grad(linalg_vector_norm_forward_func, (0))(x, p, dim, keepdim)

@test_utils.run_with_cell
def linalg_vector_norm_forward_dyn(x):
    return mint.linalg.vector_norm(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=52.00)
@pytest.mark.parametrize('p', [-np.inf, 2, 6.5])
@pytest.mark.parametrize('dim', [None, 1, (1, 2)])
@pytest.mark.parametrize('keepdim', [True, False])
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_linalg_vector_norm(mode, p, dim, keepdim):
    """
    Feature: linalg.vector_norm
    Description: Verify the result of linalg.vector_norm
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    ms.set_context(mode=mode)
    x = ms.Tensor(np.random.randn(2, 3, 4), dtype=ms.float32)
    output = linalg_vector_norm_forward_func(x, p=p, dim=dim, keepdim=keepdim)
    expect_output = vector_norm_forward_func(x, p=p, dim=dim, keepdim=keepdim)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())

    grad_output = linalg_vector_norm_backward_func(x, p=p, dim=dim, keepdim=keepdim)
    expect_grad_output = vector_norm_backward_func(x, p=p, dim=dim, keepdim=keepdim)
    assert np.allclose(grad_output.asnumpy(), expect_grad_output.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
def test_linalg_vector_norm_dyn():
    """
    Feature: pyboost function.
    Description: test mint.linalg.vector_norm with dynamic rank/shape.
    Expectation: success.
    """
    input_x1 = np.random.randn(*(3, 3)).astype(np.float32)
    input_x2 = np.random.randn(*(3, 3, 3)).astype(np.float32)
    in1 = Tensor(input_x1)
    in2 = Tensor(input_x2)
    TEST_OP(linalg_vector_norm_forward_dyn, [[in1], [in2]], disable_mode=['GRAPH_MODE_GE'])
