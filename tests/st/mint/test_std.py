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
from mindspore import Tensor, mint, ops, jit
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def std_forward(x, dim=None, correction=1, keepdim=False):
    return mint.std(x, dim=dim, correction=correction, keepdim=keepdim)


@test_utils.run_with_cell
def std_backward(x, dim=None, correction=1, keepdim=False):
    return ops.grad(std_forward, (0, 1, 2, 3))(x, dim, correction, keepdim)


@test_utils.run_with_cell
def std_forward_tensor(x, dim=None, correction=1, keepdim=False):
    out = x.std(dim, correction=correction, keepdim=keepdim)
    return out


@test_utils.run_with_cell
def std_backward_tensor(x, dim=None, correction=1, keepdim=False):
    return ops.grad(std_forward_tensor, (0, 1, 2, 3))(x, dim, correction, keepdim)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_std_tensor(mode):
    """
    Feature: mint.std
    Description: Verify the result of std tensor on Ascend
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[[-4, -6, -5, 8],
                 [3, 2, -7, 0],
                 [7, -4, -3, 8]],
                [[-7, -7, -4, -5],
                 [-6, -7, 6, -2],
                 [-2, -7, 8, -8.]]])
    expect_output = [[[2.12132025e+00, 7.07106769e-01, 7.07106769e-01, 9.19238853e+00],
                      [6.36396122e+00, 6.36396122e+00, 9.19238853e+00, 1.41421354e+00],
                      [6.36396122e+00, 2.12132025e+00, 7.77817440e+00, 1.13137083e+01]]]

    # std backward
    if mode == ms.PYNATIVE_MODE:
        output = std_forward_tensor(x, dim=0, correction=1, keepdim=True)
        input_grad = std_backward_tensor(x, dim=0, correction=1, keepdim=True)
    elif mode == ms.GRAPH_MODE:
        output = (jit(std_forward_tensor, backend="ms_backend", jit_level="O0"))(x, dim=0, correction=1, keepdim=True)
        input_grad = (jit(std_backward_tensor, backend="ms_backend", jit_level="O0"))(
            x, dim=0, correction=1, keepdim=True)
    assert input_grad.asnumpy().dtype == np.float32
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_std_norlmal(mode):
    """
    Feature: mint.std
    Description: Verify the result of std on Ascend
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[[-4, -6, -5, 8],
                 [3, 2, -7, 0],
                 [7, -4, -3, 8]],
                [[-7, -7, -4, -5],
                 [-6, -7, 6, -2],
                 [-2, -7, 8, -8.]]])
    expect_output = [[[2.12132025e+00, 7.07106769e-01, 7.07106769e-01, 9.19238853e+00],
                      [6.36396122e+00, 6.36396122e+00, 9.19238853e+00, 1.41421354e+00],
                      [6.36396122e+00, 2.12132025e+00, 7.77817440e+00, 1.13137083e+01]]]

    # std backward
    if mode == ms.PYNATIVE_MODE:
        output = std_forward(x, dim=0, correction=1, keepdim=True)
        input_grad = std_backward(x, dim=0, correction=1, keepdim=True)
    elif mode == ms.GRAPH_MODE:
        output = (jit(std_forward, backend="ms_backend", jit_level="O0"))(x, dim=0, correction=1, keepdim=True)
        input_grad = (jit(std_backward, backend="ms_backend", jit_level="O0"))(x, dim=0, correction=1, keepdim=True)
    assert input_grad.asnumpy().dtype == np.float32
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_std_default(mode):
    """
    Feature: mint.std
    Description: Verify the result of std on Ascend
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([1., 2., 3.])
    expect_output = 1.
    expect_grad = [-0.5, 0., 0.5]

    # std backward
    if mode == ms.PYNATIVE_MODE:
        output = std_forward(x)
        grad = std_backward(x)
    elif mode == ms.GRAPH_MODE:
        output = (jit(std_forward, backend="ms_backend", jit_level="O0"))(x)
        grad = (jit(std_backward, backend="ms_backend", jit_level="O0"))(x)
    assert np.allclose(output.asnumpy(), expect_output)
    assert np.allclose(grad.asnumpy(), expect_grad)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.00)
def test_mint_std_dynamic_shape():
    """
    Feature: Test std with dynamic shape.
    Description: call mint.std with valid input, dim, correction amd keepdim.
    Expectation: return the correct value.
    """
    x1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    axis1 = 0
    correction1 = 1
    keep_dims1 = False
    x2 = ms.Tensor(generate_random_input((2, 3, 4), np.float32))
    axis2 = 1
    correction2 = 2
    keep_dims2 = True
    TEST_OP(std_forward, [[x1, axis1, correction1, keep_dims1],
                          [x2, axis2, correction2, keep_dims2]],
            disable_mode=["GRAPH_MODE_GE"])
