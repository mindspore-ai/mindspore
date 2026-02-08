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
from mindspore import ops, jit, JitConfig
from mindspore import context
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def diag_forward_func(input_x, diagonal=0):
    return input_x.diag(diagonal)


@test_utils.run_with_cell
def diag_backward_func(input_x, diagonal=0):
    return ops.grad(diag_forward_func, (0, 1))(input_x, diagonal)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_diag_forward(mode):
    """
    Feature: Ops.
    Description: test op diag.
    Expectation: expect correct result.
    """

    input_x = ms.Tensor(np.array([1, 2, 5]), ms.float16)
    expect = np.array([[1, 0, 0],
                       [0, 2, 0],
                       [0, 0, 5]]).astype(np.float16)
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        out = diag_forward_func(input_x)
        out_grad = diag_backward_func(input_x)
    else:
        out = (jit(diag_forward_func, backend="ms_backend", jit_config=JitConfig(jit_level="O0")))(
            input_x, 0)
        out_grad = (jit(diag_backward_func, backend="ms_backend", jit_config=JitConfig(jit_level="O0")))(
            input_x, 0)
    assert out_grad.asnumpy().dtype == np.float16
    np.testing.assert_allclose(out.asnumpy(), expect, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
def test_diag_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function div dynamic feature.
    Expectation: expect correct result.
    """
    input_x1 = ms.Tensor([[1.3, -2.1], [-4.7, 1.0]], ms.float32)
    input_x2 = ms.Tensor([1.3, 2.5, -2.1, 4.7, 2.5, 1.0], ms.float32)
    diagonal = 0
    TEST_OP(diag_forward_func, [[input_x1, diagonal], [input_x2, diagonal]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True},)
