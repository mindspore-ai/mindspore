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
from mindspore import mint
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(x, mat, vec, beta=1, alpha=1):
    return beta * x + alpha * (mat @ vec)

@test_utils.run_with_cell
def addmv_forward_func(x, mat, vec, beta=1, alpha=1):
    return mint.addmv(x, mat, vec, beta=beta, alpha=alpha)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_addmv_normal(mode):
    """
    Feature: mint.addmv
    Description: Verify the result of mint.addmv on platform_ascend910b
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.context.set_context(jit_level='O0')
    x = generate_random_input((6,), np.float32)
    mat = generate_random_input((6, 3), np.float32)
    vec = generate_random_input((3,), np.float32)
    beta = 1.0
    alpha = 1.0

    expect = generate_expect_forward_output(x, mat, vec, beta=beta, alpha=alpha)
    output = addmv_forward_func(ms.Tensor(x), ms.Tensor(mat), ms.Tensor(vec), beta=beta, alpha=alpha)

    assert np.allclose(output.asnumpy(), expect, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
def test_mint_addmv_dynamic():
    """
    Feature: Test addmv with dynamic shape in graph mode using TEST_OP.
    Description: call mint.addmv with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((5,), np.float32)
    mat1 = generate_random_input((5, 2), np.float32)
    vec1 = generate_random_input((2,), np.float32)

    x2 = generate_random_input((2,), np.float32)
    mat2 = generate_random_input((2, 3), np.float32)
    vec2 = generate_random_input((3,), np.float32)
    beta = 1.0
    alpha = 1.0

    TEST_OP(addmv_forward_func,
            [[ms.Tensor(x1), ms.Tensor(mat1), ms.Tensor(vec1), beta, alpha],
             [ms.Tensor(x2), ms.Tensor(mat2), ms.Tensor(vec2), beta, alpha]],
            disable_mode=["GRAPH_MODE_GE"],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True})
