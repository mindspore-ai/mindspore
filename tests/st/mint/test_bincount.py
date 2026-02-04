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
from mindspore import mint, Tensor
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils

def generate_random_input(num, shape, dtype):
    return np.random.randint(num, size=shape).astype(dtype)

def generate_random_weight(shape, dtype):
    return np.random.randn(shape).astype(dtype)

@test_utils.run_with_cell
def bincount_forward_func(x, weights=None, minlength=0):
    return mint.bincount(x, weights, minlength)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_mint_bincount_ext_normal(mode):
    """
    Feature: mint.bincount
    Description: Verify the result of mint.bincount
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode)
    x = Tensor([2, 4, 1, 0, 0], ms.int64)
    weights = Tensor([0, 0.25, 0.5, 0.75, 1], ms.float32)
    expect_output = Tensor([1.75, 0.5, 0, 0, 0.25], ms.float32)
    output = bincount_forward_func(x, weights, minlength=5)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
def test_mint_bincount_ext_dynamic():
    """
    Feature: Test bincount with dynamic shape in graph mode using TEST_OP.
    Description: call mint.bincount with valid input and index.
    Expectation: return the correct value.
    """
    minlength = np.random.randint(1, 11)
    x1 = generate_random_input(minlength, 7, np.int32)
    x2 = generate_random_input(minlength, 7, np.int32)
    x3 = generate_random_weight(7, np.float32)
    TEST_OP(bincount_forward_func,
            [[Tensor(x1), Tensor(x3), minlength],
             [Tensor(x2), Tensor(x3), minlength]],
            disable_mode=["GRAPH_MODE_GE"],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_grad': True,
                         'deterministic_use_origin_inputs': True})
