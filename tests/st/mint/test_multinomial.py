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
# pylint: disable=unused-variable
import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, jit
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.uniform(0, 100, shape).astype(dtype)


def multinomial_forward_func(x, num_samples, replacement):
    return mint.multinomial(x, num_samples, replacement)


def multinomial_forward_dyn_func(x, num_samples, replacement):
    return mint.multinomial(x, num_samples, replacement).shape


@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_multinomial_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function multinomial.
    Expectation: expect correct result.
    """
    g_m = ms.Generator()
    g_m.manual_seed(5)

    x0 = generate_random_input((2,), np.float32)
    x1 = generate_random_input((2, 3), np.float32)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output0 = multinomial_forward_func(ms.Tensor(x0), 1, True)
        output1 = multinomial_forward_func(ms.Tensor(x0), 2, True)
        output2 = multinomial_forward_func(ms.Tensor(x1), 6, True)
    else:
        output0 = (jit(multinomial_forward_func, jit_level="O0"))(ms.Tensor(x0), 1, True)
        output1 = (jit(multinomial_forward_func, jit_level="O0"))(ms.Tensor(x0), 2, True)
        output2 = (jit(multinomial_forward_func, jit_level="O0"))(ms.Tensor(x1), 6, True)

    assert output0.asnumpy().shape == (1,)
    assert output1.asnumpy().shape == (2,)
    assert output2.asnumpy().shape == (2, 6)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_multinomial_dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: test multinomial forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_2 = ms.Tensor(generate_random_input((5,), np.float32))

    num_samples_1 = ms.mutable(2)
    num_samples_2 = ms.mutable(1)

    replacement_1 = ms.mutable(False)
    replacement_2 = ms.mutable(True)

    TEST_OP(multinomial_forward_dyn_func,
            [[tensor_1, num_samples_1, replacement_1], [tensor_2, num_samples_2, replacement_2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'],
            case_config={'disable_grad': True,
                         'deterministic_use_origin_inputs': True})


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_multinomial_bfloat16(mode):
    """
    Feature: test multinomial functional API.
    Description: testcase for multinomial functional API.
    Expectation: the result match with expected result.
    """
    x = generate_random_input((2, 3), np.float32)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = multinomial_forward_func(ms.Tensor(x, dtype=ms.bfloat16), 6, True)
    else:
        output = (jit(multinomial_forward_func, jit_level="O0"))(ms.Tensor(x, dtype=ms.bfloat16), 6, True)

    assert output.asnumpy().shape == (2, 6)
