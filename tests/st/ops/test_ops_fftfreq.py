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
from mindspore import ops, jit, JitConfig
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


@test_utils.run_with_cell
def fftfreq_forward_func(n, d=1.0):
    return ops.fftfreq(n, d)

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(n, d=1.0):
    return np.fft.fftfreq(n, d)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', ['GE', 'KBK', 'pynative'])
def test_ops_fftfreq_forward(mode):
    """
    Feature: ops.fftfreq
    Description: test function fftfreq forward.
    Expectation: success
    """
    n = 6
    d = 2.0
    if mode == 'pynative':
        output = fftfreq_forward_func(n, d)
    elif mode == 'KBK':
        output = (jit(fftfreq_forward_func, jit_config=JitConfig(jit_level="O0")))(n, d)
    else:
        output = (jit(fftfreq_forward_func, jit_config=JitConfig(jit_level="O2")))(n, d)
    expect = generate_expect_forward_output(n, d)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('jit_level', ["O0", "O2"])
def test_ops_fftfreq_forward_dynamic(jit_level):
    """
    Feature: ops.fftfreq
    Description: test function fftfreq forward with dynamic input.
    Expectation: success
    """
    n1 = 4
    d1 = 1.0
    n2 = 7
    d2 = 3.3

    inputs1 = [n1, d1]
    inputs2 = [n2, d2]

    TEST_OP(fftfreq_forward_func, [inputs1, inputs2],
            grad=False, jit_level=jit_level)
