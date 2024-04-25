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
from mindspore.ops.auto_generate import NormalExt
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP

normal_ext_op = NormalExt()

def generate_random_input(shape):
    return np.random.randn(*shape).astype(np.float32)

def generate_expect_backward_output():
    return 0

@test_utils.run_with_cell
def normal_dyn_shape_func(mean, std, seed, offset):
    return normal_ext_op(mean, std, seed, offset)

@test_utils.run_with_cell
def normal_forward_func(mean, std, seed, offset):
    return normal_ext_op(mean, std, seed, offset)

@test_utils.run_with_cell
def normal_backward_func(mean, std, seed, offset):
    return ms.ops.grad(normal_forward_func, (0))(mean, std, seed, offset)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_normal_forward():
    """
    Feature: pyboost function.
    Description: test function NormalExt forward.
    Expectation: expect correct result.
    """
    mean = ms.Tensor(generate_random_input((10, 10)))
    std = ms.Tensor(generate_random_input((10, 10)))
    seed = 1
    offset = 1
    output = normal_forward_func(mean, std, seed, offset)
    assert output.shape == (10, 10)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ops_normal_backward():
    """
    Feature: pyboost function.
    Description: test function normal backward.
    Expectation: expect correct result.
    """
    mean = generate_random_input((10, 10))
    std = generate_random_input((10, 10))
    output = normal_backward_func(ms.Tensor(mean), ms.Tensor(std), 1, 1)
    expect = generate_expect_backward_output()
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.parametrize('jit_level', ["O0", "O2"])
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_normal_dynamic_shape_testop(jit_level):
    """
    Feature: Test NormalExt with dynamic shape in graph mode using TEST_OP.
    Description: call NormalExt with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((10, 10))
    x2 = generate_random_input((10, 10))
    TEST_OP(normal_forward_func,
            [[ms.Tensor(x1), ms.Tensor(x1), 10, 10],
             [ms.Tensor(x2), ms.Tensor(x2), 10, 10]],
            grad=False, jit_level=jit_level)
