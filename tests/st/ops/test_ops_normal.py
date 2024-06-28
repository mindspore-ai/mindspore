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
from mindspore import mint, Generator
from mindspore.ops.auto_generate import NormalTensorTensor, NormalTensorFloat,\
     NormalFloatTensor, NormalFloatFloat
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
generator = Generator()
seed_ = ms.Tensor(1, ms.int64)
offset_ = ms.Tensor(1, ms.int64)
seed2_ = ms.Tensor(2, ms.int64)
offset2_ = ms.Tensor(2, ms.int64)

normal_tensor_tensor_op = NormalTensorTensor()
normal_tensor_float_op = NormalTensorFloat()
normal_float_tensor_op = NormalFloatTensor()
normal_float_float_op = NormalFloatFloat()

def generate_random_input(shape):
    return np.random.randn(*shape).astype(np.float32)


def generate_expect_backward_output():
    return 0


@test_utils.run_with_cell
def normal_tensor_tensor_forward_func(mean, std, seed, offset):
    return normal_tensor_tensor_op(mean, std, seed, offset)


@test_utils.run_with_cell
def normal_tensor_float_forward_func(mean, std, seed, offset):
    return normal_tensor_float_op(mean, std, seed, offset)


@test_utils.run_with_cell
def normal_float_tensor_forward_func(mean, std, seed, offset):
    return normal_float_tensor_op(mean, std, seed, offset)


@test_utils.run_with_cell
def normal_float_float_forward_func(mean, std, size, seed, offset):
    return normal_float_float_op(mean, std, size, seed, offset)


@test_utils.run_with_cell
def normal_backward_func(mean, std, seed, offset):
    return ms.ops.grad(normal_tensor_tensor_forward_func, (0))(mean, std, seed, offset)


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
    output = normal_backward_func(
        ms.Tensor(mean), ms.Tensor(std), seed_, offset_)
    expect = generate_expect_backward_output()
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_normal_tensor_tensor_forward():
    """
    Feature: pyboost function.
    Description: test function NormalTensorTensor forward.
    Expectation: expect correct result.
    """
    mean = ms.Tensor(generate_random_input((10, 10)))
    std = ms.Tensor(generate_random_input((10, 10)))
    output = normal_tensor_tensor_forward_func(mean, std, seed_, offset_)
    assert output.shape == (10, 10)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_normal_tensor_float_forward():
    """
    Feature: pyboost function.
    Description: test function NormalTensorTensor forward.
    Expectation: expect correct result.
    """
    mean = ms.Tensor(generate_random_input((10, 10)))
    std = 1.0
    output = normal_tensor_float_forward_func(mean, std, seed_, offset_)
    assert output.shape == (10, 10)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_normal_float_tensor_forward():
    """
    Feature: pyboost function.
    Description: test function NormalTensorTensor forward.
    Expectation: expect correct result.
    """
    mean = 1.0
    std = ms.Tensor(generate_random_input((10, 10)))
    output = normal_float_tensor_forward_func(mean, std, seed_, offset_)
    assert output.shape == (10, 10)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_normal_float_float_forward():
    """
    Feature: pyboost function.
    Description: test function NormalTensorTensor forward.
    Expectation: expect correct result.
    """
    mean = 1.0
    std = 1.0
    size = (10, 10)
    output = normal_float_float_forward_func(mean, std, size, seed_, offset_)
    assert output.shape == (10, 10)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_normal_tensor_tensor_dynamic_shape_testop():
    """
    Feature: Test NormalTensorTensor with dynamic shape in graph mode using TEST_OP.
    Description: call NormalTensorTensor with valid input.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((10, 10))
    x2 = generate_random_input((10, 10))
    TEST_OP(normal_tensor_tensor_op,
            [[ms.Tensor(x1), ms.Tensor(x1), seed_, offset_],
             [ms.Tensor(x2), ms.Tensor(x2), seed2_, offset2_]], 'normal_tensor_tensor',
            disable_input_check=True, disable_mode=['GRAPH_MODE'], inplace_update=True)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_normal_tensor_float_dynamic_shape_testop():
    """
    Feature: Test NormalTensorFloat with dynamic shape in graph mode using TEST_OP.
    Description: call NormalTensorFloat with valid input.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((10, 10))
    x2 = generate_random_input((10, 10))
    TEST_OP(normal_tensor_float_op,
            [[ms.Tensor(x1), 1.0, seed_, offset_],
             [ms.Tensor(x2), 1.0, seed2_, offset2_]], 'normal_tensor_float',
            disable_input_check=True, disable_mode=['GRAPH_MODE'], inplace_update=True)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_normal_float_tensor_dynamic_shape_testop():
    """
    Feature: Test NormalFloatTensor with dynamic shape in graph mode using TEST_OP.
    Description: call NormalFloatTensor with valid input.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((10, 10))
    x2 = generate_random_input((10, 10))
    TEST_OP(normal_float_tensor_op,
            [[1.0, ms.Tensor(x1), seed_, offset_],
             [1.0, ms.Tensor(x2), seed2_, offset2_]], 'normal_float_tensor',
            disable_input_check=True, disable_mode=['GRAPH_MODE'], inplace_update=True)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_normal_float_float_dynamic_shape_testop():
    """
    Feature: Test NormalFloatFloat with dynamic shape in graph mode using TEST_OP.
    Description: call NormalFloatFloat with valid input.
    Expectation: return the correct value.
    """
    TEST_OP(normal_float_float_op,
            [[1.0, 1.0, (2, 2), seed_, offset_],
             [2.0, 2.0, (2, 2), seed2_, offset2_]], 'normal_float_float',
            disable_input_check=True, disable_mode=['GRAPH_MODE'], inplace_update=True)



@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_mint_normal_func1():
    """
    Feature: Test mint.normal.
    Description: call mint.normal with valid input.
    Expectation: return the correct value.
    """
    output1 = mint.normal(1.0, 1.0, (2, 2))
    output2 = mint.normal(1.0, 1.0, (2, 2))
    assert not np.all(output1.asnumpy() == output2.asnumpy())

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_mint_normal_func2():
    """
    Feature: Test mint.normal.
    Description: call mint.normal with valid input.
    Expectation: return the correct value.
    """
    state = generator.get_state()
    output1 = mint.normal(1.0, 1.0, (2, 2), generator)
    generator.set_state(state)
    output2 = mint.normal(1.0, 1.0, (2, 2), generator)
    assert np.all(output1.asnumpy() == output2.asnumpy())


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_mint_normal_func3():
    """
    Feature: Test mint.normal.
    Description: call mint.normal with valid input.
    Expectation: return the correct value.
    """
    state = ms.get_rng_state()
    output1 = mint.normal(1.0, 1.0, (2, 2))
    ms.set_rng_state(state)
    output2 = mint.normal(1.0, 1.0, (2, 2))
    assert np.all(output1.asnumpy() == output2.asnumpy())
