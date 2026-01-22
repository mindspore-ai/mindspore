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
"""test ops normal"""
import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, Generator
from mindspore.graph.api import _pynative_executor
from mindspore.ops.auto_generate import NormalTensorTensor, NormalTensorFloat, \
    NormalFloatTensor, NormalFloatFloat
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark

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
    return ms.grad(normal_tensor_tensor_forward_func, (0))(mean, std, seed, offset)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_normal_tensor_tensor_dynamic_shape_testop():
    """
    Feature: Test NormalTensorTensor with dynamic shape in graph mode using TEST_OP.
    Description: call NormalTensorTensor with valid input.
    Expectation: return the correct value.
    """
    def normal_tensor_tensor(mean, std, seed, offset):
        return normal_tensor_tensor_op(mean, std, seed, offset)

    x1 = generate_random_input((10, 10))
    x2 = generate_random_input((10, 10))
    TEST_OP(normal_tensor_tensor,
            [[ms.Tensor(x1), ms.Tensor(x1), seed_, offset_],
             [ms.Tensor(x2), ms.Tensor(x2), seed2_, offset2_]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor'],
            case_config={'disable_input_check': True},
            inplace_update=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_normal_tensor_float_dynamic_shape_testop():
    """
    Feature: Test NormalTensorFloat with dynamic shape in graph mode using TEST_OP.
    Description: call NormalTensorFloat with valid input.
    Expectation: return the correct value.
    """
    def normal_tensor_float(mean, std, seed, offset):
        return normal_tensor_float_op(mean, std, seed, offset)

    x1 = generate_random_input((10, 10))
    x2 = generate_random_input((10, 10))
    TEST_OP(normal_tensor_float,
            [[ms.Tensor(x1), 1.0, seed_, offset_],
             [ms.Tensor(x2), 1.0, seed2_, offset2_]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_input_check': True},
            inplace_update=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_normal_float_tensor_dynamic_shape_testop():
    """
    Feature: Test NormalFloatTensor with dynamic shape in graph mode using TEST_OP.
    Description: call NormalFloatTensor with valid input.
    Expectation: return the correct value.
    """
    def normal_float_tensor(mean, std, seed, offset):
        return normal_float_tensor_op(mean, std, seed, offset)

    x1 = generate_random_input((10, 10))
    x2 = generate_random_input((10, 10))
    TEST_OP(normal_float_tensor,
            [[1.0, ms.Tensor(x1), seed_, offset_],
             [1.0, ms.Tensor(x2), seed2_, offset2_]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor'],
            case_config={'disable_input_check': True},
            inplace_update=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_normal_float_float_dynamic_shape_testop():
    """
    Feature: Test NormalFloatFloat with dynamic shape in graph mode using TEST_OP.
    Description: call NormalFloatFloat with valid input.
    Expectation: return the correct value.
    """
    def normal_float_float(mean, std, size, seed, offset):
        return normal_float_float_op(mean, std, size, seed, offset)

    TEST_OP(normal_float_float,
            [[1.0, 1.0, (2, 2), seed_, offset_],
             [2.0, 2.0, (2, 2), seed2_, offset2_]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_input_check': True},
            inplace_update=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_mint_normal_func1():
    """
    Feature: Test mint.normal.
    Description: call mint.normal with valid input.
    Expectation: return the correct value.
    """
    output1 = mint.normal(1.0, 1.0, (2, 2))
    output2 = mint.normal(1.0, 1.0, (2, 2))
    assert not np.all(output1.asnumpy() == output2.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


def normal_test_func(mean, std, size=None, *, gradient_position=()):
    @test_utils.run_with_cell
    def normal_func(mean, std, size):
        return mint.normal(mean, std, size, generator=None)

    @test_utils.run_with_cell
    def normal_grad_func(mean, std, size, gradient_position):
        return ms.grad(normal_func, gradient_position)(mean, std, size)

    out = normal_func(mean, std, size)
    if gradient_position != ():
        grads = normal_grad_func(mean, std, size, gradient_position)
        return out, grads
    return out


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['kbk', 'pynative'])
def test_mint_normal_func4(mode):
    """
    Feature: Test mint.normal.
    Description: call mint.normal with valid input.
    Expectation: return the correct value.
    """
    if mode == 'kbk':
        ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)

    # std < 0
    invalid_std = [-3, -3.3]
    for std in invalid_std:
        mean = ms.Tensor(np.random.randn(2, 3))
        with pytest.raises(ValueError):
            _ = normal_test_func(mean, std, gradient_position=(0,))
            if mode == 'pynative':
                _pynative_executor.sync()

        mean = 0.5
        with pytest.raises(ValueError):
            _ = normal_test_func(mean, std, (2, 3))
            if mode == 'pynative':
                _pynative_executor.sync()

    with pytest.raises(TypeError):
        _ = normal_test_func(0, 0, 5) # size should be tuple or list
        if mode == 'pynative':
            _pynative_executor.sync()

    # mean is python number or std is python number
    python_numbers = [3, True, False, 0.3]
    sizes = [(2, 3), [2, 3]]
    for mean in python_numbers:
        for std in python_numbers:
            for size in sizes:
                out = normal_test_func(mean, std, size)
                assert out.shape == (2, 3)

    for mean in python_numbers:
        std = ms.Tensor(np.random.randn(2, 3))
        out, grad = normal_test_func(mean, std, gradient_position=(1,))
        assert out.shape == (2, 3)
        assert grad.shape == (2, 3)

    for std in python_numbers:
        mean = ms.Tensor(np.random.randn(2, 3))
        out, grad = normal_test_func(mean, std, gradient_position=(0,))
        assert out.shape == (2, 3)
        assert grad.shape == (2, 3)
