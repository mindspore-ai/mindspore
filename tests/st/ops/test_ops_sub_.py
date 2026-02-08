import numpy as np
import pytest
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase
from tests.mark_utils import arg_mark
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore import context
from mindspore import ops

class Net1(nn.Cell):
    def construct(self, input1, other1):
        input1 = input1 * 1
        other1 = other1 * 1
        input1.sub_(other1)
        return input1

class Net2(nn.Cell):
    def construct(self, input2, other2, alpha):
        input2 = input2 * 1
        other2 = other2 * 1
        input2.sub_(other2, alpha=alpha)
        return input2

class Net3(nn.Cell):
    def construct(self, input3, other3):
        input3 -= other3
        return input3

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def inplace_sub_backward_func(input4, other4):
    grad = ops.GradOperation(get_all=True)
    return grad(Net1())(input4, other4)

def inplace_sub_alpha_backward_func(input5, other5, alpha5):
    grad = ops.GradOperation(get_all=True)
    return grad(Net2())(input5, other5, alpha5)

@test_utils.run_with_cell
def inplace_sub_dyn_func(input6, other6):
    return Net1()(input6, other6)

@test_utils.run_with_cell
def inplace_sub_alpha_dyn_func(input7, other7, rounding_mode7):
    return Net2()(input7, other7, rounding_mode7)

@test_utils.run_with_cell
def inplace_sub_char_forward_func(input8, other8):
    return Net3()(input8, other8)

@ops_binary_cases(OpsBinaryCase(input_info=[((1, 2, 3, 4), np.float16), ((1, 2, 3, 4), np.float32)],
                                output_info=[((1, 2, 3, 4), np.float16), ((1, 2, 3, 4), np.float32)],
                                extra_info='SD5B'))
def ops_sub_binary_case1(input_binary_data=None, output_binary_data=None):
    input1 = Tensor(input_binary_data[0])
    input2 = Tensor(input_binary_data[1])
    output = inplace_sub_backward_func(input1, input2)
    assert np.allclose(output[0].asnumpy(), output_binary_data[0], 1e-4, 1e-4)
    assert np.allclose(output[1].asnumpy(), output_binary_data[1], 1e-4, 1e-4)

@ops_binary_cases(OpsBinaryCase(input_info=[((1, 2, 3, 4), np.float64), ((1, 2, 3, 4), np.float32)],
                                output_info=[((1, 2, 3, 4), np.float64), ((1, 2, 3, 4), np.float32)],
                                extra_info='SD5B'))
def ops_sub_binary_case2(input_binary_data=None, output_binary_data=None):
    input1 = Tensor(input_binary_data[0])
    input2 = Tensor(input_binary_data[1])
    output = inplace_sub_alpha_backward_func(input1, input2, 3.1)
    assert np.allclose(output[0].asnumpy(), output_binary_data[0], 1e-4, 1e-4)
    assert np.allclose(output[1].asnumpy(), output_binary_data[1], 1e-4, 1e-4)

@ops_binary_cases(OpsBinaryCase(input_info=[((1, 2, 3, 4), np.float16)],
                                output_info=[((1, 2, 3, 4), np.float16)],
                                extra_info='SD5B'))
def ops_sub_binary_case3(input_binary_data=None, output_binary_data=None):
    output = inplace_sub_backward_func(Tensor(input_binary_data[0]), 7)
    assert np.allclose(output[0].asnumpy(), output_binary_data, 1e-04, 1e-04)

@ops_binary_cases(OpsBinaryCase(input_info=[((1, 2, 3, 4), np.float64)],
                                output_info=[((1, 2, 3, 4), np.float64)],
                                extra_info='SD5B'))
def ops_sub_binary_case4(input_binary_data=None, output_binary_data=None):
    output = inplace_sub_alpha_backward_func(Tensor(input_binary_data[0]), 7, 2)
    assert np.allclose(output[0].asnumpy(), output_binary_data, 1e-04, 1e-04)

@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2.0)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_inplace_sub_backward(mode):
    """
    Feature: tensor.divide()
    Description: Verify the result of tensor.divide
    Expectation: success
    """
    context.set_context(mode=mode)
    ops_sub_binary_case1()
    ops_sub_binary_case2()
    ops_sub_binary_case3()
    ops_sub_binary_case4()


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2.0)
def test_ops_inplace_div_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function l1_loss forward with dynamic shape.
    Expectation: expect correct result.
    """
    input1 = ms.Tensor(generate_random_input((7, 8, 9), np.float16))
    other1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))

    input2 = ms.Tensor(generate_random_input((8, 9), np.float16))
    other2 = ms.Tensor(generate_random_input((8, 9), np.float32))

    TEST_OP(inplace_sub_dyn_func, [[input1, other1], [input2, other2]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True},
            inplace_update=True)

    input3 = ms.Tensor(generate_random_input((7, 8, 9), np.float64))
    other3 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    alpha3 = 2

    input4 = ms.Tensor(generate_random_input((8, 9), np.float64))
    other4 = ms.Tensor(generate_random_input((8, 9), np.float32))
    alpha4 = 3

    TEST_OP(inplace_sub_alpha_dyn_func, [[input3, other3, alpha3], [input4, other4, alpha4]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True},
            inplace_update=True)

    input5 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    other5 = 7.1

    input6 = ms.Tensor(generate_random_input((8, 9), np.float32))
    other6 = 8.2

    TEST_OP(inplace_sub_dyn_func, [[input5, other5], [input6, other6]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True},
            inplace_update=True)

    input7 = ms.Tensor(generate_random_input((7, 8, 9), np.float64))
    other7 = 8
    alpha7 = 9

    input8 = ms.Tensor(generate_random_input((8, 9), np.float64))
    other8 = 4
    alpha8 = 7

    TEST_OP(inplace_sub_alpha_dyn_func, [[input7, other7, alpha7], [input8, other8, alpha8]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True},
            inplace_update=True)

@arg_mark(plat_marks=['platform_ascend910b', 'cpu_linux', 'platform_gpu'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inplace_sub_char(mode):
    """
    Feature: tensor.divide()
    Description: Verify the result of tensor.divide
    Expectation: success
    """
    context.set_context(mode=mode)
    input1 = ms.Tensor([4, 2, 3, 7], dtype=ms.float32)
    input2 = ms.Tensor([1, 2, 3, 4], dtype=ms.float32)
    output = ms.Tensor([3, 0, 0, 3])
    output_np = output.asnumpy()
    output = inplace_sub_char_forward_func(input1, input2)
    assert np.allclose(output.asnumpy(), output_np, 1e-04, 1e-04)

@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.0)
def test_inplace_sub_char_GE():
    """
    Feature: tensor.divide()
    Description: Verify the result of tensor.divide
    Expectation: success
    """
    input1 = ms.Tensor([4, 2, 3, 7], dtype=ms.float32)
    input2 = ms.Tensor([1, 2, 3, 4], dtype=ms.float32)
    output = ms.Tensor([3, 0, 0, 3])
    output_np = output.asnumpy()

    context.set_context(mode=ms.GRAPH_MODE, jit_level='O2', device_target="Ascend")
    output = inplace_sub_char_forward_func(input1, input2)
    assert np.allclose(output.asnumpy(), output_np, 1e-04, 1e-04)
