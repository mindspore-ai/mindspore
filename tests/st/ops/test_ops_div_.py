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
        input1.div_(other1)
        return input1

class Net2(nn.Cell):
    def construct(self, input2, other2, rounding_mode2):
        input2 = input2 * 1
        other2 = other2 * 1
        input2.div_(other2, rounding_mode=rounding_mode2)
        return input2

class Net3(nn.Cell):
    def construct(self, input3, other3):
        input3 /= other3
        return input3

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def inplace_div_backward_func(input4, other4):
    grad = ops.GradOperation(get_all=True)
    return grad(Net1())(input4, other4)

def inplace_div_rounding_mode_backward_func(input5, other5, rounding_mode5):
    grad = ops.GradOperation(get_all=True)
    return grad(Net2())(input5, other5, rounding_mode5)

@test_utils.run_with_cell
def inplace_div_dyn_func(input6, other6):
    return Net1()(input6, other6)

@test_utils.run_with_cell
def inplace_div_rounding_mode_dyn_func(input7, other7, rounding_mode7):
    return Net2()(input7, other7, rounding_mode7)

@test_utils.run_with_cell
def inplace_div_char_forward_func(input8, other8):
    return Net3()(input8, other8)

@ops_binary_cases(OpsBinaryCase(input_info=[((6, 64, 88, 160), np.float32), ((6, 64, 88, 160), np.float32)],
                                output_info=[((6, 64, 88, 160), np.float32), ((6, 64, 88, 160), np.float32)],
                                extra_info='SD5B'))
def ops_div_binary_case1(input_binary_data=None, output_binary_data=None):
    input1 = Tensor(input_binary_data[0])
    input2 = Tensor(input_binary_data[1])
    output = inplace_div_backward_func(input1, input2)
    assert np.allclose(output[0].asnumpy(), output_binary_data[0], 1e-4, 1e-4)
    assert np.allclose(output[1].asnumpy(), output_binary_data[1], 1e-4, 1e-4)

@ops_binary_cases(OpsBinaryCase(input_info=[((6, 64, 88, 160), np.float32)],
                                output_info=[((6, 64, 88, 160), np.float32)],
                                extra_info='SD5B'))
def ops_div_binary_case2(input_binary_data=None, output_binary_data=None):
    output = inplace_div_backward_func(Tensor(input_binary_data[0]), 7)
    assert np.allclose(output[0].asnumpy(), output_binary_data, 1e-04, 1e-04)

@ops_binary_cases(OpsBinaryCase(input_info=[((7, 9, 8, 4), np.float32), ((7, 9, 8, 4), np.float32)],
                                output_info=[((7, 9, 8, 4), np.float32), ((7, 9, 8, 4), np.float32)],
                                extra_info='SD5B'))
def ops_div_binary_case3(input_binary_data=None, output_binary_data=None):
    input1 = Tensor(input_binary_data[0])
    input2 = Tensor(input_binary_data[1])
    output = inplace_div_rounding_mode_backward_func(input1, input2, 'floor')
    assert np.allclose(output[0].asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    assert np.allclose(output[1].asnumpy(), output_binary_data[1], 1e-04, 1e-04)

@ops_binary_cases(OpsBinaryCase(input_info=[((6, 8, 3, 2), np.float32)],
                                output_info=[((6, 8, 3, 2), np.float32)],
                                extra_info='SD5B'))
def ops_div_binary_case4(input_binary_data=None, output_binary_data=None):
    output = inplace_div_rounding_mode_backward_func(Tensor(input_binary_data[0]), 9, 'floor')
    assert np.allclose(output[0].asnumpy(), output_binary_data, 1e-04, 1e-04)

@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=166.0)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_inplace_div_backward(mode):
    """
    Feature: tensor.divide()
    Description: Verify the result of tensor.divide
    Expectation: success
    """
    context.set_context(mode=mode)
    ops_div_binary_case1()
    ops_div_binary_case2()
    ops_div_binary_case3()
    ops_div_binary_case4()


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2.0)
@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
def test_ops_inplace_div_dynamic_shape(rounding_mode):
    """
    Feature: pyboost function.
    Description: test function l1_loss forward with dynamic shape.
    Expectation: expect correct result.
    """
    input1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    other1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))

    input2 = ms.Tensor(generate_random_input((8, 9), np.float32))
    other2 = ms.Tensor(generate_random_input((8, 9), np.float32))

    TEST_OP(inplace_div_dyn_func, [[input1, other1], [input2, other2]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            case_config={'disable_input_check': True, 'all_dim_zero': True},
            inplace_update=True)

    input3 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    other3 = 4

    input4 = ms.Tensor(generate_random_input((8, 9), np.float32))
    other4 = 3

    TEST_OP(inplace_div_dyn_func, [[input3, other3], [input4, other4]],
            case_config={'disable_input_check': True},
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            inplace_update=True)

    input5 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    other5 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))

    input6 = ms.Tensor(generate_random_input((8, 9), np.float32))
    other6 = ms.Tensor(generate_random_input((8, 9), np.float32))

    TEST_OP(inplace_div_rounding_mode_dyn_func, [[input5, other5, rounding_mode], [input6, other6, rounding_mode]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            case_config={'disable_input_check': True, 'all_dim_zero': True},
            inplace_update=True)

    input7 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    other7 = 8

    input8 = ms.Tensor(generate_random_input((8, 9), np.float32))
    other8 = 4

    TEST_OP(inplace_div_rounding_mode_dyn_func, [[input7, other7, rounding_mode], [input8, other8, rounding_mode]],
            case_config={'disable_input_check': True}, disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            inplace_update=True)

@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inplace_div_char(mode):
    """
    Feature: tensor.divide()
    Description: Verify the result of tensor.divide
    Expectation: success
    """
    context.set_context(mode=mode)
    input1 = ms.Tensor([1, 3, 5, 7], dtype=ms.float32)
    input2 = ms.Tensor([2, 2, 2, 2], dtype=ms.float32)
    output = ms.Tensor([0.5, 1.5, 2.5, 3.5])
    output_np = output.asnumpy()
    output = inplace_div_char_forward_func(input1, input2)
    assert np.allclose(output.asnumpy(), output_np, 1e-04, 1e-04)

@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.0)
def test_inplace_div_char_GE():
    """
    Feature: tensor.divide()
    Description: Verify the result of tensor.divide
    Expectation: success
    """
    input1 = ms.Tensor([1, 3, 5, 7], dtype=ms.float32)
    input2 = ms.Tensor([2, 2, 2, 2], dtype=ms.float32)
    output = ms.Tensor([0.5, 1.5, 2.5, 3.5])
    output_np = output.asnumpy()

    context.set_context(mode=ms.GRAPH_MODE, jit_level='O2', device_target="Ascend")
    output = inplace_div_char_forward_func(input1, input2)
    assert np.allclose(output.asnumpy(), output_np, 1e-04, 1e-04)
