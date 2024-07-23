import numpy as np
import pytest
from ..share.ops.primitive.less_ops import LessFactory
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_less_forward_input_1d_all_float16():
    """
    Feature: Ops.
    Description: less正向用例：input=1d，left_type=float16,right_type=float16.
    Expectation: expect correct result.
    """
    left_input = np.random.randn(*(2455,)).astype(np.float16)
    right_input = np.random.randn(*(2455,)).astype(np.float16)
    fact = LessFactory(left_input, right_input)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_less_forward_input_2d_all_float32():
    """
    Feature: Ops.
    Description: less正向用例：input=2d，left_type=float32,right_type=float32.
    Expectation: expect correct result.
    """
    left_input = np.random.randn(*(128, 8)).astype(np.float32)
    right_input = np.random.randn(*(128, 1)).astype(np.float32)
    fact = LessFactory(left_input, right_input)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_less_forward_input_3d_all_int32():
    """
    Feature: Ops.
    Description: less正向用例：input=3d，left_type=int32,right_type=int32.
    Expectation: expect correct result.
    """
    left_input = np.random.randint(-2147483648, 2147483647, (32, 16, 128)).astype(np.int32)
    right_input = np.random.randint(-2147483648, 2147483647, (16, 128)).astype(np.int32)
    fact = LessFactory(left_input, right_input)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_less_forward_input_4d_all_int8():
    """
    Feature: Ops.
    Description: input=4d，left_type=int8,right_type=int8.
    Expectation: expect correct result.
    """
    left_input = np.random.randint(-128, 127, (2, 16, 8, 16)).astype(np.int8)
    right_input = np.random.randint(-128, 127, (2, 16, 8, 16)).astype(np.int8)
    fact = LessFactory(left_input, right_input)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_less_forward_input_5d_uint8():
    """
    Feature: Ops.
    Description: input=5d，left_type=uint8,right_type=uint8.
    Expectation: expect correct result.
    """
    left_input = np.random.randint(-256, 255, (2, 16, 8, 16, 12)).astype(np.uint8)
    right_input = np.random.randint(-256, 255, (1, 1, 8, 16, 12)).astype(np.uint8)
    fact = LessFactory(left_input, right_input)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_less_forward_input_6d_float32_float16():
    """
    Feature: Ops.
    Description: less正向用例：input=6d，left_type=float32,right_type=float64.
    Expectation: expect correct result.
    """
    left_input = np.random.randn(*(11, 10, 8, 4, 16, 32)).astype(np.float64)
    right_input = np.random.randn(*(11, 10, 8, 4, 16, 32)).astype(np.float64)
    fact = LessFactory(left_input, right_input)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_less_forward_input_7d_float16_int32():
    """
    Feature: Ops.
    Description: less正向用例：input=7d，left_type=float16,right_type=int16.
    Expectation: expect correct result.
    """
    left_input = np.random.randn(*(8, 16, 4, 2, 1, 32, 9)).astype(np.int16)
    right_input = np.random.randint(-128, 128, (8, 16, 4, 2, 1, 1, 1)).astype(np.int16)
    fact = LessFactory(left_input, right_input)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_less_forward_input_1d_left_float32_right_bool():
    """
    Feature: Ops.
    Description: less正向用例：input=1d，left_type=float32,right_type=bool,测试隐式转换.
    Expectation: expect correct result.
    """
    left_input = np.random.randn(*(128,)).astype(np.float32)
    right_input = np.random.randn(*(128,)).astype(np.bool_)
    fact = LessFactory(left_input, right_input)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_less_forward_input_right_float():
    """
    Feature: Ops.
    Description: less正向用例：input=3d，left_type=int32,right_type=float,测试一个参数是float.
    Expectation: expect correct result.
    """
    left_input = np.random.randint(-1024, 1024, (128, 4, 1)).astype(np.int32)
    right_input = 0.56
    fact = LessFactory(left_input, right_input, rightistensor=False)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_less_forward_input_left_int():
    """
    Feature: Ops.
    Description: less正向用例：input=4d，left_type=5,right_type=float32,测试一个参数是int.
    Expectation: expect correct result.
    """
    left_input = 5
    right_input = np.random.randint(-128, 128, (8, 4, 2, 16)).astype(np.float32)
    fact = LessFactory(left_input, right_input, leftistensor=False)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_less_forward_input_left_bool():
    """
    Feature: Ops.
    Description: less正向用例：input=2d，left_type=bool,right_type=float16,测试一个参数是bool.
    Expectation: expect correct result.
    """
    left_input = True
    right_input = np.random.randint(-128, 128, (8, 16)).astype(np.float16)
    fact = LessFactory(left_input, right_input, leftistensor=False)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_less_forward_input_right_bool():
    """
    Feature: Ops.
    Description: less正向用例：input=5d，left_type=float32,right_type=bool,测试right_input=bool.
    Expectation: expect correct result.
    """
    left_input = np.random.randn(*(16, 8, 1, 1, 2)).astype(np.float32)
    right_input = False
    fact = LessFactory(left_input, right_input, rightistensor=False)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_less_input_list():
    """
    Feature: Ops.
    Description: less异常用例：参数为列表.
    Expectation: expect correct result.
    """
    left_input = [1, 2, 3, 4]
    right_input = np.random.randn(*(4,)).astype(np.float32)
    fact = LessFactory(left_input, right_input, leftistensor=False)
    with pytest.raises(TypeError):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_less_input_trulp():
    """
    Feature: Ops.
    Description: less异常用例：参数为元组.
    Expectation: expect correct result.
    """
    left_input = np.random.randn(*(4,)).astype(np.float16)
    right_input = (1, 2, 3, 4)
    fact = LessFactory(left_input, right_input, rightistensor=False)
    with pytest.raises(TypeError):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_less_input_all_number():
    """
    Feature: Ops.
    Description: less异常用例：参数都是数字.
    Expectation: expect correct result.
    """
    left_input = 8
    right_input = 6
    fact = LessFactory(left_input, right_input, leftistensor=False, rightistensor=False)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_less_input_all_bool():
    """
    Feature: Ops.
    Description: less异常用例：参数都是bool.
    Expectation: expect correct result.
    """
    left_input = True
    right_input = False
    fact = LessFactory(left_input, right_input, leftistensor=False, rightistensor=False)
    fact.forward_cmp()
