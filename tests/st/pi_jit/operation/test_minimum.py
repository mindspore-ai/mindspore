import pytest
import numpy as np
import mindspore
from mindspore import Tensor
from ..share.ops.primitive.minimum_ops import MinimumFactory
from ..share.ops.primitive.minimum_ops import Minimum
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_minimum_input_512x1_512x1():
    """
    Feature: Ops.
    Description: Minimum算子测试，input_shape (512, 1), (512, 1).
    Expectation: expect correct result.
    """
    left_input = np.random.randn(512, 1).astype(np.float16)
    right_input = np.random.randn(512, 1).astype(np.float16)
    fact = MinimumFactory(left_input, right_input, np.float16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_minimum_input_2x2_2x2():
    """
    Feature: Ops.
    Description: Minimum算子测试，input_shape (2, 2), (2, 2), dtype=np.float32.
    Expectation: expect correct result.
    """
    left_input = np.random.randn(2, 2).astype(np.float32)
    right_input = np.random.randn(2, 2).astype(np.float32)
    MinimumFactory(left_input, right_input, np.float32)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_minimum_input_3x3x3x3_3x3x3x3():
    """
    Feature: Ops.
    Description: Minimum算子测试，input_shape (3, 3, 3, 3), (3, 3, 3, 3), dtype=np.float32.
    Expectation: expect correct result.
    """
    left_input = np.random.randn(3, 3, 3, 3).astype(np.float32)
    right_input = np.random.randn(3, 3, 3, 3).astype(np.float32)
    fact = MinimumFactory(left_input, right_input, np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_minimum_input_5d():
    """
    Feature: Ops.
    Description: Minimum算子测试，input_shape 5D &隐式类型转换.
    Expectation: expect correct result.
    """
    left_input = np.random.randn(3, 3, 3, 3, 5).astype(np.float16)
    right_input = np.random.randn(3, 3, 3, 3, 5).astype(np.float32)
    fact = MinimumFactory(left_input, right_input, np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_minimum_input_6d():
    """
    Feature: Ops.
    Description: Minimum算子测试，input_shape 6D.
    Expectation: expect correct result.
    """
    left_input = np.random.randn(3, 3, 3, 3, 5, 4).astype(np.float32)
    right_input = np.random.randn(3, 3, 3, 3, 5, 4).astype(np.float32)
    fact = MinimumFactory(left_input, right_input, np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_minimum_input_7d():
    """
    Feature: Ops.
    Description: Minimum算子测试，input_shape 7D.
    Expectation: expect correct result.
    """
    left_input = np.random.randn(3, 3, 3, 3, 5, 4, 3).astype(np.float32)
    right_input = np.random.randn(3, 3, 3, 3, 5, 4, 3).astype(np.float32)
    fact = MinimumFactory(left_input, right_input, np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_minimum_input_3dtensor_scalar():
    """
    Feature: Ops.
    Description: Minimum算子测试，left_input 3dtensor, right_input scalar, dtype=np.float32.
    Expectation: expect correct result.
    """
    left_input = np.random.randn(128, 128, 64).astype(np.float32)
    right_input = np.array(3.2).astype(np.float32)
    fact = MinimumFactory(left_input, right_input, np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_minimum_input_scalar_3dtensor():
    """
    Feature: Ops.
    Description: Minimum算子测试，left_input scalar, right_input 3dtensor, dtype=np.float32.
    Expectation: expect correct result.
    """
    left_input = np.array(3.2).astype(np.float32)
    right_input = np.random.randn(128, 128, 64).astype(np.float32)
    fact = MinimumFactory(left_input, right_input, np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_minimum_forward_input_2dtensor_3dtensor_int32():
    """
    Feature: Ops.
    Description: Minimum算子异常场景测试，left_input 2dtensor, right_input 3dtensor, dtype=np.int32.
    Expectation: expect correct result.
    """
    left_input = np.random.randn(3, 3).astype(np.int32)
    right_input = np.random.randn(1, 3, 2).astype(np.int32)
    fact = MinimumFactory(left_input, right_input, grad=right_input)
    with pytest.raises(ValueError):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_minimum_dtype_int64():
    """
    Feature: Ops.
    Description: minimum, dtype=int64.
    Expectation: expect correct result.
    """
    left_input = np.random.randint(0, 25, size=(3, 3)).astype(np.int64)
    right_input = np.random.randint(0, 25, size=(3, 3)).astype(np.int64)
    fact = MinimumFactory(left_input, right_input, dtype=np.int64)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_minimum_input_bool_tensor_int32():
    """
    Feature: Ops.
    Description: minimum, bool & Tensor(int32).
    Expectation: expect correct result.
    """
    left_input_np = False
    right_input_np = np.array([-1, 0, 1])
    net = Minimum()
    out_me = net(left_input_np, Tensor(right_input_np, mindspore.int32))
    out_np = np.minimum(left_input_np, right_input_np)
    assert out_me.asnumpy().all() == out_np.all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_minimum_dtype_int8():
    """
    Feature: Ops.
    Description: minimum, dtype=int8.
    Expectation: expect correct result.
    """
    left_input = np.random.randint(0, 25, size=(3, 3)).astype(np.int8)
    right_input = np.random.randint(0, 25, size=(3, 3)).astype(np.int8)
    fact = MinimumFactory(left_input, right_input, dtype=np.int8)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_minimum_dtype_int16():
    """
    Feature: Ops.
    Description: minimum, dtype=int16.
    Expectation: expect correct result.
    """
    left_input = np.random.randint(0, 25, size=(3, 3)).astype(np.int16)
    right_input = np.random.randint(0, 25, size=(3, 3)).astype(np.int16)
    fact = MinimumFactory(left_input, right_input, dtype=np.int16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_minimum_dtype_uint16():
    """
    Feature: Ops.
    Description: minimum, dtype=uint16.
    Expectation: expect correct result.
    """
    left_input = np.random.randint(0, 25, size=(3, 3)).astype(np.uint16)
    right_input = np.random.randint(0, 25, size=(3, 3)).astype(np.uint16)
    fact = MinimumFactory(left_input, right_input, dtype=np.uint16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_minimum_dtype_uint8():
    """
    Feature: Ops.
    Description: minimum, dtype=uint8.
    Expectation: expect correct result.
    """
    left_input = np.random.randint(0, 25, size=(3, 3)).astype(np.uint8)
    right_input = np.random.randint(0, 25, size=(3, 3)).astype(np.uint8)
    fact = MinimumFactory(left_input, right_input, dtype=np.uint8)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_minimum_dtype_int32_tensor_interface():
    """
    Feature: Ops.
    Description: minimum, tensor interface, int32.
    Expectation: expect correct result.
    """
    left_input = np.random.randint(0, 25, size=(3, 3)).astype(np.int32)
    right_input = np.random.randint(0, 25, size=(3, 3)).astype(np.int32)
    out_np = np.minimum(left_input, right_input).astype(np.int32)
    output = Tensor(left_input).minimum(Tensor(right_input))
    assert output.asnumpy().all() == out_np.all()
