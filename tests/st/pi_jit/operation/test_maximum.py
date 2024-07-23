import numpy as np
import pytest
import mindspore
from ..share.ops.primitive.maximum_ops import MaximumFactory
from ..share.ops.primitive.maximum_ops import Maximum
from mindspore import Tensor
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maximum_input_512x1_512x1():
    """
    Feature: Ops.
    Description: maximum算子测试，input_shape (512, 1), (512, 1).
    Expectation: expect correct result.
    """
    left_input = np.random.randn(512, 1).astype(np.float16)
    right_input = np.random.randn(512, 1).astype(np.float16)
    fact = MaximumFactory(left_input, right_input, dtype=np.float16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maximum_input_2x2_2x2():
    """
    Feature: Ops.
    Description:maximum算子测试，input_shape (2, 2), (2, 2).
    Expectation: expect correct result.
    """
    left_input = np.random.randn(2, 2).astype(np.float32)
    right_input = np.random.randn(2, 2).astype(np.float32)
    fact = MaximumFactory(left_input, right_input)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maximum_input_3x3x3x3_3x3x3x3():
    """
    Feature: Ops.
    Description:maximum算子测试，input_shape (3, 3, 3, 3), (3, 3, 3, 3).
    Expectation: expect correct result.
    """
    left_input = np.random.randn(3, 3, 3, 3).astype(np.int8)
    right_input = np.random.randn(3, 3, 3, 3).astype(np.int8)
    fact = MaximumFactory(left_input, right_input)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maximum_input_5d():
    """
    Feature: Ops.
    Description:maximum算子测试，input_shape 5D & 隐式类型转换.
    Expectation: expect correct result.
    """
    left_input = np.random.randn(3, 3, 4, 5, 4).astype(np.float16)
    right_input = np.random.randn(3, 3, 4, 5, 4).astype(np.float32)
    fact = MaximumFactory(left_input, right_input)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maximum_input_6d():
    """
    Feature: Ops.
    Description:maximum算子测试，input_shape 6D.
    Expectation: expect correct result.
    """
    left_input = np.random.randn(3, 3, 4, 5, 4, 3).astype(np.uint8)
    right_input = np.random.randn(3, 3, 4, 5, 4, 3).astype(np.uint8)
    fact = MaximumFactory(left_input, right_input)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maximum_input_3dtensor_scalar_scalar():
    """
    Feature: Ops.
    Description:maximum算子测试，input_shape (128, 128, 64), array(3.2).
    Expectation: expect correct result.
    """
    left_input = np.random.randn(128, 128, 64).astype(np.float32)
    right_input = np.array(3.2).astype(np.float32)
    fact = MaximumFactory(left_input, right_input)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maximum_input_scalar_3dtensor_scalar():
    """
    Feature: Ops.
    Description:maximum算子测试，input_shape array(3.2), (128, 128, 64).
    Expectation: expect correct result.
    """
    left_input = np.array(3.2).astype(np.float32)
    right_input = np.random.randn(128, 128, 64).astype(np.float32)
    fact = MaximumFactory(left_input, right_input)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maximum_forward_input_1dtensor_2dtensor():
    """
    Feature: Ops.
    Description:maximum算子测试，input_shape (2), (2, 3).
    Expectation: expect correct result.
    """
    left_input = np.random.randn(2).astype(np.float32)
    right_input = np.random.randn(2, 3).astype(np.float32)
    fact = MaximumFactory(left_input, right_input)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maximum_forward_input_32x128x1024_1():
    """
    Feature: Ops.
    Description:maximum算子测试，input_shape (32, 128, 1024), (1).
    Expectation: expect correct result.
    """
    left_input = np.random.randn(32, 128, 1024).astype(np.float32)
    right_input = np.random.randn(1).astype(np.float32)
    fact = MaximumFactory(left_input, right_input)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maximum_input_tensor_bool_tensor_int32():
    """
    Feature: Ops.
    Description:maximum, Tensor(bool) & Tensor(int32).
    Expectation: expect correct result.
    """
    left_input_np = np.array([False, False, False])
    right_input_np = np.array([-1, 0, 1])
    net = Maximum()
    out_me = net(Tensor(left_input_np), Tensor(right_input_np, mindspore.int32))
    out_np = np.maximum(left_input_np, right_input_np)
    assert out_me.asnumpy().all() == out_np.all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maximum_input_randn_512x1_512x1_int32():
    """
    Feature: Ops.
    Description:maximum算子测试，input_shape (512, 1), (512, 1), dtype=int32.
    Expectation: expect correct result.
    """
    left_input = np.random.randint(0, 2560, size=(512, 1)).astype(np.int32)
    right_input = np.random.randint(0, 2560, size=(512, 1)).astype(np.int32)
    fact = MaximumFactory(left_input, right_input, dtype=np.int32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maximum_input_randint_8x1_8x1_int32():
    """
    Feature: Ops.
    Description:maximum算子测试，input_shape (8, 1), (8, 1), dtype=int32.
    Expectation: expect correct result.
    """
    left_input = np.random.randint(0, 256, size=(8, 1)).astype(np.int32)
    right_input = np.random.randint(0, 256, size=(8, 1)).astype(np.int32)
    fact = MaximumFactory(left_input, right_input, dtype=np.int32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maximum_performance_improve():
    """
    Feature: Ops.
    Description:test maximum performance,input >1w.
    Expectation: expect correct result.
    """
    input_x = np.random.random((8, 8, 64, 64)).astype(np.float32)
    input_y = np.random.random((8, 8, 64, 64)).astype(np.float32)
    fact = MaximumFactory(input_x, input_y)

    net = Maximum()
    inputs = [Tensor(fact.left_input), Tensor(fact.right_input)]
    for _ in range(50):
        net(*inputs)
