import numpy as np
import mindspore.ops.operations as op
from mindspore import Tensor
from mindspore.common import dtype
from ..share.ops.primitive.floordiv_ops import FloorDivFactory
from ..share.ops.primitive.floordiv_ops import FloorDivMock
from ..share.utils import get_empty_tensor
from ..dynamic_shape_operations.floordiv import FloorDivDynamicShapeFactory
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floordiv_forward_input_128x1024_fp32():
    """
    Description:
        1. test reid floordiv with input shape (128 * 1024, 1),float32.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = FloorDivFactory(input_shape=(128 * 1024, 1), dtype=np.float32)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floordiv_forward_input_3d_fp16():
    """
    Description:
        1. test reid floordiv with input =3D, dtype=float16.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = FloorDivFactory(input_shape=(302, 110, 10), dtype=np.float16)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floordiv_forward_input_shape_dtype_int8():
    """
    Description:
        1. test reid floordiv with input=1d,  dtype=int8.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = FloorDivFactory(input_shape=(7,), dtype=np.int8)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floordiv_forward_input_shape_dtype_int16():
    """
    Description:
        1. test reid floordiv with input=2d,  dtype=int16.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = FloorDivFactory(input_shape=(3, 3), dtype=np.int16)
    fact.right_input_np = np.random.uniform((1,)).astype(np.int8)
    fact.input_x2 = Tensor(fact.right_input_np)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floordiv_forward_input_shape_dtype_int64():
    """
    Description:
        1. test reid floordiv with input=3d,  dtype=int64.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = FloorDivFactory(input_shape=(7, 8, 10), dtype=np.int64)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floordiv_forward_input_shape_dtype_fp64():
    """
    Description:
        1. test reid floordiv with input=4d,  dtype=float64.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = FloorDivFactory(input_shape=(7, 8, 9, 10), dtype=np.float64)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floordiv_forward_input_shape_dtype_int32():
    """
    Description:
        1. test reid floordiv with input=5d,  dtype=int32.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = FloorDivFactory(input_shape=(7, 8, 9, 10, 11), dtype=np.int32)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floordiv_forward_input_shape_dtype_uint16():
    """
    Description:
        1. test reid floordiv with input=6d,  dtype=uint16.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = FloorDivFactory(input_shape=(2, 4, 3, 6, 3, 5), dtype=np.uint16)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floordiv_forward_input_shape_dtype_uint8():
    """
    Description:
        1. test reid floordiv with input=7d,  dtype=uint8.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = FloorDivFactory(input_shape=(1, 4, 6, 2, 3, 5, 7), dtype=np.uint8)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floordiv_forward_right_input_int():
    """
    Description:
        1. test  floordiv  with input1 shape (13, 8), dtype =float32 ,input2 = 5.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    left_input_np = np.random.randn(13, 8).astype(np.float32)
    right_input_np = 5
    net = op.FloorDiv()
    out = net(Tensor(left_input_np), right_input_np)
    assert "float32" in str(out.dtype).lower()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floordiv_forward_left_input_bool():
    """
    Description:
        1. test  floordiv  with input1 = True, dtype .

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    right_input_np = np.random.randn(5, 3).astype(np.float16)
    net = op.FloorDiv()
    out = net(True, Tensor(right_input_np))
    assert "float16" in str(out.dtype).lower()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floordiv_empty_tensor():
    """
    Description:
        1. test floordiv with get_empty_tensor().

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_x1 = get_empty_tensor()
    input_x2 = get_empty_tensor()
    fact = FloorDivMock(inputs=[input_x1, input_x2])
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floordiv_input_3d_int16():
    """
    Description:
        1. test floor_div with input shape 3D, type=int16.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    x1 = np.random.randint(1, 512, size=(4, 4, 4)).astype(np.int16)
    x2 = np.random.randint(1, 512, size=(4, 4, 4)).astype(np.int16)
    input_x1 = Tensor(x1)
    input_x2 = Tensor(x2)
    fact = FloorDivMock(inputs=[input_x1, input_x2])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_floordiv_input_2d_fp16():
    """
    Description:
        1. test floor_div dynamic shape with input shape 2D, type=float16.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    x = np.random.randn(2, 3)
    y = np.random.randn(2, 3)
    input_x = Tensor(x, dtype=dtype.float16)
    input_y = Tensor(y, dtype=dtype.float16)
    fact = FloorDivMock(inputs=[input_x, input_y])
    fact.forward_dynamic_shape_cmp()
    fact.grad_dynamic_shape_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_floordiv_input_1d_fp32():
    """
    Description:
        1. test floor_div dynamic shape with input shape 1D, type=float32.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    x1 = np.random.randn(2).astype(np.float32)
    x2 = np.random.randn(2).astype(np.float32)
    input_x1 = Tensor(x1)
    input_x2 = Tensor(x2)
    fact = FloorDivMock(inputs=[input_x1, input_x2])
    fact.forward_dynamic_shape_cmp()
    fact.grad_dynamic_shape_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_floordiv_params_double_2d():
    """
    Description:
        1. test floordiv with dynamic shape input, dtype=double, 2d.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_x = Tensor(np.random.rand(8, 5, 8, 5).astype(np.float64))
    input_y = Tensor(np.random.rand(8, 5, 8, 5).astype(np.float64))
    indices = Tensor(np.random.choice(4, 2, replace=False).astype(np.int32))
    fact = FloorDivDynamicShapeFactory([input_x, input_y, indices])
    fact.forward_cmp()

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_floordiv_params_float32_2d():
    """
    Description:
        1. test floordiv with dynamic shape input, dtype=float32, 2d.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_x = Tensor(np.random.rand(100, 10, 10).astype(np.float32))
    input_y = Tensor(np.random.rand(100, 10, 10).astype(np.float32))
    indices = Tensor(np.random.choice(3, 1, replace=False).astype(np.int32))
    fact = FloorDivDynamicShapeFactory([input_x, input_y, indices])
    fact.forward_cmp()
