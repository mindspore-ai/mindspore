from ..share.ops.primitive.floor_ops import FloorFactory
from mindspore import Tensor
import mindspore.ops.operations as op
import numpy as np
import pytest
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floor_input_1d_fp16():
    """
    Description:
        1. test faster_rcnn floor with input shape (512,)  forward grad.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = FloorFactory(input_shape=(512,), dtype=np.float16)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floor_input_2d_fp32():
    """
    Description:
        1. test faster_rcnn floor with input shape=2d  forward grad.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = FloorFactory(input_shape=(512, 7), dtype=np.float32)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floor_input_3d_fp16():
    """
    Description:
        1. test faster_rcnn floor with input shape=2d  forward grad.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = FloorFactory(input_shape=(256, 7, 2), dtype=np.float16)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floor_input_4d_fp32():
    """
    Description:
        1. test faster_rcnn floor with input shape=2d  forward grad.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = FloorFactory(input_shape=(20, 4, 2, 1), dtype=np.float32)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floor_input_5d_fp16():
    """
    Description:
        1. test faster_rcnn floor with input shape=2d  forward grad.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = FloorFactory(input_shape=(10, 5, 3, 4, 2), dtype=np.float16)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floor_input_6d_fp32():
    """
    Description:
        1. test faster_rcnn floor with input shape=6d  forward grad.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = FloorFactory(input_shape=(5, 7, 8, 4, 5, 8), dtype=np.float32)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floor_input_7d_fp16():
    """
    Description:
        1. test faster_rcnn floor with input shape=6d  forward grad.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    fact = FloorFactory(input_shape=(2, 6, 4, 2, 1, 4, 3), dtype=np.float16)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_floor_input_512_512_fp16():
    """
    Description:
        1. test  floor with two input shape (512,)  forward.

    Expectation:
        1. the network run ok
        2. the result is the same as psjit
    """
    input_np = np.random.randn(5,).astype(np.float16)
    with pytest.raises(TypeError):
        op.Floor(Tensor(input_np), Tensor(input_np))
