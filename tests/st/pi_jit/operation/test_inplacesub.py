import numpy as np
from ..share.ops.primitive.inplacesub_ops import InplaceSubFactory
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_inplacesub_input_1d():
    """
    Feature: Ops.
    Description: test operator InplaceSub, given (inputx_shape=1d).
    Expectation: expect correct result.
    """
    fact = InplaceSubFactory(input_shape=(16,), target_shape=(4,),
                             indices=(0, 1, 2, 3), dtype=np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_inplacesub_indices_repeat():
    """
    Feature: Ops.
    Description: test operator InplaceSub, given v repeat.
    Expectation: expect correct result.
    """
    fact = InplaceSubFactory(input_shape=(16, 8, 8, 4, 4),
                             target_shape=(2, 8, 8, 4, 4), indices=(1, 1),
                             dtype=np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_inplacesub_input_32_8_128_ind_28_float32():
    """
    Feature: Ops.
    Description: test operator InplaceSub, given (input_shape=(32,8,128),
                 indices= 28,target_shape=(3,8,128),dtype=np.float32).
    Expectation: expect correct result.
    """
    fact = InplaceSubFactory(input_shape=(32, 8, 128), indices=28,
                             target_shape=(1, 8, 128), dtype=np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_inplacesub_input_1d_float16():
    """
    Feature: Ops.
    Description: test InplaceSub with 1D input, x_dtype=Float32.
    Expectation: expect correct result.
    """
    fact = InplaceSubFactory(input_shape=(3,), indices=(2, 1, 0),
                             target_shape=(3,),
                             dtype=np.float16)
    fact.forward_dynamic_shape_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_inplacesub_input_2d_float32():
    """
    Feature: Ops.
    Description: test InplaceSub with 2D input, x_dtype=Float32.
    Expectation: expect correct result.
    """
    fact = InplaceSubFactory(input_shape=(8, 16), target_shape=(2, 16),
                             indices=(2, 1), dtype=np.float32)
    fact.forward_dynamic_shape_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_inplacesub_input_3d_float64():
    """
    Feature: Ops.
    Description: test InplaceSub with 3D input, x_dtype=Float64.
    Expectation: expect correct result.
    """
    fact = InplaceSubFactory(input_shape=(6, 200, 200), target_shape=(1, 200, 200),
                             indices=(4,), dtype=np.float64)
    fact.forward_dynamic_shape_cmp()
