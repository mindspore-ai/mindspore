import numpy as np
import pytest
from ..share.ops.primitive.dense_ops import DenseFactory
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dense_input_102x44_in_44_out_32_2d_fp32():
    '''
    Description:
        test operator dense  input_shape is 2d , dtype is np.float32

    Expectation:
        pijit result match psjit

    '''
    fact = DenseFactory(input_shape=(102, 44), in_channel=44, out_channel=32, dtype=np.float32)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dense_input_1_in_248_out_100_1d_fp16():
    '''
    Description:
        test operator dense input_shapeis 1d, dtype is np.float16

    Expectation:
        pijit result match psjit

    '''
    fact = DenseFactory(input_shape=(248,), in_channel=248, out_channel=100, dtype=np.float16)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dense_input_128_496_out_124_6d_int32():
    '''
    Description:
        test operator dense input_shape 6d

    Expectation:
        pijit result match psjit

    '''
    fact = DenseFactory(input_shape=(1, 2, 4, 5, 28, 496), in_channel=496, out_channel=124,
                        dtype=np.int32)
    fact.b_np = None
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dense_input_out_2_7d():
    '''
    Description:
        test operator dense  input_shape=7d, dtype is np.int64

    Expectation:
        pijit result match psjit

    '''
    fact = DenseFactory(input_shape=(1, 3, 5, 6, 5, 10, 102), in_channel=102, out_channel=2,
                        dtype=np.int64)
    fact.b_np = np.random.randint(-10, 10)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dense_input_3d_in_3_out_4_uint8():
    '''
    Description:
        test operator dense  input_shape 3d, uint8

    Expectation:
        pijit result match psjit

    '''
    input_shape = (5, 2, 3)
    in_channel = 3
    out_channel = 4
    fact = DenseFactory(input_shape=input_shape, in_channel=in_channel, out_channel=out_channel,
                        dtype=np.uint8)
    fact.x_np = np.random.randint(0, 100, input_shape).astype(np.uint8)
    fact.w_np = np.random.randint(0, 100, (out_channel, in_channel)).astype(np.uint8)
    fact.b_np = np.random.randint(0, 100, out_channel).astype(np.uint8)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dense_input_inf():
    '''
    Description:
        test operator dense input_shape 3d, uint8

    Expectation:
        pijit result match psjit

    '''
    input_shape = (2, 2)
    in_channel = 2
    out_channel = 2
    fact = DenseFactory(input_shape=input_shape, in_channel=in_channel, out_channel=out_channel,
                        dtype=np.float32)
    fact.x_np = np.array([[2, 4], [np.inf, 2]]).astype(np.float32)
    fact.w_np = np.array([[2, 4], [np.inf, 2]]).astype(np.float32)
    fact.b_np = np.array([2, 4]).astype(np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dense_input_nan():
    '''
    Description:
        test operator dense with nan

    Expectation:
        pijit result match psjit

    '''
    input_shape = (2, 2)
    in_channel = 2
    out_channel = 2
    fact = DenseFactory(input_shape=input_shape, in_channel=in_channel, out_channel=out_channel,
                        dtype=np.float32)
    fact.x_np = np.array([[2, np.nan], [2, 2]]).astype(np.float32)
    fact.w_np = np.array([[2, 4], [np.inf, 2]]).astype(np.float32)
    fact.b_np = np.array([2, 4]).astype(np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dense_n_np_2d():
    '''
    Description:
        test operator dense  abnormal shape,b_np=2d

    Expectation:
        pijit result match psjit

    '''
    fact = DenseFactory(input_shape=(1, 1000), in_channel=1000, out_channel=1000, dtype=np.float32)
    fact.b_np = np.random.randn(1000, 1).astype(np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dense_x_1d_w_2d():
    '''
    Description:
        test operator dense  abnormal shape, input_shape is (10),w shape is 2d

    Expectation:
        pijit result match psjit

    '''
    fact = DenseFactory(input_shape=(10,), in_channel=10, out_channel=10, dtype=np.float32)
    fact.w_np = np.random.randn(10, 10).astype(np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dense_input_2d_w_3d():
    '''
    Description:
        test operator dense  abnormal shape,w_np is 3d

    Expectation:
        pijit result match psjit

    '''
    fact = DenseFactory(input_shape=(1, 1000), in_channel=1000, out_channel=1024, dtype=np.float32)
    fact.w_np = np.random.randn(1000, 1000, 1).astype(np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dense_input_1_1000_in_1000_out_1024_bias_1000_1_abnormal():
    '''
    Description:
        test operator dense abnormal shape

    Expectation:
        pijit result match psjit

    '''
    fact = DenseFactory(input_shape=(1, 1000), in_channel=1000, out_channel=1024, dtype=np.float32)
    fact.b_np = np.ones((1000, 1)).astype(np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dense_forward_input_type_not_same():
    '''
    Description:
        test operator dense abnormal dtypes.

    Expectation:
        pijit result match psjit

    '''
    fact = DenseFactory(input_shape=(28, 3), in_channel=3, out_channel=64, dtype=np.float32)
    fact.w_np = np.random.randn(64, 3).astype(np.float32)
    fact.b_np = np.random.randn(64).astype(np.float16)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_dense_input_56x28x16x28x3_in_4_out_3_abnormal():
    '''
    Description:
        test operator dense  input_shape=(56, 28, 16, 28, 3), in_channel=4, out_channel=3

    Expectation:
        pijit result match psjit

    '''
    fact = DenseFactory(input_shape=(56, 28, 16, 28, 3), in_channel=4, out_channel=3)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        fact.forward_cmp()
