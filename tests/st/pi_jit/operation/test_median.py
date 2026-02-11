import pytest 
import numpy as np
from mindspore import Tensor, jit, context
from ..share.ops.primitive.median_ops import MedianFactory
from ..share.ops.primitive.median_ops import Median
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_median_input_1d_fp32():
    """
    Feature: Ops.
    Description: median算子正向测试 input_shape=(10, ), dtype=fp32.
    Expectation: expect correct result.
    """
    fact = MedianFactory(input_shape=(10,), global_median=False, axis=0, keep_dims=True,
                         dtype=np.float32)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_median_input_4d_int16():
    """
    Feature: Ops.
    Description: median算子正向测试 input_shape=(10, 8, 3, 2), dtype=int16.
    Expectation: expect correct result.
    """
    fact = MedianFactory(input_shape=(10, 8, 3, 2), global_median=False, axis=0,
                         keep_dims=True, dtype=np.int16)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_median_input_3d_int32():
    """
    Feature: Ops.
    Description: median算子正向测试 input_shape=(10, 9, 3), dtype=int32.
    Expectation: expect correct result.
    """
    fact = MedianFactory(input_shape=(10, 9, 3), global_median=False, axis=1,
                         keep_dims=True, dtype=np.int32)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_median_input_5d_int64():
    """
    Feature: Ops.
    Description: median算子正向测试 input_shape=(12, 2, 3, 4, 2), dtype=int64.
    Expectation: expect correct result.
    """
    fact = MedianFactory(input_shape=(12, 2, 3, 4, 2), global_median=False, axis=0,
                         keep_dims=True, dtype=np.int64)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_median_input_6d_fp32():
    """
    Feature: Ops.
    Description: median算子正向测试 input_shape=(10, 9, 1, 2, 3, 4), dtype=fp32.
    Expectation: expect correct result.
    """
    fact = MedianFactory(input_shape=(10, 9, 1, 2, 3, 4), global_median=False,
                         axis=0, keep_dims=True)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_median_input_7d_fp64():
    """
    Feature: Ops.
    Description: median算子正向测试 input_shape=(10, 1, 2, 3, 9, 8, 7), dtype=fp64.
    Expectation: expect correct result.
    """
    fact = MedianFactory(input_shape=(10, 1, 2, 3, 9, 8, 7), global_median=False,
                         axis=3, keep_dims=True, dtype=np.float64)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_median_abnormal_axis_left_out_bound():
    """
    Feature: Ops.
    Description: median算子测试 异常场景，axis out left bound.
    Expectation: expect correct result.
    """
    fact = MedianFactory(input_shape=(2, 1, 6, 32), global_median=False, axis=-5, keep_dims=False)
    with pytest.raises(ValueError,
                       match=r"For primitive\[Median\], the axis must be in \[-4,4\), but got -5."):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_median_abnormal_axis_right_out_bound():
    """
    Feature: Ops.
    Description: median算子测试 异常场景，axis out right bound.
    Expectation: expect correct result.
    """
    fact = MedianFactory(input_shape=(2, 1, 6, 32, 1, 2), global_median=False,
                         axis=6, keep_dims=True)
    with pytest.raises(ValueError,
                       match=r"For primitive\[Median\], the axis must be in \[-6,6\), but got 6."):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_median_abnormal_axis_float():
    """
    Feature: Ops.
    Description: median算子测试 异常场景，axis is float.
    Expectation: expect correct result.
    """
    fact = MedianFactory(input_shape=(2, 1, 6, 32), global_median=False, axis=1.2, keep_dims=True)
    with pytest.raises(TypeError,
                       match=r"For 'Median', the type of 'axis' should be 'int', "
                             r"but got type 'float'."):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_median_abnormal_keepdims_not_bool():
    """
    Feature: Ops.
    Description: median算子测试 异常场景，keep_dims is not bool.
    Expectation: expect correct result.
    """
    fact = MedianFactory(input_shape=(4, 5), global_median=False, axis=-1, keep_dims="False")
    with pytest.raises(TypeError,
                       match=r"For 'Median', the type of 'keep_dims' should be 'bool', "
                             r"but got type 'str'."):
        fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_median_globalmedian_true_axis_default():
    """
    Feature: Ops.
    Description: median算子测试，global_median=True， axis为默认值.
    Expectation: expect correct result.
    """
    fact = MedianFactory(input_shape=(3, 5), global_median=True, axis=0,
                         keep_dims=False, dtype=np.float32)
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_median_input_same_value():
    """
    Feature: Ops.
    Description: median算子测试，input含有多个相同中值.
    Expectation: expect correct result.
    """
    x = np.array([[2, 2, 2, 2], [2, 2, 2, 2]]).astype(np.float32)
    ps_net = Median(global_median=False, axis=1, keep_dims=True)
    jit(ps_net.construct, capture_mode="ast")(Tensor(x))
    context.set_context(mode=context.GRAPH_MODE)
    y_psjit, _ = ps_net(Tensor(x))
    pi_net = Median(global_median=False, axis=1, keep_dims=True)
    jit(ps_net.construct, capture_mode="bytecode")(Tensor(x))
    context.set_context(mode=context.PYNATIVE_MODE)
    y_pijit, _ = pi_net(Tensor(x))
    assert np.allclose(y_psjit.asnumpy(), y_pijit.asnumpy(), 0.0001, 0.0001)
