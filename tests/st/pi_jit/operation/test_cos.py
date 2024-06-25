import numpy as np
import pytest
from mindspore import Tensor
from ..share.ops.primitive.cos_ops import CosMock
from ..dynamic_shape_operations.cos import CosDynamicShapeFactory
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_cos_input_64x3125():
    '''
    Description: cos算子测试，inputa_shape=(64, 3125)

    Expectation:
        1.  output return ok and the accuracy is consistent with the benchmark.
    '''
    input_x = Tensor(np.random.randn(64, 3125).astype(np.float32))
    fact = CosMock(inputs=[input_x])
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_cos_input_shape():
    '''
    Description: cos算子测试，inputa_shape=1D--->6D

    Expectation:
        1.  output return ok and the accuracy is consistent with the benchmark.
    '''
    input_x = Tensor(np.random.randn(7,).astype(np.float32))
    fact = CosMock(inputs=[input_x])
    fact.forward_cmp()

    input_x = Tensor(np.random.randn(2, 3, 4).astype(np.float32))
    fact = CosMock(inputs=[input_x])
    fact.forward_cmp()

    input_x = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    fact = CosMock(inputs=[input_x])
    fact.forward_cmp()

    input_x = Tensor(np.random.randn(6, 2, 3, 4, 5).astype(np.float32))
    fact = CosMock(inputs=[input_x])
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_cos_input_shape_6d():
    '''
    Description: cos算子测试，inputa_shape 6D

    Expectation:
        1.  output return ok and the accuracy is consistent with the benchmark.
    '''
    input_x = Tensor(np.random.randn(2, 3, 7, 8, 4, 5).astype(np.float32))
    fact = CosMock(inputs=[input_x])
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_cos_input_float32():
    '''
    Description: Cos算子正反向dynamic shape测试,input_shape=(3, 5, 8, 10, 5), dtype=np.float32

    Expectation:
        1.  output return ok and the accuracy is consistent with the benchmark.
    '''
    input_x = Tensor(np.random.randn(3, 5, 8, 10, 5).astype(np.float32))
    fact = CosMock(inputs=[input_x])
    fact.forward_dynamic_shape_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_cos_input_float16():
    '''
    Description: Cos算子正反向dynamic shape测试,input_shape=(3, 4, 5), dtype=np.float16

    Expectation:
        1.  output return ok and the accuracy is consistent with the benchmark.
    '''
    input_x = Tensor(np.random.randn(3, 4, 5).astype(np.float64))
    fact = CosMock(inputs=[input_x])
    fact.forward_dynamic_shape_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_cos_float32():
    '''
    Description: test cos with dynamic shape input, dtype=float32

    Expectation:
        1.  output return ok and the accuracy is consistent with the benchmark.
    '''
    input_x = Tensor(np.random.rand(2, 10, 5, 10).astype(np.float32))
    indices = Tensor(np.random.choice(3, 2, replace=False).astype(np.int32))
    fact = CosDynamicShapeFactory([input_x, indices], dtype=np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_p_cos_float16():
    '''
    Description: test cos with dynamic shape input, dtype=float16

    Expectation:
        1.  output return ok and the accuracy is consistent with the benchmark.
    '''
    input_x = Tensor(np.random.rand(1, 1, 2, 4, 10).astype(np.float16))
    indices = Tensor(np.random.choice(3, 1, replace=False).astype(np.int32))
    fact = CosDynamicShapeFactory([input_x, indices], dtype=np.float16)
    fact.forward_cmp()
