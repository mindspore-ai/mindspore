import numpy as np
import pytest
from mindspore import Tensor
from ..share.ops.primitive.maxpool_ops import MaxPoolMock
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maxpool_input_1x3x224x224_float16_strides_2_valid():
    """
    Feature: Ops.
    Description: create a net which contains MaxPool for mindspore and pijit, compare their results.
    Expectation: expect correct result.
    """
    inputs = []
    inputs.append(Tensor(np.random.randint(1, 10, (1, 3, 224, 224)).astype(np.float16)))
    attributes = {"pad_mode": "VALID",
                  "kernel_size": 2,
                  "strides": 2}
    fact = MaxPoolMock(attributes=attributes, inputs=[inputs])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maxpool_input_2x2x2x2_float32_strides_2_valid():
    """
    Feature: Ops.
    Description: create a net with input shape 2x2x2x2 for mindspore and pijit, compare their results.
    Expectation: expect correct result.
    """
    inputs = []
    inputs.append(Tensor(np.random.randint(-10, 10, (2, 2, 2, 2)).astype(np.float32)))
    attributes = {"pad_mode": "same",
                  "kernel_size": 1,
                  "strides": 1}
    fact = MaxPoolMock(attributes=attributes, inputs=[inputs])
    fact.loss = 1e-3
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maxpool_input_16x1x2x8_float16_strides_2_valid():
    """
    Feature: Ops.
    Description: create a net with input shape 16x1x2x8 for mindspore and pijit, compare their results.
    Expectation: expect correct result.
    """
    inputs = []
    inputs.append(Tensor(np.random.randint(1, 10, (16, 1, 2, 8)).astype(np.float16)))
    attributes = {"pad_mode": "VALID",
                  "kernel_size": 1,
                  "strides": 2}
    fact = MaxPoolMock(attributes=attributes, inputs=[inputs])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maxpool_input_2x32x16x16_float32_strides_2_valid():
    """
    Feature: Ops.
    Description: create a net with input shape 2x32x16x16 for mindspore and pijit, compare their results.
    Expectation: expect correct result.
    """
    inputs = []
    inputs.append(Tensor(np.random.randn(2, 32, 16, 16).astype(np.float32)))
    attributes = {"pad_mode": "SAME",
                  "kernel_size": 8,
                  "strides": 1,
                  "data_format": "NHWC"}
    fact = MaxPoolMock(attributes=attributes, inputs=[inputs])
    fact.loss = 1e-3
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maxpool_input_16x32x8x12_float16_strides_2_valid():
    """
    Feature: Ops.
    Description: create a net with input shape 16x32x8x12 for mindspore and pijit, compare their results.
    Expectation: expect correct result.
    """
    inputs = []
    inputs.append(Tensor(np.random.randn(16, 32, 8, 12).astype(np.float16)))
    attributes = {"pad_mode": "valid",
                  "kernel_size": 3,
                  "strides": 2}
    fact = MaxPoolMock(attributes=attributes, inputs=[inputs])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maxpool_input_2x12x12x12_float32_strides_2_same():
    """
    Feature: Ops.
    Description: create a net with input shape 2x12x12x12 for mindspore and pijit, compare their results.
    Expectation: expect correct result.
    """
    inputs = []
    inputs.append(Tensor(np.random.randn(2, 12, 12, 12).astype(np.float32)))
    attributes = {"pad_mode": "SAMe",
                  "kernel_size": 2,
                  "strides": 5,
                  "data_format": "NHWC"}
    fact = MaxPoolMock(attributes=attributes, inputs=[inputs])
    fact.loss = 1e-3
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maxpool_input_2x256x3x3_float16_strides_2_same():
    """
    Feature: Ops.
    Description: create a net with input shape 2x256x3x3 for mindspore and pijit, compare their results.
    Expectation: expect correct result.
    """
    inputs = []
    inputs.append(Tensor(np.random.randn(2, 256, 3, 3).astype(np.float16)))
    attributes = {"pad_mode": "same",
                  "kernel_size": 8,
                  "strides": 6,
                  "data_format": "NHWC"}
    fact = MaxPoolMock(attributes=attributes, inputs=[inputs])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maxpool_input_32x32x32x32_float32_strides_2_same():
    """
    Feature: Ops.
    Description: create a net with input shape 32x32x32x32 for mindspore and pijit, compare their results.
    Expectation: expect correct result.
    """
    inputs = []
    inputs.append(Tensor(np.random.randn(32, 32, 32, 32).astype(np.float32)))
    attributes = {"pad_mode": "same",
                  "kernel_size": 8,
                  "strides": (16, 2)}
    fact = MaxPoolMock(attributes=attributes, inputs=[inputs])
    fact.loss = 1e-3
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maxpool_input_1x7x32x16_float16_strides_2d_same():
    """
    Feature: Ops.
    Description: create a net with input shape 1x7x32x16 for mindspore and pijit, compare their results.
    Expectation: expect correct result.
    """
    inputs = []
    inputs.append(Tensor(np.random.randn(1, 7, 32, 16).astype(np.float16)))
    attributes = {"pad_mode": "Valid",
                  "kernel_size": (2, 2),
                  "strides": 2}
    fact = MaxPoolMock(attributes=attributes, inputs=[inputs])
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maxpool_input_1x8x256x256_float32_strides_2d_valid():
    """
    Feature: Ops.
    Description: create a net with input shape 1x8x256x256 for mindspore and pijit, compare their results.
    Expectation: expect correct result.
    """
    inputs = []
    inputs.append(Tensor(np.random.randn(1, 8, 256, 256).astype(np.float32)))
    attributes = {"pad_mode": "same",
                  "kernel_size": (7, 7),
                  "strides": (7, 7)}
    fact = MaxPoolMock(attributes=attributes, inputs=[inputs])
    fact.loss = 1e-3
    fact.forward_cmp()
    fact.grad_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_maxpool_input_1x8x3x3_k_4_valid():
    """
    Feature: Ops.
    Description: create a net with input shape error for mindspore and pijit, compare their results.
    Expectation: expect correct result.
    """
    inputs = []
    inputs.append(Tensor(np.random.randn(1, 8, 3, 3).astype(np.float32)))
    attributes = {"pad_mode": "valid",
                  "kernel_size": 4,
                  "strides": 1}
    fact = MaxPoolMock(attributes=attributes, inputs=[inputs])
    fact.loss = 1e-3
    with pytest.raises((ValueError, RuntimeError)):
        fact.forward_cmp()
