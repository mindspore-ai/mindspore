# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test numpy type np_dtype.bfloat16."""
import pytest
import mindspore as ms
from mindspore.common import np_dtype
from mindspore.common.initializer import initializer, Constant
import numpy as np


def test_bf16_numpy():
    """
    Feature: Numpy array with type of bfloat16.
    Description: Test Numpy array with type of bfloat16.
    Expectation: Success.
    """
    x = np.array([1, 2, 3], dtype=np_dtype.bfloat16)
    assert x.dtype == np_dtype.bfloat16
    assert x.shape == (3,)
    assert np.allclose(x, np.array([1, 2, 3], dtype=np.float32))

def test_bf16_numpy_func():
    """
    Feature: Numpy functions with type of bfloat16.
    Description: Test Numpy functions with type of bfloat16.
    Expectation: Success.
    """
    x_f32 = np.array([1, 2, 3], dtype=np.float32)
    y_f32 = np.array([4, 5, 6], dtype=np.float32)
    x_bf16 = np.array([1, 2, 3], dtype=np_dtype.bfloat16)
    y_bf16 = np.array([4, 5, 6], dtype=np_dtype.bfloat16)
    # Math operations
    assert np.allclose(np.add(x_bf16, y_bf16), np.add(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.subtract(x_bf16, y_bf16), np.subtract(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.multiply(x_bf16, y_bf16), np.multiply(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.divide(x_bf16, y_bf16), np.divide(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.logaddexp(x_bf16, y_bf16), np.logaddexp(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.logaddexp2(x_bf16, y_bf16), np.logaddexp2(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.negative(x_bf16), np.negative(x_f32), 0.01, 0.01)
    assert np.allclose(np.positive(x_bf16), np.positive(x_f32), 0.01, 0.01)
    assert np.allclose(np.true_divide(x_bf16, y_bf16), np.true_divide(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.floor_divide(x_bf16, y_bf16), np.floor_divide(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.power(x_bf16, y_bf16), np.power(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.remainder(x_bf16, y_bf16), np.remainder(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.mod(x_bf16, y_bf16), np.mod(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.fmod(x_bf16, y_bf16), np.fmod(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.divmod(x_bf16, y_bf16), np.divmod(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.absolute(x_bf16), np.absolute(x_f32), 0.01, 0.01)
    assert np.allclose(np.fabs(x_bf16), np.fabs(x_f32), 0.01, 0.01)
    assert np.allclose(np.rint(x_bf16), np.rint(x_f32), 0.01, 0.01)
    assert np.allclose(np.sign(x_bf16), np.sign(x_f32), 0.01, 0.01)
    assert np.allclose(np.heaviside(x_bf16, y_bf16), np.heaviside(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.conjugate(x_bf16), np.conjugate(x_f32), 0.01, 0.01)
    assert np.allclose(np.exp(x_bf16), np.exp(x_f32), 0.01, 0.01)
    assert np.allclose(np.exp2(x_bf16), np.exp2(x_f32), 0.01, 0.01)
    assert np.allclose(np.expm1(x_bf16), np.expm1(x_f32), 0.01, 0.01)
    assert np.allclose(np.log(x_bf16), np.log(x_f32), 0.01, 0.01)
    assert np.allclose(np.log1p(x_bf16), np.log1p(x_f32), 0.01, 0.01)
    assert np.allclose(np.log2(x_bf16), np.log2(x_f32), 0.01, 0.01)
    assert np.allclose(np.log10(x_bf16), np.log10(x_f32), 0.01, 0.01)
    assert np.allclose(np.sqrt(x_bf16), np.sqrt(x_f32), 0.01, 0.01)
    assert np.allclose(np.square(x_bf16), np.square(x_f32), 0.01, 0.01)
    assert np.allclose(np.cbrt(x_bf16), np.cbrt(x_f32), 0.01, 0.01)
    assert np.allclose(np.reciprocal(x_bf16), np.reciprocal(x_f32), 0.01, 0.01)
    # Trigonometric functions
    assert np.allclose(np.sin(x_bf16), np.sin(x_f32), 0.01, 0.01)
    assert np.allclose(np.cos(x_bf16), np.cos(x_f32), 0.01, 0.01)
    assert np.allclose(np.tan(x_bf16), np.tan(x_f32), 0.01, 0.01)
    assert np.allclose(np.arcsin(x_bf16), np.arcsin(x_f32), 0.01, 0.01, equal_nan=True)
    assert np.allclose(np.arccos(x_bf16), np.arccos(x_f32), 0.01, 0.01, equal_nan=True)
    assert np.allclose(np.arctan(x_bf16), np.arctan(x_f32), 0.01, 0.01, equal_nan=True)
    assert np.allclose(np.arctan2(x_bf16, y_bf16), np.arctan2(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.hypot(x_bf16, y_bf16), np.hypot(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.sinh(x_bf16), np.sinh(x_f32), 0.01, 0.01)
    assert np.allclose(np.cosh(x_bf16), np.cosh(x_f32), 0.01, 0.01)
    assert np.allclose(np.tanh(x_bf16), np.tanh(x_f32), 0.01, 0.01)
    assert np.allclose(np.arcsinh(x_bf16), np.arcsinh(x_f32), 0.01, 0.01, equal_nan=True)
    assert np.allclose(np.arccosh(x_bf16), np.arccosh(x_f32), 0.01, 0.01, equal_nan=True)
    assert np.allclose(np.arctanh(x_bf16), np.arctanh(x_f32), 0.01, 0.01, equal_nan=True)
    assert np.allclose(np.deg2rad(x_bf16), np.deg2rad(x_f32), 0.01, 0.01)
    assert np.allclose(np.rad2deg(x_bf16), np.rad2deg(x_f32), 0.01, 0.01)
    # Comparison functions
    assert np.allclose(np.equal(x_bf16, y_bf16), np.equal(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.not_equal(x_bf16, y_bf16), np.not_equal(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.less(x_bf16, y_bf16), np.less(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.less_equal(x_bf16, y_bf16), np.less_equal(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.greater(x_bf16, y_bf16), np.greater(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.greater_equal(x_bf16, y_bf16), np.greater_equal(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.maximum(x_bf16, y_bf16), np.maximum(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.minimum(x_bf16, y_bf16), np.minimum(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.fmax(x_bf16, y_bf16), np.fmax(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.fmin(x_bf16, y_bf16), np.fmin(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.logical_and(x_bf16, y_bf16), np.logical_and(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.logical_or(x_bf16, y_bf16), np.logical_or(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.logical_xor(x_bf16, y_bf16), np.logical_xor(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.logical_not(x_bf16), np.logical_not(x_f32), 0.01, 0.01)
    # Floating point functions
    assert np.allclose(np.isfinite(x_bf16), np.isfinite(x_f32), 0.01, 0.01)
    assert np.allclose(np.isinf(x_bf16), np.isinf(x_f32), 0.01, 0.01)
    assert np.allclose(np.isnan(x_bf16), np.isnan(x_f32), 0.01, 0.01)
    assert np.allclose(np.signbit(x_bf16), np.signbit(x_f32), 0.01, 0.01)
    assert np.allclose(np.copysign(x_bf16, y_bf16), np.copysign(x_f32, y_f32), 0.01, 0.01)
    assert np.allclose(np.modf(x_bf16), np.modf(x_f32), 0.01, 0.01)
    assert np.allclose(np.ldexp(x_bf16, [1, 2, 3]), np.ldexp(x_f32, [1, 2, 3]), 0.01, 0.01)
    assert np.allclose(np.frexp(x_bf16)[0], np.frexp(x_f32)[0], 0.01, 0.01)
    assert np.allclose(np.frexp(x_bf16)[1], np.frexp(x_f32)[1], 0.01, 0.01)
    assert np.allclose(np.floor(x_bf16), np.floor(x_f32), 0.01, 0.01)
    assert np.allclose(np.ceil(x_bf16), np.ceil(x_f32), 0.01, 0.01)
    assert np.allclose(np.trunc(x_bf16), np.trunc(x_f32), 0.01, 0.01)
    assert np.allclose(np.nextafter(x_bf16, y_bf16), np.nextafter(x_f32, y_f32), 0.01, 0.01)

def test_bf16_asnumpy():
    """
    Feature: Tensor asnumpy() with type of bfloat16.
    Description: Test Tensor asnumpy() method with type of bfloat16.
    Expectation: Success.
    """
    x = ms.Tensor([1, 2, 3], dtype=ms.bfloat16)
    np_x = x.asnumpy()
    assert np_x.dtype == np_dtype.bfloat16
    assert np_x.shape == (3,)
    assert np.allclose(np_x, np.array([1, 2, 3], dtype=np.float32))


def test_bf16_bool():
    """
    Feature: Tensor __bool__ with type of bfloat16.
    Description: The __bool__ method in Tensor is implemented by numpy, test type of bfloat16.
    Expectation: Success.
    """
    x1 = ms.Tensor(1, dtype=ms.bfloat16)
    x2 = ms.Tensor([1], dtype=ms.bfloat16)
    x3 = ms.Tensor([1, 2, 3], dtype=ms.bfloat16)
    assert bool(x1) == bool(np.array(1, dtype=np.float32))
    assert bool(x2) == bool(np.array([1], dtype=np.float32))
    with pytest.raises(ValueError, match="more than one element is ambiguous"):
        bool(x3)

def test_bf16_int():
    """
    Feature: Tensor __int__ with type of bfloat16.
    Description: The __int__ method in Tensor is implemented by numpy, test type of bfloat16.
    Expectation: Success.
    """
    x1 = ms.Tensor(1, dtype=ms.bfloat16)
    x2 = ms.Tensor([1], dtype=ms.bfloat16)
    x3 = ms.Tensor([1, 2, 3], dtype=ms.bfloat16)
    assert int(x1) == int(np.array(1, dtype=np.float32))
    assert int(x2) == int(np.array([1], dtype=np.float32))
    with pytest.raises(ValueError, match="Only one element tensors can be converted to Python scalars"):
        int(x3)

def test_bf16_float():
    """
    Feature: Tensor __float__ with type of bfloat16.
    Description: The __float__ method in Tensor is implemented by numpy, test type of bfloat16.
    Expectation: Success.
    """
    x1 = ms.Tensor(1, dtype=ms.bfloat16)
    x2 = ms.Tensor([1], dtype=ms.bfloat16)
    x3 = ms.Tensor([1, 2, 3], dtype=ms.bfloat16)
    assert float(x1) == float(np.array(1, dtype=np.float32))
    assert float(x2) == float(np.array([1], dtype=np.float32))
    with pytest.raises(ValueError, match="Only one element tensors can be converted to Python scalars"):
        float(x3)

def test_bf16_index():
    """
    Feature: Tensor __index__ with type of bfloat16.
    Description: The __index__ method in Tensor is implemented by numpy, test type of bfloat16.
    Expectation: Success.
    """
    x = ms.Tensor(1, dtype=ms.bfloat16)
    with pytest.raises(ValueError, match="Only integer tensors of a single element can be converted to an index"):
        a = [1, 2, 3]
        _ = a[x]

def test_bf16_str():
    """
    Feature: Tensor __str__ with type of bfloat16.
    Description: The __str__ method in Tensor is implemented by numpy, test type of bfloat16.
    Expectation: Success.
    """
    x = ms.Tensor([1, 2, 3], dtype=ms.bfloat16)
    assert str(x) == "[1.000000 2.000000 3.000000]"

def test_bf16_item():
    """
    Feature: Tensor __item__ with type of bfloat16.
    Description: The __item__ method in Tensor is implemented by numpy, test type of bfloat16.
    Expectation: Success.
    """
    x = ms.Tensor([[1, 2, 3], [4, 5, 6]], ms.bfloat16)
    assert np.allclose(float(x.item((0, 1))), 2, rtol=0.01, atol=0.01)
    x = ms.Tensor(1.2, ms.bfloat16)
    assert np.allclose(float(x.item()), 1.2, rtol=0.01, atol=0.01)

def test_bf16_init_data():
    """
    Feature: Tensor init_data with type of bfloat16.
    Description: The init_data method in Tensor is implemented by numpy, test type of bfloat16.
    Expectation: Success.
    """
    x = initializer(Constant(1), [2, 2], ms.bfloat16)
    out = x.init_data()
    assert np.allclose(out.asnumpy(), np.array([[1, 1], [1, 1]], dtype=np.float32), rtol=0.01, atol=0.01)
