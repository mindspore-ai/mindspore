# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Test Tensor arithmetic operations."""
import numpy as np
import mindspore as ms
from mindspore import Tensor

ms.set_device("CPU")


def test_tensor_add():
    """
    Feature: Tensor addition operator.
    Description: Test element-wise addition between tensors and scalars.
    Expectation: Success.
    """
    a = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = Tensor(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    c = a + b
    assert c.shape == (3,)
    assert np.allclose(c.asnumpy(), np.array([5.0, 7.0, 9.0], dtype=np.float32))

    # scalar addition
    d = a + 1.0
    assert np.allclose(d.asnumpy(), np.array([2.0, 3.0, 4.0], dtype=np.float32))


def test_tensor_sub():
    """
    Feature: Tensor subtraction operator.
    Description: Test element-wise subtraction between tensors and scalars.
    Expectation: Success.
    """
    a = Tensor(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    b = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    c = a - b
    assert np.allclose(c.asnumpy(), np.array([3.0, 3.0, 3.0], dtype=np.float32))

    # scalar subtraction
    d = a - 2.0
    assert np.allclose(d.asnumpy(), np.array([2.0, 3.0, 4.0], dtype=np.float32))


def test_tensor_mul():
    """
    Feature: Tensor multiplication operator.
    Description: Test element-wise multiplication between tensors and scalars.
    Expectation: Success.
    """
    a = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = Tensor(np.array([2.0, 3.0, 4.0], dtype=np.float32))
    c = a * b
    assert np.allclose(c.asnumpy(), np.array([2.0, 6.0, 12.0], dtype=np.float32))

    # scalar multiplication
    d = a * 3.0
    assert np.allclose(d.asnumpy(), np.array([3.0, 6.0, 9.0], dtype=np.float32))


def test_tensor_div():
    """
    Feature: Tensor true division operator.
    Description: Test element-wise true division between tensors and scalars.
    Expectation: Success.
    """
    a = Tensor(np.array([6.0, 8.0, 9.0], dtype=np.float32))
    b = Tensor(np.array([2.0, 4.0, 3.0], dtype=np.float32))
    c = a / b
    assert np.allclose(c.asnumpy(), np.array([3.0, 2.0, 3.0], dtype=np.float32))

    # scalar division
    d = a / 2.0
    assert np.allclose(d.asnumpy(), np.array([3.0, 4.0, 4.5], dtype=np.float32))


def test_tensor_neg():
    """
    Feature: Tensor unary negation operator.
    Description: Test unary minus flips sign of each element.
    Expectation: Success.
    """
    a = Tensor(np.array([1.0, -2.0, 0.0], dtype=np.float32))
    b = -a
    assert np.allclose(b.asnumpy(), np.array([-1.0, 2.0, 0.0], dtype=np.float32))


def test_tensor_broadcast_add():
    """
    Feature: Tensor broadcasting in addition.
    Description: Test that addition broadcasts a (1, 3) tensor over a (3, 3) tensor.
    Expectation: Success.
    """
    a = Tensor(np.ones((3, 3), dtype=np.float32))
    b = Tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))  # shape (1, 3)
    c = a + b
    assert c.shape == (3, 3)
    expected = np.array([[2.0, 3.0, 4.0]] * 3, dtype=np.float32)
    assert np.allclose(c.asnumpy(), expected)


def test_tensor_inplace_add():
    """
    Feature: Tensor iadd operator.
    Description: Test += updates the tensor value in place.
    Expectation: Success.
    """
    a = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    a += Tensor(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    assert np.allclose(a.asnumpy(), np.array([2.0, 3.0, 4.0], dtype=np.float32))
