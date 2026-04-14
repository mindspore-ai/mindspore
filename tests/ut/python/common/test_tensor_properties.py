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
"""Test Tensor basic properties and operations."""
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore.common import dtype as mstype

ms.set_context(device_target="CPU")


def test_tensor_shape():
    """
    Feature: Tensor shape property.
    Description: Test Tensor shape returns correct tuple for various inputs.
    Expectation: Success.
    """
    a = Tensor([1, 2, 3])
    assert a.shape == (3,)

    b = Tensor([[1, 2], [3, 4]])
    assert b.shape == (2, 2)

    c = Tensor(np.zeros((2, 3, 4)))
    assert c.shape == (2, 3, 4)

    d = Tensor(1.0)
    assert d.shape == ()


def test_tensor_ndim():
    """
    Feature: Tensor ndim property.
    Description: Test Tensor ndim returns the correct number of dimensions.
    Expectation: Success.
    """
    scalar = Tensor(1)
    assert scalar.ndim == 0

    vec = Tensor([1, 2, 3])
    assert vec.ndim == 1

    mat = Tensor([[1, 2], [3, 4]])
    assert mat.ndim == 2

    cube = Tensor(np.zeros((2, 3, 4)))
    assert cube.ndim == 3


def test_tensor_size():
    """
    Feature: Tensor size property.
    Description: Test Tensor size returns the total number of elements.
    Expectation: Success.
    """
    a = Tensor([1, 2, 3])
    assert a.size == 3

    b = Tensor([[1, 2], [3, 4]])
    assert b.size == 4

    c = Tensor(np.zeros((2, 3, 4)))
    assert c.size == 24

    d = Tensor(1.0)
    assert d.size == 1


def test_tensor_dtype_default():
    """
    Feature: Tensor default dtype inference.
    Description: Test Tensor infers correct default dtype from Python literals.
    Expectation: Success.
    """
    a = Tensor(1)
    assert a.dtype == mstype.int64

    b = Tensor(1.0)
    assert b.dtype == mstype.float32

    c = Tensor(True)
    assert c.dtype == mstype.bool_

    d = Tensor(np.array([1, 2], dtype=np.float16))
    assert d.dtype == mstype.float16


def test_tensor_astype():
    """
    Feature: Tensor astype method.
    Description: Test Tensor astype correctly casts to target dtype.
    Expectation: Success.
    """
    a = Tensor([1, 2, 3], dtype=mstype.int32)
    b = a.astype(mstype.float32)
    assert b.dtype == mstype.float32
    assert b.shape == a.shape

    c = a.astype(mstype.float64)
    assert c.dtype == mstype.float64

    d = a.astype(mstype.bool_)
    assert d.dtype == mstype.bool_


def test_tensor_reshape():
    """
    Feature: Tensor reshape method.
    Description: Test Tensor reshape returns tensor with new shape and same data.
    Expectation: Success.
    """
    a = Tensor(np.arange(12, dtype=np.float32))
    b = a.reshape(3, 4)
    assert b.shape == (3, 4)
    assert b.size == 12

    c = a.reshape(2, 6)
    assert c.shape == (2, 6)

    # Infer dimension with -1
    d = a.reshape(4, -1)
    assert d.shape == (4, 3)


def test_tensor_reshape_invalid():
    """
    Feature: Tensor reshape with invalid shape.
    Description: Test Tensor reshape raises error when total elements mismatch.
    Expectation: Raise ValueError.
    """
    a = Tensor(np.arange(12, dtype=np.float32))
    with pytest.raises((ValueError, RuntimeError)):
        a.reshape(5, 5).asnumpy()


def test_tensor_flatten():
    """
    Feature: Tensor flatten method.
    Description: Test Tensor flatten returns a 1-D tensor with all elements.
    Expectation: Success.
    """
    a = Tensor([[1, 2, 3], [4, 5, 6]], dtype=mstype.int32)
    b = a.flatten()
    assert b.shape == (6,)
    assert b.ndim == 1
    assert np.array_equal(b.asnumpy(), np.array([1, 2, 3, 4, 5, 6], dtype=np.int32))


def test_tensor_transpose():
    """
    Feature: Tensor T property and transpose method.
    Description: Test Tensor transpose reverses axes correctly.
    Expectation: Success.
    """
    a = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    b = a.T
    assert b.shape == (3, 2)
    assert np.array_equal(b.asnumpy(), a.asnumpy().T)

    c = Tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    d = c.transpose(2, 0, 1)
    assert d.shape == (4, 2, 3)


def test_tensor_asnumpy():
    """
    Feature: Tensor asnumpy method.
    Description: Test Tensor asnumpy returns numpy array with matching data.
    Expectation: Success.
    """
    np_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a = Tensor(np_data)
    result = a.asnumpy()
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert np.array_equal(result, np_data)
