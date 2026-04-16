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
"""Test Tensor dtype validation."""
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore.common import dtype as mstype

ms.set_device("CPU")


def test_tensor_dtype_from_numpy():
    """
    Feature: Tensor dtype inference from numpy array.
    Description: Test Tensor correctly reflects dtype of the source numpy array.
    Expectation: Success.
    """
    cases = [
        (np.float32, mstype.float32),
        (np.float64, mstype.float64),
        (np.float16, mstype.float16),
        (np.int8,    mstype.int8),
        (np.int16,   mstype.int16),
        (np.int32,   mstype.int32),
        (np.int64,   mstype.int64),
        (np.uint8,   mstype.uint8),
        (np.uint16,  mstype.uint16),
        (np.uint32,  mstype.uint32),
        (np.uint64,  mstype.uint64),
        (np.bool_,   mstype.bool_),
    ]
    for np_dtype, ms_dtype in cases:
        t = Tensor(np.array([1, 0], dtype=np_dtype))
        assert t.dtype == ms_dtype, f"Expected {ms_dtype}, got {t.dtype} for np dtype {np_dtype}"


def test_tensor_dtype_explicit():
    """
    Feature: Tensor explicit dtype parameter.
    Description: Test Tensor respects the dtype= argument at construction.
    Expectation: Success.
    """
    a = Tensor([1, 2, 3], dtype=mstype.float32)
    assert a.dtype == mstype.float32

    b = Tensor([1, 2, 3], dtype=mstype.int16)
    assert b.dtype == mstype.int16

    c = Tensor([1, 2, 3], dtype=mstype.bool_)
    assert c.dtype == mstype.bool_

    d = Tensor([1.5, 2.5], dtype=mstype.float64)
    assert d.dtype == mstype.float64


def test_tensor_dtype_astype():
    """
    Feature: Tensor astype dtype casting.
    Description: Test astype correctly changes dtype while preserving values.
    Expectation: Success.
    """
    a = Tensor(np.array([1, 2, 3], dtype=np.int32))

    b = a.astype(mstype.float32)
    assert b.dtype == mstype.float32
    assert np.allclose(b.asnumpy(), np.array([1.0, 2.0, 3.0], dtype=np.float32))

    c = a.astype(mstype.float64)
    assert c.dtype == mstype.float64

    d = a.astype(mstype.int8)
    assert d.dtype == mstype.int8

    e = a.astype(mstype.bool_)
    assert e.dtype == mstype.bool_
    assert np.array_equal(e.asnumpy(), np.array([True, True, True]))


def test_tensor_dtype_preserved_through_reshape():
    """
    Feature: Tensor dtype preservation through reshape.
    Description: Test reshape does not change the tensor dtype.
    Expectation: Success.
    """
    for ms_dtype, np_dtype in [(mstype.float32, np.float32), (mstype.int64, np.int64)]:
        a = Tensor(np.arange(6, dtype=np_dtype))
        b = a.reshape(2, 3)
        assert b.dtype == ms_dtype
        assert b.shape == (2, 3)


def test_tensor_dtype_preserved_through_transpose():
    """
    Feature: Tensor dtype preservation through transpose.
    Description: Test transpose does not change the tensor dtype.
    Expectation: Success.
    """
    a = Tensor(np.arange(6, dtype=np.float16).reshape(2, 3))
    b = a.T
    assert b.dtype == mstype.float16
    assert b.shape == (3, 2)


def test_tensor_dtype_preserved_through_flatten():
    """
    Feature: Tensor dtype preservation through flatten.
    Description: Test flatten does not change the tensor dtype.
    Expectation: Success.
    """
    a = Tensor(np.array([[1, 2], [3, 4]], dtype=np.int32))
    b = a.flatten()
    assert b.dtype == mstype.int32
    assert b.shape == (4,)


def test_tensor_dtype_arithmetic_same_type():
    """
    Feature: Tensor dtype preservation through arithmetic.
    Description: Test that arithmetic between same-dtype tensors preserves dtype.
    Expectation: Success.
    """
    for np_dtype, ms_dtype in [(np.float32, mstype.float32), (np.float64, mstype.float64)]:
        a = Tensor(np.array([1.0, 2.0, 3.0], dtype=np_dtype))
        b = Tensor(np.array([4.0, 5.0, 6.0], dtype=np_dtype))
        assert (a + b).dtype == ms_dtype
        assert (a - b).dtype == ms_dtype
        assert (a * b).dtype == ms_dtype
        assert (a / b).dtype == ms_dtype


def test_tensor_dtype_comparison_returns_bool():
    """
    Feature: Tensor comparison operator dtype.
    Description: Test that comparison operators return bool dtype tensors.
    Expectation: Success.
    """
    a = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = Tensor(np.array([3.0, 2.0, 1.0], dtype=np.float32))

    assert (a == b).dtype == mstype.bool_
    assert (a != b).dtype == mstype.bool_
    assert (a < b).dtype == mstype.bool_
    assert (a > b).dtype == mstype.bool_


def test_tensor_dtype_bool_values():
    """
    Feature: Tensor comparison result values.
    Description: Test that comparison operators produce correct bool values.
    Expectation: Success.
    """
    a = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = Tensor(np.array([1.0, 3.0, 2.0], dtype=np.float32))

    eq = (a == b).asnumpy()
    assert np.array_equal(eq, np.array([True, False, False]))

    lt = (a < b).asnumpy()
    assert np.array_equal(lt, np.array([False, True, False]))

    gt = (a > b).asnumpy()
    assert np.array_equal(gt, np.array([False, False, True]))
