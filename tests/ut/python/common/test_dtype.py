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
"""Test dtype."""
import pytest
import mindspore as ms
from mindspore.common import dtype as mstype
import numpy as np

def test_dtype_equal():
    """
    Feature: Tensor dtype share one instance.
    Description: Test Tensor dtype share one instance.
    Expectation: Success.
    """
    a = ms.Tensor(1)
    assert a.dtype is ms.int64
    b = ms.Tensor(1, dtype=ms.float32)
    assert b.dtype is ms.float32

def test_dtype_methods():
    """
    Feature: Tensor dtype methods.
    Description: Test Tensor dtype methods.
    Expectation: Success.
    """
    for dtype in mstype.all_types:
        # is_floating_point
        if dtype in mstype.float_type:
            assert dtype.is_floating_point
        else:
            assert not dtype.is_floating_point

        # is_signed
        if dtype in mstype.signed_type:
            assert dtype.is_signed
        else:
            assert not dtype.is_signed

        # is_complex
        if dtype in mstype.complex_type:
            assert dtype.is_complex
        else:
            assert not dtype.is_complex

        # itemsize
        if dtype in (mstype.bool, mstype.qint4x2, mstype.int8, mstype.uint8, mstype.float8_e4m3fn, mstype.float8_e5m2,
                     mstype.hifloat8, mstype.float8_e8m0fnu):
            assert dtype.itemsize == 1
        elif dtype in (mstype.int16, mstype.uint16, mstype.float16, mstype.bfloat16):
            assert dtype.itemsize == 2
        elif dtype in (mstype.int32, mstype.uint32, mstype.float32):
            assert dtype.itemsize == 4
        elif dtype in (mstype.int64, mstype.uint64, mstype.float64, mstype.complex64):
            assert dtype.itemsize == 8
        elif dtype == mstype.complex128:
            assert dtype.itemsize == 16

        # to_real
        if dtype == mstype.complex64:
            assert dtype.to_real() == mstype.float32
        elif dtype == mstype.complex128:
            assert dtype.to_real() == mstype.float64
        else:
            assert dtype.to_real() is dtype

        # to_complex
        if dtype in mstype.complex_type:
            assert dtype.to_complex() is dtype
        elif dtype in (mstype.float32, mstype.bfloat16):
            assert dtype.to_complex() == mstype.complex64
        elif dtype == mstype.float64:
            assert dtype.to_complex() == mstype.complex128
        else:
            with pytest.raises(TypeError):
                _ = dtype.to_complex()


def test_dtype_int_long_float_bool():
    """
    Feature: Tensor dtype int long float bool.
    Description: Test Tensor dtype int long float bool.
    Expectation: Success.
    """
    a = ms.Tensor(1, dtype=mstype.int)
    assert a.dtype == mstype.int32
    b = ms.Tensor(1, dtype=mstype.long)
    assert b.dtype == mstype.int64
    c = ms.Tensor(1.0, dtype=mstype.float)
    assert c.dtype == mstype.float32
    d = ms.Tensor(True, dtype=mstype.bool)
    assert d.dtype == mstype.bool_
    e = ms.Tensor(np.array(1 + 2j), dtype=mstype.cfloat)
    assert e.dtype == mstype.complex64
    f = ms.Tensor(np.array(1 + 2j), dtype=mstype.cdouble)
    assert f.dtype == mstype.complex128

    a1 = ms.Tensor(1, dtype=ms.int)
    assert a1.dtype == mstype.int32
    b1 = ms.Tensor(1, dtype=ms.long)
    assert b1.dtype == mstype.int64
    c1 = ms.Tensor(1.0, dtype=ms.float)
    assert c1.dtype == mstype.float32
    d1 = ms.Tensor(True, dtype=ms.bool)
    assert d1.dtype == mstype.bool_
    e1 = ms.Tensor(np.array(1 + 2j), dtype=ms.cfloat)
    assert e1.dtype == mstype.complex64
    f1 = ms.Tensor(np.array(1 + 2j), dtype=ms.cdouble)
    assert f1.dtype == mstype.complex128
