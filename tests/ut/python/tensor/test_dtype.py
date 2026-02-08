# Copyright 2020 Huawei Technologies Co., Ltd
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
"""test_dtype"""
from dataclasses import dataclass
import numpy as np
import pytest

import mindspore as ms
from mindspore.common import dtype


def test_dtype_to_nptype():
    """test_dtype2nptype"""
    assert dtype._dtype_to_nptype(ms.bool_) == np.bool_  # pylint:disable=protected-access
    assert dtype._dtype_to_nptype(ms.int8) == np.int8  # pylint:disable=protected-access
    assert dtype._dtype_to_nptype(ms.int16) == np.int16  # pylint:disable=protected-access
    assert dtype._dtype_to_nptype(ms.int32) == np.int32  # pylint:disable=protected-access
    assert dtype._dtype_to_nptype(ms.int64) == np.int64  # pylint:disable=protected-access
    assert dtype._dtype_to_nptype(ms.uint8) == np.uint8  # pylint:disable=protected-access
    assert dtype._dtype_to_nptype(ms.uint16) == np.uint16  # pylint:disable=protected-access
    assert dtype._dtype_to_nptype(ms.uint32) == np.uint32  # pylint:disable=protected-access
    assert dtype._dtype_to_nptype(ms.uint64) == np.uint64  # pylint:disable=protected-access
    assert dtype._dtype_to_nptype(ms.float16) == np.float16  # pylint:disable=protected-access
    assert dtype._dtype_to_nptype(ms.float32) == np.float32  # pylint:disable=protected-access
    assert dtype._dtype_to_nptype(ms.float64) == np.float64  # pylint:disable=protected-access
    assert dtype._dtype_to_nptype(ms.complex64) == np.complex64  # pylint:disable=protected-access
    assert dtype._dtype_to_nptype(ms.complex128) == np.complex128  # pylint:disable=protected-access


def test_dtype_to_pytype():
    """test_dtype_to_pytype"""
    assert dtype._dtype_to_pytype(ms.bool_) == bool  # pylint:disable=protected-access
    assert dtype._dtype_to_pytype(ms.int8) == int  # pylint:disable=protected-access
    assert dtype._dtype_to_pytype(ms.int16) == int  # pylint:disable=protected-access
    assert dtype._dtype_to_pytype(ms.int32) == int  # pylint:disable=protected-access
    assert dtype._dtype_to_pytype(ms.int64) == int  # pylint:disable=protected-access
    assert dtype._dtype_to_pytype(ms.uint8) == int  # pylint:disable=protected-access
    assert dtype._dtype_to_pytype(ms.uint16) == int  # pylint:disable=protected-access
    assert dtype._dtype_to_pytype(ms.uint32) == int  # pylint:disable=protected-access
    assert dtype._dtype_to_pytype(ms.uint64) == int  # pylint:disable=protected-access
    assert dtype._dtype_to_pytype(ms.float16) == float  # pylint:disable=protected-access
    assert dtype._dtype_to_pytype(ms.float32) == float  # pylint:disable=protected-access
    assert dtype._dtype_to_pytype(ms.float64) == float  # pylint:disable=protected-access
    assert dtype._dtype_to_pytype(ms.complex64) == complex  # pylint:disable=protected-access
    assert dtype._dtype_to_pytype(ms.complex128) == complex  # pylint:disable=protected-access
    assert dtype._dtype_to_pytype(ms.list_) == list  # pylint:disable=protected-access
    assert dtype._dtype_to_pytype(ms.tuple_) == tuple  # pylint:disable=protected-access
    assert dtype._dtype_to_pytype(ms.string) == str  # pylint:disable=protected-access
    assert dtype._dtype_to_pytype(ms.type_none) == type(None)  # pylint:disable=protected-access


@dataclass
class Foo:
    x: int

    def inf(self):
        return self.x


def get_class_attrib_types(cls):
    """
        get attrib type of dataclass
    """
    fields = cls.__dataclass_fields__
    attr_type = [field.type for name, field in fields.items()]
    return attr_type


def test_dtype():
    """test_dtype"""
    x = 1.5
    me_type = dtype._get_py_obj_dtype(x)  # pylint:disable=protected-access
    assert me_type == ms.float64
    me_type = dtype._get_py_obj_dtype(type(x))  # pylint:disable=protected-access
    assert me_type == ms.float64

    x = 100
    me_type = dtype._get_py_obj_dtype(type(x))  # pylint:disable=protected-access
    assert me_type == ms.int64
    me_type = dtype._get_py_obj_dtype(x)  # pylint:disable=protected-access
    assert me_type == ms.int64

    x = False
    me_type = dtype._get_py_obj_dtype(type(x))  # pylint:disable=protected-access
    assert me_type == ms.bool_
    me_type = dtype._get_py_obj_dtype(x)  # pylint:disable=protected-access
    assert me_type == ms.bool_

    x = 0.1+3j
    me_type = dtype._get_py_obj_dtype(type(x))  # pylint:disable=protected-access
    assert me_type == ms.complex128
    me_type = dtype._get_py_obj_dtype(x)  # pylint:disable=protected-access
    assert me_type == ms.complex128

    # support str
    # x = "string type"

    x = [1, 2, 3]
    me_type = dtype._get_py_obj_dtype(x)  # pylint:disable=protected-access
    assert me_type == ms.list_
    me_type = dtype._get_py_obj_dtype(type(x))  # pylint:disable=protected-access
    assert me_type == ms.list_

    x = (2, 4, 5)
    me_type = dtype._get_py_obj_dtype(x)  # pylint:disable=protected-access
    assert me_type == ms.tuple_
    me_type = dtype._get_py_obj_dtype(type(x))  # pylint:disable=protected-access
    assert me_type == ms.tuple_

    y = Foo(3)
    me_type = dtype._get_py_obj_dtype(y.x)  # pylint:disable=protected-access
    assert me_type == ms.int64
    me_type = dtype._get_py_obj_dtype(type(y.x))  # pylint:disable=protected-access
    assert me_type == ms.int64

    y = Foo(3.1)
    me_type = dtype._get_py_obj_dtype(y.x)  # pylint:disable=protected-access
    assert me_type == ms.float64
    me_type = dtype._get_py_obj_dtype(type(y.x))  # pylint:disable=protected-access
    assert me_type == ms.float64

    fields = get_class_attrib_types(y)
    assert len(fields) == 1
    me_type = dtype._get_py_obj_dtype(fields[0])  # pylint:disable=protected-access
    assert me_type == ms.int64

    fields = get_class_attrib_types(Foo)
    assert len(fields) == 1
    me_type = dtype._get_py_obj_dtype(fields[0])  # pylint:disable=protected-access
    assert me_type == ms.int64

    with pytest.raises(NotImplementedError):
        x = 1.5
        dtype._get_py_obj_dtype(type(type(x)))  # pylint:disable=protected-access


def test_type_equal():
    t1 = (dtype.int32, dtype.int32)
    valid_types = [dtype.float16, dtype.float32]
    assert t1 not in valid_types
    assert dtype.int32 not in valid_types
    assert dtype.float32 in valid_types
