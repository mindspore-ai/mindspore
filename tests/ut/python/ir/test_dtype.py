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
import numpy as np
import pytest
from dataclasses import dataclass

import mindspore as ms
from mindspore.common import dtype


def test_dtype_to_nptype():
    """test_dtype2nptype"""
    assert ms.dtype_to_nptype(ms.bool_) == np.bool_
    assert ms.dtype_to_nptype(ms.int8) == np.int8
    assert ms.dtype_to_nptype(ms.int16) == np.int16
    assert ms.dtype_to_nptype(ms.int32) == np.int32
    assert ms.dtype_to_nptype(ms.int64) == np.int64
    assert ms.dtype_to_nptype(ms.uint8) == np.uint8
    assert ms.dtype_to_nptype(ms.uint16) == np.uint16
    assert ms.dtype_to_nptype(ms.uint32) == np.uint32
    assert ms.dtype_to_nptype(ms.uint64) == np.uint64
    assert ms.dtype_to_nptype(ms.float16) == np.float16
    assert ms.dtype_to_nptype(ms.float32) == np.float32
    assert ms.dtype_to_nptype(ms.float64) == np.float64


def test_dtype_to_pytype():
    """test_dtype_to_pytype"""
    assert ms.dtype_to_pytype(ms.bool_) == bool
    assert ms.dtype_to_pytype(ms.int8) == int
    assert ms.dtype_to_pytype(ms.int16) == int
    assert ms.dtype_to_pytype(ms.int32) == int
    assert ms.dtype_to_pytype(ms.int64) == int
    assert ms.dtype_to_pytype(ms.uint8) == int
    assert ms.dtype_to_pytype(ms.uint16) == int
    assert ms.dtype_to_pytype(ms.uint32) == int
    assert ms.dtype_to_pytype(ms.uint64) == int
    assert ms.dtype_to_pytype(ms.float16) == float
    assert ms.dtype_to_pytype(ms.float32) == float
    assert ms.dtype_to_pytype(ms.float64) == float
    assert ms.dtype_to_pytype(ms.list_) == list
    assert ms.dtype_to_pytype(ms.tuple_) == tuple
    assert ms.dtype_to_pytype(ms.string) == str
    assert ms.dtype_to_pytype(ms.type_none) == type(None)


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
    me_type = dtype.get_py_obj_dtype(x)
    assert me_type == ms.float64
    me_type = dtype.get_py_obj_dtype(type(x))
    assert me_type == ms.float64

    x = 100
    me_type = dtype.get_py_obj_dtype(type(x))
    assert me_type == ms.int64
    me_type = dtype.get_py_obj_dtype(x)
    assert me_type == ms.int64

    x = False
    me_type = dtype.get_py_obj_dtype(type(x))
    assert me_type == ms.bool_
    me_type = dtype.get_py_obj_dtype(x)
    assert me_type == ms.bool_

    # support str
    # x = "string type"

    x = [1, 2, 3]
    me_type = dtype.get_py_obj_dtype(x)
    assert me_type == ms.list_
    me_type = dtype.get_py_obj_dtype(type(x))
    assert me_type == ms.list_

    x = (2, 4, 5)
    me_type = dtype.get_py_obj_dtype(x)
    assert me_type == ms.tuple_
    me_type = dtype.get_py_obj_dtype(type(x))
    assert me_type == ms.tuple_

    y = Foo(3)
    me_type = dtype.get_py_obj_dtype(y.x)
    assert me_type == ms.int64
    me_type = dtype.get_py_obj_dtype(type(y.x))
    assert me_type == ms.int64

    y = Foo(3.1)
    me_type = dtype.get_py_obj_dtype(y.x)
    assert me_type == ms.float64
    me_type = dtype.get_py_obj_dtype(type(y.x))
    assert me_type == ms.float64

    fields = get_class_attrib_types(y)
    assert len(fields) == 1
    me_type = dtype.get_py_obj_dtype(fields[0])
    assert me_type == ms.int64

    fields = get_class_attrib_types(Foo)
    assert len(fields) == 1
    me_type = dtype.get_py_obj_dtype(fields[0])
    assert me_type == ms.int64

    with pytest.raises(NotImplementedError):
        x = 1.5
        dtype.get_py_obj_dtype(type(type(x)))
