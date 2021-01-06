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
# ==============================================================================
"""
Testing Mask op in DE
"""
import numpy as np
import pytest

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as ops

mstype_to_np_type = {
    mstype.bool_: np.bool,
    mstype.int8: np.int8,
    mstype.uint8: np.uint8,
    mstype.int16: np.int16,
    mstype.uint16: np.uint16,
    mstype.int32: np.int32,
    mstype.uint32: np.uint32,
    mstype.int64: np.int64,
    mstype.uint64: np.uint64,
    mstype.float16: np.float16,
    mstype.float32: np.float32,
    mstype.float64: np.float64,
    mstype.string: np.str
}


def mask_compare(array, op, constant, dtype=mstype.bool_):
    data = ds.NumpySlicesDataset([array])
    array = np.array(array)
    data = data.map(operations=ops.Mask(op, constant, dtype))
    for d in data:
        if op == ops.Relational.EQ:
            array = array == np.array(constant, dtype=array.dtype)
        elif op == ops.Relational.NE:
            array = array != np.array(constant, dtype=array.dtype)
        elif op == ops.Relational.GT:
            array = array > np.array(constant, dtype=array.dtype)
        elif op == ops.Relational.GE:
            array = array >= np.array(constant, dtype=array.dtype)
        elif op == ops.Relational.LT:
            array = array < np.array(constant, dtype=array.dtype)
        elif op == ops.Relational.LE:
            array = array <= np.array(constant, dtype=array.dtype)

        array = array.astype(dtype=mstype_to_np_type[dtype])

        np.testing.assert_array_equal(array, d[0].asnumpy())


def test_mask_int_comparison():
    for k in mstype_to_np_type:
        if k == mstype.string:
            continue
        mask_compare([1, 2, 3, 4, 5], ops.Relational.EQ, 3, k)
        mask_compare([1, 2, 3, 4, 5], ops.Relational.NE, 3, k)
        mask_compare([1, 2, 3, 4, 5], ops.Relational.LT, 3, k)
        mask_compare([1, 2, 3, 4, 5], ops.Relational.LE, 3, k)
        mask_compare([1, 2, 3, 4, 5], ops.Relational.GT, 3, k)
        mask_compare([1, 2, 3, 4, 5], ops.Relational.GE, 3, k)


def test_mask_float_comparison():
    for k in mstype_to_np_type:
        if k == mstype.string:
            continue
        mask_compare([1.5, 2.5, 3., 4.5, 5.5], ops.Relational.EQ, 3, k)
        mask_compare([1.5, 2.5, 3., 4.5, 5.5], ops.Relational.NE, 3, k)
        mask_compare([1.5, 2.5, 3., 4.5, 5.5], ops.Relational.LT, 3, k)
        mask_compare([1.5, 2.5, 3., 4.5, 5.5], ops.Relational.LE, 3, k)
        mask_compare([1.5, 2.5, 3., 4.5, 5.5], ops.Relational.GT, 3, k)
        mask_compare([1.5, 2.5, 3., 4.5, 5.5], ops.Relational.GE, 3, k)


def test_mask_float_comparison2():
    for k in mstype_to_np_type:
        if k == mstype.string:
            continue
        mask_compare([1, 2, 3, 4, 5], ops.Relational.EQ, 3.5, k)
        mask_compare([1, 2, 3, 4, 5], ops.Relational.NE, 3.5, k)
        mask_compare([1, 2, 3, 4, 5], ops.Relational.LT, 3.5, k)
        mask_compare([1, 2, 3, 4, 5], ops.Relational.LE, 3.5, k)
        mask_compare([1, 2, 3, 4, 5], ops.Relational.GT, 3.5, k)
        mask_compare([1, 2, 3, 4, 5], ops.Relational.GE, 3.5, k)


def test_mask_string_comparison():
    for k in mstype_to_np_type:
        if k == mstype.string:
            continue
        mask_compare(["1.5", "2.5", "3.", "4.5", "5.5"], ops.Relational.EQ, "3.", k)
        mask_compare(["1.5", "2.5", "3.", "4.5", "5.5"], ops.Relational.NE, "3.", k)
        mask_compare(["1.5", "2.5", "3.", "4.5", "5.5"], ops.Relational.LT, "3.", k)
        mask_compare(["1.5", "2.5", "3.", "4.5", "5.5"], ops.Relational.LE, "3.", k)
        mask_compare(["1.5", "2.5", "3.", "4.5", "5.5"], ops.Relational.GT, "3.", k)
        mask_compare(["1.5", "2.5", "3.", "4.5", "5.5"], ops.Relational.GE, "3.", k)


def test_mask_exceptions_str():
    with pytest.raises(RuntimeError) as info:
        mask_compare([1, 2, 3, 4, 5], ops.Relational.EQ, "3.5")
    assert "input datatype does not match the value datatype." in str(info.value)

    with pytest.raises(RuntimeError) as info:
        mask_compare(["1", "2", "3", "4", "5"], ops.Relational.EQ, 3.5)
    assert "input datatype does not match the value datatype." in str(info.value)

    with pytest.raises(RuntimeError) as info:
        mask_compare(["1", "2", "3", "4", "5"], ops.Relational.EQ, "3.5", mstype.string)
    assert "only support numeric datatype of input." in str(info.value)


if __name__ == "__main__":
    test_mask_int_comparison()
    test_mask_float_comparison()
    test_mask_float_comparison2()
    test_mask_string_comparison()
    test_mask_exceptions_str()
