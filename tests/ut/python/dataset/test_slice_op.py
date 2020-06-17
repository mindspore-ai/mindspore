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
Testing Slice op in DE
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as ops


def slice_compare(array, indexing):
    data = ds.NumpySlicesDataset([array])
    array = np.array(array)
    data = data.map(operations=ops.Slice(indexing))
    for d in data:
        if indexing is None:
            array = array[:]
        else:
            array = array[indexing]
        np.testing.assert_array_equal(array, d[0])


def test_slice_all():
    slice_compare([1, 2, 3, 4, 5], None)
    slice_compare([1, 2, 3, 4, 5], ...)


def test_slice_single_index():
    slice_compare([1, 2, 3, 4, 5], 0)
    slice_compare([1, 2, 3, 4, 5], 4)
    slice_compare([1, 2, 3, 4, 5], 2)
    slice_compare([1, 2, 3, 4, 5], -1)
    slice_compare([1, 2, 3, 4, 5], -5)
    slice_compare([1, 2, 3, 4, 5], -3)


def test_slice_list_index():
    slice_compare([1, 2, 3, 4, 5], [0, 1, 4])
    slice_compare([1, 2, 3, 4, 5], [4, 1, 0])
    slice_compare([1, 2, 3, 4, 5], [-1, 1, 0])
    slice_compare([1, 2, 3, 4, 5], [-1, -4, -2])
    slice_compare([1, 2, 3, 4, 5], [3, 3, 3])
    slice_compare([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])


def test_slice_slice_obj_2s():
    slice_compare([1, 2, 3, 4, 5], slice(0, 2))
    slice_compare([1, 2, 3, 4, 5], slice(2, 4))
    slice_compare([1, 2, 3, 4, 5], slice(4, 10))


def test_slice_slice_obj_1s():
    slice_compare([1, 2, 3, 4, 5], slice(1))
    slice_compare([1, 2, 3, 4, 5], slice(4))
    slice_compare([1, 2, 3, 4, 5], slice(10))


def test_slice_slice_obj_3s():
    slice_compare([1, 2, 3, 4, 5], slice(0, 2, 1))
    slice_compare([1, 2, 3, 4, 5], slice(0, 4, 1))
    slice_compare([1, 2, 3, 4, 5], slice(0, 10, 1))
    slice_compare([1, 2, 3, 4, 5], slice(0, 5, 2))
    slice_compare([1, 2, 3, 4, 5], slice(0, 2, 2))
    slice_compare([1, 2, 3, 4, 5], slice(0, 1, 2))
    slice_compare([1, 2, 3, 4, 5], slice(4, 5, 1))
    slice_compare([1, 2, 3, 4, 5], slice(2, 5, 3))


def test_slice_slice_obj_3s_double():
    slice_compare([1., 2., 3., 4., 5.], slice(0, 2, 1))
    slice_compare([1., 2., 3., 4., 5.], slice(0, 4, 1))
    slice_compare([1., 2., 3., 4., 5.], slice(0, 10, 1))
    slice_compare([1., 2., 3., 4., 5.], slice(0, 5, 2))
    slice_compare([1., 2., 3., 4., 5.], slice(0, 2, 2))
    slice_compare([1., 2., 3., 4., 5.], slice(0, 1, 2))
    slice_compare([1., 2., 3., 4., 5.], slice(4, 5, 1))
    slice_compare([1., 2., 3., 4., 5.], slice(2, 5, 3))


def test_slice_slice_obj_neg():
    slice_compare([1, 2, 3, 4, 5], slice(-1, -5, -1))
    slice_compare([1, 2, 3, 4, 5], slice(-1))
    slice_compare([1, 2, 3, 4, 5], slice(-2))
    slice_compare([1, 2, 3, 4, 5], slice(-1, -5, -2))
    slice_compare([1, 2, 3, 4, 5], slice(-5, -1, 2))
    slice_compare([1, 2, 3, 4, 5], slice(-5, -1))


def test_slice_exceptions():
    with pytest.raises(RuntimeError) as info:
        slice_compare([1, 2, 3, 4, 5], 5)
    assert "Index 5 is out of bounds [0,5)" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        slice_compare([1, 2, 3, 4, 5], slice(0))
    assert "Indices are empty, generated tensor would be empty." in str(info.value)

    with pytest.raises(RuntimeError) as info:
        slice_compare([1, 2, 3, 4, 5], slice(3, 1, 1))
    assert "Indices are empty, generated tensor would be empty." in str(info.value)

    with pytest.raises(RuntimeError) as info:
        slice_compare([1, 2, 3, 4, 5], slice(5, 10, 1))
    assert "Indices are empty, generated tensor would be empty." in str(info.value)

    with pytest.raises(RuntimeError) as info:
        slice_compare([1, 2, 3, 4, 5], slice(-1, -5, 1))
    assert "Indices are empty, generated tensor would be empty." in str(info.value)


def test_slice_all_str():
    slice_compare([b"1", b"2", b"3", b"4", b"5"], None)
    slice_compare([b"1", b"2", b"3", b"4", b"5"], ...)


def test_slice_single_index_str():
    slice_compare([b"1", b"2", b"3", b"4", b"5"], 0)
    slice_compare([b"1", b"2", b"3", b"4", b"5"], 4)
    slice_compare([b"1", b"2", b"3", b"4", b"5"], 2)
    slice_compare([b"1", b"2", b"3", b"4", b"5"], -1)
    slice_compare([b"1", b"2", b"3", b"4", b"5"], -5)
    slice_compare([b"1", b"2", b"3", b"4", b"5"], -3)


def test_slice_list_index_str():
    slice_compare([b"1", b"2", b"3", b"4", b"5"], [0, 1, 4])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], [4, 1, 0])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], [-1, 1, 0])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], [-1, -4, -2])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], [3, 3, 3])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], [1, 1, 1, 1, 1])


def test_slice_slice_obj_2s_str():
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(0, 2))
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(2, 4))
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(4, 10))


def test_slice_slice_obj_1s_str():
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(1))
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(4))
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(10))


def test_slice_slice_obj_3s_str():
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(0, 2, 1))
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(0, 4, 1))
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(0, 10, 1))
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(0, 5, 2))
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(0, 2, 2))
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(0, 1, 2))
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(4, 5, 1))
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(2, 5, 3))


def test_slice_slice_obj_neg_str():
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(-1, -5, -1))
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(-1))
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(-2))
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(-1, -5, -2))
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(-5, -1, 2))
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(-5, -1))


def test_slice_exceptions_str():
    with pytest.raises(RuntimeError) as info:
        slice_compare([b"1", b"2", b"3", b"4", b"5"], 5)
    assert "Index 5 is out of bounds [0,5)" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(0))
    assert "Indices are empty, generated tensor would be empty." in str(info.value)

    with pytest.raises(RuntimeError) as info:
        slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(3, 1, 1))
    assert "Indices are empty, generated tensor would be empty." in str(info.value)

    with pytest.raises(RuntimeError) as info:
        slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(5, 10, 1))
    assert "Indices are empty, generated tensor would be empty." in str(info.value)

    with pytest.raises(RuntimeError) as info:
        slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(-1, -5, 1))
    assert "Indices are empty, generated tensor would be empty." in str(info.value)


if __name__ == "__main__":
    test_slice_all()
    test_slice_single_index()
    test_slice_list_index()
    test_slice_slice_obj_3s()
    test_slice_slice_obj_2s()
    test_slice_slice_obj_1s()
    test_slice_slice_obj_neg()
    test_slice_exceptions()
    test_slice_slice_obj_3s_double()
    test_slice_all_str()
    test_slice_single_index_str()
    test_slice_list_index_str()
    test_slice_slice_obj_3s_str()
    test_slice_slice_obj_2s_str()
    test_slice_slice_obj_1s_str()
    test_slice_slice_obj_neg_str()
    test_slice_exceptions_str()
