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


def slice_compare(array, indexing, expected_array):
    data = ds.NumpySlicesDataset([array])
    if isinstance(indexing, list) and indexing and not isinstance(indexing[0], int):
        data = data.map(operations=ops.Slice(*indexing))
    else:
        data = data.map(operations=ops.Slice(indexing))
    for d in data.create_dict_iterator(output_numpy=True):
        np.testing.assert_array_equal(expected_array, d['column_0'])


def test_slice_all():
    slice_compare([1, 2, 3, 4, 5], None, [1, 2, 3, 4, 5])
    slice_compare([1, 2, 3, 4, 5], ..., [1, 2, 3, 4, 5])
    slice_compare([1, 2, 3, 4, 5], True, [1, 2, 3, 4, 5])


def test_slice_single_index():
    slice_compare([1, 2, 3, 4, 5], 0, [1])
    slice_compare([1, 2, 3, 4, 5], -3, [3])
    slice_compare([1, 2, 3, 4, 5], [0], [1])


def test_slice_indices_multidim():
    slice_compare([[1, 2, 3, 4, 5]], [[0], [0]], 1)
    slice_compare([[1, 2, 3, 4, 5]], [[0], [0, 3]], [[1, 4]])
    slice_compare([[1, 2, 3, 4, 5]], [0], [[1, 2, 3, 4, 5]])
    slice_compare([[1, 2, 3, 4, 5]], [[0], [0, -4]], [[1, 2]])


def test_slice_list_index():
    slice_compare([1, 2, 3, 4, 5], [0, 1, 4], [1, 2, 5])
    slice_compare([1, 2, 3, 4, 5], [4, 1, 0], [5, 2, 1])
    slice_compare([1, 2, 3, 4, 5], [-1, 1, 0], [5, 2, 1])
    slice_compare([1, 2, 3, 4, 5], [-1, -4, -2], [5, 2, 4])
    slice_compare([1, 2, 3, 4, 5], [3, 3, 3], [4, 4, 4])


def test_slice_index_and_slice():
    slice_compare([[1, 2, 3, 4, 5]], [slice(0, 1), [4]], [[5]])
    slice_compare([[1, 2, 3, 4, 5]], [[0], slice(0, 2)], [[1, 2]])
    slice_compare([[1, 2, 3, 4], [5, 6, 7, 8]], [[1], slice(2, 4, 1)], [[7, 8]])


def test_slice_slice_obj_1s():
    slice_compare([1, 2, 3, 4, 5], slice(1), [1])
    slice_compare([1, 2, 3, 4, 5], slice(4), [1, 2, 3, 4])
    slice_compare([[1, 2, 3, 4], [5, 6, 7, 8]], [slice(2), slice(2)], [[1, 2], [5, 6]])
    slice_compare([1, 2, 3, 4, 5], slice(10), [1, 2, 3, 4, 5])


def test_slice_slice_obj_2s():
    slice_compare([1, 2, 3, 4, 5], slice(0, 2), [1, 2])
    slice_compare([1, 2, 3, 4, 5], slice(2, 4), [3, 4])
    slice_compare([[1, 2, 3, 4], [5, 6, 7, 8]], [slice(0, 2), slice(1, 2)], [[2], [6]])
    slice_compare([1, 2, 3, 4, 5], slice(4, 10), [5])


def test_slice_slice_obj_2s_multidim():
    slice_compare([[1, 2, 3, 4, 5]], [slice(0, 1)], [[1, 2, 3, 4, 5]])
    slice_compare([[1, 2, 3, 4, 5]], [slice(0, 1), slice(4)], [[1, 2, 3, 4]])
    slice_compare([[1, 2, 3, 4, 5]], [slice(0, 1), slice(0, 3)], [[1, 2, 3]])
    slice_compare([[1, 2, 3, 4], [5, 6, 7, 8]], [slice(0, 2, 2), slice(2, 4, 1)], [[3, 4]])
    slice_compare([[1, 2, 3, 4], [5, 6, 7, 8]], [slice(1, 0, -1), slice(1)], [[5]])


def test_slice_slice_obj_3s():
    """
    Test passing in all parameters to the slice objects
    """
    slice_compare([1, 2, 3, 4, 5], slice(0, 2, 1), [1, 2])
    slice_compare([1, 2, 3, 4, 5], slice(0, 4, 1), [1, 2, 3, 4])
    slice_compare([1, 2, 3, 4, 5], slice(0, 10, 1), [1, 2, 3, 4, 5])
    slice_compare([1, 2, 3, 4, 5], slice(0, 5, 2), [1, 3, 5])
    slice_compare([1, 2, 3, 4, 5], slice(0, 2, 2), [1])
    slice_compare([1, 2, 3, 4, 5], slice(0, 1, 2), [1])
    slice_compare([1, 2, 3, 4, 5], slice(4, 5, 1), [5])
    slice_compare([1, 2, 3, 4, 5], slice(2, 5, 3), [3])
    slice_compare([[1, 2, 3, 4], [5, 6, 7, 8]], [slice(0, 2, 1)], [[1, 2, 3, 4], [5, 6, 7, 8]])
    slice_compare([[1, 2, 3, 4], [5, 6, 7, 8]], [slice(0, 2, 3)], [[1, 2, 3, 4]])
    slice_compare([[1, 2, 3, 4], [5, 6, 7, 8]], [slice(0, 2, 2), slice(0, 1, 2)], [[1]])
    slice_compare([[1, 2, 3, 4], [5, 6, 7, 8]], [slice(0, 2, 1), slice(0, 1, 2)], [[1], [5]])
    slice_compare([[[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 2, 3, 4], [5, 6, 7, 8]]],
                  [slice(0, 2, 1), slice(0, 1, 1), slice(0, 4, 2)],
                  [[[1, 3]], [[1, 3]]])


def test_slice_obj_3s_double():
    slice_compare([1., 2., 3., 4., 5.], slice(0, 2, 1), [1., 2.])
    slice_compare([1., 2., 3., 4., 5.], slice(0, 4, 1), [1., 2., 3., 4.])
    slice_compare([1., 2., 3., 4., 5.], slice(0, 5, 2), [1., 3., 5.])
    slice_compare([1., 2., 3., 4., 5.], slice(0, 2, 2), [1.])
    slice_compare([1., 2., 3., 4., 5.], slice(0, 1, 2), [1.])
    slice_compare([1., 2., 3., 4., 5.], slice(4, 5, 1), [5.])
    slice_compare([1., 2., 3., 4., 5.], slice(2, 5, 3), [3.])


def test_out_of_bounds_slicing():
    """
    Test passing indices outside of the input to the slice objects
    """
    slice_compare([1, 2, 3, 4, 5], slice(-15, -1), [1, 2, 3, 4])
    slice_compare([1, 2, 3, 4, 5], slice(-15, 15), [1, 2, 3, 4, 5])
    slice_compare([1, 2, 3, 4], slice(-15, -7), [])


def test_slice_multiple_rows():
    """
    Test passing in multiple rows
    """
    dataset = [[1], [3, 4, 5], [1, 2], [1, 2, 3, 4, 5, 6, 7]]
    exp_dataset = [[], [4, 5], [2], [2, 3, 4]]

    def gen():
        for row in dataset:
            yield (np.array(row),)

    data = ds.GeneratorDataset(gen, column_names=["col"])
    indexing = slice(1, 4)
    data = data.map(operations=ops.Slice(indexing))
    for (d, exp_d) in zip(data.create_dict_iterator(output_numpy=True), exp_dataset):
        np.testing.assert_array_equal(exp_d, d['col'])


def test_slice_none_and_ellipsis():
    """
    Test passing None and Ellipsis to Slice
    """
    dataset = [[1], [3, 4, 5], [1, 2], [1, 2, 3, 4, 5, 6, 7]]
    exp_dataset = [[1], [3, 4, 5], [1, 2], [1, 2, 3, 4, 5, 6, 7]]

    def gen():
        for row in dataset:
            yield (np.array(row),)

    data = ds.GeneratorDataset(gen, column_names=["col"])
    data = data.map(operations=ops.Slice(None))
    for (d, exp_d) in zip(data.create_dict_iterator(output_numpy=True), exp_dataset):
        np.testing.assert_array_equal(exp_d, d['col'])

    data = ds.GeneratorDataset(gen, column_names=["col"])
    data = data.map(operations=ops.Slice(Ellipsis))
    for (d, exp_d) in zip(data.create_dict_iterator(output_numpy=True), exp_dataset):
        np.testing.assert_array_equal(exp_d, d['col'])


def test_slice_obj_neg():
    slice_compare([1, 2, 3, 4, 5], slice(-1, -5, -1), [5, 4, 3, 2])
    slice_compare([1, 2, 3, 4, 5], slice(-1), [1, 2, 3, 4])
    slice_compare([1, 2, 3, 4, 5], slice(-2), [1, 2, 3])
    slice_compare([1, 2, 3, 4, 5], slice(-1, -5, -2), [5, 3])
    slice_compare([1, 2, 3, 4, 5], slice(-5, -1, 2), [1, 3])
    slice_compare([1, 2, 3, 4, 5], slice(-5, -1), [1, 2, 3, 4])


def test_slice_all_str():
    slice_compare([b"1", b"2", b"3", b"4", b"5"], None, [b"1", b"2", b"3", b"4", b"5"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], ..., [b"1", b"2", b"3", b"4", b"5"])


def test_slice_single_index_str():
    slice_compare([b"1", b"2", b"3", b"4", b"5"], [0, 1], [b"1", b"2"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], [0, 1], [b"1", b"2"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], [4], [b"5"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], [-1], [b"5"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], [-5], [b"1"])


def test_slice_indexes_multidim_str():
    slice_compare([[b"1", b"2", b"3", b"4", b"5"]], [[0], 0], [[b"1"]])
    slice_compare([[b"1", b"2", b"3", b"4", b"5"]], [[0], [0, 1]], [[b"1", b"2"]])


def test_slice_list_index_str():
    slice_compare([b"1", b"2", b"3", b"4", b"5"], [0, 1, 4], [b"1", b"2", b"5"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], [4, 1, 0], [b"5", b"2", b"1"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], [3, 3, 3], [b"4", b"4", b"4"])


# test str index object here
def test_slice_index_and_slice_str():
    slice_compare([[b"1", b"2", b"3", b"4", b"5"]], [slice(0, 1), 4], [[b"5"]])
    slice_compare([[b"1", b"2", b"3", b"4", b"5"]], [[0], slice(0, 2)], [[b"1", b"2"]])
    slice_compare([[b"1", b"2", b"3", b"4"], [b"5", b"6", b"7", b"8"]], [[1], slice(2, 4, 1)],
                  [[b"7", b"8"]])


def test_slice_slice_obj_1s_str():
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(1), [b"1"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(4), [b"1", b"2", b"3", b"4"])
    slice_compare([[b"1", b"2", b"3", b"4"], [b"5", b"6", b"7", b"8"]],
                  [slice(2), slice(2)],
                  [[b"1", b"2"], [b"5", b"6"]])


def test_slice_slice_obj_2s_str():
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(0, 2), [b"1", b"2"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(2, 4), [b"3", b"4"])
    slice_compare([[b"1", b"2", b"3", b"4"], [b"5", b"6", b"7", b"8"]],
                  [slice(0, 2), slice(1, 2)], [[b"2"], [b"6"]])


def test_slice_slice_obj_2s_multidim_str():
    slice_compare([[b"1", b"2", b"3", b"4", b"5"]], [slice(0, 1)], [[b"1", b"2", b"3", b"4", b"5"]])
    slice_compare([[b"1", b"2", b"3", b"4", b"5"]], [slice(0, 1), slice(4)],
                  [[b"1", b"2", b"3", b"4"]])
    slice_compare([[b"1", b"2", b"3", b"4", b"5"]], [slice(0, 1), slice(0, 3)],
                  [[b"1", b"2", b"3"]])
    slice_compare([[b"1", b"2", b"3", b"4"], [b"5", b"6", b"7", b"8"]],
                  [slice(0, 2, 2), slice(2, 4, 1)],
                  [[b"3", b"4"]])


def test_slice_slice_obj_3s_str():
    """
    Test passing in all parameters to the slice objects
    """
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(0, 2, 1), [b"1", b"2"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(0, 4, 1), [b"1", b"2", b"3", b"4"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(0, 5, 2), [b"1", b"3", b"5"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(0, 2, 2), [b"1"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(0, 1, 2), [b"1"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(4, 5, 1), [b"5"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(2, 5, 3), [b"3"])
    slice_compare([[b"1", b"2", b"3", b"4"], [b"5", b"6", b"7", b"8"]], [slice(0, 2, 1)],
                  [[b"1", b"2", b"3", b"4"], [b"5", b"6", b"7", b"8"]])
    slice_compare([[b"1", b"2", b"3", b"4"], [b"5", b"6", b"7", b"8"]], slice(0, 2, 3), [[b"1", b"2", b"3", b"4"]])
    slice_compare([[b"1", b"2", b"3", b"4"], [b"5", b"6", b"7", b"8"]],
                  [slice(0, 2, 2), slice(0, 1, 2)], [[b"1"]])
    slice_compare([[b"1", b"2", b"3", b"4"], [b"5", b"6", b"7", b"8"]],
                  [slice(0, 2, 1), slice(0, 1, 2)],
                  [[b"1"], [b"5"]])
    slice_compare([[[b"1", b"2", b"3", b"4"], [b"5", b"6", b"7", b"8"]],
                   [[b"1", b"2", b"3", b"4"], [b"5", b"6", b"7", b"8"]]],
                  [slice(0, 2, 1), slice(0, 1, 1), slice(0, 4, 2)],
                  [[[b"1", b"3"]], [[b"1", b"3"]]])


def test_slice_obj_neg_str():
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(-1, -5, -1), [b"5", b"4", b"3", b"2"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(-1), [b"1", b"2", b"3", b"4"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(-2), [b"1", b"2", b"3"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(-1, -5, -2), [b"5", b"3"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(-5, -1, 2), [b"1", b"3"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(-5, -1), [b"1", b"2", b"3", b"4"])


def test_out_of_bounds_slicing_str():
    """
    Test passing indices outside of the input to the slice objects
    """
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(-15, -1), [b"1", b"2", b"3", b"4"])
    slice_compare([b"1", b"2", b"3", b"4", b"5"], slice(-15, 15), [b"1", b"2", b"3", b"4", b"5"])

    indexing = slice(-15, -7)
    expected_array = np.array([], dtype="S")
    data = [b"1", b"2", b"3", b"4", b"5"]
    data = ds.NumpySlicesDataset([data])
    data = data.map(operations=ops.Slice(indexing))
    for d in data.create_dict_iterator(output_numpy=True):
        np.testing.assert_array_equal(expected_array, d['column_0'])


def test_slice_exceptions():
    """
    Test passing in invalid parameters
    """
    with pytest.raises(RuntimeError) as info:
        slice_compare([b"1", b"2", b"3", b"4", b"5"], [5], [b"1", b"2", b"3", b"4", b"5"])
    assert "Index 5 is out of bounds." in str(info.value)

    with pytest.raises(RuntimeError) as info:
        slice_compare([b"1", b"2", b"3", b"4", b"5"], [], [b"1", b"2", b"3", b"4", b"5"])
    assert "Both indices and slices can not be empty." in str(info.value)

    with pytest.raises(TypeError) as info:
        slice_compare([b"1", b"2", b"3", b"4", b"5"], [[[0, 1]]], [b"1", b"2", b"3", b"4", b"5"])
    assert "Argument slice_option[0] with value [0, 1] is not of type " \
           "(<class 'int'>,)." in str(info.value)

    with pytest.raises(TypeError) as info:
        slice_compare([b"1", b"2", b"3", b"4", b"5"], [[slice(3)]], [b"1", b"2", b"3", b"4", b"5"])
    assert "Argument slice_option[0] with value slice(None, 3, None) is not of type " \
           "(<class 'int'>,)." in str(info.value)


if __name__ == "__main__":
    test_slice_all()
    test_slice_single_index()
    test_slice_indices_multidim()
    test_slice_list_index()
    test_slice_index_and_slice()
    test_slice_slice_obj_1s()
    test_slice_slice_obj_2s()
    test_slice_slice_obj_2s_multidim()
    test_slice_slice_obj_3s()
    test_slice_obj_3s_double()
    test_slice_multiple_rows()
    test_slice_obj_neg()
    test_slice_all_str()
    test_slice_single_index_str()
    test_slice_indexes_multidim_str()
    test_slice_list_index_str()
    test_slice_index_and_slice_str()
    test_slice_slice_obj_1s_str()
    test_slice_slice_obj_2s_str()
    test_slice_slice_obj_2s_multidim_str()
    test_slice_slice_obj_3s_str()
    test_slice_obj_neg_str()
    test_out_of_bounds_slicing_str()
    test_slice_exceptions()
