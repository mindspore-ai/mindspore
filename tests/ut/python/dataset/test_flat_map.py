# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import numpy as np

import mindspore.dataset as ds

DATA_FILE = "../data/dataset/test_flat_map/images1.txt"
INDEX_FILE = "../data/dataset/test_flat_map/image_index.txt"


def test_flat_map_basic():
    """
    Feature: flat_map
    Description: Test basic usage
    Expectation: The result is as expected
    """
    def flat_map_func(x):
        data_dir = x.item()
        d = ds.ImageFolderDataset(data_dir)
        return d

    data = ds.TextFileDataset(DATA_FILE)
    data = data.flat_map(flat_map_func)

    count = 0
    for d in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert isinstance(d[0], np.ndarray)
        count += 1
    assert count == 52


def test_flat_map_chain_call():
    """
    Feature: flat_map
    Description: Test chain call
    Expectation: The result is as expected
    """
    def flat_map_func_1(x):
        data_dir = x.item()
        d = ds.ImageFolderDataset(data_dir)
        return d

    def flat_map_func_2(x):
        text_file = x.item()
        d = ds.TextFileDataset(text_file)
        d = d.flat_map(flat_map_func_1)
        return d

    data = ds.TextFileDataset(INDEX_FILE)
    data = data.flat_map(flat_map_func_2)

    count = 0
    for d in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert isinstance(d[0], np.ndarray)
        count += 1
    assert count == 104


def test_flat_map_one_column():
    """
    Feature: flat_map
    Description: Test with one column dataset
    Expectation: The result is as expected
    """
    dataset = ds.NumpySlicesDataset([[0, 1], [2, 3]], shuffle=False)

    def repeat(array):
        data = ds.NumpySlicesDataset(array, shuffle=False)
        data = data.repeat(2)
        return data

    dataset = dataset.flat_map(repeat)

    i = 0
    expect = np.array([0, 1, 0, 1, 2, 3, 2, 3])
    for d in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(d[0], expect[i])
        i += 1

    dataset = ds.NumpySlicesDataset([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]], shuffle=False)

    def plus(array):
        data = ds.NumpySlicesDataset(array + 1, shuffle=False)
        return data

    dataset = dataset.flat_map(plus)

    i = 0
    expect = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    for d in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(d[0], expect[i])
        i += 1


def test_flat_map_multi_column():
    """
    Feature: flat_map
    Description: Test with multi column dataset
    Expectation: The result is as expected
    """
    dataset = ds.NumpySlicesDataset(([[0, 1], [2, 3]], [[0, -1], [-2, -3]]), column_names=["col1", "col2"],
                                    shuffle=False)

    def plus_and_minus(col1, col2):
        data = ds.NumpySlicesDataset((col1 + 1, col2 - 1), shuffle=False)
        return data

    dataset = dataset.flat_map(plus_and_minus)

    i = 0
    expect_col1 = np.array([1, 2, 3, 4])
    expect_col2 = np.array([-1, -2, -3, -4])
    for d in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(d[0], expect_col1[i])
        np.testing.assert_array_equal(d[1], expect_col2[i])
        i += 1


if __name__ == "__main__":
    test_flat_map_basic()
    test_flat_map_chain_call()
    test_flat_map_one_column()
    test_flat_map_multi_column()
