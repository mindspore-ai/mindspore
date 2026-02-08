# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
"""test for dataset iterator"""
import numpy as np
import pytest

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
import mindspore.dataset as ds
from mindspore.dataset.engine.iterators import ITERATORS_LIST, _cleanup

DATA_DIR = ["../data/dataset/testTFTestAllTypes/test.data"]
SCHEMA_DIR = "../data/dataset/testTFTestAllTypes/datasetSchema.json"
COLUMNS = ["col_1d", "col_2d", "col_3d", "col_binary", "col_float",
           "col_sint16", "col_sint32", "col_sint64"]


def check(project_columns):
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=COLUMNS, shuffle=False)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=project_columns, shuffle=False)

    for data_actual, data_expected in zip(data1.create_tuple_iterator(project_columns, num_epochs=1, output_numpy=True),
                                          data2.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        assert len(data_actual) == len(data_expected)
        assert all(np.array_equal(d1, d2) for d1, d2 in zip(data_actual, data_expected))


def test_iterator_create_tuple_numpy():
    """
    Feature: Iterator
    Description: Test creating tuple Iterator with output NumPy
    Expectation: Output is equal to the expected output
    """
    check(COLUMNS)
    check(COLUMNS[0:1])
    check(COLUMNS[0:2])
    check(COLUMNS[0:7])
    check(COLUMNS[7:8])
    check(COLUMNS[0:2:8])


def test_iterator_create_dict_mstensor():
    """
    Feature: Iterator
    Description: Test creating dict Iterator with output MSTensor
    Expectation: Output is equal to the expected output
    """
    def generator():
        for i in range(64):
            yield (np.array([i], dtype=np.float32),)

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator, ["data"])

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1):
        golden = np.array([i], dtype=np.float32)
        np.testing.assert_array_equal(item["data"].asnumpy(), golden)
        assert isinstance(item["data"], Tensor)
        assert item["data"].dtype == mstype.float32
        i += 1
    assert i == 64

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, do_copy=False):
        golden = np.array([i], dtype=np.float32)
        np.testing.assert_array_equal(item["data"].asnumpy(), golden)
        assert isinstance(item["data"], Tensor)
        assert item["data"].dtype == mstype.float32
        i += 1
    assert i == 64


def test_iterator_create_tuple_mstensor():
    """
    Feature: Iterator
    Description: Test creating tuple Iterator with output MSTensor
    Expectation: Output is equal to the expected output
    """
    def generator():
        for i in range(64):
            yield (np.array([i], dtype=np.float32),)

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator, ["data"])

    i = 0
    for item in data1.create_tuple_iterator(num_epochs=1):
        golden = np.array([i], dtype=np.float32)
        np.testing.assert_array_equal(item[0].asnumpy(), golden)
        assert isinstance(item[0], Tensor)
        assert item[0].dtype == mstype.float32
        i += 1
    assert i == 64

    i = 0
    for item in data1.create_tuple_iterator(num_epochs=1, do_copy=False):
        golden = np.array([i], dtype=np.float32)
        np.testing.assert_array_equal(item[0].asnumpy(), golden)
        assert isinstance(item[0], Tensor)
        assert item[0].dtype == mstype.float32
        i += 1
    assert i == 64


def test_iterator_weak_ref():
    """
    Feature: Iterator
    Description: Test __del__ on existing tuple Iterator
    Expectation: Output is equal to the expected output or correct error is raised when expected
    """
    ITERATORS_LIST.clear()
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR)
    itr1 = data.create_tuple_iterator(num_epochs=1)
    itr2 = data.create_tuple_iterator(num_epochs=1)
    itr3 = data.create_tuple_iterator(num_epochs=1)

    assert len(ITERATORS_LIST) == 3
    assert sum(itr() is not None for itr in ITERATORS_LIST) == 3

    del itr1
    assert len(ITERATORS_LIST) == 2
    assert sum(itr() is not None for itr in ITERATORS_LIST) == 2

    del itr2
    assert len(ITERATORS_LIST) == 1
    assert sum(itr() is not None for itr in ITERATORS_LIST) == 1

    del itr3
    assert ITERATORS_LIST == []
    assert sum(itr() is not None for itr in ITERATORS_LIST) == 0

    itr1 = data.create_tuple_iterator(num_epochs=1)
    itr2 = data.create_tuple_iterator(num_epochs=1)
    itr3 = data.create_tuple_iterator(num_epochs=1)

    _cleanup()
    with pytest.raises(AttributeError) as info:
        next(itr2)
    assert "object has no attribute '_runtime_context'" in str(info.value)

    del itr1
    assert ITERATORS_LIST == []

    _cleanup()


def test_iterator_exception():
    """
    Feature: Iterator
    Description: Test Iterator with invalid input
    Expectation: Correct error is raised as expected
    """
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR)
    try:
        _ = data.create_dict_iterator(num_epochs=1, output_numpy="123")
        assert False
    except TypeError as e:
        assert "Argument output_numpy with value 123 is not of type" in str(e)

    try:
        _ = data.create_dict_iterator(num_epochs=1, output_numpy=123)
        assert False
    except TypeError as e:
        assert "Argument output_numpy with value 123 is not of type" in str(e)

    try:
        _ = data.create_tuple_iterator(num_epochs=1, output_numpy="123")
        assert False
    except TypeError as e:
        assert "Argument output_numpy with value 123 is not of type" in str(e)

    try:
        _ = data.create_tuple_iterator(num_epochs=1, output_numpy=123)
        assert False
    except TypeError as e:
        assert "Argument output_numpy with value 123 is not of type" in str(e)


class MyDict(dict):
    """MyDict class"""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __call__(self, t):
        return t


def test_tree_copy():
    """
    Feature: Iterator
    Description: Test copying the tree with a pyfunc that cannot be pickled
    Expectation: Ids of iterator data are different than pre-iterator data
    """
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=COLUMNS)
    data1 = data.map(operations=[MyDict()])

    itr = data1.create_tuple_iterator(num_epochs=1)

    assert id(data1) != id(itr.dataset)
    assert id(data) != id(itr.dataset.children[0])

    itr.release()


def test_iterator_output_string():
    """
    Feature: Iterator
    Description: Test iterator output string to np.array
    Expectation: Success
    """
    class RandomAccessDataset:
        def __init__(self):
            self._data = ["Cat", "Dog", "Monkey", "Lion", "Tiger"]
            self._label = np.zeros((5, 1))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)

    loader = RandomAccessDataset()
    dataset = ds.GeneratorDataset(source=loader, column_names=["data", "label"])

    for data in dataset:
        assert len(data) == 2
        assert isinstance(data[0], np.ndarray)
        assert np.issubdtype(data[0].dtype, np.str_)
        assert isinstance(data[1], Tensor)

    for data in dataset.create_tuple_iterator():
        assert len(data) == 2
        assert isinstance(data[0], np.ndarray)
        assert np.issubdtype(data[0].dtype, np.str_)
        assert isinstance(data[1], Tensor)

    for data in dataset.create_dict_iterator():
        assert len(data) == 2
        assert isinstance(data["data"], np.ndarray)
        assert np.issubdtype(data["data"].dtype, np.str_)
        assert isinstance(data["label"], Tensor)


if __name__ == '__main__':
    test_iterator_create_tuple_numpy()
    test_iterator_weak_ref()
    test_iterator_exception()
    test_tree_copy()
    test_iterator_output_string()
