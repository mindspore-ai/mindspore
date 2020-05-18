# Copyright 2019 Huawei Technologies Co., Ltd
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
import pytest
import copy
import mindspore.dataset as ds
from mindspore.dataset.engine.iterators import ITERATORS_LIST, _cleanup

DATA_DIR = ["../data/dataset/testTFTestAllTypes/test.data"]
SCHEMA_DIR = "../data/dataset/testTFTestAllTypes/datasetSchema.json"
COLUMNS = ["col_1d", "col_2d", "col_3d", "col_binary", "col_float",
           "col_sint16", "col_sint32", "col_sint64"]


def check(project_columns):
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=COLUMNS, shuffle=False)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=project_columns, shuffle=False)

    for data_actual, data_expected in zip(data1.create_tuple_iterator(project_columns), data2.create_tuple_iterator()):
        assert len(data_actual) == len(data_expected)
        assert all([np.array_equal(d1, d2) for d1, d2 in zip(data_actual, data_expected)])


def test_case_iterator():
    """
    Test creating tuple iterator
    """
    check(COLUMNS)
    check(COLUMNS[0:1])
    check(COLUMNS[0:2])
    check(COLUMNS[0:7])
    check(COLUMNS[7:8])
    check(COLUMNS[0:2:8])


def test_iterator_weak_ref():
    ITERATORS_LIST.clear()
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR)
    itr1 = data.create_tuple_iterator()
    itr2 = data.create_tuple_iterator()
    itr3 = data.create_tuple_iterator()

    assert len(ITERATORS_LIST) == 3
    assert sum(itr() is not None for itr in ITERATORS_LIST) == 3

    del itr1
    assert len(ITERATORS_LIST) == 3
    assert sum(itr() is not None for itr in ITERATORS_LIST) == 2

    del itr2
    assert len(ITERATORS_LIST) == 3
    assert sum(itr() is not None for itr in ITERATORS_LIST) == 1

    del itr3
    assert len(ITERATORS_LIST) == 3
    assert sum(itr() is not None for itr in ITERATORS_LIST) == 0

    itr1 = data.create_tuple_iterator()
    itr2 = data.create_tuple_iterator()
    itr3 = data.create_tuple_iterator()

    _cleanup()
    with pytest.raises(AttributeError) as info:
        itr2.get_next()
    assert "object has no attribute 'depipeline'" in str(info.value)

    del itr1
    assert len(ITERATORS_LIST) == 6
    assert sum(itr() is not None for itr in ITERATORS_LIST) == 2

    _cleanup()


class MyDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __call__(self, t):
        return t


def test_tree_copy():
    #  Testing copying the tree with a pyfunc that cannot be pickled

    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=COLUMNS)
    data1 = data.map(operations=[MyDict()])

    itr = data1.create_tuple_iterator()

    assert id(data1) != id(itr.dataset)
    assert id(data) != id(itr.dataset.input[0])
    assert id(data1.operations[0]) == id(itr.dataset.operations[0])

    itr.release()


if __name__ == '__main__':
    test_tree_copy()
