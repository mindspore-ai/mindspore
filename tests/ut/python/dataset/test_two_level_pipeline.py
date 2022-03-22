# Copyright 2022 Huawei Technologies Co., Ltd
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
This is the test module for two level pipeline.
"""
import os
import pytest

import mindspore.dataset as ds
from util_minddataset import add_and_remove_cv_file # pylint: disable=unused-import


# pylint: disable=redefined-outer-name
def test_minddtaset_generatordataset_01(add_and_remove_cv_file):
    """
    Feature: Test basic two level pipeline.
    Description: MindDataset + GeneratorDataset
    Expectation: Data Iteration Successfully.
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 1
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    data_set = ds.MindDataset(file_name + "0", columns_list, num_parallel_workers=num_readers, shuffle=None)
    dataset_size = data_set.get_dataset_size()

    class MyIterable:
        """ custom iteration """
        def __init__(self, dataset, dataset_size):
            self._iter = None
            self._index = 0
            self._dataset = dataset
            self._dataset_size = dataset_size

        def __next__(self):
            if self._index >= self._dataset_size:
                raise StopIteration
            if self._iter:
                item = next(self._iter)
                self._index += 1
                return item
            self._iter = self._dataset.create_tuple_iterator(num_epochs=1, output_numpy=True)
            return next(self)

        def __iter__(self):
            self._index = 0
            self._iter = None
            return self

        def __len__(self):
            return self._dataset_size

    dataset = ds.GeneratorDataset(source=MyIterable(data_set, dataset_size),
                                  column_names=["data", "file_name", "label"], num_parallel_workers=1)
    num_epoches = 3
    iter_ = dataset.create_dict_iterator(num_epochs=3, output_numpy=True)
    num_iter = 0
    for _ in range(num_epoches):
        for _ in iter_:
            num_iter += 1
    assert num_iter == num_epoches * dataset_size


# pylint: disable=redefined-outer-name
def test_minddtaset_generatordataset_exception_01(add_and_remove_cv_file):
    """
    Feature: Test basic two level pipeline.
    Description: invalid column name in MindDataset
    Expectation: throw expected exception.
    """
    err_columns_list = ["data", "filename", "label"]
    num_readers = 1
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    data_set = ds.MindDataset(file_name + "0", err_columns_list, num_parallel_workers=num_readers, shuffle=None)
    dataset_size = data_set.get_dataset_size()

    class MyIterable:
        """ custom iteration """
        def __init__(self, dataset, dataset_size):
            self._iter = None
            self._index = 0
            self._dataset = dataset
            self._dataset_size = dataset_size

        def __next__(self):
            if self._index >= self._dataset_size:
                raise StopIteration
            if self._iter:
                item = next(self._iter)
                self._index += 1
                return item
            self._iter = self._dataset.create_tuple_iterator(num_epochs=1, output_numpy=True)
            return next(self)

        def __iter__(self):
            self._index = 0
            self._iter = None
            return self

        def __len__(self):
            return self._dataset_size

    dataset = ds.GeneratorDataset(source=MyIterable(data_set, dataset_size),
                                  column_names=["data", "file_name", "label"], num_parallel_workers=1)
    num_epoches = 3
    iter_ = dataset.create_dict_iterator(num_epochs=3, output_numpy=True)
    num_iter = 0
    with pytest.raises(RuntimeError) as error_info:
        for _ in range(num_epoches):
            for _ in iter_:
                num_iter += 1
    assert 'Unexpected error. Invalid data, column name:' in str(error_info.value)
