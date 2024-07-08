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
import numpy as np

import mindspore.dataset as ds
from mindspore import log as logger
from util_minddataset import add_and_remove_cv_file, add_and_remove_file  # pylint: disable=unused-import


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
    num_epochs = 3
    iter_ = dataset.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)
    num_iter = 0
    for _ in range(num_epochs):
        for _ in iter_:
            num_iter += 1
    assert num_iter == num_epochs * dataset_size


# pylint: disable=redefined-outer-name
def test_minddtaset_generatordataset_exception_01(add_and_remove_cv_file):
    """
    Feature: Test basic two level pipeline.
    Description: Invalid column name in MindDataset
    Expectation: Throw expected exception.
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
    num_epochs = 3
    iter_ = dataset.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)
    num_iter = 0
    with pytest.raises(RuntimeError) as error_info:
        for _ in range(num_epochs):
            for _ in iter_:
                num_iter += 1
    assert 'Invalid data, column name:' in str(error_info.value)


# pylint: disable=redefined-outer-name
def test_minddtaset_generatordataset_exception_02(add_and_remove_file):
    """
    Feature: Test basic two level pipeline for mixed dataset.
    Description: Invalid column name in MindDataset
    Expectation: Throw expected exception.
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 1
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    file_paths = [file_name + "_cv" + str(i) for i in range(4)]
    file_paths += [file_name + "_nlp" + str(i) for i in range(4)]

    class MyIterable:
        """ custom iteration """

        def __init__(self, file_paths):
            self._iter = None
            self._index = 0
            self._idx = 0
            self._file_paths = file_paths

        def __next__(self):
            if self._index >= len(self._file_paths) * 10:
                raise StopIteration
            if self._iter:
                try:
                    item = next(self._iter)
                    self._index += 1
                except StopIteration:
                    if self._idx >= len(self._file_paths):
                        raise StopIteration
                    self._iter = None
                    return next(self)
                return item
            logger.info("load <<< {}.".format(self._file_paths[self._idx]))
            self._iter = ds.MindDataset(self._file_paths[self._idx],
                                        columns_list, num_parallel_workers=num_readers,
                                        shuffle=None).create_tuple_iterator(num_epochs=1, output_numpy=True)
            self._idx += 1
            return next(self)

        def __iter__(self):
            self._index = 0
            self._idx = 0
            self._iter = None
            return self

        def __len__(self):
            return len(self._file_paths) * 10

    dataset = ds.GeneratorDataset(source=MyIterable(file_paths),
                                  column_names=["data", "file_name", "label"], num_parallel_workers=1)
    num_epochs = 1
    iter_ = dataset.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)
    num_iter = 0
    with pytest.raises(RuntimeError) as error_info:
        for _ in range(num_epochs):
            for item in iter_:
                print("item: ", item)
                num_iter += 1
    assert 'Invalid data, column name:' in str(error_info.value)


def test_two_level_pipeline_with_multiprocessing():
    """
    Feature: Test basic two level pipeline with multiprocessing testcases.
    Description: Test basic feature on two level pipeline with multiprocessing scenario.
    Expectation: Basic feature work fine.
    """
    file_name = "../data/dataset/testPK/data"

    class DatasetGenerator:
        def __init__(self):
            data1 = ds.ImageFolderDataset(file_name)
            data1 = data1.map(DatasetGenerator.pyfunc,
                              input_columns=["image"],
                              num_parallel_workers=2,
                              python_multiprocessing=True)
            self.iter = data1.create_tuple_iterator(output_numpy=True)

        def __getitem__(self, item):
            return next(self.iter)

        def __len__(self):
            return 10

        @staticmethod
        def pyfunc(x):
            return x

    source = DatasetGenerator()
    data2 = ds.GeneratorDataset(source, ["data", "label"])
    assert data2.output_shapes() == [[159109], []]

    data3 = ds.GeneratorDataset(source, ["data", "label"])
    assert data3.output_types() == [np.uint8, np.int32]

    data4 = ds.GeneratorDataset(source, ["data", "label"])
    assert data4.get_dataset_size() == 10
    nums = 0
    for _ in data4.create_dict_iterator(output_numpy=True):
        nums += 1
    assert nums == 10
