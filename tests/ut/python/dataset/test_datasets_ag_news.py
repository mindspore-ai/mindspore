# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
import mindspore.dataset as ds

FILE_DIR = '../data/dataset/testAGNews'


def test_ag_news_dataset_basic():
    """
    Feature: Test AG News Dataset.
    Description: Read data from a single file.
    Expectation: The data is processed successfully.
    """
    buffer = []
    data = ds.AGNewsDataset(FILE_DIR, usage='all', shuffle=False)
    data = data.repeat(2)
    data = data.skip(2)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 8


def test_ag_news_dataset_one_file():
    """
    Feature: Test AG News Dataset.
    Description: Read data from a single file.
    Expectation: The data is processed successfully.
    """
    data = ds.AGNewsDataset(FILE_DIR, usage='test', shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 2


def test_ag_news_dataset_all_file():
    """
    Feature: Test AG News Dataset(usage=all).
    Description: Read train data and test data.
    Expectation: The data is processed successfully.
    """
    buffer = []
    data = ds.AGNewsDataset(FILE_DIR, usage='all', shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 5


def test_ag_news_dataset_num_samples():
    """
    Feature: Test AG News Dataset.
    Description: Read data from a single file.
    Expectation: The data is processed successfully.
    """
    data = ds.AGNewsDataset(FILE_DIR, usage='all', num_samples=4, shuffle=False)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 4


def test_ag_news_dataset_distribution():
    """
    Feature: Test AG News Dataset.
    Description: Read data from a single file.
    Expectation: The data is processed successfully.
    """
    data = ds.AGNewsDataset(FILE_DIR, usage='test', shuffle=False, num_shards=2, shard_id=0)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 1


def test_ag_news_dataset_quoted():
    """
    Feature: Test get the AG News Dataset.
    Description: Read AGNewsDataset data and get data.
    Expectation: The data is processed successfully.
    """
    data = ds.AGNewsDataset(FILE_DIR, usage='test', shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.extend([d['index'],
                       d['title'],
                       d['description']])
    assert buffer == ["3", "Background of the selection",
                      "In this day and age, the internet is growing rapidly, "
                      "the total number of connected devices is increasing and "
                      "we are entering the era of big data.",
                      "4", "Related technologies",
                      "\"Leaflet is the leading open source JavaScript library "
                      "for mobile-friendly interactive maps.\""]


def test_ag_news_dataset_size():
    """
    Feature: Test Getters.
    Description: Test get_dataset_size of AG News dataset.
    Expectation: The data is processed successfully.
    """
    data = ds.AGNewsDataset(FILE_DIR, usage='test', shuffle=False)
    assert data.get_dataset_size() == 2


def test_ag_news_dataset_exception():
    """
    Feature: Error Test.
    Description: Test the wrong input.
    Expectation: Unable to read in data.
    """
    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.AGNewsDataset(FILE_DIR, usage='test', shuffle=False)
        data = data.map(operations=exception_func, input_columns=["index"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.AGNewsDataset(FILE_DIR, usage='test', shuffle=False)
        data = data.map(operations=exception_func, input_columns=["title"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)

    try:
        data = ds.AGNewsDataset(FILE_DIR, usage='test', shuffle=False)
        data = data.map(operations=exception_func, input_columns=["description"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)


if __name__ == "__main__":
    test_ag_news_dataset_basic()
    test_ag_news_dataset_one_file()
    test_ag_news_dataset_all_file()
    test_ag_news_dataset_num_samples()
    test_ag_news_dataset_distribution()
    test_ag_news_dataset_quoted()
    test_ag_news_dataset_size()
    test_ag_news_dataset_exception()
