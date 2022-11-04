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

DATA_DIR = '../data/dataset/testYahooAnswers/'


def test_yahoo_answers_dataset_basic():
    """
    Feature: YahooAnswersDataset.
    Description: Read data from train file.
    Expectation: The data is processed successfully.
    """

    buffer = []
    data = ds.YahooAnswersDataset(DATA_DIR, usage="train", shuffle=False)
    data = data.repeat(2)
    data = data.skip(3)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 5


def test_yahoo_answers_dataset_quoted():
    """
    Feature: YahooAnswersDataset.
    Description: Read the data and compare it to expectations.
    Expectation: The data is processed successfully.
    """

    data = ds.YahooAnswersDataset(DATA_DIR, usage="test", shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.extend([d['class'],
                       d['title'],
                       d['content'],
                       d['answer']])
    assert buffer == ["4", "My pet", "My pet is a toy bear.", "He is white.",
                      "1", "My favourite seasion", "My favorite season is summer.",
                      "In summer it is often sunny and hot."]


def test_yahoo_answers_dataset_usage():
    """
    Feature: YahooAnswersDataset.
    Description: Read all files with usage all.
    Expectation: The data is processed successfully.
    """

    buffer = []
    data = ds.YahooAnswersDataset(DATA_DIR, usage="all", shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 6


def test_yahoo_answers_dataset_get_datasetsize():
    """
    Feature: YahooAnswersDataset.
    Description: Test get_dataset_size function.
    Expectation: The data is processed successfully.
    """

    data = ds.YahooAnswersDataset(DATA_DIR, usage="test", shuffle=False)
    size = data.get_dataset_size()
    assert size == 2


def test_yahoo_answers_dataset_distribution():
    """
    Feature: YahooAnswersDataset.
    Description: Test in a distributed state.
    Expectation: The data is processed successfully.
    """

    data = ds.YahooAnswersDataset(DATA_DIR, usage="test", shuffle=False, num_shards=2, shard_id=0)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 1


def test_yahoo_answers_dataset_num_samples():
    """
    Feature: YahooAnswersDataset.
    Description: Test num_samples parameter.
    Expectation: The data is processed successfully.
    """

    data = ds.YahooAnswersDataset(DATA_DIR, usage="test", shuffle=False, num_samples=2)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_yahoo_answers_dataset_exception():
    """
    Feature: YahooAnswersDataset.
    Description: Test the wrong input.
    Expectation: Unable to read data properly.
    """

    def exception_func(item):
        raise Exception("Error occur!")
    try:
        data = ds.YahooAnswersDataset(DATA_DIR, usage="test", shuffle=False)
        data = data.map(operations=exception_func, input_columns=["class"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)
    try:
        data = ds.YahooAnswersDataset(DATA_DIR, usage="test", shuffle=False)
        data = data.map(operations=exception_func, input_columns=["content"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file is" in str(e)


if __name__ == "__main__":
    test_yahoo_answers_dataset_basic()
    test_yahoo_answers_dataset_quoted()
    test_yahoo_answers_dataset_usage()
    test_yahoo_answers_dataset_get_datasetsize()
    test_yahoo_answers_dataset_distribution()
    test_yahoo_answers_dataset_num_samples()
    test_yahoo_answers_dataset_exception()
