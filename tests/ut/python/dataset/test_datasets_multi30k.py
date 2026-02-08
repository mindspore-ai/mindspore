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
"""test dataset Multi30kDataset"""
import pytest

import mindspore.dataset as ds
from mindspore.dataset import text
from mindspore import log as logger
from util import config_get_set_num_parallel_workers, config_get_set_seed

INVALID_FILE = '../data/dataset/testMulti30kDataset/invalid_dir'
DATA_ALL_FILE = '../data/dataset/testMulti30kDataset'


def test_data_file_multi30k_text():
    """
    Feature: Test Multi30k Dataset.
    Description: Read data from a single file.
    Expectation: The data is processed successfully.
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(987)
    dataset = ds.Multi30kDataset(DATA_ALL_FILE, usage="train", shuffle=False)
    count = 0
    line = ["This is the first English sentence in train.",
            "This is the second English sentence in train.",
            "This is the third English sentence in train."
            ]
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 3
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_data_file_multi30k_translation():
    """
    Feature: Test Multi30k Dataset.
    Description: Read data from a single file.
    Expectation: The data is processed successfully.
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(987)
    dataset = ds.Multi30kDataset(DATA_ALL_FILE, usage="train", shuffle=False)
    count = 0
    line = ["This is the first Germany sentence in train.",
            "This is the second Germany sentence in train.",
            "This is the third Germany sentence in train."
            ]
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["translation"]
        assert strs == line[count]
        count += 1
    assert count == 3
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_all_file_multi30k():
    """
    Feature: Test Multi30k Dataset.
    Description: Read data from all file.
    Expectation: The data is processed successfully.
    """
    dataset = ds.Multi30kDataset(DATA_ALL_FILE)
    count = 0
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["text"]))
        count += 1
    assert count == 8


def test_dataset_num_samples_none():
    """
    Feature: Test Multi30k Dataset(num_samples = default).
    Description: Test get num_samples.
    Expectation: The data is processed successfully.
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(987)
    dataset = ds.Multi30kDataset(DATA_ALL_FILE, shuffle=False)
    count = 0
    line = ["This is the first English sentence in test.",
            "This is the second English sentence in test.",
            "This is the third English sentence in test.",
            "This is the first English sentence in train.",
            "This is the second English sentence in train.",
            "This is the third English sentence in train.",
            "This is the first English sentence in valid.",
            "This is the second English sentence in valid."
            ]
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 8
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_num_shards_multi30k():
    """
    Feature: Test Multi30k Dataset(num_shards = 3).
    Description: Test get num_samples.
    Expectation: The data is processed successfully.
    """
    dataset = ds.Multi30kDataset(DATA_ALL_FILE, usage='train', num_shards=3, shard_id=1)
    count = 0
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["text"]))
        count += 1
    assert count == 1


def test_multi30k_dataset_num_samples():
    """
    Feature: Test Multi30k Dataset(num_samples = 2).
    Description: Test get num_samples.
    Expectation: The data is processed successfully.
    """
    dataset = ds.Multi30kDataset(DATA_ALL_FILE, usage="test", num_samples=2)
    count = 0
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_multi30k_dataset_shuffle_files():
    """
    Feature: Test Multi30k Dataset.
    Description: Test get all files.
    Expectation: The data is processed successfully.
    """
    dataset = ds.Multi30kDataset(DATA_ALL_FILE, shuffle=True)
    count = 0
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 8


def test_multi30k_dataset_shuffle_false():
    """
    Feature: Test Multi30k Dataset (shuffle = false).
    Description: Test get all files.
    Expectation: The data is processed successfully.
    """
    dataset = ds.Multi30kDataset(DATA_ALL_FILE, shuffle=False)
    count = 0
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 8


def test_multi30k_dataset_repeat():
    """
    Feature: Test Multi30k in distribution (repeat 3 times).
    Description: Test in a distributed state.
    Expectation: The data is processed successfully.
    """
    dataset = ds.Multi30kDataset(DATA_ALL_FILE, usage='train')
    dataset = dataset.repeat(3)
    count = 0
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 9


def test_multi30k_dataset_get_datasetsize():
    """
    Feature: Test Getters.
    Description: Test get_dataset_size of Multi30k dataset.
    Expectation: The data is processed successfully.
    """
    dataset = ds.Multi30kDataset(DATA_ALL_FILE)
    size = dataset.get_dataset_size()
    assert size == 8


def test_multi30k_dataset_exceptions():
    """
    Feature: Test Multi30k Dataset.
    Description: Test exceptions.
    Expectation: Exception thrown to be caught
    """
    with pytest.raises(ValueError) as error_info:
        _ = ds.Multi30kDataset(INVALID_FILE)
    assert "The folder ../data/dataset/testMulti30kDataset/invalid_dir does not exist or is not a directory or" \
           " permission denied" in str(error_info.value)
    with pytest.raises(ValueError) as error_info:
        _ = ds.Multi30kDataset(DATA_ALL_FILE, usage="INVALID")
    assert "Input usage is not within the valid set of ['train', 'test', 'valid', 'all']." in str(error_info.value)
    with pytest.raises(ValueError) as error_info:
        _ = ds.Multi30kDataset(DATA_ALL_FILE, usage="test", language_pair=["ch", "ja"])
    assert "language_pair can only be ['en', 'de'] or ['en', 'de'], but got ['ch', 'ja']" in str(error_info.value)
    with pytest.raises(ValueError) as error_info:
        _ = ds.Multi30kDataset(DATA_ALL_FILE, usage="test", language_pair=["en", "en", "de"])
    assert "language_pair should be a list or tuple of length 2, but got 3" in str(error_info.value)
    with pytest.raises(ValueError) as error_info:
        _ = ds.Multi30kDataset(DATA_ALL_FILE, usage='test', num_samples=-1)
    assert "num_samples exceeds the boundary between 0 and 9223372036854775807(INT64_MAX)!" in str(error_info.value)


def test_multi30k_dataset_en_pipeline():
    """
    Feature: Multi30kDataset
    Description: Test Multi30kDataset in pipeline mode
    Expectation: The data is processed successfully
    """
    expected = ["this is the first english sentence in train.",
                "this is the second english sentence in train.",
                "this is the third english sentence in train."]
    dataset = ds.Multi30kDataset(DATA_ALL_FILE, 'train', shuffle=False)
    filter_wikipedia_xml_op = text.CaseFold()
    dataset = dataset.map(input_columns=["text"], operations=filter_wikipedia_xml_op, num_parallel_workers=1)
    count = 0
    for i in dataset.create_dict_iterator(output_numpy=True):
        strs = i["text"]
        assert strs == expected[count]
        count += 1


def test_multi30k_dataset_de_pipeline():
    """
    Feature: Multi30kDataset
    Description: Test Multi30kDataset in pipeline mode
    Expectation: The data is processed successfully
    """
    expected = ["this is the first germany sentence in train.",
                "this is the second germany sentence in train.",
                "this is the third germany sentence in train."]
    dataset = ds.Multi30kDataset(DATA_ALL_FILE, 'train', shuffle=False)
    filter_wikipedia_xml_op = text.CaseFold()
    dataset = dataset.map(input_columns=["translation"], operations=filter_wikipedia_xml_op, num_parallel_workers=1)
    count = 0
    for i in dataset.create_dict_iterator(output_numpy=True):
        strs = i["translation"]
        assert strs == expected[count]
        count += 1


if __name__ == "__main__":
    test_data_file_multi30k_text()
    test_data_file_multi30k_translation()
    test_all_file_multi30k()
    test_dataset_num_samples_none()
    test_num_shards_multi30k()
    test_multi30k_dataset_num_samples()
    test_multi30k_dataset_shuffle_files()
    test_multi30k_dataset_shuffle_false()
    test_multi30k_dataset_repeat()
    test_multi30k_dataset_get_datasetsize()
    test_multi30k_dataset_exceptions()
    test_multi30k_dataset_en_pipeline()
    test_multi30k_dataset_de_pipeline()
