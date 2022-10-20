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
import pytest

import mindspore.dataset as ds
from mindspore import log as logger
from util import config_get_set_num_parallel_workers, config_get_set_seed

FILE_DIR = '../data/dataset/testPennTreebank'


def test_penn_treebank_dataset_one_file():
    """
    Feature: Test PennTreebank Dataset.
    Description: Read data from a single file.
    Expectation: The data is processed successfully.
    """
    data = ds.PennTreebankDataset(FILE_DIR, usage='test')
    count = 0
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["text"]))
        count += 1
    assert count == 3


def test_penn_treebank_dataset_train():
    """
    Feature: Test PennTreebank Dataset.
    Description: Read data from a single file.
    Expectation: The data is processed successfully.
    """
    data = ds.PennTreebankDataset(FILE_DIR, usage='train')
    count = 0
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["text"]))
        count += 1
    assert count == 3


def test_penn_treebank_dataset_valid():
    """
    Feature: Test PennTreebank Dataset.
    Description: Read data from a single file.
    Expectation: The data is processed successfully.
    """
    data = ds.PennTreebankDataset(FILE_DIR, usage='valid')
    count = 0
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["text"]))
        count += 1
    assert count == 3


def test_penn_treebank_dataset_all_file():
    """
    Feature: Test PennTreebank Dataset.
    Description: Read data from a single file.
    Expectation: The data is processed successfully.
    """
    data = ds.PennTreebankDataset(FILE_DIR, usage='all')
    count = 0
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["text"]))
        count += 1
    assert count == 9


def test_penn_treebank_dataset_num_samples_none():
    """
    Feature: Test PennTreebank Dataset.
    Description: Read data with no num_samples input.
    Expectation: The data is processed successfully.
    """
    # Do not provide a num_samples argument, so it would be None by default
    data = ds.PennTreebankDataset(FILE_DIR, usage='all')
    count = 0
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["text"]))
        count += 1
    assert count == 9


def test_penn_treebank_dataset_shuffle_false4():
    """
    Feature: Test PennTreebank Dataset.
    Description: Read data from a single file with shulle is false.
    Expectation: The data is processed successfully.
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)
    original_seed = config_get_set_seed(987)
    data = ds.PennTreebankDataset(FILE_DIR, usage='all', shuffle=False)
    count = 0
    line = [" no it was black friday ",
            " does the bank charge a fee for setting up the account ",
            " just ahead of them there was a huge fissure ",
            " clash twits poetry formulate flip loyalty splash ",
            " <unk> the wardrobe was very small in our room ",
            " <unk> <unk> the proportion of female workers in this company <unk> <unk> ",
            " you pay less for the supermaket's own brands ",
            " black white grapes ",
            " everyone in our football team is fuming "]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 9
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_penn_treebank_dataset_shuffle_false1():
    """
    Feature: Test PennTreebank Dataset.
    Description: Read data from a single file with shulle is false.
    Expectation: The data is processed successfully.
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(987)
    data = ds.PennTreebankDataset(FILE_DIR, usage='all', shuffle=False)
    count = 0
    line = [" no it was black friday ",
            " clash twits poetry formulate flip loyalty splash ",
            " you pay less for the supermaket's own brands ",
            " does the bank charge a fee for setting up the account ",
            " <unk> the wardrobe was very small in our room ",
            " black white grapes ",
            " just ahead of them there was a huge fissure ",
            " <unk> <unk> the proportion of female workers in this company <unk> <unk> ",
            " everyone in our football team is fuming "]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 9
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_penn_treebank_dataset_shuffle_files4():
    """
    Feature: Test PennTreebank Dataset.
    Description: Read data from a single file with shulle is files.
    Expectation: The data is processed successfully.
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)
    original_seed = config_get_set_seed(135)
    data = ds.PennTreebankDataset(FILE_DIR, usage='all', shuffle=ds.Shuffle.FILES)
    count = 0
    line = [" just ahead of them there was a huge fissure ",
            " does the bank charge a fee for setting up the account ",
            " no it was black friday ",
            " <unk> <unk> the proportion of female workers in this company <unk> <unk> ",
            " <unk> the wardrobe was very small in our room ",
            " clash twits poetry formulate flip loyalty splash ",
            " everyone in our football team is fuming ",
            " black white grapes ",
            " you pay less for the supermaket's own brands "]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 9
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_penn_treebank_dataset_shuffle_files1():
    """
    Feature: Test PennTreebank Dataset.
    Description: Read data from a single file with shulle is files.
    Expectation: The data is processed successfully.
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(135)
    data = ds.PennTreebankDataset(FILE_DIR, usage='all', shuffle=ds.Shuffle.FILES)
    count = 0
    line = [" just ahead of them there was a huge fissure ",
            " <unk> <unk> the proportion of female workers in this company <unk> <unk> ",
            " everyone in our football team is fuming ",
            " does the bank charge a fee for setting up the account ",
            " <unk> the wardrobe was very small in our room ",
            " black white grapes ",
            " no it was black friday ",
            " clash twits poetry formulate flip loyalty splash ",
            " you pay less for the supermaket's own brands "]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 9
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_penn_treebank_dataset_shuffle_global4():
    """
    Feature: Test PennTreebank Dataset.
    Description: Read data from a single file with shulle is global.
    Expectation: The data is processed successfully.
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)
    original_seed = config_get_set_seed(246)
    data = ds.PennTreebankDataset(FILE_DIR, usage='all', shuffle=ds.Shuffle.GLOBAL)
    count = 0
    line = [" everyone in our football team is fuming ",
            " does the bank charge a fee for setting up the account ",
            " clash twits poetry formulate flip loyalty splash ",
            " no it was black friday ",
            " just ahead of them there was a huge fissure ",
            " <unk> <unk> the proportion of female workers in this company <unk> <unk> ",
            " you pay less for the supermaket's own brands ",
            " <unk> the wardrobe was very small in our room ",
            " black white grapes "]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 9
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_penn_treebank_dataset_shuffle_global1():
    """
    Feature: Test PennTreebank Dataset.
    Description: Read data from a single file with shulle is global.
    Expectation: The data is processed successfully.
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(246)
    data = ds.PennTreebankDataset(FILE_DIR, usage='all', shuffle=ds.Shuffle.GLOBAL)
    count = 0
    line = [" everyone in our football team is fuming ",
            " does the bank charge a fee for setting up the account ",
            " clash twits poetry formulate flip loyalty splash ",
            " <unk> the wardrobe was very small in our room ",
            " black white grapes ",
            " you pay less for the supermaket's own brands ",
            " <unk> <unk> the proportion of female workers in this company <unk> <unk> ",
            " no it was black friday ",
            " just ahead of them there was a huge fissure "]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 9
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_penn_treebank_dataset_num_samples():
    """
    Feature: Test PennTreebank Dataset.
    Description: Test num_samples.
    Expectation: The data is processed successfully.
    """
    data = ds.PennTreebankDataset(FILE_DIR, usage='all', num_samples=2)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_penn_treebank_dataset_distribution():
    """
    Feature: Test PennTreebank Dataset.
    Description: Read data from a single file.
    Expectation: The data is processed successfully.
    """
    data = ds.PennTreebankDataset(FILE_DIR, usage='all', num_shards=2, shard_id=1)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 5


def test_penn_treebank_dataset_repeat():
    """
    Feature: Test PennTreebank Dataset.
    Description: Test repeat.
    Expectation: The data is processed successfully.
    """
    data = ds.PennTreebankDataset(FILE_DIR, usage='test', shuffle=False)
    data = data.repeat(3)
    count = 0
    line = [" no it was black friday ",
            " clash twits poetry formulate flip loyalty splash ",
            " you pay less for the supermaket's own brands ",
            " no it was black friday ",
            " clash twits poetry formulate flip loyalty splash ",
            " you pay less for the supermaket's own brands ",
            " no it was black friday ",
            " clash twits poetry formulate flip loyalty splash ",
            " you pay less for the supermaket's own brands ",]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 9


def test_penn_treebank_dataset_get_datasetsize():
    """
    Feature: Test PennTreebank Dataset.
    Description: Test get_datasetsize.
    Expectation: The data is processed successfully.
    """
    data = ds.PennTreebankDataset(FILE_DIR, usage='test')
    size = data.get_dataset_size()
    assert size == 3


def test_penn_treebank_dataset_device_que():
    """
    Feature: Test PennTreebank Dataset.
    Description: Test device_que.
    Expectation: The data is processed successfully.
    """
    data = ds.PennTreebankDataset(FILE_DIR, usage='test')
    data = data.device_que()
    data.send()


def test_penn_treebank_dataset_exceptions():
    """
    Feature: Test PennTreebank Dataset.
    Description: Test exceptions.
    Expectation: Exception thrown to be caught
    """
    with pytest.raises(ValueError) as error_info:
        _ = ds.PennTreebankDataset(FILE_DIR, usage='test', num_samples=-1)
    assert "num_samples exceeds the boundary" in str(error_info.value)
    with pytest.raises(ValueError) as error_info:
        _ = ds.PennTreebankDataset("does/not/exist/no.txt")
    assert str(error_info.value)
    with pytest.raises(ValueError) as error_info:
        _ = ds.PennTreebankDataset("")
    assert  str(error_info.value)
    def exception_func(item):
        raise Exception("Error occur!")
    with pytest.raises(RuntimeError) as error_info:
        data = ds.PennTreebankDataset(FILE_DIR)
        data = data.map(operations=exception_func, input_columns=["text"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    assert "map operation: [PyFunc] failed. The corresponding data file is" in str(error_info.value)


if __name__ == "__main__":
    test_penn_treebank_dataset_one_file()
    test_penn_treebank_dataset_train()
    test_penn_treebank_dataset_valid()
    test_penn_treebank_dataset_all_file()
    test_penn_treebank_dataset_num_samples_none()
    test_penn_treebank_dataset_shuffle_false4()
    test_penn_treebank_dataset_shuffle_false1()
    test_penn_treebank_dataset_shuffle_files4()
    test_penn_treebank_dataset_shuffle_files1()
    test_penn_treebank_dataset_shuffle_global4()
    test_penn_treebank_dataset_shuffle_global1()
    test_penn_treebank_dataset_num_samples()
    test_penn_treebank_dataset_distribution()
    test_penn_treebank_dataset_repeat()
    test_penn_treebank_dataset_get_datasetsize()
    test_penn_treebank_dataset_device_que()
    test_penn_treebank_dataset_exceptions()
