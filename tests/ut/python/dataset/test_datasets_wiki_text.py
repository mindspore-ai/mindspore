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

FILE_DIR = '../data/dataset/testWikiText'


def test_wiki_text_dataset_test():
    """
    Feature: Test WikiText Dataset.
    Description: Read test data from a single file.
    Expectation: The data is processed successfully.
    """
    data = ds.WikiTextDataset(FILE_DIR, usage='test', shuffle=False)
    count = 0
    test_content = [" no it was black friday ", " I am happy ", " finish math homework "]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["text"]))
        strs = i["text"]
        assert strs == test_content[count]
        count += 1
    assert count == 3


def test_wiki_text_dataset_train():
    """
    Feature: Test WikiText Dataset.
    Description: Read train data from a single file.
    Expectation: The data is processed successfully.
    """
    data = ds.WikiTextDataset(FILE_DIR, usage='train', shuffle=False)
    count = 0
    train_content = [" go to china ", " I lova MindSpore ", " black white grapes "]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["text"]))
        strs = i["text"]
        assert strs == train_content[count]
        count += 1
    assert count == 3


def test_wiki_text_dataset_valid():
    """
    Feature: Test WikiText Dataset.
    Description: Read valid data from a single file.
    Expectation: The data is processed successfully.
    """
    data = ds.WikiTextDataset(FILE_DIR, usage='valid', shuffle=False)
    count = 0
    valid_content = [" just ahead of them there was a huge fissure ", " zhejiang, china ", " MindSpore Ascend "]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["text"]))
        strs = i["text"]
        assert strs == valid_content[count]
        count += 1
    assert count == 3


def test_wiki_text_dataset_all_file():
    """
    Feature: Test WikiText Dataset.
    Description: Read data from all files.
    Expectation: The data is processed successfully.
    """
    data = ds.WikiTextDataset(FILE_DIR, usage='all')
    count = 0
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["text"]))
        count += 1
    assert count == 9


def test_wiki_text_dataset_num_samples_none():
    """
    Feature: Test WikiText Dataset.
    Description: Read data with no num_samples input.
    Expectation: The data is processed successfully.
    """
    # Do not provide a num_samples argument, so it would be None by default, which means all samples are read.
    data = ds.WikiTextDataset(FILE_DIR, usage='all')
    count = 0
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["text"]))
        count += 1
    assert count == 9


def test_wiki_text_dataset_shuffle_false_and_workers_4():
    """
    Feature: Test WikiText Dataset.
    Description: Read data from a single file with shuffle is False and num_parallel_workers=4.
    Expectation: The data is processed successfully.
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)
    original_seed = config_get_set_seed(987)
    data = ds.WikiTextDataset(FILE_DIR, usage='all', shuffle=False)
    count = 0
    line = [" no it was black friday ",
            " go to china ",
            " just ahead of them there was a huge fissure ",
            " I am happy ",
            " I lova MindSpore ",
            " zhejiang, china ",
            " finish math homework ",
            " black white grapes ",
            " MindSpore Ascend "]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 9
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_wiki_text_dataset_shuffle_false_and_workers_1():
    """
    Feature: Test WikiText Dataset.
    Description: Read data from a single file with shuffle is False and num_parallel_workers is 1.
    Expectation: The data is processed successfully.
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(987)
    data = ds.WikiTextDataset(FILE_DIR, usage='all', shuffle=False)
    count = 0
    line = [" no it was black friday ",
            " I am happy ",
            " finish math homework ",
            " go to china ",
            " I lova MindSpore ",
            " black white grapes ",
            " just ahead of them there was a huge fissure ",
            " zhejiang, china ",
            " MindSpore Ascend "]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 9
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_wiki_text_dataset_shuffle_files_and_workers_4():
    """
    Feature: Test WikiText Dataset.
    Description: Read data from a single file with shuffle is files and num_parallel_workers is 4.
    Expectation: The data is processed successfully.
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)
    original_seed = config_get_set_seed(135)
    data = ds.WikiTextDataset(FILE_DIR, usage='all', shuffle=ds.Shuffle.FILES)
    count = 0
    line = [" just ahead of them there was a huge fissure ",
            " go to china ",
            " no it was black friday ",
            " zhejiang, china ",
            " I lova MindSpore ",
            " I am happy ",
            " MindSpore Ascend ",
            " black white grapes ",
            " finish math homework "]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 9
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_wiki_text_dataset_shuffle_files_and_workers_1():
    """
    Feature: Test WikiText Dataset.
    Description: Read data from a single file with shuffle is files and num_parallel_workers is 1.
    Expectation: The data is processed successfully.
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(135)
    data = ds.WikiTextDataset(FILE_DIR, usage='all', shuffle=ds.Shuffle.FILES)
    count = 0
    line = [" just ahead of them there was a huge fissure ",
            " zhejiang, china ",
            " MindSpore Ascend ",
            " go to china ",
            " I lova MindSpore ",
            " black white grapes ",
            " no it was black friday ",
            " I am happy ",
            " finish math homework "]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 9
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_wiki_text_dataset_shuffle_global4():
    """
    Feature: Test WikiText Dataset.
    Description: Read data from a single file with shuffle is global.
    Expectation: The data is processed successfully.
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)
    original_seed = config_get_set_seed(246)
    data = ds.WikiTextDataset(FILE_DIR, usage='all', shuffle=ds.Shuffle.GLOBAL)
    count = 0
    line = [" MindSpore Ascend ",
            " go to china ",
            " I am happy ",
            " no it was black friday ",
            " just ahead of them there was a huge fissure ",
            " zhejiang, china ",
            " finish math homework ",
            " I lova MindSpore ",
            " black white grapes "]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 9
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_wiki_text_dataset_shuffle_global1():
    """
    Feature: Test WikiText Dataset.
    Description: Read data from a single file with shuffle is global.
    Expectation: The data is processed successfully.
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(246)
    data = ds.WikiTextDataset(FILE_DIR, usage='all', shuffle=ds.Shuffle.GLOBAL)
    count = 0
    line = [" MindSpore Ascend ",
            " go to china ",
            " I am happy ",
            " I lova MindSpore ",
            " black white grapes ",
            " finish math homework ",
            " zhejiang, china ",
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


def test_wiki_text_dataset_num_samples():
    """
    Feature: Test WikiText Dataset.
    Description: Test num_samples.
    Expectation: The data is processed successfully.
    """
    data = ds.WikiTextDataset(FILE_DIR, usage='all', num_samples=2)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_wiki_text_dataset_distribution():
    """
    Feature: Test WikiText Dataset.
    Description: Read data from a single file.
    Expectation: The data is processed successfully.
    """
    data = ds.WikiTextDataset(FILE_DIR, usage='all', num_shards=2, shard_id=1)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 5


def test_wiki_text_dataset_repeat():
    """
    Feature: Test WikiText Dataset.
    Description: Test repeat.
    Expectation: The data is processed successfully.
    """
    data = ds.WikiTextDataset(FILE_DIR, usage='test', shuffle=False)
    data = data.repeat(3)
    count = 0
    line = [" no it was black friday ",
            " I am happy ",
            " finish math homework ",
            " no it was black friday ",
            " I am happy ",
            " finish math homework ",
            " no it was black friday ",
            " I am happy ",
            " finish math homework ",]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 9


def test_wiki_text_dataset_get_datasetsize():
    """
    Feature: Test WikiText Dataset.
    Description: Test get_datasetsize.
    Expectation: The data is processed successfully.
    """
    data = ds.WikiTextDataset(FILE_DIR, usage='test')
    size = data.get_dataset_size()
    assert size == 3


def test_wiki_text_dataset_device_que():
    """
    Feature: Test WikiText Dataset.
    Description: Test device_que.
    Expectation: The data is processed successfully.
    """
    data = ds.WikiTextDataset(FILE_DIR, usage='test')
    data = data.device_que()
    data.send()


def test_wiki_text_dataset_exceptions():
    """
    Feature: Test WikiText Dataset.
    Description: Test exceptions.
    Expectation: Exception thrown to be caught
    """
    with pytest.raises(ValueError) as error_info:
        _ = ds.WikiTextDataset(FILE_DIR, usage='test', num_samples=-1)
    assert "num_samples exceeds the boundary" in str(error_info.value)
    with pytest.raises(ValueError) as error_info:
        _ = ds.WikiTextDataset("does/not/exist/no.txt")
    assert str(error_info.value)
    with pytest.raises(ValueError) as error_info:
        _ = ds.WikiTextDataset("")
    assert  str(error_info.value)
    def exception_func(item):
        raise Exception("Error occur!")
    with pytest.raises(RuntimeError) as error_info:
        data = ds.WikiTextDataset(FILE_DIR)
        data = data.map(operations=exception_func, input_columns=["text"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    assert "map operation: [PyFunc] failed. The corresponding data file is" in str(error_info.value)


if __name__ == "__main__":
    test_wiki_text_dataset_test()
    test_wiki_text_dataset_train()
    test_wiki_text_dataset_valid()
    test_wiki_text_dataset_all_file()
    test_wiki_text_dataset_num_samples_none()
    test_wiki_text_dataset_shuffle_false_and_workers_4()
    test_wiki_text_dataset_shuffle_false_and_workers_1()
    test_wiki_text_dataset_shuffle_files_and_workers_4()
    test_wiki_text_dataset_shuffle_files_and_workers_1()
    test_wiki_text_dataset_shuffle_global4()
    test_wiki_text_dataset_shuffle_global1()
    test_wiki_text_dataset_num_samples()
    test_wiki_text_dataset_distribution()
    test_wiki_text_dataset_repeat()
    test_wiki_text_dataset_get_datasetsize()
    test_wiki_text_dataset_device_que()
    test_wiki_text_dataset_exceptions()
