# Copyright 2020 Huawei Technologies Co., Ltd
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


DATA_FILE = "../data/dataset/testTextFileDataset/1.txt"
DATA_ALL_FILE = "../data/dataset/testTextFileDataset/*"


def test_textline_dataset_one_file():
    data = ds.TextFileDataset(DATA_FILE)
    count = 0
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["text"]))
        count += 1
    assert count == 3


def test_textline_dataset_all_file():
    data = ds.TextFileDataset(DATA_ALL_FILE)
    count = 0
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["text"]))
        count += 1
    assert count == 5


def test_textline_dataset_num_samples_none():
    # Do not provide a num_samples argument, so it would be None by default
    data = ds.TextFileDataset(DATA_FILE)
    count = 0
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["text"]))
        count += 1
    assert count == 3


def test_textline_dataset_shuffle_false4():
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)
    original_seed = config_get_set_seed(987)
    data = ds.TextFileDataset(DATA_ALL_FILE, shuffle=False)
    count = 0
    line = ["This is a text file.", "Another file.",
            "Be happy every day.", "End of file.", "Good luck to everyone."]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"].item().decode("utf8")
        assert strs == line[count]
        count += 1
    assert count == 5
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_textline_dataset_shuffle_false1():
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(987)
    data = ds.TextFileDataset(DATA_ALL_FILE, shuffle=False)
    count = 0
    line = ["This is a text file.", "Be happy every day.", "Good luck to everyone.",
            "Another file.", "End of file."]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"].item().decode("utf8")
        assert strs == line[count]
        count += 1
    assert count == 5
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_textline_dataset_shuffle_files4():
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)
    original_seed = config_get_set_seed(135)
    data = ds.TextFileDataset(DATA_ALL_FILE, shuffle=ds.Shuffle.FILES)
    count = 0
    line = ["This is a text file.", "Another file.",
            "Be happy every day.", "End of file.", "Good luck to everyone."]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"].item().decode("utf8")
        assert strs == line[count]
        count += 1
    assert count == 5
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_textline_dataset_shuffle_files1():
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(135)
    data = ds.TextFileDataset(DATA_ALL_FILE, shuffle=ds.Shuffle.FILES)
    count = 0
    line = ["This is a text file.", "Be happy every day.", "Good luck to everyone.",
            "Another file.", "End of file."]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"].item().decode("utf8")
        assert strs == line[count]
        count += 1
    assert count == 5
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_textline_dataset_shuffle_global4():
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)
    original_seed = config_get_set_seed(246)
    data = ds.TextFileDataset(DATA_ALL_FILE, shuffle=ds.Shuffle.GLOBAL)
    count = 0
    line = ["Another file.", "Good luck to everyone.", "End of file.",
            "This is a text file.", "Be happy every day."]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"].item().decode("utf8")
        assert strs == line[count]
        count += 1
    assert count == 5
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_textline_dataset_shuffle_global1():
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(246)
    data = ds.TextFileDataset(DATA_ALL_FILE, shuffle=ds.Shuffle.GLOBAL)
    count = 0
    line = ["Another file.", "Good luck to everyone.", "This is a text file.",
            "End of file.", "Be happy every day."]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"].item().decode("utf8")
        assert strs == line[count]
        count += 1
    assert count == 5
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_textline_dataset_num_samples():
    data = ds.TextFileDataset(DATA_FILE, num_samples=2)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_textline_dataset_distribution():
    data = ds.TextFileDataset(DATA_ALL_FILE, num_shards=2, shard_id=1)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 3


def test_textline_dataset_repeat():
    data = ds.TextFileDataset(DATA_FILE, shuffle=False)
    data = data.repeat(3)
    count = 0
    line = ["This is a text file.", "Be happy every day.", "Good luck to everyone.",
            "This is a text file.", "Be happy every day.", "Good luck to everyone.",
            "This is a text file.", "Be happy every day.", "Good luck to everyone."]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"].item().decode("utf8")
        assert strs == line[count]
        count += 1
    assert count == 9


def test_textline_dataset_get_datasetsize():
    data = ds.TextFileDataset(DATA_FILE)
    size = data.get_dataset_size()
    assert size == 3

def test_textline_dataset_to_device():
    data = ds.TextFileDataset(DATA_FILE, shuffle=False)
    data = data.to_device()
    data.send()

def test_textline_dataset_exceptions():
    with pytest.raises(ValueError) as error_info:
        _ = ds.TextFileDataset(DATA_FILE, num_samples=-1)
    assert "num_samples exceeds the boundary" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = ds.TextFileDataset("does/not/exist/no.txt")
    assert "The following patterns did not match any files" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = ds.TextFileDataset("")
    assert "The following patterns did not match any files" in str(error_info.value)

    def exception_func(item):
        raise Exception("Error occur!")
    with pytest.raises(RuntimeError) as error_info:
        data = ds.TextFileDataset(DATA_FILE)
        data = data.map(operations=exception_func, input_columns=["text"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    assert "map operation: [PyFunc] failed. The corresponding data files" in str(error_info.value)


if __name__ == "__main__":
    test_textline_dataset_one_file()
    test_textline_dataset_all_file()
    test_textline_dataset_num_samples_none()
    test_textline_dataset_shuffle_false4()
    test_textline_dataset_shuffle_false1()
    test_textline_dataset_shuffle_files4()
    test_textline_dataset_shuffle_files1()
    test_textline_dataset_shuffle_global4()
    test_textline_dataset_shuffle_global1()
    test_textline_dataset_num_samples()
    test_textline_dataset_distribution()
    test_textline_dataset_repeat()
    test_textline_dataset_get_datasetsize()
    test_textline_dataset_to_device()
    test_textline_dataset_exceptions()
