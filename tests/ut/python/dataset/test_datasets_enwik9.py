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
import pytest

import mindspore.dataset as ds
from mindspore import log as logger
from util import config_get_set_num_parallel_workers, config_get_set_seed

DATA_FILE = "../data/dataset/testEnWik9Dataset"


def test_enwik9_total_rows_dataset_num_samples_none():
    """
    Feature: EnWik9Dataset
    Description: Test the function while param num_samples = 0
    Expectation: The number of samples is 13
    """
    # Do not provide a num_samples argument, so it would be None by default.
    data = ds.EnWik9Dataset(DATA_FILE)
    count = 0
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["text"]))
        count += 1
    assert count == 13


def test_enwik9_total_rows_dataset_shuffle_false_parallel_worker_two():
    """
    Feature: EnWik9Dataset
    Description: Test the function while param shuffle = False
    Expectation: The samples is ordered
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(2)
    original_seed = config_get_set_seed(987)
    data = ds.EnWik9Dataset(DATA_FILE, shuffle=False)
    count = 0
    line = ["  <page>",
            "    <title>MindSpore</title>",
            "    <id>1</id>",
            "    <revision>",
            "      <id>234</id>",
            "      <timestamp>2020-01-01T00:00:00Z</timestamp>",
            "      <contributor>",
            "        <username>MS</username>",
            "        <id>567</id>",
            "      </contributor>",
            "      <text xml:space=\"preserve\">666</text>",
            "    </revision>",
            "  </page>"]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 13
    # Restore configuration.
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_enwik9_total_rows_dataset_shuffle_false_parallel_worker_one():
    """
    Feature: EnWik9Dataset
    Description: Test the function while param shuffle = False
    Expectation: The samples is ordered
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(987)
    data = ds.EnWik9Dataset(DATA_FILE, shuffle=False)
    count = 0
    line = ["  <page>",
            "    <title>MindSpore</title>",
            "    <id>1</id>",
            "    <revision>",
            "      <id>234</id>",
            "      <timestamp>2020-01-01T00:00:00Z</timestamp>",
            "      <contributor>",
            "        <username>MS</username>",
            "        <id>567</id>",
            "      </contributor>",
            "      <text xml:space=\"preserve\">666</text>",
            "    </revision>",
            "  </page>"]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 13
    # Restore configuration.
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_enwik9_total_rows_dataset_shuffle_true_parallel_worker_two():
    """
    Feature: EnWik9Dataset
    Description: Test the function while param shuffle = True
    Expectation: The samples is disorder
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(2)
    original_seed = config_get_set_seed(135)
    data = ds.EnWik9Dataset(DATA_FILE, shuffle=True)
    count = 0
    line = ["        <username>MS</username>",
            "    <title>MindSpore</title>",
            "      <id>234</id>",
            "    </revision>",
            "      </contributor>",
            "    <revision>",
            "        <id>567</id>",
            "      <timestamp>2020-01-01T00:00:00Z</timestamp>",
            "    <id>1</id>",
            "  </page>",
            "  <page>",
            "      <text xml:space=\"preserve\">666</text>",
            "      <contributor>"]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 13
    # Restore configuration.
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_enwik9_total_rows_dataset_shuffle_true_parallel_worker_one():
    """
    Feature: EnWik9Dataset
    Description: Test the function while param shuffle = True
    Expectation: The samples is disorder
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(135)
    data = ds.EnWik9Dataset(DATA_FILE, shuffle=True)
    count = 0
    line = ["        <username>MS</username>",
            "    <title>MindSpore</title>",
            "      <id>234</id>",
            "    </revision>",
            "      </contributor>",
            "    <revision>",
            "        <id>567</id>",
            "      <timestamp>2020-01-01T00:00:00Z</timestamp>",
            "    <id>1</id>",
            "  </page>",
            "  <page>",
            "      <text xml:space=\"preserve\">666</text>",
            "      <contributor>"]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 13
    # Restore configuration.
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_enwik9_dataset_num_samples():
    """
    Feature: EnWik9Dataset
    Description: Test param num_samples, while it = 2
    Expectation: The number of samples = 2
    """
    data = ds.EnWik9Dataset(DATA_FILE, num_samples=2)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_enwik9_dataset_distribution():
    """
    Feature: EnWik9Dataset
    Description: Test distribution of the dataset
    Expectation: count = 7
    """
    data = ds.EnWik9Dataset(DATA_FILE, num_shards=2, shard_id=1)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 7


def test_enwik9_total_rows_dataset_repeat():
    """
    Feature: EnWik9Dataset
    Description: Test the function whie the samples are repeat
    Expectation: count = 26
    """
    data = ds.EnWik9Dataset(DATA_FILE, shuffle=False)
    data = data.repeat(2)
    count = 0
    line = ["  <page>",
            "    <title>MindSpore</title>",
            "    <id>1</id>",
            "    <revision>",
            "      <id>234</id>",
            "      <timestamp>2020-01-01T00:00:00Z</timestamp>",
            "      <contributor>",
            "        <username>MS</username>",
            "        <id>567</id>",
            "      </contributor>",
            "      <text xml:space=\"preserve\">666</text>",
            "    </revision>",
            "  </page>",
            "  <page>",
            "    <title>MindSpore</title>",
            "    <id>1</id>",
            "    <revision>",
            "      <id>234</id>",
            "      <timestamp>2020-01-01T00:00:00Z</timestamp>",
            "      <contributor>",
            "        <username>MS</username>",
            "        <id>567</id>",
            "      </contributor>",
            "      <text xml:space=\"preserve\">666</text>",
            "    </revision>",
            "  </page>"]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        strs = i["text"]
        assert strs == line[count]
        count += 1
    assert count == 26


def test_enwik9_total_rows_dataset_get_datasetsize():
    """
    Feature: EnWik9Dataset
    Description: Test the function, get_dataset_size()
    Expectation: size = 13
    """
    data = ds.EnWik9Dataset(DATA_FILE)
    size = data.get_dataset_size()
    assert size == 13


def test_enwik9_total_rows_dataset_device_que():
    """
    Feature: EnWik9Dataset
    Description: Test the function, device_que()
    Expectation: size = 13
    """
    data = ds.EnWik9Dataset(DATA_FILE, shuffle=False)
    data = data.device_que()
    data.send()


def test_enwik9_dataset_exceptions():
    """
    Feature: EnWik9Dataset
    Description: Test the errors which appear possibly
    Expectation: The errors are expected correctly
    """
    with pytest.raises(ValueError) as error_info:
        _ = ds.EnWik9Dataset("does/not/exist/")
    assert "does not exist or is not a directory or permission denied" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = ds.EnWik9Dataset("")
    assert "The folder  does not exist or is not a directory or permission denied" in str(error_info.value)

    def exception_func(item):
        raise Exception("Error occur!")
    with pytest.raises(RuntimeError) as error_info:
        data = ds.EnWik9Dataset(DATA_FILE)
        data = data.map(operations=exception_func, input_columns=["text"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    assert "map operation: [PyFunc] failed. The corresponding data files" in str(error_info.value)


if __name__ == "__main__":
    test_enwik9_total_rows_dataset_num_samples_none()
    test_enwik9_total_rows_dataset_shuffle_false_parallel_worker_two()
    test_enwik9_total_rows_dataset_shuffle_false_parallel_worker_one()
    test_enwik9_total_rows_dataset_shuffle_true_parallel_worker_two()
    test_enwik9_total_rows_dataset_shuffle_true_parallel_worker_one()
    test_enwik9_dataset_num_samples()
    test_enwik9_dataset_distribution()
    test_enwik9_total_rows_dataset_repeat()
    test_enwik9_total_rows_dataset_get_datasetsize()
    test_enwik9_total_rows_dataset_device_que()
    test_enwik9_dataset_exceptions()
