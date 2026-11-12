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
Test Food101 dataset operators
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.log as logger

DATA_DIR = "../data/dataset/testFood101Data/"


def test_food101_basic():
    """
    Feature: Food101 dataset
    Description: Read all files
    Expectation: Throw number of data in all files
    """
    logger.info("Test Food101Dataset Op")

    # case 1: test loading default usage dataset
    data1 = ds.Food101Dataset(DATA_DIR)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter1 += 1
    assert num_iter1 == 8

    # case 2: test num_samples
    data2 = ds.Food101Dataset(DATA_DIR, num_samples=1)
    num_iter2 = 0
    for _ in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter2 += 1
    assert num_iter2 == 1

    # case 3: test repeat
    data3 = ds.Food101Dataset(DATA_DIR, num_samples=2)
    data3 = data3.repeat(5)
    num_iter3 = 0
    for _ in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter3 += 1
    assert num_iter3 == 10


def test_food101_noshuffle():
    """
    Feature: Food101 dataset
    Description: Test noshuffle
    Expectation: Throw number of data in all files
    """
    logger.info("Test Case noShuffle")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    # Note: "all" reads both "train" dataset (2 samples) and "valid" dataset (2 samples)
    data1 = ds.Food101Dataset(DATA_DIR, shuffle=False)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    assert num_iter == 8


def test_food101_usage():
    """
    Feature: Food101 dataset
    Description: Test Usage
    Expectation: Throw number of data in all files
    """
    logger.info("Test Food101Dataset usage flag")

    def test_config(usage, food101_path=DATA_DIR):
        try:
            data = ds.Food101Dataset(food101_path, usage=usage)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    # test the usage of Food101
    assert test_config("test") == 4
    assert test_config("train") == 4
    assert test_config("all") == 8
    assert "usage is not within the valid set of ['train', 'test', 'all']" in test_config(
        "invalid")

    # change to the folder that contains all Food101 files
    all_food101 = None
    if all_food101 is not None:
        assert test_config("test", all_food101) == 25250
        assert test_config("train", all_food101) == 75750
        assert test_config("all", all_food101) == 101000
        assert ds.Food101Dataset(all_food101, usage="test").get_dataset_size() == 25250
        assert ds.Food101Dataset(all_food101, usage="train").get_dataset_size() == 75750
        assert ds.Food101Dataset(all_food101, usage="all").get_dataset_size() == 101000


def test_food101_sequential_sampler():
    """
    Feature: Food101 dataset
    Description: Test SequentialSampler
    Expectation: Get correct number of data
    """
    num_samples = 1
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data1 = ds.Food101Dataset(DATA_DIR, 'test', sampler=sampler)
    data2 = ds.Food101Dataset(DATA_DIR, 'test', shuffle=False, num_samples=num_samples)
    matches_list1, matches_list2 = [], []
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1), data2.create_dict_iterator(num_epochs=1)):
        matches_list1.append(item1["image"].asnumpy())
        matches_list2.append(item2["image"].asnumpy())
        num_iter += 1
    np.testing.assert_array_equal(matches_list1, matches_list2)
    assert num_iter == num_samples


def test_food101_pipeline():
    """
    Feature: Pipeline test
    Description: Read a sample
    Expectation: The amount of each function are equal
    """
    dataset = ds.Food101Dataset(DATA_DIR, "test", num_samples=1, decode=True)
    resize_op = vision.Resize((100, 100))
    dataset = dataset.map(input_columns=["image"], operations=resize_op)
    num_iter = 0
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    assert num_iter == 1
