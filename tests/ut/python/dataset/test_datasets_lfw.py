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
import re
import pytest

import mindspore.dataset as ds
from mindspore import log as logger

DATA_DIR = "../data/dataset/testLFW"


def test_lfw_basic():
    """
    Feature: LFW
    Description: Test basic usage of LFW
    Expectation: The dataset is as expected
    """
    logger.info("Test Case basic")
    # define parameters.
    repeat_count = 2

    # apply dataset operations.
    data1 = ds.LFWDataset(DATA_DIR)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 8


def test_lfw_task():
    """
    Feature: LFW
    Description: Test basic usage of LFW
    Expectation: The dataset is as expected
    """
    logger.info("Test Case basic")
    # define parameters.
    repeat_count = 2

    # apply dataset operations.
    data1 = ds.LFWDataset(DATA_DIR, task="pairs", usage="all")
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        logger.info("image1 is {}".format(item["image1"]))
        logger.info("image2 is {}".format(item["image2"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 16


def test_lfw_usage():
    """
    Feature: LFW
    Description: Test basic usage of LFW
    Expectation: The dataset is as expected
    """
    logger.info("Test Case basic")
    # define parameters.
    repeat_count = 2

    # apply dataset operations.
    data1 = ds.LFWDataset(DATA_DIR, usage="test")
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 6


def test_lfw_image_set():
    """
    Feature: LFW
    Description: Test basic usage of LFW
    Expectation: The dataset is as expected
    """
    logger.info("Test Case basic")
    # define parameters.
    repeat_count = 2

    # apply dataset operations.
    data1 = ds.LFWDataset(DATA_DIR, image_set="funneled")
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 8


def test_lfw_num_samples():
    """
    Feature: LFW
    Description: Test basic usage of LFW
    Expectation: The dataset is as expected
    """
    logger.info("Test Case numSamples")
    # define parameters.
    repeat_count = 2

    # apply dataset operations.
    data1 = ds.LFWDataset(DATA_DIR, num_samples=4, num_parallel_workers=2)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 8

    random_sampler = ds.RandomSampler(num_samples=2, replacement=True)
    data1 = ds.LFWDataset(DATA_DIR, num_parallel_workers=2,
                          sampler=random_sampler)

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    assert num_iter == 2

    random_sampler = ds.RandomSampler(num_samples=3, replacement=False)
    data1 = ds.LFWDataset(DATA_DIR, num_parallel_workers=2,
                          sampler=random_sampler)

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    assert num_iter == 3


def test_lfw_num_shards():
    """
    Feature: LFW
    Description: Test basic usage of LFW
    Expectation: The dataset is as expected
    """
    logger.info("Test Case numShards")
    # define parameters.
    repeat_count = 2

    # apply dataset operations.
    data1 = ds.LFWDataset(DATA_DIR, num_shards=5, shard_id=1)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 2


def test_lfw_shard_id():
    """
    Feature: LFW
    Description: Test basic usage of LFW
    Expectation: The dataset is as expected
    """
    logger.info("Test Case withShardID")
    # define parameters.
    repeat_count = 2

    # apply dataset operations.
    data1 = ds.LFWDataset(DATA_DIR, num_shards=4, shard_id=1)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 2


def test_lfw_no_shuffle():
    """
    Feature: LFW
    Description: Test dataset of LFW
    Expectation: The dataset is as expected
    """
    logger.info("Test Case noShuffle")
    # define parameters.
    repeat_count = 2

    # apply dataset operations.
    data1 = ds.LFWDataset(DATA_DIR, shuffle=False)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 8


def test_lfw_decode():
    """
    Feature: LFW
    Description: Test basic usage of LFW
    Expectation: The dataset is as expected
    """
    logger.info("Test Case decode")
    # define parameters.
    repeat_count = 2

    # apply dataset operations.
    data1 = ds.LFWDataset(DATA_DIR, decode=True)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 8


def test_sequential_sampler():
    """
    Feature: LFW
    Description: Test basic usage of LFW
    Expectation: The dataset is as expected
    """
    logger.info("Test Case SequentialSampler")
    # define parameters.
    repeat_count = 2
    # apply dataset operations.
    sampler = ds.SequentialSampler(num_samples=3)
    data1 = ds.LFWDataset(DATA_DIR, sampler=sampler)
    data1 = data1.repeat(repeat_count)
    result = []
    num_iter = 0
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        result.append(item["label"])
        num_iter += 1

    assert num_iter == 6
    logger.info("Result: {}".format(result))


def test_random_and_sequentialchained_sampler():
    """
    Feature: LFW
    Description: Test basic usage of LFW
    Expectation: The dataset is as expected
    """
    logger.info("Test Case Chained Sampler - Random and Sequential, with repeat")

    # Create chained sampler, random and sequential.
    sampler = ds.RandomSampler()
    child_sampler = ds.SequentialSampler()
    sampler.add_child(child_sampler)
    # Create LFWDataset with sampler.
    data1 = ds.LFWDataset(DATA_DIR, sampler=sampler)

    data1 = data1.repeat(count=3)

    # Verify dataset size.
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 12

    # Verify number of iterations.
    num_iter = 0
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 12


def test_lfw_rename():
    """
    Feature: LFW
    Description: Test basic usage of LFW
    Expectation: The dataset is as expected
    """
    logger.info("Test Case rename")
    # define parameters.
    repeat_count = 2

    # apply dataset operations.
    data1 = ds.LFWDataset(DATA_DIR, num_samples=4)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 8

    data1 = data1.rename(input_columns=["image"], output_columns="image2")

    num_iter = 0
    # each data is a dictionary.
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        logger.info("image is {}".format(item["image2"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 8


def test_lfw_zip():
    """
    Feature: LFW
    Description: Test basic usage of LFW
    Expectation: The dataset is as expected
    """
    logger.info("Test Case zip")
    # define parameters.
    repeat_count = 2

    # apply dataset operations.
    data1 = ds.LFWDataset(DATA_DIR, num_samples=3)
    data2 = ds.LFWDataset(DATA_DIR, num_samples=3)

    data1 = data1.repeat(repeat_count)
    # rename dataset2 for no conflict.
    data2 = data2.rename(input_columns=["image", "label"],
                         output_columns=["image1", "label1"])
    data3 = ds.zip((data1, data2))

    num_iter = 0
    # each data is a dictionary.
    for item in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label".
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 3


def test_lfw_exception():
    """
    Feature: LFW
    Description: Test error cases of LFW
    Expectation: Throw exception correctly
    """
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.LFWDataset(DATA_DIR, shuffle=False, decode=True, sampler=ds.SequentialSampler(1))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.LFWDataset(DATA_DIR, sampler=ds.SequentialSampler(1), decode=True, num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.LFWDataset(DATA_DIR, decode=True, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.LFWDataset(DATA_DIR, decode=True, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.LFWDataset(DATA_DIR, decode=True, num_shards=5, shard_id=-1)

    with pytest.raises(ValueError, match=error_msg_5):
        ds.LFWDataset(DATA_DIR, decode=True, num_shards=5, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.LFWDataset(DATA_DIR, decode=True, shuffle=False, num_parallel_workers=0)

    with pytest.raises(ValueError, match=error_msg_6):
        ds.LFWDataset(DATA_DIR, decode=True, shuffle=False, num_parallel_workers=256)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.LFWDataset(DATA_DIR, decode=True, num_shards=2, shard_id="0")

    error_msg_8 = "does not exist or is not a directory or permission denied!"
    with pytest.raises(ValueError, match=error_msg_8):
        all_data = ds.LFWDataset("../data/dataset/testLFW2", decode=True)
        for _ in all_data.create_dict_iterator(num_epochs=1):
            pass

    error_msg_9 = "Input task is not within the valid set of ['people', 'pairs']."
    with pytest.raises(ValueError, match=re.escape(error_msg_9)):
        all_data = ds.LFWDataset(DATA_DIR, task="all")
        for _ in all_data.create_dict_iterator(num_epochs=1):
            pass

    error_msg_10 = "Input usage is not within the valid set of ['10fold', 'train', 'test', 'all']."
    with pytest.raises(ValueError, match=re.escape(error_msg_10)):
        all_data = ds.LFWDataset(DATA_DIR, usage="many")
        for _ in all_data.create_dict_iterator(num_epochs=1):
            pass

    error_msg_11 = "Input image_set is not within the valid set of ['original', 'funneled', 'deepfunneled']."
    with pytest.raises(ValueError, match=re.escape(error_msg_11)):
        all_data = ds.LFWDataset(DATA_DIR, image_set="all")
        for _ in all_data.create_dict_iterator(num_epochs=1):
            pass

    error_msg_12 = "Argument decode with value 123 is not of type [<class 'bool'>], but got <class 'int'>."
    with pytest.raises(TypeError, match=re.escape(error_msg_12)):
        all_data = ds.LFWDataset(DATA_DIR, decode=123)
        for _ in all_data.create_dict_iterator(num_epochs=1):
            pass


if __name__ == '__main__':
    test_lfw_basic()
    test_lfw_task()
    test_lfw_usage()
    test_lfw_image_set()
    test_lfw_num_samples()
    test_sequential_sampler()
    test_random_and_sequentialchained_sampler()
    test_lfw_num_shards()
    test_lfw_shard_id()
    test_lfw_no_shuffle()
    test_lfw_decode()
    test_lfw_rename()
    test_lfw_zip()
    test_lfw_exception()
