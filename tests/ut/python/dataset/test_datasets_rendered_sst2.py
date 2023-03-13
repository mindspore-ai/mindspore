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
Test RenderedSST2 dataset operators
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger

IMAGE_DATA_DIR = "../data/dataset/testRenderedSST2Data"
WRONG_DIR = "../data/dataset/notExist"


@pytest.mark.skip(reason="random failure")
def test_rendered_sst2_basic():
    """
    Feature: RenderedSST2Dataset
    Description: Basic test of RenderedSST2Dataset
    Expectation: The data is processed successfully
    """
    logger.info("Test RenderedSST2Dataset Op")
    # case 1: test read all data
    all_data_1 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle=False)
    all_data_2 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle=False)

    num_iter = 0
    for item1, item2 in zip(all_data_1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            all_data_2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1["label"], item2["label"])
        num_iter += 1
    assert num_iter == 12

    # case 2: test decode
    all_data_1 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, decode=True, shuffle=False)
    all_data_2 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, decode=True, shuffle=False)

    num_iter = 0
    for item1, item2 in zip(all_data_1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            all_data_2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1["label"], item2["label"])
        num_iter += 1
    assert num_iter == 12

    # case 3: test num_samples
    all_data = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_samples=4)
    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 4

    # case 4: test repeat
    all_data = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_samples=4)
    all_data = all_data.repeat(2)
    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 8

    # case 5: test get_dataset_size, resize and batch
    all_data = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_samples=8)
    all_data = all_data.map(operations=[vision.Decode(), vision.Resize((256, 256))], input_columns=["image"],
                            num_parallel_workers=1)

    assert all_data.get_dataset_size() == 8
    assert all_data.get_batch_size() == 1
    # drop_remainder is default to be False
    all_data = all_data.batch(batch_size=3)
    assert all_data.get_batch_size() == 3
    assert all_data.get_dataset_size() == 3

    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 3


def test_rendered_sst2_decode():
    """
    Feature: RenderedSST2Dataset
    Description: Validate RenderedSST2Dataset with decode
    Expectation: The data is processed successfully
    """
    logger.info("Validate RenderedSST2Dataset with decode")
    # define parameters
    repeat_count = 1

    data1 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, decode=True)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 12


def test_rendered_sst2_sequential_sampler():
    """
    Feature: RenderedSST2Dataset
    Description: Test RenderedSST2Dataset with SequentialSampler
    Expectation: The data is processed successfully
    """
    logger.info("Test RenderedSST2Dataset Op with SequentialSampler")
    num_samples = 4
    sampler = ds.SequentialSampler(num_samples=num_samples)
    all_data_1 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, sampler=sampler)
    all_data_2 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle=False, num_samples=num_samples)
    label_list_1, label_list_2 = [], []
    num_iter = 0
    for item1, item2 in zip(all_data_1.create_dict_iterator(num_epochs=1),
                            all_data_2.create_dict_iterator(num_epochs=1)):
        label_list_1.append(item1["label"].asnumpy())
        label_list_2.append(item2["label"].asnumpy())
        num_iter += 1
    np.testing.assert_array_equal(label_list_1, label_list_2)
    assert num_iter == num_samples


def test_rendered_sst2_random_sampler():
    """
    Feature: RenderedSST2Dataset
    Description: Test RenderedSST2Dataset with RandomSampler
    Expectation: The data is processed successfully
    """
    logger.info("Test RenderedSST2Dataset Op with RandomSampler")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    sampler = ds.RandomSampler()
    data1 = ds.RenderedSST2Dataset(IMAGE_DATA_DIR, sampler=sampler)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 12


def test_rendered_sst2_exception():
    """
    Feature: RenderedSST2Dataset
    Description: Test error cases for RenderedSST2Dataset
    Expectation: Throw correct error and message
    """
    logger.info("Test error cases for RenderedSST2Dataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle=False, sampler=ds.SequentialSampler(1))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, sampler=ds.SequentialSampler(1), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_shards=5, shard_id=-1)

    with pytest.raises(ValueError, match=error_msg_5):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_shards=5, shard_id=5)

    with pytest.raises(ValueError, match=error_msg_5):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.RenderedSST2Dataset(IMAGE_DATA_DIR, num_shards=2, shard_id="0")

    error_msg_8 = "does not exist or is not a directory or permission denied!"
    with pytest.raises(ValueError, match=error_msg_8):
        all_data = ds.RenderedSST2Dataset(WRONG_DIR)
        for _ in all_data.create_dict_iterator(num_epochs=1):
            pass


if __name__ == '__main__':
    test_rendered_sst2_basic()
    test_rendered_sst2_decode()
    test_rendered_sst2_sequential_sampler()
    test_rendered_sst2_random_sampler()
    test_rendered_sst2_exception()
