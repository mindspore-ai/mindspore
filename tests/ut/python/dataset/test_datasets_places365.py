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
"""
Test Places365 dataset operations
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

import mindspore.dataset as ds
from mindspore import log as logger

DATA_DIR = "../data/dataset/testPlaces365Data"


def load_places365(path):
    """
    Feature: load_places365.
    Description: Load places365.
    Expectation: Get data of places365 dataset.
    """
    images_path = os.path.realpath(os.path.join(path, 'val_256'))
    labels_path = os.path.realpath(os.path.join(path, 'places365_val.txt'))
    images = []
    labels = []
    with open(labels_path, 'r') as f:
        for line in f.readlines():
            file_path, label = line.split()
            image = np.array(Image.open(images_path + file_path))
            label = int(label)
            images.append(image)
            labels.append(label)
    return images, labels


def visualize_dataset(images, labels):
    """
    Feature: visualize_dataset.
    Description: Visualize places365 dataset.
    Expectation: Plot images.
    """
    num_samples = len(images)
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].squeeze(), cmap=plt.cm.gray)
        plt.title(labels[i])
    plt.show()


def test_places365_content_check():
    """
    Feature: test_places365_content_check.
    Description: Validate Places365Dataset image readings.
    Expectation: Get correct number of data and correct content.
    """
    logger.info("Test Places365Dataset Op with content check")
    sampler = ds.SequentialSampler(num_samples=4)
    data1 = ds.Places365Dataset(dataset_dir=DATA_DIR, usage='val', small=True, decode=True, sampler=sampler)
    _, labels = load_places365(DATA_DIR)
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label"
    image_list, label_list = [], []
    for i, data in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_list.append(data["image"])
        label_list.append("label {}".format(data["label"]))
        # due to the precision problem, the following two doesn't total equal.
        # np.testing.assert_array_equal(data["image"], images[i])
        np.testing.assert_array_equal(data["label"], labels[i])
        num_iter += 1
    assert num_iter == 4


def test_places365_basic():
    """
    Feature: test_places365_basic.
    Description: Test basic usage of Places365Dataset.
    Expectation: Get correct number of data.
    """
    logger.info("Test places365Dataset Op")

    # case 1: test loading whole dataset
    data1 = ds.Places365Dataset(DATA_DIR, usage='val', small=True, decode=True)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == 4
    # case 2: test num_samples
    data2 = ds.Places365Dataset(DATA_DIR, usage='train-standard', small=True, decode=True, num_samples=4)
    num_iter2 = 0
    for _ in data2.create_dict_iterator(num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 4

    # case 3: test repeat
    data3 = ds.Places365Dataset(DATA_DIR, usage='val', small=True, decode=True, num_samples=4)
    data3 = data3.repeat(5)
    num_iter3 = 0
    for _ in data3.create_dict_iterator(num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 20

    # case 4: test batch with drop_remainder=False
    data4 = ds.Places365Dataset(DATA_DIR, usage='val', small=True, decode=True, num_samples=4)
    assert data4.get_dataset_size() == 4
    assert data4.get_batch_size() == 1
    data4 = data4.batch(batch_size=2)  # drop_remainder is default to be False
    assert data4.get_dataset_size() == 2
    assert data4.get_batch_size() == 2
    num_iter4 = 0
    for _ in data4.create_dict_iterator(num_epochs=1):
        num_iter4 += 1
    assert num_iter4 == 2

    # case 5: test batch with drop_remainder=True
    data5 = ds.Places365Dataset(DATA_DIR, usage='train-standard', small=True, decode=True, num_samples=4)
    assert data5.get_dataset_size() == 4
    assert data5.get_batch_size() == 1
    data5 = data5.batch(batch_size=3, drop_remainder=True)  # the rest of incomplete batch will be dropped
    assert data5.get_dataset_size() == 1
    assert data5.get_batch_size() == 3
    num_iter5 = 0
    for _ in data5.create_dict_iterator(num_epochs=1):
        num_iter5 += 1
    assert num_iter5 == 1


def test_places365_pk_sampler():
    """
    Feature: test_places365_pk_sampler.
    Description: Test usage of Places365Dataset with PKSampler.
    Expectation: Get correct number of data.
    """
    logger.info("Test Places365Dataset Op with PKSampler")

    sampler = ds.PKSampler(1)
    data = ds.Places365Dataset(DATA_DIR, usage='val', small=True, decode=True, sampler=sampler)
    num_iter = 0
    golden = [0, 1]
    label_list = []
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        label_list.append(item["label"])
        num_iter += 1
    np.testing.assert_array_equal(golden, label_list)
    assert num_iter == 2


def test_places365_sequential_sampler():
    """
    Feature: test_places365_sequential_sampler.
    Description: Test usage of Places365Dataset with SequentialSampler.
    Expectation: Get correct number of data.
    """
    logger.info("Test Places365Dataset Op with SequentialSampler")
    num_samples = 4
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data1 = ds.Places365Dataset(DATA_DIR, usage='val', small=True, decode=True, sampler=sampler)
    data2 = ds.Places365Dataset(DATA_DIR, usage='val', small=True, decode=True, shuffle=False, num_samples=num_samples)
    label_list1, label_list2 = [], []
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1), data2.create_dict_iterator(num_epochs=1)):
        label_list1.append(item1["label"].asnumpy())
        label_list2.append(item2["label"].asnumpy())
        num_iter += 1
    np.testing.assert_array_equal(label_list1, label_list2)
    assert num_iter == num_samples


def test_places365_exception():
    """
    Feature: test_places365_exception.
    Description: Test error cases for Places365Dataset.
    Expectation: Raise exception.
    """
    logger.info("Test error cases for Places365Dataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.Places365Dataset(DATA_DIR, usage='val', small=True, decode=True, shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.Places365Dataset(DATA_DIR, usage='val', small=True, decode=True,
                            sampler=ds.PKSampler(3), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.Places365Dataset(DATA_DIR, usage='val', small=True, decode=True, num_shards=4)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.Places365Dataset(DATA_DIR, usage='val', small=True, decode=True, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.Places365Dataset(DATA_DIR, usage='val', small=True, decode=True, num_shards=2, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.Places365Dataset(DATA_DIR, usage='val', small=True, decode=True, num_shards=2, shard_id=2)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.Places365Dataset(DATA_DIR, usage='val', small=True, decode=True, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.Places365Dataset(DATA_DIR, usage='val', small=True, decode=True, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.Places365Dataset(DATA_DIR, usage='val', small=True, decode=True, shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.Places365Dataset(DATA_DIR, usage='val', small=True, decode=True, shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.Places365Dataset(DATA_DIR, usage='val', small=True, decode=True, num_shards=2, shard_id="0")


def test_places365_visualize(plot=False):
    """
    Feature: test_places365_visualize.
    Description: Visualize Places365Dataset results.
    Expectation: Get correct number of data and plot them.
    """
    logger.info("Test Places365Dataset visualization")

    data1 = ds.Places365Dataset(DATA_DIR, usage='val', small=True, decode=True, num_samples=4, shuffle=False)
    num_iter = 0
    image_list, label_list = [], []
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item["image"]
        label = item["label"]
        image_list.append(image)
        label_list.append("label {}".format(label))
        assert isinstance(image, np.ndarray)
        assert image.shape == (256, 256, 3)
        assert image.dtype == np.uint8
        assert label.dtype == np.uint32
        num_iter += 1
    assert num_iter == 4
    if plot:
        visualize_dataset(image_list, label_list)


def test_places365_usage():
    """
    Feature: test_places365_usage.
    Description: Validate Places365Dataset image readings.
    Expectation: Get correct number of data.
    """
    logger.info("Test Places365Dataset usage flag")

    def test_config(usage, places365_path=None):
        if places365_path is None:
            places365_path = DATA_DIR
        try:
            data = ds.Places365Dataset(places365_path, usage=usage, small=True, decode=True, shuffle=False)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            print(str(e))
            return str(e)
        return num_rows

    assert test_config("val") == 4
    assert "usage is not within the valid set of ['train-standard', 'train-challenge', 'val']" in test_config("invalid")
    assert "Argument usage with value ['list'] is not of type [<class 'str'>]" in test_config(["list"])

    # change this directory to the folder that contains all places365 files
    train_standard_files_path = DATA_DIR
    # the following tests on the entire datasets
    if train_standard_files_path is not None:
        assert test_config("train-standard", train_standard_files_path) == 4
        assert test_config("val", train_standard_files_path) == 4
    # change this directory to the folder that contains all places365 files
    train_challenge_files_path = DATA_DIR
    # the following tests on the entire datasets
    if train_challenge_files_path is not None:
        assert test_config("train-challenge", train_challenge_files_path) == 4
        assert test_config("val", train_standard_files_path) == 4


if __name__ == '__main__':
    test_places365_content_check()
    test_places365_basic()
    test_places365_pk_sampler()
    test_places365_sequential_sampler()
    test_places365_exception()
    test_places365_visualize(plot=True)
    test_places365_usage()
