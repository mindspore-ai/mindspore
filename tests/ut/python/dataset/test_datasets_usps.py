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
Test USPS dataset operations
"""
import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore import log as logger

DATA_DIR = "../data/dataset/testUSPSDataset"
WRONG_DIR = "../data/dataset/testMnistData"


def load_usps(path, usage):
    """
    load USPS data
    """
    assert usage in ["train", "test"]
    if usage == "train":
        data_path = os.path.realpath(os.path.join(path, "usps"))
    elif usage == "test":
        data_path = os.path.realpath(os.path.join(path, "usps.t"))

    with open(data_path, 'r') as f:
        raw_data = [line.split() for line in f.readlines()]
        tmp_list = [[x.split(':')[-1] for x in data[1:]] for data in raw_data]
        images = np.asarray(tmp_list, dtype=np.float32).reshape((-1, 16, 16, 1))
        images = ((cast(np.ndarray, images) + 1) / 2 * 255).astype(dtype=np.uint8)
        labels = [int(d[0]) - 1 for d in raw_data]
    return images, labels


def visualize_dataset(images, labels):
    """
    Helper function to visualize the dataset samples
    """
    num_samples = len(images)
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].squeeze(), cmap=plt.cm.gray)
        plt.title(labels[i])
    plt.show()


def test_usps_content_check():
    """
    Feature: USPSDataset
    Description: Test USPSDataset image readings with content check
    Expectation: The dataset is processed as expected
    """
    logger.info("Test USPSDataset Op with content check")
    train_data = ds.USPSDataset(DATA_DIR, "train", num_samples=10, shuffle=False)
    images, labels = load_usps(DATA_DIR, "train")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label"
    for i, data in enumerate(train_data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        for m in range(16):
            for n in range(16):
                assert (data["image"][m, n, 0] != 0 or images[i][m, n, 0] != 255) and \
                        (data["image"][m, n, 0] != 255 or images[i][m, n, 0] != 0)
                assert (data["image"][m, n, 0] == images[i][m, n, 0]) or\
                        (data["image"][m, n, 0] == images[i][m, n, 0] + 1) or\
                        (data["image"][m, n, 0] + 1 == images[i][m, n, 0])
        np.testing.assert_array_equal(data["label"], labels[i])
        num_iter += 1
    assert num_iter == 3

    test_data = ds.USPSDataset(DATA_DIR, "test", num_samples=3, shuffle=False)
    images, labels = load_usps(DATA_DIR, "test")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label"
    for i, data in enumerate(test_data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        for m in range(16):
            for n in range(16):
                if (data["image"][m, n, 0] == 0 and images[i][m, n, 0] == 255) or\
                        (data["image"][m, n, 0] == 255 and images[i][m, n, 0] == 0):
                    assert False
                if (data["image"][m, n, 0] != images[i][m, n, 0]) and\
                        (data["image"][m, n, 0] != images[i][m, n, 0] + 1) and\
                        (data["image"][m, n, 0] + 1 != images[i][m, n, 0]):
                    assert False
        np.testing.assert_array_equal(data["label"], labels[i])
        num_iter += 1
    assert num_iter == 3


def test_usps_basic():
    """
    Feature: USPSDataset
    Description: Test USPSDataset basic usage
    Expectation: The dataset is processed as expected
    """
    logger.info("Test USPSDataset Op")

    # case 1: test loading whole dataset
    train_data = ds.USPSDataset(DATA_DIR, "train")
    num_iter = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 3

    test_data = ds.USPSDataset(DATA_DIR, "test")
    num_iter = 0
    for _ in test_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 3

    # case 2: test num_samples
    train_data = ds.USPSDataset(DATA_DIR, "train", num_samples=2)
    num_iter = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 2

    # case 3: test repeat
    train_data = ds.USPSDataset(DATA_DIR, "train", num_samples=2)
    train_data = train_data.repeat(5)
    num_iter = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 10

    # case 4: test batch with drop_remainder=False
    train_data = ds.USPSDataset(DATA_DIR, "train", num_samples=3)
    assert train_data.get_dataset_size() == 3
    assert train_data.get_batch_size() == 1
    train_data = train_data.batch(batch_size=2)  # drop_remainder is default to be False
    assert train_data.get_batch_size() == 2
    assert train_data.get_dataset_size() == 2

    num_iter = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 2

    # case 5: test batch with drop_remainder=True
    train_data = ds.USPSDataset(DATA_DIR, "train", num_samples=3)
    assert train_data.get_dataset_size() == 3
    assert train_data.get_batch_size() == 1
    train_data = train_data.batch(batch_size=2, drop_remainder=True)  # the rest of incomplete batch will be dropped
    assert train_data.get_dataset_size() == 1
    assert train_data.get_batch_size() == 2
    num_iter = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 1


def test_usps_exception():
    """
    Feature: USPSDataset
    Description: Test error cases for USPSDataset
    Expectation: Correct error is thrown as expected
    """
    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.USPSDataset(DATA_DIR, "train", num_shards=10)
        ds.USPSDataset(DATA_DIR, "test", num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.USPSDataset(DATA_DIR, "train", shard_id=0)
        ds.USPSDataset(DATA_DIR, "test", shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.USPSDataset(DATA_DIR, "train", num_shards=5, shard_id=-1)
        ds.USPSDataset(DATA_DIR, "test", num_shards=5, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.USPSDataset(DATA_DIR, "train", num_shards=5, shard_id=5)
        ds.USPSDataset(DATA_DIR, "test", num_shards=5, shard_id=5)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.USPSDataset(DATA_DIR, "train", num_shards=2, shard_id=5)
        ds.USPSDataset(DATA_DIR, "test", num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.USPSDataset(DATA_DIR, "train", shuffle=False, num_parallel_workers=0)
        ds.USPSDataset(DATA_DIR, "test", shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.USPSDataset(DATA_DIR, "train", shuffle=False, num_parallel_workers=256)
        ds.USPSDataset(DATA_DIR, "test", shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.USPSDataset(DATA_DIR, "train", shuffle=False, num_parallel_workers=-2)
        ds.USPSDataset(DATA_DIR, "test", shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.USPSDataset(DATA_DIR, "train", num_shards=2, shard_id="0")
        ds.USPSDataset(DATA_DIR, "test", num_shards=2, shard_id="0")

    error_msg_8 = "invalid input shape"
    with pytest.raises(RuntimeError, match=error_msg_8):
        train_data = ds.USPSDataset(DATA_DIR, "train")
        train_data = train_data.map(operations=vision.Decode(), input_columns=["image"], num_parallel_workers=1)
        for _ in train_data.__iter__():
            pass

        test_data = ds.USPSDataset(DATA_DIR, "test")
        test_data = test_data.map(operations=vision.Decode(), input_columns=["image"], num_parallel_workers=1)
        for _ in test_data.__iter__():
            pass

    error_msg_9 = "usps does not exist or is a directory"
    with pytest.raises(RuntimeError, match=error_msg_9):
        train_data = ds.USPSDataset(WRONG_DIR, "train")
        for _ in train_data.__iter__():
            pass
    error_msg_10 = "usps.t does not exist or is a directory"
    with pytest.raises(RuntimeError, match=error_msg_10):
        test_data = ds.USPSDataset(WRONG_DIR, "test")
        for _ in test_data.__iter__():
            pass


def test_usps_visualize(plot=False):
    """
    Feature: USPSDataset
    Description: Test USPSDataset visualized results
    Expectation: The dataset is processed as expected
    """
    logger.info("Test USPSDataset visualization")

    train_data = ds.USPSDataset(DATA_DIR, "train", num_samples=3, shuffle=False)
    num_iter = 0
    image_list, label_list = [], []
    for item in train_data.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item["image"]
        label = item["label"]
        image_list.append(image)
        label_list.append("label {}".format(label))
        assert isinstance(image, np.ndarray)
        assert image.shape == (16, 16, 1)
        assert image.dtype == np.uint8
        assert label.dtype == np.uint32
        num_iter += 1
    assert num_iter == 3
    if plot:
        visualize_dataset(image_list, label_list)

    test_data = ds.USPSDataset(DATA_DIR, "test", num_samples=3, shuffle=False)
    num_iter = 0
    image_list, label_list = [], []
    for item in test_data.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item["image"]
        label = item["label"]
        image_list.append(image)
        label_list.append("label {}".format(label))
        assert isinstance(image, np.ndarray)
        assert image.shape == (16, 16, 1)
        assert image.dtype == np.uint8
        assert label.dtype == np.uint32
        num_iter += 1
    assert num_iter == 3
    if plot:
        visualize_dataset(image_list, label_list)


def test_usps_usage():
    """
    Feature: USPSDataset
    Description: Test USPSDataset image readings with usage flag
    Expectation: The dataset is processed as expected
    """
    logger.info("Test USPSDataset usage flag")

    def test_config(usage, path=None):
        path = DATA_DIR if path is None else path
        try:
            data = ds.USPSDataset(path, usage=usage, shuffle=False)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    assert test_config("train") == 3
    assert test_config("test") == 3

    assert "usage is not within the valid set of ['train', 'test', 'all']" in test_config("invalid")
    assert "Argument usage with value ['list'] is not of type [<class 'str'>]" in test_config(["list"])

    # change this directory to the folder that contains all USPS files
    all_files_path = None
    # the following tests on the entire datasets
    if all_files_path is not None:
        assert test_config("train", all_files_path) == 3
        assert test_config("test", all_files_path) == 3
        assert ds.USPSDataset(all_files_path, usage="train").get_dataset_size() == 3
        assert ds.USPSDataset(all_files_path, usage="test").get_dataset_size() == 3


def test_usps_with_map():
    """
    Feature: USPSDataset
    Description: Test doing map operation on USPSDataset
    Expectation: The dataset is processed as expected
    """
    dataset = ds.USPSDataset(DATA_DIR)
    random_crop = vision.RandomCrop((10, 10))
    dataset = dataset.map(random_crop, input_columns=["image"])
    type_cast = transforms.TypeCast(np.float32)
    dataset = dataset.map(type_cast, input_columns=["label"])
    count = 0
    for _ in dataset.create_dict_iterator(num_epochs=1):
        count += 1
    assert count == 6


if __name__ == '__main__':
    test_usps_content_check()
    test_usps_basic()
    test_usps_exception()
    test_usps_visualize(plot=True)
    test_usps_usage()
    test_usps_with_map()
