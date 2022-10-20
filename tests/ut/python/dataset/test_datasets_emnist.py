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
Test EMnist dataset operations
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger

DATA_DIR = "../data/dataset/testEMnistDataset"


def load_emnist(path, usage, name):
    """
    load EMnist data
    """
    image_path = []
    label_path = []
    image_ext = "images-idx3-ubyte"
    label_ext = "labels-idx1-ubyte"
    train_prefix = "emnist-" + name + "-train-"
    test_prefix = "emnist-" + name + "-test-"
    assert usage in ["train", "test", "all"]
    if usage == "train":
        image_path.append(os.path.realpath(os.path.join(path, train_prefix + image_ext)))
        label_path.append(os.path.realpath(os.path.join(path, train_prefix + label_ext)))
    elif usage == "test":
        image_path.append(os.path.realpath(os.path.join(path, test_prefix + image_ext)))
        label_path.append(os.path.realpath(os.path.join(path, test_prefix + label_ext)))
    elif usage == "all":
        image_path.append(os.path.realpath(os.path.join(path, test_prefix + image_ext)))
        label_path.append(os.path.realpath(os.path.join(path, test_prefix + label_ext)))
        image_path.append(os.path.realpath(os.path.join(path, train_prefix + image_ext)))
        label_path.append(os.path.realpath(os.path.join(path, train_prefix + label_ext)))
    assert len(image_path) == len(label_path)
    images = []
    labels = []
    for i, _ in enumerate(image_path):
        with open(image_path[i], 'rb') as image_file:
            image_file.read(16)
            image = np.fromfile(image_file, dtype=np.uint8)
            image = image.reshape(-1, 28, 28, 1)
            images.append(image)
        with open(label_path[i], 'rb') as label_file:
            label_file.read(8)
            label = np.fromfile(label_file, dtype=np.uint8)
            labels.append(label)

    images = np.concatenate(images, 0)
    labels = np.concatenate(labels, 0)

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


def test_emnist_content_check():
    """
    Feature: EMnistDataset
    Description: Test EMnistDataset image readings with content check
    Expectation: The dataset is processed as expected
    """
    logger.info("Test EMnistDataset Op with content check")
    # train mnist
    train_data = ds.EMnistDataset(DATA_DIR, name="mnist", usage="train", num_samples=10, shuffle=False)
    images, labels = load_emnist(DATA_DIR, "train", "mnist")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label"
    image_list, label_list = [], []
    for i, data in enumerate(train_data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_list.append(data["image"])
        label_list.append("label {}".format(data["label"]))
        np.testing.assert_array_equal(data["image"], images[i])
        np.testing.assert_array_equal(data["label"], labels[i])
        num_iter += 1
    assert num_iter == 10

    # train byclass
    train_data = ds.EMnistDataset(DATA_DIR, name="byclass", usage="train", num_samples=10, shuffle=False)
    images, labels = load_emnist(DATA_DIR, "train", "byclass")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label"
    image_list, label_list = [], []
    for i, data in enumerate(train_data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_list.append(data["image"])
        label_list.append("label {}".format(data["label"]))
        np.testing.assert_array_equal(data["image"], images[i])
        np.testing.assert_array_equal(data["label"], labels[i])
        num_iter += 1
    assert num_iter == 10

    # test
    test_data = ds.EMnistDataset(DATA_DIR, name="mnist", usage="test", num_samples=10, shuffle=False)
    images, labels = load_emnist(DATA_DIR, "test", "mnist")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label"
    image_list, label_list = [], []
    for i, data in enumerate(test_data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_list.append(data["image"])
        label_list.append("label {}".format(data["label"]))
        np.testing.assert_array_equal(data["image"], images[i])
        np.testing.assert_array_equal(data["label"], labels[i])
        num_iter += 1
    assert num_iter == 10


def test_emnist_basic():
    """
    Feature: EMnistDataset
    Description: Test basic read on EMnistDataset
    Expectation: The dataset is processed as expected
    """
    logger.info("Test EMnistDataset Op")

    # case 1: test loading whole dataset
    train_data = ds.EMnistDataset(DATA_DIR, "mnist", "train")
    num_iter1 = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == 10

    test_data = ds.EMnistDataset(DATA_DIR, "mnist", "test")
    num_iter = 0
    for _ in test_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 10

    # case 2: test num_samples
    train_data = ds.EMnistDataset(DATA_DIR, "byclass", "train", num_samples=5)
    num_iter2 = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 5

    test_data = ds.EMnistDataset(DATA_DIR, "mnist", "test", num_samples=5)
    num_iter2 = 0
    for _ in test_data.create_dict_iterator(num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 5

    # case 3: test repeat
    train_data = ds.EMnistDataset(DATA_DIR, "byclass", "train", num_samples=2)
    train_data = train_data.repeat(5)
    num_iter3 = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 10

    test_data = ds.EMnistDataset(DATA_DIR, "mnist", "test", num_samples=2)
    test_data = test_data.repeat(5)
    num_iter3 = 0
    for _ in test_data.create_dict_iterator(num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 10

    # case 4: test batch with drop_remainder=False
    train_data = ds.EMnistDataset(DATA_DIR, "byclass", "train", num_samples=10)
    assert train_data.get_dataset_size() == 10
    assert train_data.get_batch_size() == 1

    train_data = train_data.batch(batch_size=7)  # drop_remainder is default to be False
    assert train_data.get_dataset_size() == 2
    assert train_data.get_batch_size() == 7
    num_iter4 = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter4 += 1
    assert num_iter4 == 2

    test_data = ds.EMnistDataset(DATA_DIR, "mnist", "test", num_samples=10)
    assert test_data.get_dataset_size() == 10
    assert test_data.get_batch_size() == 1

    test_data = test_data.batch(
        batch_size=7)  # drop_remainder is default to be False
    assert test_data.get_dataset_size() == 2
    assert test_data.get_batch_size() == 7
    num_iter4 = 0
    for _ in test_data.create_dict_iterator(num_epochs=1):
        num_iter4 += 1
    assert num_iter4 == 2

    # case 5: test batch with drop_remainder=True
    train_data = ds.EMnistDataset(DATA_DIR, "byclass", "train", num_samples=10)
    assert train_data.get_dataset_size() == 10
    assert train_data.get_batch_size() == 1
    train_data = train_data.batch(batch_size=7, drop_remainder=True)  # the rest of incomplete batch will be dropped
    assert train_data.get_dataset_size() == 1
    assert train_data.get_batch_size() == 7
    num_iter5 = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter5 += 1
    assert num_iter5 == 1

    test_data = ds.EMnistDataset(DATA_DIR, "mnist", "test", num_samples=10)
    assert test_data.get_dataset_size() == 10
    assert test_data.get_batch_size() == 1
    test_data = test_data.batch(batch_size=7, drop_remainder=True)  # the rest of incomplete batch will be dropped
    assert test_data.get_dataset_size() == 1
    assert test_data.get_batch_size() == 7
    num_iter5 = 0
    for _ in test_data.create_dict_iterator(num_epochs=1):
        num_iter5 += 1
    assert num_iter5 == 1

    # case 6: test get_col_names
    dataset = ds.EMnistDataset(DATA_DIR, "mnist", "test", num_samples=10)
    assert dataset.get_col_names() == ["image", "label"]


def test_emnist_pk_sampler():
    """
    Feature: EMnistDataset
    Description: Test EMnistDataset with PKSampler
    Expectation: The dataset is processed as expected
    """
    logger.info("Test EMnistDataset Op with PKSampler")
    golden = [0, 0, 0, 1, 1, 1]

    sampler = ds.PKSampler(3)
    train_data = ds.EMnistDataset(DATA_DIR, "mnist", "train", sampler=sampler)
    num_iter = 0
    label_list = []
    for item in train_data.create_dict_iterator(num_epochs=1, output_numpy=True):
        label_list.append(item["label"])
        num_iter += 1
    np.testing.assert_array_equal(golden, label_list)
    assert num_iter == 6

    sampler = ds.PKSampler(3)
    test_data = ds.EMnistDataset(DATA_DIR, "mnist", "train", sampler=sampler)
    num_iter = 0
    label_list = []
    for item in test_data.create_dict_iterator(num_epochs=1, output_numpy=True):
        label_list.append(item["label"])
        num_iter += 1
    np.testing.assert_array_equal(golden, label_list)
    assert num_iter == 6


def test_emnist_sequential_sampler():
    """
    Feature: EMnistDataset
    Description: Test EMnistDataset with SequentialSampler
    Expectation: The dataset is processed as expected
    """
    logger.info("Test EMnistDataset Op with SequentialSampler")
    num_samples = 10
    sampler = ds.SequentialSampler(num_samples=num_samples)
    train_data1 = ds.EMnistDataset(DATA_DIR, "mnist", "train", sampler=sampler)
    train_data2 = ds.EMnistDataset(DATA_DIR, "mnist", "train", shuffle=False, num_samples=num_samples)
    label_list1, label_list2 = [], []
    num_iter = 0
    for item1, item2 in zip(train_data1.create_dict_iterator(num_epochs=1),
                            train_data2.create_dict_iterator(num_epochs=1)):
        label_list1.append(item1["label"].asnumpy())
        label_list2.append(item2["label"].asnumpy())
        num_iter += 1
    np.testing.assert_array_equal(label_list1, label_list2)
    assert num_iter == num_samples

    num_samples = 10
    sampler = ds.SequentialSampler(num_samples=num_samples)
    test_data1 = ds.EMnistDataset(DATA_DIR, "mnist", "test", sampler=sampler)
    test_data2 = ds.EMnistDataset(DATA_DIR, "mnist", "test", shuffle=False, num_samples=num_samples)
    label_list1, label_list2 = [], []
    num_iter = 0
    for item1, item2 in zip(test_data1.create_dict_iterator(num_epochs=1),
                            test_data2.create_dict_iterator(num_epochs=1)):
        label_list1.append(item1["label"].asnumpy())
        label_list2.append(item2["label"].asnumpy())
        num_iter += 1
    np.testing.assert_array_equal(label_list1, label_list2)
    assert num_iter == num_samples


def test_emnist_exception():
    """
    Feature: EMnistDataset
    Description: Test error cases for EMnistDataset
    Expectation: Throw correct error as expected
    """
    logger.info("Test error cases for EMnistDataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.EMnistDataset(DATA_DIR, "byclass", "train", shuffle=False, sampler=ds.PKSampler(3))
        ds.EMnistDataset(DATA_DIR, "mnist", "test", shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.EMnistDataset(DATA_DIR, "mnist", "train", sampler=ds.PKSampler(3), num_shards=2, shard_id=0)
        ds.EMnistDataset(DATA_DIR, "mnist", "test", sampler=ds.PKSampler(3), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.EMnistDataset(DATA_DIR, "byclass", "train", num_shards=10)
        ds.EMnistDataset(DATA_DIR, "mnist", "test", num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.EMnistDataset(DATA_DIR, "mnist", "train", shard_id=0)
        ds.EMnistDataset(DATA_DIR, "mnist", "test", shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.EMnistDataset(DATA_DIR, "byclass", "train", num_shards=5, shard_id=-1)
        ds.EMnistDataset(DATA_DIR, "mnist", "test", num_shards=5, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.EMnistDataset(DATA_DIR, "mnist", "train", num_shards=5, shard_id=5)
        ds.EMnistDataset(DATA_DIR, "mnist", "test", num_shards=5, shard_id=5)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.EMnistDataset(DATA_DIR, "byclass", "train", num_shards=2, shard_id=5)
        ds.EMnistDataset(DATA_DIR, "mnist", "test", num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.EMnistDataset(DATA_DIR, "mnist", "train", shuffle=False, num_parallel_workers=0)
        ds.EMnistDataset(DATA_DIR, "mnist", "test", shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.EMnistDataset(DATA_DIR, "byclass", "train", shuffle=False, num_parallel_workers=256)
        ds.EMnistDataset(DATA_DIR, "mnist", "test", shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.EMnistDataset(DATA_DIR, "mnist", "train", shuffle=False, num_parallel_workers=-2)
        ds.EMnistDataset(DATA_DIR, "mnist", "test", shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.EMnistDataset(DATA_DIR, "mnist", "train", num_shards=2, shard_id="0")
        ds.EMnistDataset(DATA_DIR, "mnist", "test", num_shards=2, shard_id="0")

    def exception_func(item):
        raise Exception("Error occur!")

    error_msg_8 = "The corresponding data file is"
    with pytest.raises(RuntimeError, match=error_msg_8):
        data = ds.EMnistDataset(DATA_DIR, "mnist", "train")
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    with pytest.raises(RuntimeError, match=error_msg_8):
        data = ds.EMnistDataset(DATA_DIR, "mnist", "train")
        data = data.map(operations=vision.Decode(), input_columns=["image"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    with pytest.raises(RuntimeError, match=error_msg_8):
        data = ds.EMnistDataset(DATA_DIR, "mnist", "train")
        data = data.map(operations=exception_func, input_columns=["label"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass


def test_emnist_visualize(plot=False):
    """
    Feature: EMnistDataset
    Description: Test EMnistDataset visualization for result
    Expectation: The dataset is processed as expected
    """
    logger.info("Test EMnistDataset visualization")

    train_data = ds.EMnistDataset(DATA_DIR, "mnist", "train", num_samples=10, shuffle=False)
    num_iter = 0
    image_list, label_list = [], []
    for item in train_data.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item["image"]
        label = item["label"]
        image_list.append(image)
        label_list.append("label {}".format(label))
        assert isinstance(image, np.ndarray)
        assert image.shape == (28, 28, 1)
        assert image.dtype == np.uint8
        assert label.dtype == np.uint32
        num_iter += 1
    assert num_iter == 10
    if plot:
        visualize_dataset(image_list, label_list)

    test_data = ds.EMnistDataset(DATA_DIR, "mnist", "test", num_samples=10, shuffle=False)
    num_iter = 0
    image_list, label_list = [], []
    for item in test_data.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item["image"]
        label = item["label"]
        image_list.append(image)
        label_list.append("label {}".format(label))
        assert isinstance(image, np.ndarray)
        assert image.shape == (28, 28, 1)
        assert image.dtype == np.uint8
        assert label.dtype == np.uint32
        num_iter += 1
    assert num_iter == 10
    if plot:
        visualize_dataset(image_list, label_list)


def test_emnist_usage():
    """
    Feature: EMnistDataset
    Description: Test EMnistDataset image readings with usage flag
    Expectation: The dataset is processed or error is thrown as expected
    """
    logger.info("Test EMnistDataset usage flag")

    def test_config(usage, emnist_path=None):
        emnist_path = DATA_DIR if emnist_path is None else emnist_path
        try:
            data = ds.EMnistDataset(emnist_path, "mnist", usage=usage, shuffle=False)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    assert test_config("train") == 10
    assert test_config("test") == 10
    assert test_config("all") == 20

    assert "usage is not within the valid set of ['train', 'test', 'all']" in test_config("invalid")
    assert "Argument usage with value ['list'] is not of type [<class 'str'>]" in test_config(["list"])

    # change this directory to the folder that contains all emnist files
    all_files_path = None

    # the following tests on the entire datasets
    if all_files_path is not None:
        assert test_config("train", all_files_path) == 10000
        assert test_config("test", all_files_path) == 60000
        assert test_config("all", all_files_path) == 70000
        assert ds.EMnistDataset(all_files_path, "mnist", usage="test").get_dataset_size() == 10000
        assert ds.EMnistDataset(all_files_path, "mnist", usage="test").get_dataset_size() == 60000
        assert ds.EMnistDataset(all_files_path, "mnist", usage="all").get_dataset_size() == 70000


def test_emnist_name():
    """
    Feature: EMnistDataset
    Description: Test EMnistDataset image readings with name flag
    Expectation: The dataset is processed or error is thrown as expected
    """
    def test_config(name, usage, emnist_path=None):
        emnist_path = DATA_DIR if emnist_path is None else emnist_path
        try:
            data = ds.EMnistDataset(emnist_path, name, usage=usage, shuffle=False)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    assert test_config("mnist", "train") == 10
    assert test_config("mnist", "test") == 10
    assert test_config("byclass", "train") == 10
    assert "name is not within the valid set of " + \
            "['byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist']" in test_config("invalid", "train")
    assert "Argument name with value ['list'] is not of type [<class 'str'>]" in test_config(["list"], "train")


if __name__ == '__main__':
    test_emnist_content_check()
    test_emnist_basic()
    test_emnist_pk_sampler()
    test_emnist_sequential_sampler()
    test_emnist_exception()
    test_emnist_visualize(plot=True)
    test_emnist_usage()
    test_emnist_name()
