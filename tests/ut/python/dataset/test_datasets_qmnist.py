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
Test QMnistDataset operations
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger

DATA_DIR = "../data/dataset/testQMnistData"


def load_qmnist(path, usage, compat=True):
    """
    load QMNIST data
    """
    image_path = []
    label_path = []
    image_ext = "images-idx3-ubyte"
    label_ext = "labels-idx2-int"
    train_prefix = "qmnist-train"
    test_prefix = "qmnist-test"
    nist_prefix = "xnist"
    assert usage in ["train", "test", "nist", "all"]
    if usage == "train":
        image_path.append(os.path.realpath(os.path.join(path, train_prefix + "-" + image_ext)))
        label_path.append(os.path.realpath(os.path.join(path, train_prefix + "-" + label_ext)))
    elif usage == "test":
        image_path.append(os.path.realpath(os.path.join(path, test_prefix + "-" + image_ext)))
        label_path.append(os.path.realpath(os.path.join(path, test_prefix + "-" + label_ext)))
    elif usage == "nist":
        image_path.append(os.path.realpath(os.path.join(path, nist_prefix + "-" + image_ext)))
        label_path.append(os.path.realpath(os.path.join(path, nist_prefix + "-" + label_ext)))
    elif usage == "all":
        image_path.append(os.path.realpath(os.path.join(path, train_prefix + "-" + image_ext)))
        label_path.append(os.path.realpath(os.path.join(path, train_prefix + "-" + label_ext)))
        image_path.append(os.path.realpath(os.path.join(path, test_prefix + "-" + image_ext)))
        label_path.append(os.path.realpath(os.path.join(path, test_prefix + "-" + label_ext)))
        image_path.append(os.path.realpath(os.path.join(path, nist_prefix + "-" + image_ext)))
        label_path.append(os.path.realpath(os.path.join(path, nist_prefix + "-" + label_ext)))

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
            label_file.read(12)
            label = np.fromfile(label_file, dtype='>u4')
            label = label.reshape(-1, 8)
            labels.append(label)

    images = np.concatenate(images, 0)
    labels = np.concatenate(labels, 0)
    if compat:
        return images, labels[:, 0]
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


def test_qmnist_content_check():
    """
    Feature: QMnistDataset
    Description: Test QMnistDataset image readings with content check
    Expectation: The dataset is processed as expected
    """
    logger.info("Test QMnistDataset Op with content check")
    for usage in ["train", "test", "nist", "all"]:
        data1 = ds.QMnistDataset(DATA_DIR, usage, True, num_samples=10, shuffle=False)
        images, labels = load_qmnist(DATA_DIR, usage, True)
        num_iter = 0
        # in this example, each dictionary has keys "image" and "label"
        image_list, label_list = [], []
        for i, data in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
            image_list.append(data["image"])
            label_list.append("label {}".format(data["label"]))
            np.testing.assert_array_equal(data["image"], images[i])
            np.testing.assert_array_equal(data["label"], labels[i])
            num_iter += 1
        assert num_iter == 10

    for usage in ["train", "test", "nist", "all"]:
        data1 = ds.QMnistDataset(DATA_DIR, usage, False, num_samples=10, shuffle=False)
        images, labels = load_qmnist(DATA_DIR, usage, False)
        num_iter = 0
        # in this example, each dictionary has keys "image" and "label"
        image_list, label_list = [], []
        for i, data in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
            image_list.append(data["image"])
            label_list.append("label {}".format(data["label"]))
            np.testing.assert_array_equal(data["image"], images[i])
            np.testing.assert_array_equal(data["label"], labels[i])
            num_iter += 1
        assert num_iter == 10


def test_qmnist_basic():
    """
    Feature: QMnistDataset
    Description: Test QMnistDataset basic usage
    Expectation: The dataset is processed as expected
    """
    logger.info("Test QMnistDataset Op")

    # case 1: test loading whole dataset
    data1 = ds.QMnistDataset(DATA_DIR, "train", True)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == 10

    # case 2: test num_samples
    data2 = ds.QMnistDataset(DATA_DIR, "train", True, num_samples=5)
    num_iter2 = 0
    for _ in data2.create_dict_iterator(num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 5

    # case 3: test repeat
    data3 = ds.QMnistDataset(DATA_DIR, "train", True)
    data3 = data3.repeat(5)
    num_iter3 = 0
    for _ in data3.create_dict_iterator(num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 50

    # case 4: test batch with drop_remainder=False
    data4 = ds.QMnistDataset(DATA_DIR, "train", True, num_samples=10)
    assert data4.get_dataset_size() == 10
    assert data4.get_batch_size() == 1
    data4 = data4.batch(batch_size=7)  # drop_remainder is default to be False
    assert data4.get_dataset_size() == 2
    assert data4.get_batch_size() == 7
    num_iter4 = 0
    for _ in data4.create_dict_iterator(num_epochs=1):
        num_iter4 += 1
    assert num_iter4 == 2

    # case 5: test batch with drop_remainder=True
    data5 = ds.QMnistDataset(DATA_DIR, "train", True, num_samples=10)
    assert data5.get_dataset_size() == 10
    assert data5.get_batch_size() == 1
    data5 = data5.batch(batch_size=3, drop_remainder=True)  # the rest of incomplete batch will be dropped
    assert data5.get_dataset_size() == 3
    assert data5.get_batch_size() == 3
    num_iter5 = 0
    for _ in data5.create_dict_iterator(num_epochs=1):
        num_iter5 += 1
    assert num_iter5 == 3

    # case 6: test get_col_names
    dataset = ds.QMnistDataset(DATA_DIR, "train", True, num_samples=10)
    assert dataset.get_col_names() == ["image", "label"]


def test_qmnist_pk_sampler():
    """
    Feature: QMnistDataset
    Description: Test QMnistDataset with PKSampler
    Expectation: The dataset is processed as expected
    """
    logger.info("Test QMnistDataset Op with PKSampler")
    golden = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sampler = ds.PKSampler(10)
    data = ds.QMnistDataset(DATA_DIR, "nist", True, sampler=sampler)
    num_iter = 0
    label_list = []
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        label_list.append(item["label"])
        num_iter += 1
    np.testing.assert_array_equal(golden, label_list)
    assert num_iter == 10


def test_qmnist_sequential_sampler():
    """
    Feature: QMnistDataset
    Description: Test QMnistDataset with SequentialSampler
    Expectation: The dataset is processed as expected
    """
    logger.info("Test QMnistDataset Op with SequentialSampler")
    num_samples = 10
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data1 = ds.QMnistDataset(DATA_DIR, "train", True, sampler=sampler)
    data2 = ds.QMnistDataset(DATA_DIR, "train", True, shuffle=False, num_samples=num_samples)
    label_list1, label_list2 = [], []
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1), data2.create_dict_iterator(num_epochs=1)):
        label_list1.append(item1["label"].asnumpy())
        label_list2.append(item2["label"].asnumpy())
        num_iter += 1
    np.testing.assert_array_equal(label_list1, label_list2)
    assert num_iter == num_samples


def test_qmnist_exception():
    """
    Feature: QMnistDataset
    Description: Test error cases for QMnistDataset
    Expectation: Correct error is thrown as expected
    """
    logger.info("Test error cases for MnistDataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.QMnistDataset(DATA_DIR, "train", True, shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.QMnistDataset(DATA_DIR, "nist", True, sampler=ds.PKSampler(1), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.QMnistDataset(DATA_DIR, "train", True, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.QMnistDataset(DATA_DIR, "train", True, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.QMnistDataset(DATA_DIR, "train", True, num_shards=5, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.QMnistDataset(DATA_DIR, "train", True, num_shards=5, shard_id=5)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.QMnistDataset(DATA_DIR, "train", True, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.QMnistDataset(DATA_DIR, "train", True, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.QMnistDataset(DATA_DIR, "train", True, shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.QMnistDataset(DATA_DIR, "train", True, shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.QMnistDataset(DATA_DIR, "train", True, num_shards=2, shard_id="0")

    def exception_func(item):
        raise Exception("Error occur!")

    error_msg_8 = "The corresponding data files"
    with pytest.raises(RuntimeError, match=error_msg_8):
        data = ds.QMnistDataset(DATA_DIR, "train", True)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    with pytest.raises(RuntimeError, match=error_msg_8):
        data = ds.QMnistDataset(DATA_DIR, "train", True)
        data = data.map(operations=vision.Decode(), input_columns=["image"], num_parallel_workers=1)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    with pytest.raises(RuntimeError, match=error_msg_8):
        data = ds.QMnistDataset(DATA_DIR, "train", True)
        data = data.map(operations=exception_func, input_columns=["label"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass


def test_qmnist_visualize(plot=False):
    """
    Feature: QMnistDataset
    Description: Test QMnistDataset visualized results
    Expectation: The dataset is processed as expected
    """
    logger.info("Test QMnistDataset visualization")

    data1 = ds.QMnistDataset(DATA_DIR, "train", True, num_samples=10, shuffle=False)
    num_iter = 0
    image_list, label_list = [], []
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
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


def test_qmnist_usage():
    """
    Feature: QMnistDataset
    Description: Test QMnistDataset image readings with usage flag
    Expectation: The dataset is processed as expected
    """
    logger.info("Test QMnistDataset usage flag")

    def test_config(usage, path=None):
        path = DATA_DIR if path is None else path
        try:
            data = ds.QMnistDataset(path, usage=usage, compat=True, shuffle=False)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    assert test_config("train") == 10
    assert test_config("test") == 10
    assert test_config("nist") == 10
    assert test_config("all") == 30
    assert "usage is not within the valid set of ['train', 'test', 'test10k', 'test50k', 'nist', 'all']" in\
           test_config("invalid")
    assert "Argument usage with value ['list'] is not of type [<class 'str'>]" in test_config(["list"])


if __name__ == '__main__':
    test_qmnist_content_check()
    test_qmnist_basic()
    test_qmnist_pk_sampler()
    test_qmnist_sequential_sampler()
    test_qmnist_exception()
    test_qmnist_visualize(plot=True)
    test_qmnist_usage()
