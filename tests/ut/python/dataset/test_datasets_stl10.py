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
Test STL10 dataset operations
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger

DATA_DIR = "../data/dataset/testSTL10Data"
WRONG_DIR = "../data/dataset/testMnistData"


def loadfile(path_to_data, path_to_labels=None):
    """
    Feature: loadfile.
    Description: Parse stl10 file.
    Expectation: Get image and label of stl10 dataset.
    """
    labels = None
    if path_to_labels:
        with open(os.path.realpath(path_to_labels), 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 1, 3, 2))
    return images, labels


def load_stl10(path, usage):
    """
    Feature: load_stl10.
    Description: Load stl10.
    Expectation: Get data of stl10 dataset.
    """
    assert usage in ["train", "test", "unlabeled", "train+unlabeled", "all"]

    if usage == "train":
        image_path = os.path.join(path, "train_X.bin")
        label_path = os.path.join(path, "train_y.bin")
        images, labels = loadfile(image_path, label_path)

    elif usage == "train+unlabeled":
        image_path = os.path.join(path, "train_X.bin")
        label_path = os.path.join(path, "train_y.bin")
        images, labels = loadfile(image_path, label_path)

        image_path = os.path.join(path, "unlabeled_X.bin")
        unlabeled_image, _ = loadfile(image_path)

        images = np.concatenate((images, unlabeled_image))
        labels = np.concatenate((labels, np.asarray([-1] * unlabeled_image.shape[0])))

    elif usage == "unlabeled":
        image_path = os.path.join(path, "unlabeled_X.bin")

        images, _ = loadfile(image_path)
        labels = np.asarray([-1] * images.shape[0])

    elif usage == "test":
        image_path = os.path.join(path, "test_X.bin")
        label_path = os.path.join(path, "test_y.bin")

        images, labels = loadfile(image_path, label_path)

    elif usage == "all":
        image_path = os.path.join(path, "test_X.bin")
        label_path = os.path.join(path, "test_y.bin")
        images, labels = loadfile(image_path, label_path)

        image_path = os.path.join(path, "train_X.bin")
        label_path = os.path.join(path, "train_y.bin")

        train_image, train_label = loadfile(image_path, label_path)

        images = np.concatenate((images, train_image))
        labels = np.concatenate((labels, train_label))

        image_path = os.path.join(path, "unlabeled_X.bin")
        unlabeled_image, _ = loadfile(image_path)

        images = np.concatenate((images, unlabeled_image))
        labels = np.concatenate((labels, np.asarray([-1] * unlabeled_image.shape[0])))

    return images, labels


def visualize_dataset(images, labels):
    """
    Feature: visualize_dataset.
    Description: Visualize stl10 dataset.
    Expectation: Plot images.
    """
    num_samples = len(images)
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.title(labels[i])
    plt.show()


def test_stl10_content_check():
    """
    Feature: test_stl10_content_check.
    Description: Validate STL10ataset image readings.
    Expectation: Get correct number of data and correct content.
    """
    logger.info("Test STL10Dataset Op with content check")
    # 1. train data.
    data1 = ds.STL10Dataset(DATA_DIR, usage="train", num_samples=1, shuffle=False)
    images, labels = load_stl10(DATA_DIR, "train")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label".
    for i, d in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(d["image"], np.transpose(images[i], (1, 2, 0)))
        np.testing.assert_array_equal(d["label"], labels[i])
        num_iter += 1
    assert num_iter == 1

    # 2. test data.
    data1 = ds.STL10Dataset(DATA_DIR, usage="test", num_samples=1, shuffle=False)
    images, labels = load_stl10(DATA_DIR, "test")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label".
    for i, d in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(d["image"], np.transpose(images[i], (1, 2, 0)))
        np.testing.assert_array_equal(d["label"], labels[i])
        num_iter += 1
    assert num_iter == 1

    # 3. unlabeled data.
    data1 = ds.STL10Dataset(DATA_DIR, usage="unlabeled", num_samples=1, shuffle=False)
    images, labels = load_stl10(DATA_DIR, "unlabeled")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label".
    for i, d in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(d["image"], np.transpose(images[i], (1, 2, 0)))
        np.testing.assert_array_equal(d["label"], labels[i])
        num_iter += 1
    assert num_iter == 1

    # 4. train+unlabeled data.
    data1 = ds.STL10Dataset(DATA_DIR, usage="train+unlabeled", num_samples=2, shuffle=False)
    images, labels = load_stl10(DATA_DIR, "train+unlabeled")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label".
    for i, d in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(d["image"], np.transpose(images[i], (1, 2, 0)))
        np.testing.assert_array_equal(d["label"], labels[i])
        num_iter += 1
    assert num_iter == 2

    # 4. all data.
    data1 = ds.STL10Dataset(DATA_DIR, usage="all", num_samples=3, shuffle=False)
    images, labels = load_stl10(DATA_DIR, "all")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label".
    for i, d in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(d["image"], np.transpose(images[i], (1, 2, 0)))
        np.testing.assert_array_equal(d["label"], labels[i])
        num_iter += 1
    assert num_iter == 3


def test_stl10_basic():
    """
    Feature: test_stl10_basic.
    Description: Test basic usage of STL10Dataset.
    Expectation: Get correct number of data.
    """
    logger.info("Test STL10Dataset Op")

    # case 1: test loading whole dataset.
    all_data = ds.STL10Dataset(DATA_DIR, "all")
    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 3

    # case 2: test num_samples.
    all_data = ds.STL10Dataset(DATA_DIR, "all", num_samples=1)
    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 1

    # case 3: test repeat.
    all_data = ds.STL10Dataset(DATA_DIR, "all", num_samples=2)
    all_data = all_data.repeat(5)
    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 10

    # case 4: test batch with drop_remainder=False.
    all_data = ds.STL10Dataset(DATA_DIR, "all", num_samples=2)
    assert all_data.get_dataset_size() == 2
    assert all_data.get_batch_size() == 1
    all_data = all_data.batch(batch_size=2)  # drop_remainder is default to be False.
    assert all_data.get_batch_size() == 2
    assert all_data.get_dataset_size() == 1

    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 1

    # case 5: test batch with drop_remainder=True.
    all_data = ds.STL10Dataset(DATA_DIR, "all", num_samples=2)
    assert all_data.get_dataset_size() == 2
    assert all_data.get_batch_size() == 1
    all_data = all_data.batch(batch_size=2, drop_remainder=True)  # the rest of incomplete batch will be dropped.
    assert all_data.get_dataset_size() == 1
    assert all_data.get_batch_size() == 2
    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 1


def test_stl10_sequential_sampler():
    """
    Feature: test_stl10_sequential_sampler.
    Description: Test usage of STL10Dataset with SequentialSampler.
    Expectation: Get correct number of data.
    """
    logger.info("Test STL10Dataset Op with SequentialSampler")
    num_samples = 2
    sampler = ds.SequentialSampler(num_samples=num_samples)
    all_data_1 = ds.STL10Dataset(DATA_DIR, "all", sampler=sampler)
    all_data_2 = ds.STL10Dataset(DATA_DIR, "all", shuffle=False, num_samples=num_samples)
    label_list_1, label_list_2 = [], []
    num_iter = 0
    for item1, item2 in zip(all_data_1.create_dict_iterator(num_epochs=1),
                            all_data_2.create_dict_iterator(num_epochs=1)):
        label_list_1.append(item1["label"].asnumpy())
        label_list_2.append(item2["label"].asnumpy())
        num_iter += 1
    np.testing.assert_array_equal(label_list_1, label_list_2)
    assert num_iter == num_samples


def test_stl10_exception():
    """
    Feature: test_stl10_exception.
    Description: Test error cases for STL10Dataset.
    Expectation: Raise exception.
    """
    logger.info("Test error cases for STL10Dataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.STL10Dataset(DATA_DIR, "all", shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.STL10Dataset(DATA_DIR, "all", sampler=ds.PKSampler(3), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.STL10Dataset(DATA_DIR, "all", num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.STL10Dataset(DATA_DIR, "all", shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.STL10Dataset(DATA_DIR, "all", num_shards=5, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.STL10Dataset(DATA_DIR, "all", num_shards=5, shard_id=5)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.STL10Dataset(DATA_DIR, "all", num_shards=2, shard_id=5)
    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.STL10Dataset(DATA_DIR, "all", shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.STL10Dataset(DATA_DIR, "all", shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.STL10Dataset(DATA_DIR, "all", shuffle=False, num_parallel_workers=-2)
    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.STL10Dataset(DATA_DIR, "all", num_shards=2, shard_id="0")

    def exception_func(item):
        raise Exception("Error occur!")

    error_msg_8 = "The corresponding data files"
    with pytest.raises(RuntimeError, match=error_msg_8):
        all_data = ds.STL10Dataset(DATA_DIR, "all")
        all_data = all_data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in all_data.__iter__():
            pass

    with pytest.raises(RuntimeError, match=error_msg_8):
        all_data = ds.STL10Dataset(DATA_DIR, "all")
        all_data = all_data.map(operations=vision.Decode(), input_columns=["image"], num_parallel_workers=1)
        for _ in all_data.__iter__():
            pass

    error_msg_9 = "does not exist or permission denied!"
    with pytest.raises(ValueError, match=error_msg_9):
        all_data = ds.STL10Dataset(WRONG_DIR, "all")
        for _ in all_data.__iter__():
            pass


def test_stl10_visualize(plot=False):
    """
    Feature: test_stl10_visualize.
    Description: Visualize STL10Dataset results.
    Expectation: Get correct number of data and plot them.
    """
    logger.info("Test STL10Dataset visualization")
    all_data = ds.STL10Dataset(DATA_DIR, "all", num_samples=2, shuffle=False)
    num_iter = 0
    image_list, label_list = [], []
    for item in all_data.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item["image"]
        label = item["label"]
        image_list.append(image)
        label_list.append("label {}".format(label))
        assert isinstance(image, np.ndarray)
        assert image.shape == (96, 96, 3)
        assert image.dtype == np.uint8
        assert label.dtype == np.int32
        num_iter += 1
    assert num_iter == 2
    if plot:
        visualize_dataset(image_list, label_list)


def test_stl10_usage():
    """
    Feature: test_stl10_usage.
    Description: Validate STL10Dataset image readings.
    Expectation: Get correct number of data.
    """
    logger.info("Test STL10Dataset usage flag")

    def test_config(usage, path=None):
        path = DATA_DIR if path is None else path
        try:
            data = ds.STL10Dataset(path, usage=usage, shuffle=False)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    assert test_config("train") == 1
    assert test_config("test") == 1
    assert test_config("unlabeled") == 1
    assert test_config("train+unlabeled") == 2
    assert test_config("all") == 3

    assert "Input usage is not within the valid set of ['train', 'test', 'unlabeled', 'train+unlabeled', 'all']."\
             in test_config("invalid")
    assert "Argument usage with value ['list'] is not of type [<class 'str'>]" in test_config(["list"])

    # change this directory to the folder that contains all STL10 files.
    all_files_path = None
    # the following tests on the entire datasets.
    if all_files_path is not None:
        assert test_config("train", all_files_path) == 1

        assert ds.STL10Dataset(all_files_path, usage="train").get_dataset_size() == 1


if __name__ == '__main__':
    test_stl10_content_check()
    test_stl10_basic()
    test_stl10_sequential_sampler()
    test_stl10_exception()
    test_stl10_visualize(plot=True)
    test_stl10_usage()
