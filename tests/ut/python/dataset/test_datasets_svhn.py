# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
Test SVHN dataset operations
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.io import loadmat

import mindspore.dataset as ds
from mindspore import log as logger

DATA_DIR = "../data/dataset/testSVHNData"
WRONG_DIR = "../data/dataset/testMnistData"


def load_mat(mode, path):
    """
    Feature: load_mat.
    Description: Load .mat file.
    Expectation: Get .mat of svhn dataset.
    """
    filename = mode + "_32x32.mat"
    mat_data = loadmat(os.path.realpath(os.path.join(path, filename)))
    data = np.transpose(mat_data['X'], [3, 0, 1, 2])
    label = mat_data['y'].astype(np.uint32).squeeze()
    np.place(label, label == 10, 0)
    return data, label


def load_svhn(path, usage):
    """
    Feature: load_svhn.
    Description: Load svhn.
    Expectation: Get data of svhn dataset.
    """
    assert usage in ["train", "test", "extra", "all"]

    usage_all = ["train", "test", "extra"]
    data = np.array([], dtype=np.uint8)
    label = np.array([], dtype=np.uint32)
    if usage == "all":
        for _usage in usage_all:
            current_data, current_label = load_mat(_usage, path)
            data = np.concatenate((data, current_data)) if data.size else current_data
            label = np.concatenate((label, current_label)) if label.size else current_label
    else:
        data, label = load_mat(usage, path)
    return data, label


def visualize_dataset(images, labels):
    """
    Feature: visualize_dataset.
    Description: Visualize svhn dataset.
    Expectation: Plot images.
    """
    num_samples = len(images)
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i])
        plt.title(labels[i])
    plt.show()


def test_svhn_content_check():
    """
    Feature: test_svhn_content_check.
    Description: Validate SVHNDataset image readings.
    Expectation: Get correct number of data and correct content.
    """
    logger.info("Test SVHNDataset Op with content check")
    train_data = ds.SVHNDataset(DATA_DIR, "train", num_samples=2, shuffle=False)
    images, labels = load_svhn(DATA_DIR, "train")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label".
    for i, data in enumerate(train_data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(data["image"], images[i])
        np.testing.assert_array_equal(data["label"], labels[i])
        num_iter += 1
    assert num_iter == 2

    test_data = ds.SVHNDataset(DATA_DIR, "test", num_samples=4, shuffle=False)
    images, labels = load_svhn(DATA_DIR, "test")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label".
    for i, data in enumerate(test_data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(data["image"], images[i])
        np.testing.assert_array_equal(data["label"], labels[i])
        num_iter += 1
    assert num_iter == 4

    extra_data = ds.SVHNDataset(DATA_DIR, "extra", num_samples=6, shuffle=False)
    images, labels = load_svhn(DATA_DIR, "extra")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label".
    for i, data in enumerate(extra_data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(data["image"], images[i])
        np.testing.assert_array_equal(data["label"], labels[i])
        num_iter += 1
    assert num_iter == 6

    all_data = ds.SVHNDataset(DATA_DIR, "all", num_samples=12, shuffle=False)
    images, labels = load_svhn(DATA_DIR, "all")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label".
    for i, data in enumerate(all_data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(data["image"], images[i])
        np.testing.assert_array_equal(data["label"], labels[i])
        num_iter += 1
    assert num_iter == 12


# Run this test in separate process since this test updates shared memory config
@pytest.mark.forked
def test_svhn_basic():
    """
    Feature: test_svhn_basic.
    Description: Test basic usage of SVHNDataset.
    Expectation: Get correct number of data.
    """
    logger.info("Test SVHNDataset Op")

    # case 1: test loading whole dataset.
    default_data = ds.SVHNDataset(DATA_DIR)
    num_iter = 0
    for _ in default_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 12

    all_data = ds.SVHNDataset(DATA_DIR, "all")
    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 12

    # case 2: test num_samples.
    train_data = ds.SVHNDataset(DATA_DIR, "train", num_samples=2)
    num_iter = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 2

    # case 3: test repeat.
    train_data = ds.SVHNDataset(DATA_DIR, "train", num_samples=2)
    train_data = train_data.repeat(5)
    num_iter = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 10

    # case 4: test batch with drop_remainder=False.
    train_data = ds.SVHNDataset(DATA_DIR, "train", num_samples=2)
    assert train_data.get_dataset_size() == 2
    assert train_data.get_batch_size() == 1
    train_data = train_data.batch(batch_size=2)  # drop_remainder is default to be False.
    assert train_data.get_batch_size() == 2
    assert train_data.get_dataset_size() == 1

    num_iter = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 1

    # case 5: test batch with drop_remainder=True.
    train_data = ds.SVHNDataset(DATA_DIR, "train", num_samples=2)
    assert train_data.get_dataset_size() == 2
    assert train_data.get_batch_size() == 1
    train_data = train_data.batch(batch_size=2, drop_remainder=True)  # the rest of incomplete batch will be dropped.
    assert train_data.get_dataset_size() == 1
    assert train_data.get_batch_size() == 2
    num_iter = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 1

    # case 6: test num_parallel_workers>1
    shared_mem_flag = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)
    all_data = ds.SVHNDataset(DATA_DIR, "all", num_parallel_workers=2)
    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 12
    ds.config.set_enable_shared_mem(shared_mem_flag)

    # case 7: test map method
    input_columns = ["image"]
    image1, image2 = [], []
    dataset = ds.SVHNDataset(DATA_DIR, "all")
    for data in dataset.create_dict_iterator(output_numpy=True):
        image1.extend(data['image'])
    operations = [(lambda x: x + x)]
    dataset = dataset.map(input_columns=input_columns, operations=operations)
    for data in dataset.create_dict_iterator(output_numpy=True):
        image2.extend(data['image'])
    assert len(image1) == len(image2)

    # case 8: test batch
    dataset = ds.SVHNDataset(DATA_DIR, "all")
    dataset = dataset.batch(batch_size=3)

    num_iter = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        num_iter += 1
    assert num_iter == 4


def test_svhn_sequential_sampler():
    """
    Feature: test_svhn_sequential_sampler.
    Description: Test usage of SVHNDataset with SequentialSampler.
    Expectation: Get correct number of data.
    """
    logger.info("Test SVHNDataset Op with SequentialSampler")
    num_samples = 2
    sampler = ds.SequentialSampler(num_samples=num_samples)
    train_data_1 = ds.SVHNDataset(DATA_DIR, "train", sampler=sampler)
    train_data_2 = ds.SVHNDataset(DATA_DIR, "train", shuffle=False, num_samples=num_samples)
    label_list_1, label_list_2 = [], []
    num_iter = 0
    for item1, item2 in zip(train_data_1.create_dict_iterator(num_epochs=1),
                            train_data_2.create_dict_iterator(num_epochs=1)):
        label_list_1.append(item1["label"].asnumpy())
        label_list_2.append(item2["label"].asnumpy())
        num_iter += 1
    np.testing.assert_array_equal(label_list_1, label_list_2)
    assert num_iter == num_samples


def test_svhn_exception():
    """
    Feature: test_svhn_exception.
    Description: Test error cases for SVHNDataset.
    Expectation: Raise exception.
    """
    logger.info("Test error cases for SVHNDataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.SVHNDataset(DATA_DIR, "train", shuffle=False, sampler=ds.SequentialSampler(1))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.SVHNDataset(DATA_DIR, "train", sampler=ds.SequentialSampler(1), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.SVHNDataset(DATA_DIR, "train", num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.SVHNDataset(DATA_DIR, "train", shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.SVHNDataset(DATA_DIR, "train", num_shards=5, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.SVHNDataset(DATA_DIR, "train", num_shards=5, shard_id=5)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.SVHNDataset(DATA_DIR, "train", num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.SVHNDataset(DATA_DIR, "train", shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.SVHNDataset(DATA_DIR, "train", shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.SVHNDataset(DATA_DIR, "train", shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.SVHNDataset(DATA_DIR, "train", num_shards=2, shard_id="0")

    error_msg_8 = "does not exist or permission denied!"
    with pytest.raises(ValueError, match=error_msg_8):
        train_data = ds.SVHNDataset(WRONG_DIR, "train")
        for _ in train_data.__iter__():
            pass


def test_svhn_visualize(plot=False):
    """
    Feature: test_svhn_visualize.
    Description: Visualize SVHNDataset results.
    Expectation: Get correct number of data and plot them.
    """
    logger.info("Test SVHNDataset visualization")

    train_data = ds.SVHNDataset(DATA_DIR, "train", num_samples=2, shuffle=False)
    num_iter = 0
    image_list, label_list = [], []
    for item in train_data.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item["image"]
        label = item["label"]
        image_list.append(image)
        label_list.append("label {}".format(label))
        assert isinstance(image, np.ndarray)
        assert image.shape == (32, 32, 3)
        assert image.dtype == np.uint8
        assert label.dtype == np.uint32
        num_iter += 1
    assert num_iter == 2
    if plot:
        visualize_dataset(image_list, label_list)


def test_svhn_usage():
    """
    Feature: test_svhn_usage.
    Description: Validate SVHNDataset image readings.
    Expectation: Get correct number of data.
    """
    logger.info("Test SVHNDataset usage flag")

    def test_config(usage, path=None):
        path = DATA_DIR if path is None else path
        try:
            data = ds.SVHNDataset(path, usage=usage, shuffle=False)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    assert test_config("train") == 2
    assert test_config("test") == 4
    assert test_config("extra") == 6
    assert test_config("all") == 12

    assert "usage is not within the valid set of ['train', 'test', 'extra', 'all']" in test_config("invalid")
    assert "Argument usage with value ['list'] is not of type [<class 'str'>]" in test_config(["list"])

    data_path = None
    # the following tests on the entire datasets.
    if data_path is not None:
        assert test_config("train", data_path) == 2
        assert test_config("test", data_path) == 4
        assert test_config("extra", data_path) == 6
        assert test_config("all", data_path) == 12
        assert ds.SVHNDataset(data_path, usage="train").get_dataset_size() == 2
        assert ds.SVHNDataset(data_path, usage="test").get_dataset_size() == 4
        assert ds.SVHNDataset(data_path, usage="extra").get_dataset_size() == 6
        assert ds.SVHNDataset(data_path, usage="all").get_dataset_size() == 12


if __name__ == '__main__':
    test_svhn_content_check()
    test_svhn_basic()
    test_svhn_sequential_sampler()
    test_svhn_exception()
    test_svhn_visualize(plot=True)
    test_svhn_usage()
