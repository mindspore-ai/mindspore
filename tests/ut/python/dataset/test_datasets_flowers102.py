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
Test Flowers102 dataset operations
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image
from scipy.io import loadmat

import mindspore.dataset as ds
from mindspore.dataset import vision
from mindspore import log as logger

DATA_DIR = "../data/dataset/testFlowers102Dataset"
WRONG_DIR = "../data/dataset/testMnistData"


def load_flowers102(path, usage):
    """
    load Flowers102 data
    """
    assert usage in ["train", "valid", "test", "all"]

    imagelabels = (loadmat(os.path.join(path, "imagelabels.mat"))["labels"][0] - 1).astype(np.uint32)
    split = loadmat(os.path.join(path, "setid.mat"))
    if usage == 'train':
        indices = split["trnid"][0].tolist()
    elif usage == 'test':
        indices = split["tstid"][0].tolist()
    elif usage == 'valid':
        indices = split["valid"][0].tolist()
    elif usage == 'all':
        indices = split["trnid"][0].tolist()
        indices += split["tstid"][0].tolist()
        indices += split["valid"][0].tolist()

    image_paths = [os.path.join(path, "jpg", "image_" + str(index).zfill(5) + ".jpg") for index in indices]
    segmentation_paths = [os.path.join(path, "segmim", "segmim_" + str(index).zfill(5) + ".jpg") for index in indices]
    images = [np.asarray(Image.open(path).convert("RGB")) for path in image_paths]
    segmentations = [np.asarray(Image.open(path).convert("RGB")) for path in segmentation_paths]
    labels = [imagelabels[index - 1] for index in indices]

    return images, segmentations, labels


def visualize_dataset(images, labels):
    """
    Helper function to visualize the dataset samples
    """
    num_samples = len(images)
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].squeeze())
        plt.title(labels[i])
    plt.show()


def test_flowers102_content_check():
    """
    Feature: Flowers102Dataset
    Description: Test Flowers102Dataset image readings with content check
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Flowers102Dataset Op with content check")
    all_data = ds.Flowers102Dataset(DATA_DIR, task="Segmentation", usage="all",
                                    num_samples=6, decode=True, shuffle=False)
    images, segmentations, labels = load_flowers102(DATA_DIR, "all")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label"
    for i, data in enumerate(all_data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(data["image"], images[i])
        np.testing.assert_array_equal(data["segmentation"], segmentations[i])
        np.testing.assert_array_equal(data["label"], labels[i])
        num_iter += 1
    assert num_iter == 6

    train_data = ds.Flowers102Dataset(DATA_DIR, task="Segmentation", usage="train",
                                      num_samples=2, decode=True, shuffle=False)
    images, segmentations, labels = load_flowers102(DATA_DIR, "train")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label"
    for i, data in enumerate(train_data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(data["image"], images[i])
        np.testing.assert_array_equal(data["segmentation"], segmentations[i])
        np.testing.assert_array_equal(data["label"], labels[i])
        num_iter += 1
    assert num_iter == 2

    test_data = ds.Flowers102Dataset(DATA_DIR, task="Segmentation", usage="test",
                                     num_samples=2, decode=True, shuffle=False)
    images, segmentations, labels = load_flowers102(DATA_DIR, "test")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label"
    for i, data in enumerate(test_data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(data["image"], images[i])
        np.testing.assert_array_equal(data["segmentation"], segmentations[i])
        np.testing.assert_array_equal(data["label"], labels[i])
        num_iter += 1
    assert num_iter == 2

    val_data = ds.Flowers102Dataset(DATA_DIR, task="Segmentation", usage="valid",
                                    num_samples=2, decode=True, shuffle=False)
    images, segmentations, labels = load_flowers102(DATA_DIR, "valid")
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label"
    for i, data in enumerate(val_data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(data["image"], images[i])
        np.testing.assert_array_equal(data["segmentation"], segmentations[i])
        np.testing.assert_array_equal(data["label"], labels[i])
        num_iter += 1
    assert num_iter == 2


def test_flowers102_basic():
    """
    Feature: Flowers102Dataset
    Description: Test basic read on Flowers102Dataset
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Flowers102Dataset Op")

    # case 1: test decode
    all_data = ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", decode=False, shuffle=False)
    all_data_1 = all_data.map(operations=[vision.Decode()], input_columns=["image"], num_parallel_workers=1)
    all_data_2 = ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", decode=True, shuffle=False)

    num_iter = 0
    for item1, item2 in zip(all_data_1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            all_data_2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1["label"], item2["label"])
        num_iter += 1
    assert num_iter == 6

    # case 2: test num_samples
    all_data = ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", decode=True, num_samples=4)
    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 4

    # case 3: test repeat
    all_data = ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", decode=True, num_samples=4)
    all_data = all_data.repeat(5)
    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 20

    # case 3: test get_dataset_size, resize and batch
    all_data = ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", decode=False, num_samples=4)
    all_data = all_data.map(operations=[vision.Decode(), vision.Resize((224, 224))], input_columns=["image"],
                            num_parallel_workers=1)

    assert all_data.get_dataset_size() == 4
    assert all_data.get_batch_size() == 1
    all_data = all_data.batch(batch_size=3)  # drop_remainder is default to be False
    assert all_data.get_batch_size() == 3
    assert all_data.get_dataset_size() == 2

    num_iter = 0
    for _ in all_data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 2

    # case 4: test get_class_indexing
    all_data = ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", decode=False, num_samples=4)
    class_indexing = all_data.get_class_indexing()
    assert class_indexing["pink primrose"] == 0
    assert class_indexing["blackberry lily"] == 101


def test_flowers102_sequential_sampler():
    """
    Feature: Flowers102Dataset
    Description: Test Flowers102Dataset with SequentialSampler
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Flowers102Dataset Op with SequentialSampler")
    num_samples = 4
    sampler = ds.SequentialSampler(num_samples=num_samples)
    all_data_1 = ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all",
                                      decode=True, sampler=sampler)
    all_data_2 = ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all",
                                      decode=True, shuffle=False, num_samples=num_samples)
    label_list_1, label_list_2 = [], []
    num_iter = 0
    for item1, item2 in zip(all_data_1.create_dict_iterator(num_epochs=1),
                            all_data_2.create_dict_iterator(num_epochs=1)):
        label_list_1.append(item1["label"].asnumpy())
        label_list_2.append(item2["label"].asnumpy())
        num_iter += 1
    np.testing.assert_array_equal(label_list_1, label_list_2)
    assert num_iter == num_samples


def test_flowers102_exception():
    """
    Feature: Flowers102Dataset
    Description: Test error cases on Flowers102Dataset
    Expectation: Correct error is thrown as expected
    """
    logger.info("Test error cases for Flowers102Dataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", shuffle=False,
                             decode=True, sampler=ds.SequentialSampler(1))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", sampler=ds.SequentialSampler(1),
                             decode=True, num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", decode=True, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", decode=True, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", decode=True, num_shards=5, shard_id=-1)

    with pytest.raises(ValueError, match=error_msg_5):
        ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", decode=True, num_shards=5, shard_id=5)

    with pytest.raises(ValueError, match=error_msg_5):
        ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", decode=True, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", decode=True,
                             shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", decode=True,
                             shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", decode=True,
                             shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", decode=True, num_shards=2, shard_id="0")


    error_msg_8 = "does not exist or is not a directory or permission denied!"
    with pytest.raises(ValueError, match=error_msg_8):
        all_data = ds.Flowers102Dataset(WRONG_DIR, task="Classification", usage="all", decode=True)
        for _ in all_data.create_dict_iterator(num_epochs=1):
            pass

    error_msg_9 = "is not of type"
    with pytest.raises(TypeError, match=error_msg_9):
        all_data = ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", decode=123)
        for _ in all_data.create_dict_iterator(num_epochs=1):
            pass


def test_flowers102_visualize(plot=False):
    """
    Feature: Flowers102Dataset
    Description: Test Flowers102Dataset visualization for results
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Flowers102Dataset visualization")

    all_data = ds.Flowers102Dataset(DATA_DIR, task="Classification", usage="all", num_samples=4,
                                    decode=True, shuffle=False)
    num_iter = 0
    image_list, label_list = [], []
    for item in all_data.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item["image"]
        label = item["label"]
        image_list.append(image)
        label_list.append("label {}".format(label))
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3
        assert image.shape[-1] == 3
        assert image.dtype == np.uint8
        assert label.dtype == np.uint32
        num_iter += 1
    assert num_iter == 4
    if plot:
        visualize_dataset(image_list, label_list)


def test_flowers102_usage():
    """
    Feature: Flowers102Dataset
    Description: Test Flowers102Dataset usage flag
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Flowers102Dataset usage flag")

    def test_config(usage):
        try:
            data = ds.Flowers102Dataset(DATA_DIR, task="Classification", usage=usage, decode=True, shuffle=False)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    assert test_config("all") == 6
    assert test_config("train") == 2
    assert test_config("test") == 2
    assert test_config("valid") == 2

    assert "usage is not within the valid set of ['train', 'valid', 'test', 'all']" in test_config("invalid")
    assert "Argument usage with value ['list'] is not of type [<class 'str'>]" in test_config(["list"])


def test_flowers102_task():
    """
    Feature: Flowers102Dataset
    Description: Test Flowers102Dataset task flag
    Expectation: The dataset is processed as expected
    """
    logger.info("Test Flowers102Dataset task flag")

    def test_config(task):
        try:
            data = ds.Flowers102Dataset(DATA_DIR, task=task, usage="all", decode=True, shuffle=False)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    assert test_config("Classification") == 6
    assert test_config("Segmentation") == 6

    assert "Input task is not within the valid set of ['Classification', 'Segmentation']" in test_config("invalid")
    assert "Argument task with value ['list'] is not of type [<class 'str'>]" in test_config(["list"])

if __name__ == '__main__':
    test_flowers102_content_check()
    test_flowers102_basic()
    test_flowers102_sequential_sampler()
    test_flowers102_exception()
    test_flowers102_visualize(plot=True)
    test_flowers102_usage()
    test_flowers102_task()
