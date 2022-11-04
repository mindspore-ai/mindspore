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

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger

DATA_DIR = "../data/dataset/testSBUDataset"
WRONG_DIR = "../data/dataset/testMnistData"


def load_sbu(path):
    """
    load SBU data
    """
    images = []
    captions = []

    file1 = os.path.realpath(os.path.join(path, 'SBU_captioned_photo_dataset_urls.txt'))
    file2 = os.path.realpath(os.path.join(path, 'SBU_captioned_photo_dataset_captions.txt'))

    for line1, line2 in zip(open(file1), open(file2)):
        url = line1.rstrip()
        image = url[23:].replace("/", "_")
        filename = os.path.join(path, 'sbu_images', image)
        if os.path.exists(filename):
            caption = line2.rstrip()
            images.append(np.asarray(Image.open(filename).convert('RGB')).astype(np.uint8))
            captions.append(caption)
    return images, captions


def visualize_dataset(images, captions):
    """
    Helper function to visualize the dataset samples
    """
    num_samples = len(images)
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].squeeze())
        plt.title(captions[i])
    plt.show()


def test_sbu_content_check():
    """
    Feature: SBUDataset
    Description: Test SBUDataset image readings with content check
    Expectation: The dataset is processed as expected
    """
    logger.info("Test SBUDataset Op with content check")
    dataset = ds.SBUDataset(DATA_DIR, decode=True, num_samples=50, shuffle=False)
    images, captions = load_sbu(DATA_DIR)
    num_iter = 0
    # in this example, each dictionary has keys "image" and "caption"
    for i, data in enumerate(dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
        assert data["image"].shape == images[i].shape
        assert data["caption"] == captions[i]
        num_iter += 1
    assert num_iter == 5


def test_sbu_case():
    """
    Feature: SBUDataset
    Description: Test SBUDataset cases
    Expectation: The dataset is processed as expected
    """
    dataset = ds.SBUDataset(DATA_DIR, decode=True)

    dataset = dataset.map(operations=[vision.Resize((224, 224))], input_columns=["image"])
    repeat_num = 4
    dataset = dataset.repeat(repeat_num)
    batch_size = 2
    dataset = dataset.batch(batch_size, drop_remainder=True)

    num = 0
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        num += 1
    # 4 x 5 / 2
    assert num == 10

    dataset = ds.SBUDataset(DATA_DIR, decode=False)

    dataset = dataset.map(operations=[vision.Decode(), vision.Resize((224, 224))], input_columns=["image"])
    repeat_num = 4
    dataset = dataset.repeat(repeat_num)
    batch_size = 2
    dataset = dataset.batch(batch_size, drop_remainder=True)

    num = 0
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        num += 1
    # 4 x 5 / 2
    assert num == 10


def test_sbu_basic():
    """
    Feature: SBUDataset
    Description: Test SBUDataset basic usage
    Expectation: The dataset is processed as expected
    """
    logger.info("Test SBUDataset Op")

    # case 1: test loading whole dataset
    dataset = ds.SBUDataset(DATA_DIR, decode=True)
    num_iter = 0
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    assert num_iter == 5


    # case 2: test num_samples
    dataset = ds.SBUDataset(DATA_DIR, decode=True, num_samples=5)
    num_iter = 0
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    assert num_iter == 5

    # case 3: test repeat
    dataset = ds.SBUDataset(DATA_DIR, decode=True, num_samples=5)
    dataset = dataset.repeat(5)
    num_iter = 0
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    assert num_iter == 25

    # case 4: test batch
    dataset = ds.SBUDataset(DATA_DIR, decode=True, num_samples=5)
    assert dataset.get_dataset_size() == 5
    assert dataset.get_batch_size() == 1

    num_iter = 0
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    assert num_iter == 5

    # case 5: test get_class_indexing
    dataset = ds.SBUDataset(DATA_DIR, decode=True, num_samples=5)
    assert dataset.get_class_indexing() == {}

    # case 6: test get_col_names
    dataset = ds.SBUDataset(DATA_DIR, decode=True, num_samples=5)
    assert dataset.get_col_names() == ["image", "caption"]


def test_sbu_sequential_sampler():
    """
    Feature: SBUDataset
    Description: Test SBUDataset wtih SequentialSampler
    Expectation: The dataset is processed as expected
    """
    logger.info("Test SBUDataset Op with SequentialSampler")
    num_samples = 5
    sampler = ds.SequentialSampler(num_samples=num_samples)
    dataset_1 = ds.SBUDataset(DATA_DIR, decode=True, sampler=sampler)
    dataset_2 = ds.SBUDataset(DATA_DIR, decode=True, shuffle=False, num_samples=num_samples)

    num_iter = 0
    for item1, item2 in zip(dataset_1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset_2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1["caption"], item2["caption"])
        num_iter += 1
    assert num_iter == num_samples


def test_sbu_exception():
    """
    Feature: SBUDataset
    Description: Test error cases for SBUDataset
    Expectation: Correct error is thrown as expected
    """
    logger.info("Test error cases for SBUDataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.SBUDataset(DATA_DIR, decode=True, shuffle=False, sampler=ds.SequentialSampler())

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.SBUDataset(DATA_DIR, decode=True, sampler=ds.SequentialSampler(), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.SBUDataset(DATA_DIR, decode=True, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.SBUDataset(DATA_DIR, decode=True, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.SBUDataset(DATA_DIR, decode=True, num_shards=5, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.SBUDataset(DATA_DIR, decode=True, num_shards=5, shard_id=5)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.SBUDataset(DATA_DIR, decode=True, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.SBUDataset(DATA_DIR, decode=True, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.SBUDataset(DATA_DIR, decode=True, shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.SBUDataset(DATA_DIR, decode=True, shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.SBUDataset(DATA_DIR, decode=True, num_shards=2, shard_id="0")

    def exception_func(item):
        raise Exception("Error occur!")

    error_msg_8 = "The corresponding data file is"
    with pytest.raises(RuntimeError, match=error_msg_8):
        dataset = ds.SBUDataset(DATA_DIR, decode=True)
        dataset = dataset.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in dataset.__iter__():
            pass

    with pytest.raises(RuntimeError, match=error_msg_8):
        dataset = ds.SBUDataset(DATA_DIR, decode=True)
        dataset = dataset.map(operations=vision.Decode(), input_columns=["image"], num_parallel_workers=1)
        for _ in dataset.__iter__():
            pass

    error_msg_9 = "does not exist or permission denied"
    with pytest.raises(ValueError, match=error_msg_9):
        dataset = ds.SBUDataset(WRONG_DIR, decode=True)
        for _ in dataset.__iter__():
            pass

    error_msg_10 = "Argument decode with value"
    with pytest.raises(TypeError, match=error_msg_10):
        dataset = ds.SBUDataset(DATA_DIR, decode="not_bool")
        for _ in dataset.__iter__():
            pass


def test_sbu_visualize(plot=False):
    """
    Feature: SBUDataset
    Description: Test SBUDataset visualized results
    Expectation: The dataset is processed as expected
    """
    logger.info("Test SBUDataset visualization")

    dataset = ds.SBUDataset(DATA_DIR, decode=True, num_samples=10, shuffle=False)
    num_iter = 0
    image_list, caption_list = [], []
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item["image"]
        caption = item["caption"]
        image_list.append(image)
        caption_list.append("caption {}".format(caption))
        assert isinstance(image, np.ndarray)

        assert image.dtype == np.uint8
        assert caption.dtype.type == np.str_
        num_iter += 1
    assert num_iter == 5
    if plot:
        visualize_dataset(image_list, caption_list)


def test_sbu_decode():
    """
    Feature: SBUDataset
    Description: Test SBUDataset image readings with decode flag
    Expectation: The dataset is processed as expected
    """
    logger.info("Test SBUDataset decode flag")

    sampler = ds.SequentialSampler(num_samples=50)
    dataset = ds.SBUDataset(dataset_dir=DATA_DIR, decode=False, sampler=sampler)
    dataset_1 = dataset.map(operations=[vision.Decode()], input_columns=["image"])

    dataset_2 = ds.SBUDataset(dataset_dir=DATA_DIR, decode=True, sampler=sampler)

    num_iter = 0
    for item1, item2 in zip(dataset_1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset_2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1["caption"], item2["caption"])
        num_iter += 1

    assert num_iter == 5


if __name__ == '__main__':
    test_sbu_content_check()
    test_sbu_basic()
    test_sbu_case()
    test_sbu_sequential_sampler()
    test_sbu_exception()
    test_sbu_visualize(plot=True)
    test_sbu_decode()
