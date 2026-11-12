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
Test FakeImage dataset operations
"""

import matplotlib.pyplot as plt
import numpy as np

import mindspore.dataset as ds
from mindspore import log as logger

num_images = 50
image_size = (28, 28, 3)
num_classes = 10
base_seed = 0


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


def test_fake_image_basic():
    """
    Feature: FakeImage
    Description: Test basic usage of FakeImage
    Expectation: The dataset is as expected
    """
    logger.info("Test FakeImageDataset Op")

    # case 1: test loading whole dataset
    train_data = ds.FakeImageDataset(num_images, image_size, num_classes, base_seed)
    num_iter1 = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == num_images

    # case 2: test num_samples
    train_data = ds.FakeImageDataset(num_images, image_size, num_classes, base_seed, num_samples=4)
    num_iter2 = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 4

    # case 3: test repeat
    train_data = ds.FakeImageDataset(num_images, image_size, num_classes, base_seed, num_samples=4)
    train_data = train_data.repeat(5)
    num_iter3 = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 20

    # case 4: test batch with drop_remainder=False, get_dataset_size, get_batch_size, get_col_names
    train_data = ds.FakeImageDataset(num_images, image_size, num_classes, base_seed, num_samples=4)
    assert train_data.get_dataset_size() == 4
    assert train_data.get_batch_size() == 1
    assert train_data.get_col_names() == ['image', 'label']
    train_data = train_data.batch(batch_size=3)  # drop_remainder is default to be False
    assert train_data.get_dataset_size() == 2
    assert train_data.get_batch_size() == 3
    num_iter4 = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter4 += 1
    assert num_iter4 == 2

    # case 5: test batch with drop_remainder=True
    train_data = ds.FakeImageDataset(num_images, image_size, num_classes, base_seed, num_samples=4)
    assert train_data.get_dataset_size() == 4
    assert train_data.get_batch_size() == 1
    train_data = train_data.batch(batch_size=3, drop_remainder=True)  # the rest of incomplete batch will be dropped
    assert train_data.get_dataset_size() == 1
    assert train_data.get_batch_size() == 3
    num_iter5 = 0
    for _ in train_data.create_dict_iterator(num_epochs=1):
        num_iter5 += 1
    assert num_iter5 == 1




def test_fake_image_sequential_sampler():
    """
    Feature: FakeImage
    Description: Test FakeImageDataset with SequentialSampler
    Expectation: The results are as expected
    """
    logger.info("Test FakeImageDataset Op with SequentialSampler")
    num_samples = 50
    sampler = ds.SequentialSampler(num_samples=num_samples)
    train_data1 = ds.FakeImageDataset(num_images, image_size, num_classes, base_seed, sampler=sampler)
    train_data2 = ds.FakeImageDataset(num_images, image_size, num_classes, base_seed, shuffle=False,
                                      num_samples=num_samples)

    label_list1, label_list2 = [], []
    num_iter = 0
    for item1, item2 in zip(train_data1.create_dict_iterator(num_epochs=1),
                            train_data2.create_dict_iterator(num_epochs=1)):
        label_list1.append(item1["label"].asnumpy())
        label_list2.append(item2["label"].asnumpy())
        num_iter += 1
    np.testing.assert_array_equal(label_list1, label_list2)
    assert num_iter == num_samples




def test_fake_image_visualize(plot=False):
    """
    Feature: FakeImage
    Description: Test FakeImageDataset visualized results
    Expectation: Get correct dataset of FakeImage
    """
    logger.info("Test FakeImageDataset visualization")

    train_data = ds.FakeImageDataset(num_images, image_size, num_classes, base_seed, num_samples=10, shuffle=False)
    num_iter = 0
    image_list, label_list = [], []
    for item in train_data.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item["image"]
        label = item["label"]
        image_list.append(image)
        label_list.append("label {}".format(label))
        assert isinstance(image, np.ndarray)
        assert image.shape == (28, 28, 3)
        assert image.dtype == np.uint8
        assert label.dtype == np.uint32
        num_iter += 1
    assert num_iter == 10
    if plot:
        visualize_dataset(image_list, label_list)


def test_fake_image_num_images():
    """
    Feature: FakeImage
    Description: Test FakeImageDataset with num images
    Expectation: Throw exception correctly or get correct dataset
    """
    logger.info("Test FakeImageDataset num_images flag")

    def test_config(test_num_images):

        try:
            data = ds.FakeImageDataset(test_num_images, image_size, num_classes, base_seed, shuffle=False)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    assert test_config(num_images) == num_images

    assert "Input num_images is not within the required interval of [1, 2147483647]." in test_config(-1)
    assert "is not of type [<class 'int'>], but got <class 'str'>." in test_config("10")


def test_fake_image_image_size():
    """
    Feature: FakeImage
    Description: Test FakeImageDataset with image size
    Expectation: Throw exception correctly or get correct dataset
    """
    logger.info("Test FakeImageDataset image_size flag")

    def test_config(test_image_size):
        try:
            data = ds.FakeImageDataset(num_images, test_image_size, num_classes, base_seed, shuffle=False)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    assert test_config(image_size) == num_images

    assert "Argument image_size[0] with value -1 is not of type [<class 'int'>], but got <class 'str'>."\
            in test_config(("-1", 28, 3))
    assert "image_size should be a list or tuple of length 3, but got 2" in test_config((2, 2))
    assert "Input image_size[0] is not within the required interval of [1, 2147483647]." in test_config((-1, 28, 3))


def test_fake_image_num_classes():
    """
    Feature: FakeImage
    Description: Test FakeImageDataset with num classes
    Expectation: Throw exception correctly or get correct dataset
    """
    logger.info("Test FakeImageDataset num_classes flag")

    def test_config(test_num_classes):
        try:
            data = ds.FakeImageDataset(num_images, image_size, test_num_classes, base_seed, shuffle=False)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    assert test_config(num_classes) == num_images

    assert "Input num_classes is not within the required interval of [1, 2147483647]." in test_config(-1)
    #should not be negative
    assert "is not of type [<class 'int'>], but got <class 'str'>." in test_config("10")
