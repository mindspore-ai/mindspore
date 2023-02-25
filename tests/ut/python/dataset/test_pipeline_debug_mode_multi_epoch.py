# Copyright 2023 Huawei Technologies Co., Ltd
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
Test multiple epoch scenarios in debug mode
"""
import math
import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import config_get_set_seed, visualize_list, config_get_set_num_parallel_workers

pytestmark = pytest.mark.forked

# tf_file_dataset description:
# test1.data: 10 samples - [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# test2.data: 10 samples - [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# test3.data: 10 samples - [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
# test4.data: 10 samples - [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
# test5.data: 10 samples - [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
TF_FILES = ["../data/dataset/tf_file_dataset/test1.data",
            "../data/dataset/tf_file_dataset/test2.data",
            "../data/dataset/tf_file_dataset/test3.data",
            "../data/dataset/tf_file_dataset/test4.data",
            "../data/dataset/tf_file_dataset/test5.data"]


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_pipeline_debug_mode_multi_epoch_celaba(my_debug_mode, plot=False):
    """
    Feature: Pipeline debug mode.
    Description: Test multiple epoch scenario using CelebADataset.
    Expectation: Output is equal to the expected output
    """
    logger.info("test_pipeline_debug_mode_multi_epoch_celaba")

    # Set configuration
    original_seed = config_get_set_seed(99)
    original_num_workers = config_get_set_num_parallel_workers(1)
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    # testCelebAData has 4 samples
    num_samples = 4

    data1 = ds.CelebADataset("../data/dataset/testCelebAData/", decode=True)

    # Confirm dataset size
    assert data1.get_dataset_size() == num_samples

    num_epoch = 2
    iter1 = data1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)
    epoch_count = 0
    sample_count = 0
    image_list = []
    for _ in range(num_epoch):
        row_count = 0
        for row_item in iter1:
            # Note: Each row in this CelebADataset pipeline has columns "image" and "attr"
            assert len(row_item) == 2
            assert row_item["image"].shape == (2268, 4032, 3)
            image = row_item["image"]
            image_list.append(image)
            row_count += 1
        assert row_count == num_samples
        epoch_count += 1
        sample_count += row_count
    assert epoch_count == num_epoch
    assert sample_count == num_samples * num_epoch
    if plot:
        visualize_list(image_list)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_workers)
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_pipeline_debug_mode_multi_epoch_celaba_take(my_debug_mode):
    """
    Feature: Pipeline debug mode.
    Description: Test multiple epoch scenario using CelebADataset with take op.
    Expectation: Output is equal to the expected output
    """
    logger.info("test_pipeline_debug_mode_multi_epoch_celaba_take")

    # Set configuration
    original_seed = config_get_set_seed(199)
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    num_samples = 4
    num_take = 3

    data1 = ds.CelebADataset("../data/dataset/testCelebAData/", num_samples=num_samples, decode=True)
    data1 = data1.take(num_take)

    num_epoch = 2
    iter1 = data1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)
    epoch_count = 0
    sample_count = 0
    for _ in range(num_epoch):
        row_count = 0
        for row_item in iter1:
            # Note: Each row in this CelebADataset pipeline has columns "image" and "attr"
            assert len(row_item) == 2
            assert row_item["image"].shape == (2268, 4032, 3)
            row_count += 1
        assert row_count == num_take
        epoch_count += 1
        sample_count += row_count
    assert epoch_count == num_epoch
    assert sample_count == num_take * num_epoch

    # Restore configuration
    ds.config.set_seed(original_seed)
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_pipeline_debug_mode_multi_epoch_cifar10_take(my_debug_mode):
    """
    Feature: Pipeline debug mode.
    Description: Test multiple epoch scenario using Cifar10Dataset with take op.
    Expectation: Output is equal to the expected output
    """
    logger.info("test_pipeline_debug_mode_multi_epoch_cifar10_take")

    # Set configuration
    original_seed = config_get_set_seed(299)
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    data_dir_10 = "../data/dataset/testCifar10Data"
    num_samples = 100
    num_repeat = 2
    num_take = 45

    data1 = ds.Cifar10Dataset(data_dir_10, num_samples=num_samples)
    data1 = data1.repeat(num_repeat)

    # Confirm dataset size (before take op applied)
    assert data1.get_dataset_size() == num_samples * num_repeat

    data1 = data1.take(num_take)

    num_epoch = 5
    iter1 = data1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)
    epoch_count = 0
    sample_count = 0
    for _ in range(num_epoch):
        row_count = 0
        for row_item in iter1:
            image = row_item["image"]
            assert image.shape == (32, 32, 3)
            row_count += 1
        assert row_count == num_take
        epoch_count += 1
        sample_count += row_count
    assert epoch_count == num_epoch
    assert sample_count == num_take * num_epoch

    # Restore configuration
    ds.config.set_seed(original_seed)
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_pipeline_debug_mode_multi_epoch_cifar10_repeat_batch(my_debug_mode):
    """
    Feature: Pipeline debug mode.
    Description: Test multiple epoch scenario using Cifar10Dataset with repeat op then batch op.
    Expectation: Output is equal to the expected output
    """
    logger.info("test_pipeline_debug_mode_multi_epoch_cifar10_repeat_batch")

    # Set configuration
    original_seed = config_get_set_seed(399)
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    data_dir_10 = "../data/dataset/testCifar10Data"
    num_samples = 40
    num_repeat = 2
    batch_size = 16

    data1 = ds.Cifar10Dataset(data_dir_10, num_samples=num_samples)
    # Add repeat then batch
    data1 = data1.repeat(num_repeat)
    data1 = data1.batch(batch_size, True)

    num_epoch = 2
    iter1 = data1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)
    epoch_count = 0
    sample_count = 0
    label_list = []
    label_golden = [[0, 7, 8, 4, 9, 1, 9, 8, 6, 2, 7, 0, 2, 1, 7, 0],
                    [1, 4, 4, 7, 9, 4, 7, 8, 3, 4, 3, 9, 4, 7, 3, 9],
                    [0, 0, 5, 6, 6, 6, 5, 4, 7, 6, 9, 0, 3, 4, 3, 7],
                    [2, 4, 0, 3, 1, 7, 7, 9, 9, 8, 7, 0, 6, 3, 7, 9],
                    [0, 8, 7, 8, 1, 6, 4, 2, 6, 4, 9, 5, 3, 2, 5, 9],
                    [2, 5, 3, 6, 2, 0, 7, 8, 3, 9, 6, 2, 4, 1, 5, 4],
                    [5, 0, 9, 8, 6, 4, 9, 2, 0, 0, 0, 2, 7, 2, 6, 4],
                    [6, 2, 7, 7, 2, 6, 9, 2, 8, 5, 7, 6, 6, 4, 6, 2],
                    [0, 7, 2, 9, 8, 6, 7, 6, 0, 3, 2, 2, 6, 8, 8, 2],
                    [4, 6, 6, 9, 8, 4, 6, 1, 0, 5, 5, 9, 2, 0, 8, 7]]
    for _ in range(num_epoch):
        row_count = 0
        label_list_per_epoch = []
        for row_item in iter1:
            image = row_item["image"]
            label = row_item["label"]
            assert image.shape == (batch_size, 32, 32, 3)
            label_list.append(label)
            label_list_per_epoch.append(label)
            row_count += 1
        logger.info("epoch_count is {}, label_list_per_epoch is {}".format(epoch_count, label_list_per_epoch))
        assert row_count == int(num_samples * num_repeat / batch_size)
        epoch_count += 1
        sample_count += row_count
    assert epoch_count == num_epoch
    assert sample_count == int(num_samples * num_repeat / batch_size) * num_epoch
    np.testing.assert_array_equal(label_list, np.array(label_golden))

    # Restore configuration
    ds.config.set_seed(original_seed)
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_pipeline_debug_mode_multi_epoch_cifar10_batch_repeat(my_debug_mode):
    """
    Feature: Pipeline debug mode.
    Description: Test multiple epoch scenario using Cifar10Dataset with batch op then repeat op.
    Expectation: Output is equal to the expected output
    """
    logger.info("test_pipeline_debug_mode_multi_epoch_cifar10_batch_repeat")
    # Set configuration
    original_seed = config_get_set_seed(499)
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    data_dir_10 = "../data/dataset/testCifar10Data"
    num_samples = 10
    batch_size = 2
    num_repeat = 2

    data1 = ds.Cifar10Dataset(data_dir_10, num_samples=num_samples)
    # Add batch then repeat
    data1 = data1.batch(batch_size, True)
    data1 = data1.repeat(num_repeat)

    num_epoch = 2
    iter1 = data1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)
    epoch_count = 0
    sample_count = 0
    for _ in range(num_epoch):
        row_count = 0
        label_list_per_epoch = []
        for row_item in iter1:
            image = row_item["image"]
            label = row_item["label"]
            assert image.shape == (batch_size, 32, 32, 3)
            label_list_per_epoch.append(label)
            row_count += 1
        logger.info("epoch_count is {}, label_list_per_epoch is {}".format(epoch_count, label_list_per_epoch))
        assert row_count == int(num_samples * num_repeat / batch_size)
        epoch_count += 1
        sample_count += row_count
    assert epoch_count == num_epoch
    assert sample_count == int(num_samples * num_repeat / batch_size) * num_epoch

    # Restore configuration
    ds.config.set_seed(original_seed)
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_pipeline_debug_mode_multi_epoch_cifar10_zip(my_debug_mode):
    """
    Feature: Pipeline debug mode.
    Description: Test multiple epoch scenario using Cifar10Dataset with zip op.
    Expectation: Output is equal to the expected output
    """
    logger.info("test_pipeline_debug_mode_multi_epoch_cifar10_zip")

    # Set configuration
    original_seed = config_get_set_seed(599)
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    data_dir_10 = "../data/dataset/testCifar10Data"
    num_samples = 20
    num_repeat = 5
    batch_size = 10

    data1 = ds.Cifar10Dataset(data_dir_10, num_samples=num_samples)

    data2 = ds.Cifar10Dataset(data_dir_10, num_samples=num_samples)
    # Rename dataset2 for no conflict
    data2 = data2.rename(input_columns=["image", "label"], output_columns=["image2", "label2"])

    # Create data3 with zip
    data3 = ds.zip((data1, data2))
    # Add repeat then batch
    data3 = data3.repeat(num_repeat)
    data3 = data3.batch(batch_size, True)

    num_epoch = 2
    iter1 = data3.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)
    epoch_count = 0
    sample_count = 0
    for _ in range(num_epoch):
        row_count = 0
        label_list_per_epoch = []
        for row_item in iter1:
            image = row_item["image"]
            label = row_item["label"]
            assert image.shape == (batch_size, 32, 32, 3)
            label_list_per_epoch.append(label)
            row_count += 1
        logger.info("epoch_count is {}, label_list_per_epoch is {}".format(epoch_count, label_list_per_epoch))
        assert row_count == int(num_samples * num_repeat / batch_size)
        epoch_count += 1
        sample_count += row_count
    assert epoch_count == num_epoch
    assert sample_count == int(num_samples * num_repeat / batch_size) * num_epoch

    # Restore configuration
    ds.config.set_seed(original_seed)
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_pipeline_debug_mode_multi_epoch_cifar10_zip_batch_repeat(my_debug_mode):
    """
    Feature: Pipeline debug mode.
    Description: Test multiple epoch scenario using Cifar10Dataset with zip, batch and repeat.
    Expectation: Output is equal to the expected output
    """
    logger.info("test_pipeline_debug_mode_multi_epoch_cifar10_zip_batch_repeat")

    # Set configuration
    original_seed = config_get_set_seed(699)
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    data_dir_10 = "../data/dataset/testCifar10Data"
    num_samples = 20
    batch_size = 10
    num_repeat = 3

    data1 = ds.Cifar10Dataset(data_dir_10, num_samples=num_samples)

    data2 = ds.Cifar10Dataset(data_dir_10, num_samples=num_samples)
    # Rename dataset2 for no conflict
    data2 = data2.rename(input_columns=["image", "label"], output_columns=["image2", "label2"])

    # Create data3 with zip
    data3 = ds.zip((data1, data2))
    # Add batch then repeat
    data3 = data3.batch(batch_size, True)
    data3 = data3.repeat(num_repeat)

    num_epoch = 2
    iter1 = data3.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)
    epoch_count = 0
    sample_count = 0
    label_list = []
    label_golden = [[5, 1, 3, 6, 2, 7, 5, 2, 1, 9],
                    [3, 0, 9, 1, 1, 2, 5, 5, 6, 3],
                    [5, 5, 8, 0, 8, 5, 4, 7, 2, 2],
                    [2, 2, 4, 8, 1, 1, 3, 0, 5, 8],
                    [1, 1, 0, 5, 5, 5, 8, 4, 4, 1],
                    [8, 2, 9, 0, 8, 1, 6, 0, 1, 8],
                    [7, 0, 6, 1, 6, 2, 7, 4, 2, 3],
                    [9, 8, 0, 2, 7, 4, 1, 9, 8, 3],
                    [7, 0, 2, 6, 2, 0, 2, 0, 7, 0],
                    [4, 7, 7, 7, 6, 5, 3, 4, 5, 9],
                    [1, 9, 7, 5, 7, 7, 2, 2, 9, 2],
                    [8, 8, 5, 1, 4, 0, 5, 5, 6, 6]]
    for _ in range(num_epoch):
        row_count = 0
        label_list_per_epoch = []
        for row_item in iter1:
            image = row_item["image"]
            label = row_item["label"]
            assert image.shape == (batch_size, 32, 32, 3)
            label_list.append(label)
            label_list_per_epoch.append(label)
            row_count += 1
        logger.info("epoch_count is {}, label_list_per_epoch is {}".format(epoch_count, label_list_per_epoch))
        assert row_count == int(num_samples * num_repeat / batch_size)
        epoch_count += 1
        sample_count += row_count
    assert epoch_count == num_epoch
    assert sample_count == int(num_samples * num_repeat / batch_size) * num_epoch
    np.testing.assert_array_equal(label_list, np.array(label_golden))

    # Restore configuration
    ds.config.set_seed(original_seed)
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_pipeline_debug_mode_multi_epoch_imagefolder(my_debug_mode, plot=False):
    """
    Feature: Pipeline debug mode.
    Description: Test multiple epoch scenario using ImageFolderDataset. Plot support provided.
    Expectation: Output is equal to the expected output
    """
    logger.info("test_pipeline_debug_mode_multi_epoch_imagefolder")

    # Set configuration
    original_seed = config_get_set_seed(899)
    original_num_workers = config_get_set_num_parallel_workers(1)
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    # Note: testImageNetData4 has 7 samples in total
    num_samples = 7

    # Use all 7 samples from the dataset
    data1 = ds.ImageFolderDataset("../data/dataset/testImageNetData4/train",
                                  decode=True)
    # Confirm dataset size
    assert data1.get_dataset_size() == num_samples

    num_epoch = 4
    iter1 = data1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)
    epoch_count = 0
    sample_count = 0
    image_list = []
    label_list = []
    label_golden = [4, 3, 5, 0, 1, 2, 6] + [3, 2, 5, 1, 0, 6, 4] + [6, 0, 1, 2, 5, 4, 3] + [3, 4, 5, 1, 0, 6, 2]
    for _ in range(num_epoch):
        row_count = 0
        label_list_per_epoch = []
        for row_item in iter1:
            image = row_item["image"]
            label = row_item["label"]
            assert image.shape == (384, 682, 3)
            image_list.append(image)
            label_list.append(label)
            label_list_per_epoch.append(label)
            row_count += 1
        logger.info("epoch_count is {}, label_list_per_epoch is {}".format(epoch_count, label_list_per_epoch))
        assert row_count == num_samples
        epoch_count += 1
        sample_count += row_count
    assert epoch_count == num_epoch
    assert sample_count == num_samples * num_epoch
    assert label_list == label_golden
    if plot:
        visualize_list(image_list)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_workers)
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode, my_shuffle", [(False, True), (True, None), (True, True), (True, False)])
def test_pipeline_debug_mode_multi_epoch_imagefolder_shuffle(my_debug_mode, my_shuffle, plot=False):
    """
    Feature: Pipeline debug mode.
    Description: Test multiple epoch scenario using ImageFolderDataset with shuffle parameter. Plot support provided.
    Expectation: Output is equal to the expected output
    """
    logger.info("test_pipeline_debug_mode_multi_epoch_imagefolder_shuffle")

    # Set configuration
    original_seed = config_get_set_seed(899)
    original_num_workers = config_get_set_num_parallel_workers(1)
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    num_samples = 5
    data1 = ds.ImageFolderDataset("../data/dataset/testImageNetData4/train",
                                  shuffle=my_shuffle,
                                  num_samples=num_samples,
                                  decode=True)

    num_epoch = 2
    iter1 = data1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)
    epoch_count = 0
    sample_count = 0
    image_list = []
    label_list = []
    if my_shuffle is False:
        # Sequential order is used
        label_golden = list(range(0, 5)) * num_epoch
    else:
        # Random order is used, according to the seed value
        label_golden = [2, 1, 3, 0, 4] + [3, 6, 2, 0, 3]
    for _ in range(num_epoch):
        row_count = 0
        label_list_per_epoch = []
        for row_item in iter1:
            image = row_item["image"]
            label = row_item["label"]
            assert image.shape == (384, 682, 3)
            image_list.append(image)
            label_list.append(label)
            label_list_per_epoch.append(label)
            row_count += 1
        logger.info("epoch_count is {}, label_list_per_epoch is {}".format(epoch_count, label_list_per_epoch))
        assert row_count == num_samples
        epoch_count += 1
        sample_count += row_count
    assert epoch_count == num_epoch
    assert sample_count == num_samples * num_epoch
    assert label_list == label_golden
    if plot:
        visualize_list(image_list)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_workers)
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_pipeline_debug_mode_multi_epoch_imagefolder_repeat(my_debug_mode, plot=False):
    """
    Feature: Pipeline debug mode.
    Description: Test multiple epoch scenario using ImageFolderDataset with repeat op. Plot support provided.
    Expectation: Output is equal to the expected output
    """
    logger.info("test_pipeline_debug_mode_multi_epoch_imagefolder_repeat")

    # Set configuration
    original_seed = config_get_set_seed(899)
    original_num_workers = config_get_set_num_parallel_workers(1)
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    num_samples = 5
    num_repeat = 3
    data1 = ds.ImageFolderDataset("../data/dataset/testImageNetData4/train",
                                  shuffle=True,
                                  num_samples=num_samples,
                                  decode=True)
    data1 = data1.repeat(num_repeat)

    num_epoch = 2
    iter1 = data1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)
    epoch_count = 0
    sample_count = 0
    image_list = []
    label_list = []
    # Random order is used, according to the seed value
    label_golden = [2, 1, 3, 0, 4] + [3, 6, 2, 0, 3] + \
                   [5, 4, 0, 1, 0] + [0, 0, 1, 3, 5] + \
                   [4, 5, 5, 2, 0] + [1, 2, 4, 4, 5]
    for _ in range(num_epoch):
        row_count = 0
        label_list_per_epoch = []
        for row_item in iter1:
            image = row_item["image"]
            label = row_item["label"]
            assert image.shape == (384, 682, 3)
            image_list.append(image)
            label_list.append(label)
            label_list_per_epoch.append(label)
            row_count += 1
        logger.info("epoch_count is {}, label_list_per_epoch is {}".format(epoch_count, label_list_per_epoch))
        assert row_count == num_samples * num_repeat
        epoch_count += 1
        sample_count += row_count
    assert epoch_count == num_epoch
    assert sample_count == num_samples * num_repeat * num_epoch
    assert label_list == label_golden

    if plot:
        visualize_list(image_list)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_workers)
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_pipeline_debug_mode_multi_epoch_imagefolder_map_pyfunc(my_debug_mode):
    """
    Feature: Pipeline debug mode.
    Description: Test multiple epoch scenario using ImageFolderDataset with map(PyFunc).
    Expectation: Output is equal to the expected output
    """
    logger.info("test_pipeline_debug_mode_multi_epoch_imagefolder_map_pyfunc")

    # Set configuration
    original_seed = config_get_set_seed(899)
    original_num_workers = config_get_set_num_parallel_workers(1)
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    data = ds.ImageFolderDataset("../data/dataset/testImageNetData4/train",
                                 sampler=ds.SequentialSampler(), decode=True)
    num_repeat = 3
    sample_row = 7
    data = data.repeat(num_repeat)
    data = data.map(operations=[(lambda x: x - 1), (lambda x: x * 2)], input_columns=["image"])
    num_epoch = 2
    epoch_count = 0
    sample_count = 0
    iter1 = data.create_dict_iterator(num_epochs=num_epoch)
    for _ in range(num_epoch):
        num_rows = 0
        for row_item in iter1:
            assert len(row_item) == 2
            assert row_item["image"].shape == (384, 682, 3)
            num_rows += 1
        assert num_rows == sample_row * num_repeat
        sample_count += num_rows
        epoch_count += 1
    assert epoch_count == num_epoch
    assert sample_count == num_repeat * num_epoch * sample_row

    err_msg = "EOF buffer encountered. User tries to fetch data beyond the specified number of epochs."
    with pytest.raises(RuntimeError, match=err_msg):
        iter1.__next__()

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_workers)
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode, my_drop, my_num_samples",
                         [(False, False, 6), (True, False, 6), (True, True, 7)])
def test_pipeline_debug_mode_multi_ep_im_batch_no_remainder(my_debug_mode, my_drop, my_num_samples, plot=False):
    """
    Feature: Pipeline debug mode.
    Description: Test multiple epoch scenario using ImageFolderDataset with batch op and no remainder.
    Expectation: Output is equal to the expected output
    """
    logger.info("test_pipeline_debug_mode_multi_ep_im_batch_no_remainder")

    # Set configuration
    original_seed = config_get_set_seed(899)
    original_num_workers = config_get_set_num_parallel_workers(1)
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    num_samples = my_num_samples
    batch_size = 2

    data1 = ds.ImageFolderDataset("../data/dataset/testImageNetData4/train",
                                  num_samples=num_samples,
                                  decode=True)
    data1 = data1.batch(batch_size, drop_remainder=my_drop)

    num_epoch = 3
    iter1 = data1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)
    epoch_count = 0
    sample_count = 0
    image_list = []
    label_list = []
    label_golden = [[2, 1], [3, 0], [4, 6]] + [[3, 6], [2, 0], [3, 5]] + [[5, 4], [0, 1], [0, 6]]
    for _ in range(num_epoch):
        row_count = 0
        label_list_per_epoch = []
        for row_item in iter1:
            image = row_item["image"]
            label = row_item["label"]
            assert image.shape == (2, 384, 682, 3)
            image_list.append(image[0])
            label_list.append(label)
            label_list_per_epoch.append(label)
            row_count += 1
        logger.info("epoch_count is {}, label_list_per_epoch is {}".format(epoch_count, label_list_per_epoch))
        assert row_count == int(num_samples / batch_size)
        epoch_count += 1
        sample_count += row_count

    assert epoch_count == num_epoch
    assert sample_count == int(num_samples / batch_size) * num_epoch
    np.testing.assert_array_equal(label_list, np.array(label_golden))
    if plot:
        visualize_list(image_list)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_workers)
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode, my_drop, my_num_samples",
                         [(False, False, 7), (True, False, 7)])
def test_pipeline_debug_mode_multi_ep_im_batch_with_remainders(my_debug_mode, my_drop, my_num_samples, plot=False):
    """
    Feature: Pipeline debug mode.
    Description: Test multiple epoch scenario using ImageFolderDataset with batch op and remainder.
    Expectation: Output is equal to the expected output
    """
    logger.info("test_pipeline_debug_mode_multi_ep_im_batch_with_remainders")

    # Set configuration
    original_seed = config_get_set_seed(899)
    original_num_workers = config_get_set_num_parallel_workers(1)
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    num_samples = my_num_samples
    batch_size = 2

    data1 = ds.ImageFolderDataset("../data/dataset/testImageNetData4/train",
                                  num_samples=num_samples,
                                  decode=True)
    data1 = data1.batch(batch_size, drop_remainder=my_drop)

    num_epoch = 3
    iter1 = data1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)
    epoch_count = 0
    sample_count = 0
    image_list = []
    label_list = []
    label_golden = [[2, 1], [3, 0], [4, 6], [5]] + [[3, 6], [2, 0], [3, 5], [1]] + [[5, 4], [0, 1], [0, 6], [6]]
    for _ in range(num_epoch):
        row_count = 0
        label_list_per_epoch = []
        for row_item in iter1:
            image = row_item["image"]
            label = row_item["label"]
            assert image.shape == (2, 384, 682, 3) or (1, 384, 682, 3)
            image_list.append(image[0])
            label_list.append(list(label))
            label_list_per_epoch.append(list(label))
            row_count += 1
        logger.info("epoch_count is {}, label_list_per_epoch is {}".format(epoch_count, label_list_per_epoch))
        assert row_count == math.ceil(num_samples / batch_size)
        epoch_count += 1
        sample_count += row_count
    assert epoch_count == num_epoch
    assert sample_count == math.ceil(num_samples / batch_size) * num_epoch
    assert label_list == label_golden
    if plot:
        visualize_list(image_list)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_workers)
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_pipeline_debug_mode_multi_epoch_tfrecord_shuffle(my_debug_mode):
    """
    Feature: Pipeline debug mode
    Description: Test multiple epoch scenario using (non-mappable) TFRecordDataset with various shuffle parameter values
    Expectation: The dataset is processed as expected
    """
    logger.info("test_pipeline_debug_mode_multi_epoch_tfrecord_shuffle")

    def test_config(seed, shuffle, my_num_samples, num_epoch):
        # Set seed configuration
        original_seed = config_get_set_seed(seed)

        # Set num_samples for calculations
        num_samples = my_num_samples if my_num_samples is not None else 50

        data1 = ds.TFRecordDataset(TF_FILES, num_samples=my_num_samples, shuffle=shuffle)
        iter1 = data1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)
        epoch_count = 0
        sample_count = 0
        res_list = []

        for _ in range(num_epoch):
            row_count = 0
            res_list_per_epoch = []
            for row_item in iter1:
                scalar = row_item["scalars"][0]
                res_list.append(scalar)
                res_list_per_epoch.append(scalar)
                row_count += 1
            logger.info("epoch_count is {}, res_list_per_epoch is {}".format(epoch_count, res_list_per_epoch))
            epoch_count += 1
            sample_count += row_count
        assert epoch_count == num_epoch
        assert sample_count == num_samples * num_epoch

        # Restore seed configuration
        ds.config.set_seed(original_seed)

        # Return results list
        return res_list

    # Enable debug mode
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    # Test various configurations

    # Test shuffle=False with all samples
    shuffle_false_golden = ([1, 11, 21, 31, 41, 2, 12, 22, 32, 42, 3, 13, 23, 33, 43, 4, 14, 24, 34, 44] + \
                            [5, 15, 25, 35, 45, 6, 16, 26, 36, 46, 7, 17, 27, 37, 47, 8, 18, 28, 38, 48] + \
                            [9, 19, 29, 39, 49, 10, 20, 30, 40, 50]) * 2
    assert test_config(1001, False, None, 2) == shuffle_false_golden

    # Test ds.Shuffle.INFILE (same as shuffle=False) with 12 samples
    shuffle_false_golden = [1, 11, 21, 31, 41, 2, 12, 22, 32, 42, 3, 13] * 2
    assert test_config(1001, ds.Shuffle.INFILE, 12, 2) == shuffle_false_golden

    # Test shuffle=FILES
    shuffle_files_golden = [11, 1, 21, 31, 41, 12, 2, 22, 32, 42, 13, 3] + \
                           [1, 11, 21, 31, 41, 2, 12, 22, 32, 42, 3, 13] + \
                           [11, 1, 21, 31, 41, 12, 2, 22, 32, 42, 13, 3] + \
                           [1, 11, 21, 31, 41, 2, 12, 22, 32, 42, 3, 13] + \
                           [11, 1, 21, 31, 41, 12, 2, 22, 32, 42, 13, 3]
    assert test_config(1, ds.Shuffle.FILES, 12, 5) == shuffle_files_golden
    assert test_config(1, ds.Shuffle.FILES, 12, 3) == shuffle_files_golden[0:36]

    shuffle_files_golden = [11, 31, 41, 21, 1, 12, 32, 42, 22, 2, 13, 33] + \
                           [31, 21, 1, 41, 11, 32, 22, 2, 42, 12, 33, 23] + \
                           [21, 41, 11, 1, 31, 22, 42, 12, 2, 32, 23, 43] + \
                           [41, 1, 31, 11, 21, 42, 2, 32, 12, 22, 43, 3] + \
                           [1, 11, 21, 31, 41, 2, 12, 22, 32, 42, 3, 13]
    assert test_config(1001, ds.Shuffle.FILES, 12, 5) == shuffle_files_golden

    # Test shuffle=GLOBAL
    shuffle_global_golden = [32, 1, 11, 42, 3, 22, 21, 2, 41, 13, 31, 12] + \
                            [1, 41, 22, 2, 3, 32, 11, 13, 21, 31, 42, 12] + \
                            [3, 1, 12, 22, 42, 2, 32, 31, 21, 13, 11, 41] + \
                            [31, 21, 3, 2, 13, 22, 32, 41, 1, 11, 12, 42] + \
                            [13, 3, 11, 22, 41, 21, 31, 32, 42, 1, 12, 2]
    result = test_config(1, ds.Shuffle.GLOBAL, 12, 5)
    assert result == shuffle_global_golden

    # Restore debug mode configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_pipeline_debug_mode_multi_epoch_tfrecord_ops(my_debug_mode):
    """
    Feature: Pipeline debug mode
    Description: Test multiple epoch scenario using (non-mappable) TFRecordDataset with various ops in data pipeline
    Expectation: The dataset is processed as expected
    """
    logger.info("test_pipeline_debug_mode_multi_epoch_tfrecord_ops")

    def test_config(seed, my_pipeline, num_epoch):
        # Set seed configuration
        original_seed = config_get_set_seed(seed)

        iter1 = my_pipeline.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)
        epoch_count = 0
        res_list = []

        for _ in range(num_epoch):
            res_list_per_epoch = []
            for row_item in iter1:
                scalars = row_item["scalars"]
                res_list.append(scalars)
                res_list_per_epoch.append(scalars)
            logger.info("epoch_count is {}, res_list_per_epoch is {}".format(epoch_count, res_list_per_epoch))
            epoch_count += 1
        assert epoch_count == num_epoch

        # Restore seed configuration
        ds.config.set_seed(original_seed)

        # Return results list
        return res_list

    # Set configuration
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    # Test various configurations

    # Test pipeline with just non-mappable source op
    data1 = ds.TFRecordDataset(TF_FILES, num_samples=16, shuffle=ds.Shuffle.FILES)
    golden1 = [21, 11, 41, 31, 1, 22, 12, 42, 32, 2, 23, 13, 43, 33, 3, 24] + \
              [41, 11, 1, 31, 21, 42, 12, 2, 32, 22, 43, 13, 3, 33, 23, 44] + \
              [1, 11, 21, 31, 41, 2, 12, 22, 32, 42, 3, 13, 23, 33, 43, 4]
    assert test_config(1202, data1, 3) == golden1

    # Test with repeat op
    data1 = ds.TFRecordDataset(TF_FILES, num_samples=8, shuffle=ds.Shuffle.FILES)
    data1 = data1.repeat(2)
    golden1 = [21, 11, 41, 31, 1, 22, 12, 42, 41, 11, 1, 31, 21, 42, 12, 2] + \
              [1, 11, 21, 31, 41, 2, 12, 22, 21, 11, 41, 31, 1, 22, 12, 42] + \
              [41, 11, 1, 31, 21, 42, 12, 2, 1, 11, 21, 31, 41, 2, 12, 22]
    assert test_config(1202, data1, 3) == golden1

    # Test with shuffle op
    data1 = ds.TFRecordDataset(TF_FILES, num_samples=16, shuffle=False)
    data1 = data1.shuffle(buffer_size=8)
    golden1 = [1, 2, 42, 21, 22, 31, 33, 32, 3, 43, 12, 11, 23, 4, 41, 13] + \
              [11, 41, 31, 32, 1, 2, 22, 23, 33, 13, 42, 4, 43, 12, 21, 3] + \
              [2, 41, 11, 3, 42, 1, 22, 43, 12, 21, 13, 23, 33, 31, 32, 4]
    result = test_config(1202, data1, 3)
    assert result == golden1

    # Test with batch op
    data1 = ds.TFRecordDataset(TF_FILES, num_samples=16, shuffle=ds.Shuffle.FILES)
    data1 = data1.batch(batch_size=8)
    golden1 = [[[21], [11], [41], [31], [1], [22], [12], [42]], [[32], [2], [23], [13], [43], [33], [3], [24]],
               [[41], [11], [1], [31], [21], [42], [12], [2]], [[32], [22], [43], [13], [3], [33], [23], [44]],
               [[1], [11], [21], [31], [41], [2], [12], [22]], [[32], [42], [3], [13], [23], [33], [43], [4]]]
    np.testing.assert_array_equal(test_config(1202, data1, 3), np.array(golden1))

    # Test with repeat op then batch
    data1 = ds.TFRecordDataset(TF_FILES, num_samples=8, shuffle=ds.Shuffle.FILES)
    data1 = data1.repeat(2)
    data1 = data1.batch(batch_size=8)
    golden1 = [[[21], [11], [41], [31], [1], [22], [12], [42]],
               [[41], [11], [1], [31], [21], [42], [12], [2]],
               [[1], [11], [21], [31], [41], [2], [12], [22]],
               [[21], [11], [41], [31], [1], [22], [12], [42]],
               [[41], [11], [1], [31], [21], [42], [12], [2]],
               [[1], [11], [21], [31], [41], [2], [12], [22]]]
    np.testing.assert_array_equal(test_config(1202, data1, 3), np.array(golden1))

    # Test with batch then repeat
    data1 = ds.TFRecordDataset(TF_FILES, num_samples=8, shuffle=ds.Shuffle.FILES)
    data1 = data1.batch(batch_size=8)
    data1 = data1.repeat(2)
    golden1 = [[[21], [11], [41], [31], [1], [22], [12], [42]],
               [[41], [11], [1], [31], [21], [42], [12], [2]],
               [[1], [11], [21], [31], [41], [2], [12], [22]],
               [[21], [11], [41], [31], [1], [22], [12], [42]],
               [[41], [11], [1], [31], [21], [42], [12], [2]],
               [[1], [11], [21], [31], [41], [2], [12], [22]]]
    np.testing.assert_array_equal(test_config(1202, data1, 3), np.array(golden1))

    # Test with map op
    data1 = ds.TFRecordDataset(TF_FILES, num_samples=16, shuffle=ds.Shuffle.FILES)
    data1 = data1.map(operations=[(lambda x: x + 1)], input_columns=["scalars"])
    golden1 = [22, 12, 42, 32, 2, 23, 13, 43, 33, 3, 24, 14, 44, 34, 4, 25] + \
              [42, 12, 2, 32, 22, 43, 13, 3, 33, 23, 44, 14, 4, 34, 24, 45] + \
              [2, 12, 22, 32, 42, 3, 13, 23, 33, 43, 4, 14, 24, 34, 44, 5]
    assert test_config(1202, data1, 3) == golden1

    # Test common pipeline: map -> batch
    data1 = ds.TFRecordDataset(TF_FILES, num_samples=16, shuffle=ds.Shuffle.FILES)
    data1 = data1.map(operations=[(lambda x: x + 1)], input_columns=["scalars"])
    data1 = data1.batch(batch_size=4)
    golden1 = [[[22], [12], [42], [32]], [[2], [23], [13], [43]], [[33], [3], [24], [14]], [[44], [34], [4], [25]],
               [[42], [12], [2], [32]], [[22], [43], [13], [3]], [[33], [23], [44], [14]], [[4], [34], [24], [45]],
               [[2], [12], [22], [32]], [[42], [3], [13], [23]], [[33], [43], [4], [14]], [[24], [34], [44], [5]]]
    np.testing.assert_array_equal(test_config(1202, data1, 3), np.array(golden1))

    # Test common pipeline: map -> repeat -> shuffle -> batch
    data1 = ds.TFRecordDataset(TF_FILES, num_samples=8, shuffle=False)
    data1 = data1.map(operations=[(lambda x: x + 1)], input_columns=["scalars"])
    data1 = data1.repeat(2)
    data1 = data1.shuffle(buffer_size=16)
    data1 = data1.batch(batch_size=8)
    golden1 = [[[2], [13], [3], [32], [2], [23], [3], [23]], [[22], [12], [13], [12], [42], [22], [42], [32]],
               [[12], [13], [3], [2], [2], [42], [32], [42]], [[23], [22], [12], [23], [3], [13], [22], [32]],
               [[3], [3], [13], [13], [23], [22], [32], [32]], [[42], [22], [12], [2], [12], [2], [42], [23]]]
    result = test_config(1202, data1, 3)
    np.testing.assert_array_equal(result, np.array(golden1))

    # Restore configuration
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_pipeline_debug_mode_multi_epoch_cifar100_per_batch_map(my_debug_mode):
    """
    Feature: Pipeline debug mode.
    Description: Test multiple epoch scenario using Cifar100Dataset with Batch op with per_batch_map parameter
    Expectation: Output is equal to the expected output
    """
    logger.info("test_pipeline_debug_mode_multi_epoch_cifar10_zip_batch_repeat")

    # Set configuration
    original_seed = config_get_set_seed(1399)
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    # Define dataset pipeline
    num_samples = 12
    cifar100_ds = ds.Cifar100Dataset("../data/dataset/testCifar100Data", num_samples=num_samples)
    cifar100_ds = cifar100_ds.map(operations=[vision.ToType(np.int32)], input_columns="fine_label")
    cifar100_ds = cifar100_ds.map(operations=[lambda z: z], input_columns="image")

    # Callable function to delete 3rd column
    def del_column(col1, col2, col3, batch_info):
        return (col1, col2,)

    # Apply Dataset Ops
    batch_size = 4
    num_repeat = 2
    # Note: Test repeat before batch
    cifar100_ds = cifar100_ds.repeat(num_repeat)
    cifar100_ds = cifar100_ds.batch(batch_size, per_batch_map=del_column,
                                    input_columns=['image', 'fine_label', 'coarse_label'],
                                    output_columns=['image', 'label'], drop_remainder=False)

    # Iterate over data pipeline
    num_epoch = 2
    iter1 = cifar100_ds.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)
    epoch_count = 0
    sample_count = 0
    label_list = []
    label_golden = [[30, 26, 59, 25], [51, 7, 89, 73], [50, 74, 56, 70],
                    [23, 11, 3, 73], [96, 98, 31, 24], [9, 4, 86, 30],
                    [35, 56, 59, 57], [43, 53, 87, 59], [58, 7, 95, 76],
                    [27, 16, 69, 30], [21, 62, 67, 67], [78, 94, 58, 27]]
    for _ in range(num_epoch):
        row_count = 0
        label_list_per_epoch = []
        for row_item in iter1:
            image = row_item["image"]
            label = row_item["label"]
            assert len(row_item) == 2
            assert image.shape == (batch_size, 32, 32, 3)
            label_list.append(label)
            label_list_per_epoch.append(label)
            row_count += 1
        logger.info("epoch_count is {}, label_list_per_epoch is {}".format(epoch_count, label_list_per_epoch))
        assert row_count == int(num_samples * num_repeat / batch_size)
        epoch_count += 1
        sample_count += row_count
    assert epoch_count == num_epoch
    assert sample_count == int(num_samples * num_repeat / batch_size) * num_epoch
    np.testing.assert_array_equal(label_list, np.array(label_golden))

    # Restore configuration
    ds.config.set_seed(original_seed)
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_pipeline_debug_mode_batch_map_get_batch_num(my_debug_mode):
    """
    Feature: Batch op
    Description: Test basic map Batch op with per_batch_map function calling get_batch_num()
    Expectation: Output is equal to the expected output
    """
    # Set configuration
    original_seed = config_get_set_seed(1499)
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    def check_res(arr1, arr2):
        assert len(arr1) == len(arr2)
        for ind, _ in enumerate(arr1):
            assert np.array_equal(arr1[ind], np.array(arr2[ind]))

    def gen(num):
        for i in range(num):
            yield (np.array([i]),)

    def invert_sign_per_batch(col_list, batch_info):
        return ([np.copy(((-1) ** batch_info.get_batch_num()) * arr) for arr in col_list],)

    def batch_map_config(num_samples, num_repeat, batch_size, num_epoch, func, res):
        data1 = ds.GeneratorDataset((lambda: gen(num_samples)), ["num"],
                                    python_multiprocessing=False)
        data1 = data1.batch(batch_size=batch_size, input_columns=["num"], per_batch_map=func,
                            python_multiprocessing=False, num_parallel_workers=1)
        data1 = data1.repeat(num_repeat)

        # Iterate over data pipeline
        iter1 = data1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)
        epoch_count = 0
        sample_count = 0
        for _ in range(num_epoch):
            row_count = 0
            for item in iter1:
                res.append(item["num"])
                row_count += 1
            assert row_count == int(num_samples * num_repeat / batch_size)
            epoch_count += 1
            sample_count += row_count
        assert epoch_count == num_epoch
        assert sample_count == int(num_samples * num_repeat / batch_size) * num_epoch

    res1, res2, res3, res4 = [], [], [], []

    # Test repeat=1, num_epochs=1
    batch_map_config(4, 1, 2, 1, invert_sign_per_batch, res1)
    check_res(res1, [[[0], [1]], [[-2], [-3]]])

    # Test repeat=2, num_epochs=1
    batch_map_config(4, 2, 2, 1, invert_sign_per_batch, res2)
    check_res(res2, [[[0], [1]], [[-2], [-3]], [[0], [1]], [[-2], [-3]]])

    # Test repeat=1, num_epochs=3
    batch_map_config(4, 1, 2, 3, invert_sign_per_batch, res3)
    check_res(res3, [[[0], [1]], [[-2], [-3]], [[0], [1]], [[-2], [-3]], [[0], [1]], [[-2], [-3]]])

    # Test repeat=2, num_epochs=2
    batch_map_config(4, 2, 2, 2, invert_sign_per_batch, res4)
    check_res(res4, [[[0], [1]], [[-2], [-3]], [[0], [1]], [[-2], [-3]],
                     [[0], [1]], [[-2], [-3]], [[0], [1]], [[-2], [-3]]])

    # Restore configuration
    ds.config.set_seed(original_seed)
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


if __name__ == '__main__':
    test_pipeline_debug_mode_multi_epoch_celaba(True, plot=True)
    test_pipeline_debug_mode_multi_epoch_celaba_take(True)
    test_pipeline_debug_mode_multi_epoch_cifar10_take(True)
    test_pipeline_debug_mode_multi_epoch_cifar10_repeat_batch(True)
    test_pipeline_debug_mode_multi_epoch_cifar10_batch_repeat(True)
    test_pipeline_debug_mode_multi_epoch_cifar10_zip(True)
    test_pipeline_debug_mode_multi_epoch_cifar10_zip_batch_repeat(True)
    test_pipeline_debug_mode_multi_epoch_imagefolder(True, plot=True)
    test_pipeline_debug_mode_multi_epoch_imagefolder_shuffle(True, True, plot=True)
    test_pipeline_debug_mode_multi_epoch_imagefolder_repeat(True, plot=True)
    test_pipeline_debug_mode_multi_epoch_imagefolder_map_pyfunc(True)
    test_pipeline_debug_mode_multi_ep_im_batch_no_remainder(True, True, 7, plot=True)
    test_pipeline_debug_mode_multi_ep_im_batch_with_remainders(True, False, 7, plot=True)
    test_pipeline_debug_mode_multi_epoch_tfrecord_shuffle(True)
    test_pipeline_debug_mode_multi_epoch_tfrecord_ops(True)
    test_pipeline_debug_mode_multi_epoch_cifar100_per_batch_map(True)
    test_pipeline_debug_mode_batch_map_get_batch_num(True)
