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
from mindspore import log as logger
from util import config_get_set_seed, visualize_list, config_get_set_num_parallel_workers

pytestmark = pytest.mark.forked


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
    test_pipeline_debug_mode_multi_ep_im_batch_no_remainder(True, True, 7, plot=True)
    test_pipeline_debug_mode_multi_ep_im_batch_with_remainder(True, False, 7, plot=True)
