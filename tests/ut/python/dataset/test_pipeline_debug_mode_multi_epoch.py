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
import pytest
import mindspore.dataset as ds
from mindspore import log as logger
from util import config_get_set_seed

pytestmark = pytest.mark.forked


@pytest.mark.parametrize("my_debug_mode", (False, True))
def test_pipeline_debug_mode_multi_epoch_celaba_take(my_debug_mode):
    """
    Feature: Pipeline debug mode.
    Description: Test creating tuple iterator in CelebA dataset with take op and multi epochs.
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
    # Use create_tuple_iterator
    iter1 = data1.create_tuple_iterator(num_epochs=num_epoch)
    epoch_count = 0
    sample_count = 0
    for _ in range(num_epoch):
        row_count = 0
        for _ in iter1:
            # in this example, each row has columns "image" and "label"
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
        for _ in iter1:
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
    num_samples = 100
    num_repeat = 2
    batch_size = 32

    data1 = ds.Cifar10Dataset(data_dir_10, num_samples=num_samples)
    # Add repeat then batch
    data1 = data1.repeat(num_repeat)
    data1 = data1.batch(batch_size, True)

    num_epoch = 5
    # Use create_tuple_iterator
    iter1 = data1.create_tuple_iterator(num_epochs=num_epoch)
    epoch_count = 0
    sample_count = 0
    for _ in range(num_epoch):
        row_count = 0
        for _ in iter1:
            row_count += 1
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
        for _ in iter1:
            row_count += 1
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
        for _ in iter1:
            row_count += 1
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
    num_repeat = 5

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
    for _ in range(num_epoch):
        row_count = 0
        for _ in iter1:
            row_count += 1
        assert row_count == int(num_samples * num_repeat / batch_size)
        epoch_count += 1
        sample_count += row_count
    assert epoch_count == num_epoch
    assert sample_count == int(num_samples * num_repeat / batch_size) * num_epoch

    # Restore configuration
    ds.config.set_seed(original_seed)
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


@pytest.mark.parametrize("my_debug_mode, my_drop_remainder, my_num_samples",
                         [(False, False, 40), (True, False, 40), (True, True, 43)])
def test_pipeline_debug_mode_multi_epoch_imagefolder_batch(my_debug_mode, my_drop_remainder, my_num_samples):
    """
    Feature: Pipeline debug mode.
    Description: Test multiple epoch scenario using ImageFolderDataset with batch op.
    Expectation: Output is equal to the expected output
    """
    logger.info("test_pipeline_debug_mode_multi_epoch_imagefolder_batch")

    # Set configuration
    original_seed = config_get_set_seed(799)
    if my_debug_mode:
        debug_mode_original = ds.config.get_debug_mode()
        ds.config.set_debug_mode(True)

    num_samples = my_num_samples
    batch_size = 5

    data1 = ds.ImageFolderDataset("../data/dataset/testPK/data", num_samples=num_samples)
    data1 = data1.batch(batch_size, drop_remainder=my_drop_remainder)

    num_epoch = 3
    iter1 = data1.create_dict_iterator(num_epochs=num_epoch, output_numpy=True)
    epoch_count = 0
    sample_count = 0
    for _ in range(num_epoch):
        row_count = 0
        for _ in iter1:
            row_count += 1
        assert row_count == int(num_samples / batch_size)
        epoch_count += 1
        sample_count += row_count
    assert epoch_count == num_epoch
    assert sample_count == int(num_samples / batch_size) * num_epoch

    # Restore configuration
    ds.config.set_seed(original_seed)
    if my_debug_mode:
        ds.config.set_debug_mode(debug_mode_original)


if __name__ == '__main__':
    test_pipeline_debug_mode_multi_epoch_celaba_take(True)
    test_pipeline_debug_mode_multi_epoch_cifar10_take(True)
    test_pipeline_debug_mode_multi_epoch_cifar10_repeat_batch(True)
    test_pipeline_debug_mode_multi_epoch_cifar10_batch_repeat(True)
    test_pipeline_debug_mode_multi_epoch_cifar10_zip(True)
    test_pipeline_debug_mode_multi_epoch_cifar10_zip_batch_repeat(True)
    test_pipeline_debug_mode_multi_epoch_imagefolder_batch(True, False, 10)
