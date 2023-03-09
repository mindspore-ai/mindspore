# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
Test Map operation's handling of rows with errors
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms as data_trans
import mindspore.dataset.vision as vision
from mindspore import log as logger
from mindspore.dataset.core.config import ErrorSamplesMode
from util import config_get_set_seed, visualize_list

# Need to run all these tests in separate processes since we are modifying a config flag
pytestmark = pytest.mark.forked

# Set global variable
TOTAL_SIZE = 100


def my_generator(ds_size):
    def generator_func():
        for i in range(ds_size):
            yield i

    return generator_func


def raise_none(x):
    return x


def raise_all(x):
    raise ZeroDivisionError


def raise_first(x):
    if x == 0:
        raise ZeroDivisionError
    return x


def raise_first_10(x):
    if x < 10:
        raise ZeroDivisionError
    return x


def raise_first_100(x):
    if x < 100:
        raise ZeroDivisionError
    return x


def raise_first_101(x):
    if x < 101:
        raise ZeroDivisionError
    return x


def raise_first_n(x):
    if x < TOTAL_SIZE // 2 - 2:
        raise ZeroDivisionError
    return x


def raise_first_m(x):
    if x < TOTAL_SIZE // 2 + 2:
        raise ZeroDivisionError
    return x


def raise_all_but_last(x):
    if x < TOTAL_SIZE - 1:
        raise ZeroDivisionError
    return x


def raise_all_but_first(x):
    if x > 0:
        raise ZeroDivisionError
    return x


def raise_last_n(x):
    if x > TOTAL_SIZE // 2 + 2:
        raise ZeroDivisionError
    return x


def raise_last_m(x):
    if x > TOTAL_SIZE // 2 - 2:
        raise ZeroDivisionError
    return x


def raise_all_odds(x):
    if x % 2 != 0:
        raise ZeroDivisionError
    return x


def raise_all_3_remainders(x):
    if x % 3 != 0:
        raise ZeroDivisionError
    return x


def run_replace_test(transforms, dataset_size, num_parallel_workers, python_multiprocessing, expected=None, epochs=1):
    """ Function to run test replace error samples mode based on input configuration. """
    data1 = ds.GeneratorDataset(my_generator(dataset_size), ["data"])
    data1 = data1.map(operations=transforms,
                      num_parallel_workers=num_parallel_workers,
                      python_multiprocessing=python_multiprocessing)

    global TOTAL_SIZE
    TOTAL_SIZE = dataset_size

    itr = data1.create_dict_iterator(num_epochs=epochs, output_numpy=True)
    count = 0
    result = []
    for _ in range(epochs):
        for _, data in enumerate(itr):
            count += 1
            if expected is not None:
                result.append(data["data"].item(0))
    assert count == dataset_size * epochs
    if expected is not None:
        assert result == expected


def run_skip_test(transforms, dataset_size, num_parallel_workers, python_multiprocessing, expected=None, epochs=1):
    """ Function to run test skip error samples mode based on input configuration. """
    data1 = ds.GeneratorDataset(my_generator(dataset_size), ["data"])
    data1 = data1.map(operations=transforms,
                      num_parallel_workers=num_parallel_workers,
                      python_multiprocessing=python_multiprocessing)

    global TOTAL_SIZE
    TOTAL_SIZE = dataset_size

    itr = data1.create_dict_iterator(num_epochs=epochs, output_numpy=True)
    count = 0
    result = []
    for _ in range(epochs):
        for _, data in enumerate(itr):
            count += 1
            if expected is not None:
                result.append(data["data"].item(0))

    if expected is not None:
        assert count == len(expected) * epochs
        assert result == expected


def test_map_replace_errors_failure():
    """
    Feature: Process Error Samples
    Description: Simple replace tests of data pipeline with error rows in Map operation
    Expectation: Exceptions are raise due to numerous error samples
    """
    error_samples_mode_original = ds.config.get_error_samples_mode()
    ds.config.set_error_samples_mode(ds.config.ErrorSamplesMode.REPLACE)

    # failure cases:
    with pytest.raises(RuntimeError) as error_info:
        run_replace_test(raise_all, 1, 1, False)
    assert "All data is garbage" in str(error_info.value)

    with pytest.raises(RuntimeError) as error_info:
        run_replace_test(raise_all, 10, 1, False)
    assert "All data is garbage" in str(error_info.value)

    ds.config.set_error_samples_mode(error_samples_mode_original)


@pytest.mark.parametrize("my_num_workers, my_mp",
                         [(1, False), (4, False), (3, True)])
def test_map_replace_errors_success1(my_num_workers, my_mp):
    """
    Feature: Process Error Samples
    Description: Simple replace tests of data pipeline with various number of error rows in different indexes
    Expectation: Data pipeline replaces error rows successfully
    """
    error_samples_mode_original = ds.config.get_error_samples_mode()
    ds.config.set_error_samples_mode(ds.config.ErrorSamplesMode.REPLACE)
    # Check if python_multiprocessing is to be enabled
    if my_mp:
        # Reduce memory required by disabling the shared memory optimization
        mem_original = ds.config.get_enable_shared_mem()
        ds.config.set_enable_shared_mem(False)

    # no error rows
    run_replace_test(raise_none, 10, my_num_workers, my_mp, list(range(10)))
    run_replace_test(raise_none, 100, my_num_workers, my_mp, list(range(100)))
    run_replace_test(raise_none, 1000, my_num_workers, my_mp, list(range(1000)))

    # 1 error row in the beginning of dataset
    run_replace_test(raise_first, 2, my_num_workers, my_mp, [1, 1])
    run_replace_test(raise_first, 3, my_num_workers, my_mp, [1, 2, 1])
    run_replace_test(raise_first, 10, my_num_workers, my_mp, list(range(1, 10)) + [1])
    run_replace_test(raise_first, 16, my_num_workers, my_mp, list(range(1, 16)) + [1])
    run_replace_test(raise_first, 17, my_num_workers, my_mp, list(range(1, 17)) + [1])
    run_replace_test(raise_first, 20, my_num_workers, my_mp, list(range(1, 17)) + [1] + list(range(17, 20)))
    run_replace_test(raise_first, 100, my_num_workers, my_mp, list(range(1, 17)) + [1] + list(range(17, 100)))

    # multiple error rows in beginning of dataset
    run_replace_test(raise_first_10, 11, my_num_workers, my_mp, [10] * 11)
    run_replace_test(raise_first_10, 12, my_num_workers, my_mp, [10, 11] * 6)
    run_replace_test(raise_first_10, 20, my_num_workers, my_mp, list(range(10, 20)) * 2)
    run_replace_test(raise_first_10, 30, my_num_workers, my_mp,
                     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] +
                     [10, 26, 11, 27, 12, 28, 13, 29, 14, 15, 16, 17, 18, 19])
    run_replace_test(raise_first_100, 1000, my_num_workers, my_mp)
    run_replace_test(raise_first_n, 20, my_num_workers, my_mp,
                     list(range(8, 20)) + list(range(8, 16)))  # ~first half (n < half)
    run_replace_test(raise_first_n, 40, my_num_workers, my_mp,
                     [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33] +
                     [18, 34, 19, 35, 20, 36, 21, 37, 22, 38, 23, 39] +
                     [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])  # ~first half (n < half)
    run_replace_test(raise_first_n, 100, my_num_workers, my_mp)  # ~first half (n < half)
    run_replace_test(raise_first_m, 100, my_num_workers, my_mp)  # ~first half (m > half)
    run_replace_test(raise_all_but_last, 2, my_num_workers, my_mp, [1, 1])
    run_replace_test(raise_all_but_last, 3, my_num_workers, my_mp, [2, 2, 2])
    run_replace_test(raise_all_but_last, 4, my_num_workers, my_mp, [3] * 4)
    run_replace_test(raise_all_but_last, 16, my_num_workers, my_mp, [15] * 16)
    run_replace_test(raise_all_but_last, 100, my_num_workers, my_mp, [99] * 100)
    run_replace_test(raise_all_but_first, 10, my_num_workers, my_mp, [0] * 10)
    run_replace_test(raise_all_but_first, 100, my_num_workers, my_mp, [0] * 100)

    # error rows in the end of dataset
    run_replace_test(raise_last_n, 10, my_num_workers, my_mp, list(range(0, 8)) + [0, 1])
    run_replace_test(raise_last_n, 20, my_num_workers, my_mp, list(range(0, 13)) + list(range(0, 7)))
    run_replace_test(raise_last_n, 40, my_num_workers, my_mp, list(range(0, 23)) + list(range(0, 16)) + [0])
    run_replace_test(raise_last_n, 100, my_num_workers, my_mp)
    run_replace_test(raise_last_m, 100, my_num_workers, my_mp)

    # error rows in different places
    run_replace_test(raise_all_odds, 10, my_num_workers, my_mp, [0, 2, 4, 6, 8] * 2)
    run_replace_test(raise_all_odds, 40, my_num_workers, my_mp,
                     [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30] +
                     [0, 32, 2, 34, 4, 36, 6, 38, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38])
    run_replace_test(raise_all_odds, 100, my_num_workers, my_mp,
                     [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 32, 2, 34, 4, 36, 6, 38] +
                     [8, 40, 10, 42, 12, 44, 14, 46, 16, 48, 18, 50, 20, 52, 22, 54, 24, 56, 26, 58, 28, 60] +
                     [30, 62, 32, 64, 34, 66, 36, 68, 38, 70, 40, 72, 42, 74, 44, 76, 46, 78, 48, 80, 50, 82] +
                     [52, 84, 54, 86, 56, 88, 58, 90, 60, 92, 62, 94, 64, 96, 66, 98, 68, 70, 72, 74, 76, 78] +
                     [80, 82, 84, 86, 88, 90, 92, 94, 96, 98])
    run_replace_test(raise_all_3_remainders, 12, my_num_workers, my_mp, [0, 3, 6, 9] * 3)
    run_replace_test(raise_all_3_remainders, 100, my_num_workers, my_mp)

    ds.config.set_error_samples_mode(error_samples_mode_original)
    if my_mp:
        # Restore configuration for shared memory
        ds.config.set_enable_shared_mem(mem_original)


@pytest.mark.parametrize("my_mp", (False, True))
def test_map_replace_errors_success2(my_mp):
    """
    Feature: Process Error Samples
    Description: Simple replace tests of data pipeline with error rows in different settings
    Expectation: Data pipeline replaces error rows successfully
    """
    error_samples_mode_original = ds.config.get_error_samples_mode()
    ds.config.set_error_samples_mode(ds.config.ErrorSamplesMode.REPLACE)
    # Check if python_multiprocessing is to be enabled
    if my_mp:
        # Reduce memory required by disabling the shared memory optimization
        mem_original = ds.config.get_enable_shared_mem()
        ds.config.set_enable_shared_mem(False)

    # multiple pyfuncs
    run_replace_test([raise_last_n, raise_last_n, raise_first_m, raise_first_n], 100, 1, my_mp)
    run_replace_test([raise_last_n, raise_last_n, raise_first_n], 100, 2, my_mp)  # n<50

    # multiple epochs
    run_replace_test(raise_last_m, 100, 1, my_mp, epochs=3)
    run_replace_test(raise_all_odds, 100, 3, my_mp, epochs=3)

    ds.config.set_error_samples_mode(error_samples_mode_original)
    if my_mp:
        # Restore configuration for shared memory
        ds.config.set_enable_shared_mem(mem_original)


@pytest.mark.parametrize("my_num_workers, my_mp",
                         [(1, False), (4, False), (3, True)])
def test_map_replace_errors_success3(my_num_workers, my_mp):
    """
    Feature: Process Error Samples
    Description: Simple replace tests of data pipeline with error rows in different pipelines
    Expectation: Data pipeline replaces error rows successfully
    """
    error_samples_mode_original = ds.config.get_error_samples_mode()
    ds.config.set_error_samples_mode(ds.config.ErrorSamplesMode.REPLACE)
    # Check if python_multiprocessing is to be enabled
    if my_mp:
        # Reduce memory required by disabling the shared memory optimization
        mem_original = ds.config.get_enable_shared_mem()
        ds.config.set_enable_shared_mem(False)

    dataset_size = 100
    global TOTAL_SIZE
    TOTAL_SIZE = dataset_size
    my_batch_size = 5

    # multiple maps
    data1 = ds.GeneratorDataset(my_generator(dataset_size), ["data"])
    data1 = data1.map(operations=[raise_all_but_last],
                      num_parallel_workers=my_num_workers, python_multiprocessing=my_mp)
    data1 = data1.map(operations=[raise_all_3_remainders],
                      num_parallel_workers=my_num_workers, python_multiprocessing=my_mp)

    # apply shuffle and batch
    data1 = data1.shuffle(buffer_size=50)
    data1 = data1.batch(batch_size=my_batch_size, drop_remainder=False)

    count = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == dataset_size / my_batch_size

    # repeat op
    transforms = [raise_all_but_first]
    data1 = ds.GeneratorDataset(my_generator(dataset_size), ["data"])
    data1 = data1.map(operations=transforms, num_parallel_workers=my_num_workers, python_multiprocessing=my_mp)
    data1 = data1.repeat(3)

    # apply shuffle and batch
    data1 = data1.shuffle(buffer_size=50)
    data1 = data1.batch(batch_size=my_batch_size, drop_remainder=False)

    count = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == (dataset_size * 3) / my_batch_size

    ds.config.set_error_samples_mode(error_samples_mode_original)
    if my_mp:
        # Restore configuration for shared memory
        ds.config.set_enable_shared_mem(mem_original)


@pytest.mark.parametrize("my_num_workers, my_mp",
                         [(1, False), (3, False), (2, True)])
def test_map_skip_errors_success1(my_num_workers, my_mp):
    """
    Feature: Process Error Samples
    Description: Simple skip tests of data pipeline with various number of error rows in different indexes
    Expectation: Data pipeline replaces error rows successfully
    """
    error_samples_mode_original = ds.config.get_error_samples_mode()
    ds.config.set_error_samples_mode(ds.config.ErrorSamplesMode.SKIP)
    # Check if python_multiprocessing is to be enabled
    if my_mp:
        # Reduce memory required by disabling the shared memory optimization
        mem_original = ds.config.get_enable_shared_mem()
        ds.config.set_enable_shared_mem(False)

    # no error rows
    run_skip_test(raise_none, 10, my_num_workers, my_mp, list(range(10)))
    run_skip_test(raise_none, 100, my_num_workers, my_mp, list(range(100)))
    run_skip_test(raise_none, 1000, my_num_workers, my_mp, list(range(1000)))

    # 1 error row in the beginning of dataset
    run_skip_test(raise_first, 2, my_num_workers, my_mp, [1])
    run_skip_test(raise_first, 3, my_num_workers, my_mp, [1, 2])
    run_skip_test(raise_first, 10, my_num_workers, my_mp, list(range(1, 10)))
    run_skip_test(raise_first, 16, my_num_workers, my_mp, list(range(1, 16)))
    run_skip_test(raise_first, 17, my_num_workers, my_mp, list(range(1, 17)))
    run_skip_test(raise_first, 20, my_num_workers, my_mp, list(range(1, 20)))
    run_skip_test(raise_first, 100, my_num_workers, my_mp, list(range(1, 100)))

    # multiple error rows in beginning of dataset
    run_skip_test(raise_first_10, 11, my_num_workers, my_mp, [10])
    run_skip_test(raise_first_10, 12, my_num_workers, my_mp, [10, 11])
    run_skip_test(raise_first_10, 20, my_num_workers, my_mp, list(range(10, 20)))
    run_skip_test(raise_first_10, 30, my_num_workers, my_mp, list(range(10, 30)))
    run_skip_test(raise_first_100, 250, my_num_workers, my_mp, list(range(100, 250)))
    run_skip_test(raise_first_100, 1000, my_num_workers, my_mp, list(range(100, 1000)))
    run_skip_test(raise_first_n, 20, my_num_workers, my_mp, list(range(8, 20)))  # ~first half (n < half)
    run_skip_test(raise_first_n, 40, my_num_workers, my_mp, list(range(18, 40)))  # ~first half (n < half)
    run_skip_test(raise_first_n, 100, my_num_workers, my_mp, list(range(48, 100)))  # ~first half (n < half)
    run_skip_test(raise_first_m, 100, my_num_workers, my_mp, list(range(52, 100)))  # ~first half (m > half)
    run_skip_test(raise_all_but_last, 2, my_num_workers, my_mp, [1])
    run_skip_test(raise_all_but_last, 3, my_num_workers, my_mp, [2])
    run_skip_test(raise_all_but_last, 4, my_num_workers, my_mp, [3])
    run_skip_test(raise_all_but_last, 16, my_num_workers, my_mp, [15])
    run_skip_test(raise_all_but_last, 100, my_num_workers, my_mp, [99])
    run_skip_test(raise_all_but_first, 10, my_num_workers, my_mp, [0])
    run_skip_test(raise_all_but_first, 100, my_num_workers, my_mp, [0])

    # error rows in the end of dataset
    run_skip_test(raise_last_n, 10, my_num_workers, my_mp, list(range(0, 8)))
    run_skip_test(raise_last_n, 20, my_num_workers, my_mp, list(range(0, 13)))
    run_skip_test(raise_last_n, 40, my_num_workers, my_mp, list(range(0, 23)))
    run_skip_test(raise_last_n, 100, my_num_workers, my_mp, list(range(0, 53)))
    run_skip_test(raise_last_m, 100, my_num_workers, my_mp, list(range(0, 49)))

    # error rows in different places
    run_skip_test(raise_all_odds, 10, my_num_workers, my_mp, [0, 2, 4, 6, 8])
    run_skip_test(raise_all_odds, 40, my_num_workers, my_mp,
                  [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38])
    run_skip_test(raise_all_odds, 100, my_num_workers, my_mp)
    run_skip_test(raise_all_3_remainders, 12, my_num_workers, my_mp, [0, 3, 6, 9])
    run_skip_test(raise_all_3_remainders, 100, my_num_workers, my_mp)

    # error rows in entire dataset
    run_skip_test(raise_all, 1, my_num_workers, my_mp, [])
    run_skip_test(raise_all, 3, my_num_workers, my_mp, [])
    run_skip_test(raise_all, 10, my_num_workers, my_mp, [])
    run_skip_test(raise_all, 100, my_num_workers, my_mp, [])

    ds.config.set_error_samples_mode(error_samples_mode_original)
    if my_mp:
        # Restore configuration for shared memory
        ds.config.set_enable_shared_mem(mem_original)


@pytest.mark.parametrize("my_error_samples_mode, my_num_workers",
                         [(ErrorSamplesMode.REPLACE, 2), (ErrorSamplesMode.SKIP, 1)])
def test_map_error_samples_imagefolder1_cop(my_error_samples_mode, my_num_workers, plot=False):
    """
    Feature: Process Error Samples
    Description: Invoke set_error_samples_mode and test ImageFolderDataset pipeline with map op
        of C++ implemented ops and sample errors.
    Expectation: The dataset is processed as expected.
    """

    def test_config(my_error_samples_mode, my_error_sample_data_file, my_num_classes, my_total_samples,
                    my_unskipped_samples, plot):
        # For ImageFolderDataset:
        # - use num_samples=None to read all samples
        # - use num_parallel_workers=1
        # - use shuffle=False which will result in sequential order of samples
        # - use decode default of False
        data3 = ds.ImageFolderDataset(my_error_sample_data_file, num_samples=None, num_parallel_workers=1,
                                      shuffle=False)
        # Use multiple map ops in pipeline.
        data3 = data3.map(operations=[data_trans.OneHot(my_num_classes)],
                          input_columns=["label"],
                          num_parallel_workers=my_num_workers)
        # 2nd map op in pipeline is Decode Op, which uses C++ implementation
        data3 = data3.map(operations=[vision.Decode()], input_columns=["image"],
                          num_parallel_workers=my_num_workers)
        data3 = data3.map(operations=[vision.ResizedCrop(50, 80, 300, 400, (100, 120))],
                          input_columns=["image"],
                          num_parallel_workers=my_num_workers)

        images_list = []
        if my_error_samples_mode == ErrorSamplesMode.REPLACE:
            # Error samples are to be replaced
            count = 0
            for item in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
                count += 1
                image_c = item["image"]
                images_list.append(image_c)
            assert count == my_total_samples
        elif my_error_samples_mode == ErrorSamplesMode.SKIP:
            # Error samples are to be skipped
            count = 0
            for item in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
                count += 1
                image_c = item["image"]
                images_list.append(image_c)
            assert count == my_unskipped_samples
        else:
            with pytest.raises(RuntimeError) as error_info:
                for _ in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
                    pass
            assert "map operation: [Decode] failed" in str(error_info.value)
        if plot:
            visualize_list(images_list, None, visualize_mode=1)

    # Set configuration for error_samples_mode
    error_samples_mode_original = ds.config.get_error_samples_mode()
    ds.config.set_error_samples_mode(my_error_samples_mode)

    # Test empty sample (which is first error sample when samples are read sequentially)
    test_config(my_error_samples_mode, "../data/dataset/testImageNetError/Sample1_empty/train", 1, 3, 2, plot)
    # Test corrupt sample (which is a middle error sample when samples are read sequentially)
    test_config(my_error_samples_mode, "../data/dataset/testImageNetError/Sample2_corrupt1mid/train", 3, 6, 5, plot)
    # Test text sample, instead of image sample (which is a final error sample when samples are read sequentially)
    test_config(my_error_samples_mode, "../data/dataset/testImageNetError/Sample3_text/train", 1, 3, 2, plot)

    # Restore configuration for error_samples_mode
    ds.config.set_error_samples_mode(error_samples_mode_original)


@pytest.mark.parametrize("my_error_samples_mode, my_num_workers, my_mp",
                         [(ErrorSamplesMode.REPLACE, 3, False), (ErrorSamplesMode.REPLACE, 2, True),
                          (ErrorSamplesMode.SKIP, 2, False), (ErrorSamplesMode.SKIP, 4, True)])
def test_map_error_samples_imagefolder1_pyop(my_error_samples_mode, my_num_workers, my_mp, plot=False):
    """
    Feature: Process Error Samples
    Description: Invoke set_error_samples_mode and test ImageFolderDataset pipeline with map op
        of Python implemented ops and sample errors.
    Expectation: The dataset is processed as expected.
    """

    def test_config(my_error_samples_mode, my_error_sample_data_file, my_num_classes, my_total_samples,
                    my_unskipped_samples, plot):
        # For ImageFolderDataset:
        # - use num_samples=None to read all samples
        # - use num_parallel_workers=1
        # - use shuffle=False which will result in sequential order of samples
        # - use decode default of False
        data4 = ds.ImageFolderDataset(my_error_sample_data_file, num_samples=None, num_parallel_workers=1,
                                      shuffle=False)
        # Use multiple map ops in pipeline.
        data4 = data4.map(operations=[data_trans.OneHot(my_num_classes)],
                          input_columns=["label"],
                          num_parallel_workers=my_num_workers, python_multiprocessing=my_mp)
        # 2nd map op in pipeline includes Decode Op, which uses Python implementation
        data4 = data4.map(operations=[vision.Decode(to_pil=True),
                                      vision.Resize((120, 150)),
                                      vision.RandomCrop((10, 10), (100, 120))],
                          input_columns=["image"],
                          num_parallel_workers=my_num_workers, python_multiprocessing=my_mp)
        # Note: ToPIL op added so that Python implementation of RandomVerticalFlip is selected
        data4 = data4.map(operations=[vision.ToPIL(),
                                      vision.RandomVerticalFlip(1.0)],
                          input_columns=["image"],
                          num_parallel_workers=my_num_workers, python_multiprocessing=my_mp)
        data4 = data4.map(operations=[vision.ToTensor()],
                          input_columns=["image"],
                          num_parallel_workers=my_num_workers)

        images_list = []
        if my_error_samples_mode == ErrorSamplesMode.REPLACE:
            # Error samples are to be replaced
            count = 0
            for item in data4.create_dict_iterator(num_epochs=1, output_numpy=True):
                count += 1
                image_py = (item["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
                images_list.append(image_py)
            assert count == my_total_samples
        elif my_error_samples_mode == ErrorSamplesMode.SKIP:
            # Error samples are to be skipped
            count = 0
            for item in data4.create_dict_iterator(num_epochs=1, output_numpy=True):
                count += 1
                image_py = (item["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
                images_list.append(image_py)
            assert count == my_unskipped_samples
        else:
            with pytest.raises(RuntimeError) as error_info:
                for _ in data4.create_dict_iterator(num_epochs=1, output_numpy=True):
                    pass
            assert "map operation: [PyFunc] failed" in str(error_info.value)
        if plot:
            visualize_list(images_list, None, visualize_mode=1)

    # Set configuration for error_samples_mode
    error_samples_mode_original = ds.config.get_error_samples_mode()
    ds.config.set_error_samples_mode(my_error_samples_mode)

    # Check if python_multiprocessing is to be enabled
    if my_mp:
        # Reduce memory required by disabling the shared memory optimization
        mem_original = ds.config.get_enable_shared_mem()
        ds.config.set_enable_shared_mem(False)

    # Test empty sample (which is first error sample when samples are read sequentially)
    test_config(my_error_samples_mode, "../data/dataset/testImageNetError/Sample1_empty/train", 1, 3, 2, plot)
    # Test corrupt sample (which is a middle error sample when samples are read sequentially)
    test_config(my_error_samples_mode, "../data/dataset/testImageNetError/Sample2_corrupt1mid/train", 3, 6, 5, plot)
    # Test text sample, instead of image sample (which is a final error sample when samples are read sequentially)
    test_config(my_error_samples_mode, "../data/dataset/testImageNetError/Sample3_text/train", 1, 3, 2, plot)

    # Restore configuration for error_samples_mode
    ds.config.set_error_samples_mode(error_samples_mode_original)

    if my_mp:
        # Restore configuration for shared memory
        ds.config.set_enable_shared_mem(mem_original)


@pytest.mark.parametrize("my_error_samples_mode, my_num_workers, my_mp",
                         [(ErrorSamplesMode.REPLACE, 2, False), (ErrorSamplesMode.REPLACE, 3, True),
                          (ErrorSamplesMode.SKIP, 3, False), (ErrorSamplesMode.SKIP, 2, True)])
def test_map_error_samples_imagefolder2(my_error_samples_mode, my_num_workers, my_mp):
    """
    Feature: Process Error Samples
    Description: Invoke set_error_samples_mode and test ImageFolderDataset pipeline with multiple map ops
        plus sample errors.
    Expectation: The dataset is processed as expected.
    """

    def test_config(my_error_samples_mode, my_seed, my_error_sample_data_file, my_total_samples, my_unskipped_samples):
        # Set configuration since default Random Sampler is used
        original_seed = config_get_set_seed(my_seed)

        # Create dataset pipeline which includes at least one sample error
        my_sampler = ds.RandomSampler(replacement=False, num_samples=None)
        data1 = ds.ImageFolderDataset(my_error_sample_data_file, sampler=my_sampler,
                                      num_parallel_workers=my_num_workers)
        data1 = data1.map(operations=[(lambda z: z)],
                          input_columns=["image"],
                          num_parallel_workers=my_num_workers,
                          python_multiprocessing=my_mp)
        # Add map op to the pipeline which will encounter error samples. Use Python implemented ops
        # Note: Decode is not the first op in the list of operations
        data1 = data1.map(operations=[(lambda x: x),
                                      vision.Decode(to_pil=True),
                                      (lambda y: y),
                                      vision.RandomVerticalFlip(0.8)],
                          input_columns=["image"],
                          num_parallel_workers=my_num_workers,
                          python_multiprocessing=my_mp)
        data1 = data1.map(operations=[vision.RandomHorizontalFlip(1.0)],
                          input_columns=["image"],
                          num_parallel_workers=my_num_workers, python_multiprocessing=my_mp)

        if my_error_samples_mode == ErrorSamplesMode.REPLACE:
            # Error samples are to be replaced
            count = 0
            for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
                count += 1
            assert count == my_total_samples
        elif my_error_samples_mode == ErrorSamplesMode.SKIP:
            # Error samples are to be skipped
            count = 0
            for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
                count += 1
            assert count == my_unskipped_samples
        else:
            with pytest.raises(RuntimeError) as error_info:
                for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
                    pass
            assert "map operation: [PyFunc] failed" in str(error_info.value)

        # Restore configuration
        ds.config.set_seed(original_seed)

    # Set configuration for error_samples_mode
    error_samples_mode_original = ds.config.get_error_samples_mode()
    ds.config.set_error_samples_mode(my_error_samples_mode)

    # Check if python_multiprocessing is to be enabled
    if my_mp:
        # Reduce memory required by disabling the shared memory optimization
        mem_original = ds.config.get_enable_shared_mem()
        ds.config.set_enable_shared_mem(False)

    # Test empty sample
    test_config(my_error_samples_mode, 2, "../data/dataset/testImageNetError/Sample1_empty/train", 3, 2)
    # Test corrupt sample
    test_config(my_error_samples_mode, 3, "../data/dataset/testImageNetError/Sample2_corrupt1mid/train", 6, 5)
    # Test text sample, instead of image sample
    test_config(my_error_samples_mode, 1, "../data/dataset/testImageNetError/Sample3_text/train", 3, 2)

    # Restore configuration for error_samples_mode
    ds.config.set_error_samples_mode(error_samples_mode_original)

    if my_mp:
        # Restore configuration for shared memory
        ds.config.set_enable_shared_mem(mem_original)


@pytest.mark.parametrize("my_error_samples_mode, my_num_workers, my_mp",
                         [(ErrorSamplesMode.REPLACE, 3, False), (ErrorSamplesMode.REPLACE, 4, True),
                          (ErrorSamplesMode.SKIP, 5, False), (ErrorSamplesMode.SKIP, 3, True)])
def test_map_error_samples_imagefolder3(my_error_samples_mode, my_num_workers, my_mp):
    """
    Feature: Process Error Samples
    Description: Invoke set_error_samples_mode and test ImageFolderDataset pipeline with map op
        plus multiple sample errors.
    Expectation: The dataset is processed as expected.
    """

    def test_config(my_error_samples_mode, my_seed, my_take, my_repeat, my_total_unskipped_rows):
        # Set configuration since default Random Sampler is used
        original_seed = config_get_set_seed(my_seed)

        # Create dataset pipelines with multiple error samples
        # Dataset TotalRows TotalBadSamples
        # data4       3           3
        # data1       3           1
        # data2       6           1
        # data3       3           1
        #            ==          ==
        # datafinal  15           6
        data1 = ds.ImageFolderDataset("../data/dataset/testImageNetError/Sample1_empty/train",
                                      num_samples=None, shuffle=True, num_parallel_workers=my_num_workers)
        data2 = ds.ImageFolderDataset("../data/dataset/testImageNetError/Sample2_corrupt1mid/train",
                                      num_samples=None, shuffle=True, num_parallel_workers=my_num_workers)
        data3 = ds.ImageFolderDataset("../data/dataset/testImageNetError/Sample3_text/train",
                                      num_samples=None, shuffle=True, num_parallel_workers=my_num_workers)
        data4 = ds.ImageFolderDataset("../data/dataset/testImageNetError/Sample4_corruptall/train",
                                      num_samples=None, shuffle=True, num_parallel_workers=my_num_workers)
        # Concat the multiple datasets together
        datafinal = data4 + data1 + data2 + data3
        datafinal = datafinal.take(my_take).repeat(my_repeat)
        total_rows = my_take * my_repeat
        # Add map ops to the pipeline. Use Python implemented ops
        datafinal = datafinal.map(operations=[vision.Decode(to_pil=True),
                                              vision.RandomHorizontalFlip(0.7)],
                                  input_columns=["image"],
                                  num_parallel_workers=my_num_workers,
                                  python_multiprocessing=my_mp)
        datafinal = datafinal.map(operations=[vision.ToPIL(),
                                              vision.RandomVerticalFlip(0.8)],
                                  input_columns=["image"],
                                  num_parallel_workers=my_num_workers,
                                  python_multiprocessing=my_mp)
        # Apply dataset ops
        datafinal = datafinal.shuffle(buffer_size=100)

        if my_error_samples_mode == ErrorSamplesMode.REPLACE:
            # Error samples are to be replaced
            count = 0
            for _ in datafinal.create_dict_iterator(num_epochs=1, output_numpy=True):
                count += 1
            assert count == total_rows
        elif my_error_samples_mode == ErrorSamplesMode.SKIP:
            # Error samples are to be skipped
            count = 0
            for _ in datafinal.create_dict_iterator(num_epochs=1, output_numpy=True):
                count += 1
            logger.info("Number of data in datafinal: {}".format(count))
            assert count == my_total_unskipped_rows
        else:
            with pytest.raises(RuntimeError) as error_info:
                for _ in datafinal.create_dict_iterator(num_epochs=1, output_numpy=True):
                    pass
            assert "map operation: [PyFunc] failed" in str(error_info.value)

        # Restore configuration
        ds.config.set_seed(original_seed)

    # Set configuration for error_samples_mode
    error_samples_mode_original = ds.config.get_error_samples_mode()
    ds.config.set_error_samples_mode(my_error_samples_mode)

    # Check if python_multiprocessing is to be enabled
    if my_mp:
        # Reduce memory required by disabling the shared memory optimization
        mem_original = ds.config.get_enable_shared_mem()
        ds.config.set_enable_shared_mem(False)

    # Test different scenarios
    test_config(my_error_samples_mode, 5001, 15, 1, 9)
    test_config(my_error_samples_mode, 5002, 15, 2, 18)
    test_config(my_error_samples_mode, 5003, 10, 5, 27)
    test_config(my_error_samples_mode, 5004, 12, 4, 28)

    # Restore configuration for error_samples_mode
    ds.config.set_error_samples_mode(error_samples_mode_original)

    if my_mp:
        # Restore configuration for shared memory
        ds.config.set_enable_shared_mem(mem_original)


if __name__ == '__main__':
    test_map_replace_errors_failure()
    test_map_replace_errors_success1(2, True)
    test_map_replace_errors_success2(True)
    test_map_replace_errors_success3(3, False)
    test_map_skip_errors_success1(3, True)
    test_map_error_samples_imagefolder1_cop(ErrorSamplesMode.RETURN, 4, plot=True)
    test_map_error_samples_imagefolder1_pyop(ErrorSamplesMode.REPLACE, 3, True, plot=True)
    test_map_error_samples_imagefolder2(ErrorSamplesMode.REPLACE, 4, True)
    test_map_error_samples_imagefolder3(ErrorSamplesMode.SKIP, 3, True)
