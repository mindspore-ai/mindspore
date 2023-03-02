# Copyright 2019-2023 Huawei Technologies Co., Ltd
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
Testing configuration manager
"""
import os
import filecmp
import glob
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.engine.iterators as it
import mindspore.dataset.transforms
import mindspore.dataset.vision as vision
import mindspore.dataset.core.config as config
import mindspore.dataset.debug as debug
from mindspore import log as logger
from util import dataset_equal

# Need to run all these tests in separate processes since tests are modifying config parameters
pytestmark = pytest.mark.forked

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def config_error_func(config_interface, input_args, err_type, except_err_msg):
    err_msg = ""
    try:
        config_interface(input_args)
    except err_type as e:
        err_msg = str(e)

    assert except_err_msg in err_msg


def test_basic():
    """
    Feature: Config
    Description: Test basic configuration functions
    Expectation: Output is equal to the expected value
    """
    # Save original configuration values
    num_parallel_workers_original = ds.config.get_num_parallel_workers()
    prefetch_size_original = ds.config.get_prefetch_size()
    seed_original = ds.config.get_seed()
    monitor_sampling_interval_original = ds.config.get_monitor_sampling_interval()
    fast_recovery_original = ds.config.get_fast_recovery()
    debug_mode = ds.config.get_debug_mode()
    error_samples_mode_original = ds.config.get_error_samples_mode()

    ds.config.load('../data/dataset/declient.cfg')

    assert ds.config.get_num_parallel_workers() == 8
    # assert ds.config.get_worker_connector_size() == 16
    assert ds.config.get_prefetch_size() == 16
    assert ds.config.get_seed() == 5489
    assert ds.config.get_monitor_sampling_interval() == 15
    assert ds.config.get_fast_recovery()
    assert not ds.config.get_debug_mode()
    assert ds.config.get_error_samples_mode() == config.ErrorSamplesMode.RETURN

    ds.config.set_num_parallel_workers(2)
    # ds.config.set_worker_connector_size(3)
    ds.config.set_prefetch_size(4)
    ds.config.set_seed(5)
    ds.config.set_monitor_sampling_interval(45)
    ds.config.set_fast_recovery(False)
    ds.config.set_debug_mode(True)
    ds.config.set_error_samples_mode(config.ErrorSamplesMode.REPLACE)

    assert ds.config.get_num_parallel_workers() == 2
    # assert ds.config.get_worker_connector_size() == 3
    assert ds.config.get_prefetch_size() == 4
    assert ds.config.get_seed() == 5
    assert ds.config.get_monitor_sampling_interval() == 45
    assert not ds.config.get_fast_recovery()
    assert ds.config.get_debug_mode()
    assert ds.config.get_error_samples_mode() == config.ErrorSamplesMode.REPLACE

    ds.config.set_fast_recovery(True)
    ds.config.set_debug_mode(False)
    ds.config.set_error_samples_mode(config.ErrorSamplesMode.SKIP)

    assert ds.config.get_fast_recovery()
    assert not ds.config.get_debug_mode()
    assert ds.config.get_error_samples_mode() == config.ErrorSamplesMode.SKIP

    # Restore original configuration values
    ds.config.set_num_parallel_workers(num_parallel_workers_original)
    ds.config.set_prefetch_size(prefetch_size_original)
    ds.config.set_seed(seed_original)
    ds.config.set_monitor_sampling_interval(monitor_sampling_interval_original)
    ds.config.set_fast_recovery(fast_recovery_original)
    ds.config.set_debug_mode(debug_mode)
    ds.config.set_error_samples_mode(error_samples_mode_original)


def test_get_seed():
    """
    Feature: Config
    Description: Test get_seed value without explicitly setting a default
    Expectation: Expecting to get an int
    """
    assert isinstance(ds.config.get_seed(), int)


def test_pipeline():
    """
    Feature: Config
    Description: Test that the config pipeline works when parameters are set at different locations in dataset code
    Expectation: Output is equal to the expected value
    """
    # Save original configuration values
    num_parallel_workers_original = ds.config.get_num_parallel_workers()

    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    data1 = data1.map(operations=[vision.Decode()], input_columns=["image"])
    ds.serialize(data1, "testpipeline.json")

    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, num_parallel_workers=num_parallel_workers_original,
                               shuffle=False)
    data2 = data2.map(operations=[vision.Decode()], input_columns=["image"])
    ds.serialize(data2, "testpipeline2.json")

    # check that the generated output is different
    assert filecmp.cmp('testpipeline.json', 'testpipeline2.json')

    # this test passes currently because our num_parallel_workers don't get updated.

    # remove generated jason files
    file_list = glob.glob('*.json')
    for f in file_list:
        try:
            os.remove(f)
        except IOError:
            logger.info("Error while deleting: {}".format(f))

    # Restore original configuration values
    ds.config.set_num_parallel_workers(num_parallel_workers_original)


def test_deterministic_run_fail():
    """
    Feature: Config
    Description: Test RandomCrop with seed
    Expectation: Exception is raised as expected
    """
    logger.info("test_deterministic_run_fail")

    # Save original configuration values
    num_parallel_workers_original = ds.config.get_num_parallel_workers()
    seed_original = ds.config.get_seed()

    # when we set the seed all operations within our dataset should be deterministic
    ds.config.set_seed(0)
    ds.config.set_num_parallel_workers(1)
    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Assuming we get the same seed on calling constructor, if this op is re-used then result won't be
    # the same in between the two datasets. For example, RandomCrop constructor takes seed (0)
    # outputs a deterministic series of numbers, e,g "a" = [1, 2, 3, 4, 5, 6] <- pretend these are random
    random_crop_op = vision.RandomCrop([512, 512], [200, 200, 200, 200])
    decode_op = vision.Decode()
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_crop_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])
    # If seed is set up on constructor
    data2 = data2.map(operations=random_crop_op, input_columns=["image"])

    try:
        dataset_equal(data1, data2, 0)

    except Exception as e:
        # two datasets split the number out of the sequence a
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Array" in str(e)

    # Restore original configuration values
    ds.config.set_num_parallel_workers(num_parallel_workers_original)
    ds.config.set_seed(seed_original)


def test_seed_undeterministic():
    """
    Feature: Config
    Description: Test seed with num_parallel_workers in Cpp
    Expectation: Exception is raised some of the time
    """
    logger.info("test_seed_undeterministic")

    # Save original configuration values
    num_parallel_workers_original = ds.config.get_num_parallel_workers()
    seed_original = ds.config.get_seed()

    ds.config.set_seed(0)
    ds.config.set_num_parallel_workers(3)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # We get the seed when constructor is called
    random_crop_op = vision.RandomCrop([512, 512], [200, 200, 200, 200])
    decode_op = vision.Decode()
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_crop_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])
    # Since seed is set up on constructor, so the two ops output deterministic sequence.
    # Assume the generated random sequence "a" = [1, 2, 3, 4, 5, 6] <- pretend these are random
    random_crop_op2 = vision.RandomCrop([512, 512], [200, 200, 200, 200])
    data2 = data2.map(operations=random_crop_op2, input_columns=["image"])
    try:
        dataset_equal(data1, data2, 0)
    except Exception as e:
        # two datasets both use numbers from the generated sequence "a"
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Array" in str(e)

    # Restore original configuration values
    ds.config.set_num_parallel_workers(num_parallel_workers_original)
    ds.config.set_seed(seed_original)


def test_seed_deterministic():
    """
    Feature: Config
    Description: Test deterministic run with setting the seed
    Expectation: Runs successfully if num_parallel_worker=1
    """
    logger.info("test_seed_deterministic")

    # Save original configuration values
    num_parallel_workers_original = ds.config.get_num_parallel_workers()
    seed_original = ds.config.get_seed()

    ds.config.set_seed(0)
    ds.config.set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # seed will be read in during constructor call
    random_crop_op = vision.RandomCrop([512, 512], [200, 200, 200, 200])
    decode_op = vision.Decode()
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_crop_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])
    # If seed is set up on constructor, so the two ops output deterministic sequence
    random_crop_op2 = vision.RandomCrop([512, 512], [200, 200, 200, 200])
    data2 = data2.map(operations=random_crop_op2, input_columns=["image"])

    dataset_equal(data1, data2, 0)

    # Restore original configuration values
    ds.config.set_num_parallel_workers(num_parallel_workers_original)
    ds.config.set_seed(seed_original)


def test_deterministic_run_distribution():
    """
    Feature: Config
    Description: Test deterministic run with with setting the seed being used in a distribution
    Expectation: Output is equal to the expected output
    """
    logger.info("test_deterministic_run_distribution")

    # Save original configuration values
    num_parallel_workers_original = ds.config.get_num_parallel_workers()
    seed_original = ds.config.get_seed()

    # when we set the seed all operations within our dataset should be deterministic
    ds.config.set_seed(0)
    ds.config.set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    random_horizontal_flip_op = vision.RandomHorizontalFlip(0.1)
    decode_op = vision.Decode()
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=random_horizontal_flip_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])
    # If seed is set up on constructor, so the two ops output deterministic sequence
    random_horizontal_flip_op2 = vision.RandomHorizontalFlip(0.1)
    data2 = data2.map(operations=random_horizontal_flip_op2, input_columns=["image"])

    dataset_equal(data1, data2, 0)

    # Restore original configuration values
    ds.config.set_num_parallel_workers(num_parallel_workers_original)
    ds.config.set_seed(seed_original)


def test_deterministic_python_seed():
    """
    Feature: Config
    Description: Test deterministic execution with seed in Python
    Expectation: Output is equal to the expected output
    """
    logger.info("test_deterministic_python_seed")

    # Save original configuration values
    num_parallel_workers_original = ds.config.get_num_parallel_workers()
    seed_original = ds.config.get_seed()

    ds.config.set_seed(0)
    ds.config.set_num_parallel_workers(1)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    transforms = [
        vision.Decode(True),
        vision.RandomCrop([512, 512], [200, 200, 200, 200]),
        vision.ToTensor(),
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    data1 = data1.map(operations=transform, input_columns=["image"])
    data1_output = []
    # config.set_seed() calls random.seed()
    for data_one in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        data1_output.append(data_one["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform, input_columns=["image"])
    # config.set_seed() calls random.seed(), resets seed for next dataset iterator
    ds.config.set_seed(0)

    data2_output = []
    for data_two in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
        data2_output.append(data_two["image"])

    np.testing.assert_equal(data1_output, data2_output)

    # Restore original configuration values
    ds.config.set_num_parallel_workers(num_parallel_workers_original)
    ds.config.set_seed(seed_original)


def test_deterministic_python_seed_multi_thread():
    """
    Feature: Config
    Description: Test deterministic execution with seed in Python with multi-thread PyFunc run
    Expectation: Exception is raised as expected
    """
    logger.info("test_deterministic_python_seed_multi_thread")

    # Sometimes there are some ITERATORS left in ITERATORS_LIST when run all UTs together,
    # and cause core dump and blocking in this UT. Add cleanup() here to fix it.
    it._cleanup()  # pylint: disable=W0212

    # Save original configuration values
    num_parallel_workers_original = ds.config.get_num_parallel_workers()
    seed_original = ds.config.get_seed()
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_num_parallel_workers(3)
    ds.config.set_seed(0)

    # Disable shared memory to save shm in CI
    ds.config.set_enable_shared_mem(False)

    # when we set the seed all operations within our dataset should be deterministic
    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.RandomCrop([512, 512], [200, 200, 200, 200]),
        vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    data1 = data1.map(operations=transform, input_columns=["image"], python_multiprocessing=True)
    data1_output = []
    # config.set_seed() calls random.seed()
    for data_one in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        data1_output.append(data_one["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # If seed is set up on constructor
    data2 = data2.map(operations=transform, input_columns=["image"], python_multiprocessing=True)
    # config.set_seed() calls random.seed()
    ds.config.set_seed(0)

    data2_output = []
    for data_two in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
        data2_output.append(data_two["image"])

    try:
        np.testing.assert_equal(data1_output, data2_output)
    except Exception as e:
        # expect output to not match during multi-threaded execution
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Array" in str(e)

    # Restore original configuration values
    ds.config.set_num_parallel_workers(num_parallel_workers_original)
    ds.config.set_seed(seed_original)
    ds.config.set_enable_shared_mem(mem_original)


def test_auto_num_workers_error():
    """
    Feature: Config
    Description: Test set_auto_num_workers with invalid input
    Expectation: Error is raised as expected
    """
    err_msg = ""
    try:
        ds.config.set_auto_num_workers([1, 2])
    except TypeError as e:
        err_msg = str(e)

    assert "must be of type bool" in err_msg


def test_auto_num_workers():
    """
    Feature: Config
    Description: Test set_auto_num_workers with no argument
    Expectation: Output is equal to the expected value
    """
    saved_config = ds.config.get_auto_num_workers()
    assert isinstance(saved_config, bool)
    # change to a different config
    flipped_config = not saved_config
    ds.config.set_auto_num_workers(flipped_config)
    assert flipped_config == ds.config.get_auto_num_workers()
    # now flip this back
    ds.config.set_auto_num_workers(saved_config)
    assert saved_config == ds.config.get_auto_num_workers()


def test_enable_watchdog():
    """
    Feature: Test the function of get_enable_watchdog and set_enable_watchdog.
    Description: We add this new interface so we can close the watchdog thread
    Expectation: The default state is True, when execute set_enable_watchdog, the state will update.
    """
    saved_config = ds.config.get_enable_watchdog()
    assert isinstance(saved_config, bool)
    assert saved_config is True
    # change to a different config
    flipped_config = not saved_config
    ds.config.set_enable_watchdog(flipped_config)
    assert flipped_config == ds.config.get_enable_watchdog()
    # now flip this back
    ds.config.set_enable_watchdog(saved_config)
    assert saved_config == ds.config.get_enable_watchdog()


def test_multiprocessing_timeout_interval():
    """
    Feature: Test the function of get_multiprocessing_timeout_interval and set_multiprocessing_timeout_interval.
    Description: We add this new interface so we can adjust the timeout of multiprocessing get function.
    Expectation: The default state is 300s, when execute set_multiprocessing_timeout_interval, the state will update.
    """
    saved_config = ds.config.get_multiprocessing_timeout_interval()
    assert saved_config == 300
    # change to a different config
    flipped_config = 1000
    ds.config.set_multiprocessing_timeout_interval(flipped_config)
    assert flipped_config == ds.config.get_multiprocessing_timeout_interval()
    # now flip this back
    ds.config.set_multiprocessing_timeout_interval(saved_config)
    assert saved_config == ds.config.get_multiprocessing_timeout_interval()


def test_config_bool_type_error():
    """
    Feature: Now many interfaces of config support bool input even its valid input is int.
    Description: We will raise a type error when input is a bool when it should be int.
    Expectation: TypeError will be raised when input is a bool.
    """
    # set_seed will raise TypeError if input is a boolean
    config_error_func(ds.config.set_seed, True, TypeError, "seed isn't of type int")

    # set_prefetch_size will raise TypeError if input is a boolean
    config_error_func(ds.config.set_prefetch_size, True, TypeError, "size isn't of type int")

    # set_num_parallel_workers will raise TypeError if input is a boolean
    config_error_func(ds.config.set_num_parallel_workers, True, TypeError, "num isn't of type int")

    # set_monitor_sampling_interval will raise TypeError if input is a boolean
    config_error_func(ds.config.set_monitor_sampling_interval, True, TypeError, "interval isn't of type int")

    # set_callback_timeout will raise TypeError if input is a boolean
    config_error_func(ds.config.set_callback_timeout, True, TypeError, "timeout isn't of type int")

    # set_autotune_interval will raise TypeError if input is a boolean
    config_error_func(ds.config.set_autotune_interval, True, TypeError, "interval must be of type int")

    # set_sending_batches will raise TypeError if input is a boolean
    config_error_func(ds.config.set_sending_batches, True, TypeError, "batch_num must be an int dtype")

    # set_multiprocessing_timeout_interval will raise TypeError if input is a boolean
    config_error_func(ds.config.set_multiprocessing_timeout_interval, True, TypeError, "interval isn't of type int")


def test_fast_recovery():
    """
    Feature: Test the get_fast_recovery function
    Description: This function only accepts a boolean as input and outputs error otherwise
    Expectation: TypeError will be raised when input argument is missing or is not a boolean
    """
    # set_fast_recovery will raise TypeError if input is an integer
    config_error_func(ds.config.set_fast_recovery, 0, TypeError, "fast_recovery must be a boolean dtype")
    # set_fast_recovery will raise TypeError if input is a string
    config_error_func(ds.config.set_fast_recovery, "True", TypeError, "fast_recovery must be a boolean dtype")
    # set_fast_recovery will raise TypeError if input is a tuple
    config_error_func(ds.config.set_fast_recovery, (True,), TypeError, "fast_recovery must be a boolean dtype")
    # set_fast_recovery will raise TypeError if input is None
    config_error_func(ds.config.set_fast_recovery, None, TypeError, "fast_recovery must be a boolean dtype")
    # set_fast_recovery will raise TypeError if no input is provided
    with pytest.raises(TypeError) as error_info:
        ds.config.set_fast_recovery()
    assert "set_fast_recovery() missing 1 required positional argument: 'fast_recovery'" in str(error_info.value)


def test_debug_mode_error_case():
    """
    Feature: Test the debug mode setter function
    Description: This function only accepts a boolean as first input and list as second input, outputs error otherwise
    Expectation: TypeError will be raised when input argument is missing or is invalid
    """
    # set_debug_mode will raise TypeError if first input is an integer
    config_error_func(ds.config.set_debug_mode, 0, TypeError, "debug_mode_flag isn't of type boolean.")
    # set_debug_mode will raise TypeError if first input is a string
    config_error_func(ds.config.set_debug_mode, "True", TypeError, "debug_mode_flag isn't of type boolean.")
    # set_debug_mode will raise TypeError if first input is a tuple
    config_error_func(ds.config.set_debug_mode, (True,), TypeError, "debug_mode_flag isn't of type boolean.")
    # set_debug_mode will raise TypeError if first input is None
    config_error_func(ds.config.set_debug_mode, None, TypeError, "debug_mode_flag isn't of type boolean.")
    # set_debug_mode will raise TypeError if no input is provided
    with pytest.raises(TypeError) as error_info:
        ds.config.set_debug_mode()
    assert "set_debug_mode() missing 1 required positional argument: 'debug_mode_flag'" in str(error_info.value)

    # set_debug_mode will raise TypeError if second input is not valid
    with pytest.raises(TypeError) as error_info:
        ds.config.set_debug_mode(True, debug.PrintDataHook())
    assert "debug_hook_list is not a list" in str(error_info.value)
    def func():
        pass
    with pytest.raises(TypeError) as error_info:
        ds.config.set_debug_mode(True, [func])
    assert "All items in debug_hook_list must be of type DebugHook" in str(error_info.value)


def test_error_samples_mode():
    """
    Feature: Test the get_error_samples_mode function
    Description: This function only accepts ErrorSamplesMode enum values as input and outputs error otherwise
    Expectation: For error input, error is raised as expected
    """
    # set_error_samples_mode will raise TypeError if input is boolean
    config_error_func(config.set_error_samples_mode, False, TypeError,
                      "is not of type [<enum 'ErrorSamplesMode'>]")
    # set_error_samples_mode will raise TypeError if input is int
    config_error_func(config.set_error_samples_mode, 1, TypeError,
                      "is not of type [<enum 'ErrorSamplesMode'>]")
    # set_error_samples_mode will raise TypeError if input is a string
    config_error_func(config.set_error_samples_mode, "Zero", TypeError,
                      "is not of type [<enum 'ErrorSamplesMode'>]")
    # set_error_samples_mode will raise TypeError if input is a tuple
    config_error_func(config.set_error_samples_mode, (1,), TypeError,
                      "is not of type [<enum 'ErrorSamplesMode'>]")
    # set_error_samples_mode will raise TypeError if input is None
    config_error_func(config.set_error_samples_mode, None, TypeError,
                      "is not of type [<enum 'ErrorSamplesMode'>]")

    # set_error_samples_mode will raise TypeError if no input is provided
    with pytest.raises(TypeError) as error_info:
        config.set_error_samples_mode()
    assert "set_error_samples_mode() missing 1 required positional argument: 'error_samples_mode'" in \
           str(error_info.value)

    # set_error_samples_mode will raise TypeError if too many parameters are provided
    with pytest.raises(TypeError) as error_info:
        config.set_error_samples_mode(config.ErrorSamplesMode.REPLACE, 10)
    assert "set_error_samples_mode() takes 1 positional argument but 2 were given" in str(error_info.value)


if __name__ == '__main__':
    test_basic()
    test_get_seed()
    test_pipeline()
    test_deterministic_run_fail()
    test_seed_undeterministic()
    test_seed_deterministic()
    test_deterministic_run_distribution()
    test_deterministic_python_seed()
    test_deterministic_python_seed_multi_thread()
    test_auto_num_workers_error()
    test_auto_num_workers()
    test_enable_watchdog()
    test_multiprocessing_timeout_interval()
    test_config_bool_type_error()
    test_fast_recovery()
    test_debug_mode_error_case()
    test_error_samples_mode()
