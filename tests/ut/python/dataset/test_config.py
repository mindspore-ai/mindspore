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

import mindspore.dataset as ds
from mindspore.dataset.core.config import _get_num_parallel_workers
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import dataset_equal

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def config_error_func(config_interface, input_args, err_type, except_err_msg):
    err_msg = ""
    try:
        config_interface(input_args)
    except err_type as e:
        err_msg = str(e)

    assert except_err_msg in err_msg


def test_get_seed():
    """
    Feature: Config
    Description: Test get_seed value without explicitly setting a default
    Expectation: Expecting to get an int
    """
    assert isinstance(ds.config.get_seed(), int)




def test_deterministic_run_fail():
    """
    Feature: Config
    Description: Test RandomCrop with seed
    Expectation: Exception is raised as expected
    """
    logger.info("test_deterministic_run_fail")

    # Save original configuration values
    num_parallel_workers_original = _get_num_parallel_workers()
    seed_original = ds.config.get_seed()

    # when we set the seed all operations within our dataset should be deterministic
    ds.config.set_seed(0)
    ds.config.set_num_parallel_workers(1)
    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    # Assuming we get the same seed on calling constructor, if this op is reused then result won't be
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

    except Exception as e:  # pylint: disable=broad-exception-caught
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
    num_parallel_workers_original = _get_num_parallel_workers()
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
    except Exception as e:  # pylint: disable=broad-exception-caught
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
    num_parallel_workers_original = _get_num_parallel_workers()
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
    num_parallel_workers_original = _get_num_parallel_workers()
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
