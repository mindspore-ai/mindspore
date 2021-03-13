# Copyright 2020 Huawei Technologies Co., Ltd
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
Testing RandomPosterize op in DE
"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore import log as logger
from util import visualize_list, save_and_check_md5, \
    config_get_set_seed, config_get_set_num_parallel_workers, diff_mse

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_random_posterize_op_c(plot=False, run_golden=False):
    """
    Test RandomPosterize in C transformations (uses assertion on mse as using md5 could have jpeg decoding
    inconsistencies)
    """
    logger.info("test_random_posterize_op_c")

    original_seed = config_get_set_seed(55)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # define map operations
    transforms1 = [
        c_vision.Decode(),
        c_vision.RandomPosterize((1, 8))
    ]

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transforms1, input_columns=["image"])
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=[c_vision.Decode()], input_columns=["image"])

    image_posterize = []
    image_original = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = item1["image"]
        image2 = item2["image"]
        image_posterize.append(image1)
        image_original.append(image2)

    # check mse as md5 can be inconsistent.
    # mse = 2.9668956 is calculated from
    # a thousand runs of diff_mse(np.array(image_original), np.array(image_posterize)) that all produced the same mse.
    # allow for an error of 0.0000005
    assert abs(2.9668956 - diff_mse(np.array(image_original), np.array(image_posterize))) <= 0.0000005

    if run_golden:
        # check results with md5 comparison
        filename = "random_posterize_01_result_c.npz"
        save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)

    if plot:
        visualize_list(image_original, image_posterize)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_posterize_op_fixed_point_c(plot=False, run_golden=True):
    """
    Test RandomPosterize in C transformations with fixed point
    """
    logger.info("test_random_posterize_op_c")

    # define map operations
    transforms1 = [
        c_vision.Decode(),
        c_vision.RandomPosterize(1)
    ]

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transforms1, input_columns=["image"])
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=[c_vision.Decode()], input_columns=["image"])

    image_posterize = []
    image_original = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = item1["image"]
        image2 = item2["image"]
        image_posterize.append(image1)
        image_original.append(image2)

    if run_golden:
        # check results with md5 comparison
        filename = "random_posterize_fixed_point_01_result_c.npz"
        save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)

    if plot:
        visualize_list(image_original, image_posterize)


def test_random_posterize_default_c_md5(plot=False, run_golden=True):
    """
    Test RandomPosterize C Op (default params) with md5 comparison
    """
    logger.info("test_random_posterize_default_c_md5")
    original_seed = config_get_set_seed(5)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    # define map operations
    transforms1 = [
        c_vision.Decode(),
        c_vision.RandomPosterize()
    ]

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transforms1, input_columns=["image"])
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=[c_vision.Decode()], input_columns=["image"])

    image_posterize = []
    image_original = []
    for item1, item2 in zip(data1.create_dict_iterator(output_numpy=True, num_epochs=1),
                            data2.create_dict_iterator(output_numpy=True, num_epochs=1)):
        image1 = item1["image"]
        image2 = item2["image"]
        image_posterize.append(image1)
        image_original.append(image2)

    if run_golden:
        # check results with md5 comparison
        filename = "random_posterize_01_default_result_c.npz"
        save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)

    if plot:
        visualize_list(image_original, image_posterize)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_posterize_exception_bit():
    """
    Test RandomPosterize: out of range input bits and invalid type
    """
    logger.info("test_random_posterize_exception_bit")
    # Test max > 8
    try:
        _ = c_vision.RandomPosterize((1, 9))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input is not within the required interval of [1, 8]."
    # Test min < 1
    try:
        _ = c_vision.RandomPosterize((0, 7))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input is not within the required interval of [1, 8]."
    # Test max < min
    try:
        _ = c_vision.RandomPosterize((8, 1))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input is not within the required interval of [1, 8]."
    # Test wrong type (not uint8)
    try:
        _ = c_vision.RandomPosterize(1.1)
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Argument bits with value 1.1 is not of type (<class 'list'>, <class 'tuple'>, <class 'int'>)."
    # Test wrong number of bits
    try:
        _ = c_vision.RandomPosterize((1, 1, 1))
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Size of bits should be a single integer or a list/tuple (min, max) of length 2."


def test_rescale_with_random_posterize():
    """
    Test RandomPosterize: only support CV_8S/CV_8U
    """
    logger.info("test_rescale_with_random_posterize")

    DATA_DIR_10 = "../data/dataset/testCifar10Data"
    dataset = ds.Cifar10Dataset(DATA_DIR_10)

    rescale_op = c_vision.Rescale((1.0 / 255.0), 0.0)
    dataset = dataset.map(operations=rescale_op, input_columns=["image"])

    random_posterize_op = c_vision.RandomPosterize((4, 8))
    dataset = dataset.map(operations=random_posterize_op, input_columns=["image"], num_parallel_workers=1)

    try:
        _ = dataset.output_shapes()
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "input image data type can not be float" in str(e)


if __name__ == "__main__":
    test_random_posterize_op_c(plot=False, run_golden=False)
    test_random_posterize_op_fixed_point_c(plot=False)
    test_random_posterize_default_c_md5(plot=False)
    test_random_posterize_exception_bit()
    test_rescale_with_random_posterize()
