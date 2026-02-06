# Copyright 2025 Huawei Technologies Co., Ltd
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
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import visualize_list, save_and_check_md5, \
    config_get_set_seed, config_get_set_num_parallel_workers, diff_mse

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


def test_random_posterize_op_c(plot=False, run_golden=False):
    """
    Feature: RandomPosterize op
    Description: Test RandomPosterize in Cpp implementation
    Expectation: Passes mse assertion (using md5 could have jpeg decoding inconsistencies, so not used)
    """
    logger.info("test_random_posterize_op_c")

    original_seed = config_get_set_seed(55)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # define map operations
    transforms1 = [
        vision.Decode(),
        vision.RandomPosterize((1, 8))
    ]

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transforms1, input_columns=["image"])
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=[vision.Decode()], input_columns=["image"])

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
    Feature: RandomPosterize op
    Description: Test RandomPosterize in Cpp implementation with fixed point
    Expectation: Passes md5 check
    """
    logger.info("test_random_posterize_op_c")
    original_seed = config_get_set_seed(55)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # define map operations
    transforms1 = [
        vision.Decode(),
        vision.RandomPosterize(1)
    ]

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transforms1, input_columns=["image"])
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=[vision.Decode()], input_columns=["image"])

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

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_posterize_default_c_md5(plot=False, run_golden=True):
    """
    Feature: RandomPosterize op
    Description: Test RandomPosterize C op (default params) with md5 comparison
    Expectation: Passes md5 comparison check
    """
    logger.info("test_random_posterize_default_c_md5")
    original_seed = config_get_set_seed(5)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    # define map operations
    transforms1 = [
        vision.Decode(),
        vision.RandomPosterize()
    ]

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transforms1, input_columns=["image"])
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=[vision.Decode()], input_columns=["image"])

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
    Feature: RandomPosterize op
    Description: Test RandomPosterize with out of range input bits and invalid type
    Expectation: Correct error is thrown as expected
    """
    logger.info("test_random_posterize_exception_bit")
    # Test max > 8
    try:
        _ = vision.RandomPosterize((1, 9))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input is not within the required interval of [1, 8]."
    # Test min < 1
    try:
        _ = vision.RandomPosterize((0, 7))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input is not within the required interval of [1, 8]."
    # Test max < min
    try:
        _ = vision.RandomPosterize((8, 1))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input is not within the required interval of [1, 8]."
    # Test wrong type (not uint8)
    try:
        _ = vision.RandomPosterize(1.1)
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == ("Argument bits with value 1.1 is not of type [<class 'list'>, <class 'tuple'>, "
                          "<class 'int'>], but got <class 'float'>.")
    # Test wrong number of bits
    try:
        _ = vision.RandomPosterize((1, 1, 1))
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Size of bits should be a single integer or a list/tuple (min, max) of length 2."


def test_rescale_with_random_posterize():
    """
    Feature: RandomPosterize op
    Description: Test RandomPosterize rescale (only support CV_8S/CV_8U)
    Expectation: Error is raised as expected
    """
    logger.info("test_rescale_with_random_posterize")

    DATA_DIR_10 = "../data/dataset/testCifar10Data"
    dataset = ds.Cifar10Dataset(DATA_DIR_10)

    rescale_op = vision.Rescale((1.0 / 255.0), 0.0)
    dataset = dataset.map(operations=rescale_op, input_columns=["image"])

    random_posterize_op = vision.RandomPosterize((4, 8))
    dataset = dataset.map(operations=random_posterize_op, input_columns=["image"], num_parallel_workers=1)

    try:
        _ = dataset.output_shapes()
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "data type of input image should be int" in str(e)


def test_random_posterize_operation_01():
    """
    Feature: RandomPosterize operation
    Description: Testing the normal functionality of the RandomPosterize operator
    Expectation: The Output is equal to the expected output
    """
    # Test RandomPosterize function bits is (1, 8)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    bits = (1, 8)
    random_posterize_op = vision.RandomPosterize(bits=bits)
    dataset2 = dataset2.map(input_columns=["image"], operations=random_posterize_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomPosterize function bits is 8
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    bits = 8
    random_posterize_op = vision.RandomPosterize(bits=bits)
    dataset2 = dataset2.map(input_columns=["image"], operations=random_posterize_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomPosterize function no para
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    random_posterize_op = vision.RandomPosterize()
    dataset2 = dataset2.map(input_columns=["image"], operations=random_posterize_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomPosterize function no bits
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        random_posterize_op = vision.RandomPosterize()
        image = vision.ToNumpy()(image)
        _ = random_posterize_op(image)

    # Test RandomPosterize function bits is 1
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    bits = 1
    with Image.open(image_bmp) as image:
        random_posterize_op = vision.RandomPosterize(bits)
        _ = random_posterize_op(image)

    # Test RandomPosterize function bits is 8
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    bits = 8
    with Image.open(image_png) as image:
        random_posterize_op = vision.RandomPosterize(bits)
        _ = random_posterize_op(image)

    # Test RandomPosterize function bits is [2, 3]
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    bits = [2, 3]
    with Image.open(image_gif) as image:
        random_posterize_op = vision.RandomPosterize(bits)
        _ = random_posterize_op(image)

    # Test RandomPosterize function bits is (4, 4)
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    bits = (4, 4)
    with Image.open(image_jpg) as image:
        random_posterize_op = vision.RandomPosterize(bits)
        _ = random_posterize_op(image)

    # Test RandomPosterize function input.shape is (256, 256, 3)
    bits = [1, 8]
    image = np.random.randn(256, 256, 3).astype(np.uint8)
    random_posterize_op = vision.RandomPosterize(bits)
    _ = random_posterize_op(image)

    # Test RandomPosterize function input.shape is (128, 128, 1)
    bits = (2, 8)
    image = np.random.randint(0, 255, (128, 128, 1)).astype(np.uint8)
    random_posterize_op = vision.RandomPosterize(bits)
    random_posterize_op(image)

    # Test RandomPosterize function input is <H, W>
    image = np.random.randint(-255, 255, (128, 128)).astype(np.int8)
    random_posterize_op = vision.RandomPosterize()
    _ = random_posterize_op(image)


def test_random_posterize_exception_01():
    """
    Feature: RandomPosterize operation
    Description: Testing the RandomPosterize Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test RandomPosterize function bits is (0, 8)
    bits = (0, 8)
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        vision.RandomPosterize(bits=bits)

    # Test RandomPosterize function bits is (1, 9)
    bits = (1, 9)
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        vision.RandomPosterize(bits=bits)

    # Test RandomPosterize function bits is (8, 1)
    bits = (8, 1)
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        vision.RandomPosterize(bits=bits)

    # Test RandomPosterize function bits is (1, 7.5)
    bits = (1, 7.5)
    with pytest.raises(TypeError, match="Argument bits"):
        vision.RandomPosterize(bits=bits)

    # Test RandomPosterize function bits is []
    bits = []
    with pytest.raises(TypeError, match="Size of bits should be a single integer or a list/tuple"):
        vision.RandomPosterize(bits=bits)

    # Test RandomPosterize function bits is [1]
    bits = [1]
    with pytest.raises(TypeError, match="Size of bits should be a single integer or a list/tuple"):
        vision.RandomPosterize(bits=bits)

    # Test RandomPosterize function bits is ""
    bits = ""
    with pytest.raises(TypeError, match="Argument bits"):
        vision.RandomPosterize(bits=bits)

    # Test RandomPosterize function input is 4d
    image = np.random.randint(-255, 255, (128, 128, 3, 3)).astype(np.int8)
    random_posterize_op = vision.RandomPosterize()
    with pytest.raises(RuntimeError, match="Posterize: input image is not in shape of <H,W,C> or <H,W>"):
        random_posterize_op(image)

    # Test RandomPosterize function input is 1d
    image = np.random.randint(-255, 255, (128,)).astype(np.int8)
    random_posterize_op = vision.RandomPosterize()
    with pytest.raises(RuntimeError, match="Posterize: input image is not in shape of <H,W,C> or <H,W>"):
        random_posterize_op(image)

    # Test RandomPosterize function input is list
    image = list(np.random.randint(-255, 255, (128, 128)).astype(np.uint8))
    random_posterize_op = vision.RandomPosterize()
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        random_posterize_op(image)

    # Test RandomPosterize function input is list,tolist
    image = np.random.randint(-255, 255, (128, 128)).astype(np.uint8).tolist()
    random_posterize_op = vision.RandomPosterize()
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        random_posterize_op(image)

    # Test RandomPosterize function bits is {4}
    bits = {4}
    with pytest.raises(TypeError, match="Argument bits with value \\{4\\} is not of type"):
        vision.RandomPosterize(bits)

    # Test RandomPosterize function bits is np.array([2, 5])
    bits = np.array([2, 5])
    with pytest.raises(TypeError, match="Argument bits with value \\[2 5\\] is not of type"):
        vision.RandomPosterize(bits)

    # Test RandomPosterize function bits is [2, 5, 6]
    bits = [2, 5, 6]
    with pytest.raises(TypeError,
                       match="Size of bits should be a single integer or a list/tuple \\(min, max\\) of length 2."):
        vision.RandomPosterize(bits)

    # Test RandomPosterize function bits is 2.5
    bits = 2.5
    with pytest.raises(TypeError, match=("Argument bits with value 2.5 is not of type \\[<class "
                                         "'list'>, <class 'tuple'>, <class 'int'>\\].")):
        vision.RandomPosterize(bits)

    # Test RandomPosterize function bits is True
    bits = True
    with pytest.raises(TypeError, match=("Argument bits with value True is not of type \\(<class 'list'>, "
                                         "<class 'tuple'>, <class 'int'>\\)")):
        vision.RandomPosterize(bits)

    # Test RandomPosterize function two Parameter
    with pytest.raises(TypeError, match="too many positional arguments"):
        vision.RandomPosterize(3, 6)


if __name__ == "__main__":
    test_random_posterize_op_c(plot=False, run_golden=False)
    test_random_posterize_op_fixed_point_c(plot=False)
    test_random_posterize_default_c_md5(plot=False)
    test_random_posterize_exception_bit()
    test_rescale_with_random_posterize()
    test_random_posterize_operation_01()
    test_random_posterize_exception_01()
