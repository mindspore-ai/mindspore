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
Testing RandomAffine op in DE
"""
import cv2
import numpy as np
import os
from PIL import Image
import pytest

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms as t_trans
import mindspore.dataset.vision.transforms as vision
from mindspore.dataset.vision.utils import Inter
from mindspore import log as logger
from util import visualize_list, save_and_check_md5, save_and_check_md5_pil, \
    config_get_set_seed, config_get_set_num_parallel_workers

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
MNIST_DATA_DIR = "../data/dataset/testMnistData"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


DATA_DIR_1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")


def test_random_affine_op(plot=False):
    """
    Feature: RandomAffine op
    Description: Test RandomAffine in Python transformations
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_affine_op")
    # define map operations
    transforms1 = [
        vision.Decode(True),
        vision.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), resample=Inter.NEAREST),
        vision.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)

    transforms2 = [
        vision.Decode(True),
        vision.ToTensor()
    ]
    transform2 = t_trans.Compose(transforms2)

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform1, input_columns=["image"])
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform2, input_columns=["image"])

    image_affine = []
    image_original = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_affine.append(image1)
        image_original.append(image2)
    if plot:
        visualize_list(image_original, image_affine)


def test_random_affine_op_c(plot=False):
    """
    Feature: RandomAffine op
    Description: Test RandomAffine in Cpp implementation
    Expectation: The dataset is processed as expected
    """
    logger.info("test_random_affine_op_c")
    # define map operations
    transforms1 = [
        vision.Decode(),
        vision.RandomAffine(degrees=0, translate=(0.5, 0.5, 0, 0), resample=Inter.AREA)
    ]

    transforms2 = [
        vision.Decode()
    ]

    #  First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transforms1, input_columns=["image"])
    #  Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transforms2, input_columns=["image"])

    image_affine = []
    image_original = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = item1["image"]
        image2 = item2["image"]
        image_affine.append(image1)
        image_original.append(image2)
    if plot:
        visualize_list(image_original, image_affine)


def test_random_affine_md5():
    """
    Feature: RandomAffine op
    Description: Test RandomAffine with md5 comparisons
    Expectation: Passes md5 comparison test
    """
    logger.info("test_random_affine_md5")
    original_seed = config_get_set_seed(55)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    # define map operations
    transforms = [
        vision.Decode(True),
        vision.RandomAffine(degrees=(-5, 15), translate=(0.1, 0.3),
                            scale=(0.9, 1.1), shear=(-10, 10, -5, 5)),
        vision.ToTensor()
    ]
    transform = t_trans.Compose(transforms)

    #  Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=transform, input_columns=["image"])

    # check results with md5 comparison
    filename = "random_affine_01_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_random_affine_c_md5():
    """
    Feature: RandomAffine op
    Description: Test RandomAffine C Op with md5 comparison
    Expectation: Passes the md5 comparison test
    """
    logger.info("test_random_affine_c_md5")
    original_seed = config_get_set_seed(1)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    # define map operations
    transforms = [
        vision.Decode(),
        vision.RandomAffine(degrees=(-5, 15), translate=(-0.1, 0.1, -0.3, 0.3),
                            scale=(0.9, 1.1), shear=(-10, 10, -5, 5))
    ]

    #  Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=transforms, input_columns=["image"])

    # check results with md5 comparison
    filename = "random_affine_01_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_random_affine_default_c_md5():
    """
    Feature: RandomAffine op
    Description: Test RandomAffine C Op with default parameters with md5 comparison
    Expectation: Passes the md5 comparison test
    """
    logger.info("test_random_affine_default_c_md5")
    original_seed = config_get_set_seed(1)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    # define map operations
    transforms = [
        vision.Decode(),
        vision.RandomAffine(degrees=0)
    ]

    #  Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=transforms, input_columns=["image"])

    # check results with md5 comparison
    filename = "random_affine_01_default_c_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_random_affine_py_exception_non_pil_images():
    """
    Feature: RandomAffine op
    Description: Test RandomAffine with input image of ndarray and not PIL
    Expectation: Error is raised as expected
    """
    logger.info("test_random_affine_exception_negative_degrees")
    dataset = ds.MnistDataset(MNIST_DATA_DIR, num_samples=3, num_parallel_workers=3)
    try:
        transform = t_trans.Compose([vision.ToTensor(),
                                                          vision.RandomAffine(degrees=(15, 15))])
        dataset = dataset.map(operations=transform, input_columns=["image"], num_parallel_workers=3)
        for _ in dataset.create_dict_iterator(num_epochs=1):
            pass
    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Pillow image" in str(e)


def test_random_affine_exception_negative_degrees():
    """
    Feature: RandomAffine op
    Description: Test RandomAffine with input degrees in negative
    Expectation: Error is raised as expected
    """
    logger.info("test_random_affine_exception_negative_degrees")
    try:
        _ = vision.RandomAffine(degrees=-15)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input degrees is not within the required interval of [0, 16777216]."


def test_random_affine_exception_translation_range():
    """
    Feature: RandomAffine op
    Description: Test RandomAffine where translation value is not in [-1, 1]
    Expectation: Error is raised as expected
    """
    logger.info("test_random_affine_exception_translation_range")
    try:
        _ = vision.RandomAffine(degrees=15, translate=(0.1, 1.5))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input translate at 1 is not within the required interval of [-1.0, 1.0]."
    logger.info("test_random_affine_exception_translation_range")
    try:
        _ = vision.RandomAffine(degrees=15, translate=(-2, 1.5))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input translate at 0 is not within the required interval of [-1.0, 1.0]."


def test_random_affine_exception_scale_value():
    """
    Feature: RandomAffine op
    Description: Test RandomAffine where scale is not valid
    Expectation: Error is raised as expected
    """
    logger.info("test_random_affine_exception_scale_value")
    try:
        _ = vision.RandomAffine(degrees=15, scale=(0.0, 0.0))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input scale[1] must be greater than 0."

    try:
        _ = vision.RandomAffine(degrees=15, scale=(2.0, 1.1))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input scale[1] must be equal to or greater than scale[0]."


def test_random_affine_exception_shear_value():
    """
    Feature: RandomAffine op
    Description: Test RandomAffine where sheer is a number but is not positive
    Expectation: Error is raised as expected
    """
    logger.info("test_random_affine_exception_shear_value")
    try:
        _ = vision.RandomAffine(degrees=15, shear=-5)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input shear must be greater than 0."

    try:
        _ = vision.RandomAffine(degrees=15, shear=(5, 1))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input shear[1] must be equal to or greater than shear[0]"

    try:
        _ = vision.RandomAffine(degrees=15, shear=(5, 1, 2, 8))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input shear[1] must be equal to or greater than shear[0] and " \
                         "shear[3] must be equal to or greater than shear[2]."

    try:
        _ = vision.RandomAffine(degrees=15, shear=(5, 9, 2, 1))
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input shear[1] must be equal to or greater than shear[0] and " \
                         "shear[3] must be equal to or greater than shear[2]."


def test_random_affine_exception_degrees_size():
    """
    Feature: RandomAffine op
    Description: Test RandomAffine where degrees is a list or tuple and its length is not 2
    Expectation: Error is raised as expected
    """
    logger.info("test_random_affine_exception_degrees_size")
    try:
        _ = vision.RandomAffine(degrees=[15])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "If degrees is a sequence, the length must be 2."


def test_random_affine_exception_translate_size():
    """
    Feature: RandomAffine op
    Description: Test RandomAffine where translate is not a list or tuple of length 2
    Expectation: Error is raised as expected
    """
    logger.info("test_random_affine_exception_translate_size")
    try:
        _ = vision.RandomAffine(degrees=15, translate=0.1)
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(
            e) == "Argument translate with value 0.1 is not of type [<class 'list'>," \
                  " <class 'tuple'>], but got <class 'float'>."


def test_random_affine_exception_scale_size():
    """
    Feature: RandomAffine op
    Description: Test RandomAffine where scale is not a list or tuple of length 2
    Expectation: Error is raised as expected
    """
    logger.info("test_random_affine_exception_scale_size")
    try:
        _ = vision.RandomAffine(degrees=15, scale=0.5)
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Argument scale with value 0.5 is not of type [<class 'tuple'>," \
                         " <class 'list'>], but got <class 'float'>."


def test_random_affine_exception_shear_size():
    """
    Feature: RandomAffine op
    Description: Test RandomAffine where shear is not a list or tuple of length 2 or 4
    Expectation: Error is raised as expected
    """
    logger.info("test_random_affine_exception_shear_size")
    try:
        _ = vision.RandomAffine(degrees=15, shear=(-5, 5, 10))
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "shear must be of length 2 or 4."


def test_random_affine_op_exception_c_resample():
    """
    Feature: RandomAffine
    Description: Test RandomAffine with unsupported resample values for NumPy input in eager mode
    Expectation: Exception is raised as expected
    """
    logger.info("test_random_affine_op_exception_c_resample")

    image = cv2.imread("../data/dataset/apple.jpg")

    with pytest.raises(RuntimeError) as error_info:
        random_affine_op = vision.RandomAffine(degrees=0, translate=(0.5, 0.5, 0, 0), resample=Inter.PILCUBIC)
        _ = random_affine_op(image)
    assert "RandomAffine: Invalid InterpolationMode" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        random_affine_op = vision.RandomAffine(degrees=2, translate=(0.2, 0.2, 0, 0), resample=Inter.ANTIALIAS)
        _ = random_affine_op(image)
    assert "Input image should be a Pillow image." in str(error_info.value)


def test_random_affine_op_exception_py_resample():
    """
    Feature: RandomAffine
    Description: Test RandomAffine with unsupported resample values for PIL input in eager mode
    Expectation: Exception is raised as expected
    """
    logger.info("test_random_affine_op_exception_py_resample")

    image = Image.open("../data/dataset/apple.jpg").convert("RGB")

    with pytest.raises(TypeError) as error_info:
        random_affine_op = vision.RandomAffine(degrees=0, translate=(0.5, 0.5, 0, 0), resample=Inter.PILCUBIC)
        _ = random_affine_op(image)
    assert "Current Interpolation is not supported with PIL input." in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        random_affine_op = vision.RandomAffine(degrees=2, translate=(0.2, 0.2, 0, 0), resample=Inter.AREA)
        _ = random_affine_op(image)
    assert "Current Interpolation is not supported with PIL input." in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        random_affine_op = vision.RandomAffine(degrees=1, translate=(0.1, 0.1, 0, 0), resample=Inter.ANTIALIAS)
        _ = random_affine_op(image)
    # Note: Lower PILLOW versions like 7.2.0 return "image.LANCZOS/Image.ANTIALIAS (1) cannot be used."
    #     Higher PILLOW versions like 9.0.1 return "Image.Resampling.LANCZOS (1) cannot be used."
    #     since ANTIALIAS is deprecated and replaced with LANCZOS.
    assert "LANCZOS" in str(error_info.value)
    assert "cannot be used." in str(error_info.value)


def test_random_affine_operation_01():
    """
    Feature: RandomAffine operation
    Description: Testing the normal functionality of the RandomAffine operator
    Expectation: The Output is equal to the expected output
    """
    # The parameter degrees is a float value within the normal range.
    degrees = 20.5
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    random_affine_op = vision.RandomAffine(degrees=degrees)
    dataset2 = dataset2.map(input_columns=["image"], operations=random_affine_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # The parameter degrees has two distinct values within the normal range.
    degrees = (20.0, 30.0)
    translate = [0.2, 0.8]
    scale = (0.9, 1.1)
    shear = 100
    resample = Inter.BILINEAR
    fill_value = 1
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    random_affine_op = vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                                          resample=resample, fill_value=fill_value)
    dataset2 = dataset2.map(input_columns=["image"], operations=random_affine_op)

    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # When the parameter degrees is negative, the RandomAffine interface call succeeds.
    degrees = (-10, 10)
    translate = (0.5, 0.8)
    scale = (0.9, 1.1)
    shear = (5, 10, 5, 10)
    resample = Inter.NEAREST
    fill_value = 0
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    random_affine_op = vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                                          resample=resample, fill_value=fill_value)
    dataset2 = dataset2.map(input_columns=["image"], operations=random_affine_op)

    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # When the parameter degrees is 0, the RandomAffine interface call succeeds.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = [10, 20]
    resample = Inter.AREA
    fill_value = (1, 2, 3)
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    random_affine_op = vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                                          resample=resample, fill_value=fill_value)
    dataset2 = dataset2.map(input_columns=["image"], operations=random_affine_op)

    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # When the parameter translate is 0 or 1, the RandomAffine interface call succeeds.
    degrees = 15
    translate = [0, 1]
    scale = None
    shear = [1.0, 2.0, 3.0, 4.0]
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    random_affine_op = vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    dataset2 = dataset2.map(input_columns=["image"], operations=random_affine_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # When the tx_min value in the translate parameter is negative, the RandomAffine interface call succeeds.
    degrees = 15
    translate = (-0.1, 0.8)
    scale = [1, 2]
    shear = None
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    random_affine_op = vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    dataset2 = dataset2.map(input_columns=["image"], operations=random_affine_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass


def test_random_affine_operation_02():
    """
    Feature: RandomAffine operation
    Description: Testing the normal functionality of the RandomAffine operator
    Expectation: The Output is equal to the expected output
    """
    # When the parameter "shear" is a 4-tuple, the RandomAffine interface call succeeds.
    degrees = 15
    translate = (-0.1, 0.8)
    scale = [1, 2]
    shear = (-0.5, 0.5, -0.5, 0.5)
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    random_affine_op = vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    dataset2 = dataset2.map(input_columns=["image"], operations=random_affine_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # When the parameter "shear" is negative, the RandomAffine interface call succeeds.
    degrees = 15
    translate = (-0.1, 0.8)
    scale = [1, 2]
    shear = [-10, 10]
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    random_affine_op = vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    dataset2 = dataset2.map(input_columns=["image"], operations=random_affine_op)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # When the input image is in JPG format, the RandomAffine interface call succeeds.
    with Image.open(image_jpg) as image:
        degrees = 0
        translate = (-0.8, 1)
        scale = (0, 2)
        shear = 0.001
        resample = Inter.BILINEAR
        fill_value = 1
        random_affine_op = vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                                              resample=resample, fill_value=fill_value)
        _ = random_affine_op(image)

    # When the input image is in PNG format, the RandomAffine interface call succeeds.
    image = cv2.imread(image_png)
    degrees = 11.56
    translate = [0.5, 0.6]
    scale = (1, 1)
    shear = 167772168
    resample = Inter.NEAREST
    fill_value = 255
    random_affine_op = vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                                          resample=resample, fill_value=fill_value)
    _ = random_affine_op(image)

    # When the input image is a GIF, the RandomAffine interface call succeeds.
    with Image.open(image_gif) as image:
        degrees = [-100, 180]
        translate = (-0.3, 0.1)
        scale = [2.1, 1024]
        shear = 10.1
        resample = Inter.BICUBIC
        fill_value = 0
        random_affine_op = vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                                              resample=resample, fill_value=fill_value)
        _ = random_affine_op(image)

    # When the input image is in BMP format, the RandomAffine interface call succeeds.
    image = cv2.imread(image_bmp)
    degrees = (360, 360)
    translate = [0, 0]
    scale = [0, 16777216]
    shear = [2, 8]
    resample = Inter.BILINEAR
    fill_value = (180, 26, 109)
    random_affine_op = vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                                          resample=resample, fill_value=fill_value)
    _ = random_affine_op(image)

    # When the input shape is (346, 489, 3), the RandomAffine interface call succeeds.
    image = np.random.randn(346, 489, 3)
    degrees = [0, 180]
    translate = (0, 0)
    scale = (0.01, 0.03)
    shear = (0.6, 0.6)
    resample = Inter.BICUBIC
    fill_value = (64, 64, 64)
    random_affine_op = vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                                          resample=resample, fill_value=fill_value)
    _ = random_affine_op(image)

    # When the input is a PIL image and fill_value is a tuple, the RandomAffine interface call succeeds.
    image = cv2.imread(image_jpg)
    image = vision.ToPIL()(image)
    degrees = 728.5
    fill_value = (100, 200, 220)
    random_affine_op = vision.RandomAffine(degrees=degrees, fill_value=fill_value)
    _ = random_affine_op(image)


def test_random_affine_operation_03():
    """
    Feature: RandomAffine operation
    Description: Testing the normal functionality of the RandomAffine operator
    Expectation: The Output is equal to the expected output
    """
    # Input shape is (128, 128, 1)
    image = np.random.randint(0, 255, (128, 128, 1)).astype(np.uint8)
    degrees = (4.2, 8.0)
    translate = [-0.4, 1, 0, 0.5]
    scale = (0.1, 0.2)
    shear = (1, 4, 3, 3)
    resample = Inter.NEAREST
    fill_value = (180, 180, 200)
    random_affine_op = vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                                          resample=resample, fill_value=fill_value)
    random_affine_op(image)

    # Input shape is (128, 128, 4)
    image = np.random.randint(0, 255, (128, 128, 4)).astype(np.uint8)
    degrees = [0.6, 0.6]
    translate = (-0.4, -0.2, -1, -1)
    scale = [16777215, 16777215.1]
    shear = [0.8, 10, 3.6, 4.1]
    resample = Inter.BILINEAR
    fill_value = (0, 255, 230)
    random_affine_op = vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                                          resample=resample, fill_value=fill_value)
    _ = random_affine_op(image)

    # Input shape is (192, 263)
    image = np.random.randn(192, 263)
    degrees = 728.5
    translate = [0.3, 0.3, 0.3, 0.3]
    scale = [18, 21]
    shear = (16777216, 16777216.1, 16777211, 16777216)
    resample = Inter.BILINEAR
    fill_value = (12, 8, 12)
    random_affine_op = vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                                          resample=resample, fill_value=fill_value)
    _ = random_affine_op(image)

    # Pipeline mode, combination enhancement, RandomAffine interface call successful
    # First dataset
    dataset = ds.ImageFolderDataset(DATA_DIR_1, 1)
    transforms_list = [
        vision.Decode(),
        vision.RandomAffine(10, [0, 1], (1, 5), 20.1, Inter.NEAREST, 0),
        vision.ToTensor()
    ]
    transform = t_trans.Compose(transforms_list)
    dataset = dataset.map(input_columns=["image"], operations=transform)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_random_affine_exception_01():
    """
    Feature: RandomAffine operation
    Description: Testing the RandomAffine Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When all degrees parameters are negative, the RandomAffine interface call fails.
    degrees = -0.1
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    with pytest.raises(ValueError, match="Input degrees is not within the required interval of \\[0, 16777216\\]."):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # The first value of the parameter degrees is greater than the second value.
    degrees = [10, 5]
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    with pytest.raises(ValueError, match="degrees should be in \\(min,max\\) format. Got \\(max,min\\)"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # The parameter "degrees" has a string value.
    degrees = "10"
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    with pytest.raises(TypeError, match=("Argument degrees with value 10 is not of type \\[<class 'int'>, <class "
                                         "'float'>, <class 'list'>, <class 'tuple'>\\], but got <class 'str'>.")):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # When the degrees parameter is set to 3, the RandomAffine interface call fails.
    degrees = (10, 20, 30)
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    with pytest.raises(TypeError, match="If degrees is a sequence, the length must be 2"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # When the degrees parameter is an array, the RandomAffine interface call fails.
    degrees = [20]
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    with pytest.raises(TypeError, match="If degrees is a sequence, the length must be 2"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # When the degrees parameter is an array, the RandomAffine interface call fails.
    degrees = ("10", "20")
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    with pytest.raises(TypeError, match="Argument degrees\\[0\\] with value 10 is not of type"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # When the degrees parameter is not set, the RandomAffine interface call fails.
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    with pytest.raises(TypeError, match="missing a required argument: 'degrees'"):
        vision.RandomAffine(translate=translate, scale=scale)

    # In the parameter translate, tx_min is 0.8 and tx_max is -1.
    degrees = 15
    translate = [0.8, -1]
    scale = (0.9, 1.1)
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    with pytest.raises(RuntimeError,
                       match=r"RandomAffine: minimum of 'translate'\(translate_range\) on x is greater "
                             "than maximum: min = 0.800000, max = -1.000000"):
        random_affine_op = vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)
        dataset2 = dataset2.map(input_columns=["image"], operations=random_affine_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # The parameter tx_min in the translate function is greater than tx_max.
    degrees = 15
    translate = (1.1, 0.5)
    scale = (0.9, 1.1)
    with pytest.raises(ValueError,
                       match="Input translate at 0 is not within the required interval of \\[-1.0, 1.0\\]."):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # When the parameter translate value exceeds 1.0, the RandomAffine interface call fails.
    degrees = 15
    translate = [0.5, 1.1]
    scale = (0.9, 1.1)
    with pytest.raises(ValueError,
                       match="Input translate at 1 is not within the required interval of \\[-1.0, 1.0\\]."):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # When the parameter translate has only one value, the RandomAffine interface call fails.
    degrees = 15
    translate = [0.5]
    scale = (0.9, 1.1)
    with pytest.raises(TypeError, match="translate should be a list or tuple of length 2 or 4"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # When the number of translate values is 3, the RandomAffine interface call fails.
    degrees = 15
    translate = (0.1, 0.2, 0.3)
    scale = (0.9, 1.1)
    with pytest.raises(TypeError, match="translate should be a list or tuple of length 2 or 4"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)


def test_random_affine_exception_02():
    """
    Feature: RandomAffine operation
    Description: Testing the RandomAffine Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the parameter translate is set to string, the RandomAffine interface call fails.
    degrees = 15
    translate = ("0.2", "0.8")
    scale = (0.9, 1.1)
    with pytest.raises(TypeError, match=("Argument translate\\[0\\] with value 0.2 is not of "
                                         "type \\[<class 'int'>, <class 'float'>\\]")):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # The first value in the scale parameter is greater than the second value.
    degrees = 15
    translate = (0.1, 0.1)
    scale = (1.1, 0.9)
    with pytest.raises(ValueError, match="Input scale\\[1\\] must be equal to or greater than scale\\[0\\]"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # When only one value is set for the scale parameter, the RandomAffine interface call fails.
    degrees = 15
    translate = (0.1, 0.1)
    scale = [1.2]
    with pytest.raises(TypeError, match="scale should be a list or tuple of length 2"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # When the scale parameter is set to three values, the RandomAffine interface call fails.
    degrees = 15
    translate = (0.1, 0.1)
    scale = (0.6, 0.9, 1.1)
    with pytest.raises(TypeError, match="scale should be a list or tuple of length 2"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # When the parameter "scale" is a string, the RandomAffine interface call fails.
    degrees = 15
    translate = (0.1, 0.1)
    scale = ("0.9", "1.1")
    with pytest.raises(TypeError, match=("Argument scale\\[0\\] with value 0.9 is not of "
                                         "type \\[<class 'int'>, <class 'float'>\\]")):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # When the scale parameter is negative, the RandomAffine interface call fails.
    degrees = 15
    translate = (0.1, 0.1)
    scale = [-0.1, 0.9]
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 16777216\\]"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # When scale[0] is greater than scale[1] in the scale parameter, the RandomAffine interface call fails.
    degrees = 15
    translate = (0.1, 0.1)
    scale = (0.8, 0)
    with pytest.raises(ValueError, match="Input scale\\[1\\] must be equal to or greater than scale\\[0\\]"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # When the parameter shear is 0, the RandomAffine interface call fails.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = 0
    with pytest.raises(ValueError, match="Input shear must be greater than 0"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)

    # When the parameter shear is negative, the RandomAffine interface call fails.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = -0.1
    with pytest.raises(ValueError, match="Input shear must be greater than 0"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)

    # Parameter shear is such that shear[0] is greater than shear[1].
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = [10, 5]
    with pytest.raises(ValueError, match="Input shear\\[1\\] must be equal to or greater than shear\\[0\\]"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)

    # Parameter shear: shear[2] is greater than shear[3]
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = (1.0, 2.0, 4.0, 3.0)
    with pytest.raises(ValueError, match=("Input shear\\[1\\] must be equal to or greater than shear\\[0\\] "
                                          "and shear\\[3\\] must be equal to or greater than shear\\[2\\]")):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)

    # When the parameter "shear" is of type "string", the RandomAffine interface call fails.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = "10"
    with pytest.raises(TypeError, match=("Argument shear with value 10 is not of type \\[<class "
                                         "'numbers.Number'>, <class 'tuple'>, <class 'list'>\\]")):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)

    # When the number of shear parameters equals 1, the RandomAffine interface call fails.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = [5]
    with pytest.raises(TypeError, match="shear must be of length 2 or 4"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)


def test_random_affine_exception_03():
    """
    Feature: RandomAffine operation
    Description: Testing the RandomAffine Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the number of shear parameters equals 3, the RandomAffine interface call fails.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = (5, 10, 20)
    with pytest.raises(TypeError, match="shear must be of length 2 or 4"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)

    # When the number of shear parameters equals 5, the RandomAffine interface call fails.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = [1.0, 2.0, 3.0, 4.0, 5.0]
    with pytest.raises(TypeError, match="shear must be of length 2 or 4"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)

    # When the parameter shear is set to two strings, the RandomAffine interface call fails.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = ("5", "10")
    with pytest.raises(TypeError, match=("Argument shear\\[0\\] with value 5 is not of "
                                         "type \\[<class 'int'>, <class 'float'>\\]")):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)

    # When the parameter shear is set to 4 strings, the RandomAffine interface call fails.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = ["1.0", "2.0", "3.0", "4.0"]
    with pytest.raises(TypeError, match=("Argument shear\\[0\\] with value 1.0 is "
                                         "not of type \\[<class 'int'>, <class 'float'>\\]")):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)

    # When the parameter "resample" is of type "string", the RandomAffine interface call fails.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = [-10, 10]
    resample = "Inter.NEAREST"
    with pytest.raises(TypeError,
                       match="Argument resample with value Inter.NEAREST is not of type \\[<enum 'Inter'>\\]"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample)

    # When the parameter "resample" is a list, the RandomAffine interface call fails.
    degrees = [1, 4]
    resample = [Inter.BILINEAR]
    with pytest.raises(TypeError, match="Argument resample with value \\[\\<Inter.BILINEAR: 2\\>\\] is not of type"):
        vision.RandomAffine(degrees=degrees, resample=resample)

    # When the parameter "resample" is an Inter, the RandomAffine interface call fails.
    degrees = [1, 4]
    resample = Inter
    with pytest.raises(TypeError,
                       match="Argument resample with value <enum 'Inter'> is not of type \\[<enum 'Inter'>\\]."):
        vision.RandomAffine(degrees=degrees, resample=resample)

    # When the parameter "resample" is an int, the RandomAffine interface call fails.
    degrees = [1, 4]
    resample = 10
    with pytest.raises(TypeError, match="Argument resample with value 10 is not of type"):
        vision.RandomAffine(degrees=degrees, resample=resample)

    # When the fill_value parameter is negative, the RandomAffine interface call fails.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = [-10, 10]
    resample = Inter.NEAREST
    fill_value = -1
    with pytest.raises(ValueError, match=r"Input fill_value is not within the required interval of \[0, 255\]"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                            resample=resample, fill_value=fill_value)

    # When the fill_value parameter is empty, the RandomAffine interface call fails.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = [-10, 10]
    resample = Inter.NEAREST
    fill_value = ()
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample,
                            fill_value=fill_value)

    # When the fill_value parameter is a 2-tuple, the RandomAffine interface call fails.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = [-10, 10]
    resample = Inter.NEAREST
    fill_value = (1, 2)
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample,
                            fill_value=fill_value)


def test_random_affine_exception_04():
    """
    Feature: RandomAffine operation
    Description: Testing the RandomAffine Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the fill_value parameter is a float, the RandomAffine interface call fails.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = [-10, 10]
    resample = Inter.NEAREST
    fill_value = 1.0
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample,
                            fill_value=fill_value)

    # When the fill_value parameter is a string, the RandomAffine interface call fails.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = [-10, 10]
    resample = Inter.NEAREST
    fill_value = "1"
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample,
                            fill_value=fill_value)

    # When the fill_value parameter is a 3-tuple string, the RandomAffine interface call fails.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = [-10, 10]
    resample = Inter.NEAREST
    fill_value = ("1", "2", "3")
    with pytest.raises(TypeError, match=r"Argument fill_value\[0\] with value 1 is not of type \[<class 'int'>\], "
                                        r"but got <class 'str'>."):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample,
                            fill_value=fill_value)

    # When the fill_value parameter exceeds 255, the RandomAffine interface call fails.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = [-10, 10]
    resample = Inter.NEAREST
    fill_value = 256
    with pytest.raises(ValueError, match=r"Input fill_value is not within the required interval of \[0, 255\]."):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample,
                            fill_value=fill_value)

    # When the fill_value parameter is a 4-tuple, the RandomAffine interface call fails.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = [-10, 10]
    resample = Inter.NEAREST
    fill_value = (1, 2, 3, 4)
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample,
                            fill_value=fill_value)

    # When the fill_value parameter is a list, the RandomAffine interface call fails.
    degrees = 0
    translate = (0.1, 0.1)
    scale = (0.9, 1.1)
    shear = [-10, 10]
    resample = Inter.NEAREST
    fill_value = [1, 2, 3]
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample,
                            fill_value=fill_value)

    # When the input is a list, the RandomAffine interface call fails.
    image = np.random.randn(600, 450, 3).tolist()
    degrees = [30, 60]
    random_affine_op = vision.RandomAffine(degrees=degrees)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        random_affine_op(image)

    # When the input is a tuple, the RandomAffine interface call fails.
    image = tuple(np.random.randn(600, 450, 3))
    degrees = [30, 60]
    random_affine_op = vision.RandomAffine(degrees=degrees)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>."):
        random_affine_op(image)

    # When the input image shape is one-dimensional, the RandomAffine interface call fails.
    image = np.fromfile(image_jpg, dtype=np.uint8)
    degrees = [30, 60]
    random_affine_op = vision.RandomAffine(degrees=degrees)
    with pytest.raises(RuntimeError,
                       match=r"Affine: input tensor is not in shape of <H,W> or <H,W,C>, but got rank: 1. "
                             r"You may need to perform Decode first"):
        random_affine_op(image)

    # When inputting shape as (600, 450, 3, 3), the RandomAffine interface call fails.
    image = np.random.randn(600, 450, 3, 3)
    degrees = [30, 60]
    random_affine_op = vision.RandomAffine(degrees=degrees)
    with pytest.raises(RuntimeError, match="Affine: input tensor is not in shape of <H,W> or <H,W,C>, but got rank: 4"):
        random_affine_op(image)


def test_random_affine_exception_05():
    """
    Feature: RandomAffine operation
    Description: Testing the RandomAffine Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the parameter "degrees" is a numpy array, the RandomAffine interface call fails.
    degrees = np.array([1, 4])
    with pytest.raises(TypeError, match="Argument degrees with value \\[1 4\\] is not of type"):
        vision.RandomAffine(degrees=degrees)

    # When the degrees parameter is a Tensor, the RandomAffine interface call fails.
    degrees = ms.Tensor([1, 4])
    with pytest.raises(TypeError) as e:
        vision.RandomAffine(degrees=degrees)
    assert ("Argument degrees with value {} is not of type [<class 'int'>, <class 'float'>, <class 'list'>, "
            "<class 'tuple'>]").format(degrees) in str(e)

    # When the parameter "degrees" is of type "set", the RandomAffine interface call fails.
    degrees = {100, 150}
    with pytest.raises(TypeError, match="Argument degrees with value {100, 150} is not "
                                        "of type \\[<class 'int'>, <class 'float'>, <class 'list'>, "
                                        "<class 'tuple'>\\], but got <class 'set'>."):
        vision.RandomAffine(degrees=degrees)

    # When the parameter "translate" is a NumPy array, the RandomAffine interface call fails.
    degrees = [1, 4]
    translate = np.array([0.8, 1])
    with pytest.raises(TypeError, match="Argument translate with value \\[0.8 1. \\] is not of type"):
        vision.RandomAffine(degrees=degrees, translate=translate)

    # When the parameter "translate" is a tensor, the RandomAffine interface call fails.
    degrees = [1, 4]
    translate = ms.Tensor([0.8, 1])
    with pytest.raises(TypeError) as e:
        vision.RandomAffine(degrees=degrees, translate=translate)
    assert "Argument translate with value {} is not of type [<class 'list'>, <class 'tuple'>]".format(translate) in str(
        e)

    # When the number of translate values is 5, the RandomAffine interface call fails.
    degrees = [1, 4]
    translate = [-1, 0, 0.3, 0.5, 0.8]
    with pytest.raises(TypeError, match="translate should be a list or tuple of length 2 or 4."):
        vision.RandomAffine(degrees=degrees, translate=translate)

    # When the parameter "translate" is an "int", the "RandomAffine" interface call fails.
    degrees = [1, 4]
    translate = 0.5
    with pytest.raises(TypeError,
                       match="Argument translate with value 0.5 is not of type \\[<class 'list'>, <class 'tuple'>\\]."):
        vision.RandomAffine(degrees=degrees, translate=translate)

    # When the scale parameter value exceeds the maximum value, the RandomAffine interface call fails.
    degrees = [1, 4]
    translate = [-1, 0, 0.3, 0.5]
    scale = [10, 16777217]
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 16777216\\]."):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # When the parameter scale is an integer, the RandomAffine interface call fails.
    degrees = [1, 4]
    translate = [-1, 0, 0.3, 0.5]
    scale = 500
    with pytest.raises(TypeError, match="Argument scale with value 500 is not of type "):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # When the parameter scale is a numpy, the RandomAffine interface call fails.
    degrees = [1, 4]
    translate = [-1, 0, 0.3, 0.5]
    scale = np.array([1, 10])
    with pytest.raises(TypeError, match="Argument scale with value \\[ 1 10\\] is not of type"):
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)

    # When the parameter scale is a ms.Tensor, the RandomAffine interface call fails.
    degrees = [1, 4]
    translate = [-1, 0, 0.3, 0.5]
    scale = ms.Tensor([1, 10])
    with pytest.raises(TypeError) as e:
        vision.RandomAffine(degrees=degrees, translate=translate, scale=scale)
    assert "Argument scale with value {} is not of type [<class 'tuple'>, <class 'list'>]".format(scale) in str(e)

    # When the fill_value parameter is a numpy array, the RandomAffine interface call fails.
    degrees = [1, 4]
    fill_value = np.array([10, 20, 30])
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.RandomAffine(degrees=degrees, fill_value=fill_value)


    # The shape of the input image is not <H, W> or <H, W, C>. The RandomAffine interface call failed.
    image = np.random.randint(0, 255, (658, 714, 10, 3)).astype(np.uint8)
    op = vision.RandomAffine(degrees=10.8)
    with pytest.raises(RuntimeError, match="Affine: input tensor is not in shape of <H,W>"
                                           " or <H,W,C>, but got rank: 4"):
        op(image)


if __name__ == "__main__":
    test_random_affine_op(plot=True)
    test_random_affine_op_c(plot=True)
    test_random_affine_md5()
    test_random_affine_c_md5()
    test_random_affine_default_c_md5()
    test_random_affine_py_exception_non_pil_images()
    test_random_affine_exception_negative_degrees()
    test_random_affine_exception_translation_range()
    test_random_affine_exception_scale_value()
    test_random_affine_exception_shear_value()
    test_random_affine_exception_degrees_size()
    test_random_affine_exception_translate_size()
    test_random_affine_exception_scale_size()
    test_random_affine_exception_shear_size()
    test_random_affine_op_exception_c_resample()
    test_random_affine_op_exception_py_resample()
    test_random_affine_operation_01()
    test_random_affine_operation_02()
    test_random_affine_operation_03()
    test_random_affine_exception_01()
    test_random_affine_exception_02()
    test_random_affine_exception_03()
    test_random_affine_exception_04()
    test_random_affine_exception_05()
