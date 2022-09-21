# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from PIL import Image
import pytest
import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.vision as vision
from mindspore.dataset.vision.utils import Inter
from mindspore import log as logger
from util import visualize_list, save_and_check_md5, save_and_check_md5_pil, \
    config_get_set_seed, config_get_set_num_parallel_workers

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
MNIST_DATA_DIR = "../data/dataset/testMnistData"


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
    transform1 = mindspore.dataset.transforms.Compose(transforms1)

    transforms2 = [
        vision.Decode(True),
        vision.ToTensor()
    ]
    transform2 = mindspore.dataset.transforms.Compose(transforms2)

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
    transform = mindspore.dataset.transforms.Compose(transforms)

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
        transform = mindspore.dataset.transforms.Compose([vision.ToTensor(),
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
        _ = vision.RandomAffine(degrees=15, translate=(0.1))
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
        _ = vision.RandomAffine(degrees=15, scale=(0.5))
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
