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
Testing Affine op in DE
"""
import cv2
import numpy as np
import os
import pytest
from PIL import Image

from mindspore import log as logger
import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms as t_trans
import mindspore.dataset.vision.transforms as vision
from mindspore.dataset.vision import Inter
from util import visualize_list, diff_mse

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"
DATA_DIR_1 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train")
image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")


def test_affine_exception_degrees_type():
    """
    Feature: Test Affine degrees type
    Description: Input the type of degrees is list
    Expectation: Got an exception to raise TyoeError
    """
    logger.info("test_affine_exception_degrees_type")
    try:
        _ = vision.Affine(degrees=[15.0], translate=[-1, 1], scale=1.0, shear=[1, 1])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Argument degrees with value [15.0] is not of type [<class 'int'>, <class 'float'>], " \
                         "but got <class 'list'>."


def test_affine_exception_scale_value():
    """
    Feature: Test Affine(scale is not valid)
    Description: Input scale is not valid
    Expectation: Got an exception to raise ValueError
    """
    logger.info("test_affine_exception_scale_value")
    try:
        _ = vision.Affine(degrees=15, translate=[1, 1], scale=0.0, shear=10)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input scale must be greater than 0."

    try:
        _ = vision.Affine(degrees=15, translate=[1, 1], scale=-0.2, shear=10)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "Input scale must be greater than 0."


def test_affine_exception_shear_size():
    """
    Feature: Test Affine(shear is not list or a tuple of length 2)
    Description: Input shear is not list or a tuple of length 2
    Expectation: Got an exception to raise TypeError
    """
    logger.info("test_affine_shear_size")
    try:
        _ = vision.Affine(degrees=15, translate=[1, 1], scale=1.5, shear=[1.5, 3.5, 3.5])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "The length of shear should be 2."


def test_affine_exception_translate_size():
    """
    Feature: Test Affine(translate is not list or a tuple of length 2)
    Description: Input translate is not list or a tuple of length 2
    Expectation: Got an exception to raise TypeError
    """
    logger.info("test_affine_exception_translate_size")
    try:
        _ = vision.Affine(degrees=15, translate=[1, 1, 1], scale=1.9, shear=[10.1])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "The length of translate should be 2."


def test_affine_exception_translate_value():
    """
    Feature: Test Affine(translate value)
    Description: Input translate is not a sequence
    Expectation: Got an exception to raise TypeError
    """
    logger.info("test_affine_exception_translate_value")
    try:
        _ = vision.Affine(degrees=15, translate=(0.1,), scale=2.1, shear=[1.5, 1.5], resample=vision.Inter.BILINEAR)
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert str(e) == "The length of translate should be 2."


def test_affine_pipeline(plot=False):
    """
    Feature: Affine
    Description: Test Affine in pipeline mode
    Expectation: The dataset is processed as expected
    """
    # First dataset
    transforms_list = t_trans.Compose([vision.Decode(), vision.Resize([64, 64])])
    dataset = ds.TFRecordDataset(DATA_DIR,
                                 SCHEMA_DIR,
                                 columns_list=["image"],
                                 shuffle=False)
    dataset = dataset.map(operations=transforms_list, input_columns=["image"])

    # Second dataset
    affine_transforms_list = t_trans.Compose([vision.Decode(),
                                              vision.Resize([64, 64]),
                                              vision.Affine(degrees=15, translate=[0.2, 0.2],
                                                            scale=1.1, shear=[10.0, 10.0])])
    affine_dataset = ds.TFRecordDataset(DATA_DIR,
                                        SCHEMA_DIR,
                                        columns_list=["image"],
                                        shuffle=False)
    affine_dataset = affine_dataset.map(operations=affine_transforms_list, input_columns=["image"])

    num_image = 0
    image_list = []
    affine_image_list = []
    for image, affine_image in zip(dataset.create_dict_iterator(num_epochs=1, output_numpy=True),
                                   affine_dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_image += 1
        image_list.append(image["image"])
        affine_image_list.append(affine_image["image"])

    assert num_image == 3

    if plot:
        visualize_list(image_list, affine_image_list)


def test_affine_eager():
    """
    Feature: Affine op
    Description: Test eager support for Affine Cpp implementation
    Expectation: The output data is the same as the result of cv2.warpAffine
    """
    img_in = np.array([[[211, 192, 16], [146, 176, 190], [103, 86, 18], [23, 194, 246]],
                       [[17, 86, 38], [180, 162, 43], [197, 198, 224], [109, 3, 195]],
                       [[172, 197, 74], [33, 52, 136], [120, 185, 76], [105, 23, 221]],
                       [[197, 50, 36], [82, 187, 119], [124, 193, 164], [181, 8, 11]]], dtype=np.uint8)

    affine_op1 = vision.Affine(degrees=30, translate=[0.5, 0.5], scale=1.0, shear=[0, 0])
    img_out1 = affine_op1(img_in)
    exp1 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [211, 192, 16]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [211, 192, 16]],
                     [[0, 0, 0], [0, 0, 0], [172, 197, 74], [180, 162, 43]]], dtype=np.uint8)

    affine_op2 = vision.Affine(degrees=30, translate=[0.5, 0.5], scale=1.0, shear=[10, 10])
    img_out2 = affine_op2(img_in)
    exp2 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [211, 192, 16]],
                     [[0, 0, 0], [0, 0, 0], [172, 197, 74], [180, 162, 43]]], dtype=np.uint8)

    affine_op3 = vision.Affine(degrees=30, translate=[0.5, 0.5], scale=1.2, shear=5)
    img_out3 = affine_op3(img_in)
    exp3 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [211, 192, 16]],
                     [[0, 0, 0], [0, 0, 0], [17, 86, 38], [17, 86, 38]],
                     [[0, 0, 0], [172, 197, 74], [172, 197, 74], [180, 162, 43]]], dtype=np.uint8)

    mse1 = diff_mse(img_out1, exp1)
    mse2 = diff_mse(img_out2, exp2)
    mse3 = diff_mse(img_out3, exp3)
    assert mse1 < 0.001 and mse2 < 0.001 and mse3 < 0.001


def test_affine_operation_01():
    """
    Feature: Affine operation
    Description: Testing the normal functionality of the Affine operator
    Expectation: The Output is equal to the expected output
    """
    # Affine operator: The degrees parameter is set to 20.0
    degrees = 20.0
    translate = (-1, 1)
    scale = 0.9
    shear = (50.0, 100.0)
    dataset1 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    affine_op = vision.Affine(degrees=degrees, translate=translate, shear=shear, scale=scale)
    dataset2 = dataset2.map(input_columns=["image"], operations=affine_op)
    for _ in zip(dataset1.create_dict_iterator(output_numpy=True), dataset2.create_dict_iterator(output_numpy=True)):
        pass

    # Affine operator: Input data is of type float64, with 4 channels.
    image = np.random.randint(0, 255, (128, 128, 4)).astype(np.float64)
    degrees = 10
    translate = (1, 1)
    scale = 2.2
    shear = 1.8
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    _ = affine_op(image)

    # Affine operator: The degrees parameter is set to 0
    degrees = 0
    translate = [-0, 0]
    scale = 11.11
    shear = [10.12, 22.09]
    resample = Inter.AREA
    fill_value = (1, 2, 3)
    dataset1 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                              resample=resample, fill_value=fill_value)
    dataset2 = dataset2.map(input_columns=["image"], operations=affine_op)

    for _ in zip(dataset1.create_dict_iterator(output_numpy=True), dataset2.create_dict_iterator(output_numpy=True)):
        pass

    # Affine operator executed normally: The translate parameter is a valid array, and resample is set to BICUBIC.
    degrees = 15.1
    translate = [-1, 0.9]
    scale = 100.10
    shear = (10.0, 10.0)
    resample = Inter.BICUBIC
    dataset1 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample)
    dataset2 = dataset2.map(input_columns=["image"], operations=affine_op)

    for _ in zip(dataset1.create_dict_iterator(output_numpy=True), dataset2.create_dict_iterator(output_numpy=True)):
        pass

    # Affine operator executed normally: The translate parameter is set to -1
    degrees = 123.321
    translate = (-1, -1)
    scale = 0.01
    shear = (5.1, 0.10)
    resample = Inter.BILINEAR
    dataset1 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample)
    dataset2 = dataset2.map(input_columns=["image"], operations=affine_op)

    for _ in zip(dataset1.create_dict_iterator(output_numpy=True), dataset2.create_dict_iterator(output_numpy=True)):
        pass

    # Affine operator executed normally: The translate parameter is set to [0, 1]
    degrees = 150.5
    translate = [0, 1]
    scale = 1.1112
    shear = [1.0, 4.0]
    dataset1 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    dataset2 = dataset2.map(input_columns=["image"], operations=affine_op)
    for _ in zip(dataset1.create_dict_iterator(output_numpy=True), dataset2.create_dict_iterator(output_numpy=True)):
        pass


def test_affine_operation_02():
    """
    Feature: Affine operation
    Description: Testing the normal functionality of the Affine operator
    Expectation: The Output is equal to the expected output
    """
    # The affine operator executes normally: The parameter shear contains negative values.
    degrees = 1.005
    translate = (-0.45, 0)
    scale = 256.0
    shear = [-180.0, 180.0]
    resample = Inter.BICUBIC
    fill_value = 255
    dataset1 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample,
                              fill_value=fill_value)
    dataset2 = dataset2.map(input_columns=["image"], operations=affine_op)
    for _ in zip(dataset1.create_dict_iterator(output_numpy=True), dataset2.create_dict_iterator(output_numpy=True)):
        pass

    # Affine operator executed normally: Input data is a JPG image.
    with Image.open(image_jpg) as image:
        degrees = 0
        translate = (-0.15648, 1)
        scale = 100.2
        shear = 0.001
        resample = Inter.BILINEAR
        fill_value = 1
        affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                                  resample=resample, fill_value=fill_value)
        _ = affine_op(image)

    # Affine operator executed normally: Input data is a PNG image.
    image = cv2.imread(image_png)
    degrees = 11.56
    translate = [0, -0.100]
    scale = 1.0
    shear = 167.0
    resample = Inter.NEAREST
    fill_value = 255
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                              resample=resample, fill_value=fill_value)
    _ = affine_op(image)

    # Affine operator executed normally: Input data is a GIF image.
    with Image.open(image_gif) as image:
        degrees = 167
        translate = [-0.256, 0.255]
        scale = 2.1
        shear = 10
        resample = Inter.BICUBIC
        affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample)
        _ = affine_op(image)

    # Affine operator executed normally: Input data is a BMP image.
    image = cv2.imread(image_bmp)
    degrees = 180
    translate = [0, 0]
    scale = 180
    shear = [2.2, 8.8]
    resample = Inter.BILINEAR
    fill_value = (180, 26, 109)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                              resample=resample, fill_value=fill_value)
    _ = affine_op(image)

    # Affine operator executed normally: Input data shape is (346, 489, 3), shear is 0.
    image = np.random.randn(346, 489, 3).astype(np.float32)
    degrees = 180.0
    translate = (0, 0)
    scale = 0.03
    shear = 0
    resample = Inter.BICUBIC
    fill_value = (64, 64, 64)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                              resample=resample, fill_value=fill_value)
    _ = affine_op(image)

    # Affine operator executed normally: Input data shape is (128, 128, 1)
    image = np.random.randint(0, 255, (128, 128, 1)).astype(np.uint8)
    degrees = 4.2
    translate = [-1, 1]
    scale = 0.2
    shear = (3.0, 3.0)
    resample = Inter.NEAREST
    fill_value = (180, 180, 200)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                              resample=resample, fill_value=fill_value)
    affine_op(image)


def test_affine_operation_03():
    """
    Feature: Affine operation
    Description: Testing the normal functionality of the Affine operator
    Expectation: The Output is equal to the expected output
    """
    # Affine operator executed normally: Input data shape is (128, 128, 4)
    image = np.random.randint(0, 255, (128, 128, 4)).astype(np.uint8)
    degrees = 0.6
    translate = (-1, -1)
    scale = 16777215.1
    shear = [0.8, 4.1]
    resample = Inter.BILINEAR
    fill_value = (0, 255, 230)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                              resample=resample, fill_value=fill_value)
    _ = affine_op(image)

    # 输入shape为(192, 263)时，Affine接口调用成功
    # Affine operator executed normally: Input data shape is (192, 263)
    image = np.random.randn(192, 263, 23)
    degrees = 179.9
    translate = [0, 1]
    scale = 18.21
    shear = (100.1, 100.0)
    resample = Inter.BILINEAR
    fill_value = (12, 8, 12)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                              resample=resample, fill_value=fill_value)
    _ = affine_op(image)

    # Affine operator executed normally: Testing pipeline combination mode
    ds1 = ds.ImageFolderDataset(DATA_DIR_1, 1)
    transforms0 = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform0 = t_trans.Compose(transforms0)
    ds1 = ds1.map(input_columns=["image"], operations=transform0)
    ds2 = ds.ImageFolderDataset(DATA_DIR_1, 1)
    transforms1 = [
        vision.Decode(),
        vision.Affine(10, [0, 1], 20.1, (1.0, 5.0), Inter.NEAREST, 0),
        vision.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for _ in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        break

    # Affine operator executed normally: parameter degrees set to 10
    image = cv2.imread(image_jpg)
    degrees = 10
    translate = (0, 1)
    scale = 0.9
    shear = (0.8, 4.1)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    _ = affine_op(image)

    # Affine operator executed normally: parameter translate set to [0.8, -1]
    degrees = 15.1
    translate = [0.8, -1]
    scale = 0.9
    shear = (0.9, 1.1)
    dataset1 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(DATA_DIR_1, shuffle=False, decode=True)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    dataset2 = dataset2.map(input_columns=["image"], operations=affine_op)
    for _ in zip(dataset1.create_dict_iterator(output_numpy=True), dataset2.create_dict_iterator(output_numpy=True)):
        pass

    # Affine operator executed normally: The scale parameter is of type int.
    image = np.random.randn(192, 263, 23)
    degrees = 150.2
    translate = [-1, 1]
    scale = 180
    shear = (11.1, 0.11)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    affine_op(image)


def test_affine_operation_04():
    """
    Feature: Affine operation
    Description: Testing the normal functionality of the Affine operator
    Expectation: The Output is equal to the expected output
    """
    # Affine operator executed normally: The shear parameter is -180
    image = np.random.randn(192, 263, 23)
    degrees = 100
    translate = (-1, 0)
    scale = 11.10
    shear = -180
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    _ = affine_op(image)

    # The Affine operator executes normally: The parameter `shear` is of type tuple.
    image = cv2.imread(image_jpg)
    degrees = 10
    translate = (-1, 1)
    scale = 1.1
    shear = (-1, 1)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    _ = affine_op(image)

    # Affine operator executed normally: Input is a PIL image, fill_value is a tuple.
    image = np.random.randn(192, 263, 3).astype(np.uint8)
    image = vision.ToPIL()(image)
    degrees = 128.5
    translate = (0, 1)
    scale = 0.9
    shear = [10.0, 100.0]
    fill_value = (100, 200, 220)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value)
    affine_op(image)


def test_affine_exception_01():
    """
    Feature: Affine operation
    Description: Testing the Affine Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Affine operator exception scenarios: When the shear parameter exceeds 180 or falls below -180.
    image = np.random.randn(192, 263, 3).astype(np.float32)
    degrees = 180
    translate = (-1, 0)
    scale = 11.10
    shear = 180.123
    with pytest.raises(ValueError, match="Input shear is not within the required interval of \\[-180, 180\\]."):
        _ = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="CPU")(image)

    shear = -180.123
    with pytest.raises(ValueError, match="Input shear is not within the required interval of \\[-180, 180\\]."):
        _ = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="CPU")(image)

    # Affine Operator Exception Scenario: Parameter `degrees` value is a string
    degrees = "10"
    translate = (10, 1)
    scale = 1.1
    shear = 0.8
    with pytest.raises(TypeError, match=("Argument degrees with value 10 is not of type \\[<class 'int'>,"
                                         " <class 'float'>\\], but got <class 'str'>.")):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    image = cv2.imread(image_bmp)
    with pytest.raises(ValueError, match="Input degrees is not within the required interval of \\[-180, 180\\]."):
        _ = vision.Affine(degrees=-250, translate=(-1, 1), scale=3.01, shear=[0.8, 4.1]).device(
            device_target="CPU")(image)
    image = cv2.imread(image_bmp)
    with pytest.raises(ValueError, match="Input degrees is not within the required interval of \\[-180, 180\\]."):
        _ = vision.Affine(degrees=250, translate=(-1, 1), scale=3.01, shear=[0.8, 4.1]).device(
            device_target="CPU")(image)

    # Affine Operator Exception Scenario: Parameter `degrees` value is a tuple
    image = np.random.randint(0, 255, (128, 128, 4)).astype(np.uint8)
    degrees = (10, 20)
    translate = (1, 19)
    scale = 2.2
    shear = 1.8
    with pytest.raises(TypeError, match="Argument degrees with value \\(10, 20\\) is not of type \\[<class 'int'>,"
                                        " <class 'float'>\\], but got <class 'tuple'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)

    # Affine Operator Exception Scenario: The degrees parameter is an array.
    image = np.random.randn(192, 263)
    degrees = [20]
    translate = (0, 1)
    scale = 1.8
    shear = (0.9, 1.1)
    with pytest.raises(TypeError, match="Argument degrees with value \\[20\\] is not of type \\[<class 'int'>,"
                                        " <class 'float'>\\], but got <class 'list'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)

    # Affine Operator Exception Scenario: When the degrees parameter is not set
    translate = (0, 1)
    scale = 0.9
    shear = (0.9, 1.1)
    with pytest.raises(TypeError, match="missing a required argument: 'degrees'"):
        vision.Affine(translate=translate, scale=scale, shear=shear)

    # Affine Operator Exception Scenario: Translate Value Exceeds 1
    image = cv2.imread(image_jpg)
    degrees = 15.0
    translate = (0, 240.12)
    scale = 0.9
    shear = (0.9, 1.1)
    with pytest.raises(ValueError, match="Input translate\\[1\\] is not within the required interval of"
                                         " \\[-1.0, 1.0\\]."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)

    # Affine Operator Exception Scenario: Only One Value in the translate Parameter
    image = np.random.randn(192, 263)
    degrees = 15.2
    translate = [0]
    scale = 10.9
    shear = (11.1, 0.11)
    with pytest.raises(TypeError, match="The length of translate should be 2."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)

    # Affine Operator Exception Scenario: Number of translate values is 4
    image = np.random.randn(192, 263)
    degrees = 15.2
    translate = [0, -1, -1, 1]
    scale = 10.9
    shear = (11.1, 0.11)
    with pytest.raises(TypeError, match="The length of translate should be 2."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)

    # Affine Operator Exception Scenario: Parameter `translate` Value is a String
    image = np.random.randn(192, 263)
    degrees = 100.2
    translate = ("0", "1")
    scale = 10.9
    shear = (11.1, 0.11)
    with pytest.raises(TypeError, match="Argument translate\\[0\\] with value 0 is not of type \\[<class 'int'>,"
                                        " <class 'float'>\\], but got <class 'str'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)

    # Affine Operator Exception Scenario: Parameter `translate` Value is a bool
    image = np.random.randn(192, 263)
    degrees = 100.2
    translate = False
    scale = 10.9
    shear = (11.1, 0.11)
    with pytest.raises(TypeError, match="Argument translate with value False is not of type \\[<class 'list'>,"
                                        " <class 'tuple'>\\], but got <class 'bool'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)

    # Affine Operator Exception Scenario: translate Parameter Not Set
    degrees = 100.2
    scale = 10.9
    shear = (11.1, 0.11)
    with pytest.raises(TypeError, match="missing a required argument: 'translate'"):
        vision.Affine(degrees=degrees, scale=scale, shear=shear)


def test_affine_exception_02():
    """
    Feature: Affine operation
    Description: Testing the Affine Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Affine Operator Exception Scenario: Scale Parameter is a Tuple
    image = np.random.randn(192, 263)
    degrees = 4.2
    translate = (0, 1)
    scale = (0.6, 0.9, 1.1)
    shear = (11.1, 0.11)
    with pytest.raises(TypeError, match="Argument scale with value \\(0.6, 0.9, 1.1\\) is not of type \\[<class 'int'>,"
                                        " <class 'float'>\\], but got <class 'tuple'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)

    # Affine Operator Exception Scenario: Scale Parameter is a String
    image = np.random.randn(192, 263)
    degrees = 15.32547
    translate = (-1, 0)
    scale = "0.9"
    shear = (11.1, 0.11)
    with pytest.raises(TypeError, match="Argument scale with value 0.9 is not of type \\[<class 'int'>,"
                                        " <class 'float'>\\], but got <class 'str'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)

    # Affine Operator Exception Scenario: Negative Scale Parameter
    image = np.random.randn(192, 263)
    degrees = 15.15
    translate = [0, 1]
    scale = -0.1
    shear = (11.1, 11.11)
    with pytest.raises(ValueError, match="Input scale must be greater than 0"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)

    # Affine Operator Exception Scenario: Scale Parameter Set to 0
    image = np.random.randn(192, 263)
    degrees = 15.15
    translate = [0, 1]
    scale = 0
    shear = (11.1, 11.11)
    with pytest.raises(ValueError, match="Input scale must be greater than 0"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)

    # Affine Operator Exception Scenario: Scale Parameter is a list
    image = np.random.randn(192, 263)
    degrees = 15.32547
    translate = (-1, 0)
    scale = [11.1, 0.11]
    shear = (11.1, 0.11)
    with pytest.raises(TypeError, match="Argument scale with value \\[11.1, 0.11\\] is not of type \\[<class 'int'>,"
                                        " <class 'float'>\\], but got <class 'list'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)

    # Affine Operator Exception Scenario: scale Parameter Not Set
    degrees = 200.2
    translate = (-1, 0)
    shear = (11.1, 0.11)
    with pytest.raises(TypeError, match="missing a required argument: 'scale'"):
        vision.Affine(degrees=degrees, translate=translate, shear=shear)

    # Affine Operator Exception Scenario: shear Parameter Not Set
    degrees = 200.2
    translate = (-1, 0)
    scale = 11.1
    with pytest.raises(TypeError, match="missing a required argument: 'shear'"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale)

    # Affine Operator Exception Scenario: The parameter `shear` is a tuple of length 4.
    image = np.random.randn(192, 263)
    degrees = 0
    translate = (1, 1)
    scale = 360
    shear = (1.0, 2.0, 4.0, 3.0)
    with pytest.raises(TypeError, match="The length of shear should be 2."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)

    # Affine Operator Exception Scenario: Parameter `shear` is of type string
    degrees = 0
    translate = (0, 1)
    scale = 720
    shear = "10"
    with pytest.raises(TypeError, match=("Argument shear with value 10 is not of type \\[<class "
                                         "'numbers.Number'>, <class 'tuple'>, <class 'list'>\\]")):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)

    # Affine Operator Exception Scenario: The parameter `shear` is a list of length 1.
    image = cv2.imread(image_jpg)
    degrees = 10
    translate = (-1, 1)
    scale = 1.1
    shear = [5]
    with pytest.raises(TypeError, match="The length of shear should be 2."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)

    # Affine Operator Exception Scenario: The parameter `shear` is a tuple of length 3.
    image = np.random.randn(192, 263)
    degrees = 0
    translate = (0, 1)
    scale = 100
    shear = (1.0, 2.0, 3.0)
    with pytest.raises(TypeError, match="The length of shear should be 2."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)


def test_affine_exception_03():
    """
    Feature: Affine operation
    Description: Testing the Affine Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Affine Operator Exception Scenario: The parameter `shear` is a tuple of length 2, where the element type is string
    degrees = 120
    translate = [-1, 1]
    scale = 120
    shear = ("5", "10")
    with pytest.raises(TypeError, match="Argument shear\\[0\\] with value 5 is not of type \\[<class 'int'>,"
                                        " <class 'float'>\\], but got <class 'str'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)

    # Affine Operator Exception Scenario: Parameter `resample` is of type string
    degrees = 120
    translate = [-1, 1]
    scale = 120
    shear = [5.0, 10.1]
    resample = "Inter.NEAREST"
    with pytest.raises(TypeError,
                       match="Argument resample with value Inter.NEAREST is not of type \\[<enum 'Inter'>\\]"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample)

    # Affine Operator Exception Scenario: Parameter `resample` is of type list
    degrees = 120
    translate = [-1, 1]
    scale = 120
    shear = [5.0, 10.1]
    resample = [Inter.BILINEAR]
    with pytest.raises(TypeError, match="Argument resample with value \\[\\<Inter.BILINEAR: 2\\>\\] is not of type"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample)

    # Affine Operator Exception Scenario: Parameter `resample` is Inter
    degrees = 123
    translate = [0, 1]
    scale = 456
    shear = [15.0, 128.1]
    resample = Inter
    with pytest.raises(TypeError,
                       match="Argument resample with value <enum 'Inter'> is not of type \\[<enum 'Inter'>\\]."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample)

    # Affine Operator Exception Scenario: Parameter `resample` is of type int
    degrees = 123.1
    translate = [0, 1]
    scale = 456
    shear = [15.0, 128.1]
    resample = 10
    with pytest.raises(TypeError, match="Argument resample with value 10 is not of type"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample)

    # Affine Operator Exception Scenario: Negative Value for fill_value Parameter
    degrees = 123.1
    translate = [0, 1]
    scale = 456.9
    shear = (15.0, 128.1)
    resample = Inter.NEAREST
    fill_value = -1
    with pytest.raises(ValueError, match=r"Input fill_value is not within the required interval of \[0, 255\]"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample,
                      fill_value=fill_value)

    # Affine Operator Exception Scenario: Parameter `fill_value` is empty
    degrees = 10
    translate = (0, 1)
    scale = 1.1
    shear = [-10.0, 10.0]
    fill_value = ()
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value)

    # Affine Operator Exception Scenario: The fill_value parameter is a tuple of length 2
    degrees = 10
    translate = (0, 1)
    scale = 1.1
    shear = [-10.0, 10.0]
    fill_value = (1, 2)
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value)

    # Affine Operator Exception Scenario: Parameter `fill_value` is of type float
    degrees = 1
    translate = [0, 1]
    scale = 1.10
    shear = [10.0, -10.0]
    fill_value = 1.0
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value)

    # Affine Operator Exception Scenario: Parameter `fill_value` is of type string
    degrees = 0
    translate = (0, 1)
    scale = 0.9
    shear = [-10.0, 100.1]
    fill_value = "1"
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value)


def test_affine_exception_04():
    """
    Feature: Affine operation
    Description: Testing the Affine Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Affine Operator Exception Scenario: The fill_value parameter is a 3-tuple with string elements
    degrees = 0.123
    translate = (1, 0)
    scale = 101.1
    shear = [1.1, 12.12]
    fill_value = ("1", "2", 3)
    with pytest.raises(TypeError, match=r"Argument fill_value\[0\] with value 1 is not of type \[<class 'int'>\], "
                                        r"but got <class 'str'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value)

    # Affine Operator Exception Scenario: Parameter `fill_value` exceeds 255
    degrees = 120.123
    translate = (-1, 0)
    scale = 11.1
    shear = [-1.0, 10.1]
    fill_value = 256
    with pytest.raises(ValueError, match=r"Input fill_value is not within the required interval of \[0, 255\]."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value)

    # Affine Operator Exception Scenario: Parameter `fill_value` is a 4-tuple
    degrees = 121.123
    translate = (-1, 0)
    scale = 101.1
    shear = [-1.12, 10.1]
    fill_value = (1, 2, 3, 4)
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value)

    # Affine Operator Exception Scenario: When the fill_value parameter is a tuple with elements exceeding 255
    degrees = 120.123
    translate = (-1, 0)
    scale = 11.1
    shear = [-10.14, 10.1]
    fill_value = (1, 2, 256)
    with pytest.raises(ValueError, match=r"Input fill_value\[2\] is not within the required interval of \[0, 255\]."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value)

    # Affine Operator Exception Scenario: Input is a list
    image = np.random.randn(600, 450, 3).tolist()
    degrees = 0.1
    translate = (0, 1)
    scale = 0.9
    shear = [0.0, 100.0]
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        affine_op(image)

    # Affine Operator Exception Scenario: Input is a tuple
    image = tuple(np.random.randn(600, 450, 3))
    degrees = 10.1
    translate = (0, 1)
    scale = 10.9
    shear = [1.0, 100.0]
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>."):
        affine_op(image)

    # Affine Operator Exception Scenario: Input shape is (600, 450, 3, 3)
    image = np.random.randn(600, 450, 3, 3)
    degrees = 0.1
    translate = (0, 1)
    scale = 0.9
    shear = [10.0, 100.0]
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    with pytest.raises(RuntimeError,
                       match=".*Affine: input tensor is not in shape of <H,W,C> or <H,W>, but got rank: 4.*"):
        affine_op(image)


def test_affine_exception_05():
    """
    Feature: Affine operation
    Description: Testing the Affine Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Affine Operator Exception Scenario: All degrees Parameters Are Negative
    image = cv2.imread(image_bmp)
    degrees = -1000.1
    translate = (-1, 1)
    scale = 3.01
    shear = [0.8, 4.1]
    with pytest.raises(ValueError, match="Input degrees is not within the required interval of \\[-180, 180\\]."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)

    # Affine Operator Exception Scenario: Parameter degrees value exceeds 16777216
    image = cv2.imread(image_bmp)
    degrees = 16777217
    translate = (-1, 1)
    scale = 3.01
    shear = [0.8, 4.1]
    with pytest.raises(ValueError, match="Input degrees is not within the required interval of \\[-180, 180\\]."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)

    # Affine Operator Exception Scenario: Parameter shear is None
    with Image.open(image_jpg) as image:
        degrees = 0
        translate = (-1, 0)
        scale = 1
        shear = None
        resample = Inter.BILINEAR
        fill_value = 1
        with pytest.raises(TypeError, match="Argument shear with value None is not of type \\[<class 'numbers.Number'>,"
                                            " <class 'tuple'>, <class 'list'>\\], but got <class 'NoneType'>."):
            vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample,
                          fill_value=fill_value)(image)


if __name__ == "__main__":
    test_affine_exception_degrees_type()
    test_affine_exception_scale_value()
    test_affine_exception_shear_size()
    test_affine_exception_translate_size()
    test_affine_exception_translate_value()
    test_affine_pipeline(plot=False)
    test_affine_eager()
    test_affine_operation_01()
    test_affine_operation_02()
    test_affine_operation_03()
    test_affine_operation_04()
    test_affine_exception_01()
    test_affine_exception_02()
    test_affine_exception_03()
    test_affine_exception_04()
    test_affine_exception_05()
