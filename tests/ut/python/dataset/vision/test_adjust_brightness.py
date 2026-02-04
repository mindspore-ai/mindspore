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
Testing AdjustBrightness op in DE
"""
import numpy as np
import os
import pytest
from numpy.testing import assert_allclose
from PIL import Image

import mindspore as ms
import mindspore.dataset.vision.transforms as vision
import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms
from mindspore import log as logger
from util import diff_mse

DATA_DIR = "../data/dataset/testImageNetData/train/"
MNIST_DATA_DIR = "../data/dataset/testMnistData"

DATA_DIR_2 = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

TEST_DATA_DATASET_FUNC = "../data/dataset/"


def generate_numpy_random_rgb(shape):
    """
    Only generate floating points that are fractions like n / 256, since they
    are RGB pixels. Some low-precision floating point types in this test can't
    handle arbitrary precision floating points well.
    """
    return np.random.randint(0, 256, shape) / 255.


def test_adjust_brightness_eager(plot=False):
    """
    Feature: AdjustBrightness op
    Description: Test AdjustBrightness in eager mode
    Expectation: Output is the same as expected output
    """
    # Eager 3-channel
    image_file = "../data/dataset/testImageNetData/train/class1/1_1.jpg"
    img = np.fromfile(image_file, dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = vision.Decode()(img)
    img_adjustbrightness = vision.AdjustBrightness(1)(img)
    if plot:
        visualize_image(img, img_adjustbrightness)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_adjustbrightness),
                                                         img_adjustbrightness.shape))
    mse = diff_mse(img_adjustbrightness, img)
    logger.info("MSE= {}".format(str(mse)))
    assert mse == 0


def test_adjust_brightness_invalid_brightness_factor_param():
    """
    Feature: AdjustBrightness op
    Description: Test improper parameters for AdjustBrightness implementation
    Expectation: Throw ValueError exception and TypeError exception
    """
    logger.info("Test AdjustBrightness implementation with invalid ignore parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        trans = mindspore.dataset.transforms.transforms.Compose([
            vision.Decode(True),
            vision.Resize((224, 224)),
            vision.AdjustBrightness(brightness_factor=-10.0),
            vision.ToTensor()
        ])
        data_set = data_set.map(operations=[trans], input_columns=["image"])
    except ValueError as error:
        logger.info("Got an exception in AdjustBrightness: {}".format(str(error)))
        assert "Input brightness_factor is not within the required interval of " in str(error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        trans = mindspore.dataset.transforms.transforms.Compose([
            vision.Decode(True),
            vision.Resize((224, 224)),
            vision.AdjustBrightness(brightness_factor=[1, 2]),
            vision.ToTensor()
        ])
        data_set = data_set.map(operations=[trans], input_columns=["image"])
    except TypeError as error:
        logger.info("Got an exception in AdjustBrightness: {}".format(str(error)))
        assert "is not of type [<class 'float'>, <class 'int'>], but got" in str(error)


def test_adjust_brightness_pipeline():
    """
    Feature: AdjustBrightness op
    Description: Test AdjustBrightness in pipeline mode
    Expectation: Output is the same as expected output
    """
    # First dataset
    transforms1 = [vision.Decode(True), vision.Resize([64, 64]), vision.ToTensor()]
    transforms1 = mindspore.dataset.transforms.transforms.Compose(
        transforms1)
    ds1 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds1 = ds1.map(operations=transforms1, input_columns=["image"])

    # Second dataset
    transforms2 = [
        vision.Decode(True),
        vision.Resize([64, 64]),
        vision.AdjustBrightness(1.0),
        vision.ToTensor()
    ]
    transform2 = mindspore.dataset.transforms.transforms.Compose(
        transforms2)
    ds2 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds2 = ds2.map(operations=transform2, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(ds1.create_dict_iterator(num_epochs=1),
                            ds2.create_dict_iterator(num_epochs=1)):
        num_iter += 1
        ori_img = data1["image"].asnumpy()
        cvt_img = data2["image"].asnumpy()
        assert_allclose(ori_img.flatten(),
                        cvt_img.flatten(),
                        rtol=1e-5,
                        atol=0)
        mse = diff_mse(ori_img, cvt_img)
        logger.info("MSE= {}".format(str(mse)))
        assert mse == 0


def test_adjust_brightness_operation_01():
    """
    Feature: AdjustBrightness operation
    Description: Testing the normal functionality of the AdjustBrightness operator
    Expectation: The Output is equal to the expected output
    """
    # AdjustBrightness operator, normal testing, pipeline mode, brightness_factor=1, input image is numpy
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    ds1 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False, decode=True)
    ds2 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False, decode=True)
    ds2 = ds2.map(operations=vision.AdjustBrightness(1), input_columns=["image"])
    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True, num_epochs=1),
                            ds2.create_dict_iterator(output_numpy=True, num_epochs=1)):
        assert (data1["image"] == data2["image"]).all()

    # AdjustBrightness operator, normal testing, pipeline mode, brightness_factor=0, input image is PIL
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")

    ds1 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False)

    ds2 = ds.ImageFolderDataset(dataset_dir=dataset_dir, shuffle=False)
    op_list = [vision.Decode(to_pil=True), vision.Resize([64, 64]), vision.AdjustBrightness(0),
               vision.ToTensor()]
    ds2 = ds2.map(operations=op_list, input_columns=["image"])

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True, num_epochs=1),
                            ds2.create_dict_iterator(output_numpy=True, num_epochs=1)):
        assert sum(data2["image"].flatten()) == sum((data1["image"] * 0).flatten())

    # AdjustBrightness operator, normal testing, eager mode, input image is numpy and uint8 data, brightness_factor=1
    image = np.random.randint(0, 256, (20, 20, 3)).astype(np.uint8)
    op = vision.AdjustBrightness(1)
    out = op(image)
    assert out.shape == (20, 20, 3)
    assert out.dtype == np.uint8
    assert (out == image).all()

    # AdjustBrightness operator, normal testing, input image is
    # a numpy array of shape <H,W,3> with float32 data, brightness_factor=2.666
    image = np.random.randint(0, 256, (20, 20, 3)) / 255.0
    image = image.astype(np.float32)
    op = vision.AdjustBrightness(2.666)
    out = op(image)
    assert out.shape == (20, 20, 3)
    assert out.dtype == np.float32
    assert (out == image * 2.666).all()

    # AdjustBrightness operator, normal testing, input image is
    # a numpy array of shape <H,W,3> with float32 data, brightness_factor=2666
    image = np.random.randint(0, 256, (20, 20, 3)) / 255.0
    image = image.astype(np.float32)
    op = vision.AdjustBrightness(2666)
    out = op(image)
    assert out.shape == (20, 20, 3)
    assert out.dtype == np.float32
    assert (out == image * 2666).all()

    # AdjustBrightness operator, normal testing, input image is
    # a numpy array of shape <H,W,3> with float32 data, brightness_factor=0.2666
    image = np.random.randint(0, 256, (20, 20, 3)) / 255.0
    image = image.astype(np.float32)
    op = vision.AdjustBrightness(0.2666)
    out = op(image)
    assert out.shape == (20, 20, 3)
    assert out.dtype == np.float32
    assert (out == image * 0.2666).all()

    # AdjustBrightness operator, normal testing, input image is
    # a numpy array of shape <H,W,3> with float32 data, brightness_factor=1.00001
    image = np.random.randint(0, 256, (20, 20, 3)) / 255.0
    image = image.astype(np.float32)
    op = vision.AdjustBrightness(1.00001)
    out = op(image)
    assert out.shape == (20, 20, 3)
    assert out.dtype == np.float32
    assert (out == image * 1.00001).all()

    # AdjustBrightness operator, normal testing, input image is
    # a numpy array of shape <H,W,3> with float64 data, brightness_factor=0
    image = np.random.randint(0, 256, (20, 20, 3)) / 255.0
    image = image.astype(np.float64)
    op = vision.AdjustBrightness(0)
    out = op(image)
    assert out.shape == (20, 20, 3)
    assert out.dtype == np.float64
    assert (out == image * 0).all()

    # AdjustBrightness operator, normal testing, input image is PIL, brightness_factor=0.001
    image = np.random.randint(0, 256, (20, 20, 3)).astype(np.uint8)
    image = vision.ToPIL()(image)
    op = vision.AdjustBrightness(0.001)
    out = op(image)
    assert np.array(out).shape == (20, 20, 3)

    # AdjustBrightness operator, normal testing, input image is a BMP image
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    with Image.open(image_path) as image:
        op = vision.AdjustBrightness(2)
        out = op(image)
        assert np.array(out).shape == np.array(image).shape


def test_adjust_brightness_operation_02():
    """
    Feature: AdjustBrightness operation
    Description: Testing the normal functionality of the AdjustBrightness operator
    Expectation: The Output is equal to the expected output
    """
    # AdjustBrightness operator, normal testing, input image is a JPG image
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_path) as image:
        op = vision.AdjustBrightness(2)
        out = op(image)
        assert np.array(out).shape == np.array(image).shape

    # AdjustBrightness operator, normal testing, input image is a PNG image
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    with Image.open(image_path) as image:
        op = vision.AdjustBrightness(2)
        out = op(image)
        assert np.array(out).shape == np.array(image).shape

    # AdjustBrightness operator, normal testing, input image is a GIF image, mode=RGB
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_path).convert("RGB") as image:
        op = vision.AdjustBrightness(2)
        out = op(image)
        assert np.array(out).shape == np.array(image).shape


def test_adjust_brightness_exception_01():
    """
    Feature: AdjustBrightness operation
    Description: Testing the AdjustBrightness Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # AdjustBrightness operator, anomaly testing, input image is a GIF image, mode=P
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    with Image.open(image_path) as image:
        op = vision.AdjustBrightness(2)
        with pytest.raises(ValueError, match="image has wrong mode"):
            op(image)

    # AdjustBrightness operator, error testing, input image is numpy type, image dimensions: <H,W,1>
    image = np.random.randint(0, 256, (20, 20, 1)).astype(np.uint8)
    op = vision.AdjustBrightness(2)
    op(image)

    # AdjustBrightness operator, error testing, input image is numpy type, image dimensions: <H,W,2>
    image = np.random.randint(0, 256, (20, 20, 2)).astype(np.uint8)
    op = vision.AdjustBrightness(2)
    try:
        op(image)
    except RuntimeError as e:
        assert "AdjustBrightness: the channel of image tensor does not match the " \
               "requirement of operator. Expecting tensor in channel of (1, 3). But got channel 2." in str(e)

    # AdjustBrightness operator, error testing, input image is numpy type, image dimensions: <H,W,4>
    image = np.random.randint(0, 256, (20, 20, 4)).astype(np.uint8)
    op = vision.AdjustBrightness(2)
    try:
        op(image)
    except RuntimeError as e:
        assert "AdjustBrightness: the channel of image tensor does not match the " \
               "requirement of operator. Expecting tensor in channel of (1, 3). But got channel 4." in str(e)

    # AdjustBrightness operator, error testing, input image is numpy type, image dimensions: <H,W,5>
    image = np.random.randint(0, 256, (20, 20, 5)).astype(np.uint8)
    op = vision.AdjustBrightness(2)
    try:
        op(image)
    except RuntimeError as e:
        assert "AdjustBrightness: the channel of image tensor does not match the " \
               "requirement of operator. Expecting tensor in channel of (1, 3). But got channel 5." in str(e)

    # AdjustBrightness operator, error testing, input image is numpy type, image dimensions: <H,W,20>
    image = np.random.randint(0, 256, (20, 20, 20)).astype(np.uint8)
    op = vision.AdjustBrightness(2)
    try:
        op(image)
    except RuntimeError as e:
        assert "AdjustBrightness: the channel of image tensor does not match the " \
               "requirement of operator. Expecting tensor in channel of (1, 3). But got channel 20." in str(e)

    # AdjustBrightness operator, error testing, input image is of numpy type, image dimension is one-dimensional data
    image = np.random.randint(0, 256, (20,)).astype(np.uint8)
    op = vision.AdjustBrightness(2)
    try:
        op(image)
    except RuntimeError as e:
        assert "AdjustBrightness: the dimension of image tensor does not match the " \
               "requirement of operator. Expecting tensor in dimension of (2, 3), " \
               "in shape of <H, W> or <H, W, C>. But got dimension 1. You may need to perform Decode first." in str(e)

    # AdjustBrightness operator, error testing, input image is of numpy type, image dimension is two-dimensional data
    image = np.random.randint(0, 256, (20, 20)).astype(np.uint8)
    op = vision.AdjustBrightness(2)
    op(image)

    # AdjustBrightness operator, error testing, input image is of numpy type, image dimension is four-dimensional data
    image = np.random.randint(0, 256, (20, 20, 3, 5)).astype(np.uint8)
    op = vision.AdjustBrightness(2)
    try:
        op(image)
    except RuntimeError as e:
        assert "AdjustBrightness: the dimension of image tensor does not match the " \
               "requirement of operator. Expecting tensor in dimension of (2, 3), in shape " \
               "of <H, W> or <H, W, C>. But got dimension 4." in str(e)

    # AdjustBrightness operator, error testing, input image is numpy bytes data
    image = np.random.randint(0, 256, (20, 20, 3)).astype("S")
    op = vision.AdjustBrightness(2)
    with pytest.raises(RuntimeError,
                       match=r"AdjustBrightness: the data type of image tensor "
                             r"does not match the requirement of operator. Expecting tensor in "
                             r"type of \(bool, int8, uint8, int16, uint16, int32, float16, float32, "
                             r"float64\). But got type bytes."):
        op(image)

    # AdjustBrightness operator, exception testing, brightness_factor is of type str
    with pytest.raises(TypeError, match=r"Argument brightness_factor with value 1 is not of type "
                                        r"\[<class 'float'>, <class 'int'>\], but got <class 'str'>."):
        vision.AdjustBrightness(brightness_factor='1')

    # AdjustBrightness operator, exception testing, brightness_factor is of type list
    with pytest.raises(TypeError, match=r"Argument brightness_factor with value \[1\] is not of type "
                                        r"\[<class 'float'>, <class 'int'>\], but got <class 'list'>."):
        vision.AdjustBrightness(brightness_factor=[1])

    # AdjustBrightness operator, exception testing, brightness_factor is of type bool
    with pytest.raises(TypeError, match=r"Argument brightness_factor with value True is not of type "
                                        r"\(<class 'float'>, <class 'int'>\), but got <class 'bool'>."):
        vision.AdjustBrightness(brightness_factor=True)


def test_adjust_brightness_exception_02():
    """
    Feature: AdjustBrightness operation
    Description: Testing the AdjustBrightness Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # AdjustBrightness operator, exception testing, brightness_factor is of type tuple
    with pytest.raises(TypeError, match=r"Argument brightness_factor with value \(1, 2\) is not of type "
                                        r"\[<class 'float'>, <class 'int'>\], but got <class 'tuple'>."):
        vision.AdjustBrightness(brightness_factor=(1, 2))

    # AdjustBrightness operator, exception test, brightness_factor set to -0.00001
    with pytest.raises(ValueError, match=r"Input brightness_factor is not within the required interval "
                                         r"of \[0, 16777216\]."):
        vision.AdjustBrightness(brightness_factor=-0.00001)

    # AdjustBrightness operator, exception test, brightness_factor set to 16777216.00001
    with pytest.raises(ValueError, match=r"Input brightness_factor is not within the required interval "
                                         r"of \[0, 16777216\]."):
        vision.AdjustBrightness(brightness_factor=16777216.00001)

    # AdjustBrightness operator, exception testing, input image is of type int
    image = 100
    op = vision.AdjustBrightness(2)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'int'>."):
        op(image)

    # AdjustBrightness operator, exception testing, input image is of type str
    image = '100'
    op = vision.AdjustBrightness(2)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'str'>."):
        op(image)

    # AdjustBrightness operator, exception testing, input image is of type float
    image = 100.0
    op = vision.AdjustBrightness(2)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'float'>."):
        op(image)

    # AdjustBrightness operator, exception testing, input image is of type list
    image = np.random.randint(0, 256, (10, 10, 3)).tolist()
    op = vision.AdjustBrightness(2)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        op(image)

    # AdjustBrightness operator, exception testing, input image is of type tuple
    image = (1, 2, 3)
    op = vision.AdjustBrightness(2)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>."):
        op(image)

    # AdjustBrightness operator, exception testing, input image is ms.Tensor
    image = ms.Tensor(np.random.randint(0, 256, (10, 10, 3)))
    op = vision.AdjustBrightness(2)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got "
                                        "<class 'mindspore.common.tensor.Tensor'>."):
        op(image)


if __name__ == "__main__":
    test_adjust_brightness_eager()
    test_adjust_brightness_invalid_brightness_factor_param()
    test_adjust_brightness_pipeline()
    test_adjust_brightness_operation_01()
    test_adjust_brightness_operation_02()
    test_adjust_brightness_exception_01()
    test_adjust_brightness_exception_02()
